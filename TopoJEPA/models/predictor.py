"""
Predictor 模块
==============
JEPA 的核心: 从视觉嵌入 S_V + 查询条件 → 预测目标嵌入 Ŝ_Y

借鉴:
  - jepa/src/models/predictor.py 的 VisionTransformerPredictor 架构
  - VL-JEPA 论文中用 Llama-3 Transformer 层做 predictor
  - 我们用轻量 Transformer 层, 支持双向注意力

与原始 JEPA predictor 的关键区别:
  1. 原始 JEPA: context patches → predict target patches (自监督)
  2. 我们: visual embeddings + query → predict text/label embedding (有监督)
  3. 增加了 pooling + projection 到共享嵌入空间

原始 JEPA predictor 的核心设计:
  - 将 context tokens 和 target tokens 拼接后送入 self-attention (不是 cross-attention)
  - 只取 target 部分的输出
  - 我们借鉴这个 concat + self-attention 的设计, 但 target 用 query tokens 替代
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


class PredictorBlock(nn.Module):
    """
    Predictor 的 Transformer Block

    支持可选的双向注意力 (VL-JEPA 发现双向注意力优于因果注意力)
    默认用双向 (full self-attention), 可切换为因果 (causal)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_bidirectional: bool = True,
    ):
        super().__init__()
        self.use_bidirectional = use_bidirectional

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def _get_attn_mask(self, seq_len: int, device: torch.device) -> Optional[torch.Tensor]:
        """生成因果注意力 mask (若需要)"""
        if self.use_bidirectional:
            return None
        # 因果 mask: 上三角为 -inf
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(
        self,
        x: torch.Tensor,                          # [B, N, D]
        key_padding_mask: Optional[torch.Tensor] = None,  # [B, N] True=padding
    ) -> torch.Tensor:
        # Pre-LN + self-attention
        x_norm = self.norm1(x)
        attn_mask = self._get_attn_mask(x.shape[1], x.device)
        # 非必要时不返回注意力权重，避免 [B, H, N, N] 级别的显存占用。
        attn_out = self.attn(
            x_norm, x_norm, x_norm,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        x = x + attn_out
        # Pre-LN + FFN
        x = x + self.mlp(self.norm2(x))
        return x


class EmbeddingPredictor(nn.Module):
    """
    嵌入预测器: (S_V, X_Q) → Ŝ_Y

    架构 (参考 JEPA predictor 的 concat 设计):
      1. 视觉嵌入 S_V [B, N_v, D_v] 投影到 predictor 维度 → [B, N_v, D_p]
      2. 查询嵌入 X_Q [B, N_q, D_p] (可选, Stage 2 才有)
      3. 拼接 [visual; query] → [B, N_v+N_q, D_p], 送入 Transformer 层
      4. 对输出做 average pooling (排除 padding)
      5. 投影到共享嵌入空间 → [B, D_out]

    参数量预算: ~100-500M (根据 depth 和 embed_dim 调整)
    """

    def __init__(
        self,
        visual_dim: int = 768,            # 视觉编码器输出维度
        predictor_dim: int = 512,         # predictor 内部维度
        output_dim: int = 768,            # 输出嵌入维度 (共享空间)
        depth: int = 6,                   # Transformer 层数
        num_heads: int = 8,               # 注意力头数
        mlp_ratio: float = 4.0,           # FFN 扩展倍数
        max_query_tokens: int = 128,      # 最大查询 token 数
        max_visual_tokens: int = 512,     # 视觉 token 上限 (压缩后再做注意力)
        use_bidirectional_attn: bool = True,  # 是否用双向注意力
        dropout: float = 0.0,
    ):
        """初始化嵌入预测器"""
        super().__init__()
        self.visual_dim = visual_dim
        self.predictor_dim = predictor_dim
        self.output_dim = output_dim
        self.max_query_tokens = max_query_tokens
        self.max_visual_tokens = max_visual_tokens

        # 输入投影: visual_dim → predictor_dim
        self.visual_proj = nn.Linear(visual_dim, predictor_dim)

        # 查询嵌入表 (用于无 query 时的 learnable query token)
        # Stage 1 无 query 时, 用一个可学习的 [PRED] token 替代
        self.pred_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        nn.init.trunc_normal_(self.pred_token, std=0.02)

        # Transformer 层
        self.blocks = nn.ModuleList([
            PredictorBlock(
                dim=predictor_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_bidirectional=use_bidirectional_attn,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(predictor_dim)

        # 输出投影: predictor_dim → output_dim
        self.output_proj = nn.Sequential(
            nn.Linear(predictor_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def _compress_visual_tokens(self, visual_embeds: torch.Tensor) -> torch.Tensor:
        """
        将视觉 token 序列压缩到上限，控制自注意力 O(N^2) 显存。

        使用等间距下采样（index_select）替代 adaptive_avg_pool1d。
        这样可避免某些 CUDA 版本在 AdaptiveAveragePooling backward 的内核断言。
        """
        if self.max_visual_tokens is None or self.max_visual_tokens <= 0:
            return visual_embeds
        n_tokens = visual_embeds.shape[1]
        if n_tokens <= self.max_visual_tokens:
            return visual_embeds

        # 等间距索引:
        # idx[k] = floor(k * n_tokens / max_visual_tokens), k=0..max_visual_tokens-1
        # 在 n_tokens > max_visual_tokens 时，idx 单调递增且基本覆盖全序列。
        idx = torch.div(
            torch.arange(self.max_visual_tokens, device=visual_embeds.device) * n_tokens,
            self.max_visual_tokens,
            rounding_mode="floor",
        ).long()
        return visual_embeds.index_select(1, idx)

    def forward(
        self,
        visual_embeds: torch.Tensor,       # [B, N_v, D_v] 视觉嵌入
        query_tokens: Optional[torch.Tensor] = None,  # [B, N_q, D_p] 查询嵌入 (可选)
        query_mask: Optional[torch.Tensor] = None,     # [B, N_q] padding mask (1=有效, 0=padding)
    ) -> torch.Tensor:
        """
        前向传播: 预测目标嵌入

        Returns:
            S_Y_hat: [B, D_out] 预测的目标嵌入
        """
        B = visual_embeds.shape[0]

        # 0. 视觉 token 压缩 (先压缩再投影，降低后续注意力与线性层开销)
        visual_embeds = self._compress_visual_tokens(visual_embeds)

        # 1. 投影视觉嵌入
        x_visual = self.visual_proj(visual_embeds)  # [B, N_v, D_p]
        N_v = x_visual.shape[1]

        # 2. 拼接查询或可学习 token
        if query_tokens is not None:
            # Stage 2: concat visual + query
            x = torch.cat([x_visual, query_tokens], dim=1)  # [B, N_v+N_q, D_p]

            # 构造 key_padding_mask: visual tokens 全有效, query 用 query_mask
            if query_mask is not None:
                # query_mask: 1=有效, 0=padding → key_padding_mask: True=padding
                visual_mask = torch.zeros(B, N_v, dtype=torch.bool, device=x.device)
                q_padding = ~query_mask.bool()  # 0→True (padding)
                padding_mask = torch.cat([visual_mask, q_padding], dim=1)  # [B, N_v+N_q]
            else:
                padding_mask = None
        else:
            # Stage 1: concat visual + learnable [PRED] token
            pred_tokens = self.pred_token.expand(B, -1, -1)  # [B, 1, D_p]
            x = torch.cat([x_visual, pred_tokens], dim=1)     # [B, N_v+1, D_p]
            padding_mask = None

        # 3. Transformer forward
        for blk in self.blocks:
            x = blk(x, key_padding_mask=padding_mask)

        x = self.norm(x)

        # 4. Pooling + 投影
        S_Y_hat = self.pool_and_project(x, padding_mask)
        return S_Y_hat

    def forward_with_intermediate(
        self,
        visual_embeds: torch.Tensor,
        query_tokens: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        前向传播, 同时返回中间层特征 (用于拓扑分析)

        Returns:
            S_Y_hat: [B, D_out] 预测的目标嵌入
            intermediates: List[[B, N, D_p]] 每层的中间特征
        """
        B = visual_embeds.shape[0]
        intermediates = []

        visual_embeds = self._compress_visual_tokens(visual_embeds)

        # 1. 投影
        x_visual = self.visual_proj(visual_embeds)
        N_v = x_visual.shape[1]

        # 2. 拼接
        if query_tokens is not None:
            x = torch.cat([x_visual, query_tokens], dim=1)
            if query_mask is not None:
                visual_mask = torch.zeros(B, N_v, dtype=torch.bool, device=x.device)
                q_padding = ~query_mask.bool()
                padding_mask = torch.cat([visual_mask, q_padding], dim=1)
            else:
                padding_mask = None
        else:
            pred_tokens = self.pred_token.expand(B, -1, -1)
            x = torch.cat([x_visual, pred_tokens], dim=1)
            padding_mask = None

        # 3. Transformer 逐层 forward, 收集中间特征
        for blk in self.blocks:
            x = blk(x, key_padding_mask=padding_mask)
            intermediates.append(x.clone())

        x = self.norm(x)

        # 4. Pool + project
        S_Y_hat = self.pool_and_project(x, padding_mask)
        return S_Y_hat, intermediates

    def pool_and_project(
        self,
        x: torch.Tensor,                   # [B, N, D_p] Transformer 输出
        mask: Optional[torch.Tensor] = None,  # [B, N] True=padding (key_padding_mask 格式)
    ) -> torch.Tensor:
        """
        对非 padding token 做 average pooling, 然后投影到共享嵌入空间

        Returns:
            embedding: [B, D_out]
        """
        if mask is not None:
            # mask: True=padding → 有效 mask = ~mask
            valid_mask = ~mask                         # [B, N]  True=有效
            valid_mask_f = valid_mask.unsqueeze(-1).float()  # [B, N, 1]
            # 加权平均
            x_sum = (x * valid_mask_f).sum(dim=1)      # [B, D_p]
            count = valid_mask_f.sum(dim=1).clamp(min=1)  # [B, 1]
            pooled = x_sum / count
        else:
            # 全有效, 简单平均
            pooled = x.mean(dim=1)                     # [B, D_p]

        # 投影 + L2 归一化
        projected = self.output_proj(pooled)           # [B, D_out]
        projected = F.normalize(projected, dim=-1)
        return projected
