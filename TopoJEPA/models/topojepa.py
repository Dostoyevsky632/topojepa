"""
TopoJEPA 主模型
================
将所有组件整合为一个统一的模型

架构图 (标准 JEPA 范式):
  Image → VisualEncoder f_θ (online) → S_V ─┐
                                              ├→ EmbeddingPredictor h_φ → Ŝ_Y ──┐
  Query → tokenizer (持有于此, Stage 2) ─────┘                                    │ L_jepa
                                                                                   │
  Image → VisualEncoder g_ξ (EMA target, stop-grad) → pool → S_Y ────────────────┘
                                                                                   │
                                      TopologicalBranch                            │ L_topo
                                  (消费 Ŝ_Y / S_Y, 产出 diagrams)  ───────────────┘

  Stage 2 额外链路:
  feature_maps → DetectionHead / SegmentationHead → predictions ──→ L_task

训练流程:
  Stage 1 (预训练): query-free, L_jepa + L_topo
    - S_V = f_θ(image), S_Y = g_ξ(image) (EMA visual encoder)
    - Ŝ_Y = h_φ(S_V), 预测 S_Y
  Stage 2 (微调): query-conditioned, L_jepa + L_topo + L_task
    - query_texts 通过 tokenizer 编码, 作为 predictor 的条件输入

职责边界:
  - EMA: 本模型不自行实现, 由外部 utils.ema.ExponentialMovingAverage 管理
    EMA 作用在 visual_encoder (标准 JEPA: target encoder = EMA of online encoder)
  - text_encoder: 仅用于 Stage 2 query-conditioned prediction 和推理接口
  - Tokenizer: 由本模型持有, 在 forward_predict() 中将 List[str] → token tensor
  - 拓扑计算: TopologicalBranch 负责产出 diagrams, TopoJEPA 不二次计算
  - 检测/分割: DetectionHead/SegmentationHead 作为可选子模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple

from .encoder import VisualEncoder, TextEncoder
from .predictor import EmbeddingPredictor
from .topo_branch import TopologicalBranch
from .seg_head import SegmentationHead


# ============================================================
# 内置 fallback tokenizer (零外部依赖)
# ============================================================

class _CharTokenizer:
    """
    纯 Python 字符级 tokenizer — 当 transformers / sentence-transformers
    均不可用(未安装、网络异常、缓存异常)时的兜底方案

    接口兼容 HuggingFace tokenizer 的最小子集:
      - __call__(texts) → {"input_ids": LongTensor, "attention_mask": LongTensor}
      - vocab_size 属性

    映射规则: ord(char) % vocab_size, padding=0
    """

    def __init__(self, vocab_size: int = 30522, max_length: int = 128):
        self.vocab_size = vocab_size
        self.max_length = max_length

    def __call__(
        self,
        texts: List[str],
        **_kwargs,                             # 忽略 padding/truncation/return_tensors 等
    ) -> Dict[str, torch.Tensor]:
        B = len(texts)
        max_len = min(
            self.max_length,
            max((len(t) for t in texts), default=1),
        )
        max_len = max(max_len, 1)  # 至少 1

        input_ids = torch.zeros(B, max_len, dtype=torch.long)
        attention_mask = torch.zeros(B, max_len, dtype=torch.long)

        for i, text in enumerate(texts):
            for j, ch in enumerate(text[:max_len]):
                input_ids[i, j] = ord(ch) % self.vocab_size
                attention_mask[i, j] = 1
            if len(text) == 0:
                attention_mask[i, 0] = 1  # 空文本也保留一个有效 token

        return {"input_ids": input_ids, "attention_mask": attention_mask}


# ============================================================
# 标准化数据契约 (所有模块共用)
# ============================================================
# PersistenceDiagram: Tuple[torch.Tensor, torch.Tensor]
#   - births: [K] 各拓扑特征的 birth 值
#   - deaths: [K] 各拓扑特征的 death 值
#
# TopoInfo: Dict[str, Any]
#   - "pred_diagrams": List[PersistenceDiagram]  (每个同调维度一个)
#   - "target_diagrams": List[PersistenceDiagram]
#   - "feature_diagrams": Optional[List[PersistenceDiagram]]
#   - "pred_betti": torch.Tensor [max_dim+1]
#   - "target_betti": torch.Tensor [max_dim+1]
#
# ModelOutput: Dict[str, Any]
#   - "S_Y_hat": [B, D]  预测嵌入
#   - "S_Y": [B, D]  目标嵌入
#   - "topo_info": TopoInfo
#   - "feature_maps": List[torch.Tensor]  多尺度特征图
#   - "predictions": Optional[Dict]  检测头输出 (Stage 2)
# ============================================================


class DetectionNeck(nn.Module):
    """
    轻量 FPN Neck: 在 backbone feature_maps 和 DetectionHead 之间提供可训练特征融合

    架构 (自顶向下 + 横向连接):
      P5 → lateral → ─────────────────────────── out5
      P4 → lateral → + upsample(P5') → smooth → out4
      P3 → lateral → + upsample(P4') → smooth → out3

    输入: [P3, P4, P5] (来自冻结/半冻结 backbone)
    输出: [P3', P4', P5'] (通道统一为 fpn_channels, 可训练)
    """

    def __init__(
        self,
        in_channels: List[int],
        fpn_channels: int = 256,
    ):
        super().__init__()
        self.fpn_channels = fpn_channels
        n_scales = len(in_channels)

        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(ch, fpn_channels, 1) for ch in in_channels
        ])

        self.smooth_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(fpn_channels),
                nn.SiLU(inplace=True),
            )
            for _ in range(n_scales)
        ])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, feature_maps: List[torch.Tensor]) -> List[torch.Tensor]:
        laterals = [conv(fm) for conv, fm in zip(self.lateral_convs, feature_maps)]

        # top-down
        for i in range(len(laterals) - 1, 0, -1):
            h, w = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=(h, w), mode="nearest"
            )

        return [smooth(lat) for smooth, lat in zip(self.smooth_convs, laterals)]


class DetectionHead(nn.Module):
    """
    轻量检测头: 从多尺度特征图产出检测预测

    封装 ultralytics Detect 模块, 或自定义简化版本
    确保输出格式与 v8DetectionLoss 兼容:
      {"boxes": [B, reg, A], "scores": [B, nc, A], "feats": [feat_maps]}
    """

    def __init__(
        self,
        in_channels: List[int],               # 各尺度特征图通道数 [P3_ch, P4_ch, P5_ch]
        num_classes: int = 5,
        reg_max: int = 16,
    ):
        """初始化检测头"""
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.num_levels = len(in_channels)

        # 每个尺度: 分类头 + 回归头
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()

        for ch in in_channels:
            # 分类分支: 2 层 Conv → num_classes
            self.cls_convs.append(nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1),
                nn.BatchNorm2d(ch),
                nn.SiLU(inplace=True),
                nn.Conv2d(ch, ch, 3, padding=1),
                nn.BatchNorm2d(ch),
                nn.SiLU(inplace=True),
            ))
            self.cls_preds.append(nn.Conv2d(ch, num_classes, 1))

            # 回归分支: 2 层 Conv → reg_max * 4
            self.reg_convs.append(nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1),
                nn.BatchNorm2d(ch),
                nn.SiLU(inplace=True),
                nn.Conv2d(ch, ch, 3, padding=1),
                nn.BatchNorm2d(ch),
                nn.SiLU(inplace=True),
            ))
            self.reg_preds.append(nn.Conv2d(ch, reg_max * 4, 1))

    def forward(
        self,
        feature_maps: List[torch.Tensor],      # [P3, P4, P5]
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            predictions: {
                "boxes": [B, reg_max*4, num_anchors],
                "scores": [B, num_classes, num_anchors],
                "feats": feature_maps  (透传, 供 loss 计算 anchors)
            }
        """
        all_cls = []
        all_reg = []

        for i, fm in enumerate(feature_maps):
            # 分类
            cls_feat = self.cls_convs[i](fm)
            cls_pred = self.cls_preds[i](cls_feat)         # [B, nc, H_i, W_i]
            cls_pred = cls_pred.flatten(2)                 # [B, nc, H_i*W_i]
            all_cls.append(cls_pred)

            # 回归
            reg_feat = self.reg_convs[i](fm)
            reg_pred = self.reg_preds[i](reg_feat)         # [B, reg*4, H_i, W_i]
            reg_pred = reg_pred.flatten(2)                 # [B, reg*4, H_i*W_i]
            all_reg.append(reg_pred)

        return {
            "boxes": torch.cat(all_reg, dim=2),   # [B, reg*4, total_anchors]
            "scores": torch.cat(all_cls, dim=2),   # [B, nc, total_anchors]
            "feats": feature_maps,                 # 透传
        }

    @staticmethod
    def from_ultralytics_detect(
        detect_module: "nn.Module",
    ) -> "DetectionHead":
        """
        从 ultralytics Detect() 模块反构建 DetectionHead

        ultralytics.nn.modules.head.Detect 的关键属性:
          - nc: int  (类别数)
          - reg_max: int  (DFL 回归最大值)
          - cv2: nn.ModuleList  (回归分支, 每尺度 Sequential)
          - cv3: nn.ModuleList  (分类分支, 每尺度 Sequential)
          - ch: list[int] 或可从 cv2[i][0].conv.in_channels 推断

        构建逻辑:
          1. 读取 nc, reg_max
          2. 推断 in_channels
          3. 创建 DetectionHead (结构匹配)
          4. 尽量拷贝权重 (结构不完全对齐时仅做 best-effort)
        """
        # ---- 提取参数 ----
        nc = getattr(detect_module, 'nc', 5)
        reg_max = getattr(detect_module, 'reg_max', 16)

        # 推断 in_channels: 优先从 cv2 (回归分支) 获取输入通道数
        in_channels: List[int] = []
        cv2_list = getattr(detect_module, 'cv2', None)
        cv3_list = getattr(detect_module, 'cv3', None)

        if cv2_list is not None:
            for branch in cv2_list:
                # branch 通常是 Sequential, 第一层是 Conv (ultralytics 自定义)
                first = branch[0] if hasattr(branch, '__getitem__') else None
                if first is not None:
                    # ultralytics Conv 包装器: first.conv 是实际的 nn.Conv2d
                    conv = getattr(first, 'conv', first)
                    if hasattr(conv, 'in_channels'):
                        in_channels.append(conv.in_channels)
                    elif hasattr(conv, 'weight'):
                        in_channels.append(conv.weight.shape[1])

        if not in_channels:
            # 回退: 尝试从 detect_module.ch 属性
            ch = getattr(detect_module, 'ch', None)
            if ch is not None and hasattr(ch, '__len__'):
                in_channels = list(ch)
            else:
                raise ValueError(
                    "Cannot infer in_channels from detect_module. "
                    "Ensure it is an ultralytics Detect module with cv2/cv3 attributes."
                )

        # ---- 构建 DetectionHead ----
        head = DetectionHead(
            in_channels=in_channels,
            num_classes=nc,
            reg_max=reg_max,
        )

        # ---- Best-effort 权重拷贝 ----
        # ultralytics cv3 → cls 分支, cv2 → reg 分支
        # 结构可能不完全对齐 (ultralytics 用自定义 Conv 而非 nn.Sequential)
        # 只拷贝最终 1×1 预测层 (最关键)
        if cv3_list is not None:
            for i, branch in enumerate(cv3_list):
                if i >= len(head.cls_preds):
                    break
                # 最后一层通常是 1×1 conv 输出
                last = branch[-1] if hasattr(branch, '__getitem__') else None
                src_conv = getattr(last, 'conv', last) if last is not None else None
                if src_conv is not None and hasattr(src_conv, 'weight'):
                    tgt_conv = head.cls_preds[i]
                    if src_conv.weight.shape == tgt_conv.weight.shape:
                        tgt_conv.weight.data.copy_(src_conv.weight.data)
                        if src_conv.bias is not None and tgt_conv.bias is not None:
                            tgt_conv.bias.data.copy_(src_conv.bias.data)

        if cv2_list is not None:
            for i, branch in enumerate(cv2_list):
                if i >= len(head.reg_preds):
                    break
                last = branch[-1] if hasattr(branch, '__getitem__') else None
                src_conv = getattr(last, 'conv', last) if last is not None else None
                if src_conv is not None and hasattr(src_conv, 'weight'):
                    tgt_conv = head.reg_preds[i]
                    if src_conv.weight.shape == tgt_conv.weight.shape:
                        tgt_conv.weight.data.copy_(src_conv.weight.data)
                        if src_conv.bias is not None and tgt_conv.bias is not None:
                            tgt_conv.bias.data.copy_(src_conv.bias.data)

        return head


class TopoJEPA(nn.Module):
    """
    TopoJEPA: 拓扑正则化的联合嵌入预测架构

    核心创新:
      1. 在 JEPA 嵌入空间中引入持久同调约束
      2. 双空间拓扑保真: 特征空间 + 嵌入空间
      3. 自适应拓扑权重调度 (训练初期低, 后期高)

    职责:
      - 持有所有子模块 (encoder, predictor, text_encoder, topo_branch, detection_head)
      - 持有 tokenizer (query text → token tensor 的转换)
      - 不持有 EMA (由外部 ExponentialMovingAverage 管理)
      - 不持有 target_encoder (由外部 EMA.build_target() 创建并管理)
    """

    def __init__(
        self,
        visual_encoder: VisualEncoder,
        text_encoder: TextEncoder,
        predictor: EmbeddingPredictor,
        topo_branch: TopologicalBranch,
        detection_neck: Optional[DetectionNeck] = None,  # Stage 2 FPN neck
        detection_head: Optional[DetectionHead] = None,  # Stage 2 检测
        segmentation_head: Optional[SegmentationHead] = None,  # Stage 2 分割
        embed_dim: int = 768,
        tokenizer_name: str = "all-MiniLM-L6-v2",  # query tokenizer
    ):
        """初始化 TopoJEPA"""
        super().__init__()
        self.visual_encoder = visual_encoder
        self.text_encoder = text_encoder
        self.predictor = predictor
        self.topo_branch = topo_branch
        self.detection_neck = detection_neck
        self.detection_head = detection_head
        self.segmentation_head = segmentation_head
        self.embed_dim = embed_dim
        # tokenizer 延迟初始化 (避免 import 时加载)
        self._tokenizer_name = tokenizer_name
        self._tokenizer = None
        self._query_embed = None  # 延迟初始化

        # 拓扑计算频率控制: 每 N 步才真正计算一次 PH, 其余步复用缓存
        # 设为 1 = 每步都算 (无缓存), >1 = 每 N 步算一次
        self._topo_every_n_steps: int = 1
        self._topo_step_counter: int = 0
        self._cached_topo_info: Optional[Dict] = None

    # ============ Tokenizer (归属于本模型) ============

    # fallback tokenizer 的 vocab size (与 TextEncoder._FALLBACK_VOCAB_SIZE 对齐)
    _FALLBACK_VOCAB_SIZE = 30522

    @property
    def tokenizer(self):
        """
        延迟加载 tokenizer

        加载顺序:
          1. transformers.AutoTokenizer (最佳)
          2. sentence-transformers 的内置 tokenizer
          3. 内置 _CharTokenizer (纯 Python, 零依赖)

        任何外部库的 ImportError / 网络异常 / 缓存异常都会被捕获并 fallback
        """
        if self._tokenizer is not None:
            return self._tokenizer

        # 尝试 transformers
        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_name)
            return self._tokenizer
        except Exception:
            pass

        # 尝试 sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            st = SentenceTransformer(self._tokenizer_name)
            self._tokenizer = st.tokenizer
            return self._tokenizer
        except Exception:
            pass

        # 内置 fallback
        self._tokenizer = _CharTokenizer(
            vocab_size=self._FALLBACK_VOCAB_SIZE,
            max_length=self.predictor.max_query_tokens,
        )
        return self._tokenizer

    def _ensure_query_embed(self, device: torch.device):
        """延迟初始化 query embedding 层 (tokenizer 加载后才知道 vocab_size)"""
        if self._query_embed is not None:
            return
        vocab_size = getattr(self.tokenizer, 'vocab_size', self._FALLBACK_VOCAB_SIZE)
        predictor_dim = self.predictor.predictor_dim
        self._query_embed = nn.Embedding(vocab_size, predictor_dim).to(device)
        nn.init.trunc_normal_(self._query_embed.weight, std=0.02)

    def tokenize_query(
        self,
        query_texts: List[str],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将查询文本转换为 token tensor

        Returns:
            query_tokens: [B, N_q, D_p] 嵌入后的 token (经过 embedding layer)
            query_mask: [B, N_q] padding mask (1=有效, 0=padding)
        """
        self._ensure_query_embed(device)

        tok = self.tokenizer

        if isinstance(tok, _CharTokenizer):
            # 内置 fallback: 直接返回 tensor
            encoded = tok(query_texts)
        else:
            # HuggingFace tokenizer
            encoded = tok(
                query_texts,
                padding=True,
                truncation=True,
                max_length=self.predictor.max_query_tokens,
                return_tensors="pt",
            )

        input_ids = encoded["input_ids"].to(device)          # [B, N_q]
        attention_mask = encoded["attention_mask"].to(device)  # [B, N_q] 1=有效

        # Embedding
        query_tokens = self._query_embed(input_ids)           # [B, N_q, D_p]
        query_mask = attention_mask                            # [B, N_q]

        return query_tokens, query_mask

    # ============ 前向传播子步骤 ============

    def forward_visual(
        self,
        images: torch.Tensor,                 # [B, C, H, W]
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        视觉前向: 图像 → 视觉嵌入 + 多尺度特征图

        Returns:
            S_V: [B, N, D] 视觉嵌入序列 (patch tokens)
            feature_maps: [P3, P4, P5] 多尺度特征图, 或 None (ViT 模式)
        """
        return self.visual_encoder.forward_with_features(images)

    @staticmethod
    def _has_query(query_texts: Optional[List[str]]) -> bool:
        """判断 query 列表是否有效 (非 None、非空列表、且至少有一条非空字符串)"""
        if query_texts is None:
            return False
        return any(q for q in query_texts)  # 全空字符串 → False

    def forward_predict(
        self,
        S_V: torch.Tensor,                    # [B, N, D] 视觉嵌入
        query_texts: Optional[List[str]] = None,  # 查询文本 (Stage 2)
    ) -> torch.Tensor:
        """
        预测前向: 视觉嵌入 + 查询 → 预测目标嵌入

        内部流程:
          1. 判断 query 是否有效 (None / 空列表 / 全空字符串 均视为无 query)
          2. 有 query → self.tokenize_query() → predictor(S_V, tokens, mask)
          3. 无 query → predictor(S_V)  (query-free, Stage 1)

        Stage 1 的 DataLoader 会传 query_texts=["", "", ...],
        本方法通过 _has_query() 识别为无 query, 不走 tokenizer 路径。

        Returns:
            S_Y_hat: [B, D] 预测的目标嵌入
        """
        if self._has_query(query_texts):
            # Stage 2: query-conditioned prediction
            device = S_V.device
            query_tokens, query_mask = self.tokenize_query(query_texts, device)
            S_Y_hat = self.predictor(S_V, query_tokens, query_mask)
        else:
            # Stage 1: query-free, 用 learnable [PRED] token
            S_Y_hat = self.predictor(S_V)
        return S_Y_hat

    def forward_target(
        self,
        images: torch.Tensor,                     # [B, C, H, W]
        target_encoder: Optional[nn.Module] = None,  # 外部传入的 EMA visual encoder
    ) -> torch.Tensor:
        """
        目标前向 (标准 JEPA): 图像 → EMA visual encoder → 池化 → S_Y

        标准 JEPA 范式: target encoder g_ξ 是 online encoder f_θ 的 EMA 副本,
        对同一张图像编码, 产出 stop-gradient 的目标嵌入。

        如果提供了 target_encoder (EMA visual encoder), 用它编码 (训练时)
        否则用 self.visual_encoder (推理时)

        Returns:
            S_Y: [B, D] 目标嵌入 (池化后)
        """
        encoder = target_encoder if target_encoder is not None else self.visual_encoder
        with torch.no_grad():
            S_V_target = encoder.forward_with_features(images)[0]  # [B, N, D]
        # 池化: mean over token dimension → [B, D]
        S_Y = S_V_target.mean(dim=1)
        return S_Y

    def forward_topo(
        self,
        S_Y_hat: torch.Tensor,                # [B, D] 预测嵌入 (已池化)
        S_Y: torch.Tensor,                    # [B, D] 目标嵌入 (已池化)
        feature_maps: Optional[List[torch.Tensor]] = None,
    ) -> Dict:
        """
        拓扑前向: 委托给 TopologicalBranch 计算

        接口约定:
          - 输入: [B, D] 池化后的嵌入向量 (不是 [B, N, D])
          - TopologicalBranch 将 batch 中的 B 个 D 维向量视为点云
          - 输出: 标准化的 TopoInfo 字典

        Returns:
            TopoInfo dict (见文件顶部契约定义)
        """
        return self.topo_branch(S_Y_hat, S_Y, feature_maps)

    def forward_detect(
        self,
        feature_maps: List[torch.Tensor],
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        检测前向: 多尺度特征图 → (可选 Neck) → 检测预测 (仅 Stage 2)

        Returns:
            predictions: {"boxes", "scores", "feats"} 或 None (无检测头时)
        """
        if self.detection_head is None:
            return None
        fms = self.detection_neck(feature_maps) if self.detection_neck is not None else feature_maps
        return self.detection_head(fms)

    def forward_segment(
        self,
        feature_maps: List[torch.Tensor],
        target_size: Optional[int] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        分割前向: 多尺度特征图 → 分割预测 (仅 Stage 2, DRIVE/Inria)

        Returns:
            predictions: {"seg_logits": [B, C, H, W]} 或 None (无分割头时)
        """
        if self.segmentation_head is None:
            return None
        return self.segmentation_head(feature_maps, target_size=target_size)

    def set_topo_interval(self, every_n_steps: int) -> None:
        """
        设置拓扑计算频率

        Args:
            every_n_steps: 每 N 步真正计算一次 PH, 其余步复用缓存的 topo_info
                1 = 每步都算 (默认, 无缓存)
                4 = 每 4 步算一次, 节省 ~75% 拓扑开销
        """
        self._topo_every_n_steps = max(1, every_n_steps)
        self._topo_step_counter = 0
        self._cached_topo_info = None

    # ============ 完整前向 ============

    def forward(
        self,
        images: torch.Tensor,                 # [B, C, H, W]
        query_texts: Optional[List[str]] = None,  # 查询文本 (Stage 2)
        target_encoder: Optional[nn.Module] = None,  # 外部 EMA visual encoder
    ) -> Dict:
        """
        完整前向传播 (标准 JEPA 范式)

        数据流:
          1. images → online visual encoder f_θ → S_V [B, N, D] + feature_maps
          2. S_V + query_texts → predictor h_φ → S_Y_hat [B, D]
          3. images → EMA visual encoder g_ξ → pool → S_Y [B, D] (stop-grad)
          4. (S_Y_hat, S_Y, feature_maps) → TopologicalBranch → topo_info
          5. feature_maps → DetectionHead/SegmentationHead → predictions (Stage 2)

        Returns:
            ModelOutput dict:
              - "S_Y_hat": [B, D] 预测嵌入
              - "S_Y": [B, D] 目标嵌入
              - "topo_info": TopoInfo dict (由 TopologicalBranch 产出)
              - "feature_maps": List[Tensor] 多尺度特征图
              - "predictions": Dict 检测/分割头输出 (Stage 2, 否则 None)
        """
        # 1. Visual encoding (online encoder f_θ)
        S_V, feature_maps = self.forward_visual(images)

        # 2. Embedding prediction (predictor h_φ)
        S_Y_hat = self.forward_predict(S_V, query_texts)

        # 3. Target encoding (EMA visual encoder g_ξ, stop-grad)
        S_Y = self.forward_target(images, target_encoder)

        # 4. Topological analysis (输入 [B, D], 不是 [B, N, D])
        # 性能优化: 每 N 步才真正计算 PH, 其余步复用缓存
        should_compute_topo = (
            self._topo_every_n_steps <= 1
            or self._cached_topo_info is None
            or (self._topo_step_counter % self._topo_every_n_steps == 0)
            or not self.training  # eval 模式始终计算
        )
        if should_compute_topo:
            topo_info = self.forward_topo(S_Y_hat, S_Y, feature_maps)
            # 缓存 detach 副本 (不占梯度图)
            self._cached_topo_info = {
                k: (v.detach() if isinstance(v, torch.Tensor) else
                    [(b.detach(), d.detach()) for b, d in v] if isinstance(v, list) else v)
                for k, v in topo_info.items()
            }
        else:
            topo_info = self._cached_topo_info
        if self.training:
            self._topo_step_counter += 1

        # 5. Detection (Stage 2 only, RDD)
        predictions = self.forward_detect(feature_maps) if feature_maps is not None else None

        # 6. Segmentation (Stage 2 only, DRIVE/Inria)
        seg_predictions = self.forward_segment(feature_maps) if feature_maps is not None else None

        return {
            "S_Y_hat": S_Y_hat,
            "S_Y": S_Y,
            "topo_info": topo_info,
            "feature_maps": feature_maps,
            "predictions": predictions,
            "seg_predictions": seg_predictions,
        }

    # ============ 推理接口 ============

    @torch.no_grad()
    def classify(
        self,
        images: torch.Tensor,                 # [B, C, H, W]
        candidate_labels: List[str],          # 候选类别文本列表
        query: Optional[str] = None,
    ) -> torch.Tensor:
        """
        开放词汇分类: 编码候选标签, 与预测嵌入比较, 选最近的

        流程:
          1. 编码所有候选标签 → label_embeds [K, D]
          2. 编码图像 → S_V, 然后 predict → S_Y_hat [B, D]
          3. 计算余弦相似度 → [B, K]
          4. argmax → class_indices [B]

        Returns:
            class_indices: [B] 预测的类别索引
        """
        # 编码候选标签
        label_embeds = self.text_encoder(candidate_labels)  # [K, D]

        # 编码图像 + 预测
        S_V, _ = self.forward_visual(images)
        query_texts = [query] * images.shape[0] if query else None
        S_Y_hat = self.forward_predict(S_V, query_texts)  # [B, D]

        # 余弦相似度 (S_Y_hat 和 label_embeds 已经 L2 归一化)
        similarity = torch.matmul(S_Y_hat, label_embeds.T)  # [B, K]
        class_indices = similarity.argmax(dim=-1)            # [B]
        return class_indices

    @torch.no_grad()
    def retrieve(
        self,
        query_text: str,
        image_embeddings: torch.Tensor,        # [N, D] 预计算的图像嵌入库
    ) -> torch.Tensor:
        """
        文本→图像检索

        流程:
          1. 编码查询文本 → query_embed [1, D]
          2. 与图像嵌入库计算余弦相似度

        Returns:
            scores: [N] 相似度分数
        """
        query_embed = self.text_encoder([query_text])  # [1, D]
        # 图像嵌入已经 L2 归一化
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        scores = torch.matmul(image_embeddings, query_embed.squeeze(0))  # [N]
        return scores
