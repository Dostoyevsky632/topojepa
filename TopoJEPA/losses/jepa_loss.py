"""
JEPA 损失函数
=============
嵌入空间的预测损失 + 反坍塌正则化

参考:
  - jepa/app/vjepa/train.py: L1 loss + variance regularization
  - VL-JEPA: InfoNCE (双向对比损失)
  - 我们同时支持两种, 可通过配置选择

InfoNCE 的数学分解 (Wang & Isola, 2020):
  L_infonce = L_alignment + L_uniformity
  - alignment: 最小化正对的嵌入距离
  - uniformity: 最大化 batch 内嵌入的均匀分布
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class JEPALoss(nn.Module):
    """
    JEPA 嵌入预测损失

    支持模式:
      - "l1": |Ŝ_Y - S_Y|  (原始 V-JEPA)
      - "l2": ||Ŝ_Y - S_Y||²
      - "cosine": 1 - cos(Ŝ_Y, S_Y)
      - "infonce": 双向 InfoNCE (VL-JEPA, 推荐)
    """

    def __init__(
        self,
        loss_type: str = "infonce",           # "l1" | "l2" | "cosine" | "infonce"
        temperature: float = 0.07,            # InfoNCE 温度参数
        reg_coeff: float = 0.0,               # 方差正则化系数 (V-JEPA 式)
        label_smoothing: float = 0.0,         # 标签平滑
    ):
        """初始化 JEPA 损失"""
        super().__init__()
        assert loss_type in ("l1", "l2", "cosine", "infonce"), \
            f"Unsupported loss_type: {loss_type}"
        self.loss_type = loss_type
        self.temperature = temperature
        self.reg_coeff = reg_coeff
        self.label_smoothing = label_smoothing

    def forward(
        self,
        pred_embed: torch.Tensor,             # [B, D] 预测嵌入 Ŝ_Y
        target_embed: torch.Tensor,            # [B, D] 目标嵌入 S_Y
        target_texts: Optional[List[str]] = None,
    ) -> dict:
        """
        计算嵌入预测损失

        Returns:
            dict with:
              - "loss": 总损失 (标量)
              - "loss_align": 对齐损失
              - "loss_uniform": 均匀性损失 (InfoNCE 模式)
              - "loss_reg": 方差正则化 (V-JEPA 模式)
              - "similarity": 平均余弦相似度 (监控用)
        """
        # L2 归一化 (对 infonce 和 cosine 是必要的; 对 l1/l2 也归一化以保持量纲一致)
        # 数值稳定: 在 fp32 中做归一化和相似度，避免 amp/fp16 下溢出或溢出。
        pred_norm = F.normalize(pred_embed.float(), dim=-1)
        target_norm = F.normalize(target_embed.float(), dim=-1)

        # 监控: 平均余弦相似度
        similarity = (pred_norm * target_norm).sum(dim=-1).mean()

        # --- 主损失 ---
        if self.loss_type == "infonce":
            loss_align = self.infonce_loss(
                pred_norm,
                target_norm,
                target_texts=target_texts,
            )
            # InfoNCE 隐含了 alignment + uniformity, 不单独分开
            result = {
                "loss_align": loss_align,
                "loss_uniform": torch.tensor(0.0, device=pred_embed.device),
            }
        elif self.loss_type == "l1":
            loss_align = F.l1_loss(pred_norm, target_norm.detach())
            result = {
                "loss_align": loss_align,
                "loss_uniform": torch.tensor(0.0, device=pred_embed.device),
            }
        elif self.loss_type == "l2":
            loss_align = F.mse_loss(pred_norm, target_norm.detach())
            result = {
                "loss_align": loss_align,
                "loss_uniform": torch.tensor(0.0, device=pred_embed.device),
            }
        elif self.loss_type == "cosine":
            # 1 - cos(pred, target), 在 L2 归一化后等于 0.5 * ||pred - target||^2
            loss_align = (1.0 - (pred_norm * target_norm.detach()).sum(dim=-1)).mean()
            result = {
                "loss_align": loss_align,
                "loss_uniform": torch.tensor(0.0, device=pred_embed.device),
            }
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        # --- 方差正则化 ---
        loss_reg = torch.tensor(0.0, device=pred_embed.device)
        if self.reg_coeff > 0.0:
            loss_reg = self.variance_regularization(pred_embed)

        result["loss_reg"] = loss_reg
        result["similarity"] = similarity.detach()
        result["loss"] = result["loss_align"] + self.reg_coeff * loss_reg

        return result

    def infonce_loss(
        self,
        pred: torch.Tensor,                   # [B, D] L2-normalized
        target: torch.Tensor,                  # [B, D] L2-normalized
        target_texts: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        双向 InfoNCE 损失

        L = 0.5 * (CE(sim_p2t, labels) + CE(sim_t2p, labels))
        其中 sim = pred @ target.T / temperature

        target 走 detach (类似 JEPA 的 stop-gradient on Y-encoder)
        """
        B = pred.shape[0]
        # 相似度矩阵
        sim = torch.matmul(pred, target.detach().T) / self.temperature  # [B, B]
        labels = torch.arange(B, device=pred.device)

        # 若 batch 内 target_text 存在重复, 使用 multi-positive InfoNCE
        # 避免把语义等价样本错误当作负样本, 使 Stage1 不再卡在 ln(B) 附近。
        pos_mask = self._build_positive_mask(target_texts, B, pred.device)
        if pos_mask is not None:
            loss_p2t = self._multi_positive_nce(sim, pos_mask)
            loss_t2p = self._multi_positive_nce(sim.T, pos_mask.T)
        else:
            # 标准单正样本 InfoNCE
            loss_p2t = F.cross_entropy(sim, labels, label_smoothing=self.label_smoothing)
            loss_t2p = F.cross_entropy(sim.T, labels, label_smoothing=self.label_smoothing)
        return 0.5 * (loss_p2t + loss_t2p)

    @staticmethod
    def _build_positive_mask(
        target_texts: Optional[List[str]],
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """
        构建 multi-positive mask:
          pos_mask[i, j] = True 表示样本 i 与 j 语义等价 (同 target_text)
        """
        if target_texts is None or len(target_texts) != batch_size:
            return None

        groups = {}
        for i, t in enumerate(target_texts):
            key = t if isinstance(t, str) else str(t)
            groups.setdefault(key, []).append(i)

        # 没有重复文本时退化为标准单正样本 InfoNCE, 保持 CE + label_smoothing 语义
        if all(len(idxs) == 1 for idxs in groups.values()):
            return None

        pos_mask = torch.zeros(batch_size, batch_size, dtype=torch.bool, device=device)
        for idxs in groups.values():
            idx = torch.tensor(idxs, device=device, dtype=torch.long)
            pos_mask[idx.unsqueeze(1), idx.unsqueeze(0)] = True
        return pos_mask

    @staticmethod
    def _multi_positive_nce(
        logits: torch.Tensor,                 # [B, B]
        pos_mask: torch.Tensor,               # [B, B] bool
    ) -> torch.Tensor:
        """
        Multi-positive InfoNCE:
          L_i = -log ( sum_{j in Pos(i)} exp(logits_ij) / sum_k exp(logits_ik) )

        该形式把“同文本样本”从负样本中剔除, 但不强迫所有正样本都同时靠近;
        对 batch 内重复样本更稳健。
        """
        pos_logits = logits.masked_fill(~pos_mask, float("-inf"))
        log_pos = torch.logsumexp(pos_logits, dim=1)
        log_all = torch.logsumexp(logits, dim=1)
        loss = -(log_pos - log_all)
        return loss.mean()

    def variance_regularization(
        self,
        embeddings: torch.Tensor,              # [B, D]
    ) -> torch.Tensor:
        """
        方差正则化: 防止嵌入坍塌到同一个点
        L_reg = mean(ReLU(1 - std(z_d)))

        沿 batch 维度计算每个维度的标准差,
        如果 std < 1 → 有惩罚 (鼓励 std >= 1, 即各维度充分展开)

        参考 jepa/app/vjepa/train.py line 448-449,
        以及 VICReg (Bardes et al., 2022)
        """
        # [D] 每个维度在 batch 上的标准差
        # unbiased=False: 避免 B=1 时 Bessel 修正除以 0 → NaN
        std = embeddings.std(dim=0, unbiased=False)  # [D]
        # ReLU(1 - std): 当 std < 1 时有惩罚
        loss = F.relu(1.0 - std).mean()
        return loss
