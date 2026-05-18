"""
分割损失 (Segmentation Loss)
==============================
Dice + BCE 组合损失, 用于像素级分割任务 (DRIVE, Inria)

数据契约:
  输入 predictions 来自 SegmentationHead.forward():
    {"seg_logits": [B, C, H, W]}
  输入 batch 来自 DataLoader:
    {"masks": [B, 1, H, W] 或 [B, H, W]}

算法:
  L_seg = α * DiceLoss + β * BCEWithLogitsLoss
  - Dice loss: 1 - 2|P∩G|/(|P|+|G|), 对类别不平衡鲁棒
  - BCE loss: 像素级二值交叉熵, 提供稳定梯度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


def _dice_loss(
    pred: torch.Tensor,      # [B, C, H, W] logits
    target: torch.Tensor,    # [B, C, H, W] binary {0, 1}
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    Dice loss (可微)

    L = 1 - (2 * |P ∩ G| + smooth) / (|P| + |G| + smooth)

    对 sigmoid(pred) 和 target 计算
    """
    pred_prob = torch.sigmoid(pred)

    # Flatten spatial dims
    pred_flat = pred_prob.reshape(pred_prob.shape[0], -1)   # [B, C*H*W]
    target_flat = target.reshape(target.shape[0], -1)       # [B, C*H*W]

    intersection = (pred_flat * target_flat).sum(dim=-1)     # [B]
    union = pred_flat.sum(dim=-1) + target_flat.sum(dim=-1)  # [B]

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return (1.0 - dice).mean()


class SegmentationLoss(nn.Module):
    """
    分割损失: Dice + BCE

    L_seg = dice_weight * DiceLoss + bce_weight * BCEWithLogitsLoss
    """

    def __init__(
        self,
        num_classes: int = 1,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
        smooth: float = 1.0,
    ):
        """初始化分割损失"""
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        计算分割损失

        Args:
            predictions: {"seg_logits": [B, C, H, W]}
            batch: {"masks": [B, 1, H, W] 或 [B, H, W]}

        Returns:
            dict with: loss, loss_dice, loss_bce
        """
        logits = predictions["seg_logits"]  # [B, C, H, W]
        masks = batch.get("masks")

        if masks is None or masks.numel() == 0:
            zero = logits.sum() * 0.0
            return {"loss": zero, "loss_dice": zero, "loss_bce": zero}

        # 确保 masks 形状匹配 logits
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)  # [B, H, W] → [B, 1, H, W]

        # 如果空间尺寸不匹配, 调整 masks
        if masks.shape[2:] != logits.shape[2:]:
            masks = F.interpolate(
                masks.float(), size=logits.shape[2:],
                mode="nearest",
            )

        target = masks.float()

        # Dice loss
        loss_dice = _dice_loss(logits, target, smooth=self.smooth)

        # BCE loss
        loss_bce = F.binary_cross_entropy_with_logits(logits, target, reduction="mean")

        total = self.dice_weight * loss_dice + self.bce_weight * loss_bce

        return {
            "loss": total,
            "loss_dice": loss_dice,
            "loss_bce": loss_bce,
        }
