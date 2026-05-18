"""
分割头 (Segmentation Head)
===========================
FPN-style decoder: 从多尺度特征图产出像素级分割预测

架构:
  P5 → lateral_conv → upsample → ─┐
  P4 → lateral_conv → ────────── + → smooth_conv → upsample → ─┐
  P3 → lateral_conv → ──────────────────────────────────────── + → smooth_conv
                                                                     ↓
                                                              upsample → 1x1 conv → logits [B, C, H, W]

输入: feature_maps [P3, P4, P5] 来自 VisualEncoder
输出: {"seg_logits": [B, num_classes, H_out, W_out]}
  H_out, W_out 是输入图像尺寸 (通过上采样恢复)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional


class SegmentationHead(nn.Module):
    """
    FPN-style 分割解码器

    从多尺度特征图 [P3, P4, P5] 解码为像素级分割 logits
    """

    def __init__(
        self,
        in_channels: List[int],           # 各尺度特征图通道数 [P3_ch, P4_ch, P5_ch]
        num_classes: int = 1,             # 分割类别数 (DRIVE/Inria 二值分割 = 1)
        fpn_channels: int = 256,          # FPN 内部通道数
        output_stride: int = 4,           # 输出相对于输入的下采样倍率 (最终上采样到此)
    ):
        """初始化分割头"""
        super().__init__()
        self.num_classes = num_classes
        self.fpn_channels = fpn_channels
        self.output_stride = output_stride

        n_scales = len(in_channels)

        # Lateral connections: 1x1 conv 统一通道数
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(ch, fpn_channels, 1) for ch in in_channels
        ])

        # Smooth convolutions: 3x3 conv 消除上采样锯齿
        self.smooth_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(fpn_channels),
                nn.ReLU(inplace=True),
            )
            for _ in range(n_scales)
        ])

        # 最终预测头: 3x3 conv + 1x1 conv
        self.head = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_channels, num_classes, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        feature_maps: List[torch.Tensor],     # [P3, P4, P5] 各 [B, C_i, H_i, W_i]
        target_size: Optional[int] = None,     # 目标输出尺寸 (H=W), None 则自动推断
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            feature_maps: 多尺度特征图 [P3, P4, P5]
            target_size: 输出空间尺寸, None 则上采样到 P3 尺寸的 output_stride 倍

        Returns:
            {"seg_logits": [B, num_classes, H_out, W_out]}
        """
        assert len(feature_maps) == len(self.lateral_convs), \
            f"Expected {len(self.lateral_convs)} feature maps, got {len(feature_maps)}"

        # Lateral connections
        laterals = [conv(fm) for conv, fm in zip(self.lateral_convs, feature_maps)]

        # Top-down pathway (从最粗到最细)
        for i in range(len(laterals) - 1, 0, -1):
            h, w = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=(h, w), mode="bilinear", align_corners=False
            )

        # Smooth
        fpn_outs = [smooth(lat) for smooth, lat in zip(self.smooth_convs, laterals)]

        # 使用最高分辨率 (P3 级) 的特征
        out = fpn_outs[0]

        # 上采样到目标尺寸
        if target_size is not None:
            out = F.interpolate(out, size=(target_size, target_size),
                                mode="bilinear", align_corners=False)

        # 预测
        seg_logits = self.head(out)

        return {"seg_logits": seg_logits}
