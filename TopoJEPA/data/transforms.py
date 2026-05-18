"""
数据增强
========
TopoJEPA 专用的数据增强

关键设计: 增强应保持拓扑不变性
  - 颜色变换 ✅ (不改变拓扑)
  - 翻转/旋转 ✅ (不改变拓扑)
  - Mosaic ⚠️ (可能改变拓扑, 需要同步更新标注)
  - 裁剪 ⚠️ (可能截断拓扑结构)

字段契约 (与全链路 ultralytics 风格一致):
  输入/输出均使用: "img", "bboxes", "cls"
  不使用: "image", "class_ids"
"""

import random
import numpy as np
import cv2
import torch
from typing import Optional, Tuple, Dict


class TopoJEPATransforms:
    """
    TopoJEPA 数据增强管道

    两个入口:
      - __call__(img, bboxes, cls): numpy 空间增强 (resize + hsv + flip)
      - normalize(img): tensor 归一化 (ToTensor + ImageNet normalize)

    增强在 numpy (H, W, C, uint8) 空间操作, 然后转 tensor
    bbox 是归一化坐标 (cx, cy, w, h ∈ [0,1]), 翻转时同步变换
    """

    def __init__(
        self,
        img_size: int = 640,
        augment: bool = True,
        hsv_h: float = 0.015,
        hsv_s: float = 0.7,
        hsv_v: float = 0.4,
        flip_lr: float = 0.5,
        flip_ud: float = 0.0,
        scale: Tuple[float, float] = (0.5, 1.5),
        translate: float = 0.1,
        mosaic: float = 0.0,                   # Mosaic 默认关闭 (保持拓扑)
    ):
        """初始化增强管道"""
        self.img_size = img_size
        self.augment = augment
        self.hsv_h = hsv_h
        self.hsv_s = hsv_s
        self.hsv_v = hsv_v
        self.flip_lr = flip_lr
        self.flip_ud = flip_ud
        self.scale = scale
        self.translate = translate
        self.mosaic = mosaic

        # ImageNet 均值/标准差 (用于 normalize)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # ============ 核心入口 ============

    def __call__(
        self,
        img: np.ndarray,                       # [H, W, C] uint8 BGR (cv2 读入)
        bboxes: Optional[torch.Tensor] = None,  # [num_obj, 4] (cx, cy, w, h) 归一化
        cls: Optional[torch.Tensor] = None,     # [num_obj, 1]
    ) -> Dict:
        """
        应用增强: numpy 空间操作 → 输出 tensor

        Returns:
            dict with:
              - "img": [C, H, W] float32 tensor (normalized)
              - "bboxes": [num_obj, 4] 同步变换后的 bbox (letterbox 坐标系)
              - "cls": [num_obj, 1] 类别 ID (不变)
        """
        orig_h, orig_w = img.shape[:2]

        # --- Resize (letterbox) ---
        img, ratio, (pad_w, pad_h) = self._letterbox(img, self.img_size)

        # --- bbox 从原图归一化坐标 → letterbox 归一化坐标 ---
        if bboxes is not None and bboxes.shape[0] > 0:
            bboxes = self._remap_bboxes(bboxes, orig_w, orig_h, ratio, pad_w, pad_h, self.img_size)

        if self.augment:
            # --- HSV 色彩增强 (拓扑安全) ---
            img = self._augment_hsv(img)

            # --- 水平翻转 ---
            if random.random() < self.flip_lr:
                img = np.fliplr(img).copy()
                if bboxes is not None and bboxes.shape[0] > 0:
                    bboxes[:, 0] = 1.0 - bboxes[:, 0]  # cx → 1 - cx

            # --- 垂直翻转 ---
            if random.random() < self.flip_ud:
                img = np.flipud(img).copy()
                if bboxes is not None and bboxes.shape[0] > 0:
                    bboxes[:, 1] = 1.0 - bboxes[:, 1]  # cy → 1 - cy

        # --- numpy → tensor + normalize ---
        img_tensor = self._to_tensor_and_normalize(img)

        return {
            "img": img_tensor,
            "bboxes": bboxes if bboxes is not None else torch.zeros(0, 4),
            "cls": cls if cls is not None else torch.zeros(0, 1),
        }

    # ============ Bbox 重映射 ============

    @staticmethod
    def _remap_bboxes(
        bboxes: torch.Tensor,                  # [N, 4] (cx, cy, w, h) 原图归一化
        orig_w: int,
        orig_h: int,
        ratio: float,
        pad_w: int,
        pad_h: int,
        target_size: int,
    ) -> torch.Tensor:
        """
        将 bbox 从原图归一化坐标映射到 letterbox 后的归一化坐标

        原图坐标 → 像素坐标 → 缩放 → 加 pad → letterbox 归一化坐标

        例: 原图 2044×3650, letterbox 到 640:
          ratio = 640/3650 ≈ 0.1753
          new_w = 640, new_h = 358
          pad_h = (640-358)//2 = 141
          原图 cy=0.6406 → pixel=0.6406*2044=1309.4 → scaled=229.5 → +pad=370.5 → norm=370.5/640=0.579
        """
        bboxes = bboxes.clone()

        # 原图归一化 → 原图像素
        bboxes[:, 0] *= orig_w   # cx_pixel
        bboxes[:, 1] *= orig_h   # cy_pixel
        bboxes[:, 2] *= orig_w   # w_pixel
        bboxes[:, 3] *= orig_h   # h_pixel

        # 原图像素 → 缩放后像素
        bboxes[:, 0] = bboxes[:, 0] * ratio + pad_w   # cx_scaled + pad
        bboxes[:, 1] = bboxes[:, 1] * ratio + pad_h   # cy_scaled + pad
        bboxes[:, 2] *= ratio                           # w_scaled
        bboxes[:, 3] *= ratio                           # h_scaled

        # 缩放后像素 → letterbox 归一化
        bboxes /= target_size

        # Clamp 到 [0, 1]
        bboxes.clamp_(0.0, 1.0)

        return bboxes

    # ============ Letterbox Resize ============

    @staticmethod
    def _letterbox(
        img: np.ndarray,
        target_size: int,
        color: Tuple[int, int, int] = (114, 114, 114),
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        等比缩放 + padding 到正方形, 不拉伸

        Returns:
            img: [target_size, target_size, C] uint8
            ratio: 缩放比例
            (pad_w, pad_h): 填充量
        """
        h, w = img.shape[:2]
        ratio = min(target_size / h, target_size / w)
        new_w, new_h = int(round(w * ratio)), int(round(h * ratio))

        if (new_w, new_h) != (w, h):
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 计算 padding
        pad_w = (target_size - new_w) // 2
        pad_h = (target_size - new_h) // 2

        # 对称填充
        img = cv2.copyMakeBorder(
            img,
            pad_h, target_size - new_h - pad_h,
            pad_w, target_size - new_w - pad_w,
            cv2.BORDER_CONSTANT, value=color,
        )
        return img, ratio, (pad_w, pad_h)

    # ============ HSV 色彩增强 ============

    def _augment_hsv(self, img: np.ndarray) -> np.ndarray:
        """
        HSV 色彩空间随机增强 (拓扑安全: 不改变形状/位置)

        参考 ultralytics/ultralytics/data/augment.py
        """
        if self.hsv_h == 0 and self.hsv_s == 0 and self.hsv_v == 0:
            return img

        # 随机增益
        r = np.random.uniform(-1, 1, 3) * [self.hsv_h, self.hsv_s, self.hsv_v] + 1

        # BGR → HSV
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        dtype = img.dtype

        # 构造查找表
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        # 应用 LUT
        im_hsv = cv2.merge((
            cv2.LUT(hue, lut_hue),
            cv2.LUT(sat, lut_sat),
            cv2.LUT(val, lut_val),
        ))
        img = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)
        return img

    # ============ numpy → tensor ============

    def _to_tensor_and_normalize(self, img: np.ndarray) -> torch.Tensor:
        """
        numpy BGR uint8 → float32 tensor [C, H, W], ImageNet normalized

        流程: BGR→RGB → /255 → CHW → normalize
        """
        # BGR → RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # HWC uint8 → CHW float32 [0, 1]
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # ImageNet normalize
        img = (img - self.mean) / self.std
        return img

    # ============ 逆归一化 (可视化用) ============

    def denormalize(self, img_tensor: torch.Tensor) -> np.ndarray:
        """
        tensor [C, H, W] → numpy [H, W, C] uint8 RGB (可视化用)
        """
        img = img_tensor.clone()
        img = img * self.std + self.mean
        img = img.clamp(0, 1)
        img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return img
