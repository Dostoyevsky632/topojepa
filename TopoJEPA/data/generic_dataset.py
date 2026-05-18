"""
通用数据集适配器
=================
为多数据集验证 (DRIVE, Inria 等) 提供统一接口

论文通用性验证需要至少 2-3 个数据集:
  1. RDD (道路损伤) ← 主数据集, 用 RDDTopoDataset
  2. DRIVE / CHASE_DB1 (视网膜血管) ← 拓扑结构丰富
  3. Inria Aerial (遥感建筑) ← 边界拓扑完整性

所有数据集适配到统一输出格式, 与 RDDTopoDataset.__getitem__() 一致

依赖: cv2 (必须), numpy, torch
"""

import logging
import random
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Optional, Dict, List, Tuple
from pathlib import Path

from .transforms import TopoJEPATransforms

logger = logging.getLogger("topojepa.data")


# ============================================================
# 通用基类
# ============================================================

class GenericTopoDataset(Dataset):
    """
    通用数据集基类

    子类只需实现:
      - _scan_samples() → 填充 self.samples
      - _load_sample(idx) → (image_bgr, cls_tensor, bboxes_tensor)
      - _generate_target_text(cls, bboxes) → str

    输出格式统一 (per-sample, ultralytics 字段名):
      {
        "img": [C, H, W],
        "target_text": str,
        "query_text": str,
        "bboxes": [num_obj, 4],     # (cx, cy, w, h) 归一化
        "cls": [num_obj, 1],        # 类别 ID (float tensor)
        "img_path": str,
      }

    经 collate_fn 后变为 per-batch:
      "cls" → [N_total, 1], "bboxes" → [N_total, 4], "batch_idx" → [N_total]
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        img_size: int = 640,
        text_strategy: str = "class_name",
        augment: bool = True,
    ):
        """初始化通用数据集"""
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = img_size
        self.text_strategy = text_strategy
        self.augment = augment

        # 增强管道 (复用 RDD 的 transforms)
        self.transforms = TopoJEPATransforms(
            img_size=img_size,
            augment=(augment and split == "train"),
        )

        # 子类在 _scan_samples() 中填充
        self.samples: List = []
        self._scan_samples()

    def _scan_samples(self):
        """子类实现: 扫描数据集目录, 填充 self.samples"""
        raise NotImplementedError(
            f"{self.__class__.__name__}._scan_samples 尚未实现。"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """
        统一 __getitem__ 流程:
          1. _load_sample() 加载 numpy 图像 + 标注 (+ 可选 mask)
          2. transforms() 增强
          3. _generate_target_text() 生成文本
          4. 组装输出 dict
        """
        sample = self._load_sample(idx)
        # 兼容: _load_sample 返回 4 元组 (img, cls, bboxes, path) 或
        #        5 元组 (img, cls, bboxes, path, mask)
        if len(sample) == 5:
            img_bgr, cls, bboxes, img_path, mask = sample
        else:
            img_bgr, cls, bboxes, img_path = sample
            mask = None

        # 增强 (letterbox + hsv + flip)
        augmented = self.transforms(img_bgr, bboxes, cls)

        # 文本
        target_text = self._generate_target_text(augmented["cls"], augmented["bboxes"])
        query_text = self._generate_query_text(augmented["cls"])

        result = {
            "img": augmented["img"],             # [C, H, W]
            "target_text": target_text,
            "query_text": query_text,
            "bboxes": augmented["bboxes"],       # [num_obj, 4]
            "cls": augmented["cls"],             # [num_obj, 1]
            "img_path": str(img_path),
        }

        # 分割 mask (DRIVE/Inria 等分割数据集提供)
        if mask is not None:
            # mask: [H, W] uint8 → 归一化 float tensor, resize 到 img_size
            mask_t = torch.from_numpy(mask).float() / 255.0  # [H, W]
            mask_t = F.interpolate(
                mask_t.unsqueeze(0).unsqueeze(0),
                size=(self.img_size, self.img_size),
                mode="nearest",
            ).squeeze(0)  # [1, H, W]
            result["mask"] = mask_t

        return result

    def _load_sample(self, idx: int) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor, str]:
        """
        子类实现: 加载原始数据

        Returns:
            img_bgr: [H, W, C] uint8 BGR numpy array
            cls: [num_obj, 1] float tensor
            bboxes: [num_obj, 4] (cx, cy, w, h) 归一化 float tensor
            img_path: str 图片路径
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}._load_sample 尚未实现。"
        )

    def _generate_target_text(self, cls: torch.Tensor, bboxes: torch.Tensor) -> str:
        """子类实现: 生成目标文本"""
        raise NotImplementedError(
            f"{self.__class__.__name__}._generate_target_text 尚未实现。"
        )

    def _generate_query_text(self, cls: torch.Tensor) -> str:
        """生成查询文本 (默认空, Stage 1)"""
        return ""

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """
        复用 RDD 的 collate 逻辑: per-sample → per-batch

        支持可选的分割 mask 字段
        """
        imgs = []
        target_texts = []
        query_texts = []
        all_cls = []
        all_bboxes = []
        all_batch_idx = []
        img_paths = []
        all_masks = []

        for i, sample in enumerate(batch):
            imgs.append(sample["img"])
            target_texts.append(sample["target_text"])
            query_texts.append(sample["query_text"])
            img_paths.append(sample["img_path"])

            n_obj = sample["cls"].shape[0]
            if n_obj > 0:
                all_cls.append(sample["cls"])
                all_bboxes.append(sample["bboxes"])
                all_batch_idx.append(
                    torch.full((n_obj,), i, dtype=torch.float32))

            # 分割 mask (可选)
            if "mask" in sample:
                all_masks.append(sample["mask"])

        imgs = torch.stack(imgs, dim=0)

        if len(all_cls) > 0:
            cls = torch.cat(all_cls, dim=0)
            bboxes = torch.cat(all_bboxes, dim=0)
            batch_idx = torch.cat(all_batch_idx, dim=0)
        else:
            cls = torch.zeros(0, 1)
            bboxes = torch.zeros(0, 4)
            batch_idx = torch.zeros(0)

        result = {
            "img": imgs,
            "target_text": target_texts,
            "query_text": query_texts,
            "cls": cls,
            "bboxes": bboxes,
            "batch_idx": batch_idx,
            "img_path": img_paths,
        }

        # 分割 masks (可选, DRIVE/Inria 等分割数据集提供)
        if len(all_masks) == len(batch):
            result["masks"] = torch.stack(all_masks, dim=0)  # [B, 1, H, W]

        return result


# ============================================================
# DRIVE 视网膜血管数据集
# ============================================================

# DRIVE 目录结构 (标准):
#   DRIVE/
#     training/
#       images/       *.tif   (20 张, 584×565 或类似)
#       1st_manual/   *.gif   (手工分割 mask)
#       mask/         *.gif   (FOV mask)
#     test/
#       images/       *.tif
#       1st_manual/   *.gif
#       mask/         *.gif

class DRIVETopoDataset(GenericTopoDataset):
    """
    DRIVE 视网膜血管数据集适配器

    输入: 视网膜图像 + 血管分割 mask
    目标文本: 血管结构描述 (如 "retinal vessel tree with N bifurcations")
    拓扑: 血管的连通性 (Betti_0) 和环路 (Betti_1)

    标注: 分割 mask → 转为伪 bbox (mask 的 connected components 的外接矩形)
    类别:
      0 = vessel segment (血管段)
    """

    SPLIT_MAP = {
        "train": "training",
        "val": "test",
        "test": "test",
    }

    def _scan_samples(self):
        """扫描 DRIVE 目录"""
        split_dir = self.SPLIT_MAP.get(self.split, "training")
        img_dir = self.data_root / split_dir / "images"

        if not img_dir.exists():
            logger.warning("DRIVE image dir not found: %s", img_dir)
            return

        mask_dir = self.data_root / split_dir / "1st_manual"
        if not mask_dir.exists():
            # 某些版本用 manual1 或其他名称
            for alt_name in ["manual1", "gt", "labels"]:
                alt = self.data_root / split_dir / alt_name
                if alt.exists():
                    mask_dir = alt
                    break

        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in (".tif", ".png", ".jpg", ".jpeg", ".bmp"):
                continue

            # 匹配 mask: 尝试多种命名约定
            # DRIVE 标准: 21_training.tif → 21_manual1.gif
            stem = img_path.stem
            # 提取前缀数字 (如 "21_training" → "21")
            num_prefix = stem.split("_")[0]

            mask_path = None
            for ext in [".gif", ".png", ".tif", ".bmp"]:
                for pattern in [
                    f"{num_prefix}_manual1{ext}",
                    f"{num_prefix}{ext}",
                    f"{stem}{ext}",
                ]:
                    candidate = mask_dir / pattern
                    if candidate.exists():
                        mask_path = candidate
                        break
                if mask_path:
                    break

            self.samples.append({
                "img_path": img_path,
                "mask_path": mask_path,  # 可能为 None
            })

        logger.info("DRIVE [%s]: found %d samples", self.split, len(self.samples))

    def _load_sample(self, idx: int):
        """加载 DRIVE 图像和 mask, mask → 伪 bbox + 原始 mask"""
        info = self.samples[idx]
        img_path = str(info["img_path"])

        img = cv2.imread(img_path)
        if img is None:
            raise IOError(f"Failed to read DRIVE image: {img_path}")

        h, w = img.shape[:2]

        # 加载 mask
        cls_list = []
        bbox_list = []
        raw_mask = None

        if info["mask_path"] is not None and Path(info["mask_path"]).exists():
            mask = cv2.imread(str(info["mask_path"]), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                # 二值化
                _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                cls_list, bbox_list = self._mask_to_bboxes(binary, h, w)
                raw_mask = binary  # [H, W] uint8 {0, 255}

        if len(cls_list) > 0:
            cls = torch.tensor(cls_list, dtype=torch.float32).unsqueeze(-1)  # [N, 1]
            bboxes = torch.tensor(bbox_list, dtype=torch.float32)  # [N, 4]
        else:
            cls = torch.zeros(0, 1)
            bboxes = torch.zeros(0, 4)

        return img, cls, bboxes, img_path, raw_mask

    @staticmethod
    def _mask_to_bboxes(
        binary_mask: np.ndarray,
        h: int,
        w: int,
        min_area: int = 50,
        max_components: int = 50,
    ) -> Tuple[List, List]:
        """
        分割 mask → connected components → 归一化 bbox (cx, cy, w, h)

        过滤面积过小的组件 (噪声)
        最多返回 max_components 个 (按面积降序)
        """
        # connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )

        cls_list = []
        bbox_list = []
        areas = []

        for i in range(1, num_labels):  # 跳过背景 (label=0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            x0 = stats[i, cv2.CC_STAT_LEFT]
            y0 = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]

            # 归一化 (cx, cy, w, h)
            cx = (x0 + bw / 2) / w
            cy = (y0 + bh / 2) / h
            nw = bw / w
            nh = bh / h

            cls_list.append(0)  # class=0 (vessel segment)
            bbox_list.append([cx, cy, nw, nh])
            areas.append(area)

        # 按面积降序, 取前 max_components
        if len(areas) > max_components:
            order = np.argsort(areas)[::-1][:max_components]
            cls_list = [cls_list[i] for i in order]
            bbox_list = [bbox_list[i] for i in order]

        return cls_list, bbox_list

    def _generate_target_text(self, cls: torch.Tensor, bboxes: torch.Tensor) -> str:
        """生成 DRIVE 目标文本: 描述血管结构"""
        n_segments = cls.shape[0]

        if n_segments == 0:
            return "A retinal image with no visible vessel segments."

        # 分析空间分布
        if bboxes.shape[0] > 1:
            centers = bboxes[:, :2]
            spread = centers.std(dim=0).mean().item()
            if spread > 0.2:
                layout = "widely distributed"
            elif spread > 0.1:
                layout = "moderately distributed"
            else:
                layout = "concentrated"
        else:
            layout = "isolated"

        # 估计分支复杂度 (基于组件数)
        if n_segments > 20:
            complexity = "complex"
        elif n_segments > 10:
            complexity = "moderate"
        else:
            complexity = "simple"

        return (
            f"A retinal image showing a {complexity} vessel tree with "
            f"{n_segments} {layout} vessel segments."
        )


# ============================================================
# Inria Aerial 建筑物分割数据集
# ============================================================

# Inria 目录结构:
#   AerialImageDataset/
#     train/
#       images/  *.tif   (5000×5000, 很大!)
#       gt/      *.tif   (二值 mask)
#     test/
#       images/  *.tif

class InriaTopoDataset(GenericTopoDataset):
    """
    Inria Aerial 建筑物分割数据集适配器

    输入: 卫星图像 + 建筑物分割 mask
    目标文本: 建筑物布局描述 (如 "dense urban area with N buildings")
    拓扑: 建筑物边界的闭合性 (Betti_1)

    标注: 分割 mask → 连通组件的外接矩形作为伪 bbox
    类别:
      0 = building (建筑物)

    注意: 原图 5000×5000 非常大, 默认裁剪随机 patch
    """

    SPLIT_MAP = {
        "train": "train",
        "val": "train",    # Inria 无官方 val split, 常用 train 后5张
        "test": "test",
    }

    # Inria 标准: train 有 180 张 (5 cities × 36), 常用最后 5*5 = 25 做 val
    VAL_CITIES = {"bellingham", "bloomington", "innsbruck", "sfo", "tyrol-e"}
    VAL_START_IDX = 31  # 每个城市的第 31-36 张做 val

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        img_size: int = 640,
        text_strategy: str = "class_name",
        augment: bool = True,
        patch_size: int = 1024,
        patches_per_image: int = 4,
    ):
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        super().__init__(data_root, split, img_size, text_strategy, augment)

    def _scan_samples(self):
        """扫描 Inria 目录"""
        split_dir = self.SPLIT_MAP.get(self.split, "train")
        img_dir = self.data_root / split_dir / "images"

        if not img_dir.exists():
            logger.warning("Inria image dir not found: %s", img_dir)
            return

        gt_dir = self.data_root / split_dir / "gt"

        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in (".tif", ".png", ".jpg"):
                continue

            # val split: 筛选特定图片
            if self.split == "val":
                stem = img_path.stem.lower()
                # 检查是否属于 val 子集
                city = stem.rstrip("0123456789")
                idx_str = stem[len(city):]
                try:
                    idx_num = int(idx_str)
                except ValueError:
                    continue
                if idx_num < self.VAL_START_IDX:
                    continue
            elif self.split == "train":
                # train 排除 val 的图片
                stem = img_path.stem.lower()
                city = stem.rstrip("0123456789")
                idx_str = stem[len(city):]
                try:
                    idx_num = int(idx_str)
                except ValueError:
                    idx_num = 0
                if idx_num >= self.VAL_START_IDX:
                    continue

            # 匹配 GT mask
            gt_path = None
            if gt_dir.exists():
                for ext in [".tif", ".png"]:
                    candidate = gt_dir / (img_path.stem + ext)
                    if candidate.exists():
                        gt_path = candidate
                        break

            # 每张大图生成多个 patch 条目
            for patch_idx in range(self.patches_per_image):
                self.samples.append({
                    "img_path": img_path,
                    "gt_path": gt_path,
                    "patch_idx": patch_idx,
                })

        logger.info("Inria [%s]: found %d sample patches (%d images x %d patches)",
                     self.split, len(self.samples),
                     len(self.samples) // max(self.patches_per_image, 1),
                     self.patches_per_image)

    def _load_sample(self, idx: int):
        """加载 Inria 图像 patch, mask → 伪 bbox + 原始 mask patch"""
        info = self.samples[idx]
        img_path = str(info["img_path"])

        img = cv2.imread(img_path)
        if img is None:
            raise IOError(f"Failed to read Inria image: {img_path}")

        H, W = img.shape[:2]

        # 加载 GT mask
        gt_mask = None
        if info["gt_path"] is not None and Path(info["gt_path"]).exists():
            gt_mask = cv2.imread(str(info["gt_path"]), cv2.IMREAD_GRAYSCALE)

        # 裁剪 patch
        ps = min(self.patch_size, H, W)
        if self.augment and self.split == "train":
            y0 = random.randint(0, max(0, H - ps))
            x0 = random.randint(0, max(0, W - ps))
        else:
            # 确定性网格裁剪 (val/test)
            patch_idx = info["patch_idx"]
            grid_cols = max(1, W // ps)
            row = patch_idx // grid_cols
            col = patch_idx % grid_cols
            y0 = min(row * ps, max(0, H - ps))
            x0 = min(col * ps, max(0, W - ps))

        img_patch = img[y0:y0 + ps, x0:x0 + ps]

        # 对应的 mask patch
        cls_list = []
        bbox_list = []
        raw_mask_patch = None
        if gt_mask is not None:
            mask_patch = gt_mask[y0:y0 + ps, x0:x0 + ps]
            _, binary = cv2.threshold(mask_patch, 127, 255, cv2.THRESH_BINARY)
            cls_list, bbox_list = self._mask_to_bboxes(binary, ps, ps)
            raw_mask_patch = binary  # [ps, ps] uint8 {0, 255}

        if len(cls_list) > 0:
            cls = torch.tensor(cls_list, dtype=torch.float32).unsqueeze(-1)
            bboxes = torch.tensor(bbox_list, dtype=torch.float32)
        else:
            cls = torch.zeros(0, 1)
            bboxes = torch.zeros(0, 4)

        return img_patch, cls, bboxes, img_path, raw_mask_patch

    @staticmethod
    def _mask_to_bboxes(
        binary_mask: np.ndarray,
        h: int,
        w: int,
        min_area: int = 200,
        max_components: int = 50,
    ) -> Tuple[List, List]:
        """分割 mask → connected components → 归一化 bbox"""
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )

        cls_list = []
        bbox_list = []
        areas = []

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            x0 = stats[i, cv2.CC_STAT_LEFT]
            y0 = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]

            cx = (x0 + bw / 2) / w
            cy = (y0 + bh / 2) / h
            nw = bw / w
            nh = bh / h

            cls_list.append(0)  # class=0 (building)
            bbox_list.append([cx, cy, nw, nh])
            areas.append(area)

        if len(areas) > max_components:
            order = np.argsort(areas)[::-1][:max_components]
            cls_list = [cls_list[i] for i in order]
            bbox_list = [bbox_list[i] for i in order]

        return cls_list, bbox_list

    def _generate_target_text(self, cls: torch.Tensor, bboxes: torch.Tensor) -> str:
        """生成 Inria 目标文本: 描述建筑物布局"""
        n_buildings = cls.shape[0]

        if n_buildings == 0:
            return "An aerial view of an area with no visible buildings."

        # 分析密度
        if bboxes.shape[0] > 1:
            areas = (bboxes[:, 2] * bboxes[:, 3])
            total_coverage = areas.sum().item()
            avg_area = areas.mean().item()

            if total_coverage > 0.3:
                density = "dense urban"
            elif total_coverage > 0.1:
                density = "suburban"
            else:
                density = "rural"

            if avg_area > 0.05:
                size_desc = "large"
            elif avg_area > 0.01:
                size_desc = "medium-sized"
            else:
                size_desc = "small"
        else:
            density = "sparse"
            size_desc = "isolated"

        return (
            f"An aerial view of a {density} area with "
            f"{n_buildings} {size_desc} buildings."
        )
