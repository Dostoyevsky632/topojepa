"""
RDD 数据集适配器
=================
将 YOLO 格式的道路损伤检测数据集适配为 TopoJEPA 训练格式

YOLO 格式:
  image: xxx.jpg
  label: xxx.txt → "class_id cx cy w h\n" (每行一个目标)

TopoJEPA 需要:
  - image: 图像张量
  - target_text: 结构化文本描述 (送入 Y-Encoder)
  - query_text: 查询文本 (Stage 2, 可选)
  - 检测标注 (Stage 2): 按 ultralytics batch 契约

Batch 字段契约 (ultralytics 风格, 全项目统一):
  ┌──────────────────────────────────────────────────────┐
  │  字段名       │ 形状              │ 说明              │
  ├──────────────────────────────────────────────────────┤
  │ "img"         │ [B, C, H, W]     │ 图像张量           │
  │ "target_text" │ List[str]        │ 目标文本 (B 条)     │
  │ "query_text"  │ List[str]        │ 查询文本 (B 条)     │
  │ "cls"         │ [N_total, 1]     │ 类别 ID (整个batch) │
  │ "bboxes"      │ [N_total, 4]     │ (cx, cy, w, h) 归一化 │
  │ "batch_idx"   │ [N_total]        │ 每个 obj 所属图片索引  │
  │ "img_path"    │ List[str]        │ 图片路径 (B 条)     │
  └──────────────────────────────────────────────────────┘
"""

import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from typing import Optional, Dict, List, Tuple
from pathlib import Path

from .transforms import TopoJEPATransforms


# ============================================================
# RDD 类别映射
# ============================================================

RDD_CLASS_NAMES = {
    0: "Longitudinal Crack",
    1: "Transverse Crack",
    2: "Alligator Crack",
    3: "Pothole",
    4: "Other Damage",
}

RDD_CLASS_DESCRIPTIONS = {
    0: "A longitudinal crack running along the road direction",
    1: "A transverse crack running across the road",
    2: "An alligator crack with interconnected pattern resembling scales",
    3: "A pothole with surface depression or missing material",
    4: "Other road surface damage",
}

# Stage 2 查询文本池 (随机采样)
_QUERY_POOL = [
    "What types of road damage are present in this image?",
    "Describe the road surface condition.",
    "Identify and locate the defects on this road.",
    "What road damage can you see?",
    "Analyze the structural integrity of this road surface.",
]


class RDDTopoDataset(Dataset):
    """
    RDD 道路损伤数据集 (TopoJEPA 格式)

    目录结构预期:
      data_root/
        train/
          images/  *.jpg
          labels/  *.txt
        val/
          images/  *.jpg
          labels/  *.txt
        test/
          images/  *.jpg
          labels/  *.txt   (可能不完整)
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        img_size: int = 640,
        text_strategy: str = "class_name",  # "class_name" | "spatial" | "topological"
        class_names: Optional[dict] = None,
        class_descriptions: Optional[dict] = None,
        augment: bool = True,
        stage: int = 1,
    ):
        """初始化数据集"""
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = img_size
        self.text_strategy = text_strategy
        self.class_names = class_names or RDD_CLASS_NAMES
        self.class_descriptions = class_descriptions or RDD_CLASS_DESCRIPTIONS
        self.stage = stage

        # 增强管道
        self.transforms = TopoJEPATransforms(
            img_size=img_size,
            augment=(augment and split == "train"),
        )

        # 扫描样本: 以 images 目录为基准, 匹配对应 label
        self.img_dir = self.data_root / split / "images"
        self.lbl_dir = self.data_root / split / "labels"

        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

        # 收集所有 (img_path, lbl_path) 对
        self.samples: List[Tuple[Path, Path]] = []
        for img_path in sorted(self.img_dir.glob("*.jpg")):
            lbl_path = self.lbl_dir / (img_path.stem + ".txt")
            self.samples.append((img_path, lbl_path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """
        返回单个样本 (per-sample, collate 前)

        Returns:
            dict with:
              - "img": [C, H, W] float32 tensor (normalized)
              - "target_text": str 目标文本
              - "query_text": str 查询文本
              - "bboxes": [num_obj, 4] (cx, cy, w, h) 归一化
              - "cls": [num_obj, 1] float tensor
              - "img_path": str
        """
        img_path, lbl_path = self.samples[idx]

        # --- 读取图像 ---
        img = cv2.imread(str(img_path))
        if img is None:
            raise IOError(f"Failed to read image: {img_path}")

        # --- 读取标注 ---
        cls, bboxes = self._load_yolo_labels(str(lbl_path))

        # --- 数据增强 (在 numpy 空间, bbox 同步变换) ---
        augmented = self.transforms(img, bboxes, cls)

        # --- 生成文本 ---
        target_text = self._generate_target_text(
            augmented["cls"], augmented["bboxes"], self.text_strategy)
        query_text = self._generate_query_text(augmented["cls"])

        return {
            "img": augmented["img"],             # [C, H, W]
            "target_text": target_text,
            "query_text": query_text,
            "bboxes": augmented["bboxes"],       # [num_obj, 4]
            "cls": augmented["cls"],             # [num_obj, 1]
            "img_path": str(img_path),
        }

    # ============ 标注加载 ============

    def _load_yolo_labels(
        self,
        label_path: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        加载 YOLO 格式标注文件

        每行: class_id cx cy w h

        如果文件不存在或为空, 返回空 tensor (无目标的图像)

        Returns:
            cls: [num_obj, 1] float tensor
            bboxes: [num_obj, 4] float tensor (cx, cy, w, h)
        """
        lbl = Path(label_path)
        if not lbl.exists():
            return torch.zeros(0, 1), torch.zeros(0, 4)

        import warnings
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = np.loadtxt(str(lbl), dtype=np.float32)
        except Exception:
            return torch.zeros(0, 1), torch.zeros(0, 4)

        if data.size == 0:
            return torch.zeros(0, 1), torch.zeros(0, 4)

        # 确保 2D: 单行时 np.loadtxt 返回 1D
        if data.ndim == 1:
            data = data[np.newaxis, :]

        # data: [N, 5] → class_id, cx, cy, w, h
        cls = torch.from_numpy(data[:, 0:1]).float()   # [N, 1]
        bboxes = torch.from_numpy(data[:, 1:5]).float()  # [N, 4]

        return cls, bboxes

    # ============ 文本生成策略 ============

    def _generate_target_text(
        self,
        cls: torch.Tensor,                     # [num_obj, 1]
        bboxes: torch.Tensor,                  # [num_obj, 4]
        strategy: str = "class_name",
    ) -> str:
        """
        根据标注生成目标文本

        策略:
          "class_name": 仅类别名, 适合 Stage 1
          "spatial":    类别名 + 空间位置, 适合 Stage 2
          "topological": 类别名 + 拓扑描述 (连通/分叉)
        """
        if cls.shape[0] == 0:
            return "No road damage detected. The road surface appears intact."

        class_ids = cls[:, 0].long().tolist()

        if strategy == "class_name":
            return self._text_class_name(class_ids)
        elif strategy == "spatial":
            return self._text_spatial(class_ids, bboxes)
        elif strategy == "topological":
            return self._text_topological(class_ids, bboxes)
        else:
            return self._text_class_name(class_ids)

    @staticmethod
    def _article(word: str) -> str:
        """返回正确的不定冠词 a/an"""
        return "an" if word[0].lower() in "aeiou" else "a"

    def _text_class_name(self, class_ids: List[int]) -> str:
        """策略 1: 仅列举类别名"""
        # 去重计数
        from collections import Counter
        counts = Counter(class_ids)

        parts = []
        for cid, cnt in sorted(counts.items()):
            name = self.class_names.get(cid, f"Class {cid}")
            name_lower = name.lower()
            if cnt == 1:
                # "other damage" 不加冠词; 其余用 a/an
                if name_lower.startswith("other"):
                    parts.append(name_lower)
                else:
                    parts.append(f"{self._article(name)} {name_lower}")
            else:
                parts.append(f"{cnt} {name_lower}{'s' if not name_lower.endswith('s') else ''}")

        return "This road has " + ", ".join(parts) + "."

    def _text_spatial(self, class_ids: List[int], bboxes: torch.Tensor) -> str:
        """策略 2: 类别名 + 空间位置描述"""
        descriptions = []
        for i, cid in enumerate(class_ids):
            name = self.class_names.get(cid, f"Class {cid}")
            spatial = self._bbox_to_spatial_description(
                bboxes[i], name, (self.img_size, self.img_size))
            descriptions.append(spatial)

        # 最多描述 5 个目标 (避免文本过长)
        if len(descriptions) > 5:
            descriptions = descriptions[:5]
            descriptions.append(f"and {len(class_ids) - 5} more damages")

        return "; ".join(descriptions) + "."

    def _text_topological(self, class_ids: List[int], bboxes: torch.Tensor) -> str:
        """
        策略 3: 拓扑描述 — 分析 bbox 间的空间邻近关系推测连通性

        简化启发: 如果同类 bbox 的中心距离 < 阈值, 认为它们"连通"
        """
        from collections import Counter
        counts = Counter(class_ids)

        parts = []
        for cid, cnt in sorted(counts.items()):
            name = self.class_names.get(cid, f"Class {cid}")
            if cnt == 1:
                article = self._article(name)
                parts.append(f"{article} isolated {name.lower()}")
            else:
                # 检查同类 bbox 是否相邻
                mask = [i for i, c in enumerate(class_ids) if c == cid]
                centers = bboxes[mask, :2]  # [k, 2] (cx, cy)
                # 两两距离
                dists = torch.cdist(centers.unsqueeze(0), centers.unsqueeze(0)).squeeze(0)
                # 平均 bbox 大小作为邻接阈值
                avg_size = bboxes[mask, 2:4].mean().item()
                connected = (dists < avg_size * 2).sum().item() - len(mask)  # 减去对角线

                if connected > 0:
                    parts.append(
                        f"{cnt} connected {name.lower()}s forming a network pattern")
                else:
                    parts.append(f"{cnt} scattered {name.lower()}s")

        return "This road has " + ", ".join(parts) + "."

    def _generate_query_text(
        self,
        cls: torch.Tensor,
    ) -> str:
        """
        生成查询文本

        Stage 1: 返回空字符串 (query-free, 与 plan 一致)
          → 训练侧 forward_predict 收到空字符串时不 tokenize, 不注入 query
        Stage 2: 从查询池随机采样 (增加多样性)
        """
        if self.stage == 1:
            return ""
        else:
            return random.choice(_QUERY_POOL)

    def _bbox_to_spatial_description(
        self,
        bbox: torch.Tensor,                    # [4] (cx, cy, w, h)
        class_name: str,
        img_size: Tuple[int, int],
    ) -> str:
        """
        将 bbox 转化为空间描述

        3×3 网格: left/center/right × top/middle/bottom

        例: (0.25, 0.33, 0.1, 0.3) + "Longitudinal Crack"
         → "a longitudinal crack at the upper-left region"
        """
        cx, cy = bbox[0].item(), bbox[1].item()
        w, h = bbox[2].item(), bbox[3].item()

        # 水平位置
        if cx < 0.33:
            h_pos = "left"
        elif cx < 0.67:
            h_pos = "center"
        else:
            h_pos = "right"

        # 垂直位置
        if cy < 0.33:
            v_pos = "upper"
        elif cy < 0.67:
            v_pos = "middle"
        else:
            v_pos = "lower"

        # 大小描述
        area = w * h
        if area < 0.01:
            size_desc = "small"
        elif area < 0.05:
            size_desc = "medium"
        else:
            size_desc = "large"

        article = self._article(size_desc)
        return f"{article} {size_desc} {class_name.lower()} at the {v_pos}-{h_pos} region"

    # ============ Collate ============

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """
        自定义 collate: per-sample → per-batch (ultralytics 风格)

        将每张图的 cls/bboxes 拼接为 [N_total, ...],
        并自动生成 batch_idx 标记每个 obj 属于哪张图

        Input:
            batch: List of per-sample dicts from __getitem__

        Output:
            {
              "img": [B, C, H, W],
              "target_text": List[str],  len=B
              "query_text": List[str],   len=B
              "cls": [N_total, 1],
              "bboxes": [N_total, 4],
              "batch_idx": [N_total],
              "img_path": List[str],     len=B
            }
        """
        imgs = []
        target_texts = []
        query_texts = []
        all_cls = []
        all_bboxes = []
        all_batch_idx = []
        img_paths = []

        for i, sample in enumerate(batch):
            imgs.append(sample["img"])
            target_texts.append(sample["target_text"])
            query_texts.append(sample["query_text"])
            img_paths.append(sample["img_path"])

            n_obj = sample["cls"].shape[0]
            if n_obj > 0:
                all_cls.append(sample["cls"])             # [n_obj, 1]
                all_bboxes.append(sample["bboxes"])       # [n_obj, 4]
                all_batch_idx.append(
                    torch.full((n_obj,), i, dtype=torch.float32))  # [n_obj]

        # 拼接图像 → [B, C, H, W]
        imgs = torch.stack(imgs, dim=0)

        # 拼接变长的检测标注
        if len(all_cls) > 0:
            cls = torch.cat(all_cls, dim=0)           # [N_total, 1]
            bboxes = torch.cat(all_bboxes, dim=0)     # [N_total, 4]
            batch_idx = torch.cat(all_batch_idx, dim=0)  # [N_total]
        else:
            cls = torch.zeros(0, 1)
            bboxes = torch.zeros(0, 4)
            batch_idx = torch.zeros(0)

        return {
            "img": imgs,
            "target_text": target_texts,
            "query_text": query_texts,
            "cls": cls,
            "bboxes": bboxes,
            "batch_idx": batch_idx,
            "img_path": img_paths,
        }
