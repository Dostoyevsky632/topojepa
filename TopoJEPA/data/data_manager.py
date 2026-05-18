"""
数据管理器
==========
构建 DataLoader, 支持单数据集和多数据集训练

职责:
  - build_dataloader: 根据 dataset_name 实例化对应 Dataset + 绑定 collate_fn
  - build_multi_dataset_loader: 多数据集加权混合 (论文通用性验证)
"""

import torch
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from typing import Optional, Dict, List

from .rdd_dataset import RDDTopoDataset


def _get_dataset_registry():
    """
    数据集注册表 (延迟导入, 避免未安装依赖时 crash)

    注意: drive/inria 适配器骨架已存在但尚未实现 (__getitem__ raises NotImplementedError)
    调用时会给出明确错误信息而非静默的空数据集
    """
    from .generic_dataset import DRIVETopoDataset, InriaTopoDataset
    return {
        "rdd": RDDTopoDataset,
        "drive": DRIVETopoDataset,
        "inria": InriaTopoDataset,
    }


def build_dataloader(
    dataset_name: str = "rdd",
    data_root: str = "",
    split: str = "train",
    batch_size: int = 16,
    img_size: int = 640,
    num_workers: int = 8,
    pin_memory: bool = True,
    stage: int = 1,
    text_strategy: str = "class_name",
    augment: bool = True,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    """
    构建 TopoJEPA 数据加载器

    Args:
        dataset_name: "rdd" | "drive" | "inria"
        data_root: 数据集根目录
        split: "train" | "val" | "test"
        batch_size: 批大小
        img_size: 图像尺寸 (正方形)
        num_workers: 数据加载线程数
        pin_memory: 是否 pin memory
        stage: 训练阶段 (1=预训练, 2=微调)
        text_strategy: 文本生成策略
        augment: 是否数据增强
        distributed: 是否分布式训练
        rank: 当前进程 rank
        world_size: 总进程数

    Returns:
        DataLoader
    """
    # 选择数据集类
    registry = _get_dataset_registry()
    dataset_cls = registry.get(dataset_name)
    if dataset_cls is None:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(registry.keys())}"
        )

    # 构建数据集
    # RDDTopoDataset 有完整参数; 其他数据集是 GenericTopoDataset 子类, 参数子集兼容
    dataset_kwargs = dict(
        data_root=data_root,
        split=split,
        img_size=img_size,
        text_strategy=text_strategy,
        augment=augment,
    )
    if dataset_name == "rdd":
        dataset_kwargs["stage"] = stage

    dataset = dataset_cls(**dataset_kwargs)

    # 分布式采样器
    sampler = None
    shuffle = (split == "train")
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
        )
        shuffle = False  # sampler 负责 shuffle

    # 选择 collate_fn
    collate_fn = getattr(dataset, "collate_fn", None)
    if collate_fn is None:
        # GenericTopoDataset 子类可能没有自定义 collate_fn
        # 复用 RDDTopoDataset.collate_fn (格式统一)
        collate_fn = RDDTopoDataset.collate_fn

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=sampler,
        collate_fn=collate_fn,
        drop_last=(split == "train"),
        persistent_workers=(num_workers > 0),
    )

    return dataloader


def build_multi_dataset_loader(
    datasets_config: List[Dict],
    batch_size: int = 16,
    img_size: int = 640,
    num_workers: int = 8,
    pin_memory: bool = True,
    stage: int = 1,
    text_strategy: str = "class_name",
    augment: bool = True,
) -> DataLoader:
    """
    构建多数据集混合加载器 (论文中用于验证通用性)

    使用 WeightedRandomSampler 按权重从各数据集采样

    Args:
        datasets_config: 数据集配置列表, 每个元素:
            {
              "name": "rdd",          # 数据集名
              "root": "path/to/data",  # 数据根目录
              "weight": 1.0,           # 采样权重 (越大采越多)
            }
        batch_size: 批大小
        img_size: 图像尺寸
        num_workers: 数据加载线程数
        pin_memory: 是否 pin memory
        stage: 训练阶段
        text_strategy: 文本生成策略
        augment: 是否数据增强

    Returns:
        DataLoader
    """
    datasets = []
    sample_weights = []

    for cfg in datasets_config:
        name = cfg["name"]
        root = cfg["root"]
        weight = cfg.get("weight", 1.0)

        registry = _get_dataset_registry()
        dataset_cls = registry.get(name)
        if dataset_cls is None:
            raise ValueError(f"Unknown dataset: {name}. Available: {list(registry.keys())}")

        dataset_kwargs = dict(
            data_root=root,
            split="train",
            img_size=img_size,
            text_strategy=text_strategy,
            augment=augment,
        )
        if name == "rdd":
            dataset_kwargs["stage"] = stage

        ds = dataset_cls(**dataset_kwargs)
        datasets.append(ds)

        # 为每个样本赋权重
        sample_weights.extend([weight] * len(ds))

    # 拼接所有数据集
    combined = ConcatDataset(datasets)

    # 加权随机采样器 (replacement=True 确保小数据集不会被跳过)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(combined),
        replacement=True,
    )

    dataloader = DataLoader(
        combined,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=RDDTopoDataset.collate_fn,  # 统一 collate 格式
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )

    return dataloader
