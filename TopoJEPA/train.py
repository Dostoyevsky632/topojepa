"""
TopoJEPA 训练脚本
==================
两阶段训练:
  Stage 1 (预训练): 大规模 caption 对齐, L_jepa + L_topo, query-free
  Stage 2 (微调):   检测/VQA 能力, L_jepa + L_topo + L_detect, query-conditioned

组件职责边界:
  - EMA: 由 utils.ema.ExponentialMovingAverage 管理 (非模型内部)
  - 拓扑权重调度: 由 TopoJEPACriterion.get_topo_weight() 唯一管理
  - 学习率调度: 由 utils.schedulers.CosineScheduler 管理
  - Tokenizer: 由 TopoJEPA 模型持有

运行方式:
  cd TopoJEPA && python train.py --config configs/topojepa_rdd.yaml --stage 1
  或
  python -m TopoJEPA.train --config TopoJEPA/configs/topojepa_rdd.yaml --stage 1
"""

import os
import sys
import time
import math
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

# 兼容: 从 TopoJEPA/ 目录内运行或从项目根运行
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from models import TopoJEPA, VisualEncoder, TextEncoder, EmbeddingPredictor, TopologicalBranch, DetectionNeck, DetectionHead
from losses import TopoJEPACriterion, JEPALoss, TopologicalLoss, TopoDetectionLoss
from losses.detection_loss import _decode_dfl, _make_anchors, _cxcywh_to_xyxy
from data import build_dataloader
from utils import ExponentialMovingAverage, CosineScheduler
from utils.metrics import compute_detection_metrics, compute_segmentation_metrics

logger = logging.getLogger("topojepa")


# ============================================================
# 配置加载
# ============================================================

def _load_config(config_path: str) -> Dict[str, Any]:
    """
    加载 YAML 配置文件, 回退到默认值

    尝试 yaml → 失败则返回空 dict (由各 build 函数用默认参数)
    """
    if not config_path or not Path(config_path).exists():
        logger.warning("Config file not found: %s, using defaults", config_path)
        return {}
    try:
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        logger.warning("PyYAML not installed, using defaults")
        return {}


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="TopoJEPA Training")

    parser.add_argument("--config", type=str, default="configs/topojepa_rdd.yaml")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 12],
                        help="1=pretrain, 2=finetune, 12=merged (joint training from scratch)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override config epochs")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override config batch_size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override config lr")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output dir (default: from config or runs/topojepa)")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--data_root", type=str, default="",
                        help="Override data root path")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


# ============================================================
# 构建函数
# ============================================================

def build_model(cfg: Dict, stage: int = 1) -> TopoJEPA:
    """
    根据配置构建 TopoJEPA 模型

    Stage 1:  detection_head=None
    Stage 2:  detection_head=DetectionHead(...)
    Stage 12: 与 Stage 2 相同结构 (合并训练)
    """
    mcfg = cfg.get("model", {})
    embed_dim = mcfg.get("embed_dim", 256)

    # Visual encoder
    visual_encoder = VisualEncoder(
        backbone_type=mcfg.get("backbone_type", "vit"),
        embed_dim=embed_dim,
        freeze=mcfg.get("freeze", True),
        pretrained_path=mcfg.get("pretrained_path", ""),
    )

    # Text encoder
    te_cfg = mcfg.get("text_encoder", {})
    text_encoder = TextEncoder(
        model_name=te_cfg.get("model_name", "all-MiniLM-L6-v2"),
        embed_dim=embed_dim,
        max_length=te_cfg.get("max_length", 128),
        freeze_base=te_cfg.get("freeze_base", False),
        lr_multiplier=te_cfg.get("lr_multiplier", 0.05),
    )

    # Predictor
    pred_cfg = mcfg.get("predictor", {})
    predictor = EmbeddingPredictor(
        visual_dim=pred_cfg.get("visual_dim", embed_dim),
        predictor_dim=pred_cfg.get("predictor_dim", embed_dim // 2),
        output_dim=pred_cfg.get("output_dim", embed_dim),
        depth=pred_cfg.get("depth", 6),
        num_heads=pred_cfg.get("num_heads", 8),
        mlp_ratio=pred_cfg.get("mlp_ratio", 4.0),
        max_query_tokens=pred_cfg.get("max_query_tokens", 128),
        max_visual_tokens=pred_cfg.get("max_visual_tokens", 512),
        use_bidirectional_attn=pred_cfg.get("use_bidirectional_attn", True),
        dropout=pred_cfg.get("dropout", 0.0),
    )

    # Topo branch
    tb_cfg = mcfg.get("topo_branch", {})
    topo_branch = TopologicalBranch(
        embed_dim=embed_dim,
        max_homology_dim=tb_cfg.get("max_homology_dim", 1),
        max_points=tb_cfg.get("max_points", 128),
        persistence_threshold=tb_cfg.get("persistence_threshold", 0.01),
    )

    # Stage 2 任务头: 检测 (RDD) 和分割 (DRIVE/Inria) 互斥
    # 由 model.task_type 决定: "detection" (默认) 或 "segmentation"
    task_type = mcfg.get("task_type", "detection")
    ve_in_channels = getattr(visual_encoder, "_feature_channels", None)

    detection_neck = None
    detection_head = None
    segmentation_head = None
    if stage in (2, 12):
        if task_type == "detection":
            dh_cfg = mcfg.get("detection_head", {})
            cfg_in_channels = dh_cfg.get("in_channels", None)

            # Stage 2 要求检测头输入通道与 visual_encoder 实际特征一致
            if mcfg.get("backbone_type", "vit") == "yolo_backbone" and ve_in_channels:
                in_channels = list(ve_in_channels)
                if cfg_in_channels is not None and list(cfg_in_channels) != in_channels:
                    logger.warning(
                        "detection_head.in_channels=%s mismatches visual feature channels=%s; "
                        "using visual feature channels.",
                        list(cfg_in_channels), in_channels
                    )
            else:
                in_channels = cfg_in_channels or [256, 512, 1024]

            # FPN Neck: 可训练特征融合层
            neck_cfg = mcfg.get("detection_neck", {})
            fpn_channels = neck_cfg.get("fpn_channels", 256)
            detection_neck = DetectionNeck(
                in_channels=in_channels,
                fpn_channels=fpn_channels,
            )
            neck_out_channels = [fpn_channels] * len(in_channels)
            logger.info("Stage 2: DetectionNeck built (fpn_channels=%d)", fpn_channels)

            detection_head = DetectionHead(
                in_channels=neck_out_channels,
                num_classes=dh_cfg.get("num_classes", 5),
                reg_max=dh_cfg.get("reg_max", 16),
            )
            logger.info("Stage 2: task_type=detection, DetectionHead built")

        elif task_type == "segmentation":
            sh_cfg = mcfg.get("segmentation_head", {})
            seg_in_channels = sh_cfg.get("in_channels", None)
            if mcfg.get("backbone_type", "vit") == "yolo_backbone" and ve_in_channels:
                seg_in_channels = list(ve_in_channels)
            else:
                seg_in_channels = seg_in_channels or [256, 512, 1024]

            from models.seg_head import SegmentationHead
            segmentation_head = SegmentationHead(
                in_channels=seg_in_channels,
                num_classes=sh_cfg.get("num_classes", 1),
                fpn_channels=sh_cfg.get("fpn_channels", 256),
            )
            logger.info("Stage 2: task_type=segmentation, SegmentationHead built (classes=%d)",
                        sh_cfg.get("num_classes", 1))
        else:
            logger.warning("Unknown task_type=%s, no task head built", task_type)

    # Stage 2/12: 解冻 backbone 最后几层, 让任务损失能驱动特征学习
    if stage in (2, 12):
        unfreeze_n = mcfg.get("stage2_unfreeze_layers", 3)
        n_unfrozen = visual_encoder.partial_unfreeze(n_layers=unfreeze_n)
        logger.info(
            "Stage 2: unfroze last %d backbone layers (%d params now trainable)",
            unfreeze_n, n_unfrozen,
        )

    model = TopoJEPA(
        visual_encoder=visual_encoder,
        text_encoder=text_encoder,
        predictor=predictor,
        topo_branch=topo_branch,
        detection_neck=detection_neck,
        detection_head=detection_head,
        segmentation_head=segmentation_head,
        embed_dim=embed_dim,
        tokenizer_name=mcfg.get("tokenizer_name", "all-MiniLM-L6-v2"),
    )
    return model


def build_criterion(cfg: Dict, stage: int = 1, experiment_mode: str = "topojepa") -> TopoJEPACriterion:
    """
    构建组合损失函数

    拓扑权重调度完全由 TopoJEPACriterion 内部管理
    不需要额外的 topo_scheduler
    """
    lcfg = cfg.get("loss", {})

    jepa_cfg = lcfg.get("jepa", {})
    jepa_loss = JEPALoss(
        loss_type=jepa_cfg.get("loss_type", "infonce"),
        temperature=jepa_cfg.get("temperature", 0.07),
        reg_coeff=jepa_cfg.get("reg_coeff", 0.0),
        label_smoothing=jepa_cfg.get("label_smoothing", 0.0),
    )

    topo_cfg = lcfg.get("topo", {})
    topo_loss = TopologicalLoss(
        loss_components=topo_cfg.get("loss_components", ["fidelity", "collapse"]),
        wasserstein_order=topo_cfg.get("wasserstein_order", 2),
        use_sliced_wasserstein=topo_cfg.get("use_sliced_wasserstein", True),
        num_slices=topo_cfg.get("num_slices", 50),
        fidelity_weight=topo_cfg.get("fidelity_weight", 1.0),
        preserve_weight=topo_cfg.get("preserve_weight", 0.5),
        collapse_weight=topo_cfg.get("collapse_weight", 0.5),
        persistence_threshold=topo_cfg.get("persistence_threshold", 0.01),
    )

    # 任务损失: 检测和分割互斥, 由 model.task_type 决定
    mcfg = cfg.get("model", {})
    task_type = mcfg.get("task_type", "detection")

    detect_loss = None
    seg_loss = None
    if stage in (2, 12):
        if task_type == "detection":
            det_cfg = lcfg.get("detection", {})
            detect_loss = TopoDetectionLoss(
                num_classes=det_cfg.get("num_classes", 5),
                box_gain=det_cfg.get("box_gain", 7.5),
                cls_gain=det_cfg.get("cls_gain", 0.5),
                dfl_gain=det_cfg.get("dfl_gain", 1.5),
            )
        elif task_type == "segmentation":
            seg_cfg = lcfg.get("segmentation", {})
            from losses.seg_loss import SegmentationLoss
            seg_loss = SegmentationLoss(
                num_classes=seg_cfg.get("num_classes", 1),
                dice_weight=seg_cfg.get("dice_weight", 1.0),
                bce_weight=seg_cfg.get("bce_weight", 1.0),
            )

    comb_cfg = lcfg.get("combined", {})
    criterion = TopoJEPACriterion(
        jepa_loss=jepa_loss,
        topo_loss=topo_loss,
        detect_loss=detect_loss,
        seg_loss=seg_loss,
        topo_weight_init=comb_cfg.get("topo_weight_init", 0.01),
        topo_weight_final=comb_cfg.get("topo_weight_final", 1.0),
        topo_warmup_steps=comb_cfg.get("topo_warmup_steps", 2000),
        topo_adaptive=comb_cfg.get("topo_adaptive", True),
        topo_loss_ratio_target=comb_cfg.get("topo_loss_ratio_target", 1.0),
        topo_loss_ratio_ema=comb_cfg.get("topo_loss_ratio_ema", 0.99),
        topo_weight_min=comb_cfg.get("topo_weight_min", 0.01),
        topo_weight_max=comb_cfg.get("topo_weight_max", 5.0),
        topo_loss_max=comb_cfg.get("topo_loss_max", 10.0),
        jepa_weight=comb_cfg.get("jepa_weight", 1.0),
        detect_weight=comb_cfg.get("detect_weight", 1.0),
        seg_weight=comb_cfg.get("seg_weight", 1.0),
        experiment_mode=experiment_mode,
        training_stage=stage,
    )
    return criterion


def build_optimizer(
    model: TopoJEPA,
    lr: float,
    weight_decay: float = 0.04,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
) -> torch.optim.Optimizer:
    """
    构建优化器 (分组参数)

    参数分组:
      - visual_encoder: 冻结 (lr=0) 或极低学习率
      - predictor: 正常学习率
      - text_encoder: 通过 get_lr_params() 细分
      - topo_branch: 正常学习率 (如有可学习参数)
      - detection_head: 正常学习率 (Stage 2)
    """
    param_groups = []
    seen_params = set()

    # --- text_encoder: 细粒度分组 ---
    for group in model.text_encoder.get_lr_params():
        group_lr = lr * group.get("lr_multiplier", 1.0)
        params = [p for p in group["params"] if p.requires_grad]
        if params:
            param_groups.append({
                "params": params,
                "lr": group_lr,
                "weight_decay": weight_decay,
                "name": group.get("name", "text_encoder"),
            })
            for p in params:
                seen_params.add(id(p))

    # --- predictor: 正常学习率 ---
    pred_params = [p for p in model.predictor.parameters()
                   if p.requires_grad and id(p) not in seen_params]
    if pred_params:
        param_groups.append({
            "params": pred_params,
            "lr": lr,
            "weight_decay": weight_decay,
            "name": "predictor",
        })
        for p in pred_params:
            seen_params.add(id(p))

    # --- visual_encoder: 冻结时跳过, 解冻层用 lr * 0.1 ---
    ve_params = [p for p in model.visual_encoder.parameters()
                 if p.requires_grad and id(p) not in seen_params]
    if ve_params:
        param_groups.append({
            "params": ve_params,
            "lr": lr * 0.1,
            "weight_decay": weight_decay,
            "name": "visual_encoder",
        })
        for p in ve_params:
            seen_params.add(id(p))

    # --- topo_branch ---
    tb_params = [p for p in model.topo_branch.parameters()
                 if p.requires_grad and id(p) not in seen_params]
    if tb_params:
        param_groups.append({
            "params": tb_params,
            "lr": lr,
            "weight_decay": weight_decay,
            "name": "topo_branch",
        })
        for p in tb_params:
            seen_params.add(id(p))

    # --- detection_neck (Stage 2/12): 随机初始化, 适度提高 lr ---
    if model.detection_neck is not None:
        dn_params = [p for p in model.detection_neck.parameters()
                     if p.requires_grad and id(p) not in seen_params]
        if dn_params:
            param_groups.append({
                "params": dn_params,
                "lr": lr * 3.0,
                "weight_decay": weight_decay,
                "name": "detection_neck",
            })
            for p in dn_params:
                seen_params.add(id(p))

    # --- detection_head (Stage 2/12): 随机初始化, 适度提高 lr ---
    if model.detection_head is not None:
        dh_params = [p for p in model.detection_head.parameters()
                     if p.requires_grad and id(p) not in seen_params]
        if dh_params:
            param_groups.append({
                "params": dh_params,
                "lr": lr * 3.0,
                "weight_decay": weight_decay,
                "name": "detection_head",
            })
            for p in dh_params:
                seen_params.add(id(p))

    # --- segmentation_head (Stage 2/12): 随机初始化, 适度提高 lr ---
    if model.segmentation_head is not None:
        sh_params = [p for p in model.segmentation_head.parameters()
                     if p.requires_grad and id(p) not in seen_params]
        if sh_params:
            param_groups.append({
                "params": sh_params,
                "lr": lr * 3.0,
                "weight_decay": weight_decay,
                "name": "segmentation_head",
            })
            for p in sh_params:
                seen_params.add(id(p))

    # --- 剩余参数 (query_embed 等) ---
    remaining = [p for p in model.parameters()
                 if p.requires_grad and id(p) not in seen_params]
    if remaining:
        param_groups.append({
            "params": remaining,
            "lr": lr,
            "weight_decay": weight_decay,
            "name": "other",
        })

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=lr,  # default lr, 被 group-level lr 覆盖
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )
    return optimizer


# ============================================================
# 训练 / 评估
# ============================================================

def _empty_det_prediction() -> Dict[str, np.ndarray]:
    return {
        "boxes": np.zeros((0, 4), dtype=np.float32),
        "scores": np.array([], dtype=np.float32),
        "labels": np.array([], dtype=np.int64),
    }


def _empty_det_gt() -> Dict[str, np.ndarray]:
    return {
        "boxes": np.zeros((0, 4), dtype=np.float32),
        "labels": np.array([], dtype=np.int64),
    }


def _decode_predictions_for_metrics(
    predictions: Dict[str, Any],
    conf_thresh: float = 0.25,
    topk: int = 300,
    nms_iou_thresh: float = 0.65,
) -> List[Dict[str, np.ndarray]]:
    """
    将 DetectionHead 原始输出解码为评估格式:
      {"boxes": [K,4] xyxy(归一化), "scores": [K], "labels": [K]}

    流程: conf 过滤 → top-k → batched NMS (逐类去重叠)
    """
    from torchvision.ops import batched_nms

    scores = predictions["scores"]            # [B, nc, A] logits
    raw_boxes = predictions["boxes"]          # [B, reg_max*4, A]
    feats = predictions["feats"]              # List[Tensor]
    B = int(scores.shape[0])

    # 与 TopoDetectionLoss 一致的几何推断
    feat_shapes = [(f.shape[2], f.shape[3]) for f in feats]
    base_stride = 8
    max_feat_h = max(h for h, _ in feat_shapes)
    strides = [int(base_stride * (max_feat_h / h)) for h, _ in feat_shapes]
    anchors, stride_t = _make_anchors(feat_shapes, strides, scores.device)
    img_size = max_feat_h * base_stride

    reg_max = raw_boxes.shape[1] // 4
    pred_boxes = _decode_dfl(raw_boxes, anchors, stride_t, reg_max, img_size)  # [B, A, 4]
    pred_boxes = pred_boxes.clamp(0.0, 1.0)
    pred_scores = torch.sigmoid(scores).permute(0, 2, 1)  # [B, A, nc]

    decoded: List[Dict[str, np.ndarray]] = []
    for b in range(B):
        conf, labels = pred_scores[b].max(dim=1)  # [A], [A]
        keep = conf > conf_thresh
        if not keep.any():
            decoded.append(_empty_det_prediction())
            continue

        boxes_b = pred_boxes[b][keep]
        conf_b = conf[keep]
        labels_b = labels[keep]

        if conf_b.numel() > topk:
            top_idx = conf_b.topk(topk).indices
            boxes_b = boxes_b[top_idx]
            conf_b = conf_b[top_idx]
            labels_b = labels_b[top_idx]

        # NMS: 逐类去除重叠框
        nms_keep = batched_nms(boxes_b, conf_b, labels_b, nms_iou_thresh)
        boxes_b = boxes_b[nms_keep]
        conf_b = conf_b[nms_keep]
        labels_b = labels_b[nms_keep]

        decoded.append({
            "boxes": boxes_b.detach().cpu().numpy().astype(np.float32),
            "scores": conf_b.detach().cpu().numpy().astype(np.float32),
            "labels": labels_b.detach().cpu().numpy().astype(np.int64),
        })

    return decoded


def _extract_ground_truth_for_metrics(
    batch: Dict[str, Any],
    batch_size: int,
) -> List[Dict[str, np.ndarray]]:
    """从 collate 后 batch 中提取逐图 GT。"""
    gt_cls = batch.get("cls")
    gt_boxes = batch.get("bboxes")
    gt_batch_idx = batch.get("batch_idx")

    if gt_cls is None or gt_boxes is None or gt_batch_idx is None or gt_cls.numel() == 0:
        return [_empty_det_gt() for _ in range(batch_size)]

    out: List[Dict[str, np.ndarray]] = []
    for b in range(batch_size):
        mask = gt_batch_idx == b
        if not mask.any():
            out.append(_empty_det_gt())
            continue

        cls_b = gt_cls[mask].long().squeeze(-1)
        boxes_b = _cxcywh_to_xyxy(gt_boxes[mask]).clamp(0.0, 1.0)
        out.append({
            "boxes": boxes_b.detach().cpu().numpy().astype(np.float32),
            "labels": cls_b.detach().cpu().numpy().astype(np.int64),
        })
    return out


def _iou_1vN_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """单框 vs 多框 IoU, 输入都是 xyxy。"""
    if boxes.size == 0:
        return np.array([], dtype=np.float32)

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = area_a + area_b - inter
    return inter / (union + 1e-10)


def _compute_mean_iou_matched(
    predictions: List[Dict[str, np.ndarray]],
    ground_truths: List[Dict[str, np.ndarray]],
    iou_thr: float = 0.5,
) -> float:
    """
    计算匹配样本的平均 IoU:
      - 每个 GT 只在同类预测里找 best IoU
      - 仅统计 IoU>=iou_thr 的匹配
    """
    matched_ious: List[float] = []
    for pred, gt in zip(predictions, ground_truths):
        p_boxes = pred.get("boxes", np.zeros((0, 4), dtype=np.float32))
        p_labels = pred.get("labels", np.array([], dtype=np.int64))
        g_boxes = gt.get("boxes", np.zeros((0, 4), dtype=np.float32))
        g_labels = gt.get("labels", np.array([], dtype=np.int64))
        if len(p_boxes) == 0 or len(g_boxes) == 0:
            continue

        for gi in range(len(g_boxes)):
            same_cls = p_labels == g_labels[gi]
            if not np.any(same_cls):
                continue
            ious = _iou_1vN_xyxy(g_boxes[gi], p_boxes[same_cls])
            if ious.size == 0:
                continue
            best = float(np.max(ious))
            if best >= iou_thr:
                matched_ious.append(best)

    if not matched_ious:
        return 0.0
    return float(np.mean(matched_ious))


def train_one_epoch(
    model: TopoJEPA,
    criterion: TopoJEPACriterion,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: CosineScheduler,
    ema: ExponentialMovingAverage,
    target_encoder: nn.Module,                 # EMA 管理的 target encoder
    scaler: Optional[GradScaler],
    epoch: int,
    global_step: int,
    device: torch.device,
    clip_grad: float = 10.0,
    use_amp: bool = True,
    log_every: int = 50,
    use_progress: bool = True,
) -> Dict[str, float]:
    """
    训练一个 epoch

    核心循环:
      1. Forward: model(images, query_texts, target_encoder)
      2. Loss: criterion(model_output, batch, step)
         (拓扑权重由 criterion 内部调度, 无外部 scheduler)
      3. Backward + Optimizer step
      4. EMA update: ema.update(global_step)
      5. LR schedule: lr_scheduler.step(global_step)

    Returns:
        metrics: {loss, loss_jepa, loss_topo, loss_detect, lr, topo_weight, ...}
    """
    model.train()
    criterion.train()

    metric_accum: Dict[str, float] = {}
    n_batches = 0
    nonfinite_count = 0
    nonfinite_consecutive = 0

    progress = None
    iterable = dataloader
    if use_progress and tqdm is not None:
        progress = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Train E{epoch}",
            dynamic_ncols=True,
            leave=False,
            disable=not sys.stderr.isatty(),
        )
        iterable = progress

    for batch in iterable:
        # --- 数据搬运 ---
        images = batch["img"].to(device, non_blocking=True)
        # batch 内所有 tensor 搬到 device (cls/bboxes/batch_idx 在 Stage2 被 detection_loss 使用)
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)
        target_texts = batch["target_text"]
        if isinstance(target_texts, list) and len(target_texts) > 0:
            keys = [t if isinstance(t, str) else str(t) for t in target_texts]
            unique_ratio = len(set(keys)) / float(len(keys))
            metric_accum["text_dup_frac"] = metric_accum.get("text_dup_frac", 0.0) + (1.0 - unique_ratio)

        query_texts = batch.get("query_text")
        if isinstance(query_texts, list) and len(query_texts) > 0:
            non_empty = sum(1 for q in query_texts if isinstance(q, str) and q.strip() != "")
            metric_accum["query_nonempty_ratio"] = metric_accum.get("query_nonempty_ratio", 0.0) + (
                non_empty / float(len(query_texts))
            )
        # Stage 1: query_texts 全为空字符串时等效 query-free
        if query_texts is not None and all((isinstance(q, str) and q.strip() == "") for q in query_texts):
            query_texts = None

        # --- LR 调度 ---
        current_lr = lr_scheduler.step(global_step)
        for pg in optimizer.param_groups:
            # 保持分组内的 lr_multiplier 比例
            base_lr = pg.get("_base_lr", None)
            if base_lr is None:
                # 首次: 记录初始 lr 比例
                pg["_base_lr"] = pg["lr"]
                base_lr = pg["lr"]
            # 按比例缩放
            pg["lr"] = current_lr * (base_lr / lr_scheduler.init_value) \
                if lr_scheduler.init_value > 0 else current_lr

        # --- Forward (标准 JEPA: images → online encoder + EMA target encoder) ---
        amp_enabled = use_amp and scaler is not None
        amp_device = "cuda" if device.type == "cuda" else "cpu"
        with autocast(amp_device, enabled=amp_enabled):
            model_output = model(
                images,
                query_texts=query_texts,
                target_encoder=target_encoder,
            )
            loss_output = criterion(model_output, batch=batch, step=global_step)
            loss = loss_output["loss"]

        # 数值守护: 非有限 loss 直接跳过该 step，防止参数被 NaN 污染。
        if not torch.isfinite(loss):
            nonfinite_count += 1
            nonfinite_consecutive += 1
            s_hat_finite = torch.isfinite(model_output["S_Y_hat"]).all().item()
            s_y_finite = torch.isfinite(model_output["S_Y"]).all().item()
            l_jepa = loss_output.get("loss_jepa", torch.tensor(float("nan"), device=device))
            l_topo = loss_output.get("loss_topo", torch.tensor(float("nan"), device=device))
            l_det = loss_output.get("loss_detect", torch.tensor(float("nan"), device=device))
            # 限频日志：前 3 次必打，之后每 50 次打一次，避免刷屏破坏进度条。
            if nonfinite_count <= 3 or nonfinite_count % 50 == 0:
                logger.warning(
                    "Non-finite loss at epoch=%d step=%d (count=%d), skipping update | "
                    "loss=%s jepa=%s topo=%s detect=%s S_Y_hat_finite=%s S_Y_finite=%s",
                    epoch,
                    global_step,
                    nonfinite_count,
                    str(float(loss.detach().cpu()) if torch.isfinite(loss.detach().cpu()) else "nan/inf"),
                    str(float(l_jepa.detach().cpu()) if torch.isfinite(l_jepa.detach().cpu()) else "nan/inf"),
                    str(float(l_topo.detach().cpu()) if torch.isfinite(l_topo.detach().cpu()) else "nan/inf"),
                    str(float(l_det.detach().cpu()) if torch.isfinite(l_det.detach().cpu()) else "nan/inf"),
                    bool(s_hat_finite),
                    bool(s_y_finite),
                )
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            # 若连续爆炸，尽早失败并给出明确处置建议。
            if nonfinite_consecutive >= 20:
                raise RuntimeError(
                    "Too many consecutive non-finite losses. "
                    "Try disabling AMP (--no_amp) for Stage 2."
                )
            continue
        else:
            nonfinite_consecutive = 0

        # --- Backward ---
        optimizer.zero_grad(set_to_none=True)
        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        # --- EMA update ---
        ema.update(global_step)

        # --- 记录指标 ---
        for k, v in loss_output.items():
            if isinstance(v, torch.Tensor) and v.ndim == 0:
                val = v.item()
            elif isinstance(v, (int, float)):
                val = float(v)
            else:
                continue
            metric_accum[k] = metric_accum.get(k, 0.0) + val

        metric_accum["lr"] = metric_accum.get("lr", 0.0) + current_lr
        n_batches += 1
        global_step += 1

        # --- 进度显示/降噪日志 ---
        if progress is not None:
            if n_batches % max(log_every, 1) == 0 or n_batches == 1:
                avg_loss = metric_accum.get("loss", 0.0) / max(n_batches, 1)
                progress.set_postfix({
                    "step": global_step,
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{current_lr:.2e}",
                    "alpha": f"{float(loss_output.get('topo_weight', 0.0)):.3f}",
                })
        else:
            if n_batches % max(log_every, 1) == 1:
                avg_loss = metric_accum.get("loss", 0.0) / max(n_batches, 1)
                logger.info(
                    "Epoch %d | Step %d | Loss %.4f | LR %.2e | α %.3f",
                    epoch, global_step, avg_loss, current_lr,
                    loss_output.get("topo_weight", 0.0),
                )

    if progress is not None:
        progress.close()

    # 平均指标
    metrics = {k: v / max(n_batches, 1) for k, v in metric_accum.items()}
    metrics["nonfinite_skips"] = float(nonfinite_count)
    metrics["global_step"] = global_step
    metrics["epoch"] = epoch
    return metrics


@torch.no_grad()
def evaluate(
    model: TopoJEPA,
    dataloader: torch.utils.data.DataLoader,
    criterion: TopoJEPACriterion,
    target_encoder: nn.Module,
    device: torch.device,
    epoch: int,
    global_step: int = 0,
    use_amp: bool = True,
    use_progress: bool = True,
) -> Dict[str, float]:
    """
    验证评估

    Args:
        global_step: 当前训练步数, 传给 criterion 以使 topo_weight
                     与训练进度一致 (criterion.eval() 下只读不写 EMA)

    Returns:
        metrics: {val_loss, cosine_sim, topo_fidelity, ...}
    """
    model.eval()
    criterion.eval()
    target_encoder.eval()  # 防御性: 确保 target encoder 在 eval

    metric_accum: Dict[str, float] = {}
    n_batches = 0
    total_correct = 0
    total_correct_strict = 0
    total_samples = 0
    det_predictions_all: List[Dict[str, np.ndarray]] = []
    det_ground_truths_all: List[Dict[str, np.ndarray]] = []
    seg_predictions_all: List[np.ndarray] = []
    seg_ground_truths_all: List[np.ndarray] = []

    progress = None
    iterable = dataloader
    if use_progress and tqdm is not None:
        progress = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Val   E{epoch}",
            dynamic_ncols=True,
            leave=False,
            disable=not sys.stderr.isatty(),
        )
        iterable = progress

    for batch in iterable:
        images = batch["img"].to(device, non_blocking=True)
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)
        target_texts = batch["target_text"]
        if isinstance(target_texts, list) and len(target_texts) > 0:
            keys = [t if isinstance(t, str) else str(t) for t in target_texts]
            unique_ratio = len(set(keys)) / float(len(keys))
            metric_accum["text_dup_frac"] = metric_accum.get("text_dup_frac", 0.0) + (1.0 - unique_ratio)
        # Stage 2: 传递 query_text, 保持与训练一致
        query_texts = batch.get("query_text")
        if query_texts is not None and all((isinstance(q, str) and q.strip() == "") for q in query_texts):
            query_texts = None

        amp_device = "cuda" if device.type == "cuda" else "cpu"
        with autocast(amp_device, enabled=use_amp and device.type == "cuda"):
            model_output = model(
                images,
                query_texts=query_texts,
                target_encoder=target_encoder,
            )
            loss_output = criterion(model_output, batch=batch, step=global_step)

        # --- 嵌入预测准确率 ---
        # 标准 JEPA: 用 cosine similarity 做 top-1 匹配
        # S_Y_hat[i] 应最接近 S_Y[i] (同一张图像的 online vs EMA 嵌入)
        S_Y_hat = model_output["S_Y_hat"]  # [B, D]
        S_Y = model_output["S_Y"]          # [B, D]
        sim = torch.nn.functional.cosine_similarity(S_Y_hat.unsqueeze(1),
                                                     S_Y.unsqueeze(0), dim=-1)  # [B, B]
        preds = sim.argmax(dim=1)  # [B]
        labels = torch.arange(S_Y_hat.shape[0], device=device)
        total_correct_strict += (preds == labels).sum().item()
        # 标准 JEPA: 每个样本只有一个正样本 (自身), Acc(pos) = Acc(strict)
        total_correct += (preds == labels).sum().item()
        total_samples += S_Y_hat.shape[0]

        # 记录
        for k, v in loss_output.items():
            if isinstance(v, torch.Tensor) and v.ndim == 0:
                val = v.item()
            elif isinstance(v, (int, float)):
                val = float(v)
            else:
                continue
            metric_accum[k] = metric_accum.get(k, 0.0) + val

        # --- 检测评估 (Stage 2) ---
        det_pred_raw = model_output.get("predictions")
        if det_pred_raw is not None:
            batch_preds = _decode_predictions_for_metrics(
                det_pred_raw,
                conf_thresh=0.01,
                topk=300,
            )
            batch_gts = _extract_ground_truth_for_metrics(batch, images.shape[0])
            det_predictions_all.extend(batch_preds)
            det_ground_truths_all.extend(batch_gts)

        # --- 分割评估 (Stage 2, DRIVE/Inria) ---
        seg_pred_raw = model_output.get("seg_predictions")
        if seg_pred_raw is not None and batch.get("masks") is not None:
            seg_logits = seg_pred_raw["seg_logits"]  # [B, C, H, W]
            gt_masks = batch["masks"]  # [B, 1, H, W]
            # 上采样 logits 到 mask 尺寸
            if seg_logits.shape[2:] != gt_masks.shape[2:]:
                seg_logits = torch.nn.functional.interpolate(
                    seg_logits, size=gt_masks.shape[2:], mode="bilinear", align_corners=False
                )
            seg_pred_np = torch.sigmoid(seg_logits).detach().cpu().numpy()
            seg_gt_np = gt_masks.detach().cpu().numpy()
            seg_predictions_all.append(seg_pred_np)
            seg_ground_truths_all.append(seg_gt_np)

        n_batches += 1

        if progress is not None and (n_batches % 20 == 0 or n_batches == 1):
            avg_val_loss = metric_accum.get("loss", 0.0) / max(n_batches, 1)
            progress.set_postfix({
                "val_loss": f"{avg_val_loss:.4f}",
            })

    if progress is not None:
        progress.close()

    metrics = {f"val_{k}": v / max(n_batches, 1) for k, v in metric_accum.items()}
    metrics["val_retrieval_acc_pos"] = total_correct / max(total_samples, 1)
    metrics["val_retrieval_acc_strict"] = total_correct_strict / max(total_samples, 1)
    # 向后兼容已有日志/选择 best 的逻辑
    metrics["val_retrieval_acc"] = metrics["val_retrieval_acc_pos"]
    if len(det_predictions_all) > 0 and len(det_predictions_all) == len(det_ground_truths_all):
        det_metrics = compute_detection_metrics(
            det_predictions_all,
            det_ground_truths_all,
            iou_thresholds=[0.5],
        )
        metrics["val_det_precision"] = float(det_metrics.get("precision", 0.0))
        metrics["val_det_recall"] = float(det_metrics.get("recall", 0.0))
        metrics["val_det_f1"] = float(det_metrics.get("f1", 0.0))
        metrics["val_det_map50"] = float(det_metrics.get("mAP50", 0.0))
        metrics["val_det_iou"] = _compute_mean_iou_matched(
            det_predictions_all, det_ground_truths_all, iou_thr=0.5
        )

    # --- 分割指标 ---
    if len(seg_predictions_all) > 0:
        all_seg_pred = np.concatenate(seg_predictions_all, axis=0)
        all_seg_gt = np.concatenate(seg_ground_truths_all, axis=0)
        seg_metrics = compute_segmentation_metrics(all_seg_pred, all_seg_gt)
        metrics["val_seg_dice"] = seg_metrics["dice"]
        metrics["val_seg_iou"] = seg_metrics["iou"]
        metrics["val_seg_pixel_acc"] = seg_metrics["pixel_accuracy"]

    metrics["epoch"] = epoch
    return metrics


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(
    model: TopoJEPA,
    target_encoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: TopoJEPACriterion,
    epoch: int,
    global_step: int,
    metrics: Dict,
    save_path: str,
) -> None:
    """保存检查点 (包含 target_encoder 和 criterion 状态)"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "target_encoder_state_dict": target_encoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "criterion_state_dict": criterion.state_dict(),
        "metrics": metrics,
    }
    torch.save(checkpoint, save_path)
    logger.info("Checkpoint saved: %s", save_path)


def load_checkpoint(
    model: TopoJEPA,
    target_encoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: TopoJEPACriterion,
    checkpoint_path: str,
    device: torch.device,
) -> tuple:
    """
    加载检查点

    Returns:
        (epoch, global_step, full_resume)
        full_resume=True  表示同阶段完整恢复(包含 optimizer/criterion 状态)
        full_resume=False 表示跨阶段权重迁移初始化(仅模型相关权重), 训练应从 epoch=0 开始
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = checkpoint.get("model_state_dict", {})

    ckpt_has_detection = any(k.startswith("detection_head.") for k in model_state.keys())
    model_has_detection = model.detection_head is not None
    stage_mismatch = (ckpt_has_detection != model_has_detection)

    # 检查结构差异 (新增 neck 等模块)
    ckpt_has_neck = any(k.startswith("detection_neck.") for k in model_state.keys())
    model_has_neck = model.detection_neck is not None
    structure_changed = stage_mismatch or (ckpt_has_neck != model_has_neck)

    if structure_changed:
        # 跨阶段或结构变化: 宽松加载
        incompatible = model.load_state_dict(model_state, strict=False)
        if checkpoint.get("target_encoder_state_dict") is not None:
            target_encoder.load_state_dict(checkpoint["target_encoder_state_dict"], strict=False)
        logger.warning(
            "Structure-changed checkpoint loaded as initialization: %s | "
            "missing=%d unexpected=%d. Optimizer/Criterion states are skipped.",
            checkpoint_path,
            len(getattr(incompatible, "missing_keys", [])),
            len(getattr(incompatible, "unexpected_keys", [])),
        )
        return 0, 0, False

    # 同阶段恢复: 严格加载
    model.load_state_dict(model_state)
    target_encoder.load_state_dict(checkpoint["target_encoder_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "criterion_state_dict" in checkpoint:
        criterion.load_state_dict(checkpoint["criterion_state_dict"])
    logger.info("Checkpoint loaded: %s (epoch %d, step %d)",
                checkpoint_path, checkpoint["epoch"], checkpoint["global_step"])
    return checkpoint["epoch"], checkpoint["global_step"], True


# ============================================================
# Main
# ============================================================

def main():
    """主函数"""
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Logging (tqdm 兼容) ---
    # 使用 TqdmStreamHandler 让日志通过 tqdm.write 输出, 避免打断进度条
    class _TqdmStreamHandler(logging.StreamHandler):
        """通过 tqdm.write 输出日志, 避免破坏进度条"""
        def emit(self, record):
            try:
                msg = self.format(record)
                if tqdm is not None:
                    tqdm.write(msg, file=self.stream)
                else:
                    self.stream.write(msg + self.terminator)
                    self.flush()
            except Exception:
                self.handleError(record)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # 清除已有 handler, 防止重复
    root_logger.handlers.clear()
    handler = _TqdmStreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
    root_logger.addHandler(handler)

    # --- Config ---
    cfg = _load_config(args.config)
    stage = args.stage
    training_cfg = cfg.get("training", {})
    # stage 12 先查 "stage12"，回退到 "stage2"
    if stage == 12:
        stage_cfg = training_cfg.get("stage12", training_cfg.get("stage2", {}))
    else:
        stage_cfg = training_cfg.get(f"stage{stage}", {})

    # CLI overrides
    epochs = args.epochs or stage_cfg.get("epochs", 50)
    batch_size = args.batch_size or stage_cfg.get("batch_size", 32)
    lr = args.lr or stage_cfg.get("lr", 5e-4)
    final_lr = stage_cfg.get("final_lr", 1e-6)
    weight_decay = training_cfg.get("weight_decay", 0.04) if "weight_decay" not in stage_cfg else stage_cfg["weight_decay"]
    warmup_epochs = stage_cfg.get("warmup_epochs", 5)
    text_strategy = stage_cfg.get("text_strategy", "class_name")
    clip_grad = stage_cfg.get("clip_grad", training_cfg.get("clip_grad", 10.0))
    # Stage 2 默认禁用 AMP: 解冻 backbone BN + fp16 容易数值溢出 (NaN)
    # 用户可通过 config 中 stage2.amp=true 或 CLI --amp 强制开启
    stage2_amp_default = False  # Stage 2/12 默认关闭 AMP
    if stage in (2, 12):
        cfg_amp = stage_cfg.get("amp", stage2_amp_default)
    else:
        cfg_amp = stage_cfg.get("amp", training_cfg.get("amp", True))
    use_amp = bool(cfg_amp) and not args.no_amp and torch.cuda.is_available()

    # --- Device ---
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = torch.device("cpu")
        use_amp = False
    else:
        device = torch.device(args.device)
    logger.info("AMP enabled: %s", use_amp)

    # --- Data ---
    data_cfg = cfg.get("data", {})
    data_root = args.data_root or data_cfg.get("root", "RDD_SPLIT")
    # 允许配置里写相对路径；统一按项目根目录解析，便于直接使用仓库内数据集
    data_root_path = Path(data_root)
    if not data_root_path.is_absolute():
        candidate = (_SCRIPT_DIR.parent / data_root_path).resolve()
        if candidate.exists():
            data_root_path = candidate
    data_root = str(data_root_path)
    img_size = data_cfg.get("img_size", 640)
    num_workers = args.num_workers if args.num_workers is not None else data_cfg.get("num_workers", 8)

    logger.info("Building dataloaders...")
    t_dl = time.time()
    train_loader = build_dataloader(
        dataset_name=data_cfg.get("dataset", "rdd"),
        data_root=data_root,
        split="train",
        batch_size=batch_size,
        img_size=img_size,
        num_workers=num_workers,
        stage=stage,
        text_strategy=text_strategy,
        augment=True,
    )

    val_loader = build_dataloader(
        dataset_name=data_cfg.get("dataset", "rdd"),
        data_root=data_root,
        split="val",
        batch_size=batch_size,
        img_size=img_size,
        num_workers=num_workers,
        stage=stage,
        text_strategy=text_strategy,
        augment=False,
    )
    logger.info(
        "Dataloaders ready | train_batches=%d | val_batches=%d | elapsed=%.1fs",
        len(train_loader), len(val_loader), time.time() - t_dl
    )

    # --- Model ---
    logger.info("Building model...")
    t_model = time.time()
    model = build_model(cfg, stage=stage)
    model = model.to(device)
    # 拓扑计算频率: 每 N 步才真正计算 PH, 其余步复用缓存
    topo_every = cfg.get("loss", {}).get("topo_every_n_steps", 1)
    if topo_every > 1:
        model.set_topo_interval(topo_every)
        logger.info("Topo PH computed every %d steps (%.0f%% compute saved)",
                     topo_every, (1 - 1.0 / topo_every) * 100)
    if data_cfg.get("class_names"):
        logger.info("RDD class names: %s", list(data_cfg.get("class_names", {}).values()))
    logger.info("Model built in %.1fs", time.time() - t_model)
    text_backend = getattr(model.text_encoder, "_backend", "unknown")
    text_model_name = getattr(model.text_encoder, "model_name", "unknown")
    logger.info("Text encoder backend: %s | model: %s", text_backend, text_model_name)
    if text_backend != "st":
        logger.warning(
            "Text encoder is in fallback mode. Install sentence-transformers/transformers "
            "for stronger semantic supervision."
        )
    logger.info("Model parameters: %.2fM",
                sum(p.numel() for p in model.parameters()) / 1e6)
    logger.info("Trainable parameters: %.2fM",
                sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6)

    # --- Target encoder (EMA of visual encoder) ---
    # 标准 JEPA: target encoder g_ξ 是 online encoder f_θ 的 EMA 副本
    # target encoder 必须始终处于 eval 模式 (关闭 dropout/BN 训练态)
    target_encoder = ExponentialMovingAverage.build_target(model.visual_encoder)
    target_encoder = target_encoder.to(device)
    target_encoder.eval()

    ema_cfg = training_cfg.get("ema", {})
    total_steps = epochs * len(train_loader)
    ema = ExponentialMovingAverage(
        source_model=model.visual_encoder,
        target_model=target_encoder,
        ema_range=tuple(ema_cfg.get("ema_range", [0.996, 1.0])),
        total_steps=total_steps,
    )

    # --- Criterion ---
    logger.info("Building criterion...")
    t_crit = time.time()
    criterion = build_criterion(cfg, stage=stage, experiment_mode=cfg.get("experiment", {}).get("mode", "topojepa"))
    criterion = criterion.to(device)
    logger.info("Criterion built in %.1fs", time.time() - t_crit)

    # --- Optimizer ---
    logger.info("Building optimizer...")
    t_opt = time.time()
    betas = tuple(training_cfg.get("betas", [0.9, 0.999]))
    optimizer = build_optimizer(
        model,
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=training_cfg.get("eps", 1e-8),
    )
    logger.info("Optimizer built in %.1fs", time.time() - t_opt)
    for pg in optimizer.param_groups:
        n_params = sum(p.numel() for p in pg["params"])
        logger.info("  param_group %-20s | lr=%.2e | params=%.2fM",
                     pg.get("name", "?"), pg["lr"], n_params / 1e6)

    # --- LR Scheduler ---
    warmup_steps = warmup_epochs * len(train_loader)
    lr_scheduler = CosineScheduler(
        init_value=lr,
        final_value=final_lr,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        warmup_value=lr * 0.01,
    )

    # --- AMP ---
    if use_amp:
        try:
            scaler = GradScaler("cuda")
        except TypeError:
            # 兼容旧版签名
            scaler = GradScaler()
    else:
        scaler = None

    # --- Resume ---
    start_epoch = 0
    global_step = 0
    if args.resume and Path(args.resume).exists():
        loaded_epoch, loaded_step, full_resume = load_checkpoint(
            model, target_encoder, optimizer, criterion, args.resume, device)
        if full_resume:
            start_epoch = loaded_epoch + 1  # 从下一个 epoch 开始
            global_step = loaded_step
        else:
            # 跨阶段权重迁移: 从当前 stage 的 epoch 0 开始训练
            start_epoch = 0
            global_step = 0

    # --- Output ---
    output_cfg = cfg.get("output", {})
    # CLI 优先 → 配置文件 → 硬编码默认值
    output_dir = args.output_dir or output_cfg.get("dir") or "runs/topojepa"
    os.makedirs(output_dir, exist_ok=True)
    save_every = output_cfg.get("save_every", 10)
    eval_every = output_cfg.get("eval_every", 1)
    log_every = output_cfg.get("log_every", 50)

    # --- 日志文件: 训练日志同步写入文件 ---
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    from datetime import datetime
    log_filename = f"stage{stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = os.path.join(log_dir, log_filename)
    file_handler = logging.FileHandler(log_filepath, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
    root_logger.addHandler(file_handler)
    logger.info("Training log saved to: %s", log_filepath)

    # --- Stage 2 检测头 warmup: 前 N 个 epoch 冻结表示学习参数, 只训练任务头 ---
    det_warmup_epochs = stage_cfg.get("det_head_warmup_epochs", 0) if stage in (2, 12) else 0
    _warmup_freeze_groups = {"predictor", "text_encoder", "text_encoder_base",
                             "text_encoder_proj", "visual_encoder", "topo_branch", "other"}

    def _set_task_head_warmup(model, optimizer, freeze: bool):
        """Stage2 任务头 warmup: freeze=True 时冻结表示学习参数, 只训练任务头"""
        for pg in optimizer.param_groups:
            name = pg.get("name", "")
            if name in _warmup_freeze_groups:
                if freeze:
                    pg["_saved_lr"] = pg["lr"]
                    pg["lr"] = 0.0
                else:
                    if "_saved_lr" in pg:
                        pg["lr"] = pg["_saved_lr"]

    # --- Stage 12 表示预热: 前 repr_warmup_epochs 个 epoch 检测权重线性从 0 增长到 detect_weight ---
    repr_warmup_epochs = stage_cfg.get("repr_warmup_epochs", 10) if stage == 12 else 0
    _detect_weight_final = criterion.detect_weight  # 保存目标检测权重

    # --- Training loop ---
    logger.info("Starting Stage %d training: %d epochs, batch_size=%d, lr=%.2e",
                stage, epochs, batch_size, lr)
    if stage == 12 and repr_warmup_epochs > 0:
        logger.info("Stage 12 merged training: repr_warmup=%d epochs (detect_weight 0→%.1f)",
                     repr_warmup_epochs, _detect_weight_final)
    if det_warmup_epochs > 0 and stage == 2:
        logger.info("Stage 2 task-head warmup: first %d epochs train only detection/segmentation head",
                     det_warmup_epochs)

    # best model 选择: stage 2/12 检测任务用 mAP50 (越大越好), 其他用 val_loss (越小越好)
    _use_map_for_best = stage in (2, 12) and cfg.get("model", {}).get("task_type", "detection") == "detection"
    best_val_score = 0.0 if _use_map_for_best else float("inf")

    for epoch in range(start_epoch, epochs):
        # Stage 12 表示预热: 检测权重 cosine 增长 (前半段慢, 后半段快)
        if repr_warmup_epochs > 0 and stage == 12:
            if epoch < repr_warmup_epochs:
                ramp = 0.5 * (1 - math.cos(math.pi * (epoch + 1) / repr_warmup_epochs))
                criterion.detect_weight = _detect_weight_final * ramp
                if epoch == start_epoch:
                    logger.info("  [Stage12 Warmup] detect_weight ramps 0 → %.1f over %d epochs",
                                _detect_weight_final, repr_warmup_epochs)
            elif epoch == repr_warmup_epochs:
                criterion.detect_weight = _detect_weight_final
                logger.info("  [Stage12 Warmup] Epoch %d: detect_weight reached full %.1f",
                            epoch, _detect_weight_final)

        # 任务头 warmup 控制 (stage 2)
        if det_warmup_epochs > 0 and stage == 2:
            if epoch < det_warmup_epochs:
                _set_task_head_warmup(model, optimizer, freeze=True)
                if epoch == start_epoch:
                    logger.info("  [Warmup] Epoch %d: only task head is training", epoch)
            elif epoch == det_warmup_epochs:
                _set_task_head_warmup(model, optimizer, freeze=False)
                logger.info("  [Warmup] Epoch %d: all parameters unfrozen, joint training begins", epoch)
        # Train
        train_metrics = train_one_epoch(
            model=model,
            criterion=criterion,
            dataloader=train_loader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            ema=ema,
            target_encoder=target_encoder,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            device=device,
            clip_grad=clip_grad,
            use_amp=use_amp,
            log_every=log_every,
            use_progress=True,
        )
        global_step = int(train_metrics.get("global_step", global_step + len(train_loader)))

        logger.info(
            "Epoch %d Train | Loss %.4f | JEPA %.4f | Topo %.4f | Detect %.4f | Seg %.4f | Sim %.4f | TextDup %.1f%% | NonFiniteSkips %.0f | α %.3f | LR %.2e",
            epoch,
            train_metrics.get("loss", 0),
            train_metrics.get("loss_jepa", 0),
            train_metrics.get("loss_topo", 0),
            train_metrics.get("loss_detect", 0),
            train_metrics.get("loss_seg", 0),
            train_metrics.get("jepa/similarity", 0),
            100.0 * train_metrics.get("text_dup_frac", 0),
            train_metrics.get("nonfinite_skips", 0),
            train_metrics.get("topo_weight", 0),
            train_metrics.get("lr", 0),
        )

        # Evaluate
        if (epoch + 1) % eval_every == 0:
            val_metrics = evaluate(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                target_encoder=target_encoder,
                device=device,
                epoch=epoch,
                global_step=global_step,
                use_amp=use_amp,
                use_progress=True,
            )
            if "val_seg_dice" in val_metrics:
                logger.info(
                    "Epoch %d Val | Loss %.4f | Acc(pos) %.4f | Acc(strict) %.4f | Dice %.4f | SegIoU %.4f | PixAcc %.4f",
                    epoch,
                    val_metrics.get("val_loss", 0),
                    val_metrics.get("val_retrieval_acc", 0),
                    val_metrics.get("val_retrieval_acc_strict", 0),
                    val_metrics.get("val_seg_dice", 0),
                    val_metrics.get("val_seg_iou", 0),
                    val_metrics.get("val_seg_pixel_acc", 0),
                )
            elif "val_det_f1" in val_metrics:
                logger.info(
                    "Epoch %d Val | Loss %.4f | Acc(pos) %.4f | Acc(strict) %.4f | TextDup %.1f%% | F1 %.4f | IoU %.4f | mAP50 %.4f",
                    epoch,
                    val_metrics.get("val_loss", 0),
                    val_metrics.get("val_retrieval_acc", 0),
                    val_metrics.get("val_retrieval_acc_strict", 0),
                    100.0 * val_metrics.get("val_text_dup_frac", 0),
                    val_metrics.get("val_det_f1", 0),
                    val_metrics.get("val_det_iou", 0),
                    val_metrics.get("val_det_map50", 0),
                )
            else:
                logger.info(
                    "Epoch %d Val | Loss %.4f | Acc(pos) %.4f | Acc(strict) %.4f | TextDup %.1f%%",
                    epoch,
                    val_metrics.get("val_loss", 0),
                    val_metrics.get("val_retrieval_acc", 0),
                    val_metrics.get("val_retrieval_acc_strict", 0),
                    100.0 * val_metrics.get("val_text_dup_frac", 0),
                )

            # Best model
            if _use_map_for_best and "val_det_map50" in val_metrics:
                val_score = val_metrics["val_det_map50"]
                is_better = val_score > best_val_score
            else:
                val_score = val_metrics.get("val_loss", float("inf"))
                is_better = val_score < best_val_score
            if is_better:
                best_val_score = val_score
                save_checkpoint(
                    model, target_encoder, optimizer, criterion,
                    epoch, global_step, val_metrics,
                    os.path.join(output_dir, "best.pt"),
                )

        # Periodic save
        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                model, target_encoder, optimizer, criterion,
                epoch, global_step, train_metrics,
                os.path.join(output_dir, f"epoch_{epoch:04d}.pt"),
            )

    # Final save
    save_checkpoint(
        model, target_encoder, optimizer, criterion,
        epochs - 1, global_step, train_metrics,
        os.path.join(output_dir, "last.pt"),
    )
    if _use_map_for_best:
        logger.info("Training complete. Best mAP50: %.4f", best_val_score)
    else:
        logger.info("Training complete. Best val_loss: %.4f", best_val_score)


if __name__ == "__main__":
    main()
