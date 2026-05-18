"""
检测损失 (Detection Loss)
=========================
自建检测损失, 与 DetectionHead 输出格式对齐

数据契约:
  输入 predictions 来自 DetectionHead.forward():
    {"boxes": [B, reg_max*4, A], "scores": [B, nc, A], "feats": [feat_maps]}
  输入 batch 来自 DataLoader (经 collate_fn, ultralytics 风格):
    {"batch_idx": [N_total], "cls": [N_total, 1], "bboxes": [N_total, 4], "img": [B, C, H, W]}

  boxes 是 DFL raw logits: [B, reg_max*4, A]
    - 4 个方向 (left, top, right, bottom), 每个 reg_max bins
    - decode: softmax(reg_max) → expected value → anchor + offset → xyxy

  bboxes 是归一化 cxcywh 格式

算法:
  1. 从 feats 生成 anchor points (各尺度特征图的网格中心)
  2. 解码 DFL logits → ltrb offset → xyxy (归一化坐标)
  3. 简单 top-k 匹配: 每个 GT 选 IoU 最高的 k 个 anchor 作为正样本
  4. 正样本: CIOU loss (回归) + BCE (分类)
  5. 负样本: BCE with target=0 (分类)

参考: ultralytics/ultralytics/utils/loss.py v8DetectionLoss
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """[N, 4] cxcywh → xyxy"""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def _xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """[N, 4] xyxy → cxcywh"""
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)


def _bbox_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    IoU between two sets of boxes (both xyxy format)
    box1: [N, 4], box2: [M, 4] → [N, M]
    """
    area1 = (box1[:, 2] - box1[:, 0]).clamp(min=0) * (box1[:, 3] - box1[:, 1]).clamp(min=0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(min=0) * (box2[:, 3] - box2[:, 1]).clamp(min=0)

    inter_x1 = torch.max(box1[:, None, 0], box2[None, :, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[None, :, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[None, :, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[None, :, 3])

    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + eps)


def _ciou_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Complete IoU loss (可微)
    pred, target: [N, 4] xyxy format
    Returns: [N] loss per pair (1 - CIoU)
    """
    # IoU
    px1, py1, px2, py2 = pred.unbind(-1)
    tx1, ty1, tx2, ty2 = target.unbind(-1)

    pw = (px2 - px1).clamp(min=0)
    ph = (py2 - py1).clamp(min=0)
    tw = (tx2 - tx1).clamp(min=0)
    th = (ty2 - ty1).clamp(min=0)

    area_p = pw * ph
    area_t = tw * th

    inter_x1 = torch.max(px1, tx1)
    inter_y1 = torch.max(py1, ty1)
    inter_x2 = torch.min(px2, tx2)
    inter_y2 = torch.min(py2, ty2)

    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    union = area_p + area_t - inter + eps
    iou = inter / union

    # Enclosing box
    enc_x1 = torch.min(px1, tx1)
    enc_y1 = torch.min(py1, ty1)
    enc_x2 = torch.max(px2, tx2)
    enc_y2 = torch.max(py2, ty2)

    # Distance between centers
    pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
    tcx, tcy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
    rho2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2

    # Diagonal of enclosing box
    c2 = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + eps

    # Aspect ratio penalty
    v = (4.0 / (math.pi ** 2)) * (torch.atan(tw / (th + eps)) - torch.atan(pw / (ph + eps))) ** 2
    with torch.no_grad():
        alpha = v / (1.0 - iou + v + eps)

    ciou = iou - rho2 / c2 - alpha * v
    return 1.0 - ciou


def _make_anchors(
    feat_shapes: List[Tuple[int, int]],
    strides: List[int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从特征图尺寸生成 anchor 中心点 (归一化坐标)

    Returns:
        anchor_points: [total_anchors, 2] (cx, cy) 归一化到 [0, 1]
        stride_tensor: [total_anchors, 1] 每个 anchor 对应的 stride
    """
    all_points = []
    all_strides = []
    for (h, w), stride in zip(feat_shapes, strides):
        # 特征图网格
        sy = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * stride
        sx = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * stride
        grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij")
        points = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
        all_points.append(points)
        all_strides.append(torch.full((h * w, 1), stride, device=device, dtype=torch.float32))

    return torch.cat(all_points, dim=0), torch.cat(all_strides, dim=0)


def _decode_dfl(
    raw_boxes: torch.Tensor,      # [B, reg_max*4, A]
    anchor_points: torch.Tensor,  # [A, 2]
    stride_tensor: torch.Tensor,  # [A, 1]
    reg_max: int,
    img_size: int,
) -> torch.Tensor:
    """
    DFL 解码: raw logits → xyxy (归一化坐标)

    1. reshape → [B, 4, reg_max, A]
    2. softmax(dim=2) → distribution
    3. 乘以 arange(reg_max) → expected offset (in stride units)
    4. ltrb offset × stride → pixel offset
    5. anchor + offset → xyxy
    6. 归一化到 [0, 1]

    Returns:
        decoded_boxes: [B, A, 4] xyxy 归一化
    """
    B, _, A = raw_boxes.shape
    # [B, 4, reg_max, A]
    raw = raw_boxes.reshape(B, 4, reg_max, A)
    # softmax over reg_max bins
    dist = F.softmax(raw, dim=2)
    # expected value: sum(prob * arange)
    bins = torch.arange(reg_max, device=raw.device, dtype=torch.float32)
    # [B, 4, A] = sum over reg_max
    ltrb = (dist * bins[None, None, :, None]).sum(dim=2)

    # [B, A, 4] ltrb
    ltrb = ltrb.permute(0, 2, 1)  # [B, A, 4]

    # anchor_points: [A, 2] → [1, A, 2]
    anchor = anchor_points.unsqueeze(0)  # [1, A, 2]
    stride = stride_tensor.squeeze(-1).unsqueeze(0)  # [1, A]

    # ltrb → pixel offsets
    lt = ltrb[..., :2] * stride.unsqueeze(-1)  # [B, A, 2]
    rb = ltrb[..., 2:] * stride.unsqueeze(-1)  # [B, A, 2]

    # anchor_points ± offsets → xyxy
    x1y1 = anchor - lt  # [B, A, 2]
    x2y2 = anchor + rb  # [B, A, 2]
    boxes_xyxy = torch.cat([x1y1, x2y2], dim=-1)  # [B, A, 4]

    # 归一化
    boxes_xyxy = boxes_xyxy / img_size

    return boxes_xyxy


class TopoDetectionLoss(nn.Module):
    """
    检测损失

    L_detect = L_box(CIOU) + L_cls(BCE) + L_dfl

    完整的 GT-anchor 匹配 + 有监督训练:
      - 从 feats 推断 anchor points
      - 解码 DFL → bbox
      - top-k IoU 匹配
      - CIOU regression + BCE classification
    """

    def __init__(
        self,
        yolo_model: Optional[nn.Module] = None,
        num_classes: int = 5,
        box_gain: float = 7.5,
        cls_gain: float = 0.5,
        dfl_gain: float = 1.5,
        reg_max: int = 16,
        topk: int = 10,
    ):
        """初始化检测损失"""
        super().__init__()
        self.num_classes = num_classes
        self.box_gain = box_gain
        self.cls_gain = cls_gain
        self.dfl_gain = dfl_gain
        self.reg_max = reg_max
        self.topk = topk

    def _infer_geometry(
        self,
        feats: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        从特征图推断 anchor geometry

        Returns:
            anchor_points: [A, 2]
            stride_tensor: [A, 1]
            img_size: int (推断自 P3 尺度)
        """
        device = feats[0].device
        feat_shapes = [(f.shape[2], f.shape[3]) for f in feats]

        # 推断 strides: img_size / feat_size
        # 假设输入图像是正方形, P3 stride 最小
        # 常用 strides: 8, 16, 32 (3 层) 或 stride = img_h / feat_h
        # 我们从特征图大小逆推: 最大特征图对应最小 stride
        max_feat_h = max(h for h, w in feat_shapes)
        # 标准 YOLO: P3 stride=8 → feat=img/8
        # 但 img_size 未知, 用相对方式: stride_i = max_feat_h / feat_h_i * base_stride
        # 简单做法: 直接用比例
        base_stride = 8
        strides = [base_stride * (max_feat_h / h) for h, w in feat_shapes]
        strides = [int(s) for s in strides]

        img_size = max_feat_h * base_stride  # 推断的图像尺寸

        anchors, stride_t = _make_anchors(feat_shapes, strides, device)
        return anchors, stride_t, img_size

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        计算检测损失

        Returns:
            dict with: loss, loss_box, loss_cls, loss_dfl
        """
        # 数值稳定: 检测损失在 fp32 中计算，避免 amp/fp16 造成 logits 溢出。
        scores = predictions["scores"].float()   # [B, nc, A]
        raw_boxes = predictions["boxes"].float()  # [B, reg_max*4, A]
        feats = predictions["feats"]      # List[Tensor]
        device = scores.device
        B = scores.shape[0]

        # 推断 anchor geometry
        anchor_points, stride_tensor, img_size = self._infer_geometry(feats)
        A = anchor_points.shape[0]

        # 解码预测 bbox → [B, A, 4] xyxy 归一化
        pred_boxes = _decode_dfl(raw_boxes, anchor_points, stride_tensor, self.reg_max, img_size)

        # 转置 scores → [B, A, nc]
        pred_scores = scores.permute(0, 2, 1)  # [B, A, nc]

        # --- 提取 GT ---
        gt_cls = batch.get("cls")             # [N_total, 1]
        gt_bboxes = batch.get("bboxes")       # [N_total, 4] cxcywh 归一化
        gt_batch_idx = batch.get("batch_idx")  # [N_total]

        # 无 GT: 全负样本 (所有 anchor 目标为背景)
        if gt_cls is None or gt_cls.numel() == 0:
            target_cls = torch.zeros_like(pred_scores)
            loss_cls = F.binary_cross_entropy_with_logits(
                pred_scores, target_cls, reduction="mean")
            loss_box = pred_boxes.sum() * 0.0  # 保持计算图
            loss_dfl = raw_boxes.sum() * 0.0
            total = self.cls_gain * loss_cls + self.box_gain * loss_box + self.dfl_gain * loss_dfl
            return {"loss": total, "loss_box": loss_box, "loss_cls": loss_cls, "loss_dfl": loss_dfl}

        # --- Per-image 匹配 + 损失 ---
        all_loss_box = []
        all_loss_cls = []
        all_loss_dfl = []

        for b_idx in range(B):
            # 该图的 GT
            mask = (gt_batch_idx == b_idx)
            if not mask.any():
                # 无 GT: 全背景
                target_b = torch.zeros(A, self.num_classes, device=device)
                loss_cls_b = F.binary_cross_entropy_with_logits(
                    pred_scores[b_idx], target_b, reduction="mean")
                loss_box_b = pred_boxes[b_idx].sum() * 0.0
                loss_dfl_b = raw_boxes[b_idx].sum() * 0.0
                all_loss_cls.append(loss_cls_b)
                all_loss_box.append(loss_box_b)
                all_loss_dfl.append(loss_dfl_b)
                continue

            gt_cls_b = gt_cls[mask].long().squeeze(-1)    # [G]
            gt_bbox_b = gt_bboxes[mask]                   # [G, 4] cxcywh
            gt_xyxy_b = _cxcywh_to_xyxy(gt_bbox_b)       # [G, 4] xyxy 归一化
            G = gt_cls_b.shape[0]

            pred_xyxy_b = pred_boxes[b_idx]                # [A, 4]

            # --- Top-k IoU 匹配 ---
            iou = _bbox_iou(gt_xyxy_b, pred_xyxy_b)       # [G, A]
            topk_k = min(self.topk, A)
            _, topk_indices = iou.topk(topk_k, dim=1)      # [G, topk_k]

            # 构建匹配: 正样本 mask [A], 正样本对应的 GT 索引 [A]
            pos_mask = torch.zeros(A, dtype=torch.bool, device=device)
            anchor_to_gt = torch.full((A,), -1, dtype=torch.long, device=device)

            for g_idx in range(G):
                for a_idx in topk_indices[g_idx]:
                    a = int(a_idx)
                    if not pos_mask[a]:
                        # 首次匹配
                        pos_mask[a] = True
                        anchor_to_gt[a] = g_idx
                    elif iou[g_idx, a] > iou[anchor_to_gt[a], a]:
                        # 冲突: 取 IoU 更高的 GT
                        anchor_to_gt[a] = g_idx

            n_pos = pos_mask.sum().item()

            # --- 分类损失: BCE ---
            target_b = torch.zeros(A, self.num_classes, device=device)
            if n_pos > 0:
                pos_gt_cls = gt_cls_b[anchor_to_gt[pos_mask]]  # [n_pos]
                # one-hot, 但 soft: 用 IoU 值作为 target score
                pos_gt_idx = anchor_to_gt[pos_mask]
                pos_iou = iou[pos_gt_idx, torch.where(pos_mask)[0]]  # [n_pos]
                # Clamp IoU as soft label (避免 0 target)
                soft_label = pos_iou.clamp(min=0.0, max=1.0)
                for i, (cls_id, sl) in enumerate(zip(pos_gt_cls, soft_label)):
                    target_b[torch.where(pos_mask)[0][i], cls_id.clamp(0, self.num_classes - 1)] = sl

            loss_cls_b = F.binary_cross_entropy_with_logits(
                pred_scores[b_idx], target_b, reduction="mean")
            all_loss_cls.append(loss_cls_b)

            # --- 回归损失: CIOU (仅正样本) ---
            if n_pos > 0:
                pos_pred = pred_xyxy_b[pos_mask]                      # [n_pos, 4]
                pos_target = gt_xyxy_b[anchor_to_gt[pos_mask]]        # [n_pos, 4]
                loss_box_b = _ciou_loss(pos_pred, pos_target).mean()
            else:
                loss_box_b = pred_xyxy_b.sum() * 0.0
            all_loss_box.append(loss_box_b)

            # --- DFL 损失 (简化: 正样本的 ltrb softmax cross-entropy) ---
            if n_pos > 0:
                # 计算正样本的真实 ltrb offset (in stride units)
                pos_anchors = anchor_points[pos_mask]                  # [n_pos, 2]
                pos_strides = stride_tensor[pos_mask]                  # [n_pos, 1]
                pos_gt_xyxy = gt_xyxy_b[anchor_to_gt[pos_mask]] * img_size  # 反归一化

                # ltrb = (anchor - gt_x1y1, gt_x2y2 - anchor) / stride
                lt = (pos_anchors - pos_gt_xyxy[:, :2]) / pos_strides  # [n_pos, 2]
                rb = (pos_gt_xyxy[:, 2:] - pos_anchors) / pos_strides  # [n_pos, 2]
                target_ltrb = torch.cat([lt, rb], dim=-1).clamp(min=0, max=self.reg_max - 1.01)

                # DFL: cross-entropy on distribution
                # target: 连续值 → 相邻两个 bin 的插值 label
                target_ltrb_floor = target_ltrb.long()
                target_ltrb_frac = target_ltrb - target_ltrb_floor.float()

                # 提取正样本的 raw DFL logits
                pos_indices = torch.where(pos_mask)[0]
                # raw_boxes[b_idx] → [reg_max*4, A]
                pos_raw = raw_boxes[b_idx][:, pos_indices]  # [reg_max*4, n_pos]
                pos_raw = pos_raw.reshape(4, self.reg_max, n_pos).permute(2, 0, 1)  # [n_pos, 4, reg_max]

                # DFL loss: 对每个方向做交叉熵
                loss_dfl_b = torch.tensor(0.0, device=device)
                for d in range(4):
                    logits = pos_raw[:, d, :]                   # [n_pos, reg_max]
                    t_floor = target_ltrb_floor[:, d]           # [n_pos]
                    t_ceil = (t_floor + 1).clamp(max=self.reg_max - 1)
                    t_frac = target_ltrb_frac[:, d]             # [n_pos]

                    # 两个相邻 bin 的 soft cross-entropy
                    loss_l = F.cross_entropy(logits, t_floor, reduction="none")
                    loss_r = F.cross_entropy(logits, t_ceil, reduction="none")
                    loss_dfl_b = loss_dfl_b + ((1.0 - t_frac) * loss_l + t_frac * loss_r).mean()

                loss_dfl_b = loss_dfl_b / 4.0
            else:
                loss_dfl_b = raw_boxes[b_idx].sum() * 0.0
            all_loss_dfl.append(loss_dfl_b)

        # --- 平均 ---
        loss_box = torch.stack(all_loss_box).mean()
        loss_cls = torch.stack(all_loss_cls).mean()
        loss_dfl = torch.stack(all_loss_dfl).mean()

        total = self.box_gain * loss_box + self.cls_gain * loss_cls + self.dfl_gain * loss_dfl

        return {
            "loss": total,
            "loss_box": loss_box,
            "loss_cls": loss_cls,
            "loss_dfl": loss_dfl,
        }

    @staticmethod
    def from_ultralytics(
        model_path: str,
    ) -> "TopoDetectionLoss":
        """
        从 ultralytics 模型文件构建检测损失

        加载 YOLO 模型, 提取 num_classes/reg_max, 构建 TopoDetectionLoss
        """
        try:
            from ultralytics import YOLO
            yolo = YOLO(model_path)
            model = yolo.model
            # 提取参数
            nc = model.nc if hasattr(model, 'nc') else 5
            detect = model.model[-1] if hasattr(model, 'model') else None
            reg_max = detect.reg_max if detect and hasattr(detect, 'reg_max') else 16
            return TopoDetectionLoss(
                yolo_model=model,
                num_classes=nc,
                reg_max=reg_max,
            )
        except ImportError:
            raise ImportError(
                "ultralytics package required for from_ultralytics(). "
                "Install with: pip install ultralytics"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model from {model_path}: {e}")
