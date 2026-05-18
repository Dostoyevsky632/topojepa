"""
评估指标
========
除标准检测指标 (mAP) 外, 增加拓扑相关指标

拓扑指标无外部依赖 (纯 numpy/torch 实现)
检测指标支持 ultralytics 加速 (可选)
"""

import math
import torch
import numpy as np
from typing import Optional, Dict, List, Tuple

# PersistenceDiagram 类型 (兼容 torch 和 numpy)
PersistenceDiagram = Tuple  # (births, deaths)


# ============================================================
# 拓扑保真度
# ============================================================

def compute_topo_fidelity(
    pred_diagrams: List[PersistenceDiagram],
    target_diagrams: List[PersistenceDiagram],
    distance_type: str = "wasserstein",        # "wasserstein" | "bottleneck"
) -> float:
    """
    计算拓扑保真度: 预测嵌入和目标嵌入的拓扑一致性

    值越小 → 拓扑越一致

    使用 persistence 排序匹配的近似距离 (不依赖外部库)

    Returns:
        fidelity: 平均拓扑距离
    """
    total = 0.0
    n_dims = min(len(pred_diagrams), len(target_diagrams))
    if n_dims == 0:
        return 0.0

    for dim in range(n_dims):
        pb, pd = pred_diagrams[dim]
        tb, td = target_diagrams[dim]
        # 转 numpy
        if isinstance(pb, torch.Tensor):
            pb, pd = pb.detach().cpu().numpy(), pd.detach().cpu().numpy()
        if isinstance(tb, torch.Tensor):
            tb, td = tb.detach().cpu().numpy(), td.detach().cpu().numpy()

        p_pers = np.abs(pd - pb)
        t_pers = np.abs(td - tb)

        # 过滤零 persistence
        p_mask = p_pers > 1e-7
        t_mask = t_pers > 1e-7
        p_pts = np.stack([pb[p_mask], pd[p_mask]], axis=-1) if p_mask.any() else np.zeros((0, 2))
        t_pts = np.stack([tb[t_mask], td[t_mask]], axis=-1) if t_mask.any() else np.zeros((0, 2))

        if distance_type == "wasserstein":
            total += _approx_wasserstein(p_pts, t_pts)
        else:
            total += _approx_bottleneck(p_pts, t_pts)

    return total / max(n_dims, 1)


def _approx_wasserstein(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """
    近似 Wasserstein 距离 (persistence 排序匹配)
    pts_a, pts_b: [K, 2] (birth, death)
    """
    if len(pts_a) == 0 and len(pts_b) == 0:
        return 0.0

    # 各自加对角投影, 使大小相同
    diag_a = (pts_a.sum(axis=-1, keepdims=True) / 2).repeat(2, axis=-1) if len(pts_a) > 0 else np.zeros((0, 2))
    diag_b = (pts_b.sum(axis=-1, keepdims=True) / 2).repeat(2, axis=-1) if len(pts_b) > 0 else np.zeros((0, 2))

    aug_a = np.concatenate([pts_a, diag_b], axis=0) if len(diag_b) > 0 else pts_a
    aug_b = np.concatenate([pts_b, diag_a], axis=0) if len(diag_a) > 0 else pts_b

    if len(aug_a) == 0 or len(aug_b) == 0:
        return 0.0

    # 按 persistence 排序匹配
    pers_a = np.abs(aug_a[:, 1] - aug_a[:, 0])
    pers_b = np.abs(aug_b[:, 1] - aug_b[:, 0])
    order_a = np.argsort(pers_a)[::-1]
    order_b = np.argsort(pers_b)[::-1]

    n = min(len(order_a), len(order_b))
    cost = np.linalg.norm(aug_a[order_a[:n]] - aug_b[order_b[:n]], axis=-1)
    return float(cost.mean())


def _approx_bottleneck(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """近似 bottleneck 距离 (persistence 排序匹配, 取 max)"""
    if len(pts_a) == 0 and len(pts_b) == 0:
        return 0.0

    diag_a = (pts_a.sum(axis=-1, keepdims=True) / 2).repeat(2, axis=-1) if len(pts_a) > 0 else np.zeros((0, 2))
    diag_b = (pts_b.sum(axis=-1, keepdims=True) / 2).repeat(2, axis=-1) if len(pts_b) > 0 else np.zeros((0, 2))

    aug_a = np.concatenate([pts_a, diag_b], axis=0) if len(diag_b) > 0 else pts_a
    aug_b = np.concatenate([pts_b, diag_a], axis=0) if len(diag_a) > 0 else pts_b

    if len(aug_a) == 0 or len(aug_b) == 0:
        return 0.0

    pers_a = np.abs(aug_a[:, 1] - aug_a[:, 0])
    pers_b = np.abs(aug_b[:, 1] - aug_b[:, 0])
    order_a = np.argsort(pers_a)[::-1]
    order_b = np.argsort(pers_b)[::-1]

    n = min(len(order_a), len(order_b))
    cost = np.linalg.norm(aug_a[order_a[:n]] - aug_b[order_b[:n]], axis=-1)
    return float(cost.max()) if len(cost) > 0 else 0.0


# ============================================================
# Betti 数
# ============================================================

def compute_betti_numbers(
    embeddings: np.ndarray,                    # [N, D]
    threshold: float = 0.5,
    max_dim: int = 1,
) -> np.ndarray:
    """
    计算嵌入集合的 Betti 数

    使用纯 numpy 近似:
      - B0: 距离阈值下的连通分量 (Union-Find)
      - B1: 简单近似 (非 MST 边 - 冗余边 = 环路)

    Returns:
        betti: [max_dim+1] 各维度 Betti 数
    """
    N = embeddings.shape[0]
    if N <= 1:
        return np.array([N] + [0] * max_dim)

    # 距离矩阵
    diff = embeddings[:, None, :] - embeddings[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=-1))

    # --- B0: 连通分量 (Union-Find) ---
    parent = list(range(N))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edges_below = 0
    mst_edges = 0
    for i in range(N):
        for j in range(i + 1, N):
            if dist[i, j] <= threshold:
                edges_below += 1
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
                    mst_edges += 1

    # 统计连通分量
    components = len(set(find(i) for i in range(N)))
    betti = [components]

    # --- B1: 近似环路数 ---
    if max_dim >= 1:
        # 在阈值 ε 下: B1 ≈ #edges_below_threshold - #mst_edges - ... (Euler 特征近似)
        # 更精确: B1 = E - V + C (对 1-skeleton, V=N, E=edges_below, C=components)
        b1 = max(0, edges_below - (N - components))
        betti.append(b1)

    # 高维: 占位
    while len(betti) < max_dim + 1:
        betti.append(0)

    return np.array(betti[:max_dim + 1])


# ============================================================
# 嵌入坍塌程度
# ============================================================

def compute_collapse_metric(
    embeddings: np.ndarray,                    # [N, D]
) -> Dict[str, float]:
    """
    计算嵌入坍塌程度指标

    Returns:
        dict with:
          - "rank": 嵌入矩阵的有效秩 (越接近 D 越好)
          - "uniformity": 嵌入在超球面的均匀性 (越小越均匀)
          - "topo_diversity": 拓扑多样性 (B0 + 总持久性的代理)
          - "std": 各维度标准差的平均值
    """
    N, D = embeddings.shape

    # --- 有效秩 (基于奇异值) ---
    # effective rank = exp(entropy of normalized singular values)
    _, s, _ = np.linalg.svd(embeddings, full_matrices=False)
    s_norm = s / (s.sum() + 1e-10)
    s_norm = s_norm[s_norm > 1e-10]  # 过滤零
    entropy = -np.sum(s_norm * np.log(s_norm + 1e-10))
    effective_rank = float(np.exp(entropy))

    # --- 均匀性 (Wang & Isola, 2020) ---
    # uniformity = log E[exp(-2 ||z_i - z_j||^2)]
    # L2 归一化
    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    normed = embeddings / (norms + 1e-10)
    # 采样: 大 N 时随机取子集
    max_pairs = min(N, 1000)
    if N > max_pairs:
        idx = np.random.choice(N, max_pairs, replace=False)
        normed_sub = normed[idx]
    else:
        normed_sub = normed

    diff = normed_sub[:, None, :] - normed_sub[None, :, :]
    sq_dist = (diff * diff).sum(axis=-1)
    # 排除对角线
    mask = ~np.eye(len(normed_sub), dtype=bool)
    uniformity = float(np.log(np.exp(-2.0 * sq_dist[mask]).mean() + 1e-10))

    # --- 拓扑多样性 ---
    # 用距离的标准差作为代理: 高标准差 → 结构丰富
    triu_idx = np.triu_indices(min(N, 500), k=1)
    if N > 500:
        idx = np.random.choice(N, 500, replace=False)
        sub = embeddings[idx]
    else:
        sub = embeddings
    diff_sub = sub[:, None, :] - sub[None, :, :]
    dists = np.sqrt((diff_sub * diff_sub).sum(axis=-1))[triu_idx]
    topo_diversity = float(dists.std())

    # --- 各维度标准差 ---
    dim_std = float(embeddings.std(axis=0).mean())

    return {
        "rank": effective_rank,
        "uniformity": uniformity,
        "topo_diversity": topo_diversity,
        "std": dim_std,
    }


# ============================================================
# 对齐度
# ============================================================

def compute_alignment_metric(
    pred_embeddings: np.ndarray,               # [N, D]
    target_embeddings: np.ndarray,             # [N, D]
) -> Dict[str, float]:
    """
    计算对齐度指标

    Returns:
        dict with:
          - "cosine_sim": 平均余弦相似度
          - "l2_dist": 平均 L2 距离
          - "alignment": Wang & Isola alignment = E[||z_i - z_j+||^2]
    """
    N = pred_embeddings.shape[0]

    # L2 归一化
    p_norm = pred_embeddings / (np.linalg.norm(pred_embeddings, axis=-1, keepdims=True) + 1e-10)
    t_norm = target_embeddings / (np.linalg.norm(target_embeddings, axis=-1, keepdims=True) + 1e-10)

    # 余弦相似度
    cosine_sim = float((p_norm * t_norm).sum(axis=-1).mean())

    # L2 距离
    l2_dist = float(np.linalg.norm(pred_embeddings - target_embeddings, axis=-1).mean())

    # alignment (Wang & Isola): E[||f(x) - f+(x)||^alpha], alpha=2
    alignment = float(np.mean(np.sum((p_norm - t_norm) ** 2, axis=-1)))

    return {
        "cosine_sim": cosine_sim,
        "l2_dist": l2_dist,
        "alignment": alignment,
    }


# ============================================================
# 检测指标 (mAP)
# ============================================================

def compute_detection_metrics(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_thresholds: Optional[List[float]] = None,
    class_names: Optional[dict] = None,
) -> Dict[str, float]:
    """
    计算检测指标

    每个 predictions[i] / ground_truths[i] 格式:
      {"boxes": [K, 4] xyxy, "scores": [K], "labels": [K]}

    纯 numpy 实现, 不依赖 ultralytics (但规模大时建议用 pycocotools)

    Returns:
        dict with: mAP50, mAP50-95, precision, recall, f1, per-class AP
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5 + i * 0.05 for i in range(10)]  # 0.5:0.05:0.95

    # [FIX-P1-②] 收集所有类别: 同时从 GT 和 predictions 中提取
    # 确保 "GT 中没出现但模型预测了" 的类别的 FP 也被计入
    all_classes = set()
    for gt in ground_truths:
        labels = gt.get("labels", [])
        if len(labels) > 0:
            all_classes.update(labels.tolist() if hasattr(labels, 'tolist') else list(labels))
    for pred in predictions:
        labels = pred.get("labels", [])
        if len(labels) > 0:
            all_classes.update(labels.tolist() if hasattr(labels, 'tolist') else list(labels))

    if not all_classes:
        return {"mAP50": 0.0, "mAP50-95": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    all_classes = sorted(all_classes)

    # Per-class, per-threshold AP
    aps = {t: [] for t in iou_thresholds}
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for cls_id in all_classes:
        # ----- 第一遍: 收集每张图的预测 + 逐图匹配 TP/FP -----
        # 存储: (score, {threshold: is_tp}) 的列表, 用于全局排序
        det_entries = []  # List[ (score, Dict[float, int]) ]
        n_gt = 0

        # 遍历 max(len(predictions), len(ground_truths))
        # 多出的 predictions → 全是 FP; 多出的 GT → 全是 FN
        _empty_det = {"boxes": np.zeros((0, 4)), "scores": np.array([]), "labels": np.array([])}
        _empty_gt = {"boxes": np.zeros((0, 4)), "labels": np.array([])}
        n_images = max(len(ground_truths), len(predictions))

        for img_idx in range(n_images):
            gt = ground_truths[img_idx] if img_idx < len(ground_truths) else _empty_gt
            pred = predictions[img_idx] if img_idx < len(predictions) else _empty_det

            gt_boxes = np.array(gt.get("boxes", np.zeros((0, 4))))
            gt_labels = np.array(gt.get("labels", []))
            pred_boxes = np.array(pred.get("boxes", np.zeros((0, 4))))
            pred_scores = np.array(pred.get("scores", []))
            pred_labels = np.array(pred.get("labels", []))

            # 筛选该类
            gt_mask = gt_labels == cls_id
            pred_mask = pred_labels == cls_id
            gt_cls = gt_boxes[gt_mask]
            pred_cls = pred_boxes[pred_mask]
            scores_cls = pred_scores[pred_mask]

            n_gt += len(gt_cls)

            if len(pred_cls) == 0:
                continue

            # 图内按 score 降序 (贪心匹配需要)
            order = np.argsort(-scores_cls)
            pred_cls = pred_cls[order]
            scores_cls = scores_cls[order]

            # 计算 IoU
            if len(gt_cls) > 0:
                iou_matrix = _np_bbox_iou(pred_cls, gt_cls)  # [P, G]
            else:
                iou_matrix = np.zeros((len(pred_cls), 0))

            # 逐阈值贪心匹配 (图内)
            tp_flags_per_t = {t: [] for t in iou_thresholds}
            for t in iou_thresholds:
                matched = set()
                for p_idx in range(len(pred_cls)):
                    if iou_matrix.shape[1] > 0:
                        best_gt = int(iou_matrix[p_idx].argmax())
                        if iou_matrix[p_idx, best_gt] >= t and best_gt not in matched:
                            tp_flags_per_t[t].append(1)
                            matched.add(best_gt)
                        else:
                            tp_flags_per_t[t].append(0)
                    else:
                        tp_flags_per_t[t].append(0)

            # 把每个预测框记为一条 entry (score + 各阈值 TP flag)
            for p_idx in range(len(pred_cls)):
                entry_flags = {t: tp_flags_per_t[t][p_idx] for t in iou_thresholds}
                det_entries.append((float(scores_cls[p_idx]), entry_flags))

        # ----- 第二遍: [FIX-P1-①] 按全局 score 降序排列, 再累计 TP/FP -----
        det_entries.sort(key=lambda x: -x[0])

        for t in iou_thresholds:
            if len(det_entries) == 0 and n_gt == 0:
                aps[t].append(0.0)
                continue

            tp_arr = np.array([e[1][t] for e in det_entries]) if det_entries else np.array([])
            if len(tp_arr) == 0:
                # 有 GT 但无该类预测 → AP=0
                aps[t].append(0.0)
                continue

            tp_cumsum = np.cumsum(tp_arr)
            fp_cumsum = np.cumsum(1 - tp_arr)
            precision_curve = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
            recall_curve = tp_cumsum / (n_gt + 1e-10) if n_gt > 0 else np.ones_like(tp_cumsum) * 0.0

            # AP = area under precision-recall curve (all-point interpolation, COCO 风格)
            ap = _compute_ap_all_points(recall_curve, precision_curve)
            aps[t].append(ap)

        # Global TP/FP/FN at IoU=0.5
        tp_50_arr = np.array([e[1][0.5] for e in det_entries]) if det_entries else np.array([])
        total_tp += int(tp_50_arr.sum()) if len(tp_50_arr) > 0 else 0
        total_fp += int((1 - tp_50_arr).sum()) if len(tp_50_arr) > 0 else 0
        total_fn += max(0, n_gt - (int(tp_50_arr.sum()) if len(tp_50_arr) > 0 else 0))

    # Aggregate
    mAP50 = float(np.mean(aps.get(0.5, [0.0])))
    mAP_all = [float(np.mean(aps[t])) for t in iou_thresholds]
    mAP50_95 = float(np.mean(mAP_all))

    precision = total_tp / (total_tp + total_fp + 1e-10)
    recall = total_tp / (total_tp + total_fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    result = {
        "mAP50": mAP50,
        "mAP50-95": mAP50_95,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

    # Per-class AP (at IoU=0.5)
    if class_names:
        for i, cls_id in enumerate(all_classes):
            name = class_names.get(cls_id, f"class_{cls_id}")
            if i < len(aps.get(0.5, [])):
                result[f"AP50_{name}"] = aps[0.5][i]

    return result


def _np_bbox_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """numpy IoU: [N, 4] x [M, 4] → [N, M], xyxy format"""
    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = np.maximum(0, boxes_a[:, 2] - boxes_a[:, 0]) * np.maximum(0, boxes_a[:, 3] - boxes_a[:, 1])
    area_b = np.maximum(0, boxes_b[:, 2] - boxes_b[:, 0]) * np.maximum(0, boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / (union + 1e-10)


# ============================================================
# 分割指标
# ============================================================

def compute_segmentation_metrics(
    pred_masks: np.ndarray,       # [B, H, W] or [B, 1, H, W] float/bool
    gt_masks: np.ndarray,         # [B, H, W] or [B, 1, H, W] float/bool
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    计算分割评估指标: Dice coefficient, pixel IoU, pixel accuracy

    Args:
        pred_masks: 预测 mask (sigmoid 后的概率或二值)
        gt_masks: 真实 mask (二值)
        threshold: 二值化阈值

    Returns:
        dict with: dice, iou, pixel_accuracy
    """
    # 展平到 [B, H*W]
    pred = pred_masks.reshape(pred_masks.shape[0], -1)
    gt = gt_masks.reshape(gt_masks.shape[0], -1)

    # 二值化
    pred_bin = (pred > threshold).astype(np.float32)
    gt_bin = (gt > threshold).astype(np.float32)

    # Per-sample metrics, then average
    eps = 1e-7
    intersection = (pred_bin * gt_bin).sum(axis=1)
    pred_sum = pred_bin.sum(axis=1)
    gt_sum = gt_bin.sum(axis=1)

    # Dice = 2|P∩G| / (|P| + |G|)
    dice = (2.0 * intersection + eps) / (pred_sum + gt_sum + eps)

    # IoU = |P∩G| / |P∪G|
    union = pred_sum + gt_sum - intersection
    iou = (intersection + eps) / (union + eps)

    # Pixel accuracy = (TP + TN) / total
    total_pixels = pred.shape[1]
    correct = ((pred_bin == gt_bin).sum(axis=1)).astype(np.float32)
    pixel_acc = correct / total_pixels

    return {
        "dice": float(dice.mean()),
        "iou": float(iou.mean()),
        "pixel_accuracy": float(pixel_acc.mean()),
    }


def _compute_ap_all_points(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    All-point interpolation AP (COCO / VOC2010+ 风格)

    在 recall 轴上从右往左取 precision 的 running-max (单调化),
    然后计算曲线下面积。比 11-point 更稳定, 是论文标准。
    """
    # 在前后补 sentinel 值
    mrec = np.concatenate(([0.0], recall, [recall[-1] + 1e-3]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # 从右往左取 running-max → 单调不增
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # recall 变化点
    idx = np.where(mrec[1:] != mrec[:-1])[0]

    # 面积 = Σ (Δrecall × precision)
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap
