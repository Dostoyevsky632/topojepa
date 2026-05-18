"""
可视化工具
==========
论文中需要的可视化:
  1. Persistence Diagram (持久图)
  2. Betti Curves (Betti 数随阈值变化)
  3. 嵌入空间 t-SNE + 拓扑着色
  4. 训练曲线 (多损失项)

所有函数: matplotlib 可选, 无 matplotlib 时安全 skip + 日志警告
"""

import logging
import numpy as np
from typing import Optional, List, Tuple
from pathlib import Path
from PIL import Image

logger = logging.getLogger("topojepa.viz")

try:
    import matplotlib
    matplotlib.use("Agg")  # 非交互后端, 服务器安全
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    logger.warning("matplotlib not installed; visualization functions will be no-ops")


def _ensure_mpl():
    if not HAS_MPL:
        logger.warning("matplotlib not available, skipping plot")
        return False
    return True


def _save_or_show(fig, save_path: Optional[str]):
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved: %s", save_path)
    plt.close(fig)


def _load_image(path: str) -> np.ndarray:
    """Load image as RGB numpy array."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def _load_mask_from_label(label_path: str, image_shape: Tuple[int, int]) -> np.ndarray:
    """Build a binary mask from YOLO labels using bbox rectangles."""
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    lbl = Path(label_path)
    if not lbl.exists():
        return mask

    try:
        data = np.loadtxt(str(lbl), dtype=np.float32)
    except Exception:
        return mask

    if data.size == 0:
        return mask
    if data.ndim == 1:
        data = data[np.newaxis, :]

    for row in data:
        _, cx, cy, bw, bh = row[:5]
        x1 = max(0, int((cx - bw / 2) * w))
        y1 = max(0, int((cy - bh / 2) * h))
        x2 = min(w, int((cx + bw / 2) * w))
        y2 = min(h, int((cy + bh / 2) * h))
        mask[y1:y2, x1:x2] = 255

    return mask


def _find_first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists() and p.is_dir():
            return p
    return None


def _collect_dataset_images(root: Path) -> List[Path]:
    """Collect image files from common dataset directory layouts."""
    if not root.exists():
        return []

    search_dirs = [
        root,
        root / "images",
        root / "image",
        root / "imgs",
        root / "JPEGImages",
        root / "val" / "images",
        root / "train" / "images",
        root / "test" / "images",
        root / "images" / "val",
        root / "images" / "train",
        root / "training" / "images",
        root / "testing" / "images",
    ]
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    collected = []
    for d in search_dirs:
        if d.exists() and d.is_dir():
            collected.extend([p for p in d.rglob("*") if p.suffix.lower() in exts])
    unique = []
    seen = set()
    for p in sorted(collected):
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def _guess_label_path(image_path: Path) -> Optional[Path]:
    """Guess annotation path from an image path using common layouts."""
    candidates = []
    stem = image_path.stem
    parents = list(image_path.parents)

    # YOLO style: .../images/foo.jpg -> .../labels/foo.txt
    for parent in parents[:5]:
        if parent.name.lower() == "images":
            candidates.append(parent.parent / "labels" / f"{stem}.txt")
            candidates.append(parent.parent / "labels" / image_path.relative_to(parent).with_suffix(".txt"))
        if parent.name.lower() == "training":
            candidates.append(parent / "labels" / f"{stem}.txt")

    candidates.extend([
        image_path.with_suffix(".txt"),
        image_path.parent / "labels" / f"{stem}.txt",
        image_path.parent / "masks" / f"{stem}.png",
        image_path.parent / "mask" / f"{stem}.png",
        image_path.parent.parent / "mask" / f"{stem}.png",
        image_path.parent.parent / "ground_truth" / f"{stem}.png",
        image_path.parent.parent / "ground_truth" / f"{stem}.tif",
        image_path.parent.parent / "ground_truth" / f"{stem}.tiff",
        image_path.parent.parent / "mask" / f"{stem}.png",
    ])
    for c in candidates:
        if c.exists():
            return c
    return None


def _load_annotation_mask(label_or_mask_path: Optional[Path], image_shape: Tuple[int, int]) -> np.ndarray:
    if label_or_mask_path is None or not label_or_mask_path.exists():
        return np.zeros(image_shape, dtype=np.uint8)
    if label_or_mask_path.suffix.lower() == ".txt":
        return _load_mask_from_label(str(label_or_mask_path), image_shape)
    try:
        mask = np.asarray(Image.open(label_or_mask_path).convert("L"))
        if mask.shape != image_shape:
            mask_img = Image.fromarray(mask)
            mask = np.asarray(mask_img.resize((image_shape[1], image_shape[0]), Image.NEAREST))
        return mask
    except Exception:
        return np.zeros(image_shape, dtype=np.uint8)


def _make_demo_topology_panel(kind: str, size: int = 320) -> Tuple[np.ndarray, np.ndarray, str]:
    """Create a lightweight illustrative demo panel for dataset figures.

    Returns an RGB image, a binary mask, and a short topological cue label.
    """
    h = w = size
    yy, xx = np.mgrid[0:h, 0:w]

    if kind == "RDD":
        img = np.dstack([
            0.35 + 0.18 * np.sin(xx / 18.0) + 0.06 * np.cos(yy / 11.0),
            0.28 + 0.16 * np.sin((xx + yy) / 25.0),
            0.22 + 0.14 * np.cos(xx / 19.0),
        ])
        mask = np.zeros((h, w), dtype=np.uint8)
        for offset in [40, 95, 155, 220]:
            y = (0.65 * xx + offset + 16 * np.sin(xx / 28.0)).astype(int)
            valid = (y >= 0) & (y < h)
            mask[y[valid], xx[valid]] = 255
        cue = "fragmented cracks"
    elif kind == "DRIVE":
        img = np.dstack([
            0.42 + 0.10 * np.sin(xx / 24.0) + 0.04 * np.cos(yy / 15.0),
            0.44 + 0.08 * np.cos((xx - yy) / 26.0),
            0.40 + 0.10 * np.sin(yy / 21.0),
        ])
        mask = np.zeros((h, w), dtype=np.uint8)
        paths = [
            (0.15 * w, 0.55 * h, 0.38 * w, 0.35 * h, 0.62 * w, 0.30 * h, 0.86 * w, 0.48 * h),
            (0.22 * w, 0.82 * h, 0.40 * w, 0.65 * h, 0.58 * w, 0.57 * h, 0.77 * w, 0.68 * h),
        ]
        for p in paths:
            pts = np.array(p, dtype=np.float32).reshape(-1, 2)
            for i in range(len(pts) - 1):
                x1, y1 = pts[i]
                x2, y2 = pts[i + 1]
                n = int(max(abs(x2 - x1), abs(y2 - y1)) * 2) + 1
                xs = np.linspace(x1, x2, n).astype(int)
                ys = np.linspace(y1, y2, n).astype(int)
                for x, y in zip(xs, ys):
                    rr = np.arange(max(0, y - 2), min(h, y + 3))
                    cc = np.arange(max(0, x - 2), min(w, x + 3))
                    mask[np.ix_(rr, cc)] = 255
        cue = "disconnected thin vessels"
    else:
        img = np.dstack([
            0.46 + 0.14 * np.sin(xx / 34.0),
            0.47 + 0.10 * np.cos(yy / 30.0),
            0.45 + 0.08 * np.sin((xx + yy) / 40.0),
        ])
        mask = np.zeros((h, w), dtype=np.uint8)
        boxes = [(45, 55, 115, 120), (145, 40, 225, 125), (90, 155, 175, 240), (205, 165, 275, 255)]
        for x1, y1, x2, y2 in boxes:
            mask[y1:y2, x1:x2] = 255
            mask[max(0, y1-2):min(h, y1+2), x1:x2] = 0
            mask[y1:y2, max(0, x1-2):min(w, x1+2)] = 0
        cue = "incomplete building boundaries"

    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8), mask, cue


def plot_topology_dataset_figure(
    save_path: Optional[str] = None,
    title: str = "Topology-rich segmentation datasets",
    data_roots: Optional[dict] = None,
    rows: Optional[List[str]] = None,
) -> None:
    """Generate a 3x3 figure for RDD, DRIVE, and Inria.

    Layout:
      1) Input image
      2) Ground-truth mask overlay
      3) Zoomed topological cue region
    """
    if not _ensure_mpl():
        return

    if save_path is None:
        save_path = str(Path(r"C:\Users\dongma\Desktop\yolo_crack") / "figures" / "topology_datasets.png")

    rows = rows or ["RDD", "DRIVE", "Inria"]
    col_titles = ["Input image", "GT mask overlay", "Zoomed topological cue"]
    data_roots = data_roots or {
        "RDD": {
            "images": Path(r"C:\Users\dongma\Desktop\yolo_crack\RDD_SPLIT\train\images"),
            "labels": Path(r"C:\Users\dongma\Desktop\yolo_crack\RDD_SPLIT\train\labels"),
        },
        "DRIVE": {
            "images": Path(r"C:\Users\dongma\Desktop\yolo_crack\DRIVE\training\images"),
            "labels": Path(r"C:\Users\dongma\Desktop\yolo_crack\DRIVE\training\mask"),
        },
        "Inria": {
            "images": Path(r"C:\Users\dongma\Desktop\yolo_crack\inria\images"),
            "labels": Path(r"C:\Users\dongma\Desktop\yolo_crack\inria\ground_truth"),
        },
    }
    fig, axes = plt.subplots(len(rows), 3, figsize=(12.0, 10.2))
    if len(rows) == 1:
        axes = np.expand_dims(axes, 0)

    def _bbox_from_mask(mask_arr: np.ndarray, pad: int = 24):
        ys, xs = np.where(mask_arr > 0)
        if len(xs) == 0:
            h, w = mask_arr.shape
            return w // 4, h // 4, 3 * w // 4, 3 * h // 4
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        h, w = mask_arr.shape
        return max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)

    figures_dir = Path(r"C:\Users\dongma\Desktop\yolo_crack") / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for r, name in enumerate(rows):
        root_cfg = data_roots.get(name, {})
        img_root = Path(root_cfg.get("images", ""))
        lbl_root = Path(root_cfg.get("labels", ""))
        images = _collect_dataset_images(img_root)
        if images:
            candidates = []
            for p in images:
                lbl = _guess_label_path(p)
                if lbl is None and lbl_root.exists():
                    lbl = lbl_root / f"{p.stem}.txt"
                if lbl is None:
                    continue
                if lbl.exists() and lbl.stat().st_size > 0:
                    candidates.append((p, lbl))
            if not candidates:
                candidates = [(images[len(images) // 2], _guess_label_path(images[len(images) // 2]))]
            image_path, label_path = candidates[len(candidates) // 2]
            img = _load_image(str(image_path))
            mask = _load_annotation_mask(label_path, img.shape[:2])
            if name == "Inria" and (mask.max() <= 1):
                mask = (mask > 0).astype(np.uint8) * 255
        else:
            img, mask, _ = _make_demo_topology_panel(name)

        task_name = {
            "RDD": "Road crack segmentation",
            "DRIVE": "Retinal vessel segmentation",
            "Inria": "Aerial building segmentation",
        }.get(name, name)
        cue_text = {
            "RDD": "elongated and fragmented crack topology",
            "DRIVE": "thin vessel connectivity and branching",
            "Inria": "closed-boundary and region continuity constraints",
        }.get(name, "topology-sensitive region")

        # Column 2: strong GT mask overlay.
        overlay = img.astype(np.float32).copy()
        if name == "RDD":
            color = np.array([255, 40, 40], dtype=np.float32)
        elif name == "DRIVE":
            color = np.array([0, 220, 180], dtype=np.float32)
        else:
            color = np.array([255, 210, 0], dtype=np.float32)
        overlay = 0.96 * overlay
        overlay[mask > 0] = 0.08 * overlay[mask > 0] + 0.92 * color

        # Column 3: zoomed cue around the most informative region.
        x1, y1, x2, y2 = _bbox_from_mask(mask)
        zoom = img[y1:y2, x1:x2].copy()
        zoom_mask = mask[y1:y2, x1:x2]
        zoom_overlay = zoom.astype(np.float32).copy()
        zoom_overlay = 0.90 * zoom_overlay
        zoom_overlay[zoom_mask > 0] = 0.05 * zoom_overlay[zoom_mask > 0] + 0.95 * color

        # Add a clear outline of the zoom box on the original image for context.
        boxed = img.copy().astype(np.float32)
        boxed[y1:y1+4, x1:x2] = 255
        boxed[max(0, y2-4):y2, x1:x2] = 255
        boxed[y1:y2, x1:x1+4] = 255
        boxed[y1:y2, max(0, x2-4):x2] = 255

        panels = [np.clip(boxed, 0, 255).astype(np.uint8), np.clip(overlay, 0, 255).astype(np.uint8), np.clip(zoom_overlay, 0, 255).astype(np.uint8)]
        row_dir = figures_dir / name.lower()
        row_dir.mkdir(parents=True, exist_ok=True)

        row_titles = ["input", "mask_overlay", "topological_zoom"]
        for c, panel in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(panel)
            ax.axis("off")
            if r == 0:
                ax.set_title(col_titles[c], fontsize=13, pad=10)
            fig_path = row_dir / f"{name.lower()}_{row_titles[c]}.png"
            plt.imsave(fig_path, panel)

        axes[r, 0].set_ylabel(task_name, rotation=0, labelpad=22, va="center", fontsize=12, fontweight="bold")
        axes[r, 1].text(
            0.03,
            0.06,
            "GT mask",
            transform=axes[r, 1].transAxes,
            fontsize=10,
            color="white",
            va="bottom",
            ha="left",
            bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=2),
        )
        axes[r, 2].text(
            0.03,
            0.94,
            cue_text,
            transform=axes[r, 2].transAxes,
            fontsize=10.5,
            color="white",
            va="top",
            ha="left",
            bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=3),
        )

    fig.suptitle(title, fontsize=15, y=0.995)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_rdd_split_sample_sheet(
    samples: List[dict],
    save_path: Optional[str] = None,
    title: str = "RDD_SPLIT dataset visualization",
    methods: Optional[List[str]] = None,
) -> None:
    """Create a comparison sheet matching the provided layout.

    Each sample dict should contain:
      - image_path
      - label_path
      - method_masks: dict with keys like "sam_med2d", "sam", "ips"
      - row_label: e.g. "3P", "5P", "16P"
      - group_label: optional label for the left margin grouping
    """
    if not _ensure_mpl():
        return

    methods = methods or ["sam_med2d", "sam", "ips"]
    col_titles = ["Images", "GT-Masks"] + [m.upper() if m != "ips" else "IPS (Ours)" for m in methods]
    n_rows = len(samples)
    n_cols = len(col_titles)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.15 * n_cols, 2.25 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)

    for r, sample in enumerate(samples):
        img = _load_image(sample["image_path"])
        gt_mask = _load_mask_from_label(sample["label_path"], img.shape[:2])
        row_masks = sample.get("method_masks", {})

        display_items = [img, gt_mask] + [row_masks.get(m) for m in methods]
        for c, item in enumerate(display_items):
            ax = axes[r, c]
            ax.axis("off")
            if c == 0:
                ax.imshow(item)
            else:
                if item is None:
                    ax.imshow(np.zeros_like(gt_mask), cmap="gray", vmin=0, vmax=255)
                else:
                    ax.imshow(item, cmap="gray", vmin=0, vmax=255)

        axes[r, 0].set_ylabel(sample.get("row_label", ""), rotation=0, labelpad=25, va="center", fontsize=12)

    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=13, pad=10)

    fig.suptitle(title, fontsize=14, y=0.995)
    fig.tight_layout()
    _save_or_show(fig, save_path)


# ============================================================
# 持久图
# ============================================================

def plot_persistence_diagram(
    births: np.ndarray,                        # [K]
    deaths: np.ndarray,                        # [K]
    title: str = "Persistence Diagram",
    save_path: Optional[str] = None,
    homology_dims: Optional[np.ndarray] = None,
) -> None:
    """
    绘制持久图

    X 轴: birth, Y 轴: death
    对角线: birth=death (持久性为 0)
    远离对角线的点: 显著的拓扑特征
    """
    if not _ensure_mpl():
        return

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # 对角线
    max_val = max(deaths.max(), births.max(), 1.0) if len(births) > 0 else 1.0
    ax.plot([0, max_val * 1.1], [0, max_val * 1.1], "k--", alpha=0.3, linewidth=1)

    if homology_dims is not None:
        dims = np.unique(homology_dims)
        colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
        markers = ["o", "^", "s", "D"]
        for i, d in enumerate(dims):
            mask = homology_dims == d
            c = colors[i % len(colors)]
            m = markers[i % len(markers)]
            ax.scatter(births[mask], deaths[mask], c=c, marker=m, s=40,
                       alpha=0.7, edgecolors="white", linewidth=0.5,
                       label=f"H{int(d)}")
        ax.legend(framealpha=0.9)
    else:
        ax.scatter(births, deaths, c="#2196F3", s=40, alpha=0.7,
                   edgecolors="white", linewidth=0.5)

    ax.set_xlabel("Birth", fontsize=12)
    ax.set_ylabel("Death", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(-0.02 * max_val, max_val * 1.1)
    ax.set_ylim(-0.02 * max_val, max_val * 1.1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    _save_or_show(fig, save_path)


# ============================================================
# Betti 曲线
# ============================================================

def plot_betti_curves(
    births: np.ndarray,                        # [K]
    deaths: np.ndarray,                        # [K]
    max_filtration: float = 1.0,
    num_points: int = 100,
    title: str = "Betti Curves",
    save_path: Optional[str] = None,
    homology_dims: Optional[np.ndarray] = None,
) -> None:
    """
    绘制 Betti 曲线

    X 轴: filtration 值 (距离阈值)
    Y 轴: 各维度的 Betti 数
    """
    if not _ensure_mpl():
        return

    epsilons = np.linspace(0, max_filtration, num_points)

    if homology_dims is None:
        homology_dims = np.zeros(len(births))

    dims = np.unique(homology_dims).astype(int)
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    for i, d in enumerate(dims):
        mask = homology_dims == d
        b = births[mask]
        de = deaths[mask]
        betti_vals = []
        for eps in epsilons:
            # 在阈值 eps 下存活的特征数: birth <= eps < death
            alive = np.sum((b <= eps) & (de > eps))
            betti_vals.append(alive)
        c = colors[i % len(colors)]
        ax.plot(epsilons, betti_vals, color=c, linewidth=2, label=f"β{d}")
        ax.fill_between(epsilons, betti_vals, alpha=0.1, color=c)

    ax.set_xlabel("Filtration value (ε)", fontsize=12)
    ax.set_ylabel("Betti number", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.2)

    _save_or_show(fig, save_path)


# ============================================================
# 嵌入空间拓扑可视化
# ============================================================

def plot_embedding_topology(
    embeddings: np.ndarray,                    # [N, D]
    labels: Optional[np.ndarray] = None,       # [N]
    betti_numbers: Optional[np.ndarray] = None,
    method: str = "tsne",                      # "tsne" | "umap"
    title: str = "Embedding Space Topology",
    save_path: Optional[str] = None,
) -> None:
    """
    绘制嵌入空间的拓扑可视化

    将高维嵌入降到 2D, 用颜色表示类别或 Betti 数
    """
    if not _ensure_mpl():
        return

    # 降维
    coords_2d = _reduce_dims(embeddings, method)
    if coords_2d is None:
        logger.warning("Dimensionality reduction failed, skipping plot")
        return

    n_plots = 1 + (betti_numbers is not None)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    # 子图 1: 类别着色
    ax = axes[0]
    if labels is not None:
        unique_labels = np.unique(labels)
        cmap = plt.cm.get_cmap("tab10", len(unique_labels))
        for i, lbl in enumerate(unique_labels):
            mask = labels == lbl
            ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                       c=[cmap(i)], s=15, alpha=0.6, label=f"{lbl}")
        ax.legend(fontsize=8, markerscale=2, framealpha=0.9)
    else:
        ax.scatter(coords_2d[:, 0], coords_2d[:, 1],
                   c="#2196F3", s=15, alpha=0.6)
    ax.set_title(f"{title} (classes)", fontsize=12)
    ax.grid(True, alpha=0.2)

    # 子图 2: Betti 数着色
    if betti_numbers is not None and n_plots > 1:
        ax2 = axes[1]
        # 用 B0 着色 (连通分量)
        b0 = betti_numbers[:, 0] if betti_numbers.ndim > 1 else betti_numbers
        sc = ax2.scatter(coords_2d[:, 0], coords_2d[:, 1],
                         c=b0, cmap="viridis", s=15, alpha=0.6)
        fig.colorbar(sc, ax=ax2, label="β₀")
        ax2.set_title(f"{title} (topology)", fontsize=12)
        ax2.grid(True, alpha=0.2)

    fig.tight_layout()
    _save_or_show(fig, save_path)


def _reduce_dims(embeddings: np.ndarray, method: str = "tsne") -> Optional[np.ndarray]:
    """降维到 2D"""
    N, D = embeddings.shape
    if D <= 2:
        return embeddings[:, :2]

    if method == "tsne":
        try:
            from sklearn.manifold import TSNE
            perplexity = min(30, max(5, N // 4))
            # sklearn >=1.5 重命名 n_iter → max_iter
            import inspect
            _tsne_params = inspect.signature(TSNE.__init__).parameters
            iter_key = "max_iter" if "max_iter" in _tsne_params else "n_iter"
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, **{iter_key: 500})
            return tsne.fit_transform(embeddings)
        except ImportError:
            logger.warning("sklearn not installed, falling back to PCA")
            return _pca_2d(embeddings)
    elif method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            return reducer.fit_transform(embeddings)
        except ImportError:
            logger.warning("umap not installed, falling back to PCA")
            return _pca_2d(embeddings)
    else:
        return _pca_2d(embeddings)


def _pca_2d(embeddings: np.ndarray) -> np.ndarray:
    """简单 PCA 降到 2D (纯 numpy)"""
    centered = embeddings - embeddings.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ Vt[:2].T


# ============================================================
# 训练曲线
# ============================================================

def plot_training_curves(
    log_dict: dict,                            # {metric_name: [values]}
    save_path: Optional[str] = None,
) -> None:
    """
    绘制训练曲线 (多损失项)

    自动分组:
      - loss 类: loss, loss_jepa, loss_topo, loss_detect
      - topo 分项: loss_fidelity, loss_preserve, loss_collapse
      - 权重/lr: topo_weight, lr
      - 评估: val_loss, val_retrieval_acc
    """
    if not _ensure_mpl():
        return

    groups = {
        "Total Loss": ["loss", "val_loss"],
        "JEPA Loss": ["loss_jepa", "val_loss_jepa"],
        "Topo Loss": ["loss_topo", "loss_fidelity", "loss_preserve", "loss_collapse"],
        "Detect Loss": ["loss_detect", "loss_box", "loss_cls", "loss_dfl"],
        "Weights & LR": ["topo_weight", "lr", "loss_ratio_ema"],
        "Accuracy": ["val_retrieval_acc"],
    }

    # 过滤有数据的组
    active_groups = {}
    for name, keys in groups.items():
        active_keys = [k for k in keys if k in log_dict and len(log_dict[k]) > 0]
        if active_keys:
            active_groups[name] = active_keys

    if not active_groups:
        logger.warning("No data to plot")
        return

    n = len(active_groups)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#607D8B"]

    for idx, (group_name, keys) in enumerate(active_groups.items()):
        ax = axes[idx]
        for i, k in enumerate(keys):
            vals = log_dict[k]
            c = colors[i % len(colors)]
            ax.plot(vals, color=c, linewidth=1.5, alpha=0.8, label=k)
        ax.set_title(group_name, fontsize=11)
        ax.legend(fontsize=7, framealpha=0.9)
        ax.grid(True, alpha=0.2)

    # 隐藏多余子图
    for idx in range(len(active_groups), len(axes)):
        axes[idx].set_visible(False)

    fig.tight_layout()
    _save_or_show(fig, save_path)


# ============================================================
# 训练前后拓扑对比
# ============================================================

def plot_topo_comparison(
    diagrams_before: List,
    diagrams_after: List,
    title: str = "Topological Change During Training",
    save_path: Optional[str] = None,
) -> None:
    """
    对比训练前后的拓扑变化

    论文消融实验可视化: 有/无 L_topo 时嵌入空间拓扑差异
    """
    if not _ensure_mpl():
        return

    n_dims = max(len(diagrams_before), len(diagrams_after))
    fig, axes = plt.subplots(1, n_dims, figsize=(6 * n_dims, 5))
    if n_dims == 1:
        axes = [axes]

    for dim in range(n_dims):
        ax = axes[dim]

        # Before
        if dim < len(diagrams_before):
            b, d = diagrams_before[dim]
            b = np.asarray(b) if not isinstance(b, np.ndarray) else b
            d = np.asarray(d) if not isinstance(d, np.ndarray) else d
            pers = np.abs(d - b)
            mask = pers > 1e-7
            if mask.any():
                ax.scatter(b[mask], d[mask], c="#90CAF9", s=30, alpha=0.6,
                           edgecolors="#1565C0", linewidth=0.5, marker="o",
                           label="Before")

        # After
        if dim < len(diagrams_after):
            b, d = diagrams_after[dim]
            b = np.asarray(b) if not isinstance(b, np.ndarray) else b
            d = np.asarray(d) if not isinstance(d, np.ndarray) else d
            pers = np.abs(d - b)
            mask = pers > 1e-7
            if mask.any():
                ax.scatter(b[mask], d[mask], c="#FFAB91", s=30, alpha=0.6,
                           edgecolors="#BF360C", linewidth=0.5, marker="^",
                           label="After")

        # 对角线
        all_vals = []
        for diags in [diagrams_before, diagrams_after]:
            if dim < len(diags):
                b, d = diags[dim]
                vals = np.concatenate([np.asarray(b), np.asarray(d)])
                all_vals.append(vals)
        if all_vals:
            max_val = max(np.max(v) for v in all_vals if len(v) > 0)
        else:
            max_val = 1.0
        ax.plot([0, max_val * 1.1], [0, max_val * 1.1], "k--", alpha=0.3)

        ax.set_xlabel("Birth", fontsize=11)
        ax.set_ylabel("Death", fontsize=11)
        ax.set_title(f"H{dim}", fontsize=12)
        ax.legend(framealpha=0.9)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    _save_or_show(fig, save_path)
