"""
TopoJEPA 评估脚本
=================
加载 Stage 1 或 Stage 2 的 checkpoint，在 val/test 上跑一次验证，输出全部指标并可选保存 JSON/TXT。

指标包括:
  - Loss, JEPA loss, Topo loss
  - Acc(pos), Acc(strict)
  - TextDup%
  - Stage 2 检测: Precision, Recall, F1, mAP50, IoU
  - Stage 2 分割: Dice, IoU, PixelAcc

用法:
  cd TopoJEPA && python eval_topojepa.py --checkpoint runs/topojepa_stage2/best.pt --data_root ../RDD_SPLIT
  或
  python -m TopoJEPA.eval_topojepa --checkpoint runs/topojepa_stage2/best.pt --data_root RDD_SPLIT --split test --output results/eval
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

# 与 train.py 一致：保证从 TopoJEPA 目录可导入
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from train import (
    _load_config,
    build_model,
    build_criterion,
    build_optimizer,
    load_checkpoint,
    evaluate,
)
from data import build_dataloader
from utils import ExponentialMovingAverage

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("topojepa.eval")


def parse_args():
    parser = argparse.ArgumentParser(
        description="TopoJEPA 评估：加载 checkpoint 在 val/test 上计算全部指标"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint 路径 (例如 runs/topojepa_stage2/best.pt)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/topojepa_rdd.yaml",
        help="配置文件路径 (默认 configs/topojepa_rdd.yaml)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="",
        help="数据集根目录 (默认从 config data.root 读取)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="评估集划分 (默认 val)",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=None,
        choices=[1, 2],
        help="阶段 1 或 2；不指定则根据 checkpoint 自动推断",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch 大小 (默认从 config 读取)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="设备 (默认 cuda)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="结果保存目录；指定则写入 evaluation_results.json 与 evaluation_report.txt",
    )
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="禁用 AMP",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="DataLoader 进程数 (默认从 config 读取)",
    )
    return parser.parse_args()


def _resolve_path(path_str: str, base: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (base / path_str).resolve()


def _infer_stage_from_checkpoint(checkpoint_path: str) -> int:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = ckpt.get("model_state_dict", {})
    has_det = any(k.startswith("detection_head.") for k in model_state.keys())
    return 2 if has_det else 1  # stage 12 checkpoints 等价于 stage 2 结构


def _print_metrics(metrics: dict) -> None:
    print("\n" + "=" * 60)
    print("TopoJEPA 评估结果")
    print("=" * 60)

    # 通用
    loss = metrics.get("val_loss", 0.0)
    acc_pos = metrics.get("val_retrieval_acc", 0.0)
    acc_strict = metrics.get("val_retrieval_acc_strict", 0.0)
    text_dup = 100.0 * metrics.get("val_text_dup_frac", 0.0)
    print(f"\n  Loss:        {loss:.4f}")
    print(f"  Acc(pos):    {acc_pos:.4f}")
    print(f"  Acc(strict): {acc_strict:.4f}")
    print(f"  TextDup:     {text_dup:.1f}%")

    # 损失分量（若有）
    for k in ("val_loss_jepa", "val_loss_topo", "val_loss_detect", "val_loss_seg"):
        if k in metrics:
            print(f"  {k}: {metrics[k]:.4f}")

    # Stage 2 检测
    if "val_det_map50" in metrics:
        print("\n  [检测]")
        print(f"    mAP50:     {metrics.get('val_det_map50', 0):.4f}")
        print(f"    Precision: {metrics.get('val_det_precision', 0):.4f}")
        print(f"    Recall:    {metrics.get('val_det_recall', 0):.4f}")
        print(f"    F1:        {metrics.get('val_det_f1', 0):.4f}")
        print(f"    IoU(0.5):  {metrics.get('val_det_iou', 0):.4f}")

    # Stage 2 分割
    if "val_seg_dice" in metrics:
        print("\n  [分割]")
        print(f"    Dice:          {metrics.get('val_seg_dice', 0):.4f}")
        print(f"    IoU:           {metrics.get('val_seg_iou', 0):.4f}")
        print(f"    Pixel Acc:     {metrics.get('val_seg_pixel_acc', 0):.4f}")

    print("=" * 60 + "\n")


def _save_results(metrics: dict, output_dir: Path, args, checkpoint_path: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 可序列化的指标（去掉 non-serializable）
    out_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}

    results_dict = {
        "evaluation_info": {
            "checkpoint": checkpoint_path,
            "config": args.config,
            "split": args.split,
            "stage": args.stage,
            "timestamp": datetime.now().isoformat(),
        },
        "metrics": out_metrics,
    }

    json_path = output_dir / "evaluation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    logger.info("结果已保存: %s", json_path)

    txt_path = output_dir / "evaluation_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("TopoJEPA Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Split:      {args.split}\n")
        f.write(f"Stage:      {args.stage}\n")
        f.write(f"Time:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("-" * 50 + "\n")
        for k, v in sorted(out_metrics.items()):
            f.write(f"  {k}: {v}\n")
    logger.info("报告已保存: %s", txt_path)


def main():
    args = parse_args()

    # 路径解析：checkpoint 相对当前工作目录，config 相对脚本目录
    checkpoint_path = Path(args.checkpoint) if Path(args.checkpoint).is_absolute() else (Path.cwd() / args.checkpoint)
    if not checkpoint_path.exists():
        logger.error("Checkpoint 不存在: %s", checkpoint_path)
        sys.exit(1)

    config_path = _resolve_path(args.config, _SCRIPT_DIR)
    cfg = _load_config(str(config_path))
    if not cfg:
        logger.warning("配置为空，使用默认参数")

    # 推断 stage
    stage = args.stage
    if stage is None:
        stage = _infer_stage_from_checkpoint(str(checkpoint_path))
        logger.info("从 checkpoint 推断 stage=%d", stage)
    args.stage = stage

    # 设备与 AMP
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA 不可用，使用 CPU")
        device = torch.device("cpu")
        use_amp = False
    else:
        device = torch.device(args.device)
        use_amp = not args.no_amp and device.type == "cuda"

    # 数据与训练配置
    data_cfg = cfg.get("data", {})
    training_cfg = cfg.get("training", {})
    if stage == 12:
        stage_cfg = training_cfg.get("stage12", training_cfg.get("stage2", {}))
    else:
        stage_cfg = training_cfg.get(f"stage{stage}", {})
    data_root = args.data_root or data_cfg.get("root", "RDD_SPLIT")
    img_size = data_cfg.get("img_size", 640)
    batch_size = args.batch_size or stage_cfg.get("batch_size", 16)
    num_workers = args.num_workers if args.num_workers is not None else data_cfg.get("num_workers", 8)
    text_strategy = stage_cfg.get("text_strategy", "class_name")

    # 构建模型、criterion、optimizer、target_encoder
    logger.info("构建模型 (stage=%d)...", stage)
    model = build_model(cfg, stage=stage)
    model = model.to(device)

    target_encoder = ExponentialMovingAverage.build_target(model.visual_encoder)
    target_encoder = target_encoder.to(device)
    target_encoder.eval()

    criterion = build_criterion(cfg, stage=stage)
    criterion = criterion.to(device)

    lr = stage_cfg.get("lr", 3e-4)
    weight_decay = stage_cfg.get("weight_decay", training_cfg.get("weight_decay", 0.04))
    betas = tuple(training_cfg.get("betas", [0.9, 0.999]))
    optimizer = build_optimizer(model, lr=lr, weight_decay=weight_decay, betas=betas)
    # Stage 2 checkpoint 可能含 _query_embed；该参数在模型内延迟初始化，须先创建再加载
    if stage in (2, 12):
        model._ensure_query_embed(device)
    load_checkpoint(model, target_encoder, optimizer, criterion, str(checkpoint_path), device)

    # 数据加载
    logger.info("构建 DataLoader (split=%s, batch_size=%d)...", args.split, batch_size)
    loader = build_dataloader(
        dataset_name=data_cfg.get("dataset", "rdd"),
        data_root=data_root,
        split=args.split,
        batch_size=batch_size,
        img_size=img_size,
        num_workers=num_workers,
        stage=stage,
        text_strategy=text_strategy,
        augment=False,
    )

    # 评估（关闭 AMP 避免 Half/float 混合导致的 RuntimeError）
    logger.info("开始评估...")
    metrics = evaluate(
        model=model,
        dataloader=loader,
        criterion=criterion,
        target_encoder=target_encoder,
        device=device,
        epoch=0,
        global_step=0,
        use_amp=False,
        use_progress=True,
    )

    _print_metrics(metrics)

    if args.output:
        output_dir = _resolve_path(args.output, Path.cwd())
        _save_results(metrics, output_dir, args, str(checkpoint_path))

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
