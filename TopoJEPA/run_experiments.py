"""
TopoJEPA 实验总入口
===================
用于复现实验设计中的主结果、消融和效率对比。

支持的实验模式:
  - jepa_only
  - vicreg
  - barlow
  - toploss
  - topogcl
  - topojepa

支持的数据集:
  - rdd
  - drive
  - inria

默认行为:
  1. 读取论文配置文件
  2. 按数据集与 baseline 批量生成实验命令
  3. 写入统一的 manifest.json
  4. 可选直接执行训练/评估

示例:
  python -m TopoJEPA.run_experiments --datasets rdd drive inria --modes topojepa jepa_only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from train import _load_config

logger = logging.getLogger("topojepa.experiments")


@dataclass
class ExperimentSpec:
    dataset: str
    mode: str
    stage: int
    config: str
    data_root: str
    output_dir: str
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    lr: Optional[float] = None
    seed: Optional[int] = None


DEFAULT_MODES = ["topojepa", "jepa_only", "vicreg", "barlow", "toploss", "topogcl"]
DEFAULT_DATASETS = ["rdd", "drive", "inria"]
MODE_TO_STAGE = {
    "jepa_only": 1,
    "vicreg": 1,
    "barlow": 1,
    "toploss": 1,
    "topogcl": 1,
    "topojepa": 1,
}
DATASET_TO_TASK = {
    "rdd": "detection",
    "drive": "segmentation",
    "inria": "segmentation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TopoJEPA unified experiment runner")
    parser.add_argument("--config", type=str, default="configs/topojepa_rdd.yaml")
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--modes", nargs="*", default=DEFAULT_MODES)
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 12])
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="runs/experiments")
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    parser.add_argument("--execute", action="store_true", help="直接执行命令，否则只生成 manifest")
    parser.add_argument("--dry_run", action="store_true", help="只打印命令")
    parser.add_argument("--train_script", type=str, default="train.py")
    parser.add_argument("--eval_after", action="store_true", help="训练后自动评估")
    return parser.parse_args()


def _dataset_root(cfg: Dict, dataset: str, override: str) -> str:
    if override:
        return override
    data_cfg = cfg.get("data", {})
    root = data_cfg.get("root", "")
    if dataset == data_cfg.get("dataset", dataset):
        return root
    return root


def build_specs(cfg: Dict, args: argparse.Namespace) -> List[ExperimentSpec]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    specs: List[ExperimentSpec] = []
    for dataset in args.datasets:
        for mode in args.modes:
            stage = args.stage if args.stage in (1, 2, 12) else MODE_TO_STAGE.get(mode, 1)
            data_root = _dataset_root(cfg, dataset, args.data_root)
            spec_dir = output_dir / dataset / mode
            spec = ExperimentSpec(
                dataset=dataset,
                mode=mode,
                stage=stage,
                config=str(Path(args.config)),
                data_root=data_root,
                output_dir=str(spec_dir),
                seed=args.seeds[0] if args.seeds else None,
            )
            specs.append(spec)
    return specs


def build_command(spec: ExperimentSpec, args: argparse.Namespace) -> List[str]:
    cmd = [sys.executable, str(Path(_SCRIPT_DIR) / args.train_script)]
    cmd += ["--config", spec.config, "--stage", str(spec.stage), "--device", "cuda"]
    cmd += ["--output_dir", spec.output_dir]
    if spec.data_root:
        cmd += ["--data_root", spec.data_root]
    return cmd


def _patch_config_for_mode(spec: ExperimentSpec) -> Dict:
    cfg = _load_config(spec.config)
    cfg.setdefault("experiment", {})
    cfg["experiment"]["dataset"] = spec.dataset
    cfg["experiment"]["mode"] = spec.mode
    cfg["experiment"]["stage"] = spec.stage
    cfg["experiment"]["seed"] = spec.seed
    cfg.setdefault("data", {})
    cfg["data"]["dataset"] = spec.dataset
    cfg["model"] = cfg.get("model", {})
    cfg["model"]["task_type"] = DATASET_TO_TASK.get(spec.dataset, cfg["model"].get("task_type", "detection"))
    return cfg


def _write_manifest(specs: List[ExperimentSpec], out_dir: Path) -> None:
    manifest = [asdict(s) for s in specs]
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    cfg = _load_config(args.config)
    specs = build_specs(cfg, args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(specs, out_dir)

    for spec in specs:
        spec_cfg = _patch_config_for_mode(spec)
        spec_dir = Path(spec.output_dir)
        spec_dir.mkdir(parents=True, exist_ok=True)
        with open(spec_dir / "resolved_config.json", "w", encoding="utf-8") as f:
            json.dump(spec_cfg, f, indent=2, ensure_ascii=False)

        cmd = build_command(spec, args)
        logger.info("Prepared %s/%s -> %s", spec.dataset, spec.mode, " ".join(cmd))
        if args.dry_run:
            continue
        if args.execute:
            subprocess.run(cmd, check=True, cwd=str(_SCRIPT_DIR))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
