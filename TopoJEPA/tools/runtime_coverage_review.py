#!/usr/bin/env python3
"""Runtime coverage review for TopoJEPA topology branch.

This script is a lightweight runtime audit. It focuses on:
- Numerical stability (distance matrix + gradients)
- Rips H1 semantics on canonical geometric examples
- Randomized sanity checks (birth <= death, no NaN gradients)
- Cubical fallback behavior
- TopologicalBranch end-to-end forward contract
- Optional backend coverage for giotto-tda/gudhi when available

Usage:
  python TopoJEPA/tools/runtime_coverage_review.py
  python TopoJEPA/tools/runtime_coverage_review.py --random-runs 80 --strict-backends
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Sequence

import torch

# Ensure imports work no matter where script is launched from.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from TopoJEPA.models.encoder import TextEncoder, VisualEncoder
from TopoJEPA.models.predictor import EmbeddingPredictor
from TopoJEPA.models.topo_branch import DifferentiablePH, TopologicalBranch
from TopoJEPA.models.topojepa import TopoJEPA


class SkipCase(RuntimeError):
    """Raised by a test case that should be skipped."""


@dataclass
class CaseResult:
    name: str
    status: str  # PASS | FAIL | ERROR | SKIP
    duration_s: float
    detail: str = ""


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _assert_all_finite(tensors: Sequence[torch.Tensor], msg: str) -> None:
    for tensor in tensors:
        if not torch.isfinite(tensor).all().item():
            raise AssertionError(msg)


def case_distance_matrix_stability(device: torch.device) -> None:
    ph = DifferentiablePH(max_homology_dim=1, filtration_type="rips")

    points = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )

    dist = ph.compute_distance_matrix(points)
    if dist.shape != (3, 3):
        raise AssertionError(f"Unexpected distance shape: {dist.shape}")

    if not torch.allclose(dist.diag(), torch.zeros(3, device=device), atol=1e-12):
        raise AssertionError("Distance matrix diagonal must be exactly 0")

    # Duplicate points should be near epsilon, not NaN.
    if not (0.0 <= dist[0, 1].item() <= 1e-5):
        raise AssertionError(f"Unexpected duplicate-point distance: {dist[0, 1].item():.6e}")

    loss = dist.sum()
    loss.backward()

    if points.grad is None:
        raise AssertionError("Gradient is missing")
    _assert_all_finite((dist, points.grad), "Distance/gradient contains non-finite values")


def _h1_persistences(ph: DifferentiablePH, points: torch.Tensor) -> torch.Tensor:
    dist = ph.compute_distance_matrix(points)
    diagrams = ph.soft_rips_filtration(dist)
    h1_births, h1_deaths = diagrams[1]
    return h1_deaths - h1_births


def case_rips_geometry_examples(device: torch.device) -> None:
    ph = DifferentiablePH(max_homology_dim=1, filtration_type="rips")

    # Equilateral triangle: H1 persistence should be 0.
    eq = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.5, math.sqrt(3.0) / 2.0]],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    p_eq = _h1_persistences(ph, eq)
    if p_eq.max().item() > 1e-5:
        raise AssertionError(f"Equilateral triangle H1 persistence should be 0, got {p_eq.tolist()}")

    # Right triangle: same expectation.
    rt = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    p_rt = _h1_persistences(ph, rt)
    if p_rt.max().item() > 1e-5:
        raise AssertionError(f"Right triangle H1 persistence should be 0, got {p_rt.tolist()}")

    # Unit square: persistence should be around sqrt(2)-1 ~= 0.4142.
    sq = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    p_sq = _h1_persistences(ph, sq)
    max_p = p_sq.max().item()
    target = math.sqrt(2.0) - 1.0
    if abs(max_p - target) > 0.08:
        raise AssertionError(f"Square H1 persistence expected near {target:.4f}, got {max_p:.4f}")


def case_rips_random_sanity(device: torch.device, random_runs: int) -> None:
    ph = DifferentiablePH(max_homology_dim=1, filtration_type="rips")

    for n in (3, 4, 5, 8, 12):
        for _ in range(random_runs):
            points = torch.randn(n, 3, dtype=torch.float32, device=device, requires_grad=True)
            dist = ph.compute_distance_matrix(points)
            diagrams = ph.soft_rips_filtration(dist)

            loss = dist.sum()
            for births, deaths in diagrams:
                if births.shape != deaths.shape:
                    raise AssertionError("Birth/death shape mismatch")
                _assert_all_finite((births, deaths), "Diagram contains non-finite values")

                persistence = deaths - births
                if (persistence < -1e-6).any().item():
                    raise AssertionError("Found birth > death")

                loss = loss + persistence.clamp(min=0.0).sum()

            loss.backward()
            if points.grad is None:
                raise AssertionError("Gradient is missing in random sanity check")
            _assert_all_finite((points.grad,), "Random sanity gradient contains non-finite values")


def case_cubical_pytorch_fallback(device: torch.device) -> None:
    ph = DifferentiablePH(max_homology_dim=1, filtration_type="cubical")
    # Force fallback path coverage even if gudhi exists.
    ph._has_gudhi = False

    field = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )

    diagrams = ph.cubical_filtration(field)
    if len(diagrams) != 2:
        raise AssertionError(f"Expected 2 homology dimensions, got {len(diagrams)}")

    loss = torch.tensor(0.0, device=device)
    for births, deaths in diagrams:
        _assert_all_finite((births, deaths), "Cubical fallback diagram contains non-finite values")
        loss = loss + (deaths - births).clamp(min=0.0).sum()

    loss.backward()
    if field.grad is None:
        raise AssertionError("Cubical fallback gradient is missing")
    _assert_all_finite((field.grad,), "Cubical fallback gradient contains non-finite values")


def case_topobranch_end_to_end(device: torch.device) -> None:
    branch = TopologicalBranch(embed_dim=64, max_homology_dim=1, max_points=32)

    pred_embeddings = torch.randn(8, 64, device=device)
    target_embeddings = torch.randn(8, 64, device=device)
    feature_maps = [torch.randn(8, 16, 32, 32, device=device)]

    topo_info = branch(pred_embeddings, target_embeddings, feature_maps)

    expected_keys = {
        "pred_diagrams",
        "target_diagrams",
        "feature_diagrams",
        "pred_betti",
        "target_betti",
    }
    if set(topo_info.keys()) != expected_keys:
        raise AssertionError(f"Unexpected TopoInfo keys: {sorted(topo_info.keys())}")

    if tuple(topo_info["pred_betti"].shape) != (2,):
        raise AssertionError(f"Unexpected pred_betti shape: {topo_info['pred_betti'].shape}")


def case_model_forward_integration(device: torch.device) -> None:
    model = TopoJEPA(
        visual_encoder=VisualEncoder(
            backbone_type="vit",
            img_size=224,
            patch_size=16,
            embed_dim=64,
            freeze=False,
        ),
        text_encoder=TextEncoder(embed_dim=64, max_length=32, freeze_base=True),
        predictor=EmbeddingPredictor(
            visual_dim=64,
            predictor_dim=32,
            output_dim=64,
            depth=2,
            num_heads=4,
            max_query_tokens=16,
        ),
        topo_branch=TopologicalBranch(embed_dim=64, max_homology_dim=1, max_points=64),
        embed_dim=64,
    ).to(device)

    images = torch.randn(3, 3, 224, 224, device=device)
    # 标准 JEPA 签名: model(images, query_texts=..., target_encoder=...)
    # Stage 1 (query-free): query_texts=None 或不传
    out = model(images)

    if tuple(out["S_Y_hat"].shape) != (3, 64):
        raise AssertionError(f"Unexpected S_Y_hat shape: {out['S_Y_hat'].shape}")
    if tuple(out["S_Y"].shape) != (3, 64):
        raise AssertionError(f"Unexpected S_Y shape: {out['S_Y'].shape}")


def case_giotto_backend_runtime(device: torch.device) -> None:
    ph = DifferentiablePH(max_homology_dim=1, filtration_type="rips")
    if not ph._has_giotto:
        raise SkipCase("gtda not installed")

    points = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.5, 0.8], [0.2, 0.4]],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )

    diagrams = ph(points)
    h0_births, h0_deaths = diagrams[0]
    if not torch.allclose(h0_births, torch.zeros_like(h0_births), atol=1e-8):
        raise AssertionError("giotto H0 births should be exactly 0")

    loss_terms: List[torch.Tensor] = [(h0_deaths - h0_births).clamp(min=0.0).sum()]
    if len(diagrams) > 1:
        h1_births, h1_deaths = diagrams[1]
        loss_terms.append((h1_deaths - h1_births).clamp(min=0.0).sum())

    loss = torch.stack(loss_terms).sum()
    loss.backward()
    if points.grad is None:
        raise AssertionError("giotto backend gradient is missing")
    _assert_all_finite((points.grad,), "giotto backend gradient contains non-finite values")


def case_gudhi_backend_runtime(device: torch.device) -> None:
    ph = DifferentiablePH(max_homology_dim=1, filtration_type="cubical")
    if not ph._has_gudhi:
        raise SkipCase("gudhi not installed")

    field = torch.randn(8, 8, device=device, requires_grad=True)
    diagrams = ph(field)
    if len(diagrams) != 2:
        raise AssertionError(f"Expected 2 homology dimensions, got {len(diagrams)}")

    # Use differentiable terms only.
    terms: List[torch.Tensor] = []
    for births, deaths in diagrams:
        _assert_all_finite((births, deaths), "gudhi backend diagram contains non-finite values")
        if births.requires_grad or deaths.requires_grad:
            terms.append((deaths - births).abs().sum())

    if not terms:
        raise SkipCase("gudhi backend returned no differentiable pairs")

    loss = torch.stack(terms).sum()
    loss.backward()
    if field.grad is None:
        raise AssertionError("gudhi backend gradient is missing")
    _assert_all_finite((field.grad,), "gudhi backend gradient contains non-finite values")


def _run_case(name: str, fn: Callable[[], None]) -> CaseResult:
    start = time.perf_counter()
    try:
        fn()
        return CaseResult(name=name, status="PASS", duration_s=time.perf_counter() - start)
    except SkipCase as exc:
        return CaseResult(name=name, status="SKIP", duration_s=time.perf_counter() - start, detail=str(exc))
    except AssertionError as exc:
        return CaseResult(name=name, status="FAIL", duration_s=time.perf_counter() - start, detail=str(exc))
    except Exception:
        return CaseResult(
            name=name,
            status="ERROR",
            duration_s=time.perf_counter() - start,
            detail=traceback.format_exc(limit=3),
        )


def _print_report(results: List[CaseResult]) -> None:
    print("=" * 72)
    print("TopoJEPA Runtime Coverage Review")
    print("=" * 72)

    for result in results:
        tail = f" | {result.detail}" if result.detail else ""
        print(f"[{result.status:<5}] {result.name:<34} {result.duration_s:>7.3f}s{tail}")

    total = len(results)
    passed = sum(r.status == "PASS" for r in results)
    failed = sum(r.status == "FAIL" for r in results)
    errored = sum(r.status == "ERROR" for r in results)
    skipped = sum(r.status == "SKIP" for r in results)

    print("-" * 72)
    print(
        "Summary: "
        f"total={total}, pass={passed}, fail={failed}, error={errored}, skip={skipped}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Runtime coverage review for TopoJEPA topology branch")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Execution device")
    parser.add_argument("--random-runs", type=int, default=30, help="Random iterations per point count")
    parser.add_argument(
        "--strict-backends",
        action="store_true",
        help="Treat missing giotto/gudhi backend coverage as failure",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("Requested --device cuda but CUDA is unavailable.")
        return 2

    _set_seed(args.seed)
    device = torch.device(args.device)

    cases: List[tuple[str, Callable[[], None]]] = [
        ("distance_matrix_stability", lambda: case_distance_matrix_stability(device)),
        ("rips_geometry_examples", lambda: case_rips_geometry_examples(device)),
        ("rips_random_sanity", lambda: case_rips_random_sanity(device, args.random_runs)),
        ("cubical_pytorch_fallback", lambda: case_cubical_pytorch_fallback(device)),
        ("topobranch_end_to_end", lambda: case_topobranch_end_to_end(device)),
        ("model_forward_integration", lambda: case_model_forward_integration(device)),
        ("giotto_backend_runtime", lambda: case_giotto_backend_runtime(device)),
        ("gudhi_backend_runtime", lambda: case_gudhi_backend_runtime(device)),
    ]

    results = [_run_case(name, fn) for name, fn in cases]
    _print_report(results)

    fail_like = {"FAIL", "ERROR"}
    has_failures = any(r.status in fail_like for r in results)

    if args.strict_backends:
        has_missing_backend = any(
            r.status == "SKIP" and ("gtda" in r.detail or "gudhi" in r.detail)
            for r in results
        )
        if has_missing_backend:
            print("Strict backend mode: missing giotto/gudhi coverage is treated as failure.")
            has_failures = True

    return 1 if has_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
