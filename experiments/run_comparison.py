"""
Automated multi-run comparison experiment.

Runs PSO, Random Search, and SHC on a configurable list of images and saves
results to results/results.csv

Usage:
    python experiments/run_comparison.py
    python experiments/run_comparison.py --runs 5 --gens 50 --pop 50
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import torchvision
from torchvision.models import ResNet50_Weights

# Allow running as a script from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PSOConfig
from src.pso import run as pso_run
from src.baselines import random_search, stochastic_hillclimber
from src.utils import normalize_tensor, preprocess_image

# Default test image set consisting of publicly accessible ImageNet samples
DEFAULT_IMAGES = [
    "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n02033041_dowitcher.JPEG?raw=true",
    "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n01773549_barn_spider.JPEG?raw=true",
    "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n01740131_night_snake.JPEG?raw=true",
    "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n02099267_flat-coated_retriever.JPEG?raw=true",
    "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n02356798_fox_squirrel.JPEG?raw=true",
]

RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "results.csv"
CSV_FIELDNAMES = [
    "image_idx", "image_url", "method", "run", "success", "l2", "timestamp_s"
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run multi-method adversarial patch comparison.")
    p.add_argument("--runs", type=int, default=5, help="Trials per image per method.")
    p.add_argument("--gens", type=int, default=50, help="PSO generations.")
    p.add_argument("--pop", type=int, default=50, help="PSO population size.")
    p.add_argument("--patch-size", type=int, default=40)
    p.add_argument("--num-circles", type=int, default=100)
    p.add_argument("--no-gpu", action="store_true", help="Force CPU even if GPU is available.")
    return p.parse_args()


def load_model(device: torch.device):
    weights = ResNet50_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet50(weights=weights).eval().to(device)
    labels = weights.meta["categories"]
    return model, labels, weights


def main() -> None:
    args = parse_args()
    device = torch.device("cpu" if args.no_gpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_PATH.exists()

    model, labels, weights = load_model(device)
    total_queries = args.pop * args.gens

    with RESULTS_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()

        for img_idx, img_url in enumerate(DEFAULT_IMAGES):
            print(f"\n=== Image {img_idx + 1}/{len(DEFAULT_IMAGES)} ===")
            print(f"    {img_url}")

            for run_n in range(1, args.runs + 1):
                print(f"\n  Run {run_n}/{args.runs}")

                # PSO
                cfg = PSOConfig(
                    image_source=img_url,
                    pop_size=args.pop,
                    generations=args.gens,
                    patch_size=args.patch_size,
                    num_circles=args.num_circles,
                    model_factory=torchvision.models.resnet50,
                    model_weights=weights,
                    device=device,
                )
                t0 = time.time()
                pso_success, pso_l2, _, _ = pso_run(cfg, plot=False)
                writer.writerow({
                    "image_idx": img_idx, "image_url": img_url,
                    "method": "PSO", "run": run_n,
                    "success": pso_success, "l2": pso_l2,
                    "timestamp_s": round(time.time() - t0, 2),
                })
                f.flush()

                # Random Search
                t0 = time.time()
                rs_success, rs_l2, _, _ = random_search(
                    img_url, model, labels,
                    patch_size=args.patch_size,
                    num_circles=args.num_circles,
                    total_queries=total_queries,
                    device=device,
                )
                writer.writerow({
                    "image_idx": img_idx, "image_url": img_url,
                    "method": "RS", "run": run_n,
                    "success": rs_success, "l2": rs_l2,
                    "timestamp_s": round(time.time() - t0, 2),
                })
                f.flush()

                # SHC
                t0 = time.time()
                shc_success, shc_l2, _, _ = stochastic_hillclimber(
                    img_url, model, labels,
                    patch_size=args.patch_size,
                    num_circles=args.num_circles,
                    total_queries=total_queries,
                    device=device,
                )
                writer.writerow({
                    "image_idx": img_idx, "image_url": img_url,
                    "method": "SHC", "run": run_n,
                    "success": shc_success, "l2": shc_l2,
                    "timestamp_s": round(time.time() - t0, 2),
                })
                f.flush()

    print(f"\nResults saved to {RESULTS_PATH}")
    _print_summary(RESULTS_PATH)


def _print_summary(path: Path) -> None:
    """Quick summary table printed to stdout after the experiment."""
    import csv
    from collections import defaultdict
    import math

    rows = list(csv.DictReader(path.open()))
    by_method: dict[str, list] = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)

    print("\nSummary")
    header = f"{'Method':<10} {'ASR':>6} {'Mean L2':>10} {'Median L2':>10}"
    print(header)
    print("-" * len(header))

    for method, method_rows in sorted(by_method.items()):
        successes = [r for r in method_rows if r["success"] == "True"]
        asr = len(successes) / len(method_rows) if method_rows else 0.0
        l2_vals = [float(r["l2"]) for r in successes if r["l2"] != "inf"]
        mean_l2 = sum(l2_vals) / len(l2_vals) if l2_vals else math.inf
        sorted_l2 = sorted(l2_vals)
        n = len(sorted_l2)
        median_l2 = (sorted_l2[n // 2] if n % 2 else (sorted_l2[n // 2 - 1] + sorted_l2[n // 2]) / 2) if n else math.inf
        print(f"{method:<10} {asr:>6.1%} {mean_l2:>10.4f} {median_l2:>10.4f}")


if __name__ == "__main__":
    main()
