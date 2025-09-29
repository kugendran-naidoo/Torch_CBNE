from __future__ import annotations

import argparse
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import torch

from torch_cbne.lib.run_cbne_logged import run_cbne_logged


@dataclass
class TrialResult:
    epsilon: float
    iter_limit: int
    deg_limit: int
    seed: int
    estimate: float
    error: float
    duration: float
    log_path: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate CBNE parameters against ground truth")
    parser.add_argument("--path", required=True, type=Path, help="GraphML input")
    parser.add_argument("--target", required=True, type=float, help="Reference value to match")
    parser.add_argument("--epsilon", nargs="+", type=float, default=[0.1], help="Epsilon values to explore")
    parser.add_argument(
        "--deg-limit",
        nargs="+",
        type=int,
        default=[-1, 3],
        help="Degree limits to explore (-1 lets runtime pick z)",
    )
    parser.add_argument("--iter-start", type=int, default=2000, help="Starting iteration budget")
    parser.add_argument("--iter-max", type=int, default=20000, help="Maximum iteration budget")
    parser.add_argument("--iter-factor", type=float, default=1.5, help="Multiplier for iteration sweep")
    parser.add_argument("--seeds", nargs="+", type=int, default=[123], help="Random seeds to evaluate")
    parser.add_argument("--tolerance", type=float, default=5e-4, help="Stop once mean error is below this")
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=2.5e-4,
        help="Minimum mean-error improvement required to keep increasing iter_limit",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Execution device preference",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs/calibration"),
        help="Directory for per-run logs",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path to export trial metadata as JSON",
    )
    parser.add_argument(
        "--max-stalled",
        type=int,
        default=2,
        help="Number of consecutive non-improving steps before stopping a configuration",
    )
    return parser.parse_args()


def _iter_limits(start: int, max_value: int, factor: float) -> Sequence[int]:
    if start <= 0:
        raise ValueError("iter-start must be positive")
    if max_value < start:
        raise ValueError("iter-max must be greater than or equal to iter-start")
    values = [start]
    current = start
    while current < max_value:
        next_value = int(math.ceil(current * factor))
        if next_value <= current:
            next_value = current + 1
        if next_value > max_value:
            next_value = max_value
        values.append(next_value)
        current = next_value
    return values


def _format_config_name(graph_stem: str, epsilon: float, deg_limit: int, iter_limit: int, seed: int) -> str:
    eps_part = f"eps{epsilon:.3g}".replace(".", "p")
    return f"{graph_stem}_{eps_part}_deg{deg_limit}_iter{iter_limit}_seed{seed}"


def calibrate(args: argparse.Namespace) -> list[TrialResult]:
    graph_path: Path = args.path
    graph_stem = graph_path.stem
    target = args.target
    seeds: Sequence[int] = args.seeds
    epsilons: Sequence[float] = args.epsilon
    deg_limits: Sequence[int] = args.deg_limit
    tolerance: float = args.tolerance
    min_improvement: float = args.min_improvement
    max_stalled: int = max(1, args.max_stalled)
    log_dir: Path = args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    device_choice = args.device
    if device_choice == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU")
        device_choice = "cpu"
    device = torch.device(device_choice)

    iter_limits = _iter_limits(args.iter_start, args.iter_max, args.iter_factor)
    results: list[TrialResult] = []

    for epsilon in epsilons:
        for deg_limit in deg_limits:
            best_mean_error = math.inf
            stalled_steps = 0
            for iter_limit in iter_limits:
                config_results: list[TrialResult] = []
                for seed in seeds:
                    log_name = _format_config_name(graph_stem, epsilon, deg_limit, iter_limit, seed)
                    log_path = log_dir / f"{log_name}.log"
                    estimate, duration = run_cbne_logged(
                        graph_path=graph_path,
                        epsilon=epsilon,
                        iter_limit=iter_limit,
                        deg_limit=deg_limit,
                        device=device,
                        seed=seed,
                        log_path=log_path,
                    )
                    error = abs(estimate - target)
                    config_results.append(
                        TrialResult(
                            epsilon=epsilon,
                            iter_limit=iter_limit,
                            deg_limit=deg_limit,
                            seed=seed,
                            estimate=estimate,
                            error=error,
                            duration=duration,
                            log_path=str(log_path),
                        )
                    )
                    print(
                        f"eps={epsilon:.3g} deg_limit={deg_limit} iter_limit={iter_limit} seed={seed} "
                        f"estimate={estimate:.6f} error={error:.6f} duration={duration:.3f}s"
                    )
                for trial in config_results:
                    results.append(trial)

                mean_error = statistics.mean(trial.error for trial in config_results)
                if mean_error < best_mean_error - min_improvement:
                    best_mean_error = mean_error
                    stalled_steps = 0
                else:
                    stalled_steps += 1

                print(
                    f"-> mean error {mean_error:.6f} across {len(config_results)} seed(s); "
                    f"best so far {best_mean_error:.6f}"
                )

                if mean_error <= tolerance:
                    print(
                        f"Tolerance reached (<= {tolerance}); stopping iter_limit sweep for deg_limit={deg_limit}"
                    )
                    break
                if stalled_steps >= max_stalled:
                    print(
                        "No significant improvement detected in two consecutive steps; moving to next configuration"
                    )
                    break
    return results


def summarize(results: Sequence[TrialResult]) -> list[dict[str, object]]:
    buckets: dict[tuple[float, int, int], list[TrialResult]] = defaultdict(list)
    for trial in results:
        key = (trial.epsilon, trial.deg_limit, trial.iter_limit)
        buckets[key].append(trial)

    summaries: list[dict[str, object]] = []
    for (epsilon, deg_limit, iter_limit), trials in buckets.items():
        estimates = [trial.estimate for trial in trials]
        errors = [trial.error for trial in trials]
        durations = [trial.duration for trial in trials]
        summary = {
            "epsilon": epsilon,
            "deg_limit": deg_limit,
            "iter_limit": iter_limit,
            "seeds": [trial.seed for trial in trials],
            "mean_estimate": statistics.mean(estimates),
            "stdev_estimate": statistics.pstdev(estimates) if len(estimates) > 1 else 0.0,
            "mean_error": statistics.mean(errors),
            "max_error": max(errors),
            "mean_duration": statistics.mean(durations),
            "logs": [trial.log_path for trial in trials],
        }
        summaries.append(summary)

    summaries.sort(key=lambda entry: entry["mean_error"])
    return summaries


def main() -> int:
    args = _parse_args()
    results = calibrate(args)
    if not results:
        print("No calibration runs executed; check input parameters")
        return 1

    summaries = summarize(results)
    best = summaries[0]

    print("\n=== Calibration Summary (top configurations) ===")
    for entry in summaries[: min(5, len(summaries))]:
        print(
            f"eps={entry['epsilon']:.3g} deg_limit={entry['deg_limit']} iter_limit={entry['iter_limit']} "
            f"mean_error={entry['mean_error']:.6f} stdev={entry['stdev_estimate']:.6f} "
            f"mean_estimate={entry['mean_estimate']:.6f} runtime={entry['mean_duration']:.3f}s"
        )

    print("\nBest configuration:")
    print(best)

    if args.summary_json is not None:
        payload = {
            "target": args.target,
            "results": [asdict(trial) for trial in results],
            "summaries": summaries,
        }
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            data=json_dumps(payload),
            encoding="utf-8",
        )
        print(f"Wrote summary JSON to {args.summary_json}")

    return 0


def json_dumps(payload: dict[str, object]) -> str:
    import json

    return json.dumps(payload, indent=2, sort_keys=True)


if __name__ == "__main__":
    raise SystemExit(main())
