from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

from .complex import Complex, sample_markov_chain
from .config import RuntimeConfig
from .graph_loader import load_graphml
from .stats import Statistics


def log_factory(output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = output_path.open("w", encoding="utf-8")

    def _log(message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        log_handle.write(line + "\n")
        log_handle.flush()

    return _log, log_handle


def run_cbne_logged(
    graph_path: Path,
    epsilon: float,
    iter_limit: int,
    deg_limit: int,
    device: torch.device,
    seed: int | None,
    log_path: Path,
) -> tuple[float, float]:
    log, handle = log_factory(log_path)
    adjacency, spectral_gap, dimension, one_norm, _ = load_graphml(graph_path, device)
    config = RuntimeConfig(
        epsilon=epsilon,
        iter_limit=iter_limit,
        deg_limit=deg_limit,
        cbne_version="cbne",
        device=str(device),
        seed=seed,
    )

    stats = Statistics()
    complex_obj = Complex(adjacency, generator=torch.Generator(device="cpu"))
    n = adjacency.size(0)

    gamma = spectral_gap
    epsilon = config.epsilon
    delta = epsilon / 2.0
    z = math.ceil((1.0 / gamma) * math.log(2.0 / epsilon))
    if config.deg_limit != -1:
        z = config.deg_limit

    base = one_norm if one_norm != 0 and config.use_one_norm else 2.0
    numerator = 4.0 * (base ** (2 * z)) * math.log(2.0 / 0.1)
    denominator = 2.0 * (delta ** 2)
    shot_count = numerator / denominator

    if config.iter_limit != -1:
        shot_count = config.iter_limit

    shot_count = max(1, int(math.ceil(shot_count)))
    stats.set_sample_count(shot_count)
    stats.set_walk_length(z)

    log("Starting CBNE base run")
    log(f"Graph: {graph_path}")
    log(f"Spectral gap: {spectral_gap}")
    log(f"Dimension (k): {dimension}")
    log(f"One norm: {one_norm}")
    log(f"Walk length (z): {z}")
    log(f"Planned samples: {shot_count}")

    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)
        log(f"Seed set to {seed}")

    start_time = time.perf_counter()
    total = 0.0
    log_interval = max(1, shot_count // 10)

    with tqdm(
        total=shot_count,
        desc="CBNE iterations",
        unit="sample",
        file=sys.stdout,
        leave=True,
        dynamic_ncols=True,
    ) as progress_bar:
        for idx in range(1, shot_count + 1):
            iter_start = time.perf_counter()

            sample_start = time.perf_counter()
            logger = log if shot_count <= 20 else None
            start_face = complex_obj.sample_from_complex(
                dimension,
                max_attempts=5000,
                logger=logger,
            )
            sample_duration = time.perf_counter() - sample_start

            chain_start = time.perf_counter()
            value = sample_markov_chain(start_face, complex_obj, n, z, stats, generator)
            chain_duration = time.perf_counter() - chain_start

            total += value
            iter_duration = time.perf_counter() - iter_start
            progress_bar.update(1)
            progress_bar.set_postfix(
                {
                    "last_val": f"{value:.4f}",
                    "sample_s": f"{sample_duration:.4f}",
                    "chain_s": f"{chain_duration:.4f}",
                    "iter_s": f"{iter_duration:.4f}",
                }
            )

            if shot_count <= 20 or idx % log_interval == 0 or idx == shot_count:
                progress = (idx / shot_count) * 100.0
                log(
                    f"Iteration {idx}/{shot_count} | progress {progress:.1f}% | "
                    f"value={value:.6f} | sample_time={sample_duration:.4f}s | "
                    f"chain_time={chain_duration:.4f}s | iter_time={iter_duration:.4f}s"
                )

    duration = time.perf_counter() - start_time
    result = total / float(shot_count)
    log(f"Completed run in {duration:.4f} seconds, estimate={result}")

    handle.close()
    return result, duration


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run CBNE base variant with logging")
    parser.add_argument("--path", required=True, type=Path, help="Path to .graphml file")
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--iter_limit", type=int, default=1)
    parser.add_argument("--deg_limit", type=int, default=3)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--log", type=Path, default=Path("logs/cbne_run.log"), help="Log file destination"
    )
    args = parser.parse_args(argv)

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    run_cbne_logged(
        graph_path=args.path,
        epsilon=args.epsilon,
        iter_limit=args.iter_limit,
        deg_limit=args.deg_limit,
        device=device,
        seed=args.seed,
        log_path=args.log,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
