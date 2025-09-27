from __future__ import annotations

import argparse
import sys

import torch

from .cbne import estimate
from .config import RuntimeConfig
from .graph_loader import load_graphml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Torch CBNE estimator")
    parser.add_argument("-p", "--path", required=True, help="Path to .graphml input file")
    parser.add_argument("-o", "--out", default=None, help="Optional CSV output path for intermediate values")
    parser.add_argument(
        "-n",
        "--num_data_points",
        type=int,
        default=1,
        help="Number of intermediate points to compute",
    )
    parser.add_argument("-e", "--epsilon", type=float, default=0.1, help="Epsilon value")
    parser.add_argument(
        "-i",
        "--iter_limit",
        type=int,
        default=-1,
        help="Override shot count with explicit iteration budget",
    )
    parser.add_argument(
        "-d",
        "--deg_limit",
        type=int,
        default=-1,
        help="Override polynomial degree / walk length",
    )
    parser.add_argument("-s", "--output_shot_count", action="store_true", help="Print shot count and exit")
    parser.add_argument("-c", "--output_step_count", action="store_true", help="Print step count and exit")
    parser.add_argument(
        "-u",
        "--use_one_norm",
        action="store_true",
        default=True,
        help="Use one norm from graph metadata if available",
    )
    parser.add_argument(
        "--no-use-one-norm",
        dest="use_one_norm",
        action="store_false",
        help="Ignore one norm supplied in graph metadata",
    )
    parser.add_argument(
        "-a",
        "--cbne_version",
        default="cbne",
        choices=["cbne", "cbneCheby", "cbneMusco", "cbneCompressed"],
        help="Algorithm variant to run",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Execution device",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument(
        "--time_limit",
        type=int,
        default=-1,
        help="Optional time limit placeholder (not enforced, kept for parity)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = RuntimeConfig(
        epsilon=args.epsilon,
        iter_limit=args.iter_limit,
        deg_limit=args.deg_limit,
        output_shot_count=args.output_shot_count,
        output_step_count=args.output_step_count,
        use_one_norm=args.use_one_norm,
        num_data_points=args.num_data_points,
        out_path=args.out,
        cbne_version=args.cbne_version,
        device=args.device,
        seed=args.seed,
        time_limit=args.time_limit,
    )

    adjacency, spectral_gap, dimension, one_norm, betti_est = load_graphml(args.path, config.torch_device())
    value, stats = estimate(adjacency, spectral_gap, dimension, one_norm, config)

    if config.should_output_counts():
        return 0

    if value is None:
        print("No estimate produced.")
        return 0

    print(f"\nBetti estimate: {value}\n")
    print(stats.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
