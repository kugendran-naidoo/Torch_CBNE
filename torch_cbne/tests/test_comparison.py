from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from torch_cbne.lib.cbne import estimate
from torch_cbne.lib.config import RuntimeConfig
from torch_cbne.lib.graph_loader import load_graphml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CPP_ROOT = (PROJECT_ROOT.parent / "quantinuum" / "CBNE").resolve()
DEFAULT_CPP_BIN = CPP_ROOT / "build" / "cbne"
GRAPH_PATH = (CPP_ROOT / "graphs" / "Graph-1.graphml").resolve()


def require_cpp_binary() -> Path:
    binary_path = Path(os.environ.get("CBNE_CPP_BIN", DEFAULT_CPP_BIN))
    if not binary_path.exists():
        pytest.skip(
            "Skipping comparison tests because the original CBNE binary is not available. "
            "Set CBNE_CPP_BIN to point at the built executable."
        )
    return binary_path


@pytest.mark.integration
def test_python_matches_cpp(tmp_path: Path) -> None:
    binary_path = require_cpp_binary()

    config = RuntimeConfig(
        epsilon=0.1,
        iter_limit=4000,
        deg_limit=3,
        cbne_version="cbne",
        device="cpu",
        seed=42,
    )

    adjacency, spectral_gap, dimension, one_norm, _ = load_graphml(GRAPH_PATH, torch.device("cpu"))
    value_py, stats_py = estimate(adjacency, spectral_gap, dimension, one_norm, config)

    assert value_py is not None

    cpp_cmd = [
        os.fspath(binary_path),
        "-p",
        os.fspath(GRAPH_PATH),
        "-e",
        str(config.epsilon),
        "-i",
        str(config.iter_limit),
        "-d",
        str(config.deg_limit),
        "-a",
        config.cbne_version,
    ]

    result = subprocess.run(cpp_cmd, capture_output=True, text=True, check=True)
    match = re.search(r"Betti estimate:\s*([0-9eE.+-]+)", result.stdout)
    assert match, f"Failed to parse C++ output: {result.stdout}"
    value_cpp = float(match.group(1))

    difference = abs(value_py - value_cpp)
    assert difference <= 0.05, f"Estimates differ by {difference}, python={value_py}, cpp={value_cpp}"


@pytest.mark.performance
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device not available")
def test_gpu_is_faster_than_cpu() -> None:
    config_cpu = RuntimeConfig(
        epsilon=0.1,
        iter_limit=10000,
        deg_limit=3,
        cbne_version="cbne",
        device="cpu",
        seed=123,
    )
    config_gpu = RuntimeConfig(
        epsilon=0.1,
        iter_limit=config_cpu.iter_limit,
        deg_limit=config_cpu.deg_limit,
        cbne_version=config_cpu.cbne_version,
        device="cuda",
        seed=123,
    )

    adjacency_cpu, spectral_gap, dimension, one_norm, _ = load_graphml(GRAPH_PATH, torch.device("cpu"))

    # Warm up GPU
    estimate(adjacency_cpu.clone(), spectral_gap, dimension, one_norm, config_gpu)

    start_cpu = time.perf_counter()
    value_cpu, _ = estimate(adjacency_cpu.clone(), spectral_gap, dimension, one_norm, config_cpu)
    cpu_time = time.perf_counter() - start_cpu

    adjacency_gpu = adjacency_cpu.clone().to(torch.device("cuda"))
    torch.cuda.synchronize()
    start_gpu = time.perf_counter()
    value_gpu, _ = estimate(adjacency_gpu, spectral_gap, dimension, one_norm, config_gpu)
    torch.cuda.synchronize()
    gpu_time = time.perf_counter() - start_gpu

    assert value_cpu is not None and value_gpu is not None
    assert abs(value_cpu - value_gpu) <= 0.02

    improvement = cpu_time / gpu_time if gpu_time > 0 else float("inf")
    assert improvement >= 1.05, f"Expected at least 5% speedup, got ratio {improvement}"
