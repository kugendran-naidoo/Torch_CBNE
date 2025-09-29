#!/usr/bin/env zsh
set -euo pipefail

base_dir="$(cd "$(dirname "$0")" && pwd)"
project_root="${base_dir}/../.."
repo_root="${project_root}/.."
venv="${project_root}/venv_tcbne/bin/activate"

if [[ -f "${venv}" ]]; then
  source "${venv}"
else
  printf "Warning: virtualenv not found at %s\n" "${venv}"
fi

config_file="${base_dir}/calibrate_graphs_config.json"
if [[ ! -f "${config_file}" ]]; then
  printf "Config file not found: %s\n" "${config_file}" >&2
  exit 1
fi

export base_dir repo_root project_root config_file

python <<'PY'
import json
import os
import re
import subprocess
import sys
from pathlib import Path

config_path = Path(os.environ["config_file"])
base_dir = Path(os.environ["base_dir"]).resolve()
repo_root = Path(os.environ["repo_root"]).resolve()

with config_path.open("r", encoding="utf-8") as handle:
    config = json.load(handle)

logs_root = base_dir / "logs" / "calibration"
logs_root.mkdir(parents=True, exist_ok=True)

for graph, params in config.items():
    graph_path = params.get("path") or f"sample_graphs/quantinuum/{graph}.graphml"
    graph_file = (repo_root / graph_path).resolve()
    if not graph_file.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_file}")

    text = graph_file.read_text(encoding="utf-8")
    match = re.search(r"<data key='v_betti'>([^<]+)</data>", text)
    if not match:
        raise RuntimeError(f"Unable to extract ground truth for {graph_file}")
    target_val = float(match.group(1))

    log_subdir = params.get("log_subdir", graph)
    graph_log_dir = logs_root / log_subdir
    graph_log_dir.mkdir(parents=True, exist_ok=True)

    summary_name = params.get("summary_name", f"{graph}_summary.json")
    summary_path = logs_root / summary_name

    epsilon = [str(v) for v in params.get("epsilon", [0.1])]
    deg_limit = [str(v) for v in params.get("deg_limit", [-1])]
    seeds = [str(v) for v in params.get("seeds", [123])]

    iter_start = str(params.get("iter_start", 3000))
    iter_max = str(params.get("iter_max", 20000))
    iter_factor = str(params.get("iter_factor", 1.5))
    tolerance = str(params.get("tolerance", 5e-4))
    min_improvement = str(params.get("min_improvement", 2.5e-4))
    max_stalled = str(params.get("max_stalled", 2))
    device = params.get("device", "cpu")

    print()
    print(f"=== Calibrating {graph} ===")
    print(f"Graph file   : {graph_file}")
    print(f"Ground truth : {target_val}")
    print()

    cmd = [
        sys.executable,
        "calibrate_cbne.py",
        "--path",
        str(graph_file),
        "--target",
        f"{target_val}",
        "--epsilon",
        *epsilon,
        "--deg-limit",
        *deg_limit,
        "--iter-start",
        iter_start,
        "--iter-max",
        iter_max,
        "--iter-factor",
        iter_factor,
        "--seeds",
        *seeds,
        "--tolerance",
        tolerance,
        "--min-improvement",
        min_improvement,
        "--max-stalled",
        max_stalled,
        "--device",
        device,
        "--summary-json",
        str(summary_path),
        "--log-dir",
        str(graph_log_dir),
    ]

    subprocess.run(cmd, check=True, cwd=str(base_dir))

    print()
    print(f"Calibration complete. Summary: {summary_path}")
    print(f"Ground truth : {target_val}\n")
PY
