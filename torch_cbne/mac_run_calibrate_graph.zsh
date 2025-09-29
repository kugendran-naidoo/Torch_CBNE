#!/usr/bin/env zsh
set -euo pipefail

if [[ $# -lt 2 ]]; then
  printf "Usage: %s --graph <graph-path-relative-to-sample_graphs/quantinuum>\n" "${0}" >&2
  exit 1
fi

graph_path=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --graph)
      if [[ $# -lt 2 ]]; then
        printf "Error: --graph requires an argument\n" >&2
        exit 1
      fi
      graph_path="$2"
      shift 2
      ;;
    -h|--help)
      printf "Usage: %s --graph <graph-path-relative-to-sample_graphs/quantinuum>\n" "${0}"
      exit 0
      ;;
    *)
      printf "Unknown argument: %s\n" "$1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${graph_path}" ]]; then
  printf "Error: --graph must be provided\n" >&2
  exit 1
fi

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

export base_dir repo_root project_root config_file graph_path

python <<'PY'
import json
import os
import re
import subprocess
import sys
from pathlib import Path

graph_path = os.environ["graph_path"]
config_path = Path(os.environ["config_file"])
base_dir = Path(os.environ["base_dir"]).resolve()
repo_root = Path(os.environ["repo_root"]).resolve()

config = json.loads(config_path.read_text(encoding="utf-8"))
matched_graph = None
for graph, params in config.items():
    path_entry = params.get("path")
    if not path_entry:
        continue
    if path_entry == graph_path:
        matched_graph = (graph, params)
        break

if matched_graph is None:
    raise SystemExit(
        f"Graph path '{graph_path}' not found in config {config_path}; ensure the '--graph' value matches the 'path' entry"
    )

graph_name, params = matched_graph

graph_path_obj = Path(graph_path)
if not graph_path_obj.is_absolute():
    graph_file = (base_dir / graph_path_obj).resolve()
else:
    graph_file = graph_path_obj

if not graph_file.exists():
    raise FileNotFoundError(f"Graph file not found: {graph_file}")

text = graph_file.read_text(encoding="utf-8")
match = re.search(r"<data key='v_betti'>([^<]+)</data>", text)
if not match:
    raise RuntimeError(f"Unable to extract ground truth for {graph_file}")
target_val = float(match.group(1))

logs_root = base_dir / "logs" / "calibration"
logs_root.mkdir(parents=True, exist_ok=True)

log_subdir = params.get("log_subdir", graph_name)
summary_name = params.get("summary_name", f"{graph_name}_summary.json")

graph_log_dir = logs_root / log_subdir
graph_log_dir.mkdir(parents=True, exist_ok=True)
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
print(f"=== Calibrating {graph_name} ===")
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
