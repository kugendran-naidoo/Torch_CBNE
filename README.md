# Torch CBNE – Practical Guide

Python re-implementation of the Quantinuum CBNE estimator with optional GPU acceleration via PyTorch. Augment to Quantum GNN paper for PhD in progress.

## 📊 Traffic & Popularity
<!--![Clones](https://img.shields.io/endpoint?cacheSeconds=300&url=https%3A%2F%2Fgist.githubusercontent.com%2Fkugendran-naidoo%2F2b0de4f9f92a605b780e986e6d48ffcc%2Fraw%2FTorch_CBNE-clones.json%3Fv%3D1)
![Views](https://img.shields.io/endpoint?cacheSeconds=300&url=https%3A%2F%2Fgist.githubusercontent.com%2Fkugendran-naidoo%2F9b749f24de62343dc995f8d524027c39%2Fraw%2FTorch_CBNE-views.json%3Fv%3D1)-->

## 📊 Traffic & Popularity
![Clones](https://img.shields.io/endpoint?cacheSeconds=300&url=https%3A%2F%2Fgist.githubusercontent.com%2Fkugendran-naidoo%2F2b0de4f9f92a605b780e986e6d48ffcc%2Fraw%2FTorch_CBNE-clones.json%3Fv%3D2)
![Views](https://img.shields.io/endpoint?cacheSeconds=300&url=https%3A%2F%2Fgist.githubusercontent.com%2Fkugendran-naidoo%2F9b749f24de62343dc995f8d524027c39%2Fraw%2FTorch_CBNE-views.json%3Fv%3D2)
![Stars](https://img.shields.io/endpoint?cacheSeconds=300&url=https%3A%2F%2Fgist.githubusercontent.com%2Fkugendran-naidoo%2F2b0de4f9f92a605b780e986e6d48ffcc%2Fraw%2FTorch_CBNE-stars.json%3Fv%3D2)
![Commits](https://img.shields.io/endpoint?cacheSeconds=300&url=https%3A%2F%2Fgist.githubusercontent.com%2Fkugendran-naidoo%2F2b0de4f9f92a605b780e986e6d48ffcc%2Fraw%2FTorch_CBNE-commits.json%3Fv%3D2)


> Auto-updated daily at 14:00 UTC via GitHub Actions.

## 📈 Metrics
![Activity (last 4 weeks)](https://raw.githubusercontent.com/kugendran-naidoo/Torch_CBNE/main/metrics/activity_4w.png)

> Auto-updated daily at 14:00 UTC via GitHub Actions.

---
## 1. Overview

The repository hosts a PyTorch reimplementation of the CBNE estimator.  Core logic lives in the `torch_cbne/lib` package (simplicial complex sampling, configuration handling, statistics, GraphML loader).  On top of that sit a set of runners and calibration helpers that load GraphML graphs, execute `run_cbne_logged.py`, and explore parameter grids until the CBNE estimate matches a known ground truth.

Recent additions include:

- **`calibrate_cbne.py`** – Python entry point that sweeps CBNE parameters, logs progress, and writes summaries.
- **`mac_run_calibrate_graph.zsh`** – Calibrates a single graph chosen at runtime; supports alternate JSON configs.
- **Per-graph wrappers** (e.g. `mac_run_calibrate_graph_1.zsh`) – Convenience scripts that pin both the graph path and config file.
- **Config files** describing the parameter space (`calibrate_graphs_config.json` for the “standard” sweep, `calibrate_graphs_config_strict.json` for a high-accuracy pass).

Sample GraphML files live under `torch_cbne/sample_graphs/quantinuum/`.

---
## 2. Requirements

- **Python** 3.10+ (used for recent runs).
- **Poetry or pip** (repository uses editable installs via pip).
- **PyTorch** >= 2.0.0 (CPU is sufficient, CUDA optional; scripts prefer CUDA but fall back automatically).
- **Graph tooling**: GraphML files are supplied; no external dependencies needed beyond `networkx`.

Before running calibrations, ensure a virtual environment exists (example commands below create one with `python -m venv venv_tcbne`).

---
## 3. Install & Environment Setup

```bash
cd torch_cbne

# optional: create a dedicated venv if not already present
python -m venv venv_tcbne
source venv_tcbne/bin/activate

# install Torch CBNE in editable mode with dependencies
pip install --upgrade pip
pip install -e .  # reads pyproject.toml
```

To re-use the supplied environment created during previous sessions, simply `source venv_tcbne/bin/activate` inside the repository root before running any commands.

---
## 4. Repository Layout

```
torch_cbne/
├── lib/                     # Core CBNE implementation modules
│   ├── complex.py           # Simplicial complex sampler
│   ├── cbne.py              # Estimator wrapper
│   ├── config.py            # Runtime configuration object
│   ├── graph_loader.py      # GraphML loading utility
│   ├── run_cbne_logged.py   # Main runner callable from the CLI
│   └── stats.py             # Sampling statistics container
├── calibrate_cbne.py        # Parameter sweep utility
├── calibrate_graphs_config.json       # Default calibration config
├── calibrate_graphs_config_strict.json# Strict (0.1% error target) config
├── mac_run_calibrate_graph.zsh        # Single-graph runner with config override
├── mac_run_calibrate_graph_*.zsh      # Legacy wrappers (Graph 1–4)
├── sample_graphs/quantinuum/Graph-*.graphml # Example GraphML inputs
└── logs/...
```

Logs are written to `torch_cbne/logs/calibration/<graph-name>/...`.  JSON summaries contain per-configuration averages, standard deviations, and references to log files.

---
## 5. Running Calibrations

### 5.1 Calibrate a Single Graph (Default Config)

```bash
./mac_run_calibrate_graph.zsh --graph sample_graphs/quantinuum/Graph-2.graphml
```

The script looks up the matching entry in `calibrate_graphs_config.json` (matching the `path` field) and launches `calibrate_cbne.py` with the assigned parameters.

### 5.2 Calibrate a Single Graph (Strict Config)

To drive the estimator toward ±0.1 % error bounds, use the stricter configuration:

```bash
./mac_run_calibrate_graph.zsh \
  --graph sample_graphs/quantinuum/Graph-2.graphml \
  --json-config calibrate_graphs_config_strict.json
```

Strict runs use larger iteration budgets, more seeds, and tighter early-stop rules. Expect longer runtimes (potentially hours on CPU for the 200k+ sample sweeps); run on CUDA for the best throughput.

### 5.3 Direct Invocation (No Wrapper)

For custom experiments, call `calibrate_cbne.py` directly:

```bash
python calibrate_cbne.py \
  --path sample_graphs/quantinuum/Graph-1.graphml \
  --target 0.444 \
  --epsilon 0.1 \
  --deg-limit -1 3 \
  --iter-start 3000 \
  --iter-max 20000 \
  --iter-factor 1.5 \
  --seeds 123 \
  --tolerance 5e-4 \
  --min-improvement 2.5e-4 \
  --max-stalled 2 \
  --device cuda \
  --summary-json logs/calibration/Graph-1_summary.json \
  --log-dir logs/calibration/Graph-1
```

The same script can run with alternate parameter ranges to explore convergence manually.

---
## 6. Config File Structure

Each entry in `calibrate_graphs_config*.json` has the following keys:

| Key              | Type        | Meaning                                                                 |
|------------------|-------------|-------------------------------------------------------------------------| 
| `path`           | string      | GraphML path (relative to repo root or absolute); used to locate the graph and match CLI arguments. |
| `epsilon`        | array<float>| List of epsilon values to sweep.                                      |
| `deg_limit`      | array<int>  | Degree caps (−1 lets the runtime choose).                            |
| `iter_start`     | int         | Initial iteration budget (shot count).                                |
| `iter_max`       | int         | Maximum iteration budget.                                            |
| `iter_factor`    | float       | Growth multiplier between sweeps.                                    |
| `seeds`          | array<int>  | Random seeds evaluated per configuration.                            |
| `tolerance`      | float       | Early-stop threshold on mean absolute error.                         |
| `min_improvement`| float       | Required improvement to reset the stall counter.                     |
| `max_stalled`    | int         | Number of consecutive non-improving steps before abandoning a parameter line. |
| `log_subdir`     | string      | Subdirectory under `logs/calibration/` for per-run logs.              |
| `summary_name`   | string      | Filename for the summary JSON.                                       |
| `device`         | string      | Preferred device (`cuda` or `cpu`). CUDA requests fall back to CPU automatically when unavailable. |

Configs may include metadata entries (e.g., `_comment`). Runners ignore non-object entries.

---
## 7. Understanding the Output

For each parameter combination, `calibrate_cbne.py` logs:

- individual iteration values (`run_cbne_logged.py` writes detailed logs);
- overall runtime per configuration;
- mean estimate, mean absolute error, maximum error across seeds;
- links to per-run log files.

The summary JSON sorts entries by mean error so the best configurations appear at the top. For high-accuracy campaigns, examine both mean error and standard deviation to ensure the 0.1 % target is genuinely met.

---
## 8. Working With Core Modules

Advanced users can import the library directly:

```python
from torch_cbne.lib import estimate, RuntimeConfig, load_graphml

adjacency, spectral_gap, k, one_norm, _ = load_graphml(Path("sample_graphs/quantinuum/Graph-1.graphml"), torch.device("cpu"))
config = RuntimeConfig(epsilon=0.1, iter_limit=5000, deg_limit=3, cbne_version="cbne", device="cpu", seed=123)
value, stats = estimate(adjacency, spectral_gap, k, one_norm, config)
```

This mirrors the workflow inside `run_cbne_logged.py` without logging concerns.

---
## 9. Troubleshooting

| Symptom | Likely Cause & Fix |
|---------|-------------------|
| `Graph path '…' not found in config` | Ensure `--graph` matches the `path` string in the JSON (e.g., `sample_graphs/quantinuum/Graph-2.graphml`). |
| `AttributeError: 'str' object has no attribute 'get'` | Config contains metadata entries; update to the latest scripts which ignore non-dict entries. |
| CUDA requested but unavailable | PyTorch cannot see a GPU; the script will fall back to CPU automatically, but runs will be slower. |
| No improvement, sweep stops early | Increase `max_stalled`, extend `iter_max`, add more seeds, or lower `min_improvement`. |
| Estimates drift >0.1 % | Use the strict config (`calibrate_graphs_config_strict.json`) or increase iteration budgets and seeds further. |

---
## 10. Next Steps

- Integrate the calibration summaries into an experiment tracker for long-term history.
- Experiment with Bayesian optimization on the configuration space once enough runs are captured.
- Add automated convergence plots to visualize mean vs. shot count.

For questions or contributions, refer to the existing `README.md` or open discussions within the project.
