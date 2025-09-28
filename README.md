# Torch CBNE

Python re-implementation of the Quantinuum CBNE estimator with optional GPU acceleration via PyTorch. Augment to Quantum GNN paper for PhD in progress. 

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
python -m torch_cbne.cli \
  --path ../../quantinuum/CBNE/graphs/Graph-1.graphml \
  --cbne_version cbne \
  --device cuda \
  --iter_limit 4000
```

Use `--device cpu` to force a CPU execution.

## Tests

1. Build the original C++ binary (from `../quantinuum/CBNE`):

   ```bash
   mkdir -p ../quantinuum/CBNE/build
   cmake -S ../quantinuum/CBNE -B ../quantinuum/CBNE/build
   cmake --build ../quantinuum/CBNE/build
   ```

2. Export the binary path if it differs from the default location:

   ```bash
   export CBNE_CPP_BIN=/path/to/cbne
   ```

3. Run the test suite:

   ```bash
   pytest
   ```

The `performance` marked test compares GPU and CPU runtime, skipping automatically when CUDA is not available.
