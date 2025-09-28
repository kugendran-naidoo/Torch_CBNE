from __future__ import annotations

from pathlib import Path
from typing import Tuple

import networkx as nx
import torch


def load_graphml(path: str | Path, device: torch.device) -> Tuple[torch.Tensor, float, int, float, float]:
    graph = nx.read_graphml(Path(path))
    nodes = sorted(graph.nodes(), key=lambda node_id: int(node_id))
    n = len(nodes)
    adjacency = torch.zeros((n, n), dtype=torch.bool, device=device)

    index_map = {node_id: idx for idx, node_id in enumerate(nodes)}
    for u, v in graph.edges():
        i = index_map[u]
        j = index_map[v]
        adjacency[i, j] = True
        adjacency[j, i] = True

    props = graph.nodes[nodes[0]]
    spectral_gap = float(props.get("gap", 0.0))
    dimension = int(props.get("dimk", 0))
    one_norm = float(props.get("norm", 0.0))
    betti_est = float(props.get("betti", 0.0))

    return adjacency, spectral_gap, dimension, one_norm, betti_est
