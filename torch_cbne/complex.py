from __future__ import annotations

from typing import List, Tuple

import torch

from .stats import Statistics

Face = torch.Tensor


class Complex:
    """Simplicial complex helper mirroring the C++ implementation using Torch tensors."""

    def __init__(self, adjacency: torch.Tensor, generator: torch.Generator | None = None) -> None:
        if adjacency.dim() != 2 or adjacency.size(0) != adjacency.size(1):
            raise ValueError("Adjacency matrix must be square")
        self.adjacency = adjacency.to(dtype=torch.bool)
        self.n = self.adjacency.size(0)
        self.device = self.adjacency.device
        self.generator = generator or torch.Generator(device="cpu")
        self._diag_zero()

    def _diag_zero(self) -> None:
        idx = torch.arange(self.n, device=self.adjacency.device)
        self.adjacency[idx, idx] = False

    def includes_face(self, face: Face) -> bool:
        if face.numel() <= 1:
            return True
        face = face.long()
        sub = self.adjacency.index_select(0, face).index_select(1, face)
        triu = torch.triu(sub, diagonal=1)
        return bool(triu.all())

    def number_of_up_neighbours(self, face: Face) -> int:
        face = face.long()
        face_mask = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        face_mask[face] = True
        not_face = ~face_mask
        if not torch.any(not_face):
            return 0
        connections = self.adjacency.index_select(1, face)
        connected_to_all = connections.all(dim=1)
        return int(torch.sum(connected_to_all & not_face).item())

    def _compute_sign(self, face: Face, removed_index: int, new_value: int) -> int:
        index2 = int(torch.searchsorted(face, torch.tensor(new_value, device=face.device)))
        if int(face[removed_index].item()) < new_value:
            index2 -= 1
        sign = -1 * ((-1) ** removed_index) * ((-1) ** index2)
        return int(sign)

    def get_neighbours(self, face: Face) -> List[Tuple[Face, int]]:
        face = torch.sort(face.long()).values
        face_mask = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        face_mask[face] = True
        candidate_indices = torch.nonzero(~face_mask, as_tuple=False).view(-1)
        neighbours: List[Tuple[Face, int]] = []
        if candidate_indices.numel() == 0:
            return neighbours

        connections = self.adjacency.index_select(0, candidate_indices).index_select(1, face)
        missing = ~connections
        num_missing = missing.sum(dim=1)
        qualified_mask = num_missing == 1
        if not torch.any(qualified_mask):
            return neighbours
        qualified_candidates = candidate_indices.index_select(0, torch.nonzero(qualified_mask, as_tuple=False).view(-1))
        missing_indices = torch.argmax(missing[qualified_mask].to(torch.int64), dim=1)

        for candidate, missing_idx in zip(qualified_candidates.tolist(), missing_indices.tolist()):
            new_face = face.clone()
            new_face[missing_idx] = candidate
            new_face = torch.sort(new_face).values
            sign = self._compute_sign(face, missing_idx, candidate)
            neighbours.append((new_face, sign))
        return neighbours

    def sample_from_complex(self, k: int) -> Face:
        face_size = k + 1
        gen = self.generator
        while True:
            perm = torch.randperm(self.n, generator=gen, device="cpu").to(self.device)
            face = torch.sort(perm[:face_size]).values
            if self.includes_face(face):
                return face


def sample_markov_chain(
    starting_face: Face,
    complex_obj: Complex,
    n: int,
    z: int,
    stats: Statistics,
    generator: torch.Generator,
) -> float:
    face = starting_face.clone()
    product = 1.0
    device = complex_obj.device

    for _ in range(z):
        up_degree = complex_obj.number_of_up_neighbours(face)
        diagonal = 1.0 - float(up_degree + face.numel()) / float(n)
        neighbours = complex_obj.get_neighbours(face)
        column_norm = float(len(neighbours)) / float(n) + diagonal

        rnd = torch.rand(1, generator=generator).item()
        sign = 1
        if len(neighbours) > 0 and column_norm > 0.0 and rnd > (diagonal / column_norm):
            idx = torch.randint(len(neighbours), (1,), generator=generator).item()
            sign = neighbours[idx][1]
            face = neighbours[idx][0]
        product *= sign * column_norm

    if torch.equal(face, starting_face):
        stats.incr_number_of_cycles()
        return product
    return 0.0
