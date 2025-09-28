from __future__ import annotations

import math
from enum import Enum
from typing import Optional, Sequence

import torch

from .complex import Complex, sample_markov_chain
from .config import RuntimeConfig
from .polynomials import (
    get_polynomial_apers,
    get_polynomial_musco,
    get_polynomial_musco_compressed,
)
from .stats import Statistics


class PolyType(str, Enum):
    APERS = "cbneCheby"
    MUSCO = "cbneMusco"
    MUSCO_COMPRESSED = "cbneCompressed"


def _cbne_base(
    n: int,
    k: int,
    complex_obj: Complex,
    gamma: float,
    one_norm: float,
    config: RuntimeConfig,
    stats: Statistics,
    generator: torch.Generator,
) -> float:
    epsilon = config.epsilon
    delta = epsilon / 2.0
    z = math.ceil((1.0 / gamma) * math.log(2.0 / epsilon))

    if config.deg_limit != -1:
        z = config.deg_limit

    base = one_norm if one_norm != 0 and config.use_one_norm else 2.0
    numerator = 4.0 * (base ** (2 * z)) * math.log(2.0 / 0.1)
    denominator = 2.0 * delta * delta
    p = numerator / denominator

    if config.iter_limit != -1:
        p = config.iter_limit

    p = max(1, math.ceil(p))

    if config.output_shot_count:
        print(f"Shot count: {p}")
    if config.output_step_count:
        print(f"Step count: {p * z}")
    if config.should_output_counts():
        return float("nan")

    stats.set_sample_count(p)
    stats.set_walk_length(z)

    total = 0.0
    for _ in range(p):
        start_face = complex_obj.sample_from_complex(k)
        y_z = sample_markov_chain(start_face, complex_obj, n, int(math.ceil(z)), stats, generator)
        total += y_z

    return total / float(p)


def _calculate_polynomial(
    gamma: float, epsilon: float, config: RuntimeConfig
) -> tuple[Sequence[float], int, PolyType]:
    if config.cbne_version == "cbneCheby":
        poly, degree = get_polynomial_apers(gamma, epsilon)
        poly_type = PolyType.APERS
    elif config.cbne_version == "cbneMusco":
        poly, degree = get_polynomial_musco(gamma, epsilon, config.deg_limit)
        poly_type = PolyType.MUSCO
    else:
        poly, degree = get_polynomial_musco_compressed(gamma, epsilon, config.deg_limit)
        poly_type = PolyType.MUSCO_COMPRESSED
    return poly, degree, poly_type


def _cbne_chebyshev(
    n: int,
    k: int,
    complex_obj: Complex,
    gamma: float,
    one_norm: float,
    config: RuntimeConfig,
    stats: Statistics,
    generator: torch.Generator,
) -> Optional[float]:
    epsilon = config.epsilon
    poly, degree, poly_type = _calculate_polynomial(gamma, epsilon, config)

    def shot_count_for(j: int) -> int:
        error_factor = 3 if poly_type == PolyType.APERS else 2
        if poly[j] == 0:
            return 0
        delta = epsilon / (error_factor * math.ceil(degree / 2.0) * abs(poly[j]))
        eta = 1 - math.pow(0.90, 1.0 / math.ceil(degree / 2.0))
        base = one_norm if one_norm != 0 and config.use_one_norm else 2.0
        numerator = 4.0 * (base ** (2 * j)) * math.log(2.0 / eta)
        return max(0, math.ceil(numerator / (2.0 * delta * delta)))

    shot_counts = [0] * (degree + 1)
    total_shots = 0
    total_steps = 0
    for j in range(degree + 1):
        shots = shot_count_for(j)
        shot_counts[j] = shots
        total_shots += shots
        total_steps += shots * j

    if config.iter_limit != -1 and total_shots > 0:
        fraction = config.iter_limit / total_shots
        total_shots = 0
        total_steps = 0
        for j in range(degree + 1):
            scaled = math.ceil(shot_counts[j] * fraction)
            shot_counts[j] = scaled
            total_shots += scaled
            total_steps += scaled * j
    if config.output_shot_count:
        print(f"Shot count: {total_shots}")
    if config.output_step_count:
        print(f"Step count: {total_steps}")
    if config.should_output_counts():
        return None

    stats.set_sample_count(total_shots)
    stats.set_walk_length(degree)

    limit = config.num_data_points
    estimates = [0.0] * len(poly)
    realised_shots = [0] * len(poly)
    completed = [0] * len(poly)
    result = 0.0

    for datapoint in range(limit):
        for j in range(degree + 1):
            if poly[j] == 0 or shot_counts[j] == 0:
                continue
            if limit == 1:
                batch = shot_counts[j]
            else:
                def allocate(idx: int) -> int:
                    return math.floor(idx * (shot_counts[j] / float(limit)))

                batch = allocate(datapoint + 1) - allocate(datapoint)
                if datapoint == limit - 1:
                    batch = max(0, shot_counts[j] - completed[j])
                completed[j] += batch
            if batch <= 0:
                continue

            acc = 0.0
            for _ in range(batch):
                start_face = complex_obj.sample_from_complex(k)
                value = sample_markov_chain(start_face, complex_obj, n, j, stats, generator)
                acc += value
            estimates[j] += acc
            realised_shots[j] += batch

        result = 0.0
        for j in range(degree + 1):
            if realised_shots[j] == 0:
                continue
            result += poly[j] * (estimates[j] / realised_shots[j])

    return result


def estimate(
    adjacency: torch.Tensor,
    spectral_gap: float,
    dimension: int,
    one_norm: float,
    config: RuntimeConfig,
) -> tuple[Optional[float], Statistics]:
    config.validate()
    device = config.torch_device()
    adjacency = adjacency.to(device)
    generator = torch.Generator(device="cpu")
    if config.seed is not None:
        generator.manual_seed(config.seed)
    complex_obj = Complex(adjacency, generator=generator)
    stats = Statistics()
    n = adjacency.size(0)

    if config.cbne_version == "cbne":
        value = _cbne_base(n, dimension, complex_obj, spectral_gap, one_norm, config, stats, generator)
        if config.should_output_counts():
            return None, stats
        return value, stats

    value = _cbne_chebyshev(n, dimension, complex_obj, spectral_gap, one_norm, config, stats, generator)
    return value, stats
