from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

from .math_utils import binomial_coefficient


def poly_add(a: Sequence[float], b: Sequence[float]) -> List[float]:
    degree = max(len(a), len(b))
    result = [0.0] * degree
    for idx, value in enumerate(a):
        result[idx] += value
    for idx, value in enumerate(b):
        result[idx] += value
    return result


def poly_sub(a: Sequence[float], b: Sequence[float]) -> List[float]:
    degree = max(len(a), len(b))
    result = [0.0] * degree
    for idx, value in enumerate(a):
        result[idx] += value
    for idx, value in enumerate(b):
        result[idx] -= value
    return result


def poly_scale(poly: Sequence[float], scalar: float) -> List[float]:
    return [scalar * value for value in poly]


def poly_mul_x(poly: Sequence[float]) -> List[float]:
    return [0.0] + list(poly)


def get_chebyshev_coefficients(n: int) -> List[float]:
    if n == 0:
        return [1.0]
    if n == 1:
        return [0.0, 1.0]

    T_prev = [1.0]
    T_curr = [0.0, 1.0]
    for _ in range(2, n + 1):
        T_next = poly_sub(poly_scale(poly_mul_x(T_curr), 2.0), T_prev)
        T_prev, T_curr = T_curr, T_next
    return T_curr


def chebyshev_t_value(n: int, x: float) -> float:
    if n == 0:
        return 1.0
    if n == 1:
        return x
    if abs(x) <= 1:
        return math.cos(n * math.acos(x))
    return math.cosh(n * math.acosh(x))


def get_coefficients_apers(power: int, degree: int) -> List[float]:
    coeffs = [0.0] * (degree + 1)
    denom = 2.0**power
    for j in range(degree + 1):
        if (j & 1) == (power & 1):
            multiplier = 1 if j == 0 else 2
            coeffs[j] = multiplier * (binomial_coefficient(power, (power - j) // 2) / denom)
    return coeffs


def get_polynomial_apers(gamma: float, epsilon: float) -> Tuple[List[float], int]:
    d = math.ceil(math.sqrt(2.0 / gamma) * math.log(6.0 / epsilon))
    r = math.ceil((1.0 / gamma) * math.log(3.0 / epsilon))

    cheby_coeffs = [get_chebyshev_coefficients(j) for j in range(d + 1)]
    coeffs = get_coefficients_apers(r, d)

    combined = [0.0] * (d + 1)
    for j in range(d + 1):
        coeff = coeffs[j]
        if coeff == 0.0:
            continue
        for k in range(len(cheby_coeffs[j])):
            combined_index = k
            if combined_index <= d:
                combined[combined_index] += coeff * cheby_coeffs[j][k]
    return combined, d


def get_coefficients_musco(cheby_coeffs: Sequence[float], gamma: float, d: int) -> List[float]:
    res = [0.0] * (d + 1)
    factor = 1.0 / (1.0 - gamma)
    outer = 1.0 / chebyshev_t_value(d, factor)
    for j in range(d + 1):
        scale = 1.0 if j == 0 else factor**j
        res[j] = cheby_coeffs[j] * outer * scale
    return res


def get_polynomial_musco(gamma: float, epsilon: float, deg_limit: int = -1) -> Tuple[List[float], int]:
    d = math.ceil(math.sqrt(1.0 / gamma) * math.log(4.0 / epsilon))
    if deg_limit != -1:
        d = deg_limit
    cheby = get_chebyshev_coefficients(d)
    coeffs = get_coefficients_musco(cheby, gamma, d)
    return coeffs, d


def cheby_coeff(d: int, k: int, gamma: float) -> float:
    if (d & 1) != (k & 1):
        return 0.0
    m = (d - k) // 2
    term1 = (-1.0) ** m
    term2 = 2.0 ** (-2.0 * m + d - 1)
    term3 = binomial_coefficient(-m + d - 1, d - 2 * m) + binomial_coefficient(d - m, m)
    term4 = (1.0 - gamma) ** k
    return (term1 * term2 * term3) / term4


def get_coefficients_musco_compressed(
    cheby_coeffs: Sequence[float], gamma: float, d: int
) -> List[float]:
    res = [0.0] * (d + 1)
    factor = (1.0 + gamma) / (1.0 - gamma)
    outer = 1.0 / chebyshev_t_value(d, factor)
    for j in range(d + 1):
        acc = 0.0
        for k in range(j, d + 1):
            term1 = binomial_coefficient(k, j)
            term2 = (-0.5 * (1.0 - gamma)) ** (k - j)
            term3 = cheby_coeff(d, k, (gamma + 1.0) / 2.0)
            acc += term1 * term2 * term3
        res[j] = outer * acc
    return res


def get_polynomial_musco_compressed(
    gamma: float, epsilon: float, deg_limit: int = -1
) -> Tuple[List[float], int]:
    d = math.ceil(math.sqrt(1.0 / gamma) * math.log(2.0 / epsilon))
    if deg_limit != -1:
        d = deg_limit

    while True:
        cheby = get_chebyshev_coefficients(d)
        coeffs = get_coefficients_musco_compressed(cheby, gamma, d)
        if abs(coeffs[0]) < epsilon / 2.0:
            return coeffs, d
        d += 1
