from __future__ import annotations

import math
from typing import Union

Number = Union[int, float]


def is_close_to_int(value: float) -> bool:
    return math.isclose(value, round(value))


def binomial_coefficient(a: Number, b: Number) -> float:
    """Generalised binomial coefficient covering negative integer arguments."""

    if isinstance(b, float) and not is_close_to_int(b):
        raise ValueError("Second argument to binomial_coefficient must be integral")

    b_int = int(round(b))
    if b_int < 0:
        return 0.0

    if isinstance(a, float) and is_close_to_int(a):
        a = int(round(a))

    if isinstance(a, int):
        if a >= 0:
            if b_int > a:
                return 0.0
            return float(math.comb(a, b_int))
        # Negative integer case: use identity C(-n, k) = (-1)^k C(n + k - 1, k)
        n = -a
        return float(((-1) ** b_int) * math.comb(n + b_int - 1, b_int))

    # General real-valued a: use gamma function definition
    return math.exp(
        math.lgamma(a + 1) - math.lgamma(b_int + 1) - math.lgamma(a - b_int + 1)
    )
