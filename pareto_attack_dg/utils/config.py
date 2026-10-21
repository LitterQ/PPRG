from __future__ import annotations

from fractions import Fraction
from typing import Any


def parse_float_like(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if "/" in text:
        return float(Fraction(text))
    return float(text)
