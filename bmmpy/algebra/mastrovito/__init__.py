"""Mastrovito-based generators for binary parity-check matrices."""

from __future__ import annotations

from .core import Mastrovito, build_check_matrix, get_mastrovito_matrix
from .parsing import parse_poly

__all__ = [
    "Mastrovito",
    "build_check_matrix",
    "get_mastrovito_matrix",
    "parse_poly",
]