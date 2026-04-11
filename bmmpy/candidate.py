"""
Candidate type for bmmpy algorithms results.

A Candidate represents a selected subset of rows together with the weight
associated with the transformed row combination.

Candidates are usually produced by a searcher and then consumed by an apply
strategy such as GreedySelection.
"""

from __future__ import annotations

from ._bmmpy import Candidate

__all__ = ["Candidate"]