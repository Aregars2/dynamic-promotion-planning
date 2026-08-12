"""Canonical behavioral promotion classification."""
from __future__ import annotations

import pandas as pd


CANONICAL_PROMOTION_DEPTH = 0.05


def canonical_promotion_indicator(
    history: pd.DataFrame,
    *,
    depth_column: str = "discount_depth_model",
    recorded_column: str | None = None,
) -> pd.Series:
    """Return recorded-promotion OR at-least-five-percent-discount status."""
    if depth_column not in history:
        raise KeyError(f"Missing promotion depth column: {depth_column}")
    recorded = pd.Series(False, index=history.index)
    if recorded_column is not None and recorded_column in history:
        recorded = pd.to_numeric(history[recorded_column], errors="coerce").fillna(0).gt(0)
    depth = pd.to_numeric(history[depth_column], errors="coerce").fillna(0.0)
    return (recorded | depth.ge(CANONICAL_PROMOTION_DEPTH)).astype(int)


__all__ = ["CANONICAL_PROMOTION_DEPTH", "canonical_promotion_indicator"]
