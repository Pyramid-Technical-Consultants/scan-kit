"""Tests for shared plotting layout helpers."""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np

from scan_kit.common.plotting import (
    _gridspec_layout_is_manual,
    apply_tight_layout,
    scatter_with_trend,
    set_view_header,
)
def test_apply_tight_layout_skips_manual_gridspec() -> None:
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(1, 1, left=0.1, right=0.9, top=0.9, bottom=0.1)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot([0, 1], [0, 1])
    set_view_header(fig, "Manual Grid", ["s1"], ["#1f77b4"])
    assert _gridspec_layout_is_manual(fig)
    apply_tight_layout(fig)
    plt.close(fig)


def test_apply_tight_layout_with_colorbar_nested_gridspec() -> None:
    from scan_kit.common.plotting import view_grid

    fig, axes = view_grid(2, 2, width_ratios=[1, 1])
    ax = axes[0, 0]
    im = ax.pcolormesh([1, 2, 3], [1, 2], np.array([[0, 1, 2], [2, 1, 0]]))
    fig.colorbar(im, ax=ax)
    set_view_header(fig, "Colorbar Grid", ["s1"], ["#1f77b4"])
    apply_tight_layout(fig)
    plt.close(fig)


def test_scatter_with_trend_constant_x_no_rank_warning() -> None:
    fig, ax = plt.subplots()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        slope = scatter_with_trend(
            ax,
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 3.0, 4.0, 5.0],
            color="red",
            label="s",
        )
    assert slope is None
    assert not any("Polyfit may be poorly conditioned" in str(w.message) for w in caught)
    plt.close(fig)
