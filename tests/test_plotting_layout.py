"""Tests for shared plotting layout helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from scan_kit.common.plotting import (
    _gridspec_layout_is_manual,
    apply_tight_layout,
    set_view_header,
)
from scan_kit.views.beam_off_rampdown import _heatmap_energy_extent


def test_heatmap_energy_extent_expands_zero_span() -> None:
    assert _heatmap_energy_extent([220.0]) == (219.5, 220.5)
    assert _heatmap_energy_extent([220.0, 220.0]) == (219.5, 220.5)


def test_heatmap_energy_extent_preserves_span() -> None:
    assert _heatmap_energy_extent([100.0, 220.0]) == (100.0, 220.0)


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
