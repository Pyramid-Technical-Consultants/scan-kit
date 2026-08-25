"""Tests for contour level helpers."""

from scan_kit.common.ic_xy_distribution import (
    CONTOUR_LEVEL_COUNT,
    _plot_density_contours,
    contour_level_percentiles,
)


def test_contour_level_percentiles_starts_at_cutoff() -> None:
    levels = contour_level_percentiles(5.0)
    assert len(levels) == CONTOUR_LEVEL_COUNT
    assert levels[0] == 5.0
    assert levels[-1] == 97.0


def test_contour_level_percentiles_clamps_input() -> None:
    levels = contour_level_percentiles(0.0)
    assert levels[0] == 0.0


def test_contour_level_percentiles_rejects_negative() -> None:
    levels = contour_level_percentiles(-5.0)
    assert levels[0] == 0.0


def test_plot_density_contours_zero_cutoff() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(0)
    x = rng.normal(size=500) * 0.2
    y = rng.normal(size=500) * 0.2
    fig, ax = plt.subplots()
    _plot_density_contours(
        ax, x, y, "C0", lim=(-1.0, 1.0), positive_only=False,
        contour_cutoff_percentile=0.0,
    )
    plt.close(fig)
