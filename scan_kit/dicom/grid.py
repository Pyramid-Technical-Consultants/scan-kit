"""Voxel grids placed in DICOM patient coordinates (LPS, mm)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class VolumeGrid:
    """A regular grid: voxel (i, j, k) center is ``origin + axes @ (spacing * (i, j, k))``.

    Arrays on it are ``(nz, ny, nx)``. *axes* columns are the unit i, j and k directions
    in patient LPS; *origin* is the center of voxel 0.
    """

    origin: np.ndarray
    spacing: np.ndarray
    shape: tuple[int, int, int]  # (nx, ny, nz)
    axes: np.ndarray = field(default_factory=lambda: np.eye(3))
    frame_uid: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", np.asarray(self.origin, dtype=float).reshape(3))
        object.__setattr__(self, "spacing", np.asarray(self.spacing, dtype=float).reshape(3))
        object.__setattr__(self, "axes", np.asarray(self.axes, dtype=float).reshape(3, 3))
        object.__setattr__(self, "shape", tuple(int(v) for v in self.shape))
        if not np.allclose(self.axes.T @ self.axes, np.eye(3), atol=1e-4):
            raise ValueError("grid axes must be orthonormal (oblique or sheared volumes are not supported)")
        if np.any(self.spacing <= 0.0):
            raise ValueError(f"grid spacing must be positive, got {self.spacing}")

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return self.shape[::-1]

    @property
    def corner(self) -> np.ndarray:
        """Patient position of the outer corner of voxel 0."""
        return self.origin - self.axes @ (0.5 * self.spacing)

    def to_patient(self, ijk) -> np.ndarray:
        """Continuous voxel indices ``(..., 3)`` to patient mm."""
        return np.asarray(ijk, dtype=float) * self.spacing @ self.axes.T + self.origin

    def to_index(self, lps) -> np.ndarray:
        """Patient mm ``(..., 3)`` to continuous voxel indices."""
        return (np.asarray(lps, dtype=float) - self.origin) @ self.axes / self.spacing

    def to_local(self, lps) -> np.ndarray:
        """Patient mm to the grid frame: mm along the grid axes from :attr:`corner`."""
        return (np.asarray(lps, dtype=float) - self.corner) @ self.axes

    def with_frame(self, frame_uid: str) -> VolumeGrid:
        return VolumeGrid(self.origin, self.spacing, self.shape, self.axes, frame_uid)
