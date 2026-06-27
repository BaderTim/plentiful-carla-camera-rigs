"""
Quaternion helper with graceful ``pyquaternion`` fallback.

Import this module instead of duplicating the fallback stub in every file
that works with rotation quaternions.

Usage::

    from lib.quaternion import Quaternion

If ``pyquaternion`` is installed, this re-exports that class unchanged.
Otherwise a minimal implementation covering the subset of the API used by
this project (``rotation_matrix``, ``inverse``, construction from ``array``)
is provided so that the full CARLA stack is not required just to run
visualization tools.
"""

from __future__ import annotations

import numpy as np

try:
    from pyquaternion import Quaternion  # type: ignore[import]
except ImportError:  # pragma: no cover
    class Quaternion:  # type: ignore[no-redef]
        """Minimal quaternion implementation (fallback when pyquaternion is absent).

        Supports the subset of the ``pyquaternion.Quaternion`` API used by
        this project: construction from ``(w, x, y, z)`` scalars or from an
        ``array`` keyword argument, and the ``rotation_matrix`` / ``inverse``
        properties.
        """

        def __init__(
            self,
            w: float = 1.0,
            x: float = 0.0,
            y: float = 0.0,
            z: float = 0.0,
            *,
            array: "np.ndarray | list[float] | None" = None,
        ) -> None:
            if array is not None:
                self.q = np.asarray(array, dtype=np.float32)
            else:
                self.q = np.array([w, x, y, z], dtype=np.float32)

        @property
        def rotation_matrix(self) -> np.ndarray:
            """Return the 3×3 rotation matrix equivalent of this quaternion."""
            w, x, y, z = self.q
            return np.array(
                [
                    [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                    [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
                    [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
                ],
                dtype=np.float32,
            )

        @property
        def inverse(self) -> "Quaternion":
            """Return the inverse (conjugate / norm²) of this quaternion."""
            w, x, y, z = self.q
            norm_sq = float(np.sum(self.q ** 2))
            if norm_sq == 0.0:
                return Quaternion(1.0, 0.0, 0.0, 0.0)
            return Quaternion(w / norm_sq, -x / norm_sq, -y / norm_sq, -z / norm_sq)

        def __repr__(self) -> str:
            w, x, y, z = self.q
            return f"Quaternion(w={w:.4f}, x={x:.4f}, y={y:.4f}, z={z:.4f})"
