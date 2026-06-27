#!/usr/bin/env python3
"""
Visualize a single LiDAR scan as a 2-D top-down view.

Useful for verifying coordinate transformations and LiDAR data quality.
Supports nuScenes binary ``.pcd.bin`` files as well as PCD/PLY text files.

Usage::

    python3 tools/viz/visualize_lidar_scan.py path/to/scan.pcd.bin
    python3 tools/viz/visualize_lidar_scan.py path/to/scan.pcd.bin --save output.png
    python3 tools/viz/visualize_lidar_scan.py path/to/scan.pcd.bin --no-show
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def load_lidar_pcd(pcd_path: str | Path) -> np.ndarray:
    """Load a LiDAR point cloud from a file.

    Supported formats:

    - **nuScenes binary** (``.bin`` / ``.pcd.bin``): raw ``float32`` with
      5 columns ``[x, y, z, intensity, ring_index]`` or 4 columns
      ``[x, y, z, intensity]``.
    - **PCD/PLY text**: after an ``end_header`` marker, whitespace-separated
      float rows.

    Args:
        pcd_path: Path of the point-cloud file.

    Returns:
        Float32 array of shape ``(N, C)`` where C is 4 or 5.

    Raises:
        ValueError: If the file shape cannot be determined.
    """
    pcd_path = Path(pcd_path)

    if pcd_path.suffix == ".bin" or pcd_path.name.endswith(".pcd.bin"):
        raw = np.fromfile(pcd_path, dtype=np.float32)
        if len(raw) % 5 == 0:
            return raw.reshape(-1, 5)
        if len(raw) % 4 == 0:
            return raw.reshape(-1, 4)
        raise ValueError(
            f"Unexpected element count {len(raw)} — expected multiple of 4 or 5."
        )

    # Text PCD / PLY
    try:
        text = pcd_path.read_text(encoding="utf-8")
        parts = text.split("end_header\n", 1)
        if len(parts) == 2:
            return np.loadtxt(io.StringIO(parts[1])).astype(np.float32)
        return np.loadtxt(pcd_path).astype(np.float32)
    except (UnicodeDecodeError, ValueError):
        raw = np.fromfile(pcd_path, dtype=np.float32)
        if len(raw) % 5 == 0:
            return raw.reshape(-1, 5)
        if len(raw) % 4 == 0:
            return raw.reshape(-1, 4)
        raise


def visualize_lidar_topdown(
    points: np.ndarray,
    title: str = "LiDAR Top-Down View",
    figsize: tuple = (12, 10),
) -> plt.Figure:
    """Build a four-panel top-down LiDAR visualization.

    The panels show:

    1. Top-down view coloured by *intensity*.
    2. Top-down view coloured by *height (z)*.
    3. Side view (X–Z plane) coloured by *intensity*.
    4. Point-cloud statistics text.

    Args:
        points: Array of shape ``(N, 4)`` or ``(N, 5)``.  Columns:
            ``[x, y, z, intensity[, ring_index]]``.
        title: Figure super-title.
        figsize: Matplotlib figure size.

    Returns:
        :class:`matplotlib.figure.Figure` — caller is responsible for
        ``plt.show()`` / ``savefig()`` / ``close()``.
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    intensity = points[:, 3]
    has_rings = points.shape[1] >= 5
    rings = points[:, 4] if has_rings else None

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # 1 — top-down, intensity
    ax = axes[0, 0]
    sc = ax.scatter(x, y, c=intensity, s=1, cmap="viridis", alpha=0.6)
    ax.set_xlabel("X — forward (m)", fontsize=10)
    ax.set_ylabel("Y — left (m)", fontsize=10)
    ax.set_title("Top-Down (intensity)", fontsize=11)
    ax.set_aspect("equal")
    ax.axhline(0, color="r", lw=0.5, ls="--", alpha=0.5)
    ax.axvline(0, color="r", lw=0.5, ls="--", alpha=0.5)
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, label="Intensity")

    # 2 — top-down, height
    ax = axes[0, 1]
    sc2 = ax.scatter(x, y, c=z, s=1, cmap="coolwarm", alpha=0.6)
    ax.set_xlabel("X — forward (m)", fontsize=10)
    ax.set_ylabel("Y — left (m)", fontsize=10)
    ax.set_title("Top-Down (height)", fontsize=11)
    ax.set_aspect("equal")
    ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.5)
    ax.axvline(0, color="k", lw=0.5, ls="--", alpha=0.5)
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc2, ax=ax, label="Z — height (m)")

    # 3 — side view X-Z
    ax = axes[1, 0]
    sc3 = ax.scatter(x, z, c=intensity, s=1, cmap="viridis", alpha=0.6)
    ax.set_xlabel("X — forward (m)", fontsize=10)
    ax.set_ylabel("Z — up (m)", fontsize=10)
    ax.set_title("Side View (X–Z)", fontsize=11)
    ax.axhline(0, color="r", lw=0.5, ls="--", alpha=0.5, label="Ground")
    ax.axvline(0, color="r", lw=0.5, ls="--", alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    plt.colorbar(sc3, ax=ax, label="Intensity")

    # 4 — statistics
    ax = axes[1, 1]
    ax.axis("off")
    ring_info = (
        f"\n    Ring Index:\n      Min: {rings.min():.0f}\n      Max: {rings.max():.0f}"
        if has_rings
        else ""
    )
    stats_text = (
        f"\n    Point Cloud Statistics:\n"
        f"    ─────────────────────────\n"
        f"    Total Points: {len(points):,}\n"
        f"    Columns: {points.shape[1]}\n\n"
        f"    X (forward): {x.min():.2f} … {x.max():.2f} m\n"
        f"    Y (left):    {y.min():.2f} … {y.max():.2f} m\n"
        f"    Z (height):  {z.min():.2f} … {z.max():.2f} m\n\n"
        f"    Intensity:\n"
        f"      Min: {intensity.min():.4f}\n"
        f"      Max: {intensity.max():.4f}\n"
        f"      Mean: {intensity.mean():.4f}"
        f"{ring_info}\n\n"
        f"    Coordinate System: nuScenes\n"
        f"      X = forward,  Y = left,  Z = up\n"
    )
    ax.text(
        0.1, 0.95, stats_text, transform=ax.transAxes,
        fontsize=9, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.tight_layout()
    return fig


def main() -> int:
    """CLI entry point.

    Returns:
        Exit code (0 on success).
    """
    parser = argparse.ArgumentParser(
        description="Visualize a single LiDAR scan as a 2-D top-down view.",
    )
    parser.add_argument("lidar_file", help="Path to LiDAR PCD/binary file")
    parser.add_argument("--save", help="Save visualization to file (e.g. output.png)")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for saved image")
    parser.add_argument(
        "--no-show", action="store_true",
        help="Skip interactive display (useful for headless rendering)",
    )
    args = parser.parse_args()

    lidar_path = Path(args.lidar_file)
    if not lidar_path.exists():
        print(f"Error: file not found: {lidar_path}")
        return 1

    print(f"Loading: {lidar_path}")
    try:
        points = load_lidar_pcd(lidar_path)
        print(f"Loaded {len(points):,} points  (shape {points.shape})")
        fig = visualize_lidar_topdown(points, title=f"LiDAR Scan: {lidar_path.name}")
    except Exception as exc:
        print(f"Error: {exc}")
        import traceback
        traceback.print_exc()
        return 1

    if args.save:
        save_path = Path(args.save)
        fig.savefig(save_path, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)

    return 0


if __name__ == "__main__":
    sys.exit(main())
