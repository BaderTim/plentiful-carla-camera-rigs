#!/usr/bin/env python3
"""
Visualize packed annotation (pkl) files produced by the data converter.

Projects 3-D ground-truth boxes (stored in the LiDAR frame) onto every camera
image referenced in each info record.

Usage::

    python3 tools/viz/visualize_pkl.py bevdet_infos_train.pkl --root /path/to/dataset
    python3 tools/viz/visualize_pkl.py bevdet_infos_train.pkl --root . --idx 42
    python3 tools/viz/visualize_pkl.py bevdet_infos_train.pkl --output my_vis/ --no-color
"""

from __future__ import annotations

import argparse
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

# Ensure project root is importable when run directly.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib.quaternion import Quaternion

# ---------------------------------------------------------------------------
# Per-class colours (matches nuscenes_standards category tokens)
# ---------------------------------------------------------------------------
_CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "car": (0, 255, 0),
    "truck": (255, 165, 0),
    "bus": (255, 127, 80),
    "motorcycle": (255, 255, 0),
    "bicycle": (220, 20, 60),
    "pedestrian": (0, 0, 255),
    "traffic_cone": (255, 165, 0),
    "barrier": (128, 128, 128),
    "construction_vehicle": (139, 69, 19),
    "trailer": (255, 200, 0),
    "adult": (0, 0, 255),
    "child": (0, 191, 255),
    "traffic_sign": (255, 0, 255),
    "traffic_light": (255, 0, 255),
}

_DEFAULT_COLOR: Tuple[int, int, int] = (0, 255, 0)

_BOX_EDGES = [
    (4, 5), (5, 6), (6, 7), (7, 4),  # bottom
    (0, 1), (1, 2), (2, 3), (3, 0),  # top
    (0, 4), (1, 5), (2, 6), (3, 7),  # vertical
]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _project_point(
    point_3d: np.ndarray,
    intrinsic: np.ndarray,
) -> Optional[np.ndarray]:
    """Project a 3-D point in camera space to 2-D pixel coordinates.

    Args:
        point_3d: Shape ``(3,)`` — camera-space coordinates (z forward).
        intrinsic: 3×3 camera intrinsic matrix.

    Returns:
        Shape ``(2,)`` pixel coordinate, or ``None`` if behind the camera.
    """
    if point_3d[2] <= 0:
        return None
    p = intrinsic @ point_3d
    p /= p[2]
    return p[:2]


def _get_box_corners(
    center: np.ndarray,
    size: np.ndarray,
    yaw: float,
) -> np.ndarray:
    """Return 8 box corners in the LiDAR frame.

    Args:
        center: ``[x, y, z]`` of box centre.
        size: ``[l, w, h]`` — length × width × height.
        yaw: Heading angle in radians (rotation around z-axis).

    Returns:
        Shape ``(8, 3)`` array of corners.
    """
    l, w, h = size
    x_c = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_c = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    z_c = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
    corners = np.vstack([x_c, y_c, z_c])  # (3, 8)

    cos, sin = np.cos(yaw), np.sin(yaw)
    R = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])
    corners = R @ corners
    corners[0] += center[0]
    corners[1] += center[1]
    corners[2] += center[2]
    return corners.T  # (8, 3)


# ---------------------------------------------------------------------------
# Core visualisation
# ---------------------------------------------------------------------------

def visualize_pkl(
    pkl_path: str,
    dataset_root: str,
    output_dir: str,
    sample_idx: Optional[int] = None,
    use_color: bool = True,
) -> None:
    """Load a pkl info file and render annotation boxes on camera images.

    Args:
        pkl_path: Path to the ``.pkl`` annotation file.
        dataset_root: Root directory containing the image files referenced
            inside the pkl.
        output_dir: Directory to write rendered JPEG images.
        sample_idx: Zero-based index of the sample to visualize.  If
            ``None``, a random sample is chosen.
        use_color: When ``True``, colour boxes by class name; use green for
            all boxes when ``False``.
    """
    print(f"Loading {pkl_path}...")
    with open(pkl_path, "rb") as f:
        data: Dict[str, Any] = pickle.load(f)

    infos: List[Dict] = data["infos"]
    if not infos:
        print("No infos found in pkl file.")
        return

    if sample_idx is None:
        sample_idx = random.randint(0, len(infos) - 1)

    info = infos[sample_idx]
    print(
        f"Visualizing sample index {sample_idx} "
        f"(token: {info.get('token', 'N/A')})"
    )

    gt_boxes: np.ndarray = info["gt_boxes"]   # (N, 7) — [x, y, z, l, w, h, yaw]
    gt_names: List[str] = info["gt_names"]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for cam_name, cam_info in info.get("cams", {}).items():
        image_path = Path(dataset_root) / cam_info["data_path"]
        if not image_path.exists():
            print(f"  Warning: Image {image_path} not found.")
            continue

        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)

        # sensor2lidar_rotation brings points from sensor frame to LiDAR frame.
        # To project LiDAR points into this sensor: lidar → sensor is the inverse.
        r_s2l = np.array(cam_info["sensor2lidar_rotation"])   # (3, 3)
        t_s2l = np.array(cam_info["sensor2lidar_translation"])  # (3,)
        intrinsic = np.array(cam_info["cam_intrinsic"])          # (3, 3)

        for i, box in enumerate(gt_boxes):
            center, size, yaw = box[:3], box[3:6], box[6]
            corners_lidar = _get_box_corners(center, size, yaw)  # (8, 3)

            # LiDAR → camera: P_cam = (P_lidar - t_s2l) @ r_s2l
            corners_cam = (corners_lidar - t_s2l) @ r_s2l       # (8, 3)

            label = gt_names[i] if i < len(gt_names) else "unknown"
            if use_color:
                color = _CLASS_COLORS.get(label.lower(), _DEFAULT_COLOR)
            else:
                color = _DEFAULT_COLOR

            for start, end in _BOX_EDGES:
                p1_cam = corners_cam[start]
                p2_cam = corners_cam[end]
                if p1_cam[2] < 0.1 or p2_cam[2] < 0.1:
                    continue
                p1 = _project_point(p1_cam, intrinsic)
                p2 = _project_point(p2_cam, intrinsic)
                if p1 is not None and p2 is not None:
                    draw.line(
                        [(p1[0], p1[1]), (p2[0], p2[1])],
                        fill=color,
                        width=2,
                    )

        save_path = out_path / f"pkl_vis_{cam_name}_{sample_idx}.jpg"
        img.save(save_path)
        print(f"  Saved {save_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize packed annotation files (pkl) on camera images.",
    )
    parser.add_argument("pkl_path", help="Path to the annotation pkl file")
    parser.add_argument("--root", default=".", help="Dataset root directory")
    parser.add_argument("--output", default="vis_pkl", help="Output directory")
    parser.add_argument(
        "--idx", type=int, default=None, metavar="N",
        help="Sample index to visualize (default: random)",
    )
    parser.add_argument(
        "--no-color", action="store_true",
        help="Disable per-class colour coding (use green for all boxes)",
    )
    args = parser.parse_args()
    visualize_pkl(
        args.pkl_path, args.root, args.output, args.idx,
        use_color=not args.no_color,
    )


if __name__ == "__main__":
    main()
