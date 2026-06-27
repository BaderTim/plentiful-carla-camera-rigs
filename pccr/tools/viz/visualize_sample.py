#!/usr/bin/env python3
"""
Visualize a nuScenes-format sample by projecting 3-D bounding boxes onto camera images.

Usage::

    python3 tools/viz/visualize_sample.py /path/to/dataset --output vis_output/
    python3 tools/viz/visualize_sample.py /path/to/dataset --sample-idx 5 --output vis_output/
    python3 tools/viz/visualize_sample.py /path/to/dataset --sample-token <token>
    python3 tools/viz/visualize_sample.py --version           # print version and exit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

# Ensure project root is importable when run directly.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib.nuscenes_standards import (
    CAT_ADULT,
    CAT_BICYCLE,
    CAT_BUS,
    CAT_CAR,
    CAT_CHILD,
    CAT_MOTORCYCLE,
    CAT_TRAFFIC_LIGHT,
    CAT_TRAFFIC_SIGN,
    CAT_TRUCK,
)
from lib.quaternion import Quaternion

__version__ = "1.1.0"

# Per-category RGB colours.
CATEGORY_COLORS: Dict[str, Tuple[int, int, int]] = {
    CAT_CAR: (0, 255, 0),
    CAT_TRUCK: (255, 165, 0),
    CAT_BUS: (255, 127, 80),
    CAT_MOTORCYCLE: (255, 255, 0),
    CAT_BICYCLE: (220, 20, 60),
    CAT_ADULT: (0, 0, 255),
    CAT_CHILD: (0, 191, 255),
    CAT_TRAFFIC_SIGN: (255, 0, 255),
    CAT_TRAFFIC_LIGHT: (255, 0, 255),
}

# Box edges: (start_corner_idx, end_corner_idx).
_BOX_EDGES = [
    (4, 5), (5, 6), (6, 7), (7, 4),  # bottom face
    (0, 1), (1, 2), (2, 3), (3, 0),  # top face
    (0, 4), (1, 5), (2, 6), (3, 7),  # vertical pillars
]


class NuScenesVisualizer:
    """Load and visualize a nuScenes-format dataset.

    Attributes:
        dataset_dir: Root dataset directory.
        data: Dict mapping table names to their list of records.
        indices: Fast token-lookup dicts derived from :attr:`data`.
    """

    _TABLES = [
        "sample",
        "sample_data",
        "ego_pose",
        "calibrated_sensor",
        "sample_annotation",
        "category",
        "sensor",
        "instance",
    ]

    def __init__(self, dataset_dir: str) -> None:
        """Load all JSON tables from *dataset_dir*.

        Args:
            dataset_dir: Path to the dataset root.  The tool looks for JSON
                tables under ``<dataset_dir>/v1.0-trainval/``.
        """
        self.dataset_dir = Path(dataset_dir)
        self.data: Dict[str, List[Dict[str, Any]]] = {}
        self.indices: Dict[str, Dict[str, Any]] = {}
        self._load_dataset()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_dataset(self) -> None:
        """Populate :attr:`data` and :attr:`indices` from disk."""
        table_root = self.dataset_dir / "v1.0-trainval"
        print(f"Loading dataset from {table_root}...")
        for table in self._TABLES:
            json_path = table_root / f"{table}.json"
            if not json_path.exists():
                print(f"  Warning: {table}.json not found")
                self.data[table] = []
            else:
                with open(json_path) as f:
                    self.data[table] = json.load(f)
        for table in self._TABLES:
            self.indices[table] = {
                item["token"]: item for item in self.data.get(table, [])
            }

    # ------------------------------------------------------------------
    # Record accessors
    # ------------------------------------------------------------------

    def get_sample(
        self,
        sample_token: Optional[str] = None,
        sample_idx: int = 0,
    ) -> Optional[Dict[str, Any]]:
        """Return a sample by token or zero-based index.

        Args:
            sample_token: Token of the desired sample.  Takes priority over
                *sample_idx* when supplied.
            sample_idx: Fallback list index.

        Returns:
            Sample record dict, or ``None`` if not found.
        """
        if sample_token:
            return self.indices["sample"].get(sample_token)
        samples = self.data["sample"]
        return samples[sample_idx] if 0 <= sample_idx < len(samples) else None

    def get_sample_data(self, sample_token: str) -> List[Dict[str, Any]]:
        """Return all sample_data records for *sample_token*."""
        return [
            sd for sd in self.data["sample_data"]
            if sd["sample_token"] == sample_token
        ]

    def get_annotations(self, sample_token: str) -> List[Dict[str, Any]]:
        """Return all annotations for *sample_token*."""
        return [
            ann for ann in self.data["sample_annotation"]
            if ann["sample_token"] == sample_token
        ]

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _transform_matrix(translation: List[float], rotation: List[float]) -> np.ndarray:
        """Build a 4×4 homogeneous transform from nuScenes translation+rotation.

        Args:
            translation: ``[x, y, z]``.
            rotation: Quaternion ``[w, x, y, z]``.

        Returns:
            4×4 float64 transform matrix.
        """
        q = Quaternion(array=np.array(rotation))
        T = np.eye(4)
        T[:3, :3] = q.rotation_matrix
        T[:3, 3] = np.array(translation)
        return T

    @staticmethod
    def _project_point(
        point_3d: np.ndarray,
        intrinsic: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Project a 3-D camera-space point to 2-D image coordinates.

        Args:
            point_3d: Shape ``(3,)`` in camera space (z forward).
            intrinsic: 3×3 camera intrinsic matrix.

        Returns:
            Shape ``(2,)`` image coordinates, or ``None`` if behind the camera.
        """
        if point_3d[2] <= 0:
            return None
        p = intrinsic @ point_3d
        p /= p[2]
        return p[:2]

    @staticmethod
    def _clip_line(
        p1: np.ndarray,
        p2: np.ndarray,
        near_clip: float = 0.1,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Clip a 3-D line segment against the near plane ``z = near_clip``.

        Args:
            p1: Start point in camera space.
            p2: End point in camera space.
            near_clip: Near-plane distance.

        Returns:
            Clipped ``(p1, p2)`` pair, or ``None`` if fully behind the plane.
        """
        z1, z2 = p1[2], p2[2]
        if z1 < near_clip and z2 < near_clip:
            return None
        if z1 >= near_clip and z2 >= near_clip:
            return p1, p2
        t = (near_clip - z1) / (z2 - z1)
        p_clip = p1 + t * (p2 - p1)
        return (p_clip, p2) if z1 < near_clip else (p1, p_clip)

    @staticmethod
    def _get_box_corners_cam(
        annotation: Dict[str, Any],
        world_to_cam: np.ndarray,
    ) -> np.ndarray:
        """Compute 8 box corners in camera space.

        Args:
            annotation: nuScenes annotation record with ``size``, ``translation``,
                and ``rotation`` fields.
            world_to_cam: 4×4 world-to-camera transform.

        Returns:
            Shape ``(8, 3)`` array of corners in camera space.
        """
        w, length, h = annotation["size"]
        x_c = [length / 2, length / 2, -length / 2, -length / 2,
               length / 2, length / 2, -length / 2, -length / 2]
        y_c = [w / 2, -w / 2, -w / 2, w / 2,
               w / 2, -w / 2, -w / 2, w / 2]
        z_c = [h / 2, h / 2, h / 2, h / 2,
               -h / 2, -h / 2, -h / 2, -h / 2]
        corners = np.vstack([x_c, y_c, z_c])  # (3, 8)

        q = Quaternion(array=np.array(annotation["rotation"]))
        corners = q.rotation_matrix @ corners
        corners[0] += annotation["translation"][0]
        corners[1] += annotation["translation"][1]
        corners[2] += annotation["translation"][2]

        corners_hom = np.vstack([corners, np.ones((1, 8))])  # (4, 8)
        cam = world_to_cam @ corners_hom                     # (4, 8)
        return cam[:3].T                                     # (8, 3)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def visualize_sample(self, sample_token: str, output_dir: str) -> None:
        """Render annotation boxes on all camera images for *sample_token*.

        Saves one JPEG per camera to *output_dir*.

        Args:
            sample_token: nuScenes sample token.
            output_dir: Directory that will receive the rendered images.
        """
        sample = self.get_sample(sample_token)
        if sample is None:
            print(f"Sample {sample_token} not found")
            return

        print(
            f"Visualizing sample {sample['token']} "
            f"(timestamp: {sample['timestamp']})"
        )
        annotations = self.get_annotations(sample["token"])
        print(f"  {len(annotations)} annotations")

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Filter for camera sensors.
        cameras = [
            sd for sd in self.get_sample_data(sample["token"])
            if self.indices["sensor"].get(
                self.indices["calibrated_sensor"].get(
                    sd["calibrated_sensor_token"], {}
                ).get("sensor_token", ""),
                {},
            ).get("modality") == "camera"
        ]

        for cam_data in cameras:
            image_path = self.dataset_dir / cam_data["filename"]
            if not image_path.exists():
                print(f"  Image not found: {image_path}")
                continue
            try:
                img = Image.open(image_path)
                draw = ImageDraw.Draw(img)
            except Exception as exc:
                print(f"  Error loading {image_path}: {exc}")
                continue

            ego_pose = self.indices["ego_pose"][cam_data["ego_pose_token"]]
            global_to_ego = np.linalg.inv(
                self._transform_matrix(ego_pose["translation"], ego_pose["rotation"])
            )
            cal_sensor = self.indices["calibrated_sensor"][
                cam_data["calibrated_sensor_token"]
            ]
            ego_to_sensor = np.linalg.inv(
                self._transform_matrix(cal_sensor["translation"], cal_sensor["rotation"])
            )
            world_to_cam = ego_to_sensor @ global_to_ego
            intrinsic = np.array(cal_sensor["camera_intrinsic"])

            for ann in annotations:
                inst = self.indices.get("instance", {}).get(ann["instance_token"])
                cat_token = inst["category_token"] if inst else CAT_CAR
                color = CATEGORY_COLORS.get(cat_token, (255, 0, 0))
                corners = self._get_box_corners_cam(ann, world_to_cam)

                for start, end in _BOX_EDGES:
                    clipped = self._clip_line(corners[start], corners[end])
                    if clipped is None:
                        continue
                    p1 = self._project_point(clipped[0], intrinsic)
                    p2 = self._project_point(clipped[1], intrinsic)
                    if p1 is not None and p2 is not None:
                        draw.line(
                            [(p1[0], p1[1]), (p2[0], p2[1])],
                            fill=color,
                            width=2,
                        )

            sensor_rec = self.indices["sensor"][cal_sensor["sensor_token"]]
            cam_name = sensor_rec["channel"]
            save_path = out_path / f"{cam_name}_{sample['token']}.jpg"
            img.save(save_path)
            print(f"  Saved {save_path}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Project 3-D bounding boxes onto camera images for a nuScenes sample.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("dataset_dir", nargs="?", help="Path to dataset root directory")
    parser.add_argument("--output", default="vis_output", help="Output directory")
    parser.add_argument("--sample-idx", type=int, default=0, help="Sample index")
    parser.add_argument("--sample-token", help="Sample token (overrides --sample-idx)")
    parser.add_argument(
        "--version", action="store_true", help="Print version and exit"
    )
    args = parser.parse_args()

    if args.version:
        print(f"visualize_sample v{__version__}")
        return

    if not args.dataset_dir:
        parser.error("dataset_dir is required unless --version is used")

    visualizer = NuScenesVisualizer(args.dataset_dir)

    sample_token = args.sample_token
    if not sample_token:
        sample = visualizer.get_sample(sample_idx=args.sample_idx)
        if sample is None:
            print("No samples found")
            return
        sample_token = sample["token"]

    visualizer.visualize_sample(sample_token, args.output)


if __name__ == "__main__":
    main()
