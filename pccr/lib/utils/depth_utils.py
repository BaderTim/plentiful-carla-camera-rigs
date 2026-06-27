"""Depth processing utilities for CARLA sensors."""

from __future__ import annotations

import numpy as np
from typing import Dict
import carla

from .coordinate_utils import CoordinateConverter


class DepthPointCloudAssembler:
    """Convert CARLA depth images into ego/world-frame point clouds."""

    def __init__(self, max_depth_m: float = 80.0, min_depth_m: float = 0.01) -> None:
        self.max_depth_m = max_depth_m
        self.min_depth_m = min_depth_m
        self._camera_models: Dict[str, Dict[str, np.ndarray]] = {}

    def register_camera(self, camera_id: str, width: int, height: int, fov: float) -> None:
        intrinsic = CoordinateConverter.compute_camera_intrinsic(width, height, fov)
        fx = intrinsic[0][0]
        fy = intrinsic[1][1]
        cx = intrinsic[0][2]
        cy = intrinsic[1][2]

        u = (np.arange(width, dtype=np.float32) - cx) / fx
        v = (np.arange(height, dtype=np.float32) - cy) / fy
        grid_u, grid_v = np.meshgrid(u, v)

        self._camera_models[camera_id] = {
            "grid_u": grid_u.reshape(-1),
            "grid_v": grid_v.reshape(-1),
            "width": width,
            "height": height,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
        }

    def has_camera(self, camera_id: str) -> bool:
        return camera_id in self._camera_models

    def get_camera_intrinsics(self, camera_id: str) -> Dict:
        """Return stored intrinsic parameters for a registered camera."""
        model = self._camera_models[camera_id]
        return {
            "width": model["width"],
            "height": model["height"],
            "fx": model["fx"],
            "fy": model["fy"],
            "cx": model["cx"],
            "cy": model["cy"],
        }

    def decode_depth_map(self, depth_image: carla.Image) -> np.ndarray:
        """Decode a CARLA depth image into a 2-D float32 depth map (H×W) in metres."""
        return self._decode_depth(depth_image)

    def depth_image_to_world(self, camera_id: str, depth_image: carla.Image, sensor_transform: carla.Transform) -> np.ndarray:
        if camera_id not in self._camera_models:
            return np.empty((0, 3), dtype=np.float32)

        depth_map = self._decode_depth(depth_image)
        model = self._camera_models[camera_id]
        depth_flat = depth_map.reshape(-1)

        mask = (depth_flat >= self.min_depth_m) & (depth_flat <= self.max_depth_m)
        if not np.any(mask):
            return np.empty((0, 3), dtype=np.float32)

        depth_filtered = depth_flat[mask]
        grid_u = model["grid_u"][mask]
        grid_v = model["grid_v"][mask]

        x = depth_filtered * grid_u
        y = depth_filtered * grid_v
        z = depth_filtered

        points_optical = np.stack((x, y, z), axis=1)
        points_local = self._optical_to_carla(points_optical)
        return self._apply_transform(points_local, sensor_transform)

    def _decode_depth(self, depth_image: carla.Image) -> np.ndarray:
        array = np.frombuffer(depth_image.raw_data, dtype=np.uint8)
        array = array.reshape((depth_image.height, depth_image.width, 4))

        b = array[:, :, 0].astype(np.float32)
        g = array[:, :, 1].astype(np.float32)
        r = array[:, :, 2].astype(np.float32)

        # CARLA depth sensor encodes depth in range [0, 1000m]
        # See: https://carla.readthedocs.io/en/latest/ref_sensors/#depth-camera
        normalized = (r + (g * 256.0) + (b * 65536.0)) / 16777215.0
        depth_m = normalized * 1000.0  # CARLA uses fixed 1000m range
        return depth_m.astype(np.float32)

    @staticmethod
    def _optical_to_carla(points_optical: np.ndarray) -> np.ndarray:
        if points_optical.size == 0:
            return points_optical

        x_opt = points_optical[:, 0]
        y_opt = points_optical[:, 1]
        z_opt = points_optical[:, 2]

        x_car = z_opt
        y_car = x_opt
        z_car = -y_opt

        return np.stack((x_car, y_car, z_car), axis=1).astype(np.float32)

    @staticmethod
    def _apply_transform(points: np.ndarray, transform: carla.Transform) -> np.ndarray:
        if points.size == 0:
            return points

        matrix = np.array(transform.get_matrix(), dtype=np.float32)
        ones = np.ones((points.shape[0], 1), dtype=np.float32)
        hom = np.hstack((points, ones))
        world = hom @ matrix.T
        return world[:, :3]
