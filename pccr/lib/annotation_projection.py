"""
Project nuScenes-style 3D annotations onto calibrated camera images.

Used by the annotation debug tool and by offline visualization scripts to
render coloured bounding-box wireframes directly on top of saved JPEG frames.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .utils.image_utils import ImageProcessor
from .nuscenes_standards import (
    CAT_ADULT,
    CAT_BICYCLE,
    CAT_BUS,
    CAT_CAR,
    CAT_CHILD,
    CAT_MOTORCYCLE,
    CAT_TRAFFIC_LIGHT,
    CAT_TRAFFIC_SIGN,
    CAT_TRUCK,
    NuScenesStandards,
)
from .quaternion import Quaternion


class AnnotationProjector:
    """Project 3D annotation boxes onto a calibrated camera image.

    All methods are class methods or static methods — instantiation is not
    required.

    Color map is intentionally distinct per category so that crowded scenes
    remain readable.
    """

    COLOR_MAP: Dict[str, Tuple[int, int, int]] = {
        CAT_CAR: (0, 255, 0),
        CAT_TRUCK: (255, 165, 0),
        CAT_BUS: (255, 127, 80),
        CAT_MOTORCYCLE: (255, 255, 0),
        CAT_BICYCLE: (220, 20, 60),
        CAT_ADULT: (0, 0, 255),
        CAT_CHILD: (0, 191, 255),
        CAT_TRAFFIC_SIGN: (255, 0, 255),
        CAT_TRAFFIC_LIGHT: (200, 0, 255),
    }

    #: 12 edges of a unit bounding box defined as pairs of corner indices.
    EDGES: List[Tuple[int, int]] = [
        (4, 5), (5, 6), (6, 7), (7, 4),  # bottom face
        (0, 1), (1, 2), (2, 3), (3, 0),  # top face
        (0, 4), (1, 5), (2, 6), (3, 7),  # vertical pillars
    ]

    CATEGORY_NAME_BY_TOKEN: Dict[str, str] = {
        cat["token"]: cat["name"] for cat in NuScenesStandards.get_categories()
    }

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def transform_matrix(translation: List[float], rotation: List[float]) -> np.ndarray:
        """Build a 4×4 homogeneous transform from translation and quaternion.

        Args:
            translation: ``[x, y, z]`` in metres.
            rotation: Quaternion ``[w, x, y, z]``.

        Returns:
            4×4 float32 homogeneous transform matrix.
        """
        quat = Quaternion(array=rotation)
        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = quat.rotation_matrix
        mat[:3, 3] = np.array(translation, dtype=np.float32)
        return mat

    @staticmethod
    def project_point(
        point_3d: np.ndarray, intrinsic: np.ndarray
    ) -> Optional[np.ndarray]:
        """Project a camera-space 3-D point to 2-D image coordinates.

        Args:
            point_3d: ``(3,)`` array in camera space.  Points at or behind the
                camera (z ≤ 0) return ``None``.
            intrinsic: ``(3, 3)`` camera intrinsic matrix.

        Returns:
            2-D pixel coordinate ``(u, v)`` as a float array, or ``None``.
        """
        if point_3d[2] <= 0:
            return None
        point_2d = intrinsic @ point_3d
        point_2d /= point_2d[2]
        return point_2d[:2]

    @staticmethod
    def clip_line(
        point_a: np.ndarray, point_b: np.ndarray, near_clip: float = 0.1
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Clip a 3-D segment against the camera near plane (z = *near_clip*).

        Returns the (possibly shortened) segment or ``None`` if it lies entirely
        behind the near plane.

        Args:
            point_a: First endpoint in camera space (shape ``(3,)``).
            point_b: Second endpoint in camera space (shape ``(3,)``).
            near_clip: Near-plane z value (metres).
        """
        z_a, z_b = point_a[2], point_b[2]
        if z_a < near_clip and z_b < near_clip:
            return None
        if z_a >= near_clip and z_b >= near_clip:
            return point_a, point_b
        t = (near_clip - z_a) / (z_b - z_a)
        clip_pt = point_a + t * (point_b - point_a)
        return (clip_pt, point_b) if z_a < near_clip else (point_a, clip_pt)

    # ------------------------------------------------------------------
    # Core projection
    # ------------------------------------------------------------------

    @classmethod
    def get_box_corners_cam(
        cls, annotation: Dict[str, object], world_to_cam: np.ndarray
    ) -> np.ndarray:
        """Return annotation box corners in camera coordinates.

        Args:
            annotation: nuScenes ``sample_annotation`` record with ``size``
                ``[width, length, height]`` and ``translation`` / ``rotation``.
            world_to_cam: 4×4 world-to-camera transform.

        Returns:
            Array of shape ``(8, 3)`` — one row per corner in camera space.
        """
        width, length, height = annotation["size"]
        hl, hw, hh = length / 2, width / 2, height / 2

        x_c = [hl, hl, -hl, -hl, hl, hl, -hl, -hl]
        y_c = [hw, -hw, -hw, hw, hw, -hw, -hw, hw]
        z_c = [hh, hh, hh, hh, -hh, -hh, -hh, -hh]

        corners = np.vstack([x_c, y_c, z_c]).astype(np.float32)
        quat = Quaternion(array=annotation["rotation"])
        corners = quat.rotation_matrix @ corners
        t = np.array(annotation["translation"], dtype=np.float32)
        corners[0] += t[0]
        corners[1] += t[1]
        corners[2] += t[2]

        corners_hom = np.vstack([corners, np.ones((1, 8), dtype=np.float32)])
        return (world_to_cam @ corners_hom)[:3].T  # (8, 3)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    @classmethod
    def draw_annotations(
        cls,
        image: Image.Image,
        annotations: List[Dict[str, object]],
        ego_pose: Dict[str, object],
        calibrated_sensor: Dict[str, object],
        instance_by_token: Optional[Dict[str, Dict[str, object]]] = None,
        draw_labels: bool = True,
        line_width: int = 3,
    ) -> Image.Image:
        """Render projected 3-D bounding boxes onto *image* in-place.

        Args:
            image: PIL RGB image to annotate (modified in-place and returned).
            annotations: List of nuScenes ``sample_annotation`` records.
            ego_pose: nuScenes ``ego_pose`` record for this frame.
            calibrated_sensor: nuScenes ``calibrated_sensor`` record.
            instance_by_token: Optional mapping of instance token → instance
                record (used to look up the category token for colouring).
            draw_labels: When ``True`` draw category name + visibility next to
                each box.
            line_width: Pixel width for wireframe lines.

        Returns:
            The annotated PIL image.
        """
        if instance_by_token is None:
            instance_by_token = {}

        global_to_ego = np.linalg.inv(
            cls.transform_matrix(ego_pose["translation"], ego_pose["rotation"])
        )
        ego_to_sensor = np.linalg.inv(
            cls.transform_matrix(
                calibrated_sensor["translation"], calibrated_sensor["rotation"]
            )
        )
        world_to_cam = ego_to_sensor @ global_to_ego
        intrinsic = np.array(calibrated_sensor["camera_intrinsic"], dtype=np.float32)

        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        for ann in annotations:
            instance = instance_by_token.get(ann["instance_token"], {})
            cat_token = instance.get("category_token", "")
            color = cls.COLOR_MAP.get(cat_token, (255, 0, 0))
            corners_cam = cls.get_box_corners_cam(ann, world_to_cam)

            proj_pts: List[np.ndarray] = []
            for start_idx, end_idx in cls.EDGES:
                clipped = cls.clip_line(corners_cam[start_idx], corners_cam[end_idx])
                if clipped is None:
                    continue
                p_start_2d = cls.project_point(clipped[0], intrinsic)
                p_end_2d = cls.project_point(clipped[1], intrinsic)
                if p_start_2d is None or p_end_2d is None:
                    continue
                proj_pts.extend([p_start_2d, p_end_2d])
                draw.line(
                    [
                        (float(p_start_2d[0]), float(p_start_2d[1])),
                        (float(p_end_2d[0]), float(p_end_2d[1])),
                    ],
                    fill=color,
                    width=line_width,
                )

            if draw_labels and proj_pts:
                anchor = min(proj_pts, key=lambda p: p[1])
                cat_name = cls.CATEGORY_NAME_BY_TOKEN.get(
                    cat_token, cat_token[:8] if cat_token else "unknown"
                )
                vis_token = ann.get("visibility_token", "?")
                draw.text(
                    (float(anchor[0]), float(anchor[1]) - 12),
                    f"{cat_name} v{vis_token}",
                    fill=color,
                    font=font,
                )

        return image

    @classmethod
    def render_carla_capture(
        cls,
        carla_image: object,
        annotations: List[Dict[str, object]],
        ego_pose: Dict[str, object],
        calibrated_sensor: Dict[str, object],
        instance_by_token: Optional[Dict[str, Dict[str, object]]] = None,
        draw_labels: bool = True,
        line_width: int = 3,
    ) -> Image.Image:
        """Convert a CARLA image to PIL and overlay projected annotations.

        Convenience wrapper around
        :meth:`draw_annotations` that accepts a raw CARLA image object.

        Args:
            carla_image: Raw CARLA camera image.
            annotations: nuScenes ``sample_annotation`` records.
            ego_pose: nuScenes ``ego_pose`` record.
            calibrated_sensor: nuScenes ``calibrated_sensor`` record.
            instance_by_token: Optional instance lookup dict.
            draw_labels: Draw labels when ``True``.
            line_width: Line thickness in pixels.

        Returns:
            Annotated PIL image.
        """
        image = ImageProcessor.carla_image_to_pil(carla_image)
        return cls.draw_annotations(
            image,
            annotations,
            ego_pose,
            calibrated_sensor,
            instance_by_token=instance_by_token,
            draw_labels=draw_labels,
            line_width=line_width,
        )
