"""Visibility calculation via per-camera image-space ray-box depth comparison.

For each object the 8 corners of its 3D bounding box are projected into
every camera view to obtain a 2-D bounding rectangle.  Inside that
rectangle, each pixel's camera ray is intersected with the *oriented*
bounding box (OBB).  Only pixels whose ray actually hits the box are
considered; for those, the observed depth is compared against the ray's
per-pixel entry/exit depth range.

Each ray-hit pixel is classified as one of:
  - **surface**: depth within [t_enter−tol, t_exit+tol] → object mesh.
  - **occluded**: depth < t_enter−tol → something blocks the object.
  - **see-through**: depth > t_exit+tol → background visible through
    mesh gaps (motorcycle spokes, traffic-light slots, etc.).

    visibility_ratio = surface / (surface + occluded)

See-through pixels are excluded from both numerator and denominator so
that sparse/hollow objects (traffic lights, bicycles, motorcycles) are
not penalised for air gaps inside their bounding box.

The best ratio across all cameras is taken, then mapped to a nuScenes
visibility level token.
"""

import logging
from typing import Dict, Optional, Any

import numpy as np
import carla

from ..nuscenes_standards import NuScenesStandards


logger = logging.getLogger('scenario_runner')

# Maximum per-pixel depth tolerance.  Actual tolerance is adaptive,
# scaling with the ray's depth span through the box (see step 6).
DEPTH_TOLERANCE_MAX_M = 0.25

# Minimum per-pixel depth tolerance (floor for very thin objects).
DEPTH_TOLERANCE_MIN_M = 0.05

# Fraction of per-ray box span used as tolerance.
DEPTH_TOLERANCE_FRAC = 0.08

# Ignore depth pixels below this value (degenerate / no-data).
MIN_VALID_DEPTH_M = 0.01

# Projected bounding-rectangle must cover at least this many pixels,
# otherwise the camera is considered to not see the object.
MIN_PROJECTED_AREA_PX = 4

# Projected area (pixels) below which visibility is linearly scaled down.
# Acts as a natural distance proxy: far objects have small projections.
MIN_CONFIDENT_AREA = 50

# Cap the number of ray tests per object per camera to keep frame times
# reasonable for very large / close objects.
MAX_RAY_PIXELS = 10_000


class VisibilityCalculator:
    """Per-actor visibility from per-pixel ray-box depth comparison."""

    def __init__(self, world: carla.World, ego_vehicle: carla.Actor):
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.ego_vehicle_id = ego_vehicle.id
        self.visibility_levels = NuScenesStandards.get_visibility()

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def calculate_visibility(
        self,
        target_actor: carla.Actor,
        camera_depth_data: Optional[Dict[str, Dict[str, Any]]] = None,
        category_token: Optional[str] = None,
        bbox_extent: Optional[carla.Vector3D] = None,
        bbox_location: Optional[carla.Location] = None,
        bbox_transform: Optional[carla.Transform] = None,
    ) -> str:
        """Return a nuScenes visibility-level token for *target_actor*.

        Parameters
        ----------
        target_actor:
            CARLA actor (or proxy) whose bounding box is being tested.
        camera_depth_data:
            ``{camera_id: {depth_map, transform, fx, fy, cx, cy, width, height}}``
            where *depth_map* is an ``(H, W)`` float32 array in metres and
            *transform* is the ``carla.Transform`` of the depth sensor.
        category_token:
            nuScenes category token (kept for API compatibility).
        bbox_extent / bbox_location / bbox_transform:
            Override the actor's bounding box geometry.  When supplied the
            box is defined by *bbox_transform* (world pose of the box
            centre) and *bbox_extent* (half-sizes).
        """
        if not camera_depth_data:
            logger.debug("Visibility actor=%s — no depth data → 0", target_actor.id)
            return "0"

        bbox_transform = bbox_transform or target_actor.get_transform()
        if bbox_extent is None:
            bbox_extent = target_actor.bounding_box.extent

        # 8 corners in world space  (8×3)
        corners_world = self._box_corners_world(bbox_transform, bbox_extent)

        best_ratio = 0.0
        for cam_id, cam in camera_depth_data.items():
            ratio = self._camera_visibility(
                corners_world,
                bbox_transform,
                bbox_extent,
                cam["depth_map"],
                cam["transform"],
                cam["fx"], cam["fy"],
                cam["cx"], cam["cy"],
                cam["width"], cam["height"],
            )
            if ratio > best_ratio:
                best_ratio = ratio

        token = self._ratio_to_visibility_token(best_ratio)
        logger.debug(
            "Visibility actor=%s best_ratio=%.2f token=%s",
            target_actor.id, best_ratio, token,
        )
        return token

    # ------------------------------------------------------------------ #
    # Geometry helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _box_corners_world(
        transform: carla.Transform, extent: carla.Vector3D
    ) -> np.ndarray:
        """Return the 8 world-space corners of the oriented bounding box.

        Returns
        -------
        np.ndarray  (8, 3) float64
        """
        dx, dy, dz = extent.x, extent.y, extent.z
        corners = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    loc = carla.Location(x=sx * dx, y=sy * dy, z=sz * dz)
                    w = transform.transform(loc)
                    corners.append([w.x, w.y, w.z])
        return np.array(corners, dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Per-camera ray-box visibility                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _camera_visibility(
        corners_world: np.ndarray,
        bbox_transform: carla.Transform,
        bbox_extent: carla.Vector3D,
        depth_map: np.ndarray,
        sensor_transform: carla.Transform,
        fx: float, fy: float,
        cx: float, cy: float,
        width: int, height: int,
    ) -> float:
        """Compute the visibility ratio for one camera.

        Returns a value in [0, 1], where 0 means fully occluded or
        out-of-view and 1 means fully visible.

        Algorithm
        ---------
        1.  Transform the 8 box corners to pixel coordinates via the
            camera transform, discarding corners behind the lens.
        2.  Compute the tight 2-D bounding rectangle, clipped to image
            bounds.
        3.  For each pixel in the rectangle, cast a ray in camera space,
            transform it into the box-local frame, and perform a
            ray-AABB intersection test.
        4.  For rays that hit, classify each pixel:
            * **surface** — depth in ``[t_enter − tol, t_exit + tol]``
            * **occluded** — depth < ``t_enter − tol``
            * **see-through** — depth > ``t_exit + tol`` (air gap)
        5.  ``visibility = surface / (surface + occluded)``
            (see-through pixels are excluded from the ratio).
        """
        # ---- 1. Project corners to find 2D pixel rect ----
        cam_matrix = np.array(sensor_transform.get_matrix(), dtype=np.float64)
        cam_inv = np.linalg.inv(cam_matrix)
        ones_c = np.ones((corners_world.shape[0], 1), dtype=np.float64)
        hom = np.hstack((corners_world, ones_c))
        corners_cam = (cam_inv @ hom.T).T[:, :3]  # CARLA sensor frame

        # CARLA sensor → optical
        z_opt = corners_cam[:, 0]   # X_car → depth
        x_opt = corners_cam[:, 1]   # Y_car → right
        y_opt = -corners_cam[:, 2]  # -Z_car → down

        in_front = z_opt > 0.1
        if not np.any(in_front):
            return 0.0

        # Project visible corners → pixel coords
        u_proj = fx * x_opt[in_front] / z_opt[in_front] + cx
        v_proj = fy * y_opt[in_front] / z_opt[in_front] + cy

        u_min = max(int(np.floor(u_proj.min())), 0)
        u_max = min(int(np.ceil(u_proj.max())), width - 1)
        v_min = max(int(np.floor(v_proj.min())), 0)
        v_max = min(int(np.ceil(v_proj.max())), height - 1)

        rect_w = u_max - u_min + 1
        rect_h = v_max - v_min + 1
        if rect_w * rect_h < MIN_PROJECTED_AREA_PX:
            return 0.0

        # ---- 2. Build transforms for ray-box intersection ----
        box_matrix = np.array(bbox_transform.get_matrix(), dtype=np.float64)
        box_inv = np.linalg.inv(box_matrix)

        # Camera origin in box-local frame
        cam_origin_world = cam_matrix[:3, 3]
        cam_origin_box = (box_inv @ np.append(cam_origin_world, 1.0))[:3]

        # Rotation matrices
        cam_rot = cam_matrix[:3, :3]      # sensor-local → world
        box_rot_inv = box_inv[:3, :3]     # world → box-local

        ex = float(bbox_extent.x)
        ey = float(bbox_extent.y)
        ez = float(bbox_extent.z)
        extents = np.array([ex, ey, ez], dtype=np.float64)

        # ---- 3. Generate pixel grid (subsample if too large) ----
        us = np.arange(u_min, u_max + 1, dtype=np.float64)
        vs = np.arange(v_min, v_max + 1, dtype=np.float64)
        n_full = len(us) * len(vs)
        if n_full > MAX_RAY_PIXELS:
            step = int(np.ceil(np.sqrt(n_full / MAX_RAY_PIXELS)))
            us = us[::step]
            vs = vs[::step]

        uu, vv = np.meshgrid(us, vs)
        uu_flat = uu.ravel()
        vv_flat = vv.ravel()
        n_px = len(uu_flat)

        # ---- 4. Ray directions: pixel → CARLA local → world → box local ----
        # Optical direction for pixel (u,v): ((u-cx)/fx, (v-cy)/fy, 1.0)
        # CARLA local direction: (d_z_opt, d_x_opt, -d_y_opt)
        #                      = (1.0,     (u-cx)/fx, -(v-cy)/fy)
        d_car = np.empty((n_px, 3), dtype=np.float64)
        d_car[:, 0] = 1.0
        d_car[:, 1] = (uu_flat - cx) / fx
        d_car[:, 2] = -(vv_flat - cy) / fy

        d_world = (cam_rot @ d_car.T).T          # (N, 3)
        d_box = (box_rot_inv @ d_world.T).T      # (N, 3)

        # ---- 5. Vectorised ray-AABB intersection ----
        # Ray: P(t) = cam_origin_box + t * d_box
        # AABB: [-ex, ex] × [-ey, ey] × [-ez, ez]
        EPS = 1e-12
        inv_d = np.where(np.abs(d_box) > EPS, 1.0 / d_box, 1e12)

        t1 = (-extents - cam_origin_box) * inv_d  # (N, 3)
        t2 = (extents - cam_origin_box) * inv_d   # (N, 3)

        t_near = np.minimum(t1, t2)
        t_far = np.maximum(t1, t2)

        t_enter = np.max(t_near, axis=1)   # (N,)
        t_exit = np.min(t_far, axis=1)     # (N,)

        hit = (t_enter <= t_exit + EPS) & (t_exit > 0.0)
        total_hit = int(np.count_nonzero(hit))
        if total_hit == 0:
            return 0.0

        # ---- 6. Depth comparison for hit pixels ----
        # In optical space, depth along forward axis = t  (because
        # d_car[0] = 1 for every ray, so z_opt(t) = t).
        t_enter_hit = np.maximum(t_enter[hit], MIN_VALID_DEPTH_M)
        t_exit_hit = t_exit[hit]

        rows = vv_flat[hit].astype(np.intp)
        cols = uu_flat[hit].astype(np.intp)
        observed = depth_map[rows, cols]

        valid = observed >= MIN_VALID_DEPTH_M

        # Per-pixel adaptive tolerance.  Scales with the ray's depth span
        # through the box so that thin objects (signs) get tight matching
        # while deep boxes (cars) allow for mesh roughness.
        ray_span = t_exit_hit - t_enter_hit
        per_ray_tol = np.clip(
            DEPTH_TOLERANCE_FRAC * ray_span + DEPTH_TOLERANCE_MIN_M,
            DEPTH_TOLERANCE_MIN_M,
            DEPTH_TOLERANCE_MAX_M,
        )

        # Surface: depth falls within the box depth range (object mesh hit)
        surface = (
            valid &
            (observed >= t_enter_hit - per_ray_tol) &
            (observed <= t_exit_hit + per_ray_tol)
        )

        # Occluded: depth is closer than box entry (something blocks it)
        occluded = valid & (observed < t_enter_hit - per_ray_tol)

        # See-through: depth is farther than box exit (background through
        # mesh gaps — e.g. air between motorcycle spokes, traffic-light
        # heads).  These are NOT counted against the object.

        surface_count = int(np.count_nonzero(surface))
        occluded_count = int(np.count_nonzero(occluded))

        denominator = surface_count + occluded_count
        if denominator == 0:
            # All rays pass through (no mesh surface, no occluder) →
            # object is not visible from this camera.
            return 0.0

        ratio = surface_count / denominator

        # Distance-aware confidence: objects with very small projections
        # get a linearly reduced score because (a) the pixel-level
        # visibility estimate is noisy and (b) the object is barely
        # distinguishable in the image at that distance.
        projected_area = rect_w * rect_h
        if projected_area < MIN_CONFIDENT_AREA:
            ratio *= projected_area / MIN_CONFIDENT_AREA

        return ratio

    # ------------------------------------------------------------------ #
    # Visibility-level mapping                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _ratio_to_visibility_token(ratio: float) -> str:
        """Map a [0, 1] visibility ratio to a nuScenes visibility token.

        Bins (from ``nuscenes_standards.get_visibility``):
          "0" →  0– 5 %   (filtered out)
          "1" →  5–40 %
          "2" → 40–60 %
          "3" → 60–80 %
          "4" → 80–100 %
        """
        if ratio < 0.05:
            return "0"
        if ratio < 0.40:
            return "1"
        if ratio < 0.60:
            return "2"
        if ratio < 0.80:
            return "3"
        return "4"