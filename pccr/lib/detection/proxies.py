"""
Duck-typed proxy objects for CARLA environment objects and signal heads.

CARLA exposes parked vehicles, traffic signs, and individual signal-head
bounding boxes through APIs that do not return first-class ``carla.Actor``
instances.  The proxy dataclasses in this module present a minimal
actor-compatible interface so that the annotation pipeline can treat them
uniformly alongside dynamic actors.

Also contains :class:`CategoryColors` for CARLA debug visualisation and
:class:`StaticObjectFactory` for building proxy lists from the environment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List

import carla

from ..nuscenes_standards import (
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


# ============================================================================
# CategoryColors
# ============================================================================


class CategoryColors:
    """Map nuScenes category tokens to CARLA ``Color`` objects for debug drawing.

    .. note::
       Buses and trucks intentionally share orange; motorcycles and bicycles
       share yellow.  These pairs are visually similar and are unlikely to
       co-occur in busy debug views.
    """

    COLOR_MAP: Dict[str, carla.Color] = {
        CAT_CAR: carla.Color(0, 255, 0),           # Green
        CAT_ADULT: carla.Color(0, 0, 255),          # Blue
        CAT_CHILD: carla.Color(135, 206, 250),      # Light Blue
        CAT_TRUCK: carla.Color(255, 165, 0),        # Orange
        CAT_BUS: carla.Color(255, 140, 0),          # Dark Orange (distinguishable from truck)
        CAT_MOTORCYCLE: carla.Color(255, 255, 0),   # Yellow
        CAT_BICYCLE: carla.Color(200, 220, 0),      # Yellow-green (distinguishable from motor)
        CAT_TRAFFIC_LIGHT: carla.Color(255, 0, 255),
        CAT_TRAFFIC_SIGN: carla.Color(200, 0, 255),
    }

    @staticmethod
    def get_color(category_token: str) -> carla.Color:
        """Return the CARLA debug color for *category_token*.

        Falls back to grey for unknown tokens.

        Args:
            category_token: 32-char hex nuScenes category token.
        """
        return CategoryColors.COLOR_MAP.get(category_token, carla.Color(128, 128, 128))


# ============================================================================
# StaticObjectProxy
# ============================================================================


@dataclass
class StaticObjectProxy:
    """Duck-typed wrapper for a static environment vehicle or sign.

    CARLA exposes parked vehicles and environment signs through
    ``world.get_environment_objects()``, which returns plain data objects
    rather than actor instances.  This dataclass mimics the ``carla.Actor``
    interface subset used by the annotation pipeline.

    Attributes:
        id: Unique environment-object ID.
        type_id: Blueprint-style type string (e.g. ``"static.vehicle.car"``).
        transform: World transform of the object.
        bounding_box: Zero-centred local bounding box.
        source: Provenance tag — ``"environment"`` or ``"env_sign"``.
        attributes: Actor attribute dict (mocked; role set to ``"parked"``).
        is_environment_object: Always ``True``.
    """

    id: int
    type_id: str
    transform: carla.Transform
    bounding_box: carla.BoundingBox
    source: str = "environment"
    attributes: Dict[str, str] = field(default_factory=lambda: {"role_name": "parked"})
    is_environment_object: bool = True

    def get_location(self) -> carla.Location:
        """Return the world-space location."""
        return self.transform.location

    def get_transform(self) -> carla.Transform:
        """Return the world-space transform."""
        return self.transform

    def get_velocity(self) -> carla.Vector3D:
        """Static objects are always stationary."""
        return carla.Vector3D(0.0, 0.0, 0.0)


# ============================================================================
# LightHeadProxy constants
# ============================================================================

#: Visual scale applied to the XY (face) axes of raw CARLA light-box extents
#: so the annotation box better covers the physical signal-head surface.
#: Z (height) is left unscaled — CARLA already reports housing height correctly.
LIGHT_HEAD_VISUAL_SCALE_XY: float = 1.5

#: Minimum full-dimension (largest axis, metres) a light box must have to be
#: kept as a signal-head annotation.  Filters out tiny incidental components
#: such as pedestrian push-buttons (typically < 0.25 m in every axis).
LIGHT_HEAD_MIN_FULL_DIM_M: float = 0.30


# ============================================================================
# LightHeadProxy
# ============================================================================


@dataclass
class LightHeadProxy:
    """Duck-typed proxy for a single signal head of a traffic-light actor.

    Each head returned by ``actor.get_light_boxes()`` is wrapped in its own
    proxy so the annotation pipeline emits one bounding box per signal head
    rather than one merged box per actor pole.

    Attributes:
        id: Unique ID — ``actor.id * 1000 + head_index``.
        type_id: Same as parent actor ``type_id``.
        transform: World transform: location = head centre,
            rotation = actor rotation.
        bounding_box: Zero-centred local bbox (pure extent, no positional
            offset).
        source: Always ``"head_proxy"``.
        attributes: Traffic-light metadata dict.
        is_environment_object: Always ``False``.
    """

    id: int
    type_id: str
    transform: carla.Transform
    bounding_box: carla.BoundingBox
    source: str = "head_proxy"
    attributes: Dict[str, str] = field(default_factory=dict)
    is_environment_object: bool = False

    def get_location(self) -> carla.Location:
        """Return the world-space location of this signal head."""
        return self.transform.location

    def get_transform(self) -> carla.Transform:
        """Return the world-space transform of this signal head."""
        return self.transform

    def get_velocity(self) -> carla.Vector3D:
        """Signal heads are static."""
        return carla.Vector3D(0.0, 0.0, 0.0)


def expand_to_light_head_proxies(actor: carla.Actor) -> List[LightHeadProxy]:
    """Expand a traffic-light actor into per-signal-head :class:`LightHeadProxy` objects.

    Args:
        actor: A CARLA traffic-light actor.

    Returns:
        A list of :class:`LightHeadProxy` instances — one per signal head that
        passes the minimum-dimension filter.  Returns an empty list if the
        actor has no ``get_light_boxes()`` method or if all heads are filtered.
    """
    if not hasattr(actor, "get_light_boxes"):
        return []
    try:
        light_boxes = list(actor.get_light_boxes())
    except Exception:
        return []
    if not light_boxes:
        return []

    actor_rotation = actor.get_transform().rotation
    light_state_name = "unknown"
    try:
        state = actor.get_state()
        state_name = getattr(state, "name", None)
        if state_name is None:
            state_name = str(state).split(".")[-1]
        light_state_name = str(state_name).lower()
    except Exception:
        light_state_name = "unknown"
    proxies: List[LightHeadProxy] = []
    for i, box in enumerate(light_boxes):
        max_full_dim = max(box.extent.x, box.extent.y, box.extent.z) * 2.0
        if max_full_dim < LIGHT_HEAD_MIN_FULL_DIM_M:
            continue
        scaled_extent = carla.Vector3D(
            box.extent.x * LIGHT_HEAD_VISUAL_SCALE_XY,
            box.extent.y * LIGHT_HEAD_VISUAL_SCALE_XY,
            box.extent.z,  # Housing height is accurate; do not inflate.
        )
        proxies.append(
            LightHeadProxy(
                id=actor.id * 1000 + i,
                type_id=actor.type_id,
                transform=carla.Transform(box.location, actor_rotation),
                bounding_box=carla.BoundingBox(
                    carla.Location(0.0, 0.0, 0.0), scaled_extent
                ),
                attributes={"traffic_light_state": light_state_name},
            )
        )
    return proxies


# ============================================================================
# StaticObjectFactory
# ============================================================================


class StaticObjectFactory:
    """Build :class:`StaticObjectProxy` lists from CARLA environment objects.

    Covers two object classes:

    * **Static vehicles** — parked cars/trucks/buses/motorcycles/bicycles
      returned by ``world.get_environment_objects(CityObjectLabel.*)``.
    * **Static signs** — traffic-sign face panels returned by
      ``world.get_environment_objects(CityObjectLabel.TrafficSigns)``,
      de-duplicated against already-spawned sign actors.
    """

    CITY_OBJECT_LABELS: List[tuple] = [
        ("Car", "static.vehicle.car"),
        ("Truck", "static.vehicle.truck"),
        ("Bus", "static.vehicle.bus"),
        ("Motorcycle", "static.vehicle.motorcycle"),
        ("Bicycle", "static.vehicle.bicycle"),
    ]

    #: Blueprint name prefixes of actors that already cover the same world
    #: position as some env-sign objects.  These are skipped to avoid
    #: double-annotation.
    ENV_SIGN_SKIP_PREFIXES: tuple = (
        "BP_SpeedLimit",          # Covered by spawned traffic.speed_limit.* actors.
        "BP_TrafficLightNew",     # Covered by LightHeadProxy annotations.
    )

    _static_vehicle_cache: Dict[str, List[StaticObjectProxy]] = {}
    _static_sign_cache: Dict[str, List[StaticObjectProxy]] = {}

    #: Conservative lower bounds for full vehicle dimensions in metres,
    #: expressed as sorted axes ``(longest, middle, shortest)``. These are
    #: used only for coarse CARLA environment-object labels, which can include
    #: decorative map meshes that are not real vehicles.
    STATIC_VEHICLE_MIN_DIMS: Dict[str, tuple[float, float, float]] = {
        "static.vehicle.car": (2.0, 1.0, 0.8),
        "static.vehicle.truck": (4.0, 1.5, 1.5),
        "static.vehicle.bus": (4.5, 1.8, 1.8),
    }

    STATIC_VEHICLE_MIN_VOLUME_M3: Dict[str, float] = {
        "static.vehicle.car": 2.0,
        "static.vehicle.truck": 8.0,
        "static.vehicle.bus": 10.0,
    }

    @staticmethod
    def _get_world_cache_key(world: carla.World) -> str:
        """Return a stable cache key for a CARLA map."""
        try:
            return world.get_map().name
        except Exception:
            return "<unknown-map>"

    @staticmethod
    def _is_plausible_static_vehicle_bbox(
        type_id: str,
        bounding_box: carla.BoundingBox,
    ) -> bool:
        """Return ``True`` when a coarse environment-object vehicle bbox is plausible.

        CARLA's ``CityObjectLabel`` vehicle groups sometimes include tiny proxy
        meshes or malformed boxes. Those should not become nuScenes vehicle
        annotations.
        """
        min_dims = StaticObjectFactory.STATIC_VEHICLE_MIN_DIMS.get(type_id)
        min_volume = StaticObjectFactory.STATIC_VEHICLE_MIN_VOLUME_M3.get(type_id)
        if min_dims is None or min_volume is None:
            return True

        extent = getattr(bounding_box, "extent", None)
        if extent is None:
            return False

        dims = [abs(float(extent.x)) * 2.0, abs(float(extent.y)) * 2.0, abs(float(extent.z)) * 2.0]
        if any(not math.isfinite(dim) or dim <= 0.0 for dim in dims):
            return False

        sorted_dims = sorted(dims, reverse=True)
        volume = dims[0] * dims[1] * dims[2]
        return all(dim >= limit for dim, limit in zip(sorted_dims, min_dims)) and volume >= min_volume

    @staticmethod
    def get_static_vehicle_objects(world: carla.World) -> List[StaticObjectProxy]:
        """Return static parked vehicles from the CARLA environment.

        Args:
            world: CARLA world instance.

        Returns:
            List of :class:`StaticObjectProxy` instances for the supported
            vehicle categories.  Returns an empty list if the API is
            unavailable.
        """
        if not hasattr(world, "get_environment_objects") or not hasattr(
            carla, "CityObjectLabel"
        ):
            return []

        cache_key = StaticObjectFactory._get_world_cache_key(world)
        cached = StaticObjectFactory._static_vehicle_cache.get(cache_key)
        if cached is not None:
            return cached

        result: List[StaticObjectProxy] = []
        for label_name, type_id in StaticObjectFactory.CITY_OBJECT_LABELS:
            label = getattr(carla.CityObjectLabel, label_name, None)
            if label is None:
                continue
            try:
                env_objects = world.get_environment_objects(label)
            except Exception:
                continue
            for env_obj in env_objects:
                transform = getattr(env_obj, "transform", None)
                bounding_box = getattr(env_obj, "bounding_box", None)
                object_id = getattr(env_obj, "id", None)
                if transform is None or bounding_box is None or object_id is None:
                    continue
                if not StaticObjectFactory._is_plausible_static_vehicle_bbox(
                    type_id, bounding_box
                ):
                    continue
                result.append(
                    StaticObjectProxy(
                        id=int(object_id),
                        type_id=type_id,
                        transform=transform,
                        bounding_box=bounding_box,
                    )
                )
                StaticObjectFactory._static_vehicle_cache[cache_key] = result
        return result

    @staticmethod
    def get_static_sign_objects(world: carla.World) -> List[StaticObjectProxy]:
        """Return static traffic-sign environment objects as annotatable proxies.

        Speed-limit signs are already covered by spawned ``traffic.speed_limit.*``
        actors and are omitted here to avoid double-annotation.  Each returned
        sign has its depth extent padded to a minimum so the 3-D box projects
        correctly onto 2-D images.

        Args:
            world: CARLA world instance.

        Returns:
            List of :class:`StaticObjectProxy` sign instances.
        """
        if not hasattr(world, "get_environment_objects") or not hasattr(
            carla, "CityObjectLabel"
        ):
            return []

        cache_key = StaticObjectFactory._get_world_cache_key(world)
        cached = StaticObjectFactory._static_sign_cache.get(cache_key)
        if cached is not None:
            return cached
        label = getattr(carla.CityObjectLabel, "TrafficSigns", None)
        if label is None:
            return []
        try:
            env_objects = world.get_environment_objects(label)
        except Exception:
            return []

        MIN_SIGN_DEPTH_M = 0.08  # Half-extent → 0.16 m full depth.
        result: List[StaticObjectProxy] = []
        for env_obj in env_objects:
            name = getattr(env_obj, "name", "")
            if any(name.startswith(p) for p in StaticObjectFactory.ENV_SIGN_SKIP_PREFIXES):
                continue
            transform = getattr(env_obj, "transform", None)
            bounding_box = getattr(env_obj, "bounding_box", None)
            object_id = getattr(env_obj, "id", None)
            if transform is None or bounding_box is None or object_id is None:
                continue

            raw_ext = bounding_box.extent
            padded_ext = carla.Vector3D(
                raw_ext.x,
                max(raw_ext.y, MIN_SIGN_DEPTH_M),
                raw_ext.z,
            )
            # bbox.rotation is the actual world-space orientation of the sign
            # face; using transform.rotation alone gives wrong yaw for
            # BP_StreetLight_* assets.
            sign_transform = carla.Transform(bounding_box.location, bounding_box.rotation)
            local_bbox = carla.BoundingBox(carla.Location(0.0, 0.0, 0.0), padded_ext)
            result.append(
                StaticObjectProxy(
                    id=int(object_id),
                    type_id="traffic.traffic_sign",
                    transform=sign_transform,
                    bounding_box=local_bbox,
                    source="env_sign",
                    is_environment_object=False,
                )
            )
        StaticObjectFactory._static_sign_cache[cache_key] = result
        return result
