"""
CARLA actor → nuScenes category token mapping.

``ObjectCategorizer`` is the single authority that translates a CARLA
blueprint identifier (e.g. ``"vehicle.audi.etron"``) or actor type to a
32-char hex nuScenes category token.
"""

from __future__ import annotations

from typing import List, Optional

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


class ObjectCategorizer:
    """Map CARLA actors to nuScenes category tokens.

    The mapping is performed in priority order:

    1. ``traffic.*`` actors are handled by :meth:`_classify_traffic`, which
       inspects blueprint prefixes and component names.
    2. Explicit blueprint ID lookup in :attr:`EXPLICIT_MAP`.
    3. Pedestrian ``walker.pedestrian.*`` actors, classified by age attribute.
    4. Generic fallback using ``type_id`` keyword matching.
    """

    TRAFFIC_LIGHT_PREFIXES = ("traffic.traffic_light",)
    TRAFFIC_SIGN_PREFIXES = (
        "traffic.speed_limit",
        "traffic.stop",
        "traffic.yield",
        "traffic.traffic_sign",
    )

    # Component name keyword sets used to identify sign sub-types.
    STOP_SIGN_COMPONENT_KEYWORDS = ("stopsigncomponent",)
    YIELD_SIGN_COMPONENT_KEYWORDS = ("yieldsigncomponent", "yieldsign")
    ROUND_SIGN_COMPONENT_KEYWORDS = ("roundsign", "speedsign", "speed_limit", "speedlimit")
    RECT_SIGN_COMPONENT_KEYWORDS = ("trafficsign", "warningsign", "rectsign")
    UNKNOWN_SIGN_MESH_KEYWORDS = (
        "staticmesh",
        "signcomponent",
        "roadsigns_",
        "yieldsigncomponent",
        "stopsigncomponent",
    )
    BILLBOARD_SIGN_KEYWORDS = ("billboard", "checkbox", "stopbox", "stopsign")

    #: Explicit blueprint ID → category token mapping for CARLA 0.9.16.
    EXPLICIT_MAP: dict[str, str] = {
        # Bicycles
        "vehicle.bh.crossbike": CAT_BICYCLE,
        "vehicle.diamondback.century": CAT_BICYCLE,
        "vehicle.gazelle.omafiets": CAT_BICYCLE,
        # Motorcycles
        "vehicle.yamaha.yzf": CAT_MOTORCYCLE,
        "vehicle.kawasaki.ninja": CAT_MOTORCYCLE,
        "vehicle.harley-davidson.low_rider": CAT_MOTORCYCLE,
        "vehicle.vespa.zx125": CAT_MOTORCYCLE,
        # Buses
        "vehicle.volkswagen.t2": CAT_BUS,
        "vehicle.volkswagen.t2_2021": CAT_BUS,
        "vehicle.mercedes.sprinter": CAT_BUS,
        "vehicle.mitsubishi.fusorosa": CAT_BUS,
        # Trucks
        "vehicle.ford.ambulance": CAT_TRUCK,
        "vehicle.tesla.cybertruck": CAT_TRUCK,
        "vehicle.carlamotors.european_hgv": CAT_TRUCK,
        "vehicle.carlamotors.firetruck": CAT_TRUCK,
        "vehicle.carlamotors.carlacola": CAT_TRUCK,
        # Cars
        "vehicle.micro.microlino": CAT_CAR,
        "vehicle.chevrolet.impala": CAT_CAR,
        "vehicle.mini.cooper_s": CAT_CAR,
        "vehicle.mercedes.coupe": CAT_CAR,
        "vehicle.dodge.charger_police": CAT_CAR,
        "vehicle.nissan.patrol": CAT_CAR,
        "vehicle.seat.leon": CAT_CAR,
        "vehicle.toyota.prius": CAT_CAR,
        "vehicle.lincoln.mkz_2017": CAT_CAR,
        "vehicle.tesla.model3": CAT_CAR,
        "vehicle.ford.mustang": CAT_CAR,
        "vehicle.dodge.charger_2020": CAT_CAR,
        "vehicle.mercedes.coupe_2020": CAT_CAR,
        "vehicle.bmw.grandtourer": CAT_CAR,
        "vehicle.ford.crown": CAT_CAR,
        "vehicle.nissan.micra": CAT_CAR,
        "vehicle.lincoln.mkz_2020": CAT_CAR,
        "vehicle.mini.cooper_s_2021": CAT_CAR,
        "vehicle.citroen.c3": CAT_CAR,
        "vehicle.jeep.wrangler_rubicon": CAT_CAR,
        "vehicle.dodge.charger_police_2020": CAT_CAR,
        "vehicle.nissan.patrol_2021": CAT_CAR,
        "vehicle.audi.a2": CAT_CAR,
        "vehicle.audi.etron": CAT_CAR,
        # Traffic infrastructure
        "traffic.traffic_light": CAT_TRAFFIC_LIGHT,
        "traffic.speed_limit": CAT_TRAFFIC_SIGN,
        "traffic.stop": CAT_TRAFFIC_SIGN,
        "traffic.yield": CAT_TRAFFIC_SIGN,
        "traffic.traffic_sign": CAT_TRAFFIC_SIGN,
    }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def get_category(actor: carla.Actor) -> Optional[str]:
        """Return the nuScenes category token for *actor*, or ``None``.

        Args:
            actor: A CARLA actor (vehicle, walker, traffic object, or env proxy).

        Returns:
            32-char hex category token, or ``None`` if unrecognised.
        """
        type_id = actor.type_id.lower()

        if type_id.startswith("traffic."):
            return ObjectCategorizer._classify_traffic(actor)

        if type_id in ObjectCategorizer.EXPLICIT_MAP:
            return ObjectCategorizer.EXPLICIT_MAP[type_id]

        if type_id.startswith("walker.pedestrian"):
            age = actor.attributes.get("age", "adult").lower()
            return CAT_CHILD if age in ("child", "teenager") else CAT_ADULT

        if type_id.startswith("vehicle."):
            if "truck" in type_id:
                return CAT_TRUCK
            if "bus" in type_id:
                return CAT_BUS
            if "motorcycle" in type_id:
                return CAT_MOTORCYCLE
            if "bicycle" in type_id:
                return CAT_BICYCLE
            return CAT_CAR

        if type_id.startswith("static.vehicle."):
            if "truck" in type_id:
                return CAT_TRUCK
            if "bus" in type_id:
                return CAT_BUS
            if "motorcycle" in type_id:
                return CAT_MOTORCYCLE
            if "bicycle" in type_id:
                return CAT_BICYCLE
            return CAT_CAR

        return None

    # ------------------------------------------------------------------
    # Component-name helpers (shared with bbox/infra_bbox.py)
    # ------------------------------------------------------------------

    @staticmethod
    def get_component_names(actor: carla.Actor) -> List[str]:
        """Return lower-cased component names for *actor* (empty list on error).

        Args:
            actor: Any CARLA actor that may expose ``get_component_names()``.
        """
        if not hasattr(actor, "get_component_names"):
            return []
        try:
            return [name.lower() for name in actor.get_component_names()]
        except Exception:
            return []

    @staticmethod
    def has_component_keyword(component_names: List[str], keywords: tuple) -> bool:
        """Return ``True`` if any component name contains any of *keywords*.

        Args:
            component_names: Lower-cased component names from :meth:`get_component_names`.
            keywords: Tuple of keyword strings to test for substring membership.
        """
        return any(kw in name for name in component_names for kw in keywords)

    @staticmethod
    def is_small_billboard_sign(actor: carla.Actor, component_names: List[str]) -> bool:
        """Return ``True`` if *actor* looks like a small billboard/box sign.

        Filters out arrow-signs (which are road markings, not annotation targets)
        and oversized billboard panels.

        Args:
            actor: CARLA actor.
            component_names: Pre-fetched lower-cased component names.
        """
        if not ObjectCategorizer.has_component_keyword(
            component_names, ObjectCategorizer.BILLBOARD_SIGN_KEYWORDS
        ):
            return False
        if any("arrow" in name for name in component_names):
            return False

        bbox = actor.bounding_box
        full_x = bbox.extent.x * 2.0
        full_y = bbox.extent.y * 2.0
        full_z = bbox.extent.z * 2.0
        max_h = max(full_x, full_y)
        min_h = min(full_x, full_y)
        return max_h <= 3.5 and min_h <= 1.0 and full_z <= 1.25

    # ------------------------------------------------------------------
    # Internal traffic classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_traffic(actor: carla.Actor) -> Optional[str]:
        """Classify a ``traffic.*`` actor into either a light or sign token.

        Traffic light actors are expanded into per-head proxies upstream; this
        method just confirms the category.  Signs require component-name
        inspection because not all ``traffic.stop`` actors are annotatable
        signs.

        Args:
            actor: A CARLA actor whose ``type_id`` starts with ``"traffic."``.

        Returns:
            Category token, or ``None`` if the actor should not be annotated.
        """
        type_id = actor.type_id.lower()
        comp_names = ObjectCategorizer.get_component_names(actor)

        if any(type_id.startswith(p) for p in ObjectCategorizer.TRAFFIC_LIGHT_PREFIXES):
            return CAT_TRAFFIC_LIGHT

        if type_id.startswith("traffic.stop"):
            if ObjectCategorizer.has_component_keyword(
                comp_names, ObjectCategorizer.STOP_SIGN_COMPONENT_KEYWORDS
            ):
                return CAT_TRAFFIC_SIGN
            if ObjectCategorizer.is_small_billboard_sign(actor, comp_names):
                return CAT_TRAFFIC_SIGN
            return None

        if type_id.startswith("traffic.yield"):
            if comp_names and not ObjectCategorizer.has_component_keyword(
                comp_names, ObjectCategorizer.YIELD_SIGN_COMPONENT_KEYWORDS
            ):
                if ObjectCategorizer.is_small_billboard_sign(actor, comp_names):
                    return CAT_TRAFFIC_SIGN
                return None
            return CAT_TRAFFIC_SIGN

        if type_id.startswith("traffic.speed_limit"):
            return CAT_TRAFFIC_SIGN

        if type_id.startswith("traffic.traffic_sign"):
            return CAT_TRAFFIC_SIGN

        if type_id.startswith("traffic.unknown"):
            if ObjectCategorizer.is_small_billboard_sign(actor, comp_names):
                return CAT_TRAFFIC_SIGN
            if not ObjectCategorizer.has_component_keyword(
                comp_names, ObjectCategorizer.UNKNOWN_SIGN_MESH_KEYWORDS
            ):
                return None
            for kws in (
                ObjectCategorizer.STOP_SIGN_COMPONENT_KEYWORDS,
                ObjectCategorizer.YIELD_SIGN_COMPONENT_KEYWORDS,
                ObjectCategorizer.ROUND_SIGN_COMPONENT_KEYWORDS,
                ObjectCategorizer.RECT_SIGN_COMPONENT_KEYWORDS,
            ):
                if ObjectCategorizer.has_component_keyword(comp_names, kws):
                    return CAT_TRAFFIC_SIGN
            return None

        if any(type_id.startswith(p) for p in ObjectCategorizer.TRAFFIC_SIGN_PREFIXES):
            return CAT_TRAFFIC_SIGN

        return None
