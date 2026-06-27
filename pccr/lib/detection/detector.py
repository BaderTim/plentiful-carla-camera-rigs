"""
Scene object detection: query the CARLA world and return typed results.

:class:`ObjectDetector` is the primary entry point for the annotation
pipeline.  It combines dynamic actor querying, static environment object
enumeration, deduplication, and category filtering into a single API call.
"""

from __future__ import annotations

import math
from typing import Dict, List

import carla

from ..nuscenes_standards import CAT_TRAFFIC_LIGHT, CAT_TRAFFIC_SIGN
from .categorizer import ObjectCategorizer
from .proxies import (
    LightHeadProxy,
    StaticObjectFactory,
    StaticObjectProxy,
    expand_to_light_head_proxies,
)


class DetectedObject:
    """Container for a detected scene object with its metadata.

    Attributes:
        actor: The underlying CARLA actor or proxy object.
        category_token: nuScenes category token (32-char hex).
        distance: Distance from the ego vehicle in metres.
        source: Provenance tag (``"actor"``, ``"environment"``, ``"env_sign"``,
            ``"head_proxy"``).
    """

    def __init__(
        self,
        actor: carla.Actor,
        category_token: str,
        distance: float,
        source: str = "actor",
    ) -> None:
        self.actor = actor
        self.category_token = category_token
        self.distance = distance
        self.source = source

    def __repr__(self) -> str:
        return (
            f"DetectedObject({self.category_token}, {self.distance:.1f}m, "
            f"id={self.actor.id}, source={self.source})"
        )


class ObjectDetector:
    """Detect and filter scene objects within a detection radius.

    All methods are static — the class is used purely as a namespace.
    """

    #: Maximum XY distance (m) for deduplicating static vehicles against
    #: dynamic actors at the same world position.
    STATIC_DEDUP_DISTANCE_M: float = 2.5

    #: Maximum XY distance (m) for deduplicating environment sign proxies
    #: against already-annotated spawned sign actors / LightHeadProxies.
    STATIC_SIGN_DEDUP_DISTANCE_M: float = 3.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def get_objects_of_interest(
        world: carla.World,
        ego_vehicle: carla.Actor,
        max_distance: float = 80.0,
    ) -> List[DetectedObject]:
        """Return all annotatable objects within *max_distance* of *ego_vehicle*.

        The method:

        1. Iterates all dynamic actors in the world.
        2. Expands traffic-light actors into per-head
           :class:`~lib.detection.proxies.LightHeadProxy` objects.
        3. Appends static-vehicle environment objects (de-duplicated against
           dynamic actors at the same position).
        4. Appends static-sign environment objects (de-duplicated against
           spawned sign actors and light-head proxies).
        5. Sorts the result by (distance, actor_id).

        Args:
            world: CARLA world.
            ego_vehicle: The ego vehicle actor (excluded from results).
            max_distance: Maximum detection radius in metres.

        Returns:
            List of :class:`DetectedObject` instances, sorted ascending by
            distance.
        """
        objects: List[DetectedObject] = []
        ego_loc = ego_vehicle.get_location()

        for actor in world.get_actors():
            if actor.id == ego_vehicle.id:
                continue
            dist = ego_loc.distance(actor.get_location())
            if dist > max_distance:
                continue
            cat = ObjectCategorizer.get_category(actor)
            if not cat:
                continue

            if cat == CAT_TRAFFIC_LIGHT:
                heads = expand_to_light_head_proxies(actor)
                if heads:
                    objects.extend(
                        DetectedObject(h, CAT_TRAFFIC_LIGHT, dist) for h in heads
                    )
                else:
                    objects.append(DetectedObject(actor, cat, dist))
            else:
                objects.append(DetectedObject(actor, cat, dist))

        # Static parked vehicles.
        for static_obj in StaticObjectFactory.get_static_vehicle_objects(world):
            dist = ego_loc.distance(static_obj.get_location())
            if dist > max_distance:
                continue
            if ObjectDetector._is_duplicate_static_vehicle(static_obj, objects):
                continue
            cat = ObjectCategorizer.get_category(static_obj)
            if not cat:
                continue
            objects.append(
                DetectedObject(static_obj, cat, dist, source=static_obj.source)
            )

        # Static sign panels.
        for sign_obj in StaticObjectFactory.get_static_sign_objects(world):
            dist = ego_loc.distance(sign_obj.get_location())
            if dist > max_distance:
                continue
            if ObjectDetector._is_duplicate_env_sign(sign_obj, objects):
                continue
            cat = ObjectCategorizer.get_category(sign_obj)
            if not cat:
                continue
            objects.append(
                DetectedObject(sign_obj, cat, dist, source=sign_obj.source)
            )

        objects.sort(key=lambda o: (o.distance, o.actor.id))
        return objects

    @staticmethod
    def get_detection_summary(objects: List[DetectedObject]) -> Dict[str, int]:
        """Return a category-token → count dict for a list of detected objects.

        Args:
            objects: List returned by :meth:`get_objects_of_interest`.
        """
        summary: Dict[str, int] = {}
        for obj in objects:
            summary[obj.category_token] = summary.get(obj.category_token, 0) + 1
        return summary

    @staticmethod
    def filter_by_category(
        objects: List[DetectedObject], categories: List[str]
    ) -> List[DetectedObject]:
        """Filter *objects* to only those whose category is in *categories*.

        Args:
            objects: Input list.
            categories: Category token allow-list.
        """
        return [o for o in objects if o.category_token in categories]

    # ------------------------------------------------------------------
    # Deduplication helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_duplicate_static_vehicle(
        static_object: StaticObjectProxy, dynamic_objects: List[DetectedObject]
    ) -> bool:
        """Return ``True`` if a dynamic vehicle actor already covers *static_object*.

        Args:
            static_object: Candidate static-vehicle proxy.
            dynamic_objects: Already-collected objects to check against.
        """
        sloc = static_object.get_location()
        for det in dynamic_objects:
            if not det.actor.type_id.startswith("vehicle."):
                continue
            try:
                dloc = det.actor.get_location()
            except Exception:
                continue
            if sloc.distance(dloc) <= ObjectDetector.STATIC_DEDUP_DISTANCE_M:
                return True
        return False

    @staticmethod
    def _is_duplicate_env_sign(
        sign_obj: StaticObjectProxy, existing_objects: List[DetectedObject]
    ) -> bool:
        """Return ``True`` if an existing annotated object already covers *sign_obj*.

        Uses XY-only distance so that elevated ``LightHeadProxy`` locations
        (signal heads at ~5 m) still match ground-level env-sign transforms
        (z ≈ 0.1–0.3 m) at the same pole base.

        Args:
            sign_obj: Candidate environment-sign proxy.
            existing_objects: Already-collected objects to check against.
        """
        sloc = sign_obj.get_location()
        for det in existing_objects:
            if det.category_token not in (CAT_TRAFFIC_SIGN, CAT_TRAFFIC_LIGHT):
                continue
            if getattr(det.actor, "source", "actor") == "env_sign":
                continue
            try:
                oloc = det.actor.get_location()
            except Exception:
                continue
            xy_dist = math.sqrt((sloc.x - oloc.x) ** 2 + (sloc.y - oloc.y) ** 2)
            if xy_dist <= ObjectDetector.STATIC_SIGN_DEDUP_DISTANCE_M:
                return True
        return False
