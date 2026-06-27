"""
Specialised bounding-box handling for traffic infrastructure.

Traffic lights and traffic signs require custom logic because:

* CARLA does not expose a single coherent bounding box for them.
* We want per-signal-head boxes for traffic lights (not the whole pole assembly).
* We want sign-face-only boxes for traffic signs (not the support mast).

:class:`TrafficInfrastructureBBox` is a pure-utility class (all static methods)
that wraps this logic, while :class:`SignVisualProfile` carries the per-family
visual-extent parameters used by the fallback path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import carla

from ..nuscenes_standards import CAT_TRAFFIC_LIGHT, CAT_TRAFFIC_SIGN
from .bbox_builder import BoundingBoxBuilder, BoundingBoxInfo


@dataclass
class SignVisualProfile:
    """Visual-only fallback extent profile for a traffic sign face.

    Attributes:
        extent: Half-extents of the sign face panel.
        center_height: Height above the actor origin where the panel centre
            should be placed (metres).
        source: Human-readable provenance label.
    """

    extent: carla.Vector3D
    center_height: float
    source: str


class TrafficInfrastructureBBox:
    """Bounding-box helpers for traffic lights and traffic signs.

    All methods are static — the class is a namespace only.
    """

    #: Sub-strings that identify a valid sign-face mesh component.
    SIGN_COMPONENT_KEYWORDS: Tuple[str, ...] = (
        "roundsign",
        "speedlimit",
        "speed_limit",
    )
    #: Sub-strings that disqualify a component name (support / collision geo).
    SIGN_COMPONENT_EXCLUDE_KEYWORDS: Tuple[str, ...] = (
        "trigger",
        "collision",
        "root",
        "pole",
        "post",
        "support",
        "base",
        "bounds",
        "box",
    )
    #: Z-tolerance when clustering signal heads at the same height (metres).
    LIGHT_TOP_CLUSTER_Z_TOLERANCE: float = 1.0
    #: Maximum XY spread before falling back to a single representative head (metres).
    LIGHT_WIDE_CLUSTER_THRESHOLD_M: float = 2.5

    # ------------------------------------------------------------------
    # Component introspection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_component_names(actor: carla.Actor) -> List[str]:
        """Return lower-cased component names for *actor*, or ``[]`` on failure."""
        if not hasattr(actor, "get_component_names"):
            return []
        try:
            return [name.lower() for name in actor.get_component_names()]
        except Exception:
            return []

    @staticmethod
    def _is_billboard_sign_like(component_names: List[str]) -> bool:
        """Return ``True`` if *component_names* suggests a billboard-style sign."""
        has_billboard = any("billboard" in name for name in component_names)
        has_stop_style = any(
            "stopsign" in name or "stopbox" in name for name in component_names
        )
        has_arrow = any("arrow" in name for name in component_names)
        has_mesh = any(
            "staticmesh" in name
            or "signcomponent" in name
            or "roadsigns_" in name
            for name in component_names
        )
        return has_billboard and has_stop_style and not has_arrow and not has_mesh

    # ------------------------------------------------------------------
    # Default extent helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_traffic_light_extent() -> carla.Vector3D:
        """Return the default half-extent for a traffic light assembly box."""
        return carla.Vector3D(x=0.3, y=0.3, z=1.6)

    @staticmethod
    def get_traffic_sign_extent() -> carla.Vector3D:
        """Return the default half-extent for a traffic sign assembly box."""
        return carla.Vector3D(x=0.4, y=0.1, z=1.0)

    @staticmethod
    def get_round_sign_extent() -> carla.Vector3D:
        """Return the half-extent for a round sign-face component."""
        return carla.Vector3D(x=0.4, y=0.1, z=0.4)

    @staticmethod
    def get_rect_sign_extent() -> carla.Vector3D:
        """Return the half-extent for a rectangular / generic sign-face component."""
        return carla.Vector3D(x=0.45, y=0.1, z=0.6)

    @staticmethod
    def get_traffic_infrastructure_extent(category_token: str) -> carla.Vector3D:
        """Return default half-extent based on infrastructure category.

        Args:
            category_token: nuScenes category token hex string.
        """
        if category_token == CAT_TRAFFIC_LIGHT:
            return TrafficInfrastructureBBox.get_traffic_light_extent()
        if category_token == CAT_TRAFFIC_SIGN:
            return TrafficInfrastructureBBox.get_traffic_sign_extent()
        return carla.Vector3D(x=0.5, y=0.5, z=1.0)

    # ------------------------------------------------------------------
    # Sign visual-profile (fallback extent per sign family)
    # ------------------------------------------------------------------

    @staticmethod
    def get_sign_visual_profile(actor: carla.Actor) -> SignVisualProfile:
        """Return a sign-face fallback :class:`SignVisualProfile` for *actor*.

        Dispatches on ``type_id`` and component names to pick the closest
        visual profile for the sign family.

        Args:
            actor: Traffic-sign actor.

        Returns:
            :class:`SignVisualProfile` with extent, centre height, and source tag.
        """
        type_id = getattr(actor, "type_id", "").lower()
        component_names = TrafficInfrastructureBBox._get_component_names(actor)

        sign_center_height_high = 2.15
        sign_center_height_low = 1.9

        if TrafficInfrastructureBBox._is_billboard_sign_like(component_names):
            return SignVisualProfile(
                extent=carla.Vector3D(x=0.42, y=0.05, z=0.42),
                center_height=sign_center_height_high,
                source="fallback_billboard_face",
            )
        if "yield" in type_id or any(
            "yieldsigncomponent" in name or name == "yieldsign"
            for name in component_names
        ):
            return SignVisualProfile(
                extent=carla.Vector3D(x=0.50, y=0.05, z=0.50),
                center_height=sign_center_height_high,
                source="fallback_yield_face",
            )
        if "stop" in type_id or any(
            "stopsigncomponent" in name or name == "stopsign"
            for name in component_names
        ):
            return SignVisualProfile(
                extent=carla.Vector3D(x=0.45, y=0.1, z=0.45),
                center_height=sign_center_height_high,
                source="fallback_stop_face",
            )
        if "speed_limit" in type_id or "speedlimit" in type_id or any(
            "roundsign" in name or "speedsign" in name for name in component_names
        ):
            return SignVisualProfile(
                extent=carla.Vector3D(x=0.40, y=0.1, z=0.50),
                center_height=sign_center_height_high,
                source="fallback_round_face",
            )
        if "traffic_sign" in type_id:
            return SignVisualProfile(
                extent=carla.Vector3D(x=0.55, y=0.1, z=0.55),
                center_height=sign_center_height_low,
                source="fallback_rect_face",
            )
        return SignVisualProfile(
            extent=carla.Vector3D(x=0.45, y=0.05, z=0.45),
            center_height=sign_center_height_high,
            source="fallback_generic_face",
        )

    # ------------------------------------------------------------------
    # Sign component enumeration
    # ------------------------------------------------------------------

    @staticmethod
    def _is_valid_sign_component_name(name: str) -> bool:
        """Return ``True`` if *name* identifies a sign-face mesh (not support geo)."""
        n = name.lower()
        if any(kw in n for kw in TrafficInfrastructureBBox.SIGN_COMPONENT_EXCLUDE_KEYWORDS):
            return False
        return any(kw in n for kw in TrafficInfrastructureBBox.SIGN_COMPONENT_KEYWORDS)

    @staticmethod
    def _get_sign_component_extent(name: str) -> carla.Vector3D:
        """Return a compact visual half-extent for a sign-face component by name."""
        n = name.lower()
        if "roundsign" in n or "speedlimit" in n or "speed_limit" in n:
            return TrafficInfrastructureBBox.get_round_sign_extent()
        return TrafficInfrastructureBBox.get_rect_sign_extent()

    @staticmethod
    def get_sign_component_corners_list(
        actor: carla.Actor,
    ) -> List[Tuple[str, List[carla.Location], carla.Transform]]:
        """Return per-sign-face corners for each valid mesh component in *actor*.

        Args:
            actor: Traffic-sign actor with ``get_component_names()`` and
                ``get_component_world_transform(name)``.

        Returns:
            List of ``(component_name, corners_list, world_transform)`` tuples.
            Empty if the actor has no addressable components.
        """
        if not hasattr(actor, "get_component_names"):
            return []
        result = []
        try:
            for name in actor.get_component_names():
                if not TrafficInfrastructureBBox._is_valid_sign_component_name(name):
                    continue
                extent = TrafficInfrastructureBBox._get_sign_component_extent(name)
                try:
                    comp_transform = actor.get_component_world_transform(name)
                    corners = BoundingBoxBuilder.get_custom_bbox_world_corners(
                        comp_transform.location, extent, comp_transform.rotation
                    )
                    result.append((name, corners, comp_transform))
                except Exception:
                    continue
        except Exception:
            pass
        return result

    # ------------------------------------------------------------------
    # Traffic light box selection
    # ------------------------------------------------------------------

    @staticmethod
    def _get_light_boxes(actor: carla.Actor) -> List[carla.BoundingBox]:
        """Return raw light boxes from *actor*, or ``[]`` on failure."""
        if not hasattr(actor, "get_light_boxes"):
            return []
        try:
            return list(actor.get_light_boxes())
        except Exception:
            return []

    @staticmethod
    def _light_box_spread(boxes: List[carla.BoundingBox]) -> float:
        """Return the maximum pairwise centre-to-centre distance across *boxes*."""
        if len(boxes) < 2:
            return 0.0
        centers = [box.location for box in boxes]
        max_dist = 0.0
        for i, ca in enumerate(centers):
            for cb in centers[i + 1:]:
                max_dist = max(max_dist, ca.distance(cb))
        return max_dist

    @staticmethod
    def _select_representative_light_boxes(
        boxes: List[carla.BoundingBox],
    ) -> List[carla.BoundingBox]:
        """Choose the per-head boxes most representative of the visible signal face.

        Strategy:

        1. Keep only the topmost cluster (within ``LIGHT_TOP_CLUSTER_Z_TOLERANCE``).
        2. If the top cluster is compact (spread ≤ ``LIGHT_WIDE_CLUSTER_THRESHOLD_M``),
           return all of them.
        3. Otherwise return only the single head closest to the centroid.

        Args:
            boxes: All light boxes returned by ``actor.get_light_boxes()``.

        Returns:
            Subset of *boxes* to use for the annotation bbox.
        """
        if not boxes:
            return []
        max_z = max(box.location.z for box in boxes)
        top_boxes = [
            box for box in boxes
            if (max_z - box.location.z) <= TrafficInfrastructureBBox.LIGHT_TOP_CLUSTER_Z_TOLERANCE
        ]
        candidates = top_boxes or boxes

        if len(candidates) <= 1:
            return candidates
        if (
            TrafficInfrastructureBBox._light_box_spread(candidates)
            <= TrafficInfrastructureBBox.LIGHT_WIDE_CLUSTER_THRESHOLD_M
        ):
            return candidates

        centroid = carla.Location(
            x=sum(box.location.x for box in candidates) / len(candidates),
            y=sum(box.location.y for box in candidates) / len(candidates),
            z=sum(box.location.z for box in candidates) / len(candidates),
        )
        representative = min(candidates, key=lambda b: b.location.distance(centroid))
        return [representative]

    # ------------------------------------------------------------------
    # Public bbox-info factories
    # ------------------------------------------------------------------

    @staticmethod
    def get_light_bbox_info(actor: carla.Actor) -> BoundingBoxInfo:
        """Return a compact :class:`BoundingBoxInfo` for the representative signal
        head(s) of a traffic light.

        Falls back to :meth:`get_fallback_bbox_info` when no light boxes are
        available.

        Args:
            actor: Traffic-light actor.
        """
        light_boxes = TrafficInfrastructureBBox._get_light_boxes(actor)
        selected_boxes = TrafficInfrastructureBBox._select_representative_light_boxes(
            light_boxes
        )
        if selected_boxes:
            merged_corners = [
                corner
                for lb in selected_boxes
                for corner in BoundingBoxBuilder.get_carla_bbox_world_corners(lb)
            ]
            source = (
                "merged_light_boxes"
                if len(selected_boxes) > 1
                else "representative_light_box"
            )
            return BoundingBoxBuilder.bbox_info_from_world_corners(
                merged_corners,
                actor.get_transform().rotation,
                source=source,
                minimum_extent=(0.05, 0.05, 0.1),
            )
        return TrafficInfrastructureBBox.get_fallback_bbox_info(actor, CAT_TRAFFIC_LIGHT)

    @staticmethod
    def get_sign_bbox_info(actor: carla.Actor) -> BoundingBoxInfo:
        """Return a :class:`BoundingBoxInfo` derived from sign mesh components.

        Falls back to :meth:`get_fallback_bbox_info` when no valid components are
        found.

        Args:
            actor: Traffic-sign actor.
        """
        components = TrafficInfrastructureBBox.get_sign_component_corners_list(actor)
        if components:
            merged_corners = [
                corner for _, corners, _ in components for corner in corners
            ]
            return BoundingBoxBuilder.bbox_info_from_world_corners(
                merged_corners,
                actor.get_transform().rotation,
                source="components",
                minimum_extent=(0.05, 0.05, 0.1),
            )
        return TrafficInfrastructureBBox.get_fallback_bbox_info(actor, CAT_TRAFFIC_SIGN)

    @staticmethod
    def get_fallback_corners(
        actor: carla.Actor,
        category_token: str,
    ) -> List[carla.Location]:
        """Return fallback world-corners for traffic infrastructure.

        For signs a :class:`SignVisualProfile` determines the extent and
        the vertical offset; for lights a default extent is used.

        Args:
            actor: Traffic infrastructure actor.
            category_token: nuScenes category token hex string.
        """
        transform = actor.get_transform()
        if category_token == CAT_TRAFFIC_SIGN:
            profile = TrafficInfrastructureBBox.get_sign_visual_profile(actor)
            bbox_center = carla.Location(
                transform.location.x,
                transform.location.y,
                transform.location.z + profile.center_height,
            )
            return BoundingBoxBuilder.get_custom_bbox_world_corners(
                bbox_center, profile.extent, transform.rotation
            )

        extent = TrafficInfrastructureBBox.get_traffic_infrastructure_extent(category_token)
        bbox_center = carla.Location(
            transform.location.x,
            transform.location.y,
            transform.location.z + extent.z,
        )
        return BoundingBoxBuilder.get_custom_bbox_world_corners(
            bbox_center, extent, transform.rotation
        )

    @staticmethod
    def get_fallback_bbox_info(
        actor: carla.Actor,
        category_token: str,
    ) -> BoundingBoxInfo:
        """Return fallback :class:`BoundingBoxInfo` for traffic infrastructure.

        Args:
            actor: Traffic infrastructure actor.
            category_token: nuScenes category token hex string.
        """
        corners = TrafficInfrastructureBBox.get_fallback_corners(actor, category_token)
        source = "fallback"
        if category_token == CAT_TRAFFIC_SIGN:
            source = TrafficInfrastructureBBox.get_sign_visual_profile(actor).source
        return BoundingBoxBuilder.bbox_info_from_world_corners(
            corners,
            actor.get_transform().rotation,
            source=source,
            minimum_extent=(0.05, 0.05, 0.1),
        )

    @staticmethod
    def get_annotation_bbox_info(
        actor: carla.Actor,
        category_token: str,
    ) -> BoundingBoxInfo:
        """Return the canonical annotation :class:`BoundingBoxInfo` for an
        infrastructure actor.

        Dispatches to the appropriate specialised method based on *category_token*.

        Args:
            actor: Traffic infrastructure actor.
            category_token: nuScenes category token hex string.
        """
        if category_token == CAT_TRAFFIC_LIGHT:
            return TrafficInfrastructureBBox.get_light_bbox_info(actor)
        if category_token == CAT_TRAFFIC_SIGN:
            return TrafficInfrastructureBBox.get_sign_bbox_info(actor)
        return TrafficInfrastructureBBox.get_fallback_bbox_info(actor, category_token)

    # ------------------------------------------------------------------
    # Debug / visualisation helpers (retained for annotation_debug_tool)
    # ------------------------------------------------------------------

    @staticmethod
    def get_light_box_corners_list(
        actor: carla.Actor,
    ) -> List[List[carla.Location]]:
        """Return per-light-box corner lists for visualisation.

        Args:
            actor: Traffic-light actor.

        Returns:
            List of corner lists, one list per light box.  Empty when none
            are available.
        """
        return [
            BoundingBoxBuilder.get_carla_bbox_world_corners(lb)
            for lb in TrafficInfrastructureBBox._get_light_boxes(actor)
        ]
