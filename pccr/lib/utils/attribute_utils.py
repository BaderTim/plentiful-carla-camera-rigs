"""
Attribute classification utilities for CARLA objects.

Determines nuScenes attribute tokens (``moving``, ``stopped``, ``parked``, etc.)
from current CARLA actor state.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import carla

from ..nuscenes_standards import (
    ATTR_ADULT_MOVING,
    ATTR_ADULT_STANDING,
    ATTR_CYCLE_WITH_RIDER,
    ATTR_CYCLE_WITHOUT_RIDER,
    ATTR_VEHICLE_MOVING,
    ATTR_VEHICLE_PARKED,
    ATTR_VEHICLE_STOPPED,
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

# Speed thresholds for motion classification (m/s).
#   Pedestrians: 0.1 m/s  — deliberately lower to avoid classifying slow shuffle
#                            as "standing".
#   Vehicles:    0.5 m/s  — ~1.8 km/h, filters out stationary physics jitter.
_PED_SPEED_THRESHOLD_MS: float = 0.1
_VEH_SPEED_THRESHOLD_MS: float = 0.5


class AttributeClassifier:
    """Classify nuScenes motion attributes for CARLA actors.

    All methods are static — the class is used purely as a namespace.
    """

    @staticmethod
    def _get_speed(actor: carla.Actor, trajectory_state: Optional[Dict[str, Any]]) -> float:
        if trajectory_state is not None:
            return float(trajectory_state.get("speed", 0.0) or 0.0)

        try:
            velocity = actor.get_velocity()
            return (velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) ** 0.5
        except Exception:
            return 0.0

    @staticmethod
    def _is_stopped_vehicle(trajectory_state: Dict[str, Any]) -> bool:
        if trajectory_state.get("control_hand_brake"):
            return False
        if trajectory_state.get("has_any_motion") or trajectory_state.get("has_nearby_motion"):
            return True
        if trajectory_state.get("is_at_traffic_light"):
            return True
        if trajectory_state.get("traffic_light_state_code") in (1, 2):
            return True
        if float(trajectory_state.get("control_brake", 0.0) or 0.0) > 0.05:
            return True
        if float(trajectory_state.get("control_throttle", 0.0) or 0.0) > 0.05:
            return True
        if abs(float(trajectory_state.get("control_steer", 0.0) or 0.0)) > 0.05:
            return True
        if trajectory_state.get("control_reverse"):
            return True
        return False

    @staticmethod
    def get_attribute(
        actor: carla.Actor,
        category_token: str,
        trajectory_state: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Return the nuScenes attribute token for *actor*.

        Returns ``None`` for static infrastructure categories (traffic signs and
        traffic lights) because those carry no motion state; the caller should
        store an empty ``attribute_tokens`` list in the annotation record.

        Args:
            actor: CARLA actor to classify.
            category_token: The actor's nuScenes category token (32-char hex).
            trajectory_state: Optional recorded replay state for the actor at the
                current frame. When provided, use this instead of live CARLA
                kinematics for motion attributes.

        Returns:
            Attribute token string, or ``None`` for infrastructure categories.
        """
        # Static infrastructure has no motion state.
        if category_token in (CAT_TRAFFIC_SIGN, CAT_TRAFFIC_LIGHT):
            return None

        # Environment objects are always parked/static.
        if getattr(actor, "is_environment_object", False):
            if category_token in (CAT_CAR, CAT_TRUCK, CAT_BUS):
                return ATTR_VEHICLE_PARKED

        speed = AttributeClassifier._get_speed(actor, trajectory_state)

        # Adults and children.
        if category_token in (CAT_ADULT, CAT_CHILD):
            return ATTR_ADULT_MOVING if speed > _PED_SPEED_THRESHOLD_MS else ATTR_ADULT_STANDING

        # Cycles — assume rider present when moving, no rider when stationary.
        if category_token in (CAT_BICYCLE, CAT_MOTORCYCLE):
            return ATTR_CYCLE_WITH_RIDER if speed > _PED_SPEED_THRESHOLD_MS else ATTR_CYCLE_WITHOUT_RIDER

        # Wheeled vehicles.
        if category_token in (CAT_CAR, CAT_TRUCK, CAT_BUS):
            if speed > _VEH_SPEED_THRESHOLD_MS:
                return ATTR_VEHICLE_MOVING
            if trajectory_state is not None and not AttributeClassifier._is_stopped_vehicle(trajectory_state):
                return ATTR_VEHICLE_PARKED
            return ATTR_VEHICLE_STOPPED

        # Generic fallback.
        return ATTR_VEHICLE_MOVING if speed > _VEH_SPEED_THRESHOLD_MS else ATTR_VEHICLE_STOPPED
