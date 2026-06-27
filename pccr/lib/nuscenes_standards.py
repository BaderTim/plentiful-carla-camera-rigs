"""
nuScenes dataset constants and ontology helpers.

Defines all fixed token constants for categories, attributes, and visibility
levels.  Using fixed (non-random) tokens ensures consistency across datasets
and allows merging datasets from different rigs.
"""

from __future__ import annotations

from typing import Any, Dict, List


# ============================================================================
# Category token constants (32-char hex, fixed UUIDs)
# ============================================================================

CAT_CAR = "1fa93b757fc74fb9942a253c59b4f2ff"
CAT_TRUCK = "6021b5187b924d5ba0be7ec8d89695ab"
CAT_BUS = "fedb11688db84088883945752e480c2c"
CAT_MOTORCYCLE = "dfd26f200ade4d24b540184e16050022"
CAT_BICYCLE = "fc95c87aed2a4e4e99a0e8b3cd8df30f"
CAT_ADULT = "1fa93b757fc74fb99627d72a3e5ba8f9"
CAT_CHILD = "1fa93b757fc74fb99627d72a3e5ba8f0"
CAT_TRAFFIC_SIGN = "653f7efbb9514ce7b81d44070d6208c1"
CAT_TRAFFIC_LIGHT = "53f7efbb9514ce7b81d44070d62087c8"

#: Backward-compatibility alias — prefer ``CAT_ADULT`` in new code.
CAT_PEDESTRIAN = CAT_ADULT

# ============================================================================
# Attribute token constants (32-char hex, fixed UUIDs)
# ============================================================================

ATTR_VEHICLE_MOVING = "00f6fa702f3c5d51d98e8aff1d97c8f1"
ATTR_VEHICLE_PARKED = "00f6fa702f3c5d51d98e8aff1d97c8ee"
ATTR_VEHICLE_STOPPED = "00f6fa702f3c5d51d98e8aff1d97c8f2"
ATTR_ADULT_MOVING = "0d30b188a2934f738a8f6c39cd5d3d7b"
ATTR_ADULT_STANDING = "0d30b188a2934f738a8f6c39cd5d3d7c"
ATTR_CYCLE_WITH_RIDER = "a5ef34f49b6b488a8f90e3c5ae4bf5f1"
ATTR_CYCLE_WITHOUT_RIDER = "a5ef34f49b6b488a8f90e3c5ae4bf5f2"
ATTR_TRAFFIC_LIGHT_RED = "e2f9c6c3d8ee4f9d8b1aa6d0f4b2a101"
ATTR_TRAFFIC_LIGHT_YELLOW = "e2f9c6c3d8ee4f9d8b1aa6d0f4b2a102"
ATTR_TRAFFIC_LIGHT_GREEN = "e2f9c6c3d8ee4f9d8b1aa6d0f4b2a103"
ATTR_TRAFFIC_LIGHT_OFF = "e2f9c6c3d8ee4f9d8b1aa6d0f4b2a104"
ATTR_TRAFFIC_LIGHT_UNKNOWN = "e2f9c6c3d8ee4f9d8b1aa6d0f4b2a105"


class NuScenesStandards:
    """Static helpers that return the standard nuScenes ontology tables.

    All methods return lists of dicts ready to be written directly to the
    corresponding ``*.json`` annotation files.
    """

    @staticmethod
    def get_categories() -> List[Dict[str, Any]]:
        """Return the category table for this project's ontology."""
        return [
            {"token": CAT_CAR, "name": "car", "description": "Car"},
            {"token": CAT_TRUCK, "name": "truck", "description": "Truck"},
            {"token": CAT_BUS, "name": "bus", "description": "Bus"},
            {"token": CAT_MOTORCYCLE, "name": "motorcycle", "description": "Motorcycle"},
            {"token": CAT_BICYCLE, "name": "bicycle", "description": "Bicycle"},
            {"token": CAT_ADULT, "name": "adult", "description": "Adult Pedestrian"},
            {"token": CAT_CHILD, "name": "child", "description": "Child Pedestrian"},
            {"token": CAT_TRAFFIC_SIGN, "name": "traffic_sign", "description": "Traffic sign"},
            {"token": CAT_TRAFFIC_LIGHT, "name": "traffic_light", "description": "Traffic light"},
        ]

    @staticmethod
    def get_attributes() -> List[Dict[str, Any]]:
        """Return the attribute table."""
        return [
            {"token": ATTR_VEHICLE_MOVING, "name": "moving", "description": "Vehicle is moving"},
            {"token": ATTR_VEHICLE_PARKED, "name": "parked", "description": "Vehicle is parked"},
            {"token": ATTR_VEHICLE_STOPPED, "name": "stopped", "description": "Vehicle is stopped"},
            {"token": ATTR_ADULT_MOVING, "name": "adult_moving", "description": "Adult is moving"},
            {"token": ATTR_ADULT_STANDING, "name": "adult_standing", "description": "Adult is standing"},
            {
                "token": ATTR_CYCLE_WITH_RIDER,
                "name": "with_rider",
                "description": "Bicycle/motorcycle with rider",
            },
            {
                "token": ATTR_CYCLE_WITHOUT_RIDER,
                "name": "without_rider",
                "description": "Bicycle/motorcycle without rider",
            },
            {
                "token": ATTR_TRAFFIC_LIGHT_RED,
                "name": "traffic_light_red",
                "description": "Traffic light is red",
            },
            {
                "token": ATTR_TRAFFIC_LIGHT_YELLOW,
                "name": "traffic_light_yellow",
                "description": "Traffic light is yellow",
            },
            {
                "token": ATTR_TRAFFIC_LIGHT_GREEN,
                "name": "traffic_light_green",
                "description": "Traffic light is green",
            },
            {
                "token": ATTR_TRAFFIC_LIGHT_OFF,
                "name": "traffic_light_off",
                "description": "Traffic light is off",
            },
            {
                "token": ATTR_TRAFFIC_LIGHT_UNKNOWN,
                "name": "traffic_light_unknown",
                "description": "Traffic light state is unknown",
            },
        ]

    @staticmethod
    def get_visibility() -> List[Dict[str, Any]]:
        """Return the visibility table.

        Tokens ``"0"``–``"4"`` map to increasing visibility bands.  Token
        ``"0"`` means the object is essentially invisible (0–5 % of the bounding
        box projected area is unoccluded).
        """
        return [
            {"token": "0", "level": "none", "description": "0-5%: No part visible"},
            {"token": "1", "level": "partial", "description": "5-40%: Small portion visible"},
            {"token": "2", "level": "partial", "description": "40-60%: Large portion visible"},
            {"token": "3", "level": "most", "description": "60-80%: Most visible"},
            {"token": "4", "level": "full", "description": "80-100%: Fully visible"},
        ]
