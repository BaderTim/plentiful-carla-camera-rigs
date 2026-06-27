"""
Core bounding-box creation and transformation utilities for CARLA objects.

:class:`BoundingBoxBuilder` is a pure-utility class (all static methods)
that handles:

* Corner generation from an extent in local space.
* World-space transformation.
* Standard actor-bbox queries using CARLA's built-in ``bounding_box``
  attribute.
* Custom bbox creation from an arbitrary center / extent / rotation.
* Normalising an arbitrary cloud of world-space corners back into a compact
  :class:`BoundingBoxInfo`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import carla


@dataclass
class BoundingBoxInfo:
    """Normalised bbox description shared across debug and annotation code.

    Attributes:
        corners: Eight world-space corner :class:`carla.Location` objects.
        extent: Half-extents of the fitted box (CARLA-right-hand-rule units).
        transform: Centre transform of the box in world space.
        source: Human-readable provenance tag (e.g. ``"actor_bbox"``,
            ``"components"``, ``"fallback"``).
    """

    corners: List[carla.Location]
    extent: carla.Vector3D
    transform: carla.Transform
    source: str


class BoundingBoxBuilder:
    """Utility class for creating and transforming bounding boxes.

    All methods are static — the class is a namespace only.
    """

    @staticmethod
    def create_bbox_corners(extent: carla.Vector3D) -> List[carla.Location]:
        """Return the eight corners of a box in local coordinates.

        Corner order (matching CARLA convention):
        0 = bottom-front-right,  1 = bottom-front-left,
        2 = bottom-rear-left,    3 = bottom-rear-right,
        4 = top-front-right,     5 = top-front-left,
        6 = top-rear-left,       7 = top-rear-right.

        Args:
            extent: Bounding box half-extents along x, y, z.

        Returns:
            List of eight :class:`carla.Location` in local space.
        """
        return [
            carla.Location( extent.x,  extent.y, -extent.z),  # 0 bottom-front-right
            carla.Location(-extent.x,  extent.y, -extent.z),  # 1 bottom-front-left
            carla.Location(-extent.x, -extent.y, -extent.z),  # 2 bottom-rear-left
            carla.Location( extent.x, -extent.y, -extent.z),  # 3 bottom-rear-right
            carla.Location( extent.x,  extent.y,  extent.z),  # 4 top-front-right
            carla.Location(-extent.x,  extent.y,  extent.z),  # 5 top-front-left
            carla.Location(-extent.x, -extent.y,  extent.z),  # 6 top-rear-left
            carla.Location( extent.x, -extent.y,  extent.z),  # 7 top-rear-right
        ]

    @staticmethod
    def transform_corners_to_world(
        corners: List[carla.Location],
        transform: carla.Transform,
    ) -> List[carla.Location]:
        """Apply *transform* to each corner location.

        Args:
            corners: Local-space corner locations.
            transform: Transform to apply.

        Returns:
            Corresponding world-space corner locations.
        """
        return [transform.transform(corner) for corner in corners]

    @staticmethod
    def get_bbox_edges() -> List[Tuple[int, int]]:
        """Return the twelve edges of a box as (start, end) corner-index pairs.

        Returns:
            List of 12 ``(start_idx, end_idx)`` tuples.
        """
        return [
            # Bottom face
            (0, 1), (1, 2), (2, 3), (3, 0),
            # Top face
            (4, 5), (5, 6), (6, 7), (7, 4),
            # Vertical edges
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]

    @staticmethod
    def get_standard_bbox_world_corners(actor: carla.Actor) -> List[carla.Location]:
        """Return world-space corners for a standard CARLA actor's bounding box.

        For environment objects (``actor.is_environment_object == True``) the
        world transform is baked into ``actor.bounding_box.location``; for
        regular spawned actors the bbox location is in vehicle-local space.

        Args:
            actor: CARLA actor with a ``bounding_box`` attribute.

        Returns:
            List of eight world-space corner locations.
        """
        if getattr(actor, "is_environment_object", False):
            return BoundingBoxBuilder.get_carla_bbox_world_corners(actor.bounding_box)

        bbox = actor.bounding_box
        transform = actor.get_transform()
        bb_transform = carla.Transform(bbox.location)
        bbox_corners = BoundingBoxBuilder.create_bbox_corners(bbox.extent)

        world_corners = []
        for corner in bbox_corners:
            vehicle_space_corner = bb_transform.transform(corner)
            world_corners.append(transform.transform(vehicle_space_corner))
        return world_corners

    @staticmethod
    def get_actor_bbox_info(
        actor: carla.Actor,
        source: str = "actor_bbox",
    ) -> BoundingBoxInfo:
        """Return a normalised :class:`BoundingBoxInfo` for a CARLA actor.

        Args:
            actor: CARLA actor (or proxy object with ``bounding_box``).
            source: Provenance label to embed in the result.

        Returns:
            :class:`BoundingBoxInfo` with world-space corners, extent, and
            the centred world transform.
        """
        if getattr(actor, "is_environment_object", False):
            bbox = actor.bounding_box
            corners = BoundingBoxBuilder.get_carla_bbox_world_corners(bbox)
            rotation = (
                bbox.rotation if hasattr(bbox, "rotation")
                else actor.get_transform().rotation
            )
            # CARLA environment-object boxes occasionally expose signed extents.
            # Re-fitting from the world-corner cloud recovers positive half-extents
            # and keeps the exported nuScenes size physically meaningful.
            return BoundingBoxBuilder.bbox_info_from_world_corners(
                corners,
                rotation,
                source=source,
            )

        bbox = actor.bounding_box
        world_transform = actor.get_transform()
        location = world_transform.transform(bbox.location)
        transform = carla.Transform(location, world_transform.rotation)
        corners = BoundingBoxBuilder.get_standard_bbox_world_corners(actor)
        return BoundingBoxInfo(
            corners=corners,
            extent=bbox.extent,
            transform=transform,
            source=source,
        )

    @staticmethod
    def get_carla_bbox_world_corners(
        bbox: carla.BoundingBox,
    ) -> List[carla.Location]:
        """Return world corners for a :class:`carla.BoundingBox` (e.g. from
        ``actor.get_light_boxes()``).

        Args:
            bbox: A CARLA :class:`~carla.BoundingBox` whose ``location`` is
                already in world space.

        Returns:
            List of eight world-space corner locations.
        """
        bbox_corners = BoundingBoxBuilder.create_bbox_corners(bbox.extent)
        rotation = bbox.rotation if hasattr(bbox, "rotation") else carla.Rotation()
        bbox_transform = carla.Transform(bbox.location, rotation)
        return BoundingBoxBuilder.transform_corners_to_world(bbox_corners, bbox_transform)

    @staticmethod
    def get_custom_bbox_world_corners(
        center: carla.Location,
        extent: carla.Vector3D,
        rotation: carla.Rotation,
    ) -> List[carla.Location]:
        """Return world corners for a custom box with given centre, extent, and rotation.

        Args:
            center: World-space centre of the box.
            extent: Half-extents of the box.
            rotation: Orientation of the box.

        Returns:
            List of eight world-space corner locations.
        """
        bbox_corners = BoundingBoxBuilder.create_bbox_corners(extent)
        bbox_transform = carla.Transform(center, rotation)
        return BoundingBoxBuilder.transform_corners_to_world(bbox_corners, bbox_transform)

    @staticmethod
    def bbox_info_from_world_corners(
        corners: List[carla.Location],
        rotation: carla.Rotation,
        source: str,
        minimum_extent: Tuple[float, float, float] = (0.05, 0.05, 0.1),
    ) -> BoundingBoxInfo:
        """Fit a :class:`BoundingBoxInfo` from an arbitrary cloud of world corners.

        Computes the centroid, transforms corners into centroid-local space, and
        derives half-extents (clamped to *minimum_extent*).

        Args:
            corners: Collection of world-space locations defining the box hull.
            rotation: Rotation to assign to the resulting transform.
            source: Provenance label.
            minimum_extent: Per-axis minimum half-extent ``(x, y, z)``.

        Returns:
            :class:`BoundingBoxInfo` with re-normalised corners.
        """
        center = carla.Location(
            sum(c.x for c in corners) / len(corners),
            sum(c.y for c in corners) / len(corners),
            sum(c.z for c in corners) / len(corners),
        )
        transform = carla.Transform(center, rotation)
        inverse_matrix = np.array(transform.get_inverse_matrix(), dtype=np.float32)
        corners_hom = np.array(
            [[c.x, c.y, c.z, 1.0] for c in corners], dtype=np.float32
        )
        local_corners = (inverse_matrix @ corners_hom.T).T[:, :3]
        min_vals = local_corners.min(axis=0)
        max_vals = local_corners.max(axis=0)

        extent = carla.Vector3D(
            max(float(max(abs(min_vals[0]), abs(max_vals[0]))), minimum_extent[0]),
            max(float(max(abs(min_vals[1]), abs(max_vals[1]))), minimum_extent[1]),
            max(float(max(abs(min_vals[2]), abs(max_vals[2]))), minimum_extent[2]),
        )
        normalised_corners = BoundingBoxBuilder.get_custom_bbox_world_corners(
            center, extent, rotation
        )
        return BoundingBoxInfo(
            corners=normalised_corners,
            extent=extent,
            transform=transform,
            source=source,
        )
