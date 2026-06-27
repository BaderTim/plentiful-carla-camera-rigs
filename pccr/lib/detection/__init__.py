"""
Object detection and categorisation sub-package.

This package is a structured split of the former monolithic
``lib/object_detection.py`` module:

* **proxies** — duck-typed wrappers for CARLA objects that cannot be manipulated
  as first-class actors (``StaticObjectProxy``, ``LightHeadProxy``,
  ``CategoryColors``, ``StaticObjectFactory``).
* **categorizer** — maps CARLA blueprint IDs to nuScenes category tokens
  (``ObjectCategorizer``).
* **detector** — queries the CARLA world for scene objects and returns a flat
  list of ``DetectedObject`` instances (``ObjectDetector``).

Public API (re-exported here for backward compatibility with code that
previously did ``from lib.object_detection import ObjectDetector``)::

    from lib.detection import ObjectCategorizer, ObjectDetector, DetectedObject
    from lib.detection import CategoryColors, StaticObjectProxy, LightHeadProxy
"""

from .categorizer import ObjectCategorizer
from .detector import DetectedObject, ObjectDetector
from .proxies import (
    CategoryColors,
    LightHeadProxy,
    StaticObjectFactory,
    StaticObjectProxy,
)

__all__ = [
    "ObjectCategorizer",
    "ObjectDetector",
    "DetectedObject",
    "CategoryColors",
    "LightHeadProxy",
    "StaticObjectProxy",
    "StaticObjectFactory",
]
