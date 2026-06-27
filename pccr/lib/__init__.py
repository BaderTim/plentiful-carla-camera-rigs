"""
lib — CARLA-to-nuScenes shared library.

This package is designed to be imported from any top-level entry point
(``core/``, ``tools/``) after adding the project root to ``sys.path``::

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from lib.nuscenes_builder import NuScenesBuilder

Sub-packages
------------
detection
    Actor categorisation, per-head light proxies, and scene object detection.
bbox
    Bounding-box creation and traffic infrastructure bbox utilities.
utils
    Shared utility classes: coordinate transforms, image processing, logging,
    depth assembly, attribute inference, and visibility calculation.
data_converter
    nuScenes-format → BEVDet/mmdet3d pkl info converters.
"""
