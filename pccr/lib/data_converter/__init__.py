"""
data_converter
==============

Converters for transforming nuScenes-format PCCR datasets into the pkl
info format expected by BEVDet and related mmdet3d-style models.

Modules
-------
nuscenes_converter
    Build train/val pkl info files from a nuScenes-formatted dataset.
bevdet_annotations
    Enrich pkl info files with ego-coordinate GT boxes and scene tokens.
create_gt_database
    Generate a ground-truth database for copy-paste data augmentation.
"""
