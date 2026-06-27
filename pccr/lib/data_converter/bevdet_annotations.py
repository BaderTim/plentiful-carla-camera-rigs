"""BEVDet-style annotation processing for nuScenes format datasets.

This module provides functions to enrich pkl info files with additional
annotation data required by BEVDet models:
- Velocity information for each annotation
- Ego-coordinate ground truth boxes
- Scene tokens
- Occupancy paths
"""
import pickle
from os import path as osp

import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

from .nuscenes_converter import nus_categories


# Map nuScenes category names to detection class names
# Uses the PCCR categories from nuscenes_converter.py
MAP_CATEGORY_TO_DETECTION = {
    # Vehicles
    'car': 'car',
    'truck': 'truck',
    'bus': 'bus',
    'motorcycle': 'motorcycle',
    'bicycle': 'bicycle',
    # Pedestrians
    'adult': 'adult',
    'child': 'child',
    # Static objects (PCCR specific)
    'traffic_light': 'traffic_light',
    'traffic_sign': 'traffic_sign'
}

# Detection classes used for BEVDet training
# Based on PCCR categories from nuscenes_converter.py
DETECTION_CLASSES = list(nus_categories)


def get_gt(info, detection_classes=None, category_mapping=None):
    """Generate ground truth labels from info in ego coordinates.

    Transforms annotations from global coordinates to ego vehicle coordinates
    and extracts bounding box parameters for BEVDet training.

    Note on camera selection: NuScenes synchronizes all camera sensors to the 
    same timestamp. Consequently, the ego-vehicle's pose relative to the 
    global frame is the same regardless of which camera's metadata is 
    accessed. We pick the first available camera to retrieve the 
    ego2global transformation.

    Args:
        info (dict): Sample info dict containing 'cams' and 'ann_infos'.
        detection_classes (list, optional): List of detection class names.
            Defaults to DETECTION_CLASSES.
        category_mapping (dict, optional): Mapping from category names to 
            detection class names. Defaults to MAP_CATEGORY_TO_DETECTION.

    Returns:
        tuple: (gt_boxes, gt_labels) where:
            - gt_boxes: List of 9-element arrays [x, y, z, dx, dy, dz, yaw, vx, vy]
            - gt_labels: List of class indices
    """
    if detection_classes is None:
        detection_classes = DETECTION_CLASSES
    if category_mapping is None:
        category_mapping = MAP_CATEGORY_TO_DETECTION
        
    # Choose first available camera to get ego2global transformation.
    cam_info = next(iter(info['cams'].values()))
    ego2global_rotation = cam_info['ego2global_rotation']
    ego2global_translation = cam_info['ego2global_translation']
    trans = -np.array(ego2global_translation)
    rot = Quaternion(ego2global_rotation).inverse
    
    gt_boxes = []
    gt_labels = []
    
    for ann_info in info['ann_infos']:
        # Get detection class name, default to 'ignore' for unknown categories
        det_class = category_mapping.get(ann_info['category_name'], 'ignore')
        
        # Skip ignored classes
        if det_class not in detection_classes:
            continue
            
        # Create box in global coordinates
        box = Box(
            ann_info['translation'],
            ann_info['size'],
            Quaternion(ann_info['rotation']),
            velocity=ann_info.get('velocity', [0, 0, 0]),
        )
        
        # Transform to ego coordinates
        box.translate(trans)
        box.rotate(rot)
        
        # Extract box parameters
        box_xyz = np.array(box.center)
        box_dxdydz = np.array(box.wlh)[[1, 0, 2]]  # Convert wlh to dxdydz
        box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
        box_velo = np.array(box.velocity[:2])
        
        gt_box = np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])
        gt_boxes.append(gt_box)
        gt_labels.append(detection_classes.index(det_class))
        
    return gt_boxes, gt_labels


def add_ann_adj_info(data_path, info_prefix, version, out_dir, 
                     detection_classes=None, category_mapping=None):
    """Add adjacent annotation info to pkl files for BEVDet training.

    Enriches the info pkl files with:
    - Velocity information for each annotation
    - Ego-coordinate ground truth boxes (via get_gt)
    - Scene tokens
    - Occupancy paths

    Args:
        data_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version (e.g., 'v1.0-mini', 'v1.0-trainval').
        out_dir (str): Output directory where pkl files are stored.
        detection_classes (list, optional): List of detection class names.
            Defaults to DETECTION_CLASSES.
        category_mapping (dict, optional): Mapping from category names to
            detection class names. Defaults to MAP_CATEGORY_TO_DETECTION.
    """
    if detection_classes is None:
        detection_classes = DETECTION_CLASSES
    if category_mapping is None:
        category_mapping = MAP_CATEGORY_TO_DETECTION
        
    nuscenes = NuScenes(version, data_path)
    
    # Determine which sets to process based on version
    if version == 'v1.0-mini':
        sets = ['mini']
    elif version == 'v1.0-test':
        sets = ['test']
    else:
        sets = ['train', 'val']
    
    for set_name in sets:
        pkl_path = osp.join(out_dir, f'{info_prefix}_infos_{set_name}.pkl')
        if not osp.exists(pkl_path):
            print(f"Skipping {pkl_path} (not found)")
            continue
            
        print(f"Adding annotation info to {pkl_path}")
        with open(pkl_path, 'rb') as f:
            dataset = pickle.load(f)
            
        for idx in range(len(dataset['infos'])):
            # if idx % 10 == 0:
                # print(f'{idx}/{len(dataset["infos"])}')
            info = dataset['infos'][idx]
            
            # Get sample and annotation info from nuScenes
            sample = nuscenes.get('sample', info['token'])
            ann_infos = []
            for ann in sample['anns']:
                ann_info = nuscenes.get('sample_annotation', ann)
                velocity = nuscenes.box_velocity(ann_info['token'])
                if np.any(np.isnan(velocity)):
                    velocity = np.zeros(3)
                ann_info['velocity'] = velocity
                ann_infos.append(ann_info)
            
            # Store raw annotation infos temporarily for get_gt
            dataset['infos'][idx]['ann_infos'] = ann_infos
            
            # Compute ego-coordinate GT boxes and labels
            dataset['infos'][idx]['ann_infos'] = get_gt(
                dataset['infos'][idx], 
                detection_classes=detection_classes,
                category_mapping=category_mapping
            )
            
            # Add scene token
            dataset['infos'][idx]['scene_token'] = sample['scene_token']

        with open(pkl_path, 'wb') as fid:
            pickle.dump(dataset, fid)
        print(f"Saved {pkl_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Enrich nuScenes pkl info files with BEVDet annotation data.'
    )
    parser.add_argument('data_path', help='Path to the nuScenes dataset root')
    parser.add_argument('pkl_dir', help='Directory containing the pkl info files')
    parser.add_argument('--prefix', default='pccr', help='Info file prefix (default: pccr)')
    parser.add_argument('--version', default='v1.0-trainval', help='Dataset version')
    parser.add_argument('--sets', nargs='+', default=['train', 'val'],
                        help='Splits to process (default: train val)')
    _args = parser.parse_args()
    add_ann_adj_info(
        data_path=_args.data_path,
        info_prefix=_args.prefix,
        out_dir=_args.pkl_dir,
        version=_args.version,
        sets=_args.sets,
    )
