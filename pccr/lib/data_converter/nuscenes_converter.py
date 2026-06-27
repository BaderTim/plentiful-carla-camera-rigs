import json
import pickle
import numpy as np
import cv2 as _cv2
import os
from collections import OrderedDict
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from os import path as osp
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box
from typing import List, Tuple, Union

from mmdet3d.core.bbox.box_np_ops import points_cam2img

from ..nuscenes_standards import NuScenesStandards
nus_categories = tuple([cat['name'] for cat in NuScenesStandards.get_categories()])

nus_attributes = ('moving', 'parked', 'stopped',
                  'adult_moving', 'adult_standing',
                  'with_rider', 'without_rider',
                  'traffic_light_red', 'traffic_light_yellow',
                  'traffic_light_green', 'traffic_light_off',
                  'traffic_light_unknown')


def _process_path(path, dataroot, prefix=None):
    """Process file path to be relative to dataroot, optionally with a prefix."""
    path = str(path)
    dataroot = os.path.abspath(dataroot)
    abs_path = os.path.abspath(path)
    
    if abs_path.startswith(dataroot):
        rel_path = os.path.relpath(abs_path, dataroot)
    else:
        # Fallback if path is outside dataroot for some reason
        if os.getcwd() in path:
            rel_path = path.split(f'{os.getcwd()}/')[-1]
        else:
            rel_path = path
            
    if prefix:
        return os.path.join(prefix, rel_path)
    return rel_path


def create_nuscenes_infos(root_path,
                          info_prefix,
                          version='v1.0-trainval',
                          max_sweeps=10,
                          out_dir=None,
                          no_lidar=False,
                          root_path_prefix=None):
    """Create info file of nuScenes dataset.

    Generates train/val pkl info files from a nuScenes-formatted dataset root.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Dataset version. One of ``'v1.0-trainval'``,
            ``'v1.0-test'``, ``'v1.0-mini'``. Default: ``'v1.0-trainval'``.
        max_sweeps (int): Maximum number of LiDAR sweeps to include.
            Default: 10.
        out_dir (str): Output directory for pkl files. Defaults to
            *root_path*.
        no_lidar (bool): Skip LiDAR-related processing when ``True``.
        root_path_prefix (str): Path prefix stored in the pkl files.
    """
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    from nuscenes.utils import splits
    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    assert version in available_vers
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError('unknown')

    # filter existing scenes.
    available_scenes = get_available_scenes(nusc, no_lidar=no_lidar)
    available_scene_names = [s['name'] for s in available_scenes]
    
    # Original split logic
    target_train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    target_val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))

    # Fallback for custom datasets where scene names might be 'train_01', 'val_04', etc.
    if not target_train_scenes and not target_val_scenes and available_scenes:
        if version == 'v1.0-test':
            target_train_scenes = [s['name'] for s in available_scenes if 'test' in s['name'].lower()]
            if not target_train_scenes: target_train_scenes = available_scene_names
        elif version == 'v1.0-trainval':
            target_train_scenes = [s['name'] for s in available_scenes if 'train' in s['name'].lower()]
            target_val_scenes = [s['name'] for s in available_scenes if 'val' in s['name'].lower()]
            if not target_train_scenes and not target_val_scenes: target_train_scenes = available_scene_names
        elif version == 'v1.0-mini':
            target_train_scenes = [s['name'] for s in available_scenes if 'train' in s['name'].lower()]
            target_val_scenes = [s['name'] for s in available_scenes if 'val' in s['name'].lower()]
            if not target_train_scenes and not target_val_scenes: target_train_scenes = available_scene_names

    train_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in target_train_scenes
    ])
    val_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in target_val_scenes
    ])

    test = 'test' in version
    if test:
        print('test scene: {}'.format(len(train_scenes)))
    else:
        print('train scene: {}, val scene: {}'.format(
            len(train_scenes), len(val_scenes)))
    
    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc, train_scenes, val_scenes, test, max_sweeps=max_sweeps,
        no_lidar=no_lidar, root_path_prefix=root_path_prefix)


    metadata = dict(version=version)
    if out_dir is None:
        out_dir = root_path

    if test:
        print('test sample: {}'.format(len(train_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(out_dir,
                             '{}_infos_test.pkl'.format(info_prefix))
        _dump_pkl(data, info_path)
    else:
        print(info_prefix)
        print('train sample: {}, val sample: {}'.format(
            len(train_nusc_infos), len(val_nusc_infos)))
        
        if version == 'v1.0-mini':
            # For mini, merge train and val into a single info file as requested
            data = dict(infos=train_nusc_infos + val_nusc_infos, metadata=metadata)
            info_path = osp.join(out_dir, '{}_infos_mini.pkl'.format(info_prefix))
            _dump_pkl(data, info_path)
        else:
            data = dict(infos=train_nusc_infos, metadata=metadata)
            info_path = osp.join(out_dir, '{}_infos_train.pkl'.format(info_prefix))
            _dump_pkl(data, info_path)
            
            data['infos'] = val_nusc_infos
            info_val_path = osp.join(out_dir, '{}_infos_val.pkl'.format(info_prefix))
            _dump_pkl(data, info_val_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dump_pkl(data: dict, path: str) -> None:
    """Serialize *data* to a pickle file at *path*.

    Args:
        data: Python object to serialise.
        path: Destination file path.
    """
    with open(path, 'wb') as fh:
        pickle.dump(data, fh)
    print(f'Saved {path}')


def _load_pkl(path: str) -> dict:
    """Load and return the contents of a pickle file.

    Args:
        path: Source file path.

    Returns:
        Deserialised Python object.
    """
    with open(path, 'rb') as fh:
        return pickle.load(fh)


def get_available_scenes(nusc, no_lidar=False):
    """Get available scenes from the input nuscenes class.
    Given the raw data, get the information of available scenes for
    further info generation.
    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        no_lidar (bool): Whether to skip lidar-related processing.
    Returns:
        available_scenes (list[dict]): List of basic information for the
            available scenes.
    """
    available_scenes = []
    print('total scene num: {}'.format(len(nusc.scene)))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        
        # If no_lidar is requested, we don't require LIDAR_TOP to exist or be valid
        if no_lidar:
            # For camera-only, just verify at least one camera exists
            has_camera = False
            for sensor_name, sd_token in sample_rec['data'].items():
                if 'CAM' in sensor_name:
                    cam_path, _, _ = nusc.get_sample_data(sd_token)
                    if osp.exists(cam_path):
                        has_camera = True
                        break
            if has_camera:
                available_scenes.append(scene)
            continue

        if 'LIDAR_TOP' not in sample_rec['data']:
            continue
            
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                # path from lyftdataset is absolute path
                lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
                # relative path
            if not osp.exists(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num: {}'.format(len(available_scenes)))
    return available_scenes


_MAX_RADAR_SWEEPS = 10  # Internal constant; no longer a public parameter.


def _fill_trainval_infos(nusc,
                         train_scenes,
                         val_scenes,
                         test=False,
                         max_sweeps=10,
                         no_lidar=False,
                         root_path_prefix=None):
    """Generate train/val info dicts from the raw nuScenes data.

    Args:
        nusc: :class:`~nuscenes.nuscenes.NuScenes` dataset instance.
        train_scenes (list[str]): Scene tokens belonging to the training set.
        val_scenes (list[str]): Scene tokens belonging to the validation set.
        test (bool): Test mode — annotations are not included. Default: ``False``.
        max_sweeps (int): Maximum number of LiDAR sweeps. Default: 10.
        no_lidar (bool): Skip LiDAR-related processing. Default: ``False``.
        root_path_prefix (str): Path prefix stored in the pkl files.

    Returns:
        tuple[list[dict], list[dict]]: Training and validation info lists.
    """
    train_nusc_infos = []
    val_nusc_infos = []
    token2idx = {}

    n_samples = len(nusc.sample)
    for sample_idx, sample in enumerate(nusc.sample):
        if sample_idx % 50 == 0:
            print(f'  Processing sample {sample_idx + 1}/{n_samples}...')
        # Determine the reference sensor (usually LIDAR_TOP)
        if 'LIDAR_TOP' in sample['data']:
            lidar_token = sample['data']['LIDAR_TOP']
        else:
            # Fallback to another sensor if LIDAR_TOP is missing
            lidar_token = next(iter(sample['data'].values()))
            
        sd_rec = nusc.get('sample_data', lidar_token)
        cs_record = nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

        # Handle path prefixing
        lidar_path = _process_path(lidar_path, nusc.dataroot, prefix=root_path_prefix)

        info = {
            'lidar_path': lidar_path,
            'token': sample['token'],
            'sweeps': [],
            'cams': dict(),
            'radars': dict(), 
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'timestamp': sample['timestamp'],
            'prev_token': sample['prev']
        }

        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        # Dynamically find all cameras and radars
        camera_types = []
        radar_names = []
        for sensor_name, sensor_token in sample['data'].items():
            sensor_rec = nusc.get('sample_data', sensor_token)
            if sensor_rec['sensor_modality'] == 'camera':
                camera_types.append(sensor_name)
            elif sensor_rec['sensor_modality'] == 'radar':
                radar_names.append(sensor_name)

        for cam in camera_types:
            cam_token = sample['data'][cam]
            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            
            # Use the same path prefixing logic for cameras
            cam_path = _process_path(cam_path, nusc.dataroot, prefix=root_path_prefix)

            cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat,
                                         e2g_t, e2g_r_mat, cam, root_path_prefix=root_path_prefix)
            cam_info.update(cam_intrinsic=cam_intrinsic, data_path=cam_path)
            info['cams'].update({cam: cam_info})

        for radar_name in radar_names:
            radar_token = sample['data'][radar_name]
            radar_rec = nusc.get('sample_data', radar_token)
            sweeps = []

            while len(sweeps) < _MAX_RADAR_SWEEPS:
                if not radar_rec['prev'] == '':
                    radar_path, _, radar_intrin = nusc.get_sample_data(radar_token)

                    radar_info = obtain_sensor2top(nusc, radar_token, l2e_t, l2e_r_mat,
                                                e2g_t, e2g_r_mat, radar_name, root_path_prefix=root_path_prefix)
                    sweeps.append(radar_info)
                    radar_token = radar_rec['prev']
                    radar_rec = nusc.get('sample_data', radar_token)
                else:
                    radar_path, _, radar_intrin = nusc.get_sample_data(radar_token)

                    radar_info = obtain_sensor2top(nusc, radar_token, l2e_t, l2e_r_mat,
                                                e2g_t, e2g_r_mat, radar_name, root_path_prefix=root_path_prefix)
                    sweeps.append(radar_info)
            
            info['radars'].update({radar_name: sweeps})
        # obtain sweeps for a single key-frame
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec['prev'] == '':
                sweep = obtain_sensor2top(nusc, sd_rec['prev'], l2e_t,
                                          l2e_r_mat, e2g_t, e2g_r_mat, 'lidar', root_path_prefix=root_path_prefix)
                sweeps.append(sweep)
                sd_rec = nusc.get('sample_data', sd_rec['prev'])
            else:
                break
        info['sweeps'] = sweeps
        # obtain annotation
        # Attempt to load annotations if they exist (even for test split in custom datasets)
        try:
            # 1. Get the source of truth (the boxes the SDK actually sees)
            lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)
            
            # 2. Derive metadata directly from these specific boxes using their tokens
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0]
                             for b in boxes]).reshape(-1, 1)
            
            # Calculate velocity and attributes per box token, guaranteed length match
            velocity = np.array([nusc.box_velocity(b.token)[:2] for b in boxes])
            
            # Use 'boxes' to fetch the annotation records
            ann_recs = [nusc.get('sample_annotation', b.token) for b in boxes]
            
            attributes = [
                nusc.get('attribute', ann['attribute_tokens'][0])['name']
                if len(ann['attribute_tokens']) > 0 else ''
                for ann in ann_recs
            ]

            if no_lidar:
                # Filter based on visibility level (1, 2, 3, 4)
                # This ensures we don't filter out samples with 0 lidar points
                valid_flag = np.array(
                    [ann['visibility_token'] in ['1', '2', '3', '4'] for ann in ann_recs],
                    dtype=bool).reshape(-1)
            else:
                valid_flag = np.array(
                    [(ann['num_lidar_pts'] + ann['num_radar_pts']) > 0
                     for ann in ann_recs],
                    dtype=bool).reshape(-1)

            # 3. Perform coordinate transformation on the aligned velocity array
            for i in range(len(boxes)):
                velo = np.array([*velocity[i], 0.0])
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                    l2e_r_mat).T
                velocity[i] = velo[:2]

            names = [b.name for b in boxes]
            # Replace NameMapping logic with direct filter/clean
            valid_indices = []
            for i, name in enumerate(names):
                if name in nus_categories:
                    valid_indices.append(i)
            
            # Filter all annotation arrays to only keep classes in nus_categories
            if len(valid_indices) > 0:
                gt_boxes = np.concatenate([locs, dims[:, [1, 0, 2]], rots], axis=1)[valid_indices]
                names = np.array(names)[valid_indices]
                attributes = np.array(attributes)[valid_indices]
                velocity = velocity[valid_indices].reshape(-1, 2)
                num_lidar_pts = np.array([a['num_lidar_pts'] for a in ann_recs])[valid_indices]
                num_radar_pts = np.array([a['num_radar_pts'] for a in ann_recs])[valid_indices]
                valid_flag = valid_flag[valid_indices]

                info['gt_boxes'] = gt_boxes
                info['gt_names'] = names
                info['gt_attributes'] = attributes
                info['gt_velocity'] = velocity
                info['num_lidar_pts'] = num_lidar_pts
                info['num_radar_pts'] = num_radar_pts
                info['valid_flag'] = valid_flag
            else:
                info['gt_boxes'] = np.zeros((0, 7))
                info['gt_names'] = np.array([])
                info['gt_attributes'] = np.array([])
                info['gt_velocity'] = np.zeros((0, 2))
                info['num_lidar_pts'] = np.array([])
                info['num_radar_pts'] = np.array([])
                info['valid_flag'] = np.array([], dtype=bool)

        except Exception as e:
            # If no annotations are found or any error occurs, just skip GT for this frame
            pass

        if sample['scene_token'] in train_scenes:
            train_nusc_infos.append(info)
            token2idx[info['token']] = ('train', len(train_nusc_infos) - 1)
        else:
            val_nusc_infos.append(info)
            token2idx[info['token']] = ('val', len(val_nusc_infos) - 1)
    
    for info in train_nusc_infos:
        prev_token = info['prev_token']
        if prev_token == '':
            info['prev'] = -1
        else:
            prev_set, prev_idx = token2idx[prev_token]
            assert prev_set == 'train'
            info['prev'] = prev_idx

    for info in val_nusc_infos:
        prev_token = info['prev_token']
        if prev_token == '':
            info['prev'] = -1
        else:
            prev_set, prev_idx = token2idx[prev_token]
            assert prev_set == 'val'
            info['prev'] = prev_idx

    return train_nusc_infos, val_nusc_infos


def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar',
                      root_path_prefix=None):
    """Obtain the info with RT matric from general sensor to Top LiDAR.
    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.
        root_path_prefix (str): Path prefix to be saved in pkl files.
    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))

    data_path = _process_path(data_path, nusc.dataroot, prefix=root_path_prefix)

    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }
    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep['sensor2lidar_translation'] = T
    return sweep


def export_2d_annotation(root_path, info_path, version, mono3d=True):
    """Export 2d annotation from the info file and raw data.
    Args:
        root_path (str): Root path of the raw data.
        info_path (str): Path of the info file.
        version (str): Dataset version.
        mono3d (bool): Whether to export mono3d annotation. Default: True.
    """
    nusc_infos = _load_pkl(info_path)['infos']
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    
    # Dynamically determine camera types from the first info record
    if len(nusc_infos) > 0:
        camera_types = list(nusc_infos[0]['cams'].keys())
    else:
        camera_types = []

    cat2Ids = [
        dict(id=i, name=cat_name)
        for i, cat_name in enumerate(nus_categories)
    ]
    coco_ann_id = 0
    coco_2d_dict = dict(annotations=[], images=[], categories=cat2Ids)
    n_infos = len(nusc_infos)
    for info_idx, info in enumerate(nusc_infos):
        if info_idx % 50 == 0:
            print(f'  Processing info {info_idx + 1}/{n_infos}...')
        for cam in camera_types:
            cam_info = info['cams'][cam]
            coco_infos = get_2d_boxes(
                nusc,
                cam_info['sample_data_token'],
                visibilities=['', '1', '2', '3', '4'],
                mono3d=mono3d)
            _img = _cv2.imread(cam_info['data_path'])
            height, width = (_img.shape[:2] if _img is not None
                             else (cam_info.get('height', 720),
                                   cam_info.get('width', 1280)))
            
            # Use the filename directly as it is already relative to the workspace/dataset root
            file_name = cam_info['data_path']
            
            coco_2d_dict['images'].append(
                dict(
                    file_name=file_name,
                    id=cam_info['sample_data_token'],
                    token=info['token'],
                    cam2ego_rotation=cam_info['sensor2ego_rotation'],
                    cam2ego_translation=cam_info['sensor2ego_translation'],
                    ego2global_rotation=info['ego2global_rotation'],
                    ego2global_translation=info['ego2global_translation'],
                    cam_intrinsic=cam_info['cam_intrinsic'],
                    width=width,
                    height=height))
            for coco_info in coco_infos:
                if coco_info is None:
                    continue
                # add an empty key for coco format
                coco_info['segmentation'] = []
                coco_info['id'] = coco_ann_id
                coco_2d_dict['annotations'].append(coco_info)
                coco_ann_id += 1

    if mono3d:
        json_prefix = f'{info_path[:-4]}_mono3d'
    else:
        json_prefix = f'{info_path[:-4]}'
    json_out = f'{json_prefix}.coco.json'
    with open(json_out, 'w') as fh:
        json.dump(coco_2d_dict, fh, indent=2)
    print(f'Saved {json_out}')


def get_2d_boxes(nusc,
                 sample_data_token: str,
                 visibilities: List[str],
                 mono3d=True):
    """Get the 2D annotation records for a given `sample_data_token`.
    Args:
        sample_data_token (str): Sample data token belonging to a camera \
            keyframe.
        visibilities (list[str]): Visibility filter.
        mono3d (bool): Whether to get boxes with mono3d annotation.
    Return:
        list[dict]: List of 2D annotation record that belongs to the input
            `sample_data_token`.
    """

    # Get the sample data and the sample corresponding to that sample data.
    sd_rec = nusc.get('sample_data', sample_data_token)

    assert sd_rec[
        'sensor_modality'] == 'camera', 'Error: get_2d_boxes only works' \
        ' for camera sample_data!'
    if not sd_rec['is_key_frame']:
        raise ValueError(
            'The 2D re-projections are available only for keyframes.')

    s_rec = nusc.get('sample', sd_rec['sample_token'])

    # Get the calibrated sensor and ego pose
    # record to get the transformation matrices.
    cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    camera_intrinsic = np.array(cs_rec['camera_intrinsic'])

    # Get all the annotation with the specified visibilties.
    ann_recs = [
        nusc.get('sample_annotation', token) for token in s_rec['anns']
    ]
    ann_recs = [
        ann_rec for ann_rec in ann_recs
        if (ann_rec['visibility_token'] in visibilities)
    ]

    repro_recs = []

    for ann_rec in ann_recs:
        # Augment sample_annotation with token information.
        ann_rec['sample_annotation_token'] = ann_rec['token']
        ann_rec['sample_data_token'] = sample_data_token

        # Get the box in global coordinates.
        box = nusc.get_box(ann_rec['token'])

        # Move them to the ego-pose frame.
        box.translate(-np.array(pose_rec['translation']))
        box.rotate(Quaternion(pose_rec['rotation']).inverse)

        # Move them to the calibrated sensor frame.
        box.translate(-np.array(cs_rec['translation']))
        box.rotate(Quaternion(cs_rec['rotation']).inverse)

        # Filter out the corners that are not in front of the calibrated
        # sensor.
        corners_3d = box.corners()
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]

        # Project 3d box to 2d.
        corner_coords = view_points(corners_3d, camera_intrinsic,
                                    True).T[:, :2].tolist()

        # Keep only corners that fall within the image.
        final_coords = post_process_coords(corner_coords)

        # Skip if the convex hull of the re-projected corners
        # does not intersect the image canvas.
        if final_coords is None:
            continue
        else:
            min_x, min_y, max_x, max_y = final_coords

        # Generate dictionary record to be included in the .json file.
        repro_rec = generate_record(ann_rec, min_x, min_y, max_x, max_y,
                                    sample_data_token, sd_rec['filename'])

        # If mono3d=True, add 3D annotations in camera coordinates
        if mono3d and (repro_rec is not None):
            loc = box.center.tolist()
            dim = box.wlh.tolist()
            rot = [box.orientation.yaw_pitch_roll[0]]

            global_velo2d = nusc.box_velocity(box.token)[:2]
            global_velo3d = np.array([*global_velo2d, 0.0])
            e2g_r_mat = Quaternion(pose_rec['rotation']).rotation_matrix
            c2e_r_mat = Quaternion(cs_rec['rotation']).rotation_matrix
            cam_velo3d = global_velo3d @ np.linalg.inv(
                e2g_r_mat).T @ np.linalg.inv(c2e_r_mat).T
            velo = cam_velo3d[0::2].tolist()

            repro_rec['bbox_cam3d'] = loc + dim + rot
            repro_rec['velo_cam3d'] = velo

            center3d = np.array(loc).reshape([1, 3])
            center2d = points_cam2img(
                center3d, camera_intrinsic, with_depth=True)
            repro_rec['center2d'] = center2d.squeeze().tolist()
            # normalized center2D + depth
            # if samples with depth < 0 will be removed
            if repro_rec['center2d'][2] <= 0:
                continue
            
            ann_token = nusc.get('sample_annotation',
                                 box.token)['attribute_tokens']
            if len(ann_token) == 0:
                attr_name = 'None'
            else:
                attr_name = nusc.get('attribute', ann_token[0])['name']
            
            if attr_name in nus_attributes:
                attr_id = nus_attributes.index(attr_name)
                repro_rec['attribute_name'] = attr_name
                repro_rec['attribute_id'] = attr_id
            else:
                repro_rec['attribute_name'] = 'None'
                repro_rec['attribute_id'] = -1

        repro_recs.append(repro_rec)

    return repro_recs


def post_process_coords(
    corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
) -> Union[Tuple[float, float, float, float], None]:
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.
    Args:
        corner_coords (list[int]): Corner coordinates of reprojected
            bounding box.
        imsize (tuple[int]): Size of the image canvas.
    Return:
        tuple [float]: Intersection of the convex hull of the 2D box
            corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None


def generate_record(ann_rec: dict, x1: float, y1: float, x2: float, y2: float,
                    sample_data_token: str, filename: str) -> OrderedDict:
    """Generate one 2D annotation record given various informations on top of
    the 2D bounding box coordinates.
    Args:
        ann_rec (dict): Original 3d annotation record.
        x1 (float): Minimum value of the x coordinate.
        y1 (float): Minimum value of the y coordinate.
        x2 (float): Maximum value of the x coordinate.
        y2 (float): Maximum value of the y coordinate.
        sample_data_token (str): Sample data token.
        filename (str):The corresponding image file where the annotation
            is present.
    Returns:
        dict: A sample 2D annotation record.
            - file_name (str): flie name
            - image_id (str): sample data token
            - area (float): 2d box area
            - category_name (str): category name
            - category_id (int): category id
            - bbox (list[float]): left x, top y, dx, dy of 2d box
            - iscrowd (int): whether the area is crowd
    """
    repro_rec = OrderedDict()
    repro_rec['sample_data_token'] = sample_data_token
    coco_rec = dict()

    relevant_keys = [
        'attribute_tokens',
        'category_name',
        'instance_token',
        'next',
        'num_lidar_pts',
        'num_radar_pts',
        'prev',
        'sample_annotation_token',
        'sample_data_token',
        'visibility_token',
    ]

    for key, value in ann_rec.items():
        if key in relevant_keys:
            repro_rec[key] = value

    repro_rec['bbox_corners'] = [x1, y1, x2, y2]
    repro_rec['filename'] = filename

    coco_rec['file_name'] = filename
    coco_rec['image_id'] = sample_data_token
    coco_rec['area'] = (y2 - y1) * (x2 - x1)

    # Use direct list check instead of NameMapping
    cat_name = repro_rec['category_name']
    if cat_name not in nus_categories:
        return None
    
    coco_rec['category_name'] = cat_name
    coco_rec['category_id'] = nus_categories.index(cat_name)
    coco_rec['bbox'] = [x1, y1, x2 - x1, y2 - y1]
    coco_rec['iscrowd'] = 0

    return coco_rec


if __name__ == '__main__':
    create_nuscenes_infos('data/nuscenes/', 'radar_nuscenes_5sweeps')