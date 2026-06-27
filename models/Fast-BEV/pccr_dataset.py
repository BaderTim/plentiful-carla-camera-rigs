# -*- coding: utf-8 -*-
import tempfile
from os import path as osp
from contextlib import contextmanager

import mmcv
import numpy as np
import pyquaternion
from nuscenes.utils.data_classes import Box as NuScenesBox
from pyquaternion import Quaternion

from mmdet.datasets import DATASETS
from ..core.bbox import LiDARInstance3DBoxes
from .nuscenes_dataset import NuScenesDataset, lidar_nusc_box_to_global


def tofloat(x):
    return x.astype(np.float32) if x is not None else None


@contextmanager
def pccr_nuscenes_eval_scope(class_names,
                             eval_set=None,
                             scene_names=None,
                             attribute_names=None):
    from nuscenes.eval.common import loaders as common_loaders
    from nuscenes.eval.detection import constants as det_constants
    from nuscenes.eval.detection import data_classes as det_data_classes
    from nuscenes.eval.detection import utils as det_utils

    class_names = list(class_names)

    def category_to_detection_name(category_name):
        return category_name if category_name in class_names else None

    old_constants_names = det_constants.DETECTION_NAMES
    old_data_class_names = det_data_classes.DETECTION_NAMES
    old_attribute_names = det_data_classes.ATTRIBUTE_NAMES
    old_utils_mapper = det_utils.category_to_detection_name
    old_loaders_mapper = common_loaders.category_to_detection_name
    old_create_splits_scenes = common_loaders.create_splits_scenes

    def create_splits_scenes():
        splits = old_create_splits_scenes()
        if eval_set is not None and scene_names is not None:
            splits[eval_set] = list(scene_names)
        return splits

    det_constants.DETECTION_NAMES = class_names
    det_data_classes.DETECTION_NAMES = class_names
    if attribute_names is not None:
        det_data_classes.ATTRIBUTE_NAMES = list(attribute_names)
    det_utils.category_to_detection_name = category_to_detection_name
    common_loaders.category_to_detection_name = category_to_detection_name
    common_loaders.create_splits_scenes = create_splits_scenes
    try:
        yield
    finally:
        det_constants.DETECTION_NAMES = old_constants_names
        det_data_classes.DETECTION_NAMES = old_data_class_names
        det_data_classes.ATTRIBUTE_NAMES = old_attribute_names
        det_utils.category_to_detection_name = old_utils_mapper
        common_loaders.category_to_detection_name = old_loaders_mapper
        common_loaders.create_splits_scenes = old_create_splits_scenes


@DATASETS.register_module()
class PCCRMultiViewDataset(NuScenesDataset):
    CLASSES = (
        'car', 'truck', 'bus', 'bicycle', 'motorcycle', 'adult', 'child',
        'traffic_light', 'traffic_sign')
    DefaultAttribute = {name: '' for name in CLASSES}
    ErrNameMapping = {
        'trans_err': 'mATE',
        'scale_err': 'mASE',
        'orient_err': 'mAOE'
    }

    def __init__(self, class_range=None, default_det_range=80, **kwargs):
        kwargs.setdefault('with_velocity', False)
        kwargs.setdefault('prev_only', True)
        kwargs.setdefault('next_only', False)
        kwargs.setdefault('fix_direction', True)
        super().__init__(**kwargs)

        self.token2idx = {
            info['token']: idx for idx, info in enumerate(self.data_infos)
        }
        if class_range is None:
            class_range = {name: default_det_range for name in self.CLASSES}
        self.eval_detection_configs.class_names = list(self.CLASSES)
        self.eval_detection_configs.class_range = {
            name: class_range.get(name, default_det_range)
            for name in self.CLASSES
        }

    def _resolve_path(self, file_path):
        if not file_path:
            return file_path
        if osp.isabs(file_path):
            return file_path
        return osp.join(self.data_root, file_path)

    def _resolve_prev_index(self, index, offset):
        info = self.data_infos[index]
        if offset <= 0:
            return index

        prev_indices = []
        prev_ref = info.get('prev', -1)
        visited = set()
        while True:
            if isinstance(prev_ref, int):
                if prev_ref < 0 or prev_ref >= len(self.data_infos) or prev_ref in visited:
                    break
                prev_index = prev_ref
            elif isinstance(prev_ref, str):
                if prev_ref not in self.token2idx or prev_ref in visited:
                    break
                prev_index = self.token2idx[prev_ref]
            else:
                break
            prev_indices.append(prev_index)
            visited.add(prev_index)
            prev_ref = self.data_infos[prev_index].get('prev', -1)

        if not prev_indices:
            return index
        select_id = min(offset, len(prev_indices)) - 1
        return prev_indices[select_id]

    def _adjacent_offsets(self):
        if self.test_mode and self.test_adj_ids is not None:
            return [abs(x) for x in self.test_adj_ids]
        if not self.test_mode and self.train_adj_ids is not None:
            return [abs(x) for x in self.train_adj_ids]
        return list(range(1, self.n_times))

    def _build_camera_views(self, info):
        image_paths = []
        lidar2img_rts = []
        lidar2img_augs = []
        lidar2img_extras = []
        cam_order = list(info['cams'].keys())
        kws = [
            'sensor2ego_translation',
            'sensor2ego_rotation',
            'ego2global_translation',
            'ego2global_rotation',
            'sensor2lidar_rotation',
            'sensor2lidar_translation',
            'cam_intrinsic'
        ]

        for cam_name in cam_order:
            cam_info = info['cams'][cam_name]
            image_paths.append(self._resolve_path(cam_info['data_path']))

            lidar2img_extra = {kw: cam_info[kw] for kw in kws}
            lidar2img_extras.append(lidar2img_extra)

            intrinsic = cam_info['cam_intrinsic']
            lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
            lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4, dtype=np.float32)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t

            lidar2img_aug = {
                'intrin': intrinsic,
                'rot': cam_info['sensor2lidar_rotation'],
                'tran': cam_info['sensor2lidar_translation'],
                'post_rot': np.eye(3, dtype=np.float32),
                'post_tran': np.zeros(3, dtype=np.float32),
            }
            lidar2img_augs.append(lidar2img_aug)

            viewpad = np.eye(4, dtype=np.float32)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rts.append(viewpad @ lidar2cam_rt.T)

        return cam_order, image_paths, lidar2img_rts, lidar2img_augs, lidar2img_extras

    def _build_adjacent_views(self, curr_info, adj_info, cam_order):
        image_paths = []
        lidar2img_rts = []
        lidar2img_augs = []
        lidar2img_extras = []

        egocurr2global = np.eye(4, dtype=np.float32)
        egocurr2global[:3, :3] = Quaternion(curr_info['ego2global_rotation']).rotation_matrix
        egocurr2global[:3, 3] = curr_info['ego2global_translation']
        egoadj2global = np.eye(4, dtype=np.float32)
        egoadj2global[:3, :3] = Quaternion(adj_info['ego2global_rotation']).rotation_matrix
        egoadj2global[:3, 3] = adj_info['ego2global_translation']
        lidar2ego = np.eye(4, dtype=np.float32)
        lidar2ego[:3, :3] = Quaternion(curr_info['lidar2ego_rotation']).rotation_matrix
        lidar2ego[:3, 3] = curr_info['lidar2ego_translation']
        lidaradj2lidarcurr = np.linalg.inv(lidar2ego) @ np.linalg.inv(egocurr2global) \
            @ egoadj2global @ lidar2ego

        for cam_name in cam_order:
            cam_info = adj_info['cams'][cam_name]
            image_paths.append(self._resolve_path(cam_info['data_path']))

            lidar2img_aug = {
                'intrin': cam_info['cam_intrinsic'],
                'rot': cam_info['sensor2lidar_rotation'],
                'tran': cam_info['sensor2lidar_translation'],
                'post_rot': np.eye(3, dtype=np.float32),
                'post_tran': np.zeros(3, dtype=np.float32),
            }
            mat = np.eye(4, dtype=np.float32)
            mat[:3, :3] = lidar2img_aug['rot']
            mat[:3, 3] = lidar2img_aug['tran']
            mat = lidaradj2lidarcurr @ mat
            lidar2img_aug['rot'] = mat[:3, :3]
            lidar2img_aug['tran'] = mat[:3, 3]
            lidar2img_augs.append(lidar2img_aug)

            lidar2cam_r = np.linalg.inv(lidar2img_aug['rot'])
            lidar2cam_t = lidar2img_aug['tran'] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4, dtype=np.float32)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            viewpad = np.eye(4, dtype=np.float32)
            intrin = lidar2img_aug['intrin']
            viewpad[:intrin.shape[0], :intrin.shape[1]] = intrin
            lidar2img_rts.append(viewpad @ lidar2cam_rt.T)

            lidar2img_extras.append({
                'ego2global_translation': adj_info['ego2global_translation'],
                'ego2global_rotation': adj_info['ego2global_rotation'],
            })

        return image_paths, lidar2img_rts, lidar2img_augs, lidar2img_extras

    def get_data_info(self, index):
        info = self.data_infos[index]
        cam_order, image_paths, lidar2img_rts, lidar2img_augs, lidar2img_extras = \
            self._build_camera_views(info)

        if self.sequential:
            for offset in self._adjacent_offsets():
                adj_index = self._resolve_prev_index(index, offset)
                adj_info = self.data_infos[adj_index]
                adj_paths, adj_rts, adj_augs, adj_extras = self._build_adjacent_views(
                    info, adj_info, cam_order)
                image_paths.extend(adj_paths)
                lidar2img_rts.extend(adj_rts)
                lidar2img_augs.extend(adj_augs)
                lidar2img_extras.extend(adj_extras)

        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=self._resolve_path(info['lidar_path']),
            sweeps=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
            info=info,
            img_filename=image_paths,
            lidar2img=lidar2img_rts,
            lidar2img_aug=lidar2img_augs,
            lidar2img_extra=lidar2img_extras,
            num_cams=len(cam_order),
        )

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        n_cameras = len(input_dict['img_filename'])
        data_info = dict(
            sample_idx=input_dict['sample_idx'],
            pts_filename=input_dict['pts_filename'],
            img_prefix=[None] * n_cameras,
            img_info=[dict(filename=x) for x in input_dict['img_filename']],
            lidar2img=dict(
                extrinsic=[tofloat(x) for x in input_dict['lidar2img']],
                intrinsic=np.eye(4, dtype=np.float32),
                lidar2img_aug=input_dict['lidar2img_aug'],
                lidar2img_extra=input_dict['lidar2img_extra'],
                origin=np.zeros(3, dtype=np.float32),
            ),
            num_cams=input_dict['num_cams'],
        )
        if 'ann_info' in input_dict:
            gt_bboxes_3d = input_dict['ann_info']['gt_bboxes_3d']
            gt_labels_3d = input_dict['ann_info']['gt_labels_3d'].copy()
            mask = gt_labels_3d >= 0
            gt_bboxes_3d = gt_bboxes_3d[mask]
            gt_names = input_dict['ann_info']['gt_names'][mask]
            gt_labels_3d = gt_labels_3d[mask]
            data_info['ann_info'] = dict(
                gt_bboxes_3d=gt_bboxes_3d,
                gt_names=gt_names,
                gt_labels_3d=gt_labels_3d,
            )
        return data_info

    def _format_bbox(self, results, jsonfile_prefix=None):
        nusc_annos = {}

        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            boxes = pccr_output_to_nusc_box(det)
            sample_info = self.data_infos[sample_id]
            sample_token = sample_info['token']
            boxes = lidar_nusc_box_to_global(
                sample_info, boxes, self.CLASSES, self.eval_detection_configs,
                self.eval_version)
            for box in boxes:
                name = self.CLASSES[box.label]
                annos.append(dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=[0.0, 0.0],
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name=self.DefaultAttribute[name]))
            nusc_annos[sample_token] = annos

        submission = {
            'meta': self.modality,
            'results': nusc_annos,
        }
        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmcv.dump(submission, res_path)
        return res_path

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        from nuscenes import NuScenes
        from nuscenes.eval.detection.evaluate import NuScenesEval

        output_dir = osp.join(*osp.split(result_path)[:-1])
        nusc = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=False)
        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
            'v1.0-test': 'test',
        }
        eval_set = eval_set_map[self.version]
        scene_names = sorted({
            nusc.get('scene', info['scene_token'])['name']
            for info in self.data_infos
        })
        attribute_names = sorted({attr['name'] for attr in nusc.attribute})
        with pccr_nuscenes_eval_scope(
                self.CLASSES,
                eval_set=eval_set,
                scene_names=scene_names,
                attribute_names=attribute_names):
            nusc_eval = NuScenesEval(
                nusc,
                config=self.eval_detection_configs,
                result_path=result_path,
                eval_set=eval_set,
                output_dir=output_dir,
                verbose=False)
            nusc_eval.main(render_curves=False)

        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = {}
        metric_prefix = f'{result_name}_NuScenes'
        skipped_tp_errors = {'vel_err', 'attr_err'}
        for name in self.CLASSES:
            for k, v in metrics['label_aps'][name].items():
                detail[f'{metric_prefix}/{name}_AP_dist_{k}'] = float(f'{v:.4f}')
            for k, v in metrics['label_tp_errors'][name].items():
                if k in skipped_tp_errors:
                    continue
                detail[f'{metric_prefix}/{name}_{k}'] = float(f'{v:.4f}')
        for k, v in metrics['tp_errors'].items():
            if k in skipped_tp_errors:
                continue
            detail[f'{metric_prefix}/{self.ErrNameMapping[k]}'] = float(f'{v:.4f}')

        detail[f'{metric_prefix}/NDS'] = metrics['nd_score']
        detail[f'{metric_prefix}/mAP'] = metrics['mean_ap']
        return detail

    def format_results(self, results, jsonfile_prefix=None):
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.format(
                len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            result_files = {}
            for name in results[0]:
                print(f'\nFormating bboxes of {name}')
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update({name: self._format_bbox(results_, tmp_file_)})
        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 vis_mode=False,
                 show=False,
                 out_dir=None,
                 pipeline=None,
                 **kwargs):
        return NuScenesDataset.evaluate(
            self,
            results,
            metric=metric,
            logger=logger,
            jsonfile_prefix=jsonfile_prefix,
            result_names=result_names,
            show=show,
            out_dir=out_dir,
            pipeline=pipeline)


def pccr_output_to_nusc_box(detection):
    box3d = detection['boxes_3d']
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = -box3d.yaw.numpy() - np.pi / 2

    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        box_list.append(NuScenesBox(
            box_gravity_center[i],
            box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=(0.0, 0.0, 0.0)))
    return box_list