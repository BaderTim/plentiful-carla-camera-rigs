# Copyright (c) OpenMMLab. All rights reserved.
import copy
from contextlib import contextmanager
from os import path as osp

import mmcv
import numpy as np
import pyquaternion
from nuscenes.utils.data_classes import Box as NuScenesBox

from .builder import DATASETS
from .nuscenes_dataset import NuScenesDataset


@DATASETS.register_module()
class PCCRDataset(NuScenesDataset):
    """PCCR dataset with nuScenes-style storage and custom classes.

    The PCCR infos keep image and lidar paths relative to the rig root and the
    rigs expose different camera layouts. This dataset normalizes relative
    paths, patches the nuScenes evaluation mapping for custom class names, and
    optionally drops velocity from both training boxes and reported metrics.
    """

    CLASSES = (
        'car',
        'truck',
        'bus',
        'bicycle',
        'motorcycle',
        'adult',
        'child',
        'traffic_sign',
        'traffic_light',
    )

    DefaultAttribute = {
        'car': 'vehicle.parked',
        'truck': 'vehicle.parked',
        'bus': 'vehicle.moving',
        'bicycle': 'cycle.without_rider',
        'motorcycle': 'cycle.without_rider',
        'adult': 'pedestrian.moving',
        'child': 'pedestrian.moving',
        'traffic_sign': '',
        'traffic_light': '',
    }

    ErrNameMapping = {
        'trans_err': 'mATE',
        'scale_err': 'mASE',
        'orient_err': 'mAOE',
        'vel_err': 'mAVE',
        'attr_err': 'mAAE',
    }

    DEFAULT_CLASS_RANGE = {
        'car': 50,
        'truck': 50,
        'bus': 50,
        'bicycle': 40,
        'motorcycle': 40,
        'adult': 40,
        'child': 40,
        'traffic_sign': 30,
        'traffic_light': 30,
    }

    ATTRIBUTE_NAME_MAPPING = {
        'adult_moving': 'pedestrian.moving',
        'adult_standing': 'pedestrian.standing',
        'moving': 'vehicle.moving',
        'parked': 'vehicle.parked',
        'stopped': 'vehicle.stopped',
        'with_rider': 'cycle.with_rider',
        'without_rider': 'cycle.without_rider',
    }

    EGO_CAM_CANDIDATES = (
        'CAM_FRONT',
        'CAM_BUMPER_FRONT',
        'CAM_FRONT_TOP',
        'CAM_BUMPER_FRONT_LEFT',
        'CAM_BUMPER_FRONT_RIGHT',
        'CAM_FENDER_FRONT_LEFT',
        'CAM_FENDER_FRONT_RIGHT',
    )

    def __init__(self,
                 *args,
                 class_range=None,
                 discard_velocity_eval=True,
                 ego_cam='auto',
                 with_velocity=False,
                 **kwargs):
        self.class_range = copy.deepcopy(self.DEFAULT_CLASS_RANGE)
        if class_range is not None:
            self.class_range.update(class_range)
        self.discard_velocity_eval = discard_velocity_eval
        super().__init__(
            *args, ego_cam=ego_cam, with_velocity=with_velocity, **kwargs)
        self.eval_detection_configs = self._build_eval_detection_configs()

    def load_annotations(self, ann_file):
        data = mmcv.load(ann_file, file_format='pkl')
        data_infos = list(sorted(data['infos'], key=lambda info: info['timestamp']))
        data_infos = data_infos[::self.load_interval]
        for info in data_infos:
            self._normalize_info_paths(info)
            self._normalize_bevdet_ann_infos(info)
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        return data_infos

    def _normalize_bevdet_ann_infos(self, info):
        if 'ann_infos' not in info:
            return

        ann_boxes, ann_labels = info['ann_infos']
        if len(ann_boxes) != len(ann_labels):
            return

        if self.use_valid_flag and 'valid_flag' in info:
            valid_names = info['gt_names'][info['valid_flag']]
        else:
            valid_names = info['gt_names']

        if len(valid_names) != len(ann_labels):
            return

        expected_labels = []
        for name in valid_names:
            if name in self.CLASSES:
                expected_labels.append(self.CLASSES.index(name))
            else:
                expected_labels.append(-1)

        if list(ann_labels) != expected_labels:
            info['ann_infos'] = [ann_boxes, expected_labels]

    def _build_eval_detection_configs(self):
        from nuscenes.eval.detection.data_classes import DetectionConfig

        eval_detection_configs = DetectionConfig.deserialize(
            self.eval_detection_configs.serialize())
        eval_detection_configs.class_names = list(self.CLASSES)
        eval_detection_configs.class_range = {
            class_name: self.class_range[class_name]
            for class_name in self.CLASSES
        }
        return eval_detection_configs

    def _normalize_path(self, path_name):
        if not path_name:
            return path_name
        if osp.isabs(path_name):
            return path_name
        return osp.join(self.data_root, path_name)

    def _normalize_info_paths(self, info):
        info['lidar_path'] = self._normalize_path(info['lidar_path'])

        for cam_info in info.get('cams', {}).values():
            cam_info['data_path'] = self._normalize_path(cam_info['data_path'])

        for sweep in info.get('sweeps', []) or []:
            if 'data_path' in sweep:
                sweep['data_path'] = self._normalize_path(sweep['data_path'])
            if 'lidar_path' in sweep:
                sweep['lidar_path'] = self._normalize_path(sweep['lidar_path'])

        for radar_info in info.get('radars', {}).values():
            if 'data_path' in radar_info:
                radar_info['data_path'] = self._normalize_path(
                    radar_info['data_path'])

    def _get_eval_splits(self, nusc):
        dataset_scene_names = []
        seen_scene_tokens = set()
        for info in self.data_infos:
            scene_token = info.get('scene_token')
            if scene_token in seen_scene_tokens:
                continue
            seen_scene_tokens.add(scene_token)
            dataset_scene_names.append(nusc.get('scene', scene_token)['name'])

        return dict(
            train=list(dataset_scene_names),
            val=list(dataset_scene_names),
            test=list(dataset_scene_names),
            mini_train=list(dataset_scene_names),
            mini_val=list(dataset_scene_names))

    def _map_category_to_detection_name(self, category_name):
        if category_name in self.CLASSES:
            return category_name
        return self.NameMapping.get(category_name)

    def _get_ego_cam_name(self, info):
        cam_names = list(info['cams'].keys())
        if self.ego_cam not in (None, 'auto'):
            if self.ego_cam not in info['cams']:
                raise KeyError(
                    f'ego_cam {self.ego_cam} is not available for sample '
                    f'{info.get("token", "<unknown>")}. Available cameras: '
                    f'{cam_names}')
            return self.ego_cam

        for candidate in self.EGO_CAM_CANDIDATES:
            if candidate in info['cams']:
                return candidate

        return cam_names[0]

    def _get_box_velocity_and_attr(self, name, box):
        if self.with_velocity and box.shape[0] >= 9:
            box_vel = box[7:9].tolist()
            box_vel.append(0.0)
            speed = np.sqrt(box_vel[0]**2 + box_vel[1]**2)
            if speed > 0.2:
                if name in ['car', 'truck', 'bus']:
                    attr = 'vehicle.moving'
                elif name in ['bicycle', 'motorcycle']:
                    attr = 'cycle.with_rider'
                else:
                    attr = self.DefaultAttribute[name]
            else:
                if name in ['adult', 'child']:
                    attr = 'pedestrian.standing'
                elif name == 'bus':
                    attr = 'vehicle.stopped'
                else:
                    attr = self.DefaultAttribute[name]
            return box_vel, attr

        return [0.0, 0.0, 0.0], self.DefaultAttribute[name]

    def _format_bbox(self, results, jsonfile_prefix=None):
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            boxes = det['boxes_3d'].tensor.numpy()
            scores = det['scores_3d'].numpy()
            labels = det['labels_3d'].numpy()
            info = self.data_infos[sample_id]
            sample_token = info['token']

            ego_cam_name = self._get_ego_cam_name(info)
            trans = info['cams'][ego_cam_name]['ego2global_translation']
            rot = pyquaternion.Quaternion(
                info['cams'][ego_cam_name]['ego2global_rotation'])

            annos = []
            for i, box in enumerate(boxes):
                name = mapped_class_names[labels[i]]
                center = box[:3]
                wlh = box[[4, 3, 5]]
                quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box[6])
                box_vel, attr = self._get_box_velocity_and_attr(name, box)
                nusc_box = NuScenesBox(center, wlh, quat, velocity=box_vel)
                nusc_box.rotate(rot)
                nusc_box.translate(trans)
                annos.append(
                    dict(
                        sample_token=sample_token,
                        translation=nusc_box.center.tolist(),
                        size=nusc_box.wlh.tolist(),
                        rotation=nusc_box.orientation.elements.tolist(),
                        velocity=nusc_box.velocity[:2],
                        detection_name=name,
                        detection_score=float(scores[i]),
                        attribute_name=attr,
                    ))

            if sample_token in nusc_annos:
                nusc_annos[sample_token].extend(annos)
            else:
                nusc_annos[sample_token] = annos

        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    @contextmanager
    def _patched_nuscenes_eval(self, nusc):
        from nuscenes.eval.common.data_classes import EvalBoxes
        from nuscenes.eval.common import loaders as eval_loaders
        from nuscenes.eval.detection import constants as detection_constants
        from nuscenes.eval.detection import data_classes as detection_data_classes
        from nuscenes.eval.detection import evaluate as detection_evaluate
        from nuscenes.eval.detection import utils as detection_utils

        original_split_factory = eval_loaders.create_splits_scenes
        original_load_prediction = eval_loaders.load_prediction
        original_load_gt = eval_loaders.load_gt
        original_category_mapper = eval_loaders.category_to_detection_name
        original_detection_category_mapper = \
            detection_utils.category_to_detection_name
        original_detection_names = detection_constants.DETECTION_NAMES
        original_data_class_detection_names = \
            detection_data_classes.DETECTION_NAMES
        original_attribute_names = detection_constants.ATTRIBUTE_NAMES
        original_data_class_attribute_names = \
            detection_data_classes.ATTRIBUTE_NAMES
        original_evaluate_load_prediction = detection_evaluate.load_prediction
        original_evaluate_load_gt = detection_evaluate.load_gt
        original_nusc_attributes = [attr['name'] for attr in nusc.attribute]

        dataset_sample_tokens = [info['token'] for info in self.data_infos]

        def subset_eval_boxes(eval_boxes, include_empty=False):
            filtered_eval_boxes = EvalBoxes()
            for sample_token in dataset_sample_tokens:
                if sample_token in eval_boxes.sample_tokens:
                    filtered_eval_boxes.add_boxes(sample_token, eval_boxes[sample_token])
                elif include_empty:
                    filtered_eval_boxes.add_boxes(sample_token, [])
            return filtered_eval_boxes

        def load_prediction_subset(*args, **kwargs):
            pred_boxes, meta = original_load_prediction(*args, **kwargs)
            return subset_eval_boxes(pred_boxes, include_empty=True), meta

        def load_gt_subset(*args, **kwargs):
            gt_boxes = original_load_gt(*args, **kwargs)
            return subset_eval_boxes(gt_boxes, include_empty=True)

        eval_loaders.create_splits_scenes = lambda: self._get_eval_splits(nusc)
        eval_loaders.load_prediction = load_prediction_subset
        eval_loaders.load_gt = load_gt_subset
        eval_loaders.category_to_detection_name = \
            self._map_category_to_detection_name
        detection_evaluate.load_prediction = load_prediction_subset
        detection_evaluate.load_gt = load_gt_subset
        detection_utils.category_to_detection_name = \
            self._map_category_to_detection_name
        detection_constants.DETECTION_NAMES = list(self.CLASSES)
        detection_data_classes.DETECTION_NAMES = detection_constants.DETECTION_NAMES
        detection_constants.ATTRIBUTE_NAMES = sorted(
            set(original_attribute_names) |
            set(self.ATTRIBUTE_NAME_MAPPING.keys()) |
            set(self.ATTRIBUTE_NAME_MAPPING.values()))
        detection_data_classes.ATTRIBUTE_NAMES = \
            detection_constants.ATTRIBUTE_NAMES
        for attr in nusc.attribute:
            attr['name'] = self.ATTRIBUTE_NAME_MAPPING.get(attr['name'], attr['name'])
        try:
            yield
        finally:
            eval_loaders.create_splits_scenes = original_split_factory
            eval_loaders.load_prediction = original_load_prediction
            eval_loaders.load_gt = original_load_gt
            eval_loaders.category_to_detection_name = original_category_mapper
            detection_evaluate.load_prediction = original_evaluate_load_prediction
            detection_evaluate.load_gt = original_evaluate_load_gt
            detection_utils.category_to_detection_name = \
                original_detection_category_mapper
            detection_constants.DETECTION_NAMES = original_detection_names
            detection_data_classes.DETECTION_NAMES = \
                original_data_class_detection_names
            detection_constants.ATTRIBUTE_NAMES = original_attribute_names
            detection_data_classes.ATTRIBUTE_NAMES = \
                original_data_class_attribute_names
            for attr, original_name in zip(nusc.attribute, original_nusc_attributes):
                attr['name'] = original_name

    def _compute_nds(self, metrics):
        if not self.discard_velocity_eval:
            return metrics['nd_score']

        tp_scores = []
        for metric_name, metric_value in metrics['tp_errors'].items():
            if metric_name == 'vel_err':
                continue
            if metric_value is None or not np.isfinite(metric_value):
                continue
            tp_scores.append(max(0.0, 1.0 - metric_value))

        numerator = (
            self.eval_detection_configs.mean_ap_weight * metrics['mean_ap'] +
            sum(tp_scores))
        denominator = self.eval_detection_configs.mean_ap_weight + len(tp_scores)
        if denominator == 0:
            return 0.0
        return numerator / denominator

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
        with self._patched_nuscenes_eval(nusc):
            nusc_eval = NuScenesEval(
                nusc,
                config=self.eval_detection_configs,
                result_path=result_path,
                eval_set=eval_set_map[self.version],
                output_dir=output_dir,
                verbose=False)
            nusc_eval.main(render_curves=False)

        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = {}
        metric_prefix = f'{result_name}_NuScenes'
        for name in self.CLASSES:
            for key, value in metrics['label_aps'].get(name, {}).items():
                detail[f'{metric_prefix}/{name}_AP_dist_{key}'] = \
                    float('{:.4f}'.format(value))
            for key, value in metrics['label_tp_errors'].get(name, {}).items():
                if self.discard_velocity_eval and key == 'vel_err':
                    continue
                detail[f'{metric_prefix}/{name}_{key}'] = \
                    float('{:.4f}'.format(value))

        for key, value in metrics['tp_errors'].items():
            if self.discard_velocity_eval and key == 'vel_err':
                continue
            detail[f'{metric_prefix}/{self.ErrNameMapping[key]}'] = \
                float('{:.4f}'.format(value))

        detail[f'{metric_prefix}/NDS'] = self._compute_nds(metrics)
        detail[f'{metric_prefix}/mAP'] = metrics['mean_ap']
        return detail