import tempfile
import json
from collections import defaultdict
from dataclasses import dataclass
from os import path as osp
from typing import Any, Dict, List, Sequence

import mmcv
import numpy as np
import pyquaternion
import torch
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets.custom_3d import Custom3DDataset
from pyquaternion import Quaternion

from mmdet.datasets import DATASETS


@dataclass
class _EvalBox:
    sample_token: str
    translation: np.ndarray
    size: np.ndarray
    rotation: np.ndarray
    detection_name: str
    detection_score: float
    attribute_name: str = ""
    num_pts: int = -1


def _to_numpy(data: Any) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return data
    if hasattr(data, "detach"):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def _angle_diff(x: float, y: float, period: float) -> float:
    diff = (x - y + period / 2) % period - period / 2
    if diff > np.pi:
        diff -= 2 * np.pi
    return diff


def _quaternion_yaw(quaternion: Sequence[float]) -> float:
    rotated_axis = np.dot(Quaternion(quaternion).rotation_matrix, np.array([1, 0, 0]))
    return float(np.arctan2(rotated_axis[1], rotated_axis[0]))


def _center_distance(gt_box: _EvalBox, pred_box: _EvalBox) -> float:
    return float(np.linalg.norm(pred_box.translation[:2] - gt_box.translation[:2]))


def _scale_iou(gt_box: _EvalBox, pred_box: _EvalBox) -> float:
    gt_size = np.asarray(gt_box.size)
    pred_size = np.asarray(pred_box.size)
    min_wlh = np.minimum(gt_size, pred_size)
    intersection = float(np.prod(min_wlh))
    union = float(np.prod(gt_size) + np.prod(pred_size) - intersection)
    return intersection / union


def _yaw_diff(gt_box: _EvalBox, pred_box: _EvalBox, period: float = 2 * np.pi) -> float:
    yaw_gt = _quaternion_yaw(gt_box.rotation)
    yaw_pred = _quaternion_yaw(pred_box.rotation)
    return abs(_angle_diff(yaw_gt, yaw_pred, period))


def _attr_acc(gt_box: _EvalBox, pred_box: _EvalBox) -> float:
    if gt_box.attribute_name == "":
        return np.nan
    return float(gt_box.attribute_name == pred_box.attribute_name)


def _cummean(values: np.ndarray) -> np.ndarray:
    if np.isnan(values).all():
        return np.ones(len(values), dtype=np.float64)

    sums = np.nancumsum(values.astype(np.float64))
    counts = np.cumsum(~np.isnan(values))
    return np.divide(sums, counts, out=np.zeros_like(sums), where=counts != 0)


def _no_predictions_metric_data() -> Dict[str, np.ndarray]:
    recall = np.linspace(0, 1, 101)
    ones = np.ones(101, dtype=np.float64)
    zeros = np.zeros(101, dtype=np.float64)
    return {
        "recall": recall,
        "precision": zeros,
        "confidence": zeros,
        "trans_err": ones.copy(),
        "scale_err": ones.copy(),
        "orient_err": ones.copy(),
        "attr_err": ones.copy(),
    }


def _calc_ap(metric_data: Dict[str, np.ndarray], min_recall: float, min_precision: float) -> float:
    precision = np.copy(metric_data["precision"])
    precision = precision[round(100 * min_recall) + 1 :]
    precision -= min_precision
    precision[precision < 0] = 0
    return float(np.mean(precision)) / (1.0 - min_precision)


def _calc_tp(metric_data: Dict[str, np.ndarray], min_recall: float, metric_name: str) -> float:
    first_index = round(100 * min_recall) + 1
    non_zero = np.nonzero(metric_data["confidence"])[0]
    last_index = int(non_zero[-1]) if len(non_zero) else 0
    if last_index < first_index:
        return 1.0
    return float(np.mean(metric_data[metric_name][first_index : last_index + 1]))


@DATASETS.register_module()
class PCCRDataset(Custom3DDataset):
    CLASSES = (
        "car",
        "truck",
        "bus",
        "motorcycle",
        "bicycle",
        "adult",
        "child",
        "traffic_light",
        "traffic_sign",
    )

    CLASS_RANGE = {
        "car": 50,
        "truck": 50,
        "bus": 50,
        "motorcycle": 40,
        "bicycle": 40,
        "adult": 40,
        "child": 40,
        "traffic_light": 30,
        "traffic_sign": 30,
    }

    DISTANCE_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
    TP_DISTANCE_THRESHOLD = 2.0
    MIN_RECALL = 0.1
    MIN_PRECISION = 0.1
    MAX_BOXES_PER_SAMPLE = 500
    TP_METRICS = ("trans_err", "scale_err", "orient_err")
    ErrNameMapping = {
        "trans_err": "mATE",
        "scale_err": "mASE",
        "orient_err": "mAOE",
        "attr_err": "mAAE",
    }

    def __init__(
        self,
        ann_file,
        pipeline=None,
        data_root=None,
        classes=None,
        map_classes=None,
        load_interval=1,
        with_velocity=True,
        modality=None,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        test_mode=False,
        eval_version="detection_cvpr_2019",
        use_valid_flag=False,
        class_range=None,
        dataset_root=None,
        object_classes=None,
    ) -> None:
        if data_root is None:
            data_root = dataset_root
        if classes is None:
            classes = object_classes

        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        self.data_root = data_root
        self.dataset_root = data_root
        self.map_classes = map_classes
        self.with_velocity = with_velocity
        self.eval_version = eval_version
        self.class_range = dict(class_range or self.CLASS_RANGE)
        self._cached_gt_by_token = None
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
        )

        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )

    def _use_point_count_filter(self):
        if self.modality is None:
            return True

        return self.modality.get("use_lidar", False) or self.modality.get(
            "use_radar", False
        )

    def _get_ann_mask(self, info):
        if not self._use_point_count_filter():
            return np.ones(len(info["gt_names"]), dtype=bool)

        if self.use_valid_flag:
            return info["valid_flag"]

        return info["num_lidar_pts"] > 0

    def get_cat_ids(self, idx):
        info = self.data_infos[idx]
        mask = self._get_ann_mask(info)
        gt_names = set(info["gt_names"][mask])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def load_annotations(self, ann_file):
        data = mmcv.load(ann_file)
        data_infos = list(sorted(data["infos"], key=lambda entry: entry["timestamp"]))
        data_infos = data_infos[:: self.load_interval]
        self.metadata = data["metadata"]
        self.version = self.metadata["version"]
        return data_infos

    def _resolve_data_path(self, file_path):
        if not file_path or osp.exists(file_path):
            return file_path

        data_root = getattr(self, "data_root", None)
        if not data_root:
            return file_path

        normalized_path = file_path.replace("\\", "/")
        markers = (
            "/samples/",
            "/sweeps/",
            "/maps/",
            "/can_bus/",
            "/v1.0-",
            "/.voxel51/",
        )

        for marker in markers:
            marker_index = normalized_path.find(marker)
            if marker_index != -1:
                candidate = osp.join(data_root, normalized_path[marker_index + 1 :])
                if osp.exists(candidate):
                    return candidate

        candidate = osp.join(data_root, normalized_path.lstrip("/"))
        if osp.exists(candidate):
            return candidate

        return candidate

    def _resolve_sweeps(self, sweeps):
        resolved_sweeps = []
        for sweep in sweeps:
            resolved_sweep = sweep.copy()
            if "data_path" in resolved_sweep:
                resolved_sweep["data_path"] = self._resolve_data_path(
                    resolved_sweep["data_path"]
                )
            resolved_sweeps.append(resolved_sweep)
        return resolved_sweeps

    def _resolve_radars(self, radars):
        if not isinstance(radars, dict):
            return radars

        resolved_radars = {}
        for key, sweeps in radars.items():
            resolved_radars[key] = self._resolve_sweeps(sweeps)
        return resolved_radars

    def get_data_info(self, index: int) -> Dict[str, Any]:
        info = self.data_infos[index]
        pts_filename = self._resolve_data_path(info["lidar_path"])

        data = dict(
            token=info["token"],
            sample_idx=info["token"],
            pts_filename=pts_filename,
            lidar_path=pts_filename,
            sweeps=self._resolve_sweeps(info["sweeps"]),
            timestamp=info["timestamp"] / 1e6,
            location=info.get("location", None),
            radar=self._resolve_radars(info.get("radars", None)),
        )

        if data["location"] is None:
            data.pop("location")
        if data["radar"] is None:
            data.pop("radar")

        ego2global = np.eye(4, dtype=np.float32)
        ego2global[:3, :3] = Quaternion(info["ego2global_rotation"]).rotation_matrix
        ego2global[:3, 3] = info["ego2global_translation"]
        data["ego2global"] = ego2global

        lidar2ego = np.eye(4, dtype=np.float32)
        lidar2ego[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
        lidar2ego[:3, 3] = info["lidar2ego_translation"]
        data["lidar2ego"] = lidar2ego

        if self.modality["use_camera"]:
            data["img_filename"] = []
            data["filename"] = data["img_filename"]
            data["img_timestamp"] = []
            data["intrinsics"] = []
            data["extrinsics"] = []
            data["lidar2img"] = []

            data["image_paths"] = data["img_filename"]
            data["lidar2camera"] = []
            data["lidar2image"] = data["lidar2img"]
            data["camera2ego"] = []
            data["camera_intrinsics"] = data["intrinsics"]
            data["camera2lidar"] = []

            for _, camera_info in info["cams"].items():
                data["img_filename"].append(
                    self._resolve_data_path(camera_info["data_path"])
                )
                data["img_timestamp"].append(
                    camera_info.get("timestamp", info["timestamp"]) / 1e6
                )

                lidar2camera_r = np.linalg.inv(camera_info["sensor2lidar_rotation"])
                lidar2camera_t = (
                    camera_info["sensor2lidar_translation"] @ lidar2camera_r.T
                )
                lidar2camera_rt = np.eye(4, dtype=np.float32)
                lidar2camera_rt[:3, :3] = lidar2camera_r.T
                lidar2camera_rt[3, :3] = -lidar2camera_t
                data["extrinsics"].append(lidar2camera_rt)
                data["lidar2camera"].append(lidar2camera_rt.T)

                camera_intrinsics = np.eye(4, dtype=np.float32)
                camera_intrinsics[:3, :3] = camera_info["cam_intrinsic"]
                data["intrinsics"].append(camera_intrinsics)

                data["lidar2img"].append(camera_intrinsics @ lidar2camera_rt.T)

                camera2ego = np.eye(4, dtype=np.float32)
                camera2ego[:3, :3] = Quaternion(
                    camera_info["sensor2ego_rotation"]
                ).rotation_matrix
                camera2ego[:3, 3] = camera_info["sensor2ego_translation"]
                data["camera2ego"].append(camera2ego)

                camera2lidar = np.eye(4, dtype=np.float32)
                camera2lidar[:3, :3] = camera_info["sensor2lidar_rotation"]
                camera2lidar[:3, 3] = camera_info["sensor2lidar_translation"]
                data["camera2lidar"].append(camera2lidar)

        if not self.test_mode:
            data["ann_info"] = self.get_ann_info(index)
        return data

    def _set_group_flag(self) -> None:
        self.flag = np.array([len(info["cams"]) for info in self.data_infos], dtype=np.uint8)

    def get_ann_info(self, index):
        info = self.data_infos[index]
        mask = self._get_ann_mask(info)
        gt_bboxes_3d = info["gt_boxes"][mask]
        gt_names_3d = info["gt_names"][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info["gt_velocity"][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0)
        ).convert_to(self.box_mode_3d)

        return dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
        )

    def _normalize_eval_results(self, results, result_names=None):
        if not results:
            return results

        normalized = []
        result_names = result_names or ["pts_bbox"]
        primary_name = result_names[0] if result_names else None

        for det in results:
            if primary_name and primary_name in det:
                normalized.append(det[primary_name])
            else:
                normalized.append(det)
        return normalized

    def _lidar_boxes_to_eval_boxes(
        self,
        box3d: LiDARInstance3DBoxes,
        names: Sequence[str],
        scores: Sequence[float],
        sample_token: str,
    ) -> List[_EvalBox]:
        centers = _to_numpy(box3d.gravity_center)
        dims = _to_numpy(box3d.dims)
        yaw = _to_numpy(box3d.yaw)
        yaw = -yaw - np.pi / 2

        eval_boxes = []
        for index, name in enumerate(names):
            if name not in self.class_range:
                continue

            rotation = pyquaternion.Quaternion(axis=[0, 0, 1], radians=float(yaw[index]))
            eval_boxes.append(
                _EvalBox(
                    sample_token=sample_token,
                    translation=np.asarray(centers[index], dtype=np.float32),
                    size=np.asarray(dims[index], dtype=np.float32),
                    rotation=np.asarray(rotation.elements, dtype=np.float32),
                    detection_name=name,
                    detection_score=float(scores[index]),
                )
            )
        return eval_boxes

    def _output_to_eval_boxes(self, detection, sample_token: str) -> List[_EvalBox]:
        box3d = detection["boxes_3d"]
        scores = _to_numpy(detection["scores_3d"])
        labels = _to_numpy(detection["labels_3d"])
        class_names = [self.CLASSES[int(label)] for label in labels]
        eval_boxes = self._lidar_boxes_to_eval_boxes(box3d, class_names, scores, sample_token)
        eval_boxes.sort(key=lambda box: box.detection_score, reverse=True)
        return eval_boxes[: self.MAX_BOXES_PER_SAMPLE]

    def _load_gt_from_tables(self):
        if self._cached_gt_by_token is not None:
            return self._cached_gt_by_token

        table_root = osp.join(self.data_root, self.version)
        with open(osp.join(table_root, "sample_annotation.json"), "r") as handle:
            sample_annotations = json.load(handle)
        with open(osp.join(table_root, "instance.json"), "r") as handle:
            instances = json.load(handle)
        with open(osp.join(table_root, "category.json"), "r") as handle:
            categories = json.load(handle)

        category_by_token = {category["token"]: category["name"] for category in categories}
        category_by_instance = {
            instance["token"]: category_by_token[instance["category_token"]]
            for instance in instances
        }

        gt_by_token = defaultdict(list)
        for annotation in sample_annotations:
            detection_name = category_by_instance.get(annotation["instance_token"])
            if detection_name not in self.class_range:
                continue

            gt_by_token[annotation["sample_token"]].append(
                _EvalBox(
                    sample_token=annotation["sample_token"],
                    translation=np.asarray(annotation["translation"], dtype=np.float32),
                    size=np.asarray(annotation["size"], dtype=np.float32),
                    rotation=np.asarray(annotation["rotation"], dtype=np.float32),
                    detection_name=detection_name,
                    detection_score=-1.0,
                    num_pts=int(annotation["num_lidar_pts"] + annotation["num_radar_pts"]),
                )
            )

        self._cached_gt_by_token = gt_by_token
        return gt_by_token

    def _to_global(self, info, box: _EvalBox) -> _EvalBox:
        lidar_to_ego = Quaternion(info["lidar2ego_rotation"])
        ego_to_global = Quaternion(info["ego2global_rotation"])

        center = np.asarray(lidar_to_ego.rotate(box.translation), dtype=np.float32)
        center += np.asarray(info["lidar2ego_translation"], dtype=np.float32)

        radius = float(np.linalg.norm(center[:2], ord=2))
        if radius > self.class_range[box.detection_name]:
            return None

        orientation = lidar_to_ego * Quaternion(box.rotation)

        center = np.asarray(ego_to_global.rotate(center), dtype=np.float32)
        center += np.asarray(info["ego2global_translation"], dtype=np.float32)
        orientation = ego_to_global * orientation

        return _EvalBox(
            sample_token=box.sample_token,
            translation=center,
            size=box.size,
            rotation=np.asarray(orientation.elements, dtype=np.float32),
            detection_name=box.detection_name,
            detection_score=box.detection_score,
            attribute_name=box.attribute_name,
        )

    def _accumulate(self, gt_by_token, pred_by_token, class_name: str, dist_th: float):
        npos = sum(
            1
            for token in gt_by_token
            for gt_box in gt_by_token[token]
            if gt_box.detection_name == class_name
        )
        if npos == 0:
            return _no_predictions_metric_data()

        pred_boxes = [
            box
            for token in pred_by_token
            for box in pred_by_token[token]
            if box.detection_name == class_name
        ]
        pred_boxes.sort(key=lambda box: box.detection_score, reverse=True)

        true_positives = []
        false_positives = []
        confidences = []
        match_data = {
            "trans_err": [],
            "scale_err": [],
            "orient_err": [],
            "attr_err": [],
            "conf": [],
        }

        taken = set()
        for pred_box in pred_boxes:
            min_dist = np.inf
            match_gt_idx = None

            for gt_idx, gt_box in enumerate(gt_by_token.get(pred_box.sample_token, [])):
                if gt_box.detection_name != class_name:
                    continue
                if (pred_box.sample_token, gt_idx) in taken:
                    continue

                distance = _center_distance(gt_box, pred_box)
                if distance < min_dist:
                    min_dist = distance
                    match_gt_idx = gt_idx

            is_match = min_dist < dist_th
            if is_match:
                taken.add((pred_box.sample_token, match_gt_idx))
                gt_match = gt_by_token[pred_box.sample_token][match_gt_idx]

                true_positives.append(1)
                false_positives.append(0)
                confidences.append(pred_box.detection_score)

                match_data["trans_err"].append(_center_distance(gt_match, pred_box))
                match_data["scale_err"].append(1 - _scale_iou(gt_match, pred_box))
                match_data["orient_err"].append(_yaw_diff(gt_match, pred_box))
                match_data["attr_err"].append(1 - _attr_acc(gt_match, pred_box))
                match_data["conf"].append(pred_box.detection_score)
            else:
                true_positives.append(0)
                false_positives.append(1)
                confidences.append(pred_box.detection_score)

        if not match_data["trans_err"]:
            return _no_predictions_metric_data()

        true_positives = np.cumsum(true_positives).astype(np.float64)
        false_positives = np.cumsum(false_positives).astype(np.float64)
        confidences = np.asarray(confidences, dtype=np.float64)

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / float(npos)

        recall_interp = np.linspace(0, 1, 101)
        precision = np.interp(recall_interp, recall, precision, right=0)
        confidences = np.interp(recall_interp, recall, confidences, right=0)

        metric_data = {
            "recall": recall_interp,
            "precision": precision,
            "confidence": confidences,
        }

        conf_reference = np.asarray(match_data["conf"], dtype=np.float64)
        for metric_name in self.TP_METRICS:
            cumulative = _cummean(np.asarray(match_data[metric_name], dtype=np.float64))
            metric_data[metric_name] = np.interp(
                confidences[::-1], conf_reference[::-1], cumulative[::-1]
            )[::-1]

        return metric_data

    def _build_gt_boxes(self):
        gt_lookup = self._load_gt_from_tables()
        gt_by_token = {}
        for info in self.data_infos:
            ego_to_global = Quaternion(info["ego2global_rotation"])
            ego_to_global_inv = ego_to_global.inverse
            ego_translation = np.asarray(info["ego2global_translation"], dtype=np.float32)
            filtered_boxes = []

            for box in gt_lookup.get(info["token"], []):
                if box.num_pts == 0:
                    continue

                box_in_ego = np.asarray(
                    ego_to_global_inv.rotate(box.translation - ego_translation),
                    dtype=np.float32,
                )
                if np.linalg.norm(box_in_ego[:2], ord=2) >= self.class_range[box.detection_name]:
                    continue

                filtered_boxes.append(box)

            gt_by_token[info["token"]] = filtered_boxes

        return gt_by_token

    def _build_pred_boxes(self, results):
        pred_by_token = {}
        for sample_id, det in enumerate(results):
            info = self.data_infos[sample_id]
            pred_boxes = []
            for box in self._output_to_eval_boxes(det, info["token"]):
                global_box = self._to_global(info, box)
                if global_box is not None:
                    pred_boxes.append(global_box)
            pred_by_token[info["token"]] = pred_boxes
        return pred_by_token

    def _summarize_metrics(self, gt_by_token, pred_by_token):
        label_aps = {}
        label_tp_errors = {}

        for class_name in self.CLASSES:
            label_aps[class_name] = {}
            for dist_th in self.DISTANCE_THRESHOLDS:
                metric_data = self._accumulate(gt_by_token, pred_by_token, class_name, dist_th)
                label_aps[class_name][dist_th] = _calc_ap(
                    metric_data, self.MIN_RECALL, self.MIN_PRECISION
                )

            metric_data = self._accumulate(
                gt_by_token, pred_by_token, class_name, self.TP_DISTANCE_THRESHOLD
            )
            label_tp_errors[class_name] = {
                metric_name: _calc_tp(metric_data, self.MIN_RECALL, metric_name)
                for metric_name in self.TP_METRICS
            }

        mean_dist_aps = {
            class_name: float(np.mean(list(class_aps.values())))
            for class_name, class_aps in label_aps.items()
        }
        mean_ap = float(np.mean(list(mean_dist_aps.values()))) if mean_dist_aps else 0.0
        tp_errors = {}
        for metric_name in self.TP_METRICS:
            values = [label_tp_errors[class_name][metric_name] for class_name in self.CLASSES]
            tp_errors[metric_name] = float(np.nanmean(values))

        return {
            "label_aps": label_aps,
            "label_tp_errors": label_tp_errors,
            "tp_errors": tp_errors,
            "mean_ap": mean_ap,
        }

    def _metrics_to_detail(self, metrics_summary):
        detail = {}
        for name in self.CLASSES:
            for dist_th, value in metrics_summary["label_aps"][name].items():
                detail[f"object/{name}_ap_dist_{dist_th:.1f}"] = float(f"{value:.4f}")

            for metric_name, value in metrics_summary["label_tp_errors"][name].items():
                detail[f"object/{name}_{metric_name}"] = float(f"{value:.4f}")

        for metric_name, value in metrics_summary["tp_errors"].items():
            detail[f"object/{self.ErrNameMapping[metric_name]}"] = float(f"{value:.4f}")

        detail["object/map"] = metrics_summary["mean_ap"]
        return detail

    def _format_bbox(self, results, jsonfile_prefix=None):
        nusc_annos = {}
        normalized_results = self._normalize_eval_results(results)

        print("Start to convert detection format...")
        for sample_id, det in enumerate(mmcv.track_iter_progress(normalized_results)):
            info = self.data_infos[sample_id]
            annos = []
            for box in self._output_to_eval_boxes(det, info["token"]):
                global_box = self._to_global(info, box)
                if global_box is None:
                    continue

                annos.append(
                    dict(
                        sample_token=global_box.sample_token,
                        translation=global_box.translation.tolist(),
                        size=global_box.size.tolist(),
                        rotation=global_box.rotation.tolist(),
                        detection_name=global_box.detection_name,
                        detection_score=global_box.detection_score,
                        attribute_name=global_box.attribute_name,
                    )
                )
            nusc_annos[info["token"]] = annos

        submission = {"meta": self.modality, "results": nusc_annos}
        mmcv.mkdir_or_exist(jsonfile_prefix)
        result_path = osp.join(jsonfile_prefix, "results_pccr.json")
        print("Results writes to", result_path)
        mmcv.dump(submission, result_path)
        return result_path

    def _evaluate_single(
        self,
        result_path,
        logger=None,
        metric="bbox",
        result_name="pts_bbox",
    ):
        result_payload = mmcv.load(result_path)
        pred_by_token = {}
        for sample_token, boxes in result_payload["results"].items():
            pred_by_token[sample_token] = [
                _EvalBox(
                    sample_token=entry["sample_token"],
                    translation=np.asarray(entry["translation"], dtype=np.float32),
                    size=np.asarray(entry["size"], dtype=np.float32),
                    rotation=np.asarray(entry["rotation"], dtype=np.float32),
                    detection_name=entry["detection_name"],
                    detection_score=float(entry["detection_score"]),
                    attribute_name=entry.get("attribute_name", ""),
                )
                for entry in boxes
            ]

        gt_by_token = self._build_gt_boxes()
        metrics_summary = self._summarize_metrics(gt_by_token, pred_by_token)

        output_dir = osp.dirname(result_path)
        mmcv.dump(metrics_summary, osp.join(output_dir, "metrics_summary.json"))
        return self._metrics_to_detail(metrics_summary)

    def format_results(self, results, jsonfile_prefix=None):
        assert isinstance(results, list), "results must be a list"
        assert len(results) == len(self), (
            "The length of results is not equal to the dataset len: {} != {}".format(
                len(results), len(self)
            )
        )

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, "results")
        else:
            tmp_dir = None

        result_files = self._format_bbox(results, jsonfile_prefix)
        return result_files, tmp_dir

    def evaluate_map(self, results):
        thresholds = torch.tensor([0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65])

        num_classes = len(self.map_classes)
        num_thresholds = len(thresholds)

        true_positives = torch.zeros(num_classes, num_thresholds)
        false_positives = torch.zeros(num_classes, num_thresholds)
        false_negatives = torch.zeros(num_classes, num_thresholds)

        for result in results:
            prediction = result["masks_bev"]
            label = result["gt_masks_bev"]

            prediction = prediction.detach().reshape(num_classes, -1)
            label = label.detach().bool().reshape(num_classes, -1)

            prediction = prediction[:, :, None] >= thresholds
            label = label[:, :, None]

            true_positives += (prediction & label).sum(dim=1)
            false_positives += (prediction & ~label).sum(dim=1)
            false_negatives += (~prediction & label).sum(dim=1)

        ious = true_positives / (true_positives + false_positives + false_negatives + 1e-7)

        metrics = {}
        for index, name in enumerate(self.map_classes):
            metrics[f"map/{name}/iou@max"] = ious[index].max().item()
            for threshold, iou in zip(thresholds, ious[index]):
                metrics[f"map/{name}/iou@{threshold.item():.2f}"] = iou.item()
        metrics["map/mean/iou@max"] = ious.max(dim=1).values.mean().item()
        return metrics

    def evaluate(
        self,
        results,
        metric="bbox",
        jsonfile_prefix=None,
        result_names=["pts_bbox"],
        **kwargs,
    ):
        metrics = {}
        normalized_results = self._normalize_eval_results(results, result_names)

        if results and "masks_bev" in results[0]:
            metrics.update(self.evaluate_map(results))

        if normalized_results and "boxes_3d" in normalized_results[0]:
            if jsonfile_prefix is not None:
                result_files, tmp_dir = self.format_results(normalized_results, jsonfile_prefix)
                metrics.update(self._evaluate_single(result_files))
                if tmp_dir is not None:
                    tmp_dir.cleanup()
            else:
                gt_by_token = self._build_gt_boxes()
                pred_by_token = self._build_pred_boxes(normalized_results)
                metrics_summary = self._summarize_metrics(gt_by_token, pred_by_token)
                metrics.update(self._metrics_to_detail(metrics_summary))

        return metrics