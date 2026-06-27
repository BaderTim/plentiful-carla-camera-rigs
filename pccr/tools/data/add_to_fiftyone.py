#!/usr/bin/env python3
"""Load a single rig dataset into a grouped FiftyOne dataset.

This loader expects a rig directory that contains one or more nuScenes-style
split folders such as ``v1.0-mini``, ``v1.0-test``, and ``v1.0-trainval``.

Each keyframe sample is imported as a grouped sample with one slice per sensor:

* camera slices store the original image and projected 2D detections
* the lidar slice stores a generated ``.fo3d`` scene that references a cached
  ``.pcd`` file plus 3D cuboids for the sample annotations

The script also creates indices for common filter fields such as split, scene,
timestamp, sensor channel, category, visibility, instance token, and
attributes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

import fiftyone as fo

# Ensure project root is importable when the script is run directly.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib.annotation_projection import AnnotationProjector
from lib.quaternion import Quaternion


SPLIT_DIR_ORDER = ("v1.0-mini", "v1.0-test", "v1.0-trainval")

CATEGORY_COLORS = {
    "car": "#3BA55D",
    "truck": "#D98E04",
    "bus": "#C65D3A",
    "motorcycle": "#F0C419",
    "bicycle": "#2A9D8F",
    "adult": "#1D4ED8",
    "child": "#60A5FA",
    "traffic_sign": "#C026D3",
    "traffic_light": "#8B5CF6",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a single rig into a grouped FiftyOne dataset",
    )
    parser.add_argument(
        "--input-path",
        required=True,
        help="Path to the single-rig dataset root (for example output/data/R1)",
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Name of the FiftyOne dataset to create or replace",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("mini", "test", "trainval"),
        default=None,
        help="Optional subset of splits to import. Defaults to all discovered splits.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Grouped slice batch size used when inserting samples",
    )
    return parser.parse_args()


def load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def discover_split_dirs(dataset_root: Path, requested: Optional[Sequence[str]]) -> List[Tuple[str, Path]]:
    discovered: List[Tuple[str, Path]] = []

    if (dataset_root / "sample.json").exists():
        split_name = split_name_from_dir(dataset_root.name)
        if split_name is None:
            raise ValueError(
                f"Could not infer split name from direct split folder: {dataset_root}"
            )
        if requested is None or split_name in requested:
            discovered.append((split_name, dataset_root))
        return discovered

    requested_set = set(requested) if requested else None
    for dirname in SPLIT_DIR_ORDER:
        split_dir = dataset_root / dirname
        if not split_dir.exists():
            continue

        split_name = split_name_from_dir(dirname)
        if split_name is None:
            continue
        if requested_set is not None and split_name not in requested_set:
            continue
        discovered.append((split_name, split_dir))

    return discovered


def split_name_from_dir(dirname: str) -> Optional[str]:
    if dirname == "v1.0-mini":
        return "mini"
    if dirname == "v1.0-test":
        return "test"
    if dirname == "v1.0-trainval":
        return "trainval"
    return None


def sensor_slice_base_name(sensor: Dict[str, Any]) -> str:
    channel = str(sensor.get("channel") or "").strip()
    if channel:
        return channel

    modality = str(sensor.get("modality") or "sensor").upper()
    return f"{modality}_{sensor['token'][:8]}"


def camera_forward_score(calibrated_sensor: Optional[Dict[str, Any]]) -> float:
    if not calibrated_sensor:
        return 0.0

    rotation = Quaternion(array=calibrated_sensor["rotation"]).rotation_matrix
    optical_axis = rotation[:, 2]
    return float(optical_axis[0])


def choose_default_slice(
    slice_entries: Sequence[Tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]],
) -> str:
    if not slice_entries:
        raise ValueError("Cannot choose a default slice from an empty sensor list")

    def score(entry: Tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]) -> Tuple[int, float, float, str]:
        slice_name, sensor, calibrated_sensor = entry
        modality = sensor.get("modality")

        if modality == "camera":
            lateral_offset = abs(float((calibrated_sensor or {}).get("translation", [0.0, 0.0, 0.0])[1]))
            return (2, camera_forward_score(calibrated_sensor), -lateral_offset, slice_name)

        if modality == "lidar":
            return (1, 0.0, 0.0, slice_name)

        return (0, 0.0, 0.0, slice_name)

    return max(slice_entries, key=score)[0]


def ensure_binary_pcd(source_path: Path, target_path: Path) -> Path:
    if target_path.exists() and target_path.stat().st_mtime >= source_path.stat().st_mtime:
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)

    raw = np.fromfile(source_path, dtype=np.float32)
    if raw.size % 5 == 0:
        points = raw.reshape(-1, 5)[:, :4]
    elif raw.size % 4 == 0:
        points = raw.reshape(-1, 4)
    else:
        raise ValueError(
            f"Unexpected point cloud element count in {source_path}: {raw.size}"
        )

    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z intensity\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {points.shape[0]}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {points.shape[0]}\n"
        "DATA binary\n"
    ).encode("ascii")

    with target_path.open("wb") as stream:
        stream.write(header)
        stream.write(points.astype(np.float32, copy=False).tobytes())

    return target_path


def ensure_lidar_scene(
    source_lidar_path: Path,
    scene_path: Path,
    pcd_path: Path,
    annotations: Sequence[Dict[str, Any]],
    category_by_instance: Dict[str, str],
    ego_pose: Dict[str, Any],
    calibrated_sensor: Dict[str, Any],
) -> Path:
    ensure_binary_pcd(source_lidar_path, pcd_path)
    scene_path.parent.mkdir(parents=True, exist_ok=True)

    scene = fo.Scene()
    relative_pcd_path = os.path.relpath(pcd_path, scene_path.parent)
    world_to_sensor = build_world_to_sensor(ego_pose, calibrated_sensor)
    scene.add(
        fo.PointCloud(
            "lidar",
            relative_pcd_path,
            material=fo.PointCloudMaterial(
                shading_mode="intensity",
                point_size=1.2,
                attenuate_by_distance=False,
            ),
            flag_for_projection=True,
        )
    )

    for annotation in annotations:
        category = category_by_instance.get(annotation["instance_token"], "unknown")
        color = CATEGORY_COLORS.get(category, "#EF4444")
        width, length, height = annotation["size"]
        position, quaternion = transform_annotation_to_sensor(annotation, world_to_sensor)

        scene.add(
            fo.BoxGeometry(
                name=f"{category}:{annotation['token'][:8]}",
                width=float(length),
                height=float(width),
                depth=float(height),
                default_material=fo.MeshBasicMaterial(
                    color=color,
                    wireframe=True,
                    opacity=0.9,
                ),
                position=position,
                quaternion=quaternion,
            )
        )

    scene.write(str(scene_path))
    return scene_path


def build_world_to_camera(
    ego_pose: Dict[str, Any],
    calibrated_sensor: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    world_to_sensor = build_world_to_sensor(ego_pose, calibrated_sensor)
    intrinsic = np.array(calibrated_sensor["camera_intrinsic"], dtype=np.float32)
    return world_to_sensor, intrinsic


def build_world_to_sensor(
    ego_pose: Dict[str, Any],
    calibrated_sensor: Dict[str, Any],
) -> np.ndarray:
    global_to_ego = np.linalg.inv(
        AnnotationProjector.transform_matrix(
            ego_pose["translation"],
            ego_pose["rotation"],
        )
    )
    ego_to_sensor = np.linalg.inv(
        AnnotationProjector.transform_matrix(
            calibrated_sensor["translation"],
            calibrated_sensor["rotation"],
        )
    )
    return ego_to_sensor @ global_to_ego


def rotation_matrix_to_quaternion(rotation_matrix: np.ndarray) -> fo.Quaternion:
    trace = float(np.trace(rotation_matrix))

    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / scale
        y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / scale
        z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / scale
    elif rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
        scale = np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2]) * 2.0
        w = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / scale
        z = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / scale
    elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
        scale = np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2]) * 2.0
        w = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / scale
        x = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1]) * 2.0
        w = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / scale
        x = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / scale
        y = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / scale
        z = 0.25 * scale

    return fo.Quaternion(x=float(x), y=float(y), z=float(z), w=float(w))


def transform_annotation_to_sensor(
    annotation: Dict[str, Any],
    world_to_sensor: np.ndarray,
) -> Tuple[List[float], fo.Quaternion]:
    world_position = np.array([*annotation["translation"], 1.0], dtype=np.float32)
    sensor_position = world_to_sensor @ world_position

    box_rotation_world = Quaternion(array=annotation["rotation"]).rotation_matrix
    sensor_rotation = world_to_sensor[:3, :3] @ box_rotation_world

    return (
        [float(value) for value in sensor_position[:3]],
        rotation_matrix_to_quaternion(sensor_rotation),
    )


def project_annotation_to_bbox(
    annotation: Dict[str, Any],
    world_to_camera: np.ndarray,
    intrinsic: np.ndarray,
    width: int,
    height: int,
) -> Optional[Tuple[float, float, float, float]]:
    corners_cam = AnnotationProjector.get_box_corners_cam(annotation, world_to_camera)

    projected_points: List[np.ndarray] = []
    for start_idx, end_idx in AnnotationProjector.EDGES:
        clipped = AnnotationProjector.clip_line(corners_cam[start_idx], corners_cam[end_idx])
        if clipped is None:
            continue

        start_2d = AnnotationProjector.project_point(clipped[0], intrinsic)
        end_2d = AnnotationProjector.project_point(clipped[1], intrinsic)
        if start_2d is None or end_2d is None:
            continue

        projected_points.append(start_2d)
        projected_points.append(end_2d)

    if not projected_points:
        return None

    coords = np.vstack(projected_points)
    min_x = float(np.clip(coords[:, 0].min(), 0, width))
    min_y = float(np.clip(coords[:, 1].min(), 0, height))
    max_x = float(np.clip(coords[:, 0].max(), 0, width))
    max_y = float(np.clip(coords[:, 1].max(), 0, height))

    if max_x <= min_x or max_y <= min_y:
        return None

    return min_x, min_y, max_x, max_y


def annotation_filter_fields(
    annotations: Sequence[Dict[str, Any]],
    category_by_instance: Dict[str, str],
    attributes_by_token: Dict[str, str],
    visibility_by_token: Dict[str, str],
) -> Dict[str, List[str]]:
    categories: List[str] = []
    instance_tokens: List[str] = []
    visibility_levels: List[str] = []
    attribute_names: List[str] = []

    seen_categories = set()
    seen_instances = set()
    seen_visibility = set()
    seen_attributes = set()

    for annotation in annotations:
        category = category_by_instance.get(annotation["instance_token"], "unknown")
        if category not in seen_categories:
            categories.append(category)
            seen_categories.add(category)

        instance_token = annotation["instance_token"]
        if instance_token not in seen_instances:
            instance_tokens.append(instance_token)
            seen_instances.add(instance_token)

        visibility = visibility_by_token.get(annotation.get("visibility_token", ""), "unknown")
        if visibility not in seen_visibility:
            visibility_levels.append(visibility)
            seen_visibility.add(visibility)

        for attribute_token in annotation.get("attribute_tokens", []):
            attribute_name = attributes_by_token.get(attribute_token, attribute_token)
            if attribute_name not in seen_attributes:
                attribute_names.append(attribute_name)
                seen_attributes.add(attribute_name)

    return {
        "annotation_categories": categories,
        "instance_tokens": instance_tokens,
        "visibility_levels": visibility_levels,
        "attribute_names": attribute_names,
    }


def build_projected_detections(
    annotations: Sequence[Dict[str, Any]],
    ego_pose: Dict[str, Any],
    calibrated_sensor: Dict[str, Any],
    image_width: int,
    image_height: int,
    category_by_instance: Dict[str, str],
    attributes_by_token: Dict[str, str],
    visibility_by_token: Dict[str, str],
) -> Tuple[fo.Detections, Dict[str, List[str]]]:
    world_to_camera, intrinsic = build_world_to_camera(ego_pose, calibrated_sensor)
    detections: List[fo.Detection] = []
    visible_annotations: List[Dict[str, Any]] = []

    for annotation in annotations:
        bbox = project_annotation_to_bbox(
            annotation,
            world_to_camera,
            intrinsic,
            image_width,
            image_height,
        )
        if bbox is None:
            continue

        min_x, min_y, max_x, max_y = bbox
        visible_annotations.append(annotation)

        detections.append(
            fo.Detection(
                label=category_by_instance.get(annotation["instance_token"], "unknown"),
                bounding_box=[
                    min_x / image_width,
                    min_y / image_height,
                    (max_x - min_x) / image_width,
                    (max_y - min_y) / image_height,
                ],
                annotation_token=annotation["token"],
                instance_token=annotation["instance_token"],
                visibility=visibility_by_token.get(annotation.get("visibility_token", ""), "unknown"),
                attribute_names=[
                    attributes_by_token.get(token, token)
                    for token in annotation.get("attribute_tokens", [])
                ],
                num_lidar_pts=int(annotation.get("num_lidar_pts", 0)),
            )
        )

    return (
        fo.Detections(detections=detections),
        annotation_filter_fields(
            visible_annotations,
            category_by_instance,
            attributes_by_token,
            visibility_by_token,
        ),
    )


class RigSplitLoader:
    def __init__(self, dataset_root: Path, split_name: str, split_dir: Path, cache_root: Path):
        self.dataset_root = dataset_root
        self.split_name = split_name
        self.split_dir = split_dir
        self.cache_root = cache_root

        self.samples = load_json(split_dir / "sample.json")
        self.sample_data = load_json(split_dir / "sample_data.json")
        self.sample_annotations = load_json(split_dir / "sample_annotation.json")
        self.scenes = load_json(split_dir / "scene.json")
        self.logs = load_json(split_dir / "log.json")
        self.ego_poses = load_json(split_dir / "ego_pose.json")
        self.calibrated_sensors = load_json(split_dir / "calibrated_sensor.json")
        self.sensors = load_json(split_dir / "sensor.json")
        self.instances = load_json(split_dir / "instance.json")
        self.categories = load_json(split_dir / "category.json")
        self.attributes = load_json(split_dir / "attribute.json")
        self.visibility = load_json(split_dir / "visibility.json")

        self.sample_by_token = {sample["token"]: sample for sample in self.samples}
        self.scene_by_token = {scene["token"]: scene for scene in self.scenes}
        self.log_by_token = {log["token"]: log for log in self.logs}
        self.ego_pose_by_token = {pose["token"]: pose for pose in self.ego_poses}
        self.calibrated_sensor_by_token = {
            sensor["token"]: sensor for sensor in self.calibrated_sensors
        }
        self.sensor_by_token = {sensor["token"]: sensor for sensor in self.sensors}
        self.slice_name_by_sensor_token = self._build_slice_name_by_sensor_token()
        self.instance_by_token = {instance["token"]: instance for instance in self.instances}
        self.category_name_by_token = {
            category["token"]: category["name"] for category in self.categories
        }
        self.attribute_name_by_token = {
            attribute["token"]: attribute["name"] for attribute in self.attributes
        }
        self.visibility_name_by_token = {
            entry["token"]: entry.get("level", entry.get("description", "unknown"))
            for entry in self.visibility
        }

        self.category_by_instance_token = {
            instance_token: self.category_name_by_token.get(instance["category_token"], "unknown")
            for instance_token, instance in self.instance_by_token.items()
        }

        self.annotations_by_sample = defaultdict(list)
        for annotation in self.sample_annotations:
            self.annotations_by_sample[annotation["sample_token"]].append(annotation)

        self.sample_data_by_sample = defaultdict(list)
        self.slice_entries_by_modality = defaultdict(list)
        for sample_data in self.sample_data:
            calibrated_sensor = self.calibrated_sensor_by_token[sample_data["calibrated_sensor_token"]]
            sensor = self.sensor_by_token[calibrated_sensor["sensor_token"]]
            slice_name = self.slice_name_by_sensor_token[sensor["token"]]
            modality = sensor["modality"]

            self.sample_data_by_sample[sample_data["sample_token"]].append((slice_name, sample_data))
            slice_entry = (slice_name, sensor, calibrated_sensor)
            if slice_entry not in self.slice_entries_by_modality[modality]:
                self.slice_entries_by_modality[modality].append(slice_entry)

        self.frame_index_by_sample = self._build_frame_indices()

    def _build_slice_name_by_sensor_token(self) -> Dict[str, str]:
        tokens_by_base_name: Dict[str, List[str]] = defaultdict(list)
        for sensor in self.sensors:
            tokens_by_base_name[sensor_slice_base_name(sensor)].append(sensor["token"])

        slice_name_by_sensor_token: Dict[str, str] = {}
        for base_name, sensor_tokens in tokens_by_base_name.items():
            if len(sensor_tokens) == 1:
                slice_name_by_sensor_token[sensor_tokens[0]] = base_name
                continue

            for index, sensor_token in enumerate(sorted(sensor_tokens), start=1):
                slice_name_by_sensor_token[sensor_token] = f"{base_name}__{index:02d}"

        return slice_name_by_sensor_token

    def _build_frame_indices(self) -> Dict[str, int]:
        frame_indices: Dict[str, int] = {}
        for scene in self.scenes:
            sample_token = scene["first_sample_token"]
            frame_index = 0
            while sample_token:
                frame_indices[sample_token] = frame_index
                sample_token = self.sample_by_token[sample_token]["next"]
                frame_index += 1
        return frame_indices

    def all_slice_entries(self) -> List[Tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]]:
        slice_entries: List[Tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]] = []
        for modality_entries in self.slice_entries_by_modality.values():
            slice_entries.extend(modality_entries)
        return slice_entries

    def default_slice_name(self) -> str:
        return choose_default_slice(self.all_slice_entries())

    def iter_group_slices(self) -> Iterator[fo.Sample]:
        for sample in self.samples:
            sample_token = sample["token"]
            scene = self.scene_by_token[sample["scene_token"]]
            log = self.log_by_token[scene["log_token"]]
            scene_annotations = self.annotations_by_sample.get(sample_token, [])
            group = fo.Group()
            sample_sensor_data = self.sample_data_by_sample.get(sample_token, [])
            if not sample_sensor_data:
                continue

            sample_slice_entries: List[Tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]] = []
            for slice_name, sample_data in sample_sensor_data:
                calibrated_sensor = self.calibrated_sensor_by_token[sample_data["calibrated_sensor_token"]]
                sensor = self.sensor_by_token[calibrated_sensor["sensor_token"]]
                sample_slice_entries.append((slice_name, sensor, calibrated_sensor))

            default_slice = choose_default_slice(sample_slice_entries)
            sample_filter_fields = annotation_filter_fields(
                scene_annotations,
                self.category_by_instance_token,
                self.attribute_name_by_token,
                self.visibility_name_by_token,
            )

            common_fields = {
                "rig_name": self.dataset_root.name,
                "split": self.split_name,
                "scene_name": scene["name"],
                "scene_token": scene["token"],
                "scene_description": scene.get("description", ""),
                "frame_index": self.frame_index_by_sample[sample_token],
                "timestamp": int(sample["timestamp"]),
                "location": log.get("location"),
                "weather": log.get("weather"),
                "default_slice": default_slice,
                **sample_filter_fields,
            }

            for slice_name, sample_data in sample_sensor_data:
                calibrated_sensor = self.calibrated_sensor_by_token[sample_data["calibrated_sensor_token"]]
                sensor = self.sensor_by_token[calibrated_sensor["sensor_token"]]
                channel = sensor.get("channel") or slice_name
                modality = sensor["modality"]
                group_membership = group.element(slice_name)

                if modality == "camera":
                    filepath = self.dataset_root / sample_data["filename"]
                    ego_pose = self.ego_pose_by_token[sample_data["ego_pose_token"]]
                    detections, filter_fields = build_projected_detections(
                        scene_annotations,
                        ego_pose,
                        calibrated_sensor,
                        int(sample_data["width"]),
                        int(sample_data["height"]),
                        self.category_by_instance_token,
                        self.attribute_name_by_token,
                        self.visibility_name_by_token,
                    )

                    yield fo.Sample(
                        filepath=str(filepath),
                        group=group_membership,
                        group_slice=slice_name,
                        sensor_channel=channel,
                        sensor_modality=modality,
                        sensor_token=sensor["token"],
                        calibrated_sensor_token=calibrated_sensor["token"],
                        projected_detections=detections,
                        image_width=int(sample_data["width"]),
                        image_height=int(sample_data["height"]),
                        num_annotations=len(scene_annotations),
                        num_projected_annotations=len(detections.detections),
                        camera_intrinsic=calibrated_sensor["camera_intrinsic"],
                        sensor_translation=calibrated_sensor["translation"],
                        sensor_rotation=calibrated_sensor["rotation"],
                        projected_annotation_categories=filter_fields["annotation_categories"],
                        projected_instance_tokens=filter_fields["instance_tokens"],
                        projected_visibility_levels=filter_fields["visibility_levels"],
                        projected_attribute_names=filter_fields["attribute_names"],
                        **common_fields,
                    )
                    continue

                if modality != "lidar":
                    continue

                lidar_source_path = self.dataset_root / sample_data["filename"]
                scene_cache_dir = self.cache_root / self.split_name / scene["name"] / slice_name
                scene_path = scene_cache_dir / f"{sample_token}.fo3d"
                pcd_path = scene_cache_dir / f"{sample_token}.pcd"
                ensure_lidar_scene(
                    lidar_source_path,
                    scene_path,
                    pcd_path,
                    scene_annotations,
                    self.category_by_instance_token,
                    self.ego_pose_by_token[sample_data["ego_pose_token"]],
                    calibrated_sensor,
                )
                yield fo.Sample(
                    filepath=str(scene_path),
                    group=group_membership,
                    group_slice=slice_name,
                    sensor_channel=channel,
                    sensor_modality=modality,
                    sensor_token=sensor["token"],
                    calibrated_sensor_token=calibrated_sensor["token"],
                    source_lidar_path=str(lidar_source_path),
                    num_annotations=len(scene_annotations),
                    sensor_translation=calibrated_sensor["translation"],
                    sensor_rotation=calibrated_sensor["rotation"],
                    **common_fields,
                )


def batched(items: Iterable[fo.Sample], size: int) -> Iterator[List[fo.Sample]]:
    batch: List[fo.Sample] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def create_indices(dataset: fo.Dataset) -> None:
    index_fields = [
        "split",
        "rig_name",
        "scene_name",
        "scene_token",
        "default_slice",
        "frame_index",
        "timestamp",
        "group_slice",
        "sensor_channel",
        "sensor_modality",
        "sensor_token",
        "calibrated_sensor_token",
        "location",
        "weather",
        "annotation_categories",
        "visibility_levels",
        "instance_tokens",
        "attribute_names",
        "num_annotations",
        "num_projected_annotations",
        "projected_annotation_categories",
        "projected_instance_tokens",
        "projected_visibility_levels",
        "projected_attribute_names",
        "projected_detections.detections.label",
        "projected_detections.detections.instance_token",
        "projected_detections.detections.visibility",
    ]

    for field_name in index_fields:
        try:
            dataset.create_index(field_name)
        except Exception as exc:
            print(f"Warning: failed to create index on {field_name}: {exc}")


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.input_path).expanduser().resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_root}")

    split_dirs = discover_split_dirs(dataset_root, args.splits)
    if not split_dirs:
        requested = ", ".join(args.splits) if args.splits else "mini/test/trainval"
        raise ValueError(f"No matching split directories found for {requested} in {dataset_root}")

    if fo.dataset_exists(args.dataset_name):
        fo.delete_dataset(args.dataset_name)

    cache_root = dataset_root / ".voxel51" / args.dataset_name
    loaders: List[RigSplitLoader] = []
    dataset_default_candidates: List[Tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]] = []
    for split_name, split_dir in split_dirs:
        loader = RigSplitLoader(dataset_root, split_name, split_dir, cache_root)
        loaders.append(loader)
        dataset_default_candidates.extend(loader.all_slice_entries())

    if not dataset_default_candidates:
        raise ValueError(f"No sensors found in {dataset_root}")

    default_slice_counts = Counter(slice_name for slice_name, _, _ in dataset_default_candidates)
    max_count = max(default_slice_counts.values())
    dataset_default_slice = choose_default_slice(
        [
            entry
            for entry in dataset_default_candidates
            if default_slice_counts[entry[0]] == max_count
        ]
    )

    dataset = fo.Dataset(args.dataset_name)
    dataset.add_group_field("group", default=dataset_default_slice)

    total_inserted_slices = 0

    for loader in loaders:
        print(f"Loading split: {loader.split_name} from {loader.split_dir}")
        for batch in batched(loader.iter_group_slices(), args.batch_size):
            dataset.add_samples(batch)
            total_inserted_slices += len(batch)

    dataset.persistent = True
    create_indices(dataset)

    print(
        f"Created grouped dataset '{args.dataset_name}' with {len(dataset)} groups "
        f"and {total_inserted_slices} slices"
    )
    print("Load it in Python with: fo.load_dataset(%r)" % args.dataset_name)
    print("List available datasets with: fo.list_datasets()")
    print("Launch the app in python with: fo.launch_app(%r)" % args.dataset_name)
    print("Launch the app from the command line with: fiftyone app launch %r" % args.dataset_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())