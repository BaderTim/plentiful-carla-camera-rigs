#!/usr/bin/env python3
"""
Main data-capture loop: CARLA → sensor frames → nuScenes-format dataset.

DETERMINISM
-----------
Traffic and pedestrian actors are spawned once per scene via
``record_trajectories.py`` using a firetruck (largest available vehicle) as
the ego.  Replaying those pre-recorded transforms with ``set_transform()``
and physics disabled guarantees identical actor behaviour regardless of the
ego vehicle or camera rig used during capture.

Workflow
--------
1. Run ``core/record_trajectories.py`` to generate HDF5 trajectory files.
2. Run this script pointing ``--trajectories`` at that directory.

Usage::

    python core/scene_runner.py \\
        --output ./output/data \\
        --scenes configs/scenes.json \\
        --camera-rigs configs/rigs/R1.json \\
        --trajectories ./output/trajectories
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import queue
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is importable when the script is run directly.
sys.path.insert(0, str(Path(__file__).parent.parent))

import carla
import numpy as np

from lib.bbox import BoundingBoxBuilder, TrafficInfrastructureBBox
from lib.utils.depth_utils import DepthPointCloudAssembler
from lib.detection import CategoryColors, LightHeadProxy, ObjectDetector
from lib.utils.image_utils import ImageProcessor
from lib.utils.logging_utils import log_print, setup_logging
from lib.nuscenes_builder import NuScenesDatasetBuilder
from lib.nuscenes_standards import CAT_TRAFFIC_LIGHT, CAT_TRAFFIC_SIGN
from lib.trajectory_utils import TrajectoryPlayer, validate_trajectories_exist

from lib.io_workers import IOWorkersMixin

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
SENSOR_FPS = 2          # captured at 2 FPS
SIMULATION_FPS = 10     # physics ticks at 10 FPS
RECORDING_DURATION = 30.0  # seconds of recording per scenario
SETTLE_DURATION = 1.0   # settling period before recording starts

DEPTH_RESOLUTION_SCALE = 0.5
DEPTH_IMAGE_WIDTH = int(IMAGE_WIDTH * DEPTH_RESOLUTION_SCALE)
DEPTH_IMAGE_HEIGHT = int(IMAGE_HEIGHT * DEPTH_RESOLUTION_SCALE)
SENSOR_TICK_SECONDS = 1.0 / SENSOR_FPS
SPECTATOR_FOLLOW_FPS = 2.0
SPECTATOR_UPDATE_INTERVAL_TICKS = max(1, int(SIMULATION_FPS / SPECTATOR_FOLLOW_FPS))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def determine_scene_split(scene_id: str) -> str:
    """Return the nuScenes split name for a scene ID prefix.

    Args:
        scene_id: Scene identifier (e.g. ``"mini_01"``).

    Returns:
        One of ``"trainval"``, ``"test"``, ``"mini"``.
    """
    if scene_id.startswith(("train_", "val_")):
        return "trainval"
    if scene_id.startswith("test_"):
        return "test"
    if scene_id.startswith("mini_"):
        return "mini"
    return "trainval"


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

class SceneRunner(IOWorkersMixin):
    """Orchestrate CARLA scene execution and nuScenes-format data capture.

    Inherits threaded disk-I/O worker methods from
    :class:`~core.io_workers.IOWorkersMixin`.

    Attributes:
        client: CARLA client connection.
        world: Currently loaded :class:`carla.World`.
        ego_vehicle: Spawned ego actor.
        cameras: RGB camera actors.
        depth_cameras: Depth-camera actors mirroring each RGB camera.
        lidar: LiDAR actor (or ``None`` when disabled).
        camera_data: Per-camera-id latest received image dicts.
        depth_camera_data: Per-camera-id latest received depth dicts.
        depth_camera_params: Per-camera intrinsic parameter cache.
        lidar_data: Latest received LiDAR measurement dict.
        image_save_queue: Bounded queue for JPEG image serialisation.
        lidar_save_queue: Bounded queue for LiDAR PCD serialisation.
        depth_debug_queue: Bounded queue for depth PNG serialisation.
        image_save_threads: Active image-saver threads.
        lidar_save_threads: Active LiDAR-saver threads.
        depth_debug_threads: Active depth-debug-saver threads.
        nuscenes_builders: ``{rig_split_key: NuScenesDatasetBuilder}`` map.
        traffic_vehicles: Spawned traffic-vehicle actors.
        walkers: Spawned pedestrian actors.
        trajectory_player: :class:`~lib.trajectory_utils.TrajectoryPlayer`
            for the current scene (or ``None``).
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 2000,
        timeout: float = 300.0,
        debug: bool = False,
        no_lidar: bool = False,
        max_detection_distance: float = 80.0,
    ) -> None:
        """Connect to CARLA and initialise all state.

        Args:
            host: CARLA simulator hostname.
            port: CARLA simulator port.
            timeout: Client timeout in seconds.
            debug: Enable debug bounding-box visualisation.
            no_lidar: Skip LiDAR sensor setup.
            max_detection_distance: Annotation detection radius (metres).
        """
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)
        self.world = None
        self.ego_vehicle = None
        self.cameras: List[carla.Actor] = []
        self.depth_cameras: List[carla.Actor] = []
        self.lidar = None
        self.camera_data: Dict[str, Any] = {}
        self.depth_camera_data: Dict[str, Any] = {}
        self.depth_camera_params: Dict[str, Any] = {}
        self.last_captured_camera_frames: Dict[str, int] = {}
        self.last_captured_depth_frames: Dict[str, int] = {}
        self.lidar_data = None
        self.original_settings = None
        self.frame_counter = 0
        self.captured_frames: List = []
        self.sensor_tick_counter = 0
        self.recording_start_world_frame: int | None = None
        self.next_sensor_capture_frame: int | None = None
        self.last_spectator_update_world_frame: int | None = None
        self.previous_ego_pose = None
        self.recording_start_time = None

        # Bounded queues prevent RAM leaks when I/O is slower than simulation.
        self.image_save_queue: queue.Queue = queue.Queue(maxsize=200)
        self.lidar_save_queue: queue.Queue = queue.Queue(maxsize=100)
        self.depth_debug_queue: queue.Queue = queue.Queue(maxsize=100)
        self.image_save_threads: List = []
        self.lidar_save_threads: List = []
        self.depth_debug_threads: List = []

        self.debug = debug
        self.no_lidar = no_lidar
        self.capture_depth_debug = debug
        self.max_detection_distance = max_detection_distance
        self.depth_assembler = DepthPointCloudAssembler(max_depth_m=max_detection_distance)

        self.traffic_vehicles: List = []
        self.walkers: List = []
        self.trajectory_player: TrajectoryPlayer | None = None

        self.nuscenes_builders: Dict[str, NuScenesDatasetBuilder] = {}
        self.current_scene_id: str | None = None
        self.capture_timing_totals: Dict[str, float] = {}
        self.capture_timing_samples = 0

    # ------------------------------------------------------------------
    # World / sensor setup
    # ------------------------------------------------------------------

    def enable_synchronous_mode(self) -> None:
        """Enable fixed-timestep synchronous CARLA mode."""
        self.original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / SIMULATION_FPS
        settings.no_rendering_mode = False
        self.world.apply_settings(settings)
        log_print(
            f"Synchronous mode: {SIMULATION_FPS:.0f} Hz sim / {SENSOR_FPS:.1f} Hz sensor"
        )

    def disable_synchronous_mode(self) -> None:
        """Restore world settings saved before synchronous mode was enabled."""
        if self.original_settings is not None:
            try:
                self.world.apply_settings(self.original_settings)
            except Exception as exc:
                log_print(f"Warning: Could not restore world settings: {exc}", "WARNING")

    def setup_nuscenes_dataset(
        self,
        rig_name: str,
        sensor_configs: Dict,
        output_dir: str,
        split: str | None = None,
        max_detection_distance: float = 80.0,
        version: str = "v1.0",
    ) -> NuScenesDatasetBuilder:
        """Create and register a :class:`NuScenesDatasetBuilder` for a rig/split pair.

        Args:
            rig_name: Rig identifier (e.g. ``"R1"``).
            sensor_configs: Rig config dict (cameras + optional lidar).
            output_dir: Root output directory.
            split: nuScenes split (``"trainval"``, ``"test"``, ``"mini"``).
            max_detection_distance: Annotation radius (metres).
            version: Dataset version string.

        Returns:
            The newly created :class:`NuScenesDatasetBuilder`.
        """
        builder_key = f"{rig_name}_{split}" if split else rig_name
        builder = NuScenesDatasetBuilder(
            rig_name, output_dir, split, max_detection_distance, version
        )
        camera_configs = sensor_configs.get("cameras", [])
        for cam in camera_configs:
            cam["image_size_x"] = IMAGE_WIDTH
            cam["image_size_y"] = IMAGE_HEIGHT
        builder.initialize_dataset(camera_configs, sensor_configs.get("lidar"))
        self.nuscenes_builders[builder_key] = builder
        return builder

    def load_camera_rig(self, rig_path: str) -> Dict[str, Any]:
        """Load a camera rig JSON config file.

        Args:
            rig_path: Path to the rig JSON file.
        """
        with open(rig_path, "r") as f:
            return json.load(f)

    def load_scenes(self, scenes_path: str) -> Dict[str, Any]:
        """Load the scenes.json config file.

        Args:
            scenes_path: Path to the JSON file.
        """
        with open(scenes_path, "r") as f:
            return json.load(f)

    def set_weather(self, weather_preset: str) -> None:
        """Apply a named CARLA weather preset.

        Falls back to ``ClearNoon`` for unknown preset names.

        Args:
            weather_preset: CARLA weather preset string.
        """
        presets = {
            "ClearNoon": carla.WeatherParameters.ClearNoon,
            "CloudyNoon": carla.WeatherParameters.CloudyNoon,
            "WetNoon": carla.WeatherParameters.WetNoon,
            "WetCloudyNoon": carla.WeatherParameters.WetCloudyNoon,
            "SoftRainNoon": carla.WeatherParameters.SoftRainNoon,
            "MidRainyNoon": carla.WeatherParameters.MidRainyNoon,
            "HardRainNoon": carla.WeatherParameters.HardRainNoon,
            "ClearSunset": carla.WeatherParameters.ClearSunset,
            "CloudySunset": carla.WeatherParameters.CloudySunset,
            "WetSunset": carla.WeatherParameters.WetSunset,
            "SoftRainSunset": carla.WeatherParameters.SoftRainSunset,
            "MidRainSunset": carla.WeatherParameters.MidRainSunset,
            "HardRainSunset": carla.WeatherParameters.HardRainSunset,
        }
        if weather_preset not in presets:
            log_print(f"Unknown weather preset '{weather_preset}', using ClearNoon.", "WARNING")
        self.world.set_weather(presets.get(weather_preset, carla.WeatherParameters.ClearNoon))

    def spawn_ego_vehicle(self, vehicle_type: str, spawn_point_idx: int) -> carla.Transform:
        """Spawn the ego vehicle at a map spawn point.

        Args:
            vehicle_type: Blueprint ID of the vehicle to spawn.
            spawn_point_idx: Index into the map's spawn-point list.

        Returns:
            The spawn :class:`~carla.Transform` used.
        """
        spawn_points = self.world.get_map().get_spawn_points()
        if spawn_point_idx >= len(spawn_points):
            log_print(
                f"Spawn point {spawn_point_idx} unavailable, using 0.", "WARNING"
            )
            spawn_point_idx = 0

        lib = self.world.get_blueprint_library()
        vehicle_bp = lib.find(vehicle_type)
        if vehicle_bp is None:
            log_print(
                f"Vehicle '{vehicle_type}' not found — using first available.", "WARNING"
            )
            vehicle_bp = lib.filter("vehicle.*")[0]

        spawn_tf = spawn_points[spawn_point_idx]
        self.ego_vehicle = self.world.spawn_actor(vehicle_bp, spawn_tf)
        self.world.tick()
        self.world.tick()
        return spawn_tf

    def setup_cameras(self, camera_configs: List[Dict[str, Any]]) -> None:
        """Spawn RGB + depth cameras on the ego vehicle.

        Args:
            camera_configs: List of per-camera config dicts from the rig JSON.
        """
        lib = self.world.get_blueprint_library()
        camera_bp = lib.find("sensor.camera.rgb")
        depth_bp = lib.find("sensor.camera.depth")
        self.depth_assembler = DepthPointCloudAssembler(max_depth_m=self.max_detection_distance)

        camera_bp.set_attribute("image_size_x", str(IMAGE_WIDTH))
        camera_bp.set_attribute("image_size_y", str(IMAGE_HEIGHT))
        camera_bp.set_attribute("sensor_tick", str(SENSOR_TICK_SECONDS))
        depth_bp.set_attribute("image_size_x", str(DEPTH_IMAGE_WIDTH))
        depth_bp.set_attribute("image_size_y", str(DEPTH_IMAGE_HEIGHT))
        depth_bp.set_attribute("sensor_tick", str(SENSOR_TICK_SECONDS))

        for cam_cfg in camera_configs:
            camera_bp.set_attribute("fov", str(cam_cfg["fov"]))
            depth_bp.set_attribute("fov", str(cam_cfg["fov"]))

            loc = carla.Location(*cam_cfg["location"])
            rot = carla.Rotation(*cam_cfg["rotation"])
            transform = carla.Transform(loc, rot)

            camera = self.world.spawn_actor(camera_bp, transform, attach_to=self.ego_vehicle)
            cam_id = cam_cfg["id"]
            self.camera_data[cam_id] = None
            self.depth_camera_data[cam_id] = None
            self.last_captured_camera_frames[cam_id] = -1
            self.last_captured_depth_frames[cam_id] = -1
            self.depth_camera_params[cam_id] = {
                "width": DEPTH_IMAGE_WIDTH,
                "height": DEPTH_IMAGE_HEIGHT,
                "fov": cam_cfg["fov"],
            }
            self.depth_assembler.register_camera(
                cam_id, DEPTH_IMAGE_WIDTH, DEPTH_IMAGE_HEIGHT, cam_cfg["fov"]
            )

            def _rgb_cb(image, cid=cam_id, sensor=camera):
                try:
                    if (
                        hasattr(self, "camera_data")
                        and self.camera_data is not None
                        and cid in self.camera_data
                    ):
                        self.camera_data[cid] = {
                            "image": image,
                            "frame": image.frame,
                            "timestamp": image.timestamp,
                            "transform": sensor.get_transform(),
                        }
                except Exception:
                    pass

            camera.listen(_rgb_cb)
            self.cameras.append(camera)

            depth_sensor = self.world.spawn_actor(
                depth_bp, transform, attach_to=self.ego_vehicle
            )
            self.depth_cameras.append(depth_sensor)

            def _depth_cb(image, cid=cam_id, sensor=depth_sensor):
                try:
                    if (
                        hasattr(self, "depth_camera_data")
                        and self.depth_camera_data is not None
                        and cid in self.depth_camera_data
                    ):
                        self.depth_camera_data[cid] = {
                            "image": image,
                            "frame": image.frame,
                            "timestamp": image.timestamp,
                            "transform": sensor.get_transform(),
                        }
                except Exception:
                    pass

            depth_sensor.listen(_depth_cb)

    def setup_lidar(self, lidar_config: Dict[str, Any]) -> None:
        """Spawn a LiDAR sensor on the ego vehicle.

        Args:
            lidar_config: LiDAR config dict from the rig JSON.
        """
        lib = self.world.get_blueprint_library()
        lidar_bp = lib.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("channels", str(lidar_config.get("channels", 32)))
        lidar_bp.set_attribute("range", str(lidar_config.get("range", 100.0)))
        lidar_bp.set_attribute(
            "points_per_second", str(lidar_config.get("points_per_second", 56000))
        )
        lidar_bp.set_attribute("rotation_frequency", str(SIMULATION_FPS))
        lidar_bp.set_attribute("sensor_tick", str(SENSOR_TICK_SECONDS))

        loc = carla.Location(*lidar_config["location"])
        rot = carla.Rotation(*lidar_config["rotation"])
        self.lidar = self.world.spawn_actor(
            lidar_bp, carla.Transform(loc, rot), attach_to=self.ego_vehicle
        )
        self.lidar_data = None

        def _lidar_cb(measurement):
            try:
                self.lidar_data = {
                    "measurement": measurement,
                    "frame": self.frame_counter,
                    "measurement_frame": getattr(measurement, "frame", None),
                    "timestamp": measurement.timestamp,
                }
            except Exception:
                pass

        self.lidar.listen(_lidar_cb)

    # ------------------------------------------------------------------
    # Traffic spawning (trajectory-based)
    # ------------------------------------------------------------------

    def spawn_traffic_from_trajectories(
        self, trajectory_player: TrajectoryPlayer
    ) -> None:
        """Spawn traffic actors from pre-recorded trajectory metadata.

        Vehicles are spawned with physics disabled for puppet-controlled
        replay via ``set_transform()``.  Pedestrians retain physics for
        walk-animation fidelity.

        Args:
            trajectory_player: Loaded :class:`~lib.trajectory_utils.TrajectoryPlayer`.
        """
        self.traffic_vehicles = []
        self.walkers = []
        lib = self.world.get_blueprint_library()

        traffic_bps, traffic_spawns = trajectory_player.get_traffic_spawn_info()
        for i, (bp_id, (loc, rot)) in enumerate(zip(traffic_bps, traffic_spawns)):
            try:
                bp = lib.find(bp_id)
                if bp is None:
                    log_print(f"Blueprint {bp_id} not found — skipping vehicle {i}.", "WARNING")
                    self.traffic_vehicles.append(None)
                    continue
                spawn_tf = carla.Transform(
                    carla.Location(*loc), carla.Rotation(*rot)
                )
                vehicle = self.world.spawn_actor(bp, spawn_tf)
                vehicle.set_simulate_physics(False)
                self.traffic_vehicles.append(vehicle)
            except RuntimeError as exc:
                log_print(f"Failed to spawn traffic vehicle {i} ({bp_id}): {exc}", "WARNING")
                self.traffic_vehicles.append(None)

        spawned = sum(1 for v in self.traffic_vehicles if v is not None)
        log_print(f"Spawned {spawned}/{len(traffic_bps)} traffic vehicles (physics off).")

        ped_bps, ped_spawns = trajectory_player.get_pedestrian_spawn_info()
        for i, (bp_id, (loc, rot)) in enumerate(zip(ped_bps, ped_spawns)):
            try:
                bp = lib.find(bp_id)
                if bp is None:
                    log_print(f"Blueprint {bp_id} not found — skipping pedestrian {i}.", "WARNING")
                    self.walkers.append(None)
                    continue
                if bp.has_attribute("is_invincible"):
                    bp.set_attribute("is_invincible", "false")
                spawn_tf = carla.Transform(
                    carla.Location(*loc), carla.Rotation(*rot)
                )
                walker = self.world.spawn_actor(bp, spawn_tf)
                self.walkers.append(walker)
            except RuntimeError as exc:
                log_print(f"Failed to spawn pedestrian {i} ({bp_id}): {exc}", "WARNING")
                self.walkers.append(None)

        ped_spawned = sum(1 for w in self.walkers if w is not None)
        log_print(f"Spawned {ped_spawned}/{len(ped_bps)} pedestrians.")
        trajectory_player.set_traffic_vehicles(self.traffic_vehicles)
        trajectory_player.set_pedestrians(self.walkers)
        trajectory_player.set_client(self.client)

    # ------------------------------------------------------------------
    # Spectator / pose
    # ------------------------------------------------------------------

    def update_spectator_follow_ego(
        self,
        world_frame_id: int | None = None,
        force: bool = False,
    ) -> None:
        """Move the CARLA spectator to follow the ego vehicle.

        Outside debug mode, updates are throttled to reduce spectator RPC cost.
        """
        if self.ego_vehicle is None:
            return
        if not force and not self.debug:
            if world_frame_id is None:
                return
            if self.last_spectator_update_world_frame is not None:
                if (
                    world_frame_id - self.last_spectator_update_world_frame
                    < SPECTATOR_UPDATE_INTERVAL_TICKS
                ):
                    return
        t = self.ego_vehicle.get_transform()
        fwd = t.rotation.get_forward_vector()
        spec_loc = t.location + carla.Location(x=-10 * fwd.x, y=-10 * fwd.y, z=5)
        self.world.get_spectator().set_transform(carla.Transform(spec_loc, t.rotation))
        if world_frame_id is not None:
            self.last_spectator_update_world_frame = world_frame_id

    def sensors_have_frame(self, world_frame_id: int) -> bool:
        """Return ``True`` when all camera and depth sensors have *world_frame_id*."""
        return (
            all(
                d is not None and d["frame"] == world_frame_id
                for d in self.camera_data.values()
            )
            and all(
                d is not None and d["frame"] == world_frame_id
                for d in self.depth_camera_data.values()
            )
        )

    def sensors_have_fresh_data(self) -> bool:
        """Return ``True`` when all camera and depth sensors have new frames."""
        return (
            all(
                d is not None
                and d["frame"] > self.last_captured_camera_frames.get(cid, -1)
                for cid, d in self.camera_data.items()
            )
            and all(
                d is not None
                and d["frame"] > self.last_captured_depth_frames.get(cid, -1)
                for cid, d in self.depth_camera_data.items()
            )
        )

    def get_synchronized_sensor_frame(self, require_fresh: bool = False) -> int | None:
        """Return the shared RGB/depth frame id when all sensors agree.

        When ``require_fresh`` is true, the shared frame must be newer than the
        last captured frame for every sensor.
        """
        frames: List[int] = []

        for cam_id, data in self.camera_data.items():
            if data is None:
                return None
            frame_id = data["frame"]
            if require_fresh and frame_id <= self.last_captured_camera_frames.get(cam_id, -1):
                return None
            frames.append(frame_id)

        for cam_id, data in self.depth_camera_data.items():
            if data is None:
                return None
            frame_id = data["frame"]
            if require_fresh and frame_id <= self.last_captured_depth_frames.get(cam_id, -1):
                return None
            frames.append(frame_id)

        if not frames:
            return None
        if len(set(frames)) != 1:
            return None
        return frames[0]

    def get_sensor_bundle_timestamp(self, sensor_frame_id: int) -> float | None:
        """Return a representative timestamp for a synchronized sensor bundle."""
        for data in self.camera_data.values():
            if data is not None and data["frame"] == sensor_frame_id:
                return data["timestamp"]
        for data in self.depth_camera_data.values():
            if data is not None and data["frame"] == sensor_frame_id:
                return data["timestamp"]
        return None

    def mark_sensor_data_captured(self) -> None:
        """Record the latest sensor frames as consumed by the current sample."""
        for cam_id, data in self.camera_data.items():
            if data is not None:
                self.last_captured_camera_frames[cam_id] = data["frame"]
        for cam_id, data in self.depth_camera_data.items():
            if data is not None:
                self.last_captured_depth_frames[cam_id] = data["frame"]

    def wait_for_fresh_sensor_data(
        self,
        max_attempts: int = 40,
        sleep_interval: float = 0.01,
    ) -> bool:
        """Block until all camera and depth sensors deliver fresh frames."""
        for _ in range(max_attempts):
            if self.sensors_have_fresh_data():
                return True
            time.sleep(sleep_interval)
        return False

    def wait_for_synchronized_sensor_frame(
        self,
        target_frame_id: int,
        max_attempts: int = 200,
        sleep_interval: float = 0.01,
        log_failures: bool = False,
    ) -> bool:
        """Wait until all RGB and depth sensors report ``target_frame_id``."""
        for _ in range(max_attempts):
            current = self.get_synchronized_sensor_frame(require_fresh=True)
            if current == target_frame_id:
                return True
            time.sleep(sleep_interval)

        if log_failures:
            current = self.get_synchronized_sensor_frame(require_fresh=False)
            log_print(
                f"Warning: Timed out waiting for synchronized sensor frame {target_frame_id} "
                f"(current={current})",
                "WARNING",
            )
        return False

    def reset_capture_timings(self) -> None:
        """Reset running timing totals for capture-stage profiling."""
        self.capture_timing_totals = {
            "tick_ms": 0.0,
            "wait_ms": 0.0,
            "depth_ms": 0.0,
            "annotation_ms": 0.0,
            "enqueue_ms": 0.0,
        }
        self.capture_timing_samples = 0

    def log_capture_timing_summary(self) -> None:
        """Log running-average timing totals and queue pressure."""
        if self.capture_timing_samples <= 0:
            return

        averages = {
            key: value / self.capture_timing_samples
            for key, value in self.capture_timing_totals.items()
        }
        image_fill = self.image_save_queue.qsize() / self.image_save_queue.maxsize
        lidar_fill = self.lidar_save_queue.qsize() / self.lidar_save_queue.maxsize
        depth_fill = self.depth_debug_queue.qsize() / self.depth_debug_queue.maxsize
        log_print(
            "    Perf avg "
            f"tick={averages['tick_ms']:.1f}ms "
            f"wait={averages['wait_ms']:.1f}ms "
            f"depth={averages['depth_ms']:.1f}ms "
            f"annot={averages['annotation_ms']:.1f}ms "
            f"enqueue={averages['enqueue_ms']:.1f}ms "
            f"queues(image={self.image_save_queue.qsize()}/{self.image_save_queue.maxsize} "
            f"{image_fill * 100:.0f}%, "
            f"lidar={self.lidar_save_queue.qsize()}/{self.lidar_save_queue.maxsize} "
            f"{lidar_fill * 100:.0f}%, "
            f"depth={self.depth_debug_queue.qsize()}/{self.depth_debug_queue.maxsize} "
            f"{depth_fill * 100:.0f}%)"
        )

    def get_ego_pose_data(self) -> Dict[str, Any] | None:
        """Return the ego vehicle pose dict for the current tick.

        Returns:
            Dict with ``ego_pose_global``, ``ego_pose_relative``,
            ``velocity_ms``, ``speed_ms``, ``timestamp``, and
            ``sensor_tick_idx`` fields.  ``None`` if there is no ego vehicle.
        """
        if self.ego_vehicle is None:
            return None

        cur_tf = self.ego_vehicle.get_transform()
        cur_vel = self.ego_vehicle.get_velocity()
        cur_loc = cur_tf.location
        cur_rot = cur_tf.rotation

        rel_translation = [0.0, 0.0, 0.0]
        rel_rotation = [0.0, 0.0, 0.0]
        if self.previous_ego_pose is not None:
            prev_loc = self.previous_ego_pose["location"]
            prev_rot = self.previous_ego_pose["rotation"]
            rel_translation = [
                cur_loc.x - prev_loc.x,
                cur_loc.y - prev_loc.y,
                cur_loc.z - prev_loc.z,
            ]
            rel_rotation = [
                cur_rot.pitch - prev_rot.pitch,
                cur_rot.yaw - prev_rot.yaw,
                cur_rot.roll - prev_rot.roll,
            ]

        pose_data = {
            "timestamp": (
                self.recording_start_time
                + self.sensor_tick_counter * (1.0 / SENSOR_FPS)
            ),
            "sensor_tick_idx": self.sensor_tick_counter,
            "ego_pose_global": {
                "location": {"x": cur_loc.x, "y": cur_loc.y, "z": cur_loc.z},
                "rotation": {
                    "pitch": cur_rot.pitch,
                    "yaw": cur_rot.yaw,
                    "roll": cur_rot.roll,
                },
            },
            "ego_pose_relative": {
                "translation": {
                    "x": rel_translation[0],
                    "y": rel_translation[1],
                    "z": rel_translation[2],
                },
                "rotation": {
                    "pitch": rel_rotation[0],
                    "yaw": rel_rotation[1],
                    "roll": rel_rotation[2],
                },
            },
            "velocity_ms": {"x": cur_vel.x, "y": cur_vel.y, "z": cur_vel.z},
            "speed_ms": math.sqrt(cur_vel.x**2 + cur_vel.y**2 + cur_vel.z**2),
        }
        self.previous_ego_pose = {"location": cur_loc, "rotation": cur_rot}
        return pose_data

    # ------------------------------------------------------------------
    # Depth / sensor helpers
    # ------------------------------------------------------------------

    def wait_for_sensor_frame(
        self,
        world_frame_id: int,
        max_attempts: int = 200,
        sleep_interval: float = 0.01,
        log_failures: bool = True,
    ) -> bool:
        """Block until all cameras and depth sensors report *world_frame_id*.

        Args:
            world_frame_id: CARLA world frame number to wait for.
            max_attempts: Maximum polling iterations.
            sleep_interval: Sleep between polls (seconds).
            log_failures: Log per-sensor details on timeout.

        Returns:
            ``True`` if all sensors delivered the expected frame; ``False`` on
            timeout.
        """
        for _ in range(max_attempts):
            if self.sensors_have_frame(world_frame_id):
                return True
            time.sleep(sleep_interval)

        if log_failures:
            log_print(
                f"Warning: Frame timeout at world frame {world_frame_id}", "WARNING"
            )
            for cid, d in self.camera_data.items():
                if d is None:
                    log_print(f"  Camera {cid}: No data", "WARNING")
                else:
                    log_print(
                        f"  Camera {cid}: frame {d['frame']} (Δ={d['frame'] - world_frame_id})",
                        "WARNING",
                    )
        return False

    def build_depth_point_cloud(self, max_distance: float) -> np.ndarray:
        """Assemble a merged depth point cloud from all depth cameras.

        Args:
            max_distance: Discard points further than this distance (metres).

        Returns:
            Float32 array of shape ``(N, 3)`` in world coordinates.
        """
        if not self.depth_camera_data:
            return np.empty((0, 3), dtype=np.float32)

        point_sets = []
        for cam_id, entry in self.depth_camera_data.items():
            if entry is None:
                continue
            pts = self.depth_assembler.depth_image_to_world(
                cam_id, entry["image"], entry["transform"]
            )
            if pts.size:
                point_sets.append(pts)

        if not point_sets:
            return np.empty((0, 3), dtype=np.float32)

        cloud = np.vstack(point_sets)
        if self.ego_vehicle is None or max_distance <= 0:
            return cloud

        ego_loc = self.ego_vehicle.get_transform().location
        origin = np.array([ego_loc.x, ego_loc.y, ego_loc.z], dtype=np.float32)
        mask = np.linalg.norm(cloud - origin, axis=1) <= max_distance
        return cloud[mask] if np.any(mask) else np.empty((0, 3), dtype=np.float32)

    def build_camera_depth_data(self) -> Dict[str, Any]:
        """Decode depth images and bundle camera intrinsics.

        Returns:
            ``{camera_id: {depth_map, transform, fx, fy, cx, cy, width, height}}``
            ready for :meth:`~lib.visibility_utils.VisibilityCalculator.calculate_visibility`.
        """
        result: Dict[str, Any] = {}
        for cam_id, entry in self.depth_camera_data.items():
            if entry is None:
                continue
            intr = self.depth_assembler.get_camera_intrinsics(cam_id)
            result[cam_id] = {
                "depth_map": self.depth_assembler.decode_depth_map(entry["image"]),
                "transform": entry["transform"],
                **intr,
            }
        return result

    # ------------------------------------------------------------------
    # Debug visualisation
    # ------------------------------------------------------------------

    def draw_debug_bounding_boxes(self, builder: NuScenesDatasetBuilder) -> None:
        """Draw debug wireframe boxes around annotated objects in the CARLA viewport.

        No-op when ``self.debug`` is ``False`` or no world is loaded.

        Args:
            builder: Active :class:`~lib.nuscenes_builder.NuScenesDatasetBuilder`.
        """
        if not self.debug or not self.world:
            return

        max_dist = builder.annotation_builder.max_detection_distance
        objects = ObjectDetector.get_objects_of_interest(
            self.world, self.ego_vehicle, max_dist
        )
        for det in objects:
            try:
                actor = det.actor
                cat = det.category_token
                dist = det.distance
                color = CategoryColors.get_color(cat)

                if cat in (CAT_TRAFFIC_LIGHT, CAT_TRAFFIC_SIGN):
                    self._draw_traffic_infrastructure_bbox(actor, cat, color, dist)
                else:
                    bbox_info = BoundingBoxBuilder.get_actor_bbox_info(actor)
                    self._draw_bbox_wireframe(bbox_info.corners, color)
                    self._draw_bbox_label(
                        bbox_info.transform.location
                        + carla.Location(z=bbox_info.extent.z + 1.5),
                        f"{cat} ({dist:.1f}m)",
                        0.0,
                    )
            except Exception as exc:
                log_print(
                    f"Error drawing box for actor {det.actor.id}: {exc}", "DEBUG"
                )

        if objects:
            log_print(f"Drew {len(objects)} bounding boxes.", "DEBUG")

    def _draw_traffic_infrastructure_bbox(
        self,
        actor: carla.Actor,
        category_token: str,
        color: carla.Color,
        distance: float,
    ) -> None:
        """Draw a bounding box for a traffic-light or traffic-sign actor."""
        if isinstance(actor, LightHeadProxy):
            bbox_info = BoundingBoxBuilder.get_actor_bbox_info(actor, source="head_proxy")
            self.world.debug.draw_box(
                carla.BoundingBox(bbox_info.transform.location, bbox_info.extent),
                bbox_info.transform.rotation,
                thickness=0.3,
                color=color,
                life_time=1.0 / SENSOR_FPS * 2,
                persistent_lines=False,
            )
        elif getattr(actor, "source", None) == "env_sign":
            bbox_info = BoundingBoxBuilder.get_actor_bbox_info(actor, source="env_sign")
            self._draw_bbox_wireframe(bbox_info.corners, color)
        else:
            bbox_info = TrafficInfrastructureBBox.get_annotation_bbox_info(
                actor, category_token
            )
            self._draw_bbox_wireframe(bbox_info.corners, color)

        self._draw_bbox_label(
            bbox_info.transform.location,
            f"{category_token} ({distance:.1f}m)",
            bbox_info.extent.z + 0.3,
        )

    def _draw_bbox_label(
        self,
        location: carla.Location,
        text: str,
        z_offset: float,
    ) -> None:
        """Draw a text label at *location + z_offset*."""
        self.world.debug.draw_string(
            location=location + carla.Location(z=z_offset),
            text=text,
            draw_shadow=False,
            color=carla.Color(255, 255, 255),
            life_time=1.0 / SENSOR_FPS * 2,
        )

    def _draw_bbox_wireframe(
        self,
        world_corners: List[carla.Location],
        color: carla.Color,
    ) -> None:
        """Draw 12 wireframe edges connecting *world_corners*."""
        for start_idx, end_idx in BoundingBoxBuilder.get_bbox_edges():
            self.world.debug.draw_line(
                begin=world_corners[start_idx],
                end=world_corners[end_idx],
                thickness=0.3,
                color=color,
                life_time=1.0 / SENSOR_FPS * 2,
                persistent_lines=False,
            )

    # ------------------------------------------------------------------
    # Scene recording
    # ------------------------------------------------------------------

    def record_scene(
        self,
        output_dir: str,
        scene_id: str,
        rig_name: str,
        map_name: str,
        weather: str,
    ) -> None:
        """Record one scene loop: settle, then capture sensor data.

        All captured data is enqueued for threaded disk I/O.

        Args:
            output_dir: Root output directory.
            scene_id: Scene identifier string.
            rig_name: Camera rig name.
            map_name: CARLA map name.
            weather: Weather preset string.
        """
        scene_split = determine_scene_split(scene_id)
        builder_key = f"{rig_name}_{scene_split}"
        builder = self.nuscenes_builders.get(builder_key)
        if builder is None:
            log_print(f"Warning: No builder found for {builder_key}.", "WARNING")
            return

        scene_token = builder.create_scene(scene_id, map_name, weather)

        settle_frames = int(SETTLE_DURATION * SIMULATION_FPS)
        recording_frames = int(RECORDING_DURATION * SIMULATION_FPS)
        total_frames = settle_frames + recording_frames
        frames_per_capture = int(SIMULATION_FPS / SENSOR_FPS)
        expected_capture_count = recording_frames // frames_per_capture

        if self.trajectory_player is not None:
            self.trajectory_player.validate_frame_count(total_frames)

        log_print(
            f"Recording {RECORDING_DURATION}s ({settle_frames} settle + "
            f"{recording_frames} record frames, capture every {frames_per_capture} ticks)"
        )

        self.start_image_save_threads(num_threads=len(self.cameras))
        if self.lidar:
            self.start_lidar_save_threads(num_threads=1)
        if self.capture_depth_debug:
            self.start_depth_debug_threads(num_threads=max(1, len(self.cameras) // 2 or 1))

        self.frame_counter = 0
        self.sensor_tick_counter = 0
        self.recording_start_world_frame = None
        self.next_sensor_capture_frame = None
        self.captured_frames = []
        self.previous_ego_pose = None
        self.recording_start_time = time.time()
        self.lidar_data = None
        self.reset_capture_timings()
        for cam_id in self.last_captured_camera_frames:
            self.last_captured_camera_frames[cam_id] = -1
        for cam_id in self.last_captured_depth_frames:
            self.last_captured_depth_frames[cam_id] = -1

        # --- Settle period ---
        log_print(f"Settling for {SETTLE_DURATION}s...")
        last_world_frame_id = None
        for frame in range(settle_frames):
            if self.trajectory_player is not None:
                self.trajectory_player.apply_ego_frame(frame)
                self.trajectory_player.apply_frame(frame)
            last_world_frame_id = self.world.tick()
            self.update_spectator_follow_ego(last_world_frame_id)

        log_print("Settle complete — recording...")
        self.recording_start_world_frame = last_world_frame_id
        self.mark_sensor_data_captured()
        if self.recording_start_world_frame is None:
            raise RuntimeError(f"Scene {scene_id} never entered the settle loop.")

        # --- Recording loop ---
        for frame in range(recording_frames):
            abs_frame = settle_frames + frame
            if self.trajectory_player is not None:
                self.trajectory_player.apply_ego_frame(abs_frame)
                self.trajectory_player.apply_frame(abs_frame)

            tick_start = time.perf_counter()
            world_frame_id = self.world.tick()
            tick_ms = (time.perf_counter() - tick_start) * 1000.0
            self.frame_counter = frame
            self.update_spectator_follow_ego(world_frame_id)

            if self.next_sensor_capture_frame is None:
                current_sensor_frame = self.get_synchronized_sensor_frame(
                    require_fresh=True
                )
                if current_sensor_frame is None:
                    continue
                self.next_sensor_capture_frame = current_sensor_frame
            if self.next_sensor_capture_frame is None or world_frame_id < self.next_sensor_capture_frame:
                continue
            if world_frame_id > self.next_sensor_capture_frame:
                current_sensor_frame = self.get_synchronized_sensor_frame(
                    require_fresh=False
                )
                if (
                    current_sensor_frame is not None
                    and current_sensor_frame > self.next_sensor_capture_frame
                ):
                    log_print(
                        f"Warning: Missed sensor frame {self.next_sensor_capture_frame}; "
                        f"resyncing from sensor frame {current_sensor_frame} "
                        f"(world frame {world_frame_id}).",
                        "WARNING",
                    )
                    while self.next_sensor_capture_frame < current_sensor_frame:
                        self.next_sensor_capture_frame += frames_per_capture
                    if world_frame_id < self.next_sensor_capture_frame:
                        continue

            wait_start = time.perf_counter()
            sensors_ready = self.wait_for_synchronized_sensor_frame(
                self.next_sensor_capture_frame,
                log_failures=True,
            )
            wait_ms = (time.perf_counter() - wait_start) * 1000.0
            if not sensors_ready:
                current_sensor_frame = self.get_synchronized_sensor_frame(
                    require_fresh=False
                )
                raise RuntimeError(
                    "Timed out waiting for synchronized sensor frame "
                    f"{self.next_sensor_capture_frame} for scene {scene_id} "
                    f"(current={current_sensor_frame})."
                )

            if self.debug:
                self.draw_debug_bounding_boxes(builder)

            self.sensor_tick_counter += 1
            sensor_frame_id = self.next_sensor_capture_frame
            bundle_timestamp = self.get_sensor_bundle_timestamp(sensor_frame_id)
            current_timestamp = (
                bundle_timestamp
                if bundle_timestamp is not None
                else self.recording_start_time
                + (self.sensor_tick_counter - 1) * SENSOR_TICK_SECONDS
            )

            sample_token = builder.create_sample(current_timestamp, scene_token)
            ego_pose_token = builder.create_ego_pose(
                self.ego_vehicle,
                current_timestamp,
                trajectory_player=self.trajectory_player,
                trajectory_frame_idx=abs_frame,
            )
            builder.add_can_bus_data(
                self.ego_vehicle,
                current_timestamp,
                trajectory_player=self.trajectory_player,
                trajectory_frame_idx=abs_frame,
            )

            depth_start = time.perf_counter()
            camera_depth_data: Dict[str, Any] = {}
            camera_depth_data = self.build_camera_depth_data()
            depth_ms = (time.perf_counter() - depth_start) * 1000.0
            self.mark_sensor_data_captured()
            self.next_sensor_capture_frame = sensor_frame_id + frames_per_capture

            annotation_start = time.perf_counter()
            builder.create_annotations(
                self.world,
                sample_token,
                self.ego_vehicle,
                list(self.cameras),
                self.lidar_data,
                self.lidar,
                camera_depth_data=camera_depth_data,
                frame_number=self.sensor_tick_counter,
                trajectory_player=self.trajectory_player,
                trajectory_frame_idx=abs_frame,
            )
            annotation_ms = (time.perf_counter() - annotation_start) * 1000.0

            frame_data: Dict[str, Any] = {}
            enqueue_start = time.perf_counter()
            if sensors_ready:
                for cam_id, data in self.camera_data.items():
                    if data is None:
                        continue
                    filename = ImageProcessor.get_image_filename(
                        scene_id, self.sensor_tick_counter
                    )
                    relative_path = ImageProcessor.get_sample_data_filename(
                        cam_id, scene_id, self.sensor_tick_counter
                    )
                    image_path = builder.output_dir / "samples" / cam_id / filename
                    self.image_save_queue.put((data["image"], image_path))

                    sd_token = builder.create_sample_data(
                        cam_id, relative_path, sample_token, ego_pose_token, current_timestamp
                    )
                    frame_data[cam_id] = {
                        "sensor_tick_idx": self.sensor_tick_counter,
                        "global_frame": frame,
                        "timestamp": data["timestamp"],
                        "filename": filename,
                        "sample_data_token": sd_token,
                    }

                    if self.capture_depth_debug and cam_id in self.depth_camera_data:
                        depth_entry = self.depth_camera_data.get(cam_id)
                        if depth_entry is not None:
                            depth_rel = ImageProcessor.get_depth_debug_relative_path(
                                cam_id, scene_id, self.sensor_tick_counter
                            )
                            self.depth_debug_queue.put(
                                (depth_entry["image"], builder.output_dir / depth_rel)
                            )

                # LiDAR
                if self.lidar and self.lidar_data is not None:
                    ld = self.lidar_data
                    if abs(ld["frame"] - frame) <= frames_per_capture:
                        ts_us = int(current_timestamp * 1e6)
                        lidar_fn = (
                            f"n001-{scene_id}-{self.sensor_tick_counter:06d}"
                            f"__LIDAR_TOP__{ts_us}.pcd.bin"
                        )
                        lidar_rel = f"samples/LIDAR_TOP/{lidar_fn}"
                        lidar_path = builder.output_dir / "samples" / "LIDAR_TOP" / lidar_fn
                        self.lidar_save_queue.put((ld["measurement"], lidar_path))
                        lidar_sd_token = builder.create_sample_data(
                            "LIDAR_TOP", lidar_rel, sample_token, ego_pose_token, current_timestamp
                        )
                        frame_data["LIDAR_TOP"] = {
                            "sensor_tick_idx": self.sensor_tick_counter,
                            "global_frame": frame,
                            "timestamp": ld["timestamp"],
                            "filename": lidar_fn,
                            "sample_data_token": lidar_sd_token,
                        }
            enqueue_ms = (time.perf_counter() - enqueue_start) * 1000.0

            self.capture_timing_totals["tick_ms"] += tick_ms
            self.capture_timing_totals["wait_ms"] += wait_ms
            self.capture_timing_totals["depth_ms"] += depth_ms
            self.capture_timing_totals["annotation_ms"] += annotation_ms
            self.capture_timing_totals["enqueue_ms"] += enqueue_ms
            self.capture_timing_samples += 1

            self.captured_frames.append(frame_data)

            if self.sensor_tick_counter % 10 == 0:
                elapsed = self.sensor_tick_counter * (1.0 / SENSOR_FPS)
                log_print(
                    f"    {elapsed:.1f}s / {RECORDING_DURATION}s "
                    f"({elapsed / RECORDING_DURATION * 100:.1f}%) "
                    f"— {self.sensor_tick_counter} captures"
                )
                self.log_capture_timing_summary()

        # Wait for image queue to drain.
        deadline = time.time() + 30
        while not self.image_save_queue.empty() and time.time() < deadline:
            time.sleep(0.1)
        if not self.image_save_queue.empty():
            log_print("Warning: Image save timeout (30s).", "WARNING")

        self.stop_image_save_threads()
        if self.lidar:
            self.stop_lidar_save_threads()
        if self.capture_depth_debug:
            self.stop_depth_debug_threads()

        if self.sensor_tick_counter != expected_capture_count:
            raise RuntimeError(
                f"Scene {scene_id} completed with {self.sensor_tick_counter}/"
                f"{expected_capture_count} captures."
            )

        log_print(
            f"    Completed: {self.sensor_tick_counter} captures for scene {scene_id}."
        )
        time.sleep(1.0)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Destroy all sensors and the ego vehicle; restore world settings."""
        log_print("Cleanup starting...")
        try:
            self.stop_image_save_threads()
            if self.lidar:
                self.stop_lidar_save_threads()
            self.stop_depth_debug_threads()

            self.camera_data.clear()
            self.depth_camera_data.clear()
            self.lidar_data = None
            if hasattr(self, "depth_assembler") and self.depth_assembler:
                self.depth_assembler._camera_models.clear()

            def _stop(actors, label):
                log_print(f"Stopping {len(actors)} {label}...")
                for i, actor in enumerate(actors):
                    if actor is None:
                        continue
                    try:
                        actor.stop()
                    except RuntimeError as exc:
                        if "attempting to use a destroyed" not in str(exc):
                            log_print(f"Warning: stop {label} {i}: {exc}", "WARNING")
                    except Exception as exc:
                        log_print(f"Warning: stop {label} {i}: {exc}", "WARNING")

            def _destroy(actors, label):
                log_print(f"Destroying {len(actors)} {label}...")
                destroyed = 0
                for i, actor in enumerate(actors):
                    if actor is None:
                        continue
                    try:
                        _ = actor.type_id  # existence check
                        actor.destroy()
                        destroyed += 1
                        if self.world is not None:
                            try:
                                self.world.tick()
                            except Exception:
                                pass
                    except RuntimeError as exc:
                        if "attempting to use a destroyed" not in str(exc):
                            log_print(f"Warning: destroy {label} {i}: {exc}", "WARNING")
                    except Exception as exc:
                        log_print(f"Warning: destroy {label} {i}: {exc}", "WARNING")
                log_print(f"Destroyed {destroyed}/{len(actors)} {label}.")
                return destroyed

            _stop(self.cameras, "cameras")
            _stop(self.depth_cameras, "depth cameras")

            if self.world is not None:
                for _ in range(5):
                    try:
                        self.world.tick()
                    except Exception:
                        break

            _destroy(self.cameras + self.depth_cameras, "cameras/depth-cameras")

            if self.world is not None:
                for _ in range(5):
                    try:
                        self.world.tick()
                    except Exception:
                        break

            if self.lidar is not None:
                try:
                    self.lidar.stop()
                    _ = self.lidar.type_id
                    self.lidar.destroy()
                    log_print("Lidar destroyed.")
                except Exception as exc:
                    log_print(f"Warning: destroy lidar: {exc}", "WARNING")

            if self.ego_vehicle is not None:
                try:
                    _ = self.ego_vehicle.type_id
                    self.ego_vehicle.destroy()
                    log_print("Ego vehicle destroyed.")
                except Exception as exc:
                    log_print(f"Warning: destroy ego: {exc}", "WARNING")

        except Exception as exc:
            log_print(f"Error during cleanup: {exc}", "ERROR")
        finally:
            self.cameras = []
            self.camera_data = {}
            self.depth_cameras = []
            self.depth_camera_data = {}
            self.depth_camera_params = {}
            self.last_captured_camera_frames = {}
            self.last_captured_depth_frames = {}
            self.ego_vehicle = None
            self.frame_counter = 0
            self.captured_frames = []
            self.sensor_tick_counter = 0
            self.recording_start_world_frame = None
            self.next_sensor_capture_frame = None
            self.last_spectator_update_world_frame = None
            self.previous_ego_pose = None
            self.recording_start_time = None

            if self.world is not None:
                for _ in range(5):
                    try:
                        self.world.tick()
                    except Exception:
                        break

            try:
                self.disable_synchronous_mode()
            except Exception as exc:
                log_print(f"Warning: disable sync mode: {exc}", "WARNING")

            gc.collect()
            log_print("Cleanup complete.")

    # ------------------------------------------------------------------
    # Scene execution
    # ------------------------------------------------------------------

    def execute_scene(
        self,
        scene: Dict[str, Any],
        camera_rig: Dict[str, Any],
        output_dir: str,
        rig_name: str,
        trajectories_dir: str,
    ) -> None:
        """Execute one scene end-to-end for a given camera rig.

        Loads the map (if changed), enables synchronous mode, spawns the ego
        and traffic actors, runs the recording loop, saves annotation data.

        Args:
            scene: Scene config dict from ``scenes.json``.
            camera_rig: Camera rig config dict.
            output_dir: Root output directory.
            rig_name: Camera rig name.
            trajectories_dir: Directory containing trajectory HDF5 files.
        """
        scene_id = scene["id"]
        map_name = scene["map"]
        weather = scene["weather"]
        spawn_point = scene["spawn_point"]

        log_print(
            f"Executing {scene_id}  map={map_name}  weather={weather}  "
            f"rig={rig_name}  sp={spawn_point}"
        )
        self.current_scene_id = scene_id

        try:
            log_print(f"Loading trajectory data for {scene_id}...")
            self.trajectory_player = TrajectoryPlayer(trajectories_dir, scene_id)
            self.trajectory_player.validate_scene_config(scene)

            current_map = self.world.get_map().name if self.world else ""
            if not current_map.endswith(map_name):
                log_print(f"Loading map {map_name} (current: {current_map})...")
                self.world = self.client.load_world(map_name)
            else:
                log_print(f"Map {map_name} already loaded.")

            for _ in range(5):
                self.world.tick()

            self.enable_synchronous_mode()
            self.set_weather(weather)

            spawn_tf = self.spawn_ego_vehicle(camera_rig["vehicle_type"], spawn_point)
            self.trajectory_player.set_ego_vehicle(self.ego_vehicle)
            self.ego_vehicle.set_simulate_physics(False)

            spectator = self.world.get_spectator()
            spectator.set_transform(
                carla.Transform(
                    spawn_tf.location + carla.Location(z=50),
                    carla.Rotation(pitch=-90),
                )
            )

            self.setup_cameras(camera_rig["cameras"])
            if "lidar" in camera_rig and not self.no_lidar:
                self.setup_lidar(camera_rig["lidar"])

            self.spawn_traffic_from_trajectories(self.trajectory_player)
            self.record_scene(output_dir, scene_id, rig_name, map_name, weather)

            # Destroy pedestrians and traffic vehicles.
            for actor_list, label in (
                (self.walkers, "pedestrians"),
                (self.traffic_vehicles, "traffic vehicles"),
            ):
                destroyed = 0
                for actor in actor_list:
                    if actor is not None:
                        try:
                            actor.destroy()
                            destroyed += 1
                        except Exception:
                            pass
                log_print(f"Destroyed {destroyed}/{len(actor_list)} {label}.")
                actor_list.clear()

            if self.world is not None:
                for _ in range(5):
                    try:
                        self.world.tick()
                    except Exception:
                        break

        except Exception as exc:
            log_print(f"Error executing scene {scene_id}: {exc}", "ERROR")
            raise
        finally:
            self.trajectory_player = None
            self.cleanup()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments and execute all requested scenes."""
    parser = argparse.ArgumentParser(
        description="Execute CARLA scenes with camera rigs in nuScenes format."
    )
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--scenes", required=True, help="Path to scenes.json")
    parser.add_argument(
        "--camera-rigs", required=True, nargs="+",
        help="Paths to camera rig JSON files (e.g. configs/rigs/R1.json)",
    )
    parser.add_argument(
        "--trajectories", required=True,
        help="Directory containing trajectory HDF5 files (by split)",
    )
    parser.add_argument(
        "--split", choices=["trainval", "test", "mini"],
        help="Execute only scenes from this split (optional)",
    )
    parser.add_argument(
        "--version", default="v1.0",
        help="Dataset version tag (default: v1.0)",
    )
    parser.add_argument("--limit", type=int, help="Maximum scenes to execute")
    parser.add_argument("--host", default="localhost", help="CARLA host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument(
        "--max-detection-distance", type=float, default=80.0,
        help="Detection radius for annotations in metres (default: 80.0)",
    )
    parser.add_argument("--debug", action="store_true", help="Show debug bboxes")
    parser.add_argument(
        "--no-lidar", action="store_true",
        help="Disable LiDAR (annotations will have num_lidar_pts=0)",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    setup_logging(args.output, debug=args.debug)

    runner = SceneRunner(
        args.host,
        args.port,
        debug=args.debug,
        no_lidar=args.no_lidar,
        max_detection_distance=args.max_detection_distance,
    )

    if args.no_lidar:
        log_print("LiDAR disabled — num_lidar_pts will be 0.")

    # Load rigs.
    camera_rigs: Dict[str, Dict] = {}
    for rig_path in args.camera_rigs:
        rig_name = Path(rig_path).stem
        camera_rig = runner.load_camera_rig(rig_path)
        camera_rigs[rig_name] = camera_rig
        log_print(
            f"Rig {rig_name}: {camera_rig['vehicle_type']}, "
            f"{len(camera_rig['cameras'])} cameras"
            + (f", lidar={camera_rig['lidar']['id']}" if "lidar" in camera_rig else "")
        )

    # Load scenes.
    scenes_config = runner.load_scenes(args.scenes)
    if args.split:
        if args.split == "trainval":
            scenes_to_run = (
                scenes_config["scenes"].get("train", [])
                + scenes_config["scenes"].get("val", [])
            )
        else:
            scenes_to_run = scenes_config["scenes"].get(args.split, [])
    else:
        scenes_to_run = []
        for ss in scenes_config["scenes"].values():
            scenes_to_run.extend(ss)

    if args.limit and args.limit < len(scenes_to_run):
        scenes_to_run = scenes_to_run[: args.limit]
    log_print(f"Scenes to execute: {len(scenes_to_run)}")

    # Validate trajectory files.
    scene_ids = [s["id"] for s in scenes_to_run]
    missing = validate_trajectories_exist(args.trajectories, scene_ids)
    if missing:
        log_print(
            f"ERROR: Missing trajectories for {len(missing)} scenes. "
            "Run core/record_trajectories.py first.",
            "ERROR",
        )
        for sid in missing[:10]:
            log_print(f"  - {sid}", "ERROR")
        if len(missing) > 10:
            log_print(f"  ... and {len(missing) - 10} more.", "ERROR")
        sys.exit(1)

    # Sort by map name to minimise CARLA map reloads.
    scenes_to_run.sort(key=lambda s: (s.get("map", ""), s.get("id", "")))

    # Resume support.
    resume_path = Path(args.output) / "resume_state.json"
    completed_rigs: List[str] = []
    if resume_path.exists():
        try:
            with open(resume_path) as f:
                completed_rigs = json.load(f).get("completed_rigs", [])
            log_print(f"Resuming: {len(completed_rigs)} rigs already done.")
        except Exception as exc:
            log_print(f"Warning: Could not load resume state: {exc}", "WARNING")

    # Group by split for dataset builder setup.
    scenes_by_split: Dict[str, List] = {}
    for scene in scenes_to_run:
        split = determine_scene_split(scene["id"])
        scenes_by_split.setdefault(split, []).append(scene)

    for rig_name, camera_rig in camera_rigs.items():
        for split in scenes_by_split:
            runner.setup_nuscenes_dataset(
                rig_name, camera_rig, args.output, split,
                args.max_detection_distance, args.version,
            )

    total = len(scenes_to_run) * len(camera_rigs)
    log_print(f"Total executions: {total}  ({len(scenes_to_run)} scenes × {len(camera_rigs)} rigs)")

    exec_count = 0
    this_run = 0
    start_time = time.time()

    for rig_name, camera_rig in camera_rigs.items():
        if rig_name in completed_rigs:
            log_print(f"Skipping completed rig: {rig_name}")
            exec_count += len(scenes_to_run)
            continue

        log_print("=" * 60)
        log_print(f"Rig: {rig_name}")
        log_print("=" * 60)

        for i, scene in enumerate(scenes_to_run):
            exec_count += 1
            this_run += 1
            elapsed = time.time() - start_time
            pct = exec_count / total * 100
            eta = (elapsed / this_run * (total - exec_count)) if this_run > 1 else 0
            log_print(
                f"[{exec_count}/{total} {pct:.1f}%] Rig {rig_name} "
                f"scene {i+1}/{len(scenes_to_run)} — elapsed {elapsed/60:.1f}min "
                f"ETA {eta/60:.1f}min"
            )
            runner.execute_scene(scene, camera_rig, args.output, rig_name, args.trajectories)

        # Save annotations for this rig.
        for split in scenes_by_split:
            builder_key = f"{rig_name}_{split}"
            builder = runner.nuscenes_builders.get(builder_key)
            if builder and len(builder.scenes) > 0:
                log_print(f"Saving annotations: {rig_name} / {split}")
                if builder.save_annotations():
                    info = builder.get_dataset_info()
                    log_print(
                        f"Saved {info['num_scenes']} scenes / "
                        f"{info['num_samples']} samples."
                    )
                else:
                    log_print(f"Error saving {rig_name}/{split}", "ERROR")
            if builder_key in runner.nuscenes_builders:
                del runner.nuscenes_builders[builder_key]

        completed_rigs.append(rig_name)
        try:
            with open(resume_path, "w") as f:
                json.dump({"completed_rigs": completed_rigs}, f)
        except Exception as exc:
            log_print(f"Warning: could not save resume state: {exc}", "WARNING")

    total_time = time.time() - start_time
    log_print("=" * 60)
    log_print("ALL DONE")
    log_print("=" * 60)
    log_print(f"Output: {args.output}")
    log_print(f"Total executions: {exec_count} ({this_run} this run)")
    if this_run > 0:
        log_print(
            f"Time: {total_time/60:.1f}min  Avg per scene: {total_time/this_run:.1f}s"
        )


if __name__ == "__main__":
    main()
