#!/usr/bin/env python3
"""
Interactive CARLA annotation validation tool for live autopilot scenes.

Modes of operation
------------------
**Live autopilot**  (default)
    Traffic and pedestrians are spawned fresh using the configured density and
    directed by CARLA's Traffic Manager.  Use this when you want to explore
    arbitrary traffic conditions interactively.

**Trajectory replay**  (``--trajectory-file``)
    Traffic and pedestrians follow HDF5 trajectories pre-recorded by
    ``core/record_trajectories.py``.  The ego vehicle uses autopilot while
    all other actors are puppet-controlled for deterministic replay.

Controls (pygame HUD)
---------------------
    SPACE   — pause / resume simulation
    C       — capture the current frame to disk
    N       — single-step one tick while paused
    B       — toggle CARLA debug bounding boxes
    H       — toggle help text
    ESC     — quit

Usage::

    python debug/annotation_debug_tool.py \\
        --map Town01 --spawn-point 86 \\
        --camera-rig configs/rigs/R1.json \\
        --output output/debug

    # With deterministic trajectory replay:
    python debug/annotation_debug_tool.py \\
        --map Town01 --spawn-point 86 \\
        --camera-rig configs/rigs/R1.json \\
        --trajectory-file output/trajectories/Town01_sp86.h5 \\
        --output output/debug
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import carla
import numpy as np
import pygame

# Ensure project root is importable when run directly.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib.annotation_projection import AnnotationProjector
from lib.debug_scene_utils import (
    WalkerSpawnResult,
    configure_traffic_manager,
    destroy_actors,
    enable_autopilot,
    spawn_pedestrians,
    spawn_traffic,
    stop_and_destroy_walkers,
)
from lib.utils.image_utils import ImageProcessor
from lib.utils.logging_utils import log_print, setup_logging
from lib.trajectory_utils import TrajectoryPlayer
from core.scene_runner import SIMULATION_FPS, SceneRunner

HUD_WIDTH = 980
HUD_HEIGHT = 320
HUD_FPS = 30
DEFAULT_TM_PORT = 8000
DEFAULT_TRAFFIC_DENSITY = 0.3
DEFAULT_MAX_PEDESTRIANS = 50
DEFAULT_VERSION = "v1.0-debug"
DEFAULT_SPLIT = "trainval"
DEFAULT_WARMUP_TICKS = 15


class AnnotationDebugTool:
    """Run a live CARLA scene and capture debug-friendly annotation artefacts.

    Attributes:
        args: Parsed CLI arguments.
        runner: :class:`~core.scene_runner.SceneRunner` instance.
        traffic_manager: CARLA TrafficManager, or ``None`` before setup.
        builder: Active nuScenes dataset builder, or ``None`` before setup.
        scene_token: Token of the current scene record.
        session_root: Directory for all output from this session.
        traffic_vehicles: Spawned traffic-vehicle actors (live mode).
        walker_result: Spawned walker state (live mode).
        trajectory_player: Loaded trajectory player (replay mode), or
            ``None`` in live mode.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialise the debug tool, connect to CARLA, and start pygame.

        Args:
            args: Parsed command-line arguments from :func:`build_arg_parser`.
        """
        self.args = args
        self.runner = SceneRunner(
            host=args.host,
            port=args.port,
            timeout=args.timeout,
            debug=args.show_debug_boxes,
            no_lidar=args.no_lidar,
            max_detection_distance=args.max_detection_distance,
        )
        self.traffic_manager: Optional[carla.TrafficManager] = None
        self.tm_port = args.tm_port
        self.builder = None
        self.scene_token: Optional[str] = None
        self.scene_id: Optional[str] = None
        self.session_root: Optional[Path] = None
        self.traffic_vehicles: List[carla.Actor] = []
        self.walker_result = WalkerSpawnResult()
        self.paused = False
        self.show_debug_boxes = args.show_debug_boxes
        self.capture_requested = False
        self.single_step_requested = False
        self.show_help = True
        self.capture_count = 0
        self.last_world_frame: Optional[int] = None
        self.last_timestamp_s: float = 0.0
        self.last_capture_id: Optional[str] = None

        # Trajectory replay state — only used when --trajectory-file is given.
        self.trajectory_player: Optional[TrajectoryPlayer] = None
        self._traj_frame: int = 0

        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((HUD_WIDTH, HUD_HEIGHT))
        pygame.display.set_caption("Annotation Debug Tool")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 18)
        self.small_font = pygame.font.SysFont("monospace", 15)

    # ------------------------------------------------------------------
    # Session setup
    # ------------------------------------------------------------------

    def _build_session_paths(self) -> Tuple[Path, str]:
        """Create a dedicated debug output directory and a deterministic scene ID.

        Returns:
            ``(session_root, scene_id)`` — the output directory and the scene
            identifier string.
        """
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        rig_name = Path(self.args.camera_rig).stem
        scene_id = (
            self.args.scene_id
            or f"debug_{self.args.map}_sp{self.args.spawn_point}_{ts}"
        )
        root = Path(self.args.output) / f"{scene_id}_{rig_name}"
        root.mkdir(parents=True, exist_ok=True)
        return root, scene_id

    def setup(self) -> None:
        """Load the map, spawn actors, and prepare the interactive session."""
        self.session_root, self.scene_id = self._build_session_paths()
        setup_logging(str(self.session_root), debug=True)
        log_print(f"Debug session output: {self.session_root}")

        rig_config = self.runner.load_camera_rig(self.args.camera_rig)
        rig_name = Path(self.args.camera_rig).stem

        current_map = self.runner.world.get_map().name if self.runner.world else ""
        if not current_map.endswith(self.args.map):
            log_print(f"Loading map {self.args.map}...")
            self.runner.world = self.runner.client.load_world(self.args.map)
        else:
            log_print(f"Map {self.args.map} already loaded.")

        for _ in range(5):
            self.runner.world.tick()

        self.runner.enable_synchronous_mode()
        self.traffic_manager = configure_traffic_manager(
            self.runner.client,
            port=self.tm_port,
            seed=self.args.seed,
            synchronous_mode=True,
        )
        self.runner.set_weather(self.args.weather)

        self.builder = self.runner.setup_nuscenes_dataset(
            rig_name=rig_name,
            sensor_configs=rig_config,
            output_dir=str(self.session_root),
            split=self.args.split,
            max_detection_distance=self.args.max_detection_distance,
            version=self.args.version,
        )
        self.scene_token = self.builder.create_scene(
            self.scene_id,
            self.args.map,
            self.args.weather,
            description="Interactive live annotation validation session",
        )

        spawn_tf = self.runner.spawn_ego_vehicle(
            rig_config["vehicle_type"], self.args.spawn_point
        )
        self._position_spectator_topdown(spawn_tf)
        self.runner.setup_cameras(rig_config["cameras"])
        if "lidar" in rig_config and not self.args.no_lidar:
            self.runner.setup_lidar(rig_config["lidar"])

        if self.args.trajectory_file:
            self._setup_trajectory_mode()
        else:
            self._setup_live_mode(spawn_tf)

        for _ in range(self.args.warmup_ticks):
            self._tick_world(capture_if_requested=False)

    def _setup_trajectory_mode(self) -> None:
        """Load trajectory data and spawn puppet actors for replay mode."""
        log_print(f"Loading trajectory from {self.args.trajectory_file}")
        self.trajectory_player = TrajectoryPlayer.from_file(self.args.trajectory_file)
        self.runner.spawn_traffic_from_trajectories(self.trajectory_player)
        log_print(
            f"Trajectory: {self.trajectory_player.metadata['total_frames']} frames, "
            f"{self.trajectory_player.traffic_count} vehicles, "
            f"{self.trajectory_player.pedestrian_count} pedestrians"
        )
        self.trajectory_player.apply_frame(0)
        enable_autopilot(self.runner.ego_vehicle, [], self.tm_port)
        log_print("Trajectory mode: ego on autopilot, traffic/peds follow trajectories.")

    def _setup_live_mode(self, spawn_transform: carla.Transform) -> None:
        """Spawn live-autopilot traffic and pedestrians."""
        vehicles, _, _ = spawn_traffic(
            self.runner.world,
            self.args.traffic_density,
            spawn_transform,
            seed=self.args.seed,
        )
        self.traffic_vehicles = vehicles
        log_print(f"Spawned {len(self.traffic_vehicles)} traffic vehicles.")

        self.walker_result = spawn_pedestrians(
            self.runner.client,
            self.runner.world,
            max_pedestrians=self.args.max_pedestrians,
            seed=self.args.seed,
        )
        log_print(
            f"Spawned {len(self.walker_result.walkers)} pedestrians "
            f"({len(self.walker_result.controllers)} controllers)."
        )
        enable_autopilot(self.runner.ego_vehicle, self.traffic_vehicles, self.tm_port)
        log_print("Autopilot enabled for ego and traffic vehicles.")

    def _position_spectator_topdown(self, spawn_transform: carla.Transform) -> None:
        """Place the CARLA spectator above *spawn_transform* at session start."""
        self.runner.world.get_spectator().set_transform(
            carla.Transform(
                spawn_transform.location + carla.Location(z=35.0),
                carla.Rotation(pitch=-90.0),
            )
        )

    # ------------------------------------------------------------------
    # Simulation tick & capture
    # ------------------------------------------------------------------

    def _tick_world(self, capture_if_requested: bool = True) -> Optional[int]:
        """Advance the simulation one synchronous tick.

        Args:
            capture_if_requested: When ``True`` and :attr:`capture_requested`
                is set, flush the request and call
                :meth:`_capture_current_frame`.

        Returns:
            CARLA world-frame ID, or ``None`` on error.
        """
        if self.trajectory_player is not None:
            self.trajectory_player.apply_frame(self._traj_frame)
            total = self.trajectory_player.metadata["total_frames"]
            if self._traj_frame < total - 1:
                self._traj_frame += 1

        world_frame = self.runner.world.tick()
        self.last_world_frame = world_frame
        snapshot = self.runner.world.get_snapshot()
        self.last_timestamp_s = snapshot.timestamp.elapsed_seconds
        self.runner.update_spectator_follow_ego()
        self.runner.debug = self.show_debug_boxes
        if self.show_debug_boxes:
            self.runner.draw_debug_bounding_boxes(self.builder)

        if capture_if_requested and self.capture_requested:
            self.capture_requested = False
            self._capture_current_frame(world_frame, snapshot)

        return world_frame

    def _capture_current_frame(
        self, world_frame: int, snapshot: carla.WorldSnapshot
    ) -> bool:
        """Capture raw sensors, overlays, and nuScenes records for one frame.

        Args:
            world_frame: Current CARLA world-frame ID.
            snapshot: Current CARLA world snapshot.

        Returns:
            ``True`` on success, ``False`` if sensors were not synchronised.
        """
        if self.builder is None or self.scene_token is None or self.session_root is None:
            return False

        sensors_ready = self.runner.wait_for_sensor_frame(world_frame)
        if not sensors_ready:
            log_print(
                f"Skipping capture at frame {world_frame}: sensors not synced.",
                "WARNING",
            )
            return False

        capture_id = f"{int(snapshot.timestamp.elapsed_seconds * 1000):013d}"
        ts = snapshot.timestamp.elapsed_seconds

        sample_token = self.builder.create_sample(ts, self.scene_token)
        trajectory_frame_idx = (self._traj_frame - 1) if self.trajectory_player is not None else None
        ego_pose_token = self.builder.create_ego_pose(
            self.runner.ego_vehicle,
            ts,
            trajectory_player=self.trajectory_player,
            trajectory_frame_idx=trajectory_frame_idx,
        )
        self.builder.add_can_bus_data(
            self.runner.ego_vehicle,
            ts,
            trajectory_player=self.trajectory_player,
            trajectory_frame_idx=trajectory_frame_idx,
        )

        camera_depth_data = self.runner.build_camera_depth_data()
        self.builder.create_annotations(
            self.runner.world,
            sample_token,
            self.runner.ego_vehicle,
            list(self.runner.cameras),
            self.runner.lidar_data,
            self.runner.lidar,
            camera_depth_data=camera_depth_data,
            frame_number=self.capture_count + 1,
            trajectory_player=self.trajectory_player,
            trajectory_frame_idx=(self._traj_frame - 1) if self.trajectory_player is not None else None,
        )

        ego_pose_entry = self.builder.get_ego_pose(ego_pose_token)
        annotations = self.builder.get_sample_annotations(sample_token)
        instances = self.builder.get_instances_by_tokens(
            [a["instance_token"] for a in annotations]
        )
        instance_by_token = {inst["token"]: inst for inst in instances}

        saved_camera_files: List[Dict[str, Any]] = []
        for camera_id, camera_entry in self.runner.camera_data.items():
            if camera_entry is None or camera_entry["frame"] != world_frame:
                continue

            filename = ImageProcessor.get_capture_filename(capture_id, camera_id, extension="jpg")
            relative_path = f"samples/{camera_id}/{filename}"
            raw_path = self.builder.output_dir / relative_path
            ImageProcessor.carla_image_to_jpg(camera_entry["image"], raw_path, quality=95)

            sd_token = self.builder.create_sample_data(
                camera_id, relative_path, sample_token, ego_pose_token, ts
            )
            calibrated_sensor = self.builder.get_calibrated_sensor(camera_id)
            overlay_img = AnnotationProjector.render_carla_capture(
                camera_entry["image"],
                annotations,
                ego_pose_entry,
                calibrated_sensor,
                instance_by_token=instance_by_token,
            )
            overlay_path = self.builder.output_dir / "overlays" / camera_id / filename
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            overlay_img.save(overlay_path, format="JPEG", quality=95, optimize=True)

            depth_entry = self.runner.depth_camera_data.get(camera_id)
            depth_rel_path: Optional[str] = None
            if depth_entry is not None and depth_entry["frame"] == world_frame:
                depth_fn = ImageProcessor.get_capture_filename(capture_id, camera_id, extension="png")
                depth_rel_path = f"debug_depth/{camera_id}/{depth_fn}"
                depth_path = self.builder.output_dir / depth_rel_path
                ImageProcessor.carla_depth_to_png(depth_entry["image"], depth_path)

            saved_camera_files.append({
                "camera_id": camera_id,
                "sample_data_token": sd_token,
                "raw_file": relative_path,
                "overlay_file": str(overlay_path.relative_to(self.builder.output_dir)),
                "depth_file": depth_rel_path,
            })

        lidar_file_record: Optional[Dict[str, Any]] = None
        lidar_entry = self.runner.lidar_data
        if self.runner.lidar and lidar_entry is not None:
            mf = lidar_entry.get("measurement_frame")
            if mf is None or mf == world_frame:
                lidar_fn = ImageProcessor.get_capture_filename(capture_id, "LIDAR_TOP", extension="pcd.bin")
                lidar_rel = f"samples/LIDAR_TOP/{lidar_fn}"
                lidar_path = self.builder.output_dir / lidar_rel
                _save_lidar_measurement(lidar_entry["measurement"], lidar_path)
                lidar_sd_token = self.builder.create_sample_data(
                    "LIDAR_TOP", lidar_rel, sample_token, ego_pose_token, ts
                )
                lidar_file_record = {
                    "sample_data_token": lidar_sd_token,
                    "raw_file": lidar_rel,
                    "measurement_frame": mf,
                }

        capture_record: Dict[str, Any] = {
            "capture_id": capture_id,
            "scene_id": self.scene_id,
            "map": self.args.map,
            "weather": self.args.weather,
            "spawn_point": self.args.spawn_point,
            "traffic_density": self.args.traffic_density,
            "world_frame": world_frame,
            "timestamp_seconds": ts,
            "paused": self.paused,
            "debug_boxes_enabled": self.show_debug_boxes,
            "sample": self.builder.get_sample(sample_token),
            "ego_pose": ego_pose_entry,
            "sample_data": self.builder.get_sample_data_for_sample(sample_token),
            "sample_annotations": annotations,
            "instances": instances,
            "sensors": [
                self.builder.get_sensor(cam["camera_id"]) for cam in saved_camera_files
            ],
            "calibrated_sensors": [
                self.builder.get_calibrated_sensor(cam["camera_id"])
                for cam in saved_camera_files
            ],
            "camera_outputs": saved_camera_files,
            "lidar_output": lidar_file_record,
        }

        rec_path = self.builder.output_dir / "captures" / f"{capture_id}.json"
        rec_path.parent.mkdir(parents=True, exist_ok=True)
        with open(rec_path, "w", encoding="utf-8") as f:
            json.dump(capture_record, f, indent=2)

        self.builder.save_annotations()
        self.capture_count += 1
        self.last_capture_id = capture_id
        log_print(
            f"Captured frame {world_frame} at {ts:.3f}s → "
            f"{rec_path.relative_to(self.builder.output_dir)}"
        )
        return True

    # ------------------------------------------------------------------
    # HUD / input
    # ------------------------------------------------------------------

    def _render_hud(self) -> None:
        """Draw the compact text HUD showing controls and current state."""
        self.screen.fill((20, 20, 20))
        mode = "TRAJECTORY" if self.trajectory_player is not None else "LIVE AUTOPILOT"
        traj_info = ""
        if self.trajectory_player is not None:
            total = self.trajectory_player.metadata["total_frames"]
            end = " [END]" if self._traj_frame >= total - 1 else ""
            traj_info = f"   Traj: {self._traj_frame}/{total}{end}"

        status_lines = [
            f"Scene: {self.scene_id}",
            f"Map: {self.args.map}   Weather: {self.args.weather}   "
            f"Spawn: {self.args.spawn_point}",
            f"Rig: {Path(self.args.camera_rig).stem}   Mode: {mode}{traj_info}",
            f"State: {'PAUSED' if self.paused else 'RUNNING'}   "
            f"Debug boxes: {'ON' if self.show_debug_boxes else 'OFF'}",
            f"Frame: {self.last_world_frame}   "
            f"Time: {self.last_timestamp_s:.3f}s   "
            f"Captures: {self.capture_count}",
            f"Last capture: {self.last_capture_id or '-'}",
        ]
        help_lines = [
            "Controls:",
            "  SPACE  pause / resume",
            "  C      capture current frame",
            "  N      single-step (when paused)",
            "  B      toggle debug bounding boxes",
            "  H      toggle this help",
            "  ESC    quit",
        ]
        lines = status_lines + (help_lines if self.show_help else [])

        for i, line in enumerate(lines):
            surface = (self.font if i < 6 else self.small_font).render(
                line, True, (220, 220, 220)
            )
            self.screen.blit(surface, (16, 16 + i * 26))
        pygame.display.flip()

    def _handle_events(self) -> bool:
        """Process pygame events, updating session state flags.

        Returns:
            ``False`` when the user requests to quit; ``True`` otherwise.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type != pygame.KEYDOWN:
                continue
            if event.key == pygame.K_ESCAPE:
                return False
            elif event.key == pygame.K_SPACE:
                self.paused = not self.paused
                log_print(f"Simulation {'paused' if self.paused else 'resumed'}.")
            elif event.key == pygame.K_c:
                self.capture_requested = True
                log_print("Capture requested.")
            elif event.key == pygame.K_n and self.paused:
                self.single_step_requested = True
            elif event.key == pygame.K_b:
                self.show_debug_boxes = not self.show_debug_boxes
                log_print(
                    f"Debug boxes {'enabled' if self.show_debug_boxes else 'disabled'}."
                )
            elif event.key == pygame.K_h:
                self.show_help = not self.show_help
        return True

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute the interactive pause / capture loop until the user quits."""
        self.setup()
        log_print("Annotation debug tool ready.")

        running = True
        while running:
            running = self._handle_events()
            if self.paused:
                if self.single_step_requested or self.capture_requested:
                    self._tick_world(capture_if_requested=True)
                    self.single_step_requested = False
                else:
                    time.sleep(0.01)
            else:
                self._tick_world(capture_if_requested=True)
            self._render_hud()
            self.clock.tick(HUD_FPS)

    def cleanup(self) -> None:
        """Save annotation tables and destroy all spawned actors."""
        try:
            if self.builder is not None:
                self.builder.save_annotations()
        except Exception as exc:
            logging.warning(f"Annotation save failed during cleanup: {exc}")

        try:
            stop_and_destroy_walkers(self.walker_result)
        finally:
            destroy_actors(self.traffic_vehicles)

        if self.traffic_manager is not None:
            try:
                self.traffic_manager.set_synchronous_mode(False)
            except Exception:
                pass

        try:
            self.runner.cleanup()
        finally:
            pygame.quit()


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _save_lidar_measurement(
    lidar_measurement: carla.LidarMeasurement,
    output_path: Path,
) -> None:
    """Serialise a CARLA LiDAR measurement to a nuScenes-compatible binary.

    Columns written: ``[x, y, z, intensity, ring_index]`` (float32).

    Args:
        lidar_measurement: Raw CARLA LiDAR measurement.
        output_path: Destination ``.pcd.bin`` file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw = np.frombuffer(lidar_measurement.raw_data, dtype=np.float32)
    pts = raw.reshape(-1, 4).copy()
    pts[:, 1] = -pts[:, 1]  # CARLA Y is right; nuScenes Y is left.
    out = np.zeros((pts.shape[0], 5), dtype=np.float32)
    out[:, :4] = pts
    out.tofile(output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for the debug tool."""
    p = argparse.ArgumentParser(
        description="Interactive annotation validation tool for live CARLA scenes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--map", required=True, help="CARLA map (e.g. Town01)")
    p.add_argument("--spawn-point", type=int, required=True,
                   help="Ego-vehicle spawn-point index")
    p.add_argument("--camera-rig", required=True,
                   help="Path to rig JSON in configs/rigs/")
    p.add_argument("--output", default="output/debug",
                   help="Debug output root (default: output/debug)")
    p.add_argument("--scene-id",
                   help="Optional explicit scene ID for this session")
    p.add_argument("--weather", default="ClearNoon",
                   help="CARLA weather preset (default: ClearNoon)")
    p.add_argument("--traffic-density", type=float,
                   default=DEFAULT_TRAFFIC_DENSITY,
                   help="Traffic density (default: 0.3)")
    p.add_argument("--max-pedestrians", type=int,
                   default=DEFAULT_MAX_PEDESTRIANS,
                   help="Maximum pedestrians (default: 50)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for traffic spawning (default: 42)")
    p.add_argument("--warmup-ticks", type=int, default=DEFAULT_WARMUP_TICKS,
                   help="Ticks to warm up sensors before interaction")
    p.add_argument("--host", default="localhost", help="CARLA host")
    p.add_argument("--port", type=int, default=2000, help="CARLA RPC port")
    p.add_argument("--tm-port", type=int, default=DEFAULT_TM_PORT,
                   help="Traffic Manager port (default: 8000)")
    p.add_argument("--timeout", type=float, default=60.0,
                   help="CARLA client timeout (seconds)")
    p.add_argument("--max-detection-distance", type=float, default=80.0,
                   help="Annotation detection radius in metres")
    p.add_argument("--version", default=DEFAULT_VERSION,
                   help="Version tag for exported annotation tables")
    p.add_argument("--split", default=DEFAULT_SPLIT,
                   choices=["trainval", "test", "mini"],
                   help="Split name for exported tables")
    p.add_argument("--show-debug-boxes", action="store_true",
                   help="Enable debug bounding boxes from the start")
    p.add_argument("--no-lidar", action="store_true",
                   help="Disable LiDAR sensor and num_lidar_pts annotations")
    p.add_argument(
        "--trajectory-file",
        help=(
            "Path to an HDF5 trajectory file for deterministic traffic/pedestrian "
            "replay.  When omitted, live autopilot mode is used."
        ),
    )
    return p


def main() -> None:
    """CLI entry point."""
    args = build_arg_parser().parse_args()
    os.makedirs(args.output, exist_ok=True)
    tool = AnnotationDebugTool(args)
    try:
        tool.run()
    finally:
        tool.cleanup()


if __name__ == "__main__":
    main()
