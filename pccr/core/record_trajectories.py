#!/usr/bin/env python3
"""
Record traffic and pedestrian trajectories for deterministic scene replay.

Each scene is run once with a firetruck (largest vehicle) as the ego vehicle.
All traffic-vehicle and pedestrian transforms are recorded at every simulation
tick and saved to HDF5 files.  The resulting trajectory files are later
replayed by ``core/scene_runner.py`` with physics disabled via
``set_transform()``, guaranteeing identical traffic behaviour across
different ego-vehicle / camera-rig combinations.

Produced files are organised as::

    <output>/
        trainval/
            <scene_id>_trajectories.h5
        test/
            <scene_id>_trajectories.h5
        mini/
            <scene_id>_trajectories.h5

Usage::

    python core/record_trajectories.py \\
        --output ./output/trajectories \\
        --scenes configs/scenes.json \\
        --split mini
"""

import argparse
import datetime
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Ensure project root is importable when the script is run directly.
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import carla
except ImportError:
    print(
        "Error: CARLA Python API not found.  "
        "Make sure CARLA is installed and the Python path is set correctly."
    )
    sys.exit(1)

from lib.utils.logging_utils import log_print, setup_logging
from lib.trajectory_utils import TrajectoryRecorder

# ---------------------------------------------------------------------------
# Constants  (must match core/scene_runner.py)
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
SIMULATION_FPS = 10
RECORDING_DURATION = 30.0
SETTLE_DURATION = 1.0

#: Largest vehicle blueprint — ensures all camera rigs fit within the
#: recorded trajectories regardless of the ego vehicle used during capture.
RECORDING_VEHICLE_TYPE = "vehicle.carlamotors.firetruck"


def reset_random_state_for_scene() -> None:
    """Reset Python and NumPy random state for deterministic per-scene behaviour."""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)


def determine_scene_split(scene_id: str) -> str:
    """Map a scene ID prefix to a nuScenes split name.

    Args:
        scene_id: Scene identifier string (e.g. ``"mini_01"``).

    Returns:
        One of ``"trainval"``, ``"test"``, or ``"mini"``.
    """
    if scene_id.startswith(("train_", "val_")):
        return "trainval"
    if scene_id.startswith("test_"):
        return "test"
    if scene_id.startswith("mini_"):
        return "mini"
    return "trainval"


# ---------------------------------------------------------------------------
# Main runner class
# ---------------------------------------------------------------------------

class TrajectoryRecordingRunner:
    """Spawn CARLA traffic and record actor transforms per tick.

    Attributes:
        client: Connected CARLA client.
        world: The currently loaded :class:`carla.World`.
        ego_vehicle: Spawned firetruck actor (or ``None`` before spawn).
        traffic_vehicles: List of spawned traffic-vehicle actors.
        traffic_blueprint_ids: Blueprint IDs corresponding to
            *traffic_vehicles* (for HDF5 metadata).
        traffic_spawn_transforms: Spawn transforms for *traffic_vehicles*.
        walkers: Spawned pedestrian actors.
        walker_controllers: AI-walker controller actors.
        walker_blueprint_ids: Blueprint IDs for each walker.
        walker_spawn_transforms: Spawn transforms for *walkers*.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 2000,
        timeout: float = 60.0,
        debug: bool = False,
    ) -> None:
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)
        self.world = None
        self.ego_vehicle = None
        self.original_settings = None
        self.debug = debug

        self.traffic_vehicles: List[carla.Actor] = []
        self.traffic_blueprint_ids: List[str] = []
        self.traffic_spawn_transforms: List[carla.Transform] = []

        self.walkers: List[carla.Actor] = []
        self.walker_controllers: List[carla.Actor] = []
        self.walker_blueprint_ids: List[str] = []
        self.walker_spawn_transforms: List[carla.Transform] = []

    # ------------------------------------------------------------------
    # Configuration loading
    # ------------------------------------------------------------------

    def load_scenes(self, scenes_path: str) -> Dict[str, Any]:
        """Load a scenes.json configuration file.

        Args:
            scenes_path: Path to the JSON file.

        Returns:
            Parsed scenes config dictionary.
        """
        with open(scenes_path, "r") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # World setup
    # ------------------------------------------------------------------

    def set_weather(self, weather_preset: str) -> None:
        """Set CARLA weather by preset name.

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
            log_print(
                f"Unknown weather preset '{weather_preset}', using ClearNoon",
                "WARNING",
            )
        self.world.set_weather(presets.get(weather_preset, carla.WeatherParameters.ClearNoon))

    def enable_synchronous_mode(self) -> None:
        """Switch the CARLA world to fixed-timestep synchronous mode."""
        self.original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / SIMULATION_FPS
        settings.no_rendering_mode = False
        self.world.apply_settings(settings)

        tm = self.client.get_trafficmanager()
        tm.set_synchronous_mode(True)
        tm.set_random_device_seed(RANDOM_SEED)
        tm.set_hybrid_physics_mode(False)
        tm.set_global_distance_to_leading_vehicle(2.0)
        log_print(f"Synchronous mode enabled at {SIMULATION_FPS:.0f} FPS.")

    def disable_synchronous_mode(self) -> None:
        """Restore the original world settings saved before synchronous mode."""
        if self.original_settings is None:
            return
        try:
            self.world.apply_settings(self.original_settings)
        except Exception as exc:
            log_print(f"Warning: Could not restore world settings: {exc}", "WARNING")
        try:
            self.client.get_trafficmanager().set_synchronous_mode(False)
        except Exception as exc:
            log_print(
                f"Warning: Could not disable TM synchronous mode: {exc}", "WARNING"
            )

    def update_spectator_follow_ego(self) -> None:
        """Move the CARLA spectator camera to follow the ego vehicle."""
        if self.ego_vehicle is None:
            return
        t = self.ego_vehicle.get_transform()
        fwd = t.rotation.get_forward_vector()
        spec_loc = t.location + carla.Location(x=-10 * fwd.x, y=-10 * fwd.y, z=5)
        self.world.get_spectator().set_transform(carla.Transform(spec_loc, t.rotation))

    # ------------------------------------------------------------------
    # Actor spawning
    # ------------------------------------------------------------------

    def spawn_ego_vehicle(self, spawn_point_idx: int) -> carla.Transform:
        """Spawn the firetruck ego vehicle.

        Args:
            spawn_point_idx: Index into the map's spawn-point list.

        Returns:
            The actual spawn :class:`~carla.Transform` used.

        Raises:
            RuntimeError: If the firetruck blueprint is not found.
        """
        spawn_points = self.world.get_map().get_spawn_points()
        if spawn_point_idx >= len(spawn_points):
            log_print(
                f"Spawn point {spawn_point_idx} not available, using point 0.",
                "WARNING",
            )
            spawn_point_idx = 0

        bp = self.world.get_blueprint_library().find(RECORDING_VEHICLE_TYPE)
        if bp is None:
            raise RuntimeError(
                f"Recording vehicle '{RECORDING_VEHICLE_TYPE}' not found in library."
            )

        spawn_tf = spawn_points[spawn_point_idx]
        self.ego_vehicle = self.world.spawn_actor(bp, spawn_tf)
        self.world.tick()
        self.world.tick()
        log_print(f"Spawned ego vehicle: {RECORDING_VEHICLE_TYPE}")
        return spawn_tf

    def spawn_traffic(
        self,
        traffic_density: float,
        ego_spawn_point: carla.Transform,
    ) -> List[carla.Actor]:
        """Spawn traffic vehicles avoiding the ego spawn area.

        Args:
            traffic_density: Fraction of available spawn points to populate
                (``0.0``–``1.0``).
            ego_spawn_point: Ego vehicle's spawn transform (exclusion zone centre).

        Returns:
            List of successfully spawned vehicle actors.
        """
        self.traffic_vehicles = []
        self.traffic_blueprint_ids = []
        self.traffic_spawn_transforms = []

        all_sps = self.world.get_map().get_spawn_points()
        all_sps.sort(key=lambda sp: (sp.location.x, sp.location.y, sp.location.z))

        avail = [
            sp for sp in all_sps
            if ego_spawn_point.location.distance(sp.location) > 10.0
        ]
        n = min(int(traffic_density * len(avail)), len(avail))

        selected = random.sample(avail, n)
        selected.sort(key=lambda sp: (sp.location.x, sp.location.y, sp.location.z))

        blueprints = sorted(
            self.world.get_blueprint_library().filter("vehicle.*"),
            key=lambda bp: bp.id,
        )
        bp_indices = [random.randint(0, len(blueprints) - 1) for _ in selected]

        for i, sp in enumerate(selected):
            try:
                vehicle = self.world.spawn_actor(blueprints[bp_indices[i]], sp)
                self.traffic_vehicles.append(vehicle)
                self.traffic_blueprint_ids.append(blueprints[bp_indices[i]].id)
                self.traffic_spawn_transforms.append(sp)
            except RuntimeError:
                continue

        log_print(f"Spawned {len(self.traffic_vehicles)} traffic vehicles.")
        return self.traffic_vehicles

    def spawn_pedestrians(self, max_pedestrians: int = 50) -> None:
        """Spawn AI pedestrians using CARLA's navigation mesh.

        Args:
            max_pedestrians: Hard cap on number of pedestrians to spawn.
        """
        self.walkers = []
        self.walker_controllers = []
        self.walker_blueprint_ids = []
        self.walker_spawn_transforms = []

        if max_pedestrians <= 0:
            log_print("Pedestrian spawning disabled (max_pedestrians=0).")
            return

        lib = self.world.get_blueprint_library()
        ped_bps = sorted(lib.filter("walker.pedestrian.*"), key=lambda bp: bp.id)
        ctrl_bp = lib.find("controller.ai.walker")

        # Gather candidate locations from the navigation mesh.
        spawn_tfs: List[carla.Transform] = []
        for _ in range(max_pedestrians * 2):
            loc = self.world.get_random_location_from_navigation()
            if loc is not None:
                spawn_tfs.append(carla.Transform(location=loc))
        if not spawn_tfs:
            log_print("No valid pedestrian spawn points from navigation mesh.", "WARNING")
            return

        spawn_tfs.sort(key=lambda tf: (tf.location.x, tf.location.y, tf.location.z))
        spawn_tfs = spawn_tfs[:max_pedestrians]

        bp_indices = [random.randint(0, len(ped_bps) - 1) for _ in spawn_tfs]
        speeds = [1.0 + random.random() * 1.5 for _ in spawn_tfs]

        batch = []
        for i, tf in enumerate(spawn_tfs):
            bp = ped_bps[bp_indices[i]]
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")
            batch.append(carla.command.SpawnActor(bp, tf))

        results = self.client.apply_batch_sync(batch, True)
        spawned_indices: List[int] = []
        for i, res in enumerate(results):
            if not res.error:
                actor = self.world.get_actor(res.actor_id)
                if actor is not None:
                    self.walkers.append(actor)
                    self.walker_blueprint_ids.append(ped_bps[bp_indices[i]].id)
                    self.walker_spawn_transforms.append(spawn_tfs[i])
                    spawned_indices.append(i)

        if not self.walkers:
            log_print("No pedestrians successfully spawned.", "WARNING")
            return

        self.world.tick()

        ctrl_batch = [
            carla.command.SpawnActor(ctrl_bp, carla.Transform(), w)
            for w in self.walkers
        ]
        ctrl_results = self.client.apply_batch_sync(ctrl_batch, True)
        for res in ctrl_results:
            if not res.error:
                ctrl = self.world.get_actor(res.actor_id)
                if ctrl is not None:
                    self.walker_controllers.append(ctrl)

        self.world.tick()

        for idx, ctrl in enumerate(self.walker_controllers):
            try:
                ctrl.start()
                dest = self.world.get_random_location_from_navigation()
                if dest is not None:
                    ctrl.go_to_location(dest)
                speed = speeds[spawned_indices[idx]] if idx < len(spawned_indices) else 1.5
                ctrl.set_max_speed(speed)
            except RuntimeError:
                continue

        log_print(
            f"Spawned {len(self.walkers)} pedestrians with "
            f"{len(self.walker_controllers)} controllers."
        )

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_scene(self, scene: Dict[str, Any], output_dir: str, max_pedestrians: int = 100) -> None:
        """Record trajectories for a single scene.

        Args:
            scene: Scene config dict (from ``scenes.json``).
            output_dir: Root directory for HDF5 output files.
            max_pedestrians: Hard cap on number of pedestrians to spawn.
        """
        scene_id = scene["id"]
        map_name = scene["map"]
        weather = scene["weather"]
        traffic_density = scene["traffic_density"]
        spawn_point = scene["spawn_point"]

        reset_random_state_for_scene()
        log_print(
            f"Recording scene: {scene_id}  map={map_name}  "
            f"weather={weather}  density={traffic_density}  sp={spawn_point}"
        )

        try:
            # Shut down old TM to avoid port conflicts.
            try:
                self.client.get_trafficmanager().shut_down()
            except Exception:
                pass

            log_print(f"Loading map {map_name}...")
            self.world = self.client.load_world(map_name)
            for _ in range(10):
                self.world.tick()
            time.sleep(1.0)

            self.enable_synchronous_mode()
            self.set_weather(weather)

            spawn_tf = self.spawn_ego_vehicle(spawn_point)
            self.spawn_traffic(traffic_density, spawn_tf)

            tm = self.client.get_trafficmanager()
            tm_port = tm.get_port()
            tm.set_random_device_seed(RANDOM_SEED)

            self.world.set_pedestrians_seed(RANDOM_SEED)

            self.ego_vehicle.set_autopilot(True, tm_port)
            for v in self.traffic_vehicles:
                try:
                    v.set_autopilot(True, tm_port)
                except Exception:
                    pass
            log_print(
                f"Autopilot enabled for ego + {len(self.traffic_vehicles)} vehicles."
            )

            settle_frames = int(SETTLE_DURATION * SIMULATION_FPS)
            recorder = TrajectoryRecorder(
                output_dir=output_dir,
                scene_config=scene,
                simulation_fps=SIMULATION_FPS,
                settle_frames=settle_frames,
                random_seed=RANDOM_SEED,
            )
            recorder.set_ego_vehicle(self.ego_vehicle, RECORDING_VEHICLE_TYPE)
            recorder.set_traffic_vehicles(
                self.traffic_vehicles,
                self.traffic_blueprint_ids,
                self.traffic_spawn_transforms,
            )
            recorder.set_pedestrians(
                self.walkers,
                self.walker_blueprint_ids,
                self.walker_spawn_transforms,
            )

            recording_frames = int(RECORDING_DURATION * SIMULATION_FPS)
            total_frames = settle_frames + recording_frames
            log_print(
                f"Recording {total_frames} frames "
                f"({settle_frames} settle + {recording_frames} capture)."
            )

            # Spawn pedestrians after initial setup (matches scene_runner.py).
            self.spawn_pedestrians(max_pedestrians=max_pedestrians)
            recorder.set_pedestrians(
                self.walkers,
                self.walker_blueprint_ids,
                self.walker_spawn_transforms,
            )

            for frame in range(total_frames):
                self.world.tick()
                recorder.record_frame(frame)
                self.update_spectator_follow_ego()
                if frame % 50 == 0:
                    elapsed = frame / SIMULATION_FPS
                    total_t = total_frames / SIMULATION_FPS
                    log_print(
                        f"  {elapsed:.1f}s / {total_t:.1f}s "
                        f"({frame}/{total_frames} frames)"
                    )

            recorder.save()
            log_print(f"Completed recording for scene {scene_id}.")

        except Exception as exc:
            log_print(f"Error recording scene {scene_id}: {exc}", "ERROR")
            raise
        finally:
            self.cleanup()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Destroy all spawned actors and restore world settings."""
        log_print("Cleaning up...")

        for ctrl in self.walker_controllers:
            try:
                ctrl.stop()
            except Exception:
                pass
        for actor in self.walker_controllers + self.walkers + self.traffic_vehicles:
            try:
                actor.destroy()
            except Exception:
                pass
        if self.ego_vehicle is not None:
            try:
                self.ego_vehicle.destroy()
            except Exception:
                pass

        self.traffic_vehicles = []
        self.traffic_blueprint_ids = []
        self.traffic_spawn_transforms = []
        self.walkers = []
        self.walker_controllers = []
        self.walker_blueprint_ids = []
        self.walker_spawn_transforms = []
        self.ego_vehicle = None

        time.sleep(0.5)
        try:
            if self.world is not None:
                self.world.tick()
        except Exception:
            pass

        self.disable_synchronous_mode()
        log_print("Cleanup completed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments and record trajectories for all requested scenes."""
    parser = argparse.ArgumentParser(
        description="Record traffic and pedestrian trajectories for scene replay."
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for trajectory HDF5 files",
    )
    parser.add_argument(
        "--scenes", required=True,
        help="Path to scenes.json file",
    )
    parser.add_argument(
        "--split", choices=["trainval", "test", "mini"],
        help="Record only scenes from this split (optional)",
    )
    parser.add_argument(
        "--limit", type=int,
        help="Limit number of scenes to record (optional)",
    )
    parser.add_argument("--host", default="localhost", help="CARLA host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument(
        "--max-pedestrians", type=int, default=100,
        help="Maximum pedestrians per scene (default: 100)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    setup_logging(args.output, debug=args.debug)

    log_print("=" * 60)
    log_print("TRAJECTORY RECORDING")
    log_print("=" * 60)
    log_print(f"Output:    {args.output}")
    log_print(f"Scenes:    {args.scenes}")
    log_print(f"Vehicle:   {RECORDING_VEHICLE_TYPE}")
    log_print(f"Sim FPS:   {SIMULATION_FPS}")
    log_print(f"Settle:    {SETTLE_DURATION}s  Recording: {RECORDING_DURATION}s")

    runner = TrajectoryRecordingRunner(args.host, args.port, debug=args.debug)
    scenes_config = runner.load_scenes(args.scenes)

    if args.split:
        if args.split == "trainval":
            scenes_to_run = (
                scenes_config["scenes"].get("train", [])
                + scenes_config["scenes"].get("val", [])
            )
        else:
            scenes_to_run = scenes_config["scenes"].get(args.split, [])
        log_print(f"Split '{args.split}': {len(scenes_to_run)} scenes.")
    else:
        scenes_to_run = []
        for split in ("train", "val", "test", "mini"):
            scenes_to_run.extend(scenes_config["scenes"].get(split, []))
        log_print(f"All splits: {len(scenes_to_run)} scenes.")

    if args.limit and args.limit < len(scenes_to_run):
        scenes_to_run = scenes_to_run[: args.limit]
        log_print(f"Limited to {args.limit} scenes.")

    start_time = time.time()
    success = 0
    for i, scene in enumerate(scenes_to_run):
        log_print(f"\n{'='*60}")
        log_print(f"Scene {i + 1}/{len(scenes_to_run)}: {scene['id']}")
        log_print("=" * 60)
        try:
            runner.record_scene(scene, args.output, max_pedestrians=args.max_pedestrians)
            success += 1
        except Exception as exc:
            log_print(f"Failed to record {scene['id']}: {exc}", "ERROR")

    total_time = time.time() - start_time
    log_print("\n" + "=" * 60)
    log_print("RECORDING COMPLETE")
    log_print("=" * 60)
    log_print(f"Recorded: {success}/{len(scenes_to_run)} scenes  ({total_time / 60:.1f} min)")
    log_print(f"Output:   {args.output}")


if __name__ == "__main__":
    main()
