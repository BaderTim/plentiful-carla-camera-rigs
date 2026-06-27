#!/usr/bin/env python3
"""Minimal pedestrian replay comparator for recorded CARLA trajectories.

This debug script replays a single trajectory HDF5 and keeps the spectator
locked above the first successfully spawned pedestrian so pedestrian motion can
be compared visually between three modes:

``current``
    Reuse the production replay behavior: per-tick ``WalkerControl`` plus the
    periodic snap correction implemented by :class:`lib.trajectory_utils.TrajectoryPlayer`.

``waypoint``
    Spawn AI walker controllers and send each pedestrian once to its final
    recorded position at startup. Traffic and ego still follow the recorded
    transforms every tick so only pedestrian locomotion differs.

``spawn-only``
    Spawn ego, traffic, and pedestrians from the recorded initial state, start
    AI walker controllers, then let the simulation run without replay commands
    or walker destinations.

The script intentionally creates no sensors, dataset artifacts, or capture
outputs. It only restores the recorded map/weather/context and runs the
simulation for the full recorded duration.

Usage::

    python tools/debug/pedestrian_replay_compare.py \
        --trajectory output/trajectories/mini/mini_01.h5 \
        --mode current

    python tools/debug/pedestrian_replay_compare.py \
        --trajectory output/trajectories/mini/mini_01.h5 \
        --mode waypoint

    python tools/debug/pedestrian_replay_compare.py \
        --trajectory output/trajectories/mini/mini_01.h5 \
        --mode spawn-only
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

# Ensure project root is importable when run directly.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import carla
except ImportError:
    print(
        "Error: CARLA Python API not found. "
        "Make sure CARLA is installed and the Python path is set correctly."
    )
    sys.exit(1)

from lib.trajectory_utils import TrajectoryPlayer

DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_SPECTATOR_HEIGHT_METERS = 18.0
RANDOM_SEED = 42
DEFAULT_TRAFFIC_MANAGER_PORT = 8000


def configure_console_logging(debug: bool) -> logging.Logger:
    """Configure the shared logger to emit to stdout only."""
    logger = logging.getLogger("scenario_runner")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG if debug else logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


class PedestrianReplayComparator:
    """Replay a recorded CARLA trajectory with three pedestrian control modes."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.logger = configure_console_logging(args.debug)
        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(args.timeout)

        self.world: carla.World | None = None
        self.original_settings = None
        self.player = TrajectoryPlayer.from_file(args.trajectory)
        self.traffic_manager = None

        self.ego_vehicle: carla.Actor | None = None
        self.traffic_vehicles: List[carla.Actor | None] = []
        self.pedestrians: List[carla.Actor | None] = []
        self.walker_controllers: List[carla.Actor | None] = []
        self.tracked_pedestrian_index: int | None = None

    def run(self) -> None:
        """Restore scene context and replay the trajectory."""
        self._validate_trajectory()

        meta = self.player.metadata
        self.logger.info(
            "Loaded scene %s from %s", meta["scene_id"], self.player.filepath
        )
        self.logger.info(
            "Map=%s Weather=%s FPS=%s Frames=%s Pedestrians=%s Traffic=%s Mode=%s",
            meta["map_name"],
            meta["weather"],
            meta["simulation_fps"],
            meta["total_frames"],
            self.player.pedestrian_count,
            self.player.traffic_count,
            self.args.mode,
        )

        try:
            self._set_random_seeds()
            self._load_world(meta["map_name"])
            self._enable_synchronous_mode(float(meta["simulation_fps"]))
            self._configure_traffic_manager()
            self._set_pedestrians_seed()
            self._set_weather(str(meta["weather"]))

            self._spawn_ego_vehicle()
            self._spawn_traffic_vehicles()
            self._spawn_pedestrians()
            self._select_tracked_pedestrian()

            self.player.set_client(self.client)
            self.player.set_ego_vehicle(self.ego_vehicle)
            self.player.set_traffic_vehicles(self.traffic_vehicles)
            self.player.set_pedestrians(self.pedestrians)

            if self.args.mode in {"waypoint", "spawn-only"}:
                self._setup_walker_controllers(
                    assign_targets=self.args.mode == "waypoint"
                )

            self._update_spectator()
            self._replay_frames()
        finally:
            self.cleanup()

    def _set_random_seeds(self) -> None:
        """Seed Python and NumPy for deterministic helper behavior."""
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

    def _configure_traffic_manager(self) -> None:
        """Create a seeded Traffic Manager instance for determinism."""
        self.traffic_manager = self.client.get_trafficmanager(self.args.tm_port)
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(RANDOM_SEED)
        self.traffic_manager.set_hybrid_physics_mode(False)
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.0)

    def _set_pedestrians_seed(self) -> None:
        """Seed CARLA pedestrian randomness before walker control starts."""
        if self.world is None:
            raise RuntimeError("World is not loaded.")
        self.world.set_pedestrians_seed(RANDOM_SEED)

    def _validate_trajectory(self) -> None:
        """Ensure the trajectory can support the comparison run."""
        if self.player.pedestrian_count <= 0:
            raise ValueError(
                f"Trajectory {self.player.filepath} contains no pedestrians to compare."
            )
        if int(self.player.metadata["total_frames"]) <= 0:
            raise ValueError(
                f"Trajectory {self.player.filepath} contains no recorded frames."
            )

    def _load_world(self, map_name: str) -> None:
        """Load the recorded CARLA map if it is not already active."""
        current_map = self.world.get_map().name if self.world else ""
        if not current_map.endswith(map_name):
            self.logger.info("Loading map %s...", map_name)
            self.world = self.client.load_world(map_name)
        else:
            self.logger.info("Map %s already loaded.", map_name)

        for _ in range(5):
            self.world.tick()

    def _enable_synchronous_mode(self, simulation_fps: float) -> None:
        """Enable fixed-timestep synchronous mode."""
        if self.world is None:
            raise RuntimeError("World is not loaded.")

        self.original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / simulation_fps
        settings.no_rendering_mode = False
        self.world.apply_settings(settings)
        self.logger.info("Synchronous mode enabled at %.1f FPS.", simulation_fps)

    def _restore_world_settings(self) -> None:
        """Restore world settings captured before synchronous replay."""
        if self.world is None or self.original_settings is None:
            return
        try:
            self.world.apply_settings(self.original_settings)
        except Exception as exc:
            self.logger.warning("Could not restore world settings: %s", exc)
        if self.traffic_manager is not None:
            try:
                self.traffic_manager.set_synchronous_mode(False)
            except Exception as exc:
                self.logger.warning("Could not disable Traffic Manager sync: %s", exc)

    def _set_weather(self, weather_preset: str) -> None:
        """Apply the recorded weather preset."""
        if self.world is None:
            raise RuntimeError("World is not loaded.")

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
            self.logger.warning(
                "Unknown weather preset %s, using ClearNoon.", weather_preset
            )
        self.world.set_weather(
            presets.get(weather_preset, carla.WeatherParameters.ClearNoon)
        )

    def _spawn_ego_vehicle(self) -> None:
        """Spawn the recorded ego vehicle at the first recorded transform."""
        if self.world is None:
            raise RuntimeError("World is not loaded.")

        blueprint = self.world.get_blueprint_library().find(self.player.ego_blueprint_id)
        if blueprint is None:
            raise RuntimeError(
                f"Ego blueprint {self.player.ego_blueprint_id} not found in blueprint library."
            )

        transform = self._build_transform(
            self.player.ego_locations[0],
            self.player.ego_rotations[0],
        )
        self.ego_vehicle = self.world.spawn_actor(blueprint, transform)
        self.ego_vehicle.set_simulate_physics(False)
        self.world.tick()
        self.logger.info("Spawned ego vehicle %s.", self.player.ego_blueprint_id)

    def _spawn_traffic_vehicles(self) -> None:
        """Spawn recorded traffic vehicles with physics disabled."""
        if self.world is None:
            raise RuntimeError("World is not loaded.")

        traffic_bps, traffic_spawns = self.player.get_traffic_spawn_info()
        lib = self.world.get_blueprint_library()

        for index, (bp_id, (loc, rot)) in enumerate(zip(traffic_bps, traffic_spawns)):
            try:
                blueprint = lib.find(bp_id)
                if blueprint is None:
                    self.logger.warning(
                        "Blueprint %s not found, skipping traffic vehicle %s.",
                        bp_id,
                        index,
                    )
                    self.traffic_vehicles.append(None)
                    continue

                vehicle = self.world.spawn_actor(blueprint, self._build_transform(loc, rot))
                vehicle.set_simulate_physics(False)
                self.traffic_vehicles.append(vehicle)
            except RuntimeError as exc:
                self.logger.warning(
                    "Failed to spawn traffic vehicle %s (%s): %s",
                    index,
                    bp_id,
                    exc,
                )
                self.traffic_vehicles.append(None)

        spawned = sum(actor is not None for actor in self.traffic_vehicles)
        self.logger.info(
            "Spawned %s/%s traffic vehicles.", spawned, len(traffic_bps)
        )
        if spawned:
            self.world.tick()

    def _spawn_pedestrians(self) -> None:
        """Spawn recorded pedestrians at their recorded spawn transforms."""
        if self.world is None:
            raise RuntimeError("World is not loaded.")

        ped_bps, ped_spawns = self.player.get_pedestrian_spawn_info()
        lib = self.world.get_blueprint_library()

        for index, (bp_id, (loc, rot)) in enumerate(zip(ped_bps, ped_spawns)):
            try:
                blueprint = lib.find(bp_id)
                if blueprint is None:
                    self.logger.warning(
                        "Blueprint %s not found, skipping pedestrian %s.", bp_id, index
                    )
                    self.pedestrians.append(None)
                    continue

                if blueprint.has_attribute("is_invincible"):
                    blueprint.set_attribute("is_invincible", "false")

                walker = self.world.spawn_actor(blueprint, self._build_transform(loc, rot))
                self.pedestrians.append(walker)
            except RuntimeError as exc:
                self.logger.warning(
                    "Failed to spawn pedestrian %s (%s): %s", index, bp_id, exc
                )
                self.pedestrians.append(None)

        spawned = sum(actor is not None for actor in self.pedestrians)
        self.logger.info("Spawned %s/%s pedestrians.", spawned, len(ped_bps))
        if spawned:
            self.world.tick()

    def _select_tracked_pedestrian(self) -> None:
        """Pick the first successfully spawned pedestrian for spectator follow."""
        for index, pedestrian in enumerate(self.pedestrians):
            if pedestrian is not None:
                self.tracked_pedestrian_index = index
                self.logger.info("Tracking pedestrian index %s.", index)
                return
        raise RuntimeError("No pedestrians spawned successfully; nothing to observe.")

    def _setup_walker_controllers(self, assign_targets: bool) -> None:
        """Attach AI walker controllers and optionally assign final targets."""
        if self.world is None:
            raise RuntimeError("World is not loaded.")

        controller_bp = self.world.get_blueprint_library().find("controller.ai.walker")
        if controller_bp is None:
            raise RuntimeError("controller.ai.walker blueprint not found.")

        self.walker_controllers = [None] * len(self.pedestrians)
        batch = []
        controller_indices: List[int] = []

        for index, walker in enumerate(self.pedestrians):
            if walker is None:
                continue
            batch.append(carla.command.SpawnActor(controller_bp, carla.Transform(), walker))
            controller_indices.append(index)

        if not batch:
            raise RuntimeError("No pedestrians available for walker controller setup.")

        results = self.client.apply_batch_sync(batch, True)
        for index, result in zip(controller_indices, results):
            if result.error:
                self.logger.warning(
                    "Failed to spawn walker controller for pedestrian %s: %s",
                    index,
                    result.error,
                )
                continue
            controller = self.world.get_actor(result.actor_id)
            self.walker_controllers[index] = controller

        self.world.tick()

        for index, controller in enumerate(self.walker_controllers):
            if controller is None or self.pedestrians[index] is None:
                continue

            try:
                controller.start()
                if assign_targets:
                    final_location = self.player.pedestrian_locations[-1][index]
                    target = carla.Location(
                        x=float(final_location[0]),
                        y=float(final_location[1]),
                        z=float(final_location[2]),
                    )
                    controller.go_to_location(target)
            except RuntimeError as exc:
                self.logger.warning(
                    "Failed to initialize controller for pedestrian %s: %s",
                    index,
                    exc,
                )

        self.world.tick()
        ready = sum(actor is not None for actor in self.walker_controllers)
        self.logger.info(
            "Initialized %s/%s walker controllers%s.",
            ready,
            len(self.pedestrians),
            " with targets" if assign_targets else " without targets",
        )

    def _replay_frames(self) -> None:
        """Run the full recorded replay duration."""
        if self.world is None:
            raise RuntimeError("World is not loaded.")

        total_frames = int(self.player.metadata["total_frames"])
        simulation_fps = float(self.player.metadata["simulation_fps"])
        start_time = time.time()

        self.logger.info("Starting replay for %s frames.", total_frames)

        for frame_idx in range(total_frames):
            if self.args.mode != "spawn-only":
                self.player.apply_ego_frame(frame_idx)

            if self.args.mode == "current":
                self.player.apply_frame(frame_idx)
            elif self.args.mode == "waypoint":
                self._apply_traffic_frame(frame_idx)

            self.world.tick()
            self._update_spectator()

            if frame_idx % 50 == 0 or frame_idx == total_frames - 1:
                elapsed_seconds = frame_idx / simulation_fps
                self.logger.info(
                    "Replay %.1fs / %.1fs (%s/%s frames)",
                    elapsed_seconds,
                    total_frames / simulation_fps,
                    frame_idx,
                    total_frames,
                )

        total_time = time.time() - start_time
        self.logger.info("Replay finished in %.1fs wall time.", total_time)

    def _apply_traffic_frame(self, frame_idx: int) -> None:
        """Apply recorded traffic vehicle transforms without touching pedestrians."""
        batch = []

        for index, vehicle in enumerate(self.traffic_vehicles):
            if vehicle is None:
                continue
            if index >= len(self.player.traffic_locations[frame_idx]):
                continue

            transform = self._build_transform(
                self.player.traffic_locations[frame_idx][index],
                self.player.traffic_rotations[frame_idx][index],
            )
            batch.append(carla.command.ApplyTransform(vehicle, transform))

        if batch:
            self.client.apply_batch_sync(batch, False)

    def _update_spectator(self) -> None:
        """Keep the spectator directly above the tracked pedestrian."""
        if self.world is None or self.tracked_pedestrian_index is None:
            return

        pedestrian = self.pedestrians[self.tracked_pedestrian_index]
        if pedestrian is None:
            return

        try:
            transform = pedestrian.get_transform()
        except RuntimeError:
            return

        spectator = self.world.get_spectator()
        spectator.set_transform(
            carla.Transform(
                transform.location + carla.Location(z=self.args.spectator_height),
                carla.Rotation(pitch=-90.0, yaw=transform.rotation.yaw, roll=0.0),
            )
        )

    def cleanup(self) -> None:
        """Stop controllers, destroy actors, and restore original world settings."""
        self.logger.info("Cleanup starting...")

        for controller in self.walker_controllers:
            if controller is None:
                continue
            try:
                controller.stop()
            except Exception:
                pass

        actors_to_destroy: List[carla.Actor] = []
        for actor in self.walker_controllers + self.pedestrians + self.traffic_vehicles:
            if actor is not None:
                actors_to_destroy.append(actor)
        if self.ego_vehicle is not None:
            actors_to_destroy.append(self.ego_vehicle)

        for actor in actors_to_destroy:
            try:
                actor.destroy()
            except Exception:
                pass

        self.walker_controllers = []
        self.pedestrians = []
        self.traffic_vehicles = []
        self.ego_vehicle = None

        try:
            if self.world is not None:
                self.world.tick()
        except Exception:
            pass

        self._restore_world_settings()
        self.traffic_manager = None
        self.logger.info("Cleanup complete.")

    @staticmethod
    def _build_transform(location: np.ndarray | List[float], rotation: np.ndarray | List[float]) -> carla.Transform:
        """Convert numeric arrays into a CARLA transform."""
        return carla.Transform(
            carla.Location(
                x=float(location[0]),
                y=float(location[1]),
                z=float(location[2]),
            ),
            carla.Rotation(
                pitch=float(rotation[0]),
                yaw=float(rotation[1]),
                roll=float(rotation[2]),
            ),
        )


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line interface for the comparator."""
    parser = argparse.ArgumentParser(
        description=(
            "Replay one trajectory HDF5 in current, waypoint, or spawn-only mode "
            "to compare pedestrian behavior."
        )
    )
    parser.add_argument(
        "--trajectory",
        required=True,
        help="Direct path to a recorded trajectory HDF5 file.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["current", "waypoint", "spawn-only"],
        help="Pedestrian replay mode to compare.",
    )
    parser.add_argument("--host", default="localhost", help="CARLA host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument(
        "--tm-port",
        type=int,
        default=DEFAULT_TRAFFIC_MANAGER_PORT,
        help=(
            "Traffic Manager port used only for deterministic seeding "
            f"(default: {DEFAULT_TRAFFIC_MANAGER_PORT})."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"CARLA client timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS}).",
    )
    parser.add_argument(
        "--spectator-height",
        type=float,
        default=DEFAULT_SPECTATOR_HEIGHT_METERS,
        help=(
            "Top-down spectator height above the tracked pedestrian in meters "
            f"(default: {DEFAULT_SPECTATOR_HEIGHT_METERS})."
        ),
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser


def main() -> int:
    """Parse arguments and run the replay comparison."""
    args = build_arg_parser().parse_args()

    try:
        PedestrianReplayComparator(args).run()
    except KeyboardInterrupt:
        logging.getLogger("scenario_runner").info("Interrupted by user.")
        return 130
    except Exception as exc:
        logging.getLogger("scenario_runner").error("Replay failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())