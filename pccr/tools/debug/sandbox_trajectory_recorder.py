#!/usr/bin/env python3
"""Minimal sandbox for validating the extended trajectory recorder schema."""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass
from pathlib import Path
import sys

import h5py

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.trajectory_utils import TrajectoryRecorder


@dataclass
class FakeVector3D:
    x: float
    y: float
    z: float


@dataclass
class FakeLocation:
    x: float
    y: float
    z: float


@dataclass
class FakeRotation:
    pitch: float
    yaw: float
    roll: float


@dataclass
class FakeTransform:
    location: FakeLocation
    rotation: FakeRotation


@dataclass
class FakeControl:
    throttle: float = 0.0
    steer: float = 0.0
    brake: float = 0.0
    hand_brake: bool = False
    reverse: bool = False
    manual_gear_shift: bool = False
    gear: int = 0


@dataclass
class FakeTrafficLightState:
    name: str


class FakeActor:
    def __init__(self, frames, speed_limit: float = 30.0):
        self.frames = frames
        self.index = 0
        self.speed_limit = speed_limit

    def advance(self) -> None:
        if self.index < len(self.frames) - 1:
            self.index += 1

    def _frame(self):
        return self.frames[self.index]

    def get_transform(self):
        return self._frame()["transform"]

    def get_velocity(self):
        return self._frame().get("velocity", FakeVector3D(0.0, 0.0, 0.0))

    def get_angular_velocity(self):
        return self._frame().get("angular_velocity", FakeVector3D(0.0, 0.0, 0.0))

    def get_control(self):
        return self._frame().get("control", FakeControl())

    def get_speed_limit(self):
        return self._frame().get("speed_limit", self.speed_limit)

    def is_at_traffic_light(self):
        return self._frame().get("is_at_traffic_light", False)

    def get_traffic_light_state(self):
        return self._frame().get("traffic_light_state", FakeTrafficLightState("Unknown"))


def make_transform(x: float, y: float, z: float = 0.0, yaw: float = 0.0) -> FakeTransform:
    return FakeTransform(
        location=FakeLocation(x=x, y=y, z=z),
        rotation=FakeRotation(pitch=0.0, yaw=yaw, roll=0.0),
    )


def build_fake_scene() -> tuple[FakeActor, FakeActor, FakeActor]:
    ego = FakeActor(
        [
            {
                "transform": make_transform(0.0, 0.0, z=1.0),
                "velocity": FakeVector3D(0.0, 0.0, 0.0),
                "angular_velocity": FakeVector3D(0.0, 0.0, 0.0),
                "control": FakeControl(throttle=0.4, gear=1),
            },
            {
                "transform": make_transform(2.0, 0.0, z=1.0),
                "velocity": FakeVector3D(0.0, 0.0, 0.0),
                "angular_velocity": FakeVector3D(0.0, 0.0, 0.0),
                "control": FakeControl(throttle=0.5, gear=1),
            },
            {
                "transform": make_transform(4.5, 0.0, z=1.0),
                "velocity": FakeVector3D(0.2, 0.0, 0.0),
                "angular_velocity": FakeVector3D(0.0, 0.0, 0.0),
                "control": FakeControl(throttle=0.3, gear=1),
            },
        ],
        speed_limit=50.0,
    )
    traffic_vehicle = FakeActor(
        [
            {
                "transform": make_transform(10.0, 1.0, z=0.5),
                "velocity": FakeVector3D(0.0, 0.0, 0.0),
                "angular_velocity": FakeVector3D(0.0, 0.0, 0.0),
                "control": FakeControl(brake=1.0, gear=1),
                "is_at_traffic_light": True,
                "traffic_light_state": FakeTrafficLightState("Red"),
                "speed_limit": 40.0,
            },
            {
                "transform": make_transform(12.0, 1.0, z=0.5),
                "velocity": FakeVector3D(0.0, 0.0, 0.0),
                "angular_velocity": FakeVector3D(0.0, 0.0, 0.0),
                "control": FakeControl(throttle=0.6, gear=1),
                "is_at_traffic_light": False,
                "speed_limit": 40.0,
            },
            {
                "transform": make_transform(14.0, 1.0, z=0.5),
                "velocity": FakeVector3D(0.0, 0.0, 0.0),
                "angular_velocity": FakeVector3D(0.0, 0.0, 0.05),
                "control": FakeControl(throttle=0.7, gear=1),
                "is_at_traffic_light": False,
                "speed_limit": 40.0,
            },
        ],
        speed_limit=40.0,
    )
    pedestrian = FakeActor(
        [
            {
                "transform": make_transform(-1.0, -1.0, z=0.0),
                "velocity": FakeVector3D(0.2, 0.0, 0.0),
                "angular_velocity": FakeVector3D(0.0, 0.0, 0.0),
            },
            {
                "transform": make_transform(-0.5, -1.0, z=0.0),
                "velocity": FakeVector3D(0.0, 0.0, 0.0),
                "angular_velocity": FakeVector3D(0.0, 0.0, 0.0),
            },
            {
                "transform": make_transform(0.0, -1.0, z=0.0),
                "velocity": FakeVector3D(0.0, 0.0, 0.0),
                "angular_velocity": FakeVector3D(0.0, 0.0, 0.0),
            },
        ]
    )
    return ego, traffic_vehicle, pedestrian


def run_sandbox(output_root: Path) -> Path:
    scene = {
        "id": "mini_sandbox",
        "map": "Town01",
        "weather": "ClearNoon",
        "traffic_density": 0.1,
        "spawn_point": 0,
    }
    ego, traffic_vehicle, pedestrian = build_fake_scene()
    recorder = TrajectoryRecorder(
        output_dir=str(output_root),
        scene_config=scene,
        simulation_fps=2.0,
        settle_frames=0,
        random_seed=123,
    )
    recorder.set_ego_vehicle(ego, "vehicle.test.ego")
    recorder.set_traffic_vehicles(
        [traffic_vehicle],
        ["vehicle.test.traffic"],
        [traffic_vehicle.get_transform()],
    )
    recorder.set_pedestrians(
        [pedestrian],
        ["walker.test.pedestrian"],
        [pedestrian.get_transform()],
    )

    for frame_idx in range(3):
        recorder.record_frame(frame_idx)
        ego.advance()
        traffic_vehicle.advance()
        pedestrian.advance()

    recorder.save()
    return output_root / "mini" / "mini_sandbox.h5"


def validate_output(h5_path: Path) -> None:
    with h5py.File(h5_path, "r") as handle:
        assert handle["ego_vehicle/state/selected_velocities"].shape == (3, 3)
        assert handle["traffic_vehicles/state/selected_velocities"].shape == (3, 1, 3)
        assert handle["pedestrians/state/selected_velocities"].shape == (3, 1, 3)
        assert bool(handle["ego_vehicle/state/used_estimated_velocities"][1])
        assert bool(handle["traffic_vehicles/state/used_estimated_velocities"][1, 0])
        assert int(handle["traffic_vehicles/vehicle_context/traffic_light_state_codes"][0, 0]) == 2
        assert bool(handle["traffic_vehicles/vehicle_context/is_at_traffic_light"][0, 0])
        assert abs(float(handle["traffic_vehicles/state/selected_velocities"][1, 0, 0]) - 4.0) < 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the extended trajectory recorder schema without CARLA.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Optional directory for the sandbox HDF5 output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root or Path(tempfile.mkdtemp(prefix="pccr_trajectory_sandbox_"))
    h5_path = run_sandbox(output_root)
    validate_output(h5_path)
    print(f"sandbox trajectory file: {h5_path}")
    print("validated datasets:")
    with h5py.File(h5_path, "r") as handle:
        for dataset in (
            "ego_vehicle/state/selected_velocities",
            "ego_vehicle/state/used_estimated_velocities",
            "traffic_vehicles/state/selected_velocities",
            "traffic_vehicles/vehicle_context/traffic_light_state_codes",
            "pedestrians/state/selected_velocities",
        ):
            print(f"  {dataset}: shape={handle[dataset].shape}")


if __name__ == "__main__":
    main()