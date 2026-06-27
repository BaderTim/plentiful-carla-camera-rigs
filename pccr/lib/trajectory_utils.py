"""
Trajectory recording and playback utilities for deterministic CARLA scene execution.

This module provides classes for recording actor trajectories (vehicles, pedestrians)
during a reference run and replaying them deterministically for different camera rigs.

HDF5 Schema:
------------
{split}/{scene_id}.h5
├── metadata/
│   ├── scene_id (str)
│   ├── map_name (str)
│   ├── weather (str)
│   ├── traffic_density (float)
│   ├── spawn_point (int)
│   ├── simulation_fps (float)
│   ├── total_frames (int)
│   ├── settle_frames (int)
│   ├── random_seed (int)
│   └── traffic_light_state_codes (str)
├── ego_vehicle/
│   ├── blueprint_id (str)
│   ├── locations (N, 3) float32 - [x, y, z] in CARLA coords
│   ├── rotations (N, 3) float32 - [pitch, yaw, roll] in degrees
│   ├── state/ - motion signals for later attribute classification
│   └── vehicle_context/ - control and traffic-light context
├── traffic_vehicles/
│   ├── count (int)
│   ├── blueprint_ids (list of str)
│   ├── spawn_points/ (group with spawn transforms)
│   ├── locations (N, M, 3) float32 - per-frame, per-vehicle
│   ├── rotations (N, M, 3) float32 - per-frame, per-vehicle
│   ├── state/ - motion signals for later attribute classification
│   └── vehicle_context/ - control and traffic-light context
└── pedestrians/
    ├── count (int)
    ├── blueprint_ids (list of str)
    ├── spawn_points/ (group with spawn transforms)
    ├── locations (N, P, 3) float32 - per-frame, per-pedestrian
    ├── rotations (N, P, 3) float32 - per-frame, per-pedestrian
    └── state/ - motion signals for later attribute classification
"""

import h5py
import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

# Try to import carla, but allow module to be imported without it for testing
try:
    import carla
except ImportError:
    carla = None

# Interval (in simulation frames) at which pedestrian positions are snapped to
# recorded ground-truth to prevent drift accumulation during animation-based replay.
WALKER_POSITION_SNAP_INTERVAL = 10
VELOCITY_FALLBACK_THRESHOLD_MS = 0.1
REPLAY_MOTION_THRESHOLD_MS = 0.5

STATE_VECTOR_KEYS = {
    "raw_velocities",
    "estimated_velocities",
    "selected_velocities",
    "accelerations",
    "angular_velocities",
}
STATE_BOOL_KEYS = {
    "used_estimated_velocities",
    "is_at_traffic_light",
    "control_hand_brakes",
    "control_reverses",
    "control_manual_gear_shifts",
}
STATE_INT_KEYS = {
    "traffic_light_state_codes",
    "control_gears",
}
MOTION_STATE_KEYS = (
    "raw_velocities",
    "estimated_velocities",
    "selected_velocities",
    "accelerations",
    "angular_velocities",
    "used_estimated_velocities",
)
VEHICLE_CONTEXT_KEYS = (
    "speed_limits",
    "is_at_traffic_light",
    "traffic_light_state_codes",
    "control_throttles",
    "control_steers",
    "control_brakes",
    "control_hand_brakes",
    "control_reverses",
    "control_manual_gear_shifts",
    "control_gears",
)


def get_logger():
    """Get the scenario runner logger."""
    return logging.getLogger('scenario_runner')


class TrajectoryRecorder:
    """Records actor trajectories during a CARLA simulation run.
    
    Usage:
        recorder = TrajectoryRecorder(output_dir, scene_config, simulation_fps)
        recorder.set_ego_vehicle(ego_vehicle)
        recorder.set_traffic_vehicles(traffic_vehicles, blueprint_ids, spawn_transforms)
        recorder.set_pedestrians(pedestrians, blueprint_ids, spawn_transforms)
        
        for frame in range(total_frames):
            world.tick()
            recorder.record_frame(frame)
        
        recorder.save()
    """
    
    def __init__(self, output_dir: str, scene_config: Dict[str, Any], 
                 simulation_fps: float, settle_frames: int, random_seed: int):
        """Initialize the trajectory recorder.
        
        Args:
            output_dir: Base output directory for trajectory files
            scene_config: Scene configuration dict with id, map, weather, etc.
            simulation_fps: Simulation frames per second
            settle_frames: Number of settle frames before recording starts
            random_seed: Random seed used for deterministic spawning
        """
        self.output_dir = Path(output_dir)
        self.scene_config = scene_config
        self.simulation_fps = simulation_fps
        self.settle_frames = settle_frames
        self.random_seed = random_seed
        self.frame_delta_seconds = 1.0 / float(simulation_fps) if simulation_fps > 0 else 0.0
        
        # Actor references
        self.ego_vehicle = None
        self.traffic_vehicles: List = []
        self.traffic_blueprint_ids: List[str] = []
        self.traffic_spawn_transforms: List = []
        self.pedestrians: List = []
        self.pedestrian_blueprint_ids: List[str] = []
        self.pedestrian_spawn_transforms: List = []
        
        # Trajectory data storage
        self.ego_locations: List[List[float]] = []
        self.ego_rotations: List[List[float]] = []
        self.traffic_locations: List[List[List[float]]] = []  # [frame][vehicle][xyz]
        self.traffic_rotations: List[List[List[float]]] = []  # [frame][vehicle][pyr]
        self.pedestrian_locations: List[List[List[float]]] = []  # [frame][ped][xyz]
        self.pedestrian_rotations: List[List[List[float]]] = []  # [frame][ped][pyr]
        self.ego_state = self._create_state_storage(include_vehicle_context=True)
        self.traffic_state = self._create_state_storage(include_vehicle_context=True)
        self.pedestrian_state = self._create_state_storage(include_vehicle_context=False)
        
    def set_ego_vehicle(self, ego_vehicle, blueprint_id: str):
        """Set the ego vehicle to track.
        
        Args:
            ego_vehicle: CARLA ego vehicle actor
            blueprint_id: Blueprint ID string (e.g., 'vehicle.carlamotors.firetruck')
        """
        self.ego_vehicle = ego_vehicle
        self.ego_blueprint_id = blueprint_id
        
    def set_traffic_vehicles(self, vehicles: List, blueprint_ids: List[str], 
                             spawn_transforms: List):
        """Set traffic vehicles to track.
        
        Args:
            vehicles: List of CARLA vehicle actors
            blueprint_ids: List of blueprint ID strings (same order as vehicles)
            spawn_transforms: List of spawn transforms (same order as vehicles)
        """
        self.traffic_vehicles = vehicles
        self.traffic_blueprint_ids = blueprint_ids
        self.traffic_spawn_transforms = spawn_transforms
        
    def set_pedestrians(self, pedestrians: List, blueprint_ids: List[str],
                        spawn_transforms: List):
        """Set pedestrians to track.
        
        Args:
            pedestrians: List of CARLA walker actors
            blueprint_ids: List of blueprint ID strings (same order as pedestrians)
            spawn_transforms: List of spawn transforms (same order as pedestrians)
        """
        self.pedestrians = pedestrians
        self.pedestrian_blueprint_ids = blueprint_ids
        self.pedestrian_spawn_transforms = spawn_transforms

    def _create_state_storage(self, include_vehicle_context: bool) -> Dict[str, List[Any]]:
        storage = {key: [] for key in MOTION_STATE_KEYS}
        if include_vehicle_context:
            storage.update({key: [] for key in VEHICLE_CONTEXT_KEYS})
        return storage

    @staticmethod
    def _vector3_to_list(vector: Any) -> List[float]:
        if vector is None:
            return [0.0, 0.0, 0.0]
        return [
            float(getattr(vector, "x", 0.0)),
            float(getattr(vector, "y", 0.0)),
            float(getattr(vector, "z", 0.0)),
        ]

    @staticmethod
    def _vector_speed(vector: List[float]) -> float:
        return math.sqrt(sum(component * component for component in vector))

    @staticmethod
    def _state_default(key: str) -> Any:
        if key in STATE_VECTOR_KEYS:
            return [0.0, 0.0, 0.0]
        if key in STATE_BOOL_KEYS:
            return False
        if key in STATE_INT_KEYS:
            return -1 if key == "traffic_light_state_codes" else 0
        return 0.0

    def _copy_state_value(self, value: Any) -> Any:
        if isinstance(value, list):
            return value.copy()
        return value

    def _get_previous_actor_value(self, frames: List[Any], actor_idx: int, key: str) -> Any:
        if frames and len(frames[-1]) > actor_idx:
            return self._copy_state_value(frames[-1][actor_idx])
        return self._state_default(key)

    def _get_previous_ego_value(self, frames: List[Any], key: str) -> Any:
        if frames:
            return self._copy_state_value(frames[-1])
        return self._state_default(key)

    @staticmethod
    def _traffic_light_state_code(state: Any) -> int:
        if state is None:
            return -1
        name = getattr(state, "name", None)
        if name is None:
            text = str(state)
            name = text.split(".")[-1] if text else ""
        mapping = {
            "green": 0,
            "yellow": 1,
            "red": 2,
            "off": 3,
            "unknown": 4,
        }
        return mapping.get(str(name).lower(), -1)

    def _capture_actor_state(
        self,
        actor: Any,
        prev_location: Optional[List[float]],
        prev_velocity: Optional[List[float]],
        include_vehicle_context: bool,
    ) -> Tuple[List[float], List[float], Dict[str, Any]]:
        transform = actor.get_transform()
        location = [
            float(transform.location.x),
            float(transform.location.y),
            float(transform.location.z),
        ]
        rotation = [
            float(transform.rotation.pitch),
            float(transform.rotation.yaw),
            float(transform.rotation.roll),
        ]

        raw_velocity = self._vector3_to_list(getattr(actor, "get_velocity", lambda: None)())
        estimated_velocity = [0.0, 0.0, 0.0]
        selected_velocity = raw_velocity.copy()
        used_estimated_velocity = False

        if prev_location is not None and self.frame_delta_seconds > 0.0:
            estimated_velocity = [
                (location[i] - prev_location[i]) / self.frame_delta_seconds
                for i in range(3)
            ]
            if (
                self._vector_speed(raw_velocity) < VELOCITY_FALLBACK_THRESHOLD_MS
                and self._vector_speed(estimated_velocity) > VELOCITY_FALLBACK_THRESHOLD_MS
            ):
                selected_velocity = estimated_velocity.copy()
                used_estimated_velocity = True

        if prev_velocity is not None and self.frame_delta_seconds > 0.0:
            acceleration = [
                (selected_velocity[i] - prev_velocity[i]) / self.frame_delta_seconds
                for i in range(3)
            ]
        else:
            acceleration = [0.0, 0.0, 0.0]

        state = {
            "raw_velocities": raw_velocity,
            "estimated_velocities": estimated_velocity,
            "selected_velocities": selected_velocity,
            "accelerations": acceleration,
            "angular_velocities": self._vector3_to_list(
                getattr(actor, "get_angular_velocity", lambda: None)()
            ),
            "used_estimated_velocities": used_estimated_velocity,
        }

        if include_vehicle_context:
            control = getattr(actor, "get_control", lambda: None)()
            try:
                is_at_traffic_light = bool(actor.is_at_traffic_light())
            except Exception:
                is_at_traffic_light = False
            try:
                traffic_light_state = actor.get_traffic_light_state() if is_at_traffic_light else None
            except Exception:
                traffic_light_state = None

            state.update(
                {
                    "speed_limits": float(getattr(actor, "get_speed_limit", lambda: 0.0)()),
                    "is_at_traffic_light": is_at_traffic_light,
                    "traffic_light_state_codes": self._traffic_light_state_code(traffic_light_state),
                    "control_throttles": float(getattr(control, "throttle", 0.0) or 0.0),
                    "control_steers": float(getattr(control, "steer", 0.0) or 0.0),
                    "control_brakes": float(getattr(control, "brake", 0.0) or 0.0),
                    "control_hand_brakes": bool(getattr(control, "hand_brake", False)),
                    "control_reverses": bool(getattr(control, "reverse", False)),
                    "control_manual_gear_shifts": bool(
                        getattr(control, "manual_gear_shift", False)
                    ),
                    "control_gears": int(getattr(control, "gear", 0) or 0),
                }
            )

        return location, rotation, state

    def _append_state(self, storage: Dict[str, List[Any]], state: Dict[str, Any]) -> None:
        for key in storage:
            storage[key].append(state[key])

    def _write_storage_group(
        self,
        group: h5py.Group,
        storage: Dict[str, List[Any]],
        keys: Tuple[str, ...],
    ) -> None:
        for key in keys:
            if key not in storage:
                continue
            dtype = np.float32
            if key in STATE_BOOL_KEYS:
                dtype = np.bool_
            elif key in STATE_INT_KEYS:
                dtype = np.int16
            group.create_dataset(key, data=np.array(storage[key], dtype=dtype))
        
    def record_frame(self, frame_idx: int):
        """Record transforms for all tracked actors at the current frame.
        
        Args:
            frame_idx: Current frame index (0-based, includes settle frames)
        """
        # Record ego vehicle
        if self.ego_vehicle is not None:
            try:
                prev_location = self.ego_locations[-1] if self.ego_locations else None
                prev_velocity = (
                    self.ego_state["selected_velocities"][-1]
                    if self.ego_state["selected_velocities"]
                    else None
                )
                ego_location, ego_rotation, ego_state = self._capture_actor_state(
                    self.ego_vehicle,
                    prev_location=prev_location,
                    prev_velocity=prev_velocity,
                    include_vehicle_context=True,
                )
            except RuntimeError:
                ego_location = self.ego_locations[-1] if self.ego_locations else [0.0, 0.0, 0.0]
                ego_rotation = self.ego_rotations[-1] if self.ego_rotations else [0.0, 0.0, 0.0]
                ego_state = {
                    key: self._get_previous_ego_value(self.ego_state[key], key)
                    for key in self.ego_state
                }

            self.ego_locations.append(ego_location)
            self.ego_rotations.append(ego_rotation)
            self._append_state(self.ego_state, ego_state)
        
        # Record traffic vehicles
        frame_traffic_locs = []
        frame_traffic_rots = []
        frame_traffic_state = {key: [] for key in self.traffic_state}
        for vehicle_idx, vehicle in enumerate(self.traffic_vehicles):
            try:
                prev_location = None
                prev_velocity = None
                if self.traffic_locations and len(self.traffic_locations[-1]) > vehicle_idx:
                    prev_location = self.traffic_locations[-1][vehicle_idx]
                if self.traffic_state["selected_velocities"] and len(
                    self.traffic_state["selected_velocities"][-1]
                ) > vehicle_idx:
                    prev_velocity = self.traffic_state["selected_velocities"][-1][vehicle_idx]

                location, rotation, actor_state = self._capture_actor_state(
                    vehicle,
                    prev_location=prev_location,
                    prev_velocity=prev_velocity,
                    include_vehicle_context=True,
                )
                frame_traffic_locs.append(location)
                frame_traffic_rots.append(rotation)
                for key in frame_traffic_state:
                    frame_traffic_state[key].append(actor_state[key])
            except RuntimeError:
                if self.traffic_locations and len(self.traffic_locations[-1]) > len(frame_traffic_locs):
                    last_idx = len(frame_traffic_locs)
                    frame_traffic_locs.append(self.traffic_locations[-1][last_idx])
                    frame_traffic_rots.append(self.traffic_rotations[-1][last_idx])
                else:
                    frame_traffic_locs.append([0.0, 0.0, 0.0])
                    frame_traffic_rots.append([0.0, 0.0, 0.0])
                for key in frame_traffic_state:
                    frame_traffic_state[key].append(
                        self._get_previous_actor_value(self.traffic_state[key], vehicle_idx, key)
                    )
        self.traffic_locations.append(frame_traffic_locs)
        self.traffic_rotations.append(frame_traffic_rots)
        self._append_state(self.traffic_state, frame_traffic_state)
        
        # Record pedestrians
        frame_ped_locs = []
        frame_ped_rots = []
        frame_ped_state = {key: [] for key in self.pedestrian_state}
        for pedestrian_idx, pedestrian in enumerate(self.pedestrians):
            try:
                prev_location = None
                prev_velocity = None
                if self.pedestrian_locations and len(self.pedestrian_locations[-1]) > pedestrian_idx:
                    prev_location = self.pedestrian_locations[-1][pedestrian_idx]
                if self.pedestrian_state["selected_velocities"] and len(
                    self.pedestrian_state["selected_velocities"][-1]
                ) > pedestrian_idx:
                    prev_velocity = self.pedestrian_state["selected_velocities"][-1][pedestrian_idx]

                location, rotation, actor_state = self._capture_actor_state(
                    pedestrian,
                    prev_location=prev_location,
                    prev_velocity=prev_velocity,
                    include_vehicle_context=False,
                )
                frame_ped_locs.append(location)
                frame_ped_rots.append(rotation)
                for key in frame_ped_state:
                    frame_ped_state[key].append(actor_state[key])
            except RuntimeError:
                if self.pedestrian_locations and len(self.pedestrian_locations[-1]) > len(frame_ped_locs):
                    last_idx = len(frame_ped_locs)
                    frame_ped_locs.append(self.pedestrian_locations[-1][last_idx])
                    frame_ped_rots.append(self.pedestrian_rotations[-1][last_idx])
                else:
                    frame_ped_locs.append([0.0, 0.0, 0.0])
                    frame_ped_rots.append([0.0, 0.0, 0.0])
                for key in frame_ped_state:
                    frame_ped_state[key].append(
                        self._get_previous_actor_value(self.pedestrian_state[key], pedestrian_idx, key)
                    )
        self.pedestrian_locations.append(frame_ped_locs)
        self.pedestrian_rotations.append(frame_ped_rots)
        self._append_state(self.pedestrian_state, frame_ped_state)
    
    def _transform_to_arrays(self, transforms: List) -> Tuple[np.ndarray, np.ndarray]:
        """Convert list of CARLA transforms to location and rotation arrays."""
        locations = []
        rotations = []
        for t in transforms:
            locations.append([t.location.x, t.location.y, t.location.z])
            rotations.append([t.rotation.pitch, t.rotation.yaw, t.rotation.roll])
        return np.array(locations, dtype=np.float32), np.array(rotations, dtype=np.float32)
    
    def save(self):
        """Save recorded trajectories to HDF5 file."""
        scene_id = self.scene_config['id']
        split = self._determine_split(scene_id)
        
        # Create split directory
        split_dir = self.output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = split_dir / f"{scene_id}.h5"
        
        logger = get_logger()
        logger.info(f"Saving trajectories to {filepath}")
        
        with h5py.File(filepath, 'w') as f:
            # Metadata
            meta = f.create_group('metadata')
            meta.attrs['scene_id'] = self.scene_config['id']
            meta.attrs['map_name'] = self.scene_config['map']
            meta.attrs['weather'] = self.scene_config['weather']
            meta.attrs['traffic_density'] = self.scene_config['traffic_density']
            meta.attrs['spawn_point'] = self.scene_config['spawn_point']
            meta.attrs['simulation_fps'] = self.simulation_fps
            meta.attrs['total_frames'] = len(self.ego_locations)
            meta.attrs['settle_frames'] = self.settle_frames
            meta.attrs['random_seed'] = self.random_seed
            meta.attrs['traffic_light_state_codes'] = 'none=-1,green=0,yellow=1,red=2,off=3,unknown=4'
            
            # Ego vehicle
            ego = f.create_group('ego_vehicle')
            ego.attrs['blueprint_id'] = self.ego_blueprint_id
            ego.create_dataset('locations', data=np.array(self.ego_locations, dtype=np.float32))
            ego.create_dataset('rotations', data=np.array(self.ego_rotations, dtype=np.float32))
            self._write_storage_group(ego.create_group('state'), self.ego_state, MOTION_STATE_KEYS)
            self._write_storage_group(
                ego.create_group('vehicle_context'),
                self.ego_state,
                VEHICLE_CONTEXT_KEYS,
            )
            
            # Traffic vehicles
            traffic = f.create_group('traffic_vehicles')
            traffic.attrs['count'] = len(self.traffic_vehicles)
            
            # Store blueprint IDs as variable-length strings
            if self.traffic_blueprint_ids:
                dt = h5py.special_dtype(vlen=str)
                traffic.create_dataset('blueprint_ids', data=self.traffic_blueprint_ids, dtype=dt)
                
                # Spawn points
                spawn_locs, spawn_rots = self._transform_to_arrays(self.traffic_spawn_transforms)
                spawn_grp = traffic.create_group('spawn_points')
                spawn_grp.create_dataset('locations', data=spawn_locs)
                spawn_grp.create_dataset('rotations', data=spawn_rots)
                
                # Per-frame transforms
                traffic.create_dataset('locations', 
                    data=np.array(self.traffic_locations, dtype=np.float32))
                traffic.create_dataset('rotations',
                    data=np.array(self.traffic_rotations, dtype=np.float32))
                self._write_storage_group(
                    traffic.create_group('state'),
                    self.traffic_state,
                    MOTION_STATE_KEYS,
                )
                self._write_storage_group(
                    traffic.create_group('vehicle_context'),
                    self.traffic_state,
                    VEHICLE_CONTEXT_KEYS,
                )
            
            # Pedestrians
            peds = f.create_group('pedestrians')
            peds.attrs['count'] = len(self.pedestrians)
            
            if self.pedestrian_blueprint_ids:
                dt = h5py.special_dtype(vlen=str)
                peds.create_dataset('blueprint_ids', data=self.pedestrian_blueprint_ids, dtype=dt)
                
                # Spawn points
                spawn_locs, spawn_rots = self._transform_to_arrays(self.pedestrian_spawn_transforms)
                spawn_grp = peds.create_group('spawn_points')
                spawn_grp.create_dataset('locations', data=spawn_locs)
                spawn_grp.create_dataset('rotations', data=spawn_rots)
                
                # Per-frame transforms
                peds.create_dataset('locations',
                    data=np.array(self.pedestrian_locations, dtype=np.float32))
                peds.create_dataset('rotations',
                    data=np.array(self.pedestrian_rotations, dtype=np.float32))
                self._write_storage_group(
                    peds.create_group('state'),
                    self.pedestrian_state,
                    MOTION_STATE_KEYS,
                )
        
        logger.info(f"Saved {len(self.ego_locations)} frames for scene {scene_id}")
        logger.info(f"  Traffic vehicles: {len(self.traffic_vehicles)}")
        logger.info(f"  Pedestrians: {len(self.pedestrians)}")
        
    def _determine_split(self, scene_id: str) -> str:
        """Determine which dataset split a scene belongs to based on its ID."""
        if scene_id.startswith("train_") or scene_id.startswith("val_"):
            return "trainval"
        elif scene_id.startswith("test_"):
            return "test"
        elif scene_id.startswith("mini_"):
            return "mini"
        else:
            return "trainval"


class TrajectoryPlayer:
    """Plays back recorded trajectories for deterministic scene replay.
    
    Usage:
        player = TrajectoryPlayer(trajectories_dir, scene_id)
        player.validate_scene_config(scene_config)
        
        # Spawn actors using player.get_* methods
        traffic_bps, traffic_spawns = player.get_traffic_spawn_info()
        ped_bps, ped_spawns = player.get_pedestrian_spawn_info()
        
        # Set spawned actors (may be fewer than recorded due to spawn failures)
        player.set_traffic_vehicles(spawned_vehicles)
        player.set_pedestrians(spawned_pedestrians)
        
        for frame in range(total_frames):
            player.apply_frame(frame)
            world.tick()
    """
    
    def __init__(self, trajectories_dir: str, scene_id: str):
        """Load trajectory data for a scene.
        
        Args:
            trajectories_dir: Base directory containing split subdirectories with HDF5 files
            scene_id: Scene ID to load
            
        Raises:
            FileNotFoundError: If trajectory file doesn't exist
        """
        self.trajectories_dir = Path(trajectories_dir)
        self.scene_id = scene_id
        self.split = self._determine_split(scene_id)
        
        self.filepath = self.trajectories_dir / self.split / f"{scene_id}.h5"
        if not self.filepath.exists():
            raise FileNotFoundError(
                f"Trajectory file not found: {self.filepath}\n"
                f"Run record_trajectories.py first to generate trajectory data."
            )
        
        # Load data from HDF5
        self._load_data()
        
        # Actor references (set during playback setup)
        self.traffic_vehicles: List = []
        self.pedestrians: List = []
        self.traffic_actor_index_by_id: Dict[int, int] = {}
        self.pedestrian_actor_index_by_id: Dict[int, int] = {}
        self.ego_actor_id: Optional[int] = None
        self.current_frame_idx: Optional[int] = None

        # CARLA client — set via set_client() to enable batched transform application.
        self.client = None

    @classmethod
    def from_file(cls, filepath: str) -> 'TrajectoryPlayer':
        """Load trajectory data directly from an HDF5 file path.

        Alternative constructor when you have a direct path to an HDF5 file
        rather than a trajectories directory + scene_id.

        Args:
            filepath: Direct path to an HDF5 trajectory file

        Returns:
            Initialized TrajectoryPlayer instance

        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(
                f"Trajectory file not found: {path}\n"
                f"Run record_trajectories.py first to generate trajectory data."
            )

        # Peek at metadata to get scene_id for split determination
        with h5py.File(path, 'r') as f:
            scene_id = str(f['metadata'].attrs['scene_id'])

        instance = cls.__new__(cls)
        instance.trajectories_dir = path.parent
        instance.scene_id = scene_id
        instance.split = instance._determine_split(scene_id)
        instance.filepath = path
        instance.traffic_vehicles = []
        instance.pedestrians = []
        instance.traffic_actor_index_by_id = {}
        instance.pedestrian_actor_index_by_id = {}
        instance.ego_actor_id = None
        instance.current_frame_idx = None
        instance.client = None
        instance._load_data()
        return instance

    def set_client(self, client) -> None:
        """Attach a CARLA client to enable batched command dispatch.

        When set, :meth:`apply_frame` and :meth:`apply_ego_frame` send all
        transforms in a single ``apply_batch_sync`` call instead of one
        round-trip per actor.  This eliminates the socket-saturation that
        delays sensor callbacks in synchronous mode.

        Args:
            client: Connected ``carla.Client`` instance.
        """
        self.client = client

    def _determine_split(self, scene_id: str) -> str:
        """Determine which dataset split a scene belongs to based on its ID."""
        if scene_id.startswith("train_") or scene_id.startswith("val_"):
            return "trainval"
        elif scene_id.startswith("test_"):
            return "test"
        elif scene_id.startswith("mini_"):
            return "mini"
        else:
            return "trainval"

    @staticmethod
    def _load_optional_state_group(parent: h5py.Group, group_name: str) -> Dict[str, np.ndarray]:
        if group_name not in parent:
            return {}
        group = parent[group_name]
        return {name: group[name][:] for name in group.keys()}

    @staticmethod
    def _compute_speed_series(state: Dict[str, np.ndarray], key: str) -> Optional[np.ndarray]:
        values = state.get(key)
        if values is None or values.size == 0:
            return None
        return np.linalg.norm(values, axis=-1)

    @staticmethod
    def _clamp_frame_idx(frame_idx: int, total_frames: int) -> int:
        if total_frames <= 0:
            return 0
        return max(0, min(int(frame_idx), total_frames - 1))

    @staticmethod
    def _extract_state_value(
        state: Dict[str, np.ndarray],
        key: str,
        frame_idx: int,
        actor_idx: Optional[int] = None,
    ) -> Any:
        values = state.get(key)
        if values is None or values.size == 0:
            return None
        if actor_idx is None:
            value = values[frame_idx]
        else:
            if values.ndim < 2 or actor_idx >= values.shape[1]:
                return None
            value = values[frame_idx, actor_idx]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value

    @staticmethod
    def _speed_series_flags(speed_series: Optional[np.ndarray], frame_idx: int, fps: float) -> Tuple[bool, bool]:
        if speed_series is None or speed_series.size == 0:
            return False, False
        has_any_motion = bool(np.any(speed_series > REPLAY_MOTION_THRESHOLD_MS))
        radius = max(1, int(round(float(fps))))
        start = max(0, frame_idx - radius)
        end = min(speed_series.shape[0], frame_idx + radius + 1)
        has_nearby_motion = bool(np.any(speed_series[start:end] > REPLAY_MOTION_THRESHOLD_MS))
        return has_any_motion, has_nearby_motion
    
    def _load_data(self):
        """Load trajectory data from HDF5 file."""
        with h5py.File(self.filepath, 'r') as f:
            # Metadata
            meta = f['metadata']
            self.metadata = {
                'scene_id': meta.attrs['scene_id'],
                'map_name': meta.attrs['map_name'],
                'weather': meta.attrs['weather'],
                'traffic_density': meta.attrs['traffic_density'],
                'spawn_point': meta.attrs['spawn_point'],
                'simulation_fps': meta.attrs['simulation_fps'],
                'total_frames': meta.attrs['total_frames'],
                'settle_frames': meta.attrs['settle_frames'],
                'random_seed': meta.attrs['random_seed'],
                'traffic_light_state_codes': meta.attrs.get(
                    'traffic_light_state_codes',
                    'none=-1,green=0,yellow=1,red=2,off=3,unknown=4',
                ),
            }
            
            # Ego vehicle
            ego = f['ego_vehicle']
            self.ego_blueprint_id = ego.attrs['blueprint_id']
            self.ego_locations = ego['locations'][:]
            self.ego_rotations = ego['rotations'][:]
            self.ego_state = self._load_optional_state_group(ego, 'state')
            self.ego_vehicle_context = self._load_optional_state_group(ego, 'vehicle_context')
            self.ego_speed_series = self._compute_speed_series(self.ego_state, 'selected_velocities')
            
            # Traffic vehicles
            traffic = f['traffic_vehicles']
            self.traffic_count = traffic.attrs['count']
            if self.traffic_count > 0:
                self.traffic_blueprint_ids = [bp.decode() if isinstance(bp, bytes) else bp 
                                              for bp in traffic['blueprint_ids'][:]]
                self.traffic_spawn_locations = traffic['spawn_points/locations'][:]
                self.traffic_spawn_rotations = traffic['spawn_points/rotations'][:]
                self.traffic_locations = traffic['locations'][:]
                self.traffic_rotations = traffic['rotations'][:]
                self.traffic_state = self._load_optional_state_group(traffic, 'state')
                self.traffic_vehicle_context = self._load_optional_state_group(traffic, 'vehicle_context')
                self.traffic_speed_series = self._compute_speed_series(
                    self.traffic_state,
                    'selected_velocities',
                )
            else:
                self.traffic_blueprint_ids = []
                self.traffic_spawn_locations = np.array([])
                self.traffic_spawn_rotations = np.array([])
                self.traffic_locations = np.array([])
                self.traffic_rotations = np.array([])
                self.traffic_state = {}
                self.traffic_vehicle_context = {}
                self.traffic_speed_series = None
            
            # Pedestrians
            peds = f['pedestrians']
            self.pedestrian_count = peds.attrs['count']
            if self.pedestrian_count > 0:
                self.pedestrian_blueprint_ids = [bp.decode() if isinstance(bp, bytes) else bp
                                                  for bp in peds['blueprint_ids'][:]]
                self.pedestrian_spawn_locations = peds['spawn_points/locations'][:]
                self.pedestrian_spawn_rotations = peds['spawn_points/rotations'][:]
                self.pedestrian_locations = peds['locations'][:]
                self.pedestrian_rotations = peds['rotations'][:]
                self.pedestrian_state = self._load_optional_state_group(peds, 'state')
                self.pedestrian_speed_series = self._compute_speed_series(
                    self.pedestrian_state,
                    'selected_velocities',
                )
            else:
                self.pedestrian_blueprint_ids = []
                self.pedestrian_spawn_locations = np.array([])
                self.pedestrian_spawn_rotations = np.array([])
                self.pedestrian_locations = np.array([])
                self.pedestrian_rotations = np.array([])
                self.pedestrian_state = {}
                self.pedestrian_speed_series = None
    
    def validate_scene_config(self, scene_config: Dict[str, Any]):
        """Validate that scene config matches recorded trajectory metadata.
        
        Args:
            scene_config: Scene configuration dict to validate
            
        Raises:
            ValueError: If scene config doesn't match trajectory metadata
        """
        errors = []
        
        if scene_config['id'] != self.metadata['scene_id']:
            errors.append(f"Scene ID mismatch: {scene_config['id']} vs {self.metadata['scene_id']}")
        if scene_config['map'] != self.metadata['map_name']:
            errors.append(f"Map mismatch: {scene_config['map']} vs {self.metadata['map_name']}")
        if scene_config['weather'] != self.metadata['weather']:
            errors.append(f"Weather mismatch: {scene_config['weather']} vs {self.metadata['weather']}")
        if scene_config['spawn_point'] != self.metadata['spawn_point']:
            errors.append(f"Spawn point mismatch: {scene_config['spawn_point']} vs {self.metadata['spawn_point']}")
            
        if errors:
            raise ValueError(
                f"Scene config does not match recorded trajectory for {self.scene_id}:\n" +
                "\n".join(f"  - {e}" for e in errors)
            )
    
    def validate_frame_count(self, expected_frames: int):
        """Validate that trajectory has enough frames for the expected recording duration.
        
        Args:
            expected_frames: Expected total number of frames (settle + recording)
            
        Raises:
            ValueError: If trajectory has fewer frames than expected
        """
        actual_frames = self.metadata['total_frames']
        if actual_frames < expected_frames:
            raise ValueError(
                f"Frame count mismatch for scene {self.scene_id}:\n"
                f"  Expected: {expected_frames} frames\n"
                f"  Recorded: {actual_frames} frames\n"
                f"  Settle frames: {self.metadata['settle_frames']}\n"
                f"  Simulation FPS: {self.metadata['simulation_fps']}\n"
                f"  Trajectory file: {self.filepath}\n"
                f"Re-record trajectories with matching duration settings."
            )
    
    def get_traffic_spawn_info(self) -> Tuple[List[str], List[Tuple[List[float], List[float]]]]:
        """Get traffic vehicle blueprint IDs and spawn transforms.
        
        Returns:
            Tuple of (blueprint_ids, spawn_transforms) where spawn_transforms
            is a list of (location, rotation) tuples.
        """
        spawn_transforms = []
        for i in range(self.traffic_count):
            loc = self.traffic_spawn_locations[i].tolist()
            rot = self.traffic_spawn_rotations[i].tolist()
            spawn_transforms.append((loc, rot))
        return self.traffic_blueprint_ids, spawn_transforms
    
    def get_pedestrian_spawn_info(self) -> Tuple[List[str], List[Tuple[List[float], List[float]]]]:
        """Get pedestrian blueprint IDs and spawn transforms.
        
        Returns:
            Tuple of (blueprint_ids, spawn_transforms) where spawn_transforms
            is a list of (location, rotation) tuples.
        """
        spawn_transforms = []
        for i in range(self.pedestrian_count):
            loc = self.pedestrian_spawn_locations[i].tolist()
            rot = self.pedestrian_spawn_rotations[i].tolist()
            spawn_transforms.append((loc, rot))
        return self.pedestrian_blueprint_ids, spawn_transforms
    
    def set_traffic_vehicles(self, vehicles: List):
        """Set spawned traffic vehicles for playback.
        
        Args:
            vehicles: List of spawned CARLA vehicle actors. May contain None
                     for vehicles that failed to spawn.
        """
        self.traffic_vehicles = vehicles
        self.traffic_actor_index_by_id = {
            vehicle.id: idx for idx, vehicle in enumerate(vehicles) if vehicle is not None
        }
        
    def set_pedestrians(self, pedestrians: List):
        """Set spawned pedestrians for playback.
        
        Args:
            pedestrians: List of spawned CARLA walker actors. May contain None
                        for pedestrians that failed to spawn.
        """
        self.pedestrians = pedestrians
        self.pedestrian_actor_index_by_id = {
            pedestrian.id: idx
            for idx, pedestrian in enumerate(pedestrians)
            if pedestrian is not None
        }

    def _compute_walker_control(self, frame_idx: int, ped_idx: int) -> 'carla.WalkerControl':
        """Derive a WalkerControl from position deltas for animation-based replay.

        Instead of teleporting walkers via set_transform(), this method computes
        a WalkerControl from the recorded position delta between consecutive frames.
        The character movement component uses this to play realistic walk/idle
        animations while moving the walker at the correct speed and direction.

        Args:
            frame_idx: Current frame index
            ped_idx: Pedestrian index within the frame

        Returns:
            carla.WalkerControl with appropriate speed and world-space direction
        """
        total_frames = len(self.pedestrian_locations)
        next_frame = min(frame_idx + 1, total_frames - 1)

        curr_loc = self.pedestrian_locations[frame_idx][ped_idx]
        next_loc = self.pedestrian_locations[next_frame][ped_idx]

        delta_t = 1.0 / float(self.metadata['simulation_fps'])
        dx = float(next_loc[0] - curr_loc[0]) / delta_t
        dy = float(next_loc[1] - curr_loc[1]) / delta_t

        # Horizontal speed drives animation; Z is handled by character physics
        speed = float(np.sqrt(dx * dx + dy * dy))

        WALK_THRESHOLD = 0.05  # m/s — below this, use idle animation

        if speed > WALK_THRESHOLD:
            direction = carla.Vector3D(dx / speed, dy / speed, 0.0)
        else:
            # Idle: face the recorded yaw so the walker doesn't spin
            rot = self.pedestrian_rotations[frame_idx][ped_idx]
            yaw_rad = float(rot[1]) * np.pi / 180.0
            direction = carla.Vector3D(float(np.cos(yaw_rad)), float(np.sin(yaw_rad)), 0.0)
            speed = 0.0

        return carla.WalkerControl(direction=direction, speed=speed, jump=False)

    def set_ego_vehicle(self, ego_vehicle):
        """Set the ego vehicle for trajectory playback.
        
        The ego vehicle will follow the recorded trajectory path, even if it's
        a different vehicle type than the one used during recording.
        
        Args:
            ego_vehicle: CARLA ego vehicle actor to control
        """
        self.ego_vehicle = ego_vehicle
        self.ego_actor_id = getattr(ego_vehicle, 'id', None)

    def get_actor_motion_state(self, actor: Any, frame_idx: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Return recorded motion/context for a replayed actor at *frame_idx*."""
        if actor is None:
            return None

        if frame_idx is None:
            frame_idx = self.current_frame_idx
        if frame_idx is None:
            return None

        total_frames = int(self.metadata.get('total_frames', 0))
        frame_idx = self._clamp_frame_idx(frame_idx, total_frames)

        actor_id = getattr(actor, 'id', None)
        domain: Optional[str] = None
        actor_idx: Optional[int] = None
        state: Dict[str, np.ndarray] = {}
        context: Dict[str, np.ndarray] = {}
        speed_series: Optional[np.ndarray] = None

        if actor_id == self.ego_actor_id:
            domain = 'ego'
            state = self.ego_state
            context = self.ego_vehicle_context
            speed_series = self.ego_speed_series
        elif actor_id in self.traffic_actor_index_by_id:
            domain = 'traffic'
            actor_idx = self.traffic_actor_index_by_id[actor_id]
            state = self.traffic_state
            context = self.traffic_vehicle_context
            speed_series = (
                self.traffic_speed_series[:, actor_idx]
                if self.traffic_speed_series is not None and actor_idx < self.traffic_speed_series.shape[1]
                else None
            )
        elif actor_id in self.pedestrian_actor_index_by_id:
            domain = 'pedestrian'
            actor_idx = self.pedestrian_actor_index_by_id[actor_id]
            state = self.pedestrian_state
            speed_series = (
                self.pedestrian_speed_series[:, actor_idx]
                if self.pedestrian_speed_series is not None and actor_idx < self.pedestrian_speed_series.shape[1]
                else None
            )
        else:
            return None

        selected_velocity = self._extract_state_value(state, 'selected_velocities', frame_idx, actor_idx)
        if selected_velocity is None:
            return None

        speed = float(np.linalg.norm(np.array(selected_velocity, dtype=np.float32)))
        has_any_motion, has_nearby_motion = self._speed_series_flags(
            speed_series,
            frame_idx,
            float(self.metadata.get('simulation_fps', 0.0) or 0.0),
        )

        return {
            'domain': domain,
            'frame_idx': frame_idx,
            'actor_index': actor_idx,
            'selected_velocity': selected_velocity,
            'raw_velocity': self._extract_state_value(state, 'raw_velocities', frame_idx, actor_idx),
            'estimated_velocity': self._extract_state_value(
                state,
                'estimated_velocities',
                frame_idx,
                actor_idx,
            ),
            'acceleration': self._extract_state_value(state, 'accelerations', frame_idx, actor_idx),
            'angular_velocity': self._extract_state_value(
                state,
                'angular_velocities',
                frame_idx,
                actor_idx,
            ),
            'used_estimated_velocity': bool(
                self._extract_state_value(state, 'used_estimated_velocities', frame_idx, actor_idx) or False
            ),
            'speed': speed,
            'has_any_motion': has_any_motion,
            'has_nearby_motion': has_nearby_motion,
            'speed_limit': self._extract_state_value(context, 'speed_limits', frame_idx, actor_idx),
            'is_at_traffic_light': bool(
                self._extract_state_value(context, 'is_at_traffic_light', frame_idx, actor_idx) or False
            ),
            'traffic_light_state_code': self._extract_state_value(
                context,
                'traffic_light_state_codes',
                frame_idx,
                actor_idx,
            ),
            'control_throttle': float(
                self._extract_state_value(context, 'control_throttles', frame_idx, actor_idx) or 0.0
            ),
            'control_steer': float(
                self._extract_state_value(context, 'control_steers', frame_idx, actor_idx) or 0.0
            ),
            'control_brake': float(
                self._extract_state_value(context, 'control_brakes', frame_idx, actor_idx) or 0.0
            ),
            'control_hand_brake': bool(
                self._extract_state_value(context, 'control_hand_brakes', frame_idx, actor_idx) or False
            ),
            'control_reverse': bool(
                self._extract_state_value(context, 'control_reverses', frame_idx, actor_idx) or False
            ),
            'control_manual_gear_shift': bool(
                self._extract_state_value(
                    context,
                    'control_manual_gear_shifts',
                    frame_idx,
                    actor_idx,
                )
                or False
            ),
            'control_gear': int(
                self._extract_state_value(context, 'control_gears', frame_idx, actor_idx) or 0
            ),
        }
    
    def get_ego_trajectory_at_frame(self, frame_idx: int) -> Tuple[List[float], List[float]]:
        """Get the recorded ego vehicle position at a specific frame.
        
        Args:
            frame_idx: Frame index (0-based, includes settle frames)
            
        Returns:
            Tuple of (location [x,y,z], rotation [pitch,yaw,roll])
        """
        return self.ego_locations[frame_idx].tolist(), self.ego_rotations[frame_idx].tolist()
    
    def apply_ego_frame(self, frame_idx: int):
        """Apply recorded transform to the ego vehicle for a given frame.

        When a CARLA client is attached (via :meth:`set_client`), the transform
        is dispatched as a single-item batch to avoid extra round-trips.

        Args:
            frame_idx: Frame index (0-based, includes settle frames)
        """
        if carla is None:
            raise RuntimeError("CARLA module not available")

        if not hasattr(self, 'ego_vehicle') or self.ego_vehicle is None:
            return

        self.current_frame_idx = frame_idx

        try:
            loc = self.ego_locations[frame_idx]
            rot = self.ego_rotations[frame_idx]

            transform = carla.Transform(
                carla.Location(x=float(loc[0]), y=float(loc[1]), z=float(loc[2])),
                carla.Rotation(pitch=float(rot[0]), yaw=float(rot[1]), roll=float(rot[2]))
            )
            if self.client is not None:
                self.client.apply_batch_sync(
                    [carla.command.ApplyTransform(self.ego_vehicle, transform)], False
                )
            else:
                self.ego_vehicle.set_transform(transform)
        except RuntimeError as e:
            logger = get_logger()
            logger.debug(f"Could not set transform for ego vehicle: {e}")
    
    def apply_frame(self, frame_idx: int):
        """Apply recorded transforms to all tracked actors for a given frame.

        When a CARLA client is attached (via :meth:`set_client`), ALL vehicle
        and pedestrian commands are packed into a single ``apply_batch_sync``
        call.  This replaces hundreds of individual socket round-trips with
        one, eliminating the sensor-callback delays that caused frame timeouts
        in synchronous-mode recording.

        Note: This only applies transforms to traffic vehicles and pedestrians.
        Call apply_ego_frame() separately to move the ego vehicle.

        Args:
            frame_idx: Frame index (0-based, includes settle frames)
        """
        if carla is None:
            raise RuntimeError("CARLA module not available")

        logger = get_logger()
        self.current_frame_idx = frame_idx

        if self.client is not None:
            # ---- FAST PATH: one batch round-trip for the entire frame ----
            batch = []

            for i, vehicle in enumerate(self.traffic_vehicles):
                if vehicle is None:
                    continue
                if i >= len(self.traffic_locations[frame_idx]):
                    continue
                try:
                    loc = self.traffic_locations[frame_idx][i]
                    rot = self.traffic_rotations[frame_idx][i]
                    transform = carla.Transform(
                        carla.Location(x=float(loc[0]), y=float(loc[1]), z=float(loc[2])),
                        carla.Rotation(pitch=float(rot[0]), yaw=float(rot[1]), roll=float(rot[2]))
                    )
                    batch.append(carla.command.ApplyTransform(vehicle, transform))
                except RuntimeError as e:
                    logger.debug(f"Could not build transform for traffic vehicle {i}: {e}")

            for i, pedestrian in enumerate(self.pedestrians):
                if pedestrian is None:
                    continue
                if i >= len(self.pedestrian_locations[frame_idx]):
                    continue
                try:
                    control = self._compute_walker_control(frame_idx, i)
                    batch.append(carla.command.ApplyWalkerControl(pedestrian, control))

                    if frame_idx % WALKER_POSITION_SNAP_INTERVAL == 0:
                        loc = self.pedestrian_locations[frame_idx][i]
                        rot = self.pedestrian_rotations[frame_idx][i]
                        snap_transform = carla.Transform(
                            carla.Location(x=float(loc[0]), y=float(loc[1]), z=float(loc[2])),
                            carla.Rotation(pitch=float(rot[0]), yaw=float(rot[1]), roll=float(rot[2]))
                        )
                        batch.append(carla.command.ApplyTransform(pedestrian, snap_transform))
                except RuntimeError as e:
                    logger.debug(f"Could not build command for pedestrian {i}: {e}")

            if batch:
                self.client.apply_batch_sync(batch, False)

        else:
            # ---- SLOW FALLBACK: individual calls (no client attached) ----
            for i, vehicle in enumerate(self.traffic_vehicles):
                if vehicle is None:
                    continue
                if i >= len(self.traffic_locations[frame_idx]):
                    continue
                try:
                    loc = self.traffic_locations[frame_idx][i]
                    rot = self.traffic_rotations[frame_idx][i]
                    transform = carla.Transform(
                        carla.Location(x=float(loc[0]), y=float(loc[1]), z=float(loc[2])),
                        carla.Rotation(pitch=float(rot[0]), yaw=float(rot[1]), roll=float(rot[2]))
                    )
                    vehicle.set_transform(transform)
                except RuntimeError as e:
                    logger.debug(f"Could not set transform for traffic vehicle {i}: {e}")

            for i, pedestrian in enumerate(self.pedestrians):
                if pedestrian is None:
                    continue
                if i >= len(self.pedestrian_locations[frame_idx]):
                    continue
                try:
                    control = self._compute_walker_control(frame_idx, i)
                    pedestrian.apply_control(control)

                    if frame_idx % WALKER_POSITION_SNAP_INTERVAL == 0:
                        loc = self.pedestrian_locations[frame_idx][i]
                        rot = self.pedestrian_rotations[frame_idx][i]
                        snap_transform = carla.Transform(
                            carla.Location(x=float(loc[0]), y=float(loc[1]), z=float(loc[2])),
                            carla.Rotation(pitch=float(rot[0]), yaw=float(rot[1]), roll=float(rot[2]))
                        )
                        pedestrian.set_transform(snap_transform)
                except RuntimeError as e:
                    logger.debug(f"Could not control pedestrian {i}: {e}")


def validate_trajectories_exist(trajectories_dir: str, scene_ids: List[str]) -> List[str]:
    """Validate that trajectory files exist for all requested scenes.
    
    Args:
        trajectories_dir: Base directory containing split subdirectories with HDF5 files
        scene_ids: List of scene IDs to validate
        
    Returns:
        List of missing scene IDs (empty if all exist)
    """
    trajectories_path = Path(trajectories_dir)
    missing = []
    
    for scene_id in scene_ids:
        # Determine split from scene ID
        if scene_id.startswith("train_") or scene_id.startswith("val_"):
            split = "trainval"
        elif scene_id.startswith("test_"):
            split = "test"
        elif scene_id.startswith("mini_"):
            split = "mini"
        else:
            split = "trainval"
        
        filepath = trajectories_path / split / f"{scene_id}.h5"
        if not filepath.exists():
            missing.append(scene_id)
    
    return missing
