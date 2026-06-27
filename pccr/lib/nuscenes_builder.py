"""
Main nuScenes dataset builder for CARLA scenes.
Coordinates all components to generate nuScenes-compatible dataset.

Log handling:
- Each scene gets its own log token and log file
- Logs are saved to logs/{scene_name}.json
- log.json in annotation folder contains all logs for the split

CAN bus handling:
- Per-scene files: {scene_name}_pose.json, {scene_name}_ms_imu.json, etc.
- Meta file with statistics: {scene_name}_meta.json
- Empty message types are skipped
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import carla

from .token_manager import TokenManager
from .utils.coordinate_utils import CoordinateConverter
from .can_bus_simulator import CANBusSimulator
from .utils.image_utils import ImageProcessor
from .nuscenes_standards import NuScenesStandards
from .annotation_builder import AnnotationBuilder


class NuScenesDatasetBuilder:
    """Main class for building nuScenes-compatible dataset"""
    
    def __init__(self, rig_name: str, output_dir: str, split: Optional[str] = None,
                 max_detection_distance: float = 80.0, version: str = "v1.0") -> None:
        self.rig_name: str = rig_name
        self.split: Optional[str] = split
        self.version: str = version
        self.output_dir: Path = Path(output_dir) / rig_name
        
        # Initialize components
        self.token_manager: TokenManager = TokenManager()
        self.coordinate_converter: CoordinateConverter = CoordinateConverter()
        self.can_simulator: CANBusSimulator = CANBusSimulator()
        self.image_processor: ImageProcessor = ImageProcessor()
        self.annotation_builder: AnnotationBuilder = AnnotationBuilder(self.token_manager, max_detection_distance)
        
        # nuScenes data tables
        self.scenes: List[Dict[str, Any]] = []
        self.samples: List[Dict[str, Any]] = []
        self.sample_data: List[Dict[str, Any]] = []
        self.ego_poses: List[Dict[str, Any]] = []
        self.sensors: List[Dict[str, Any]] = []
        self.calibrated_sensors: List[Dict[str, Any]] = []
        self.logs: List[Dict[str, Any]] = []
        
        # CAN bus data per scene: {scene_name: {message_type: [entries]}}
        self.can_bus_data_by_scene: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        self.current_scene_name: Optional[str] = None
        
        # Standard nuScenes ontology
        self.categories: List[Dict[str, Any]] = NuScenesStandards.get_categories()
        self.attributes: List[Dict[str, Any]] = NuScenesStandards.get_attributes()
        self.visibility: List[Dict[str, Any]] = NuScenesStandards.get_visibility()
        
        # Sample data linking
        self.sample_data_by_sensor: Dict[str, List[str]] = {}  # Track sample_data by sensor for next/prev linking
        
        # Camera and lidar configuration
        self.camera_configs: Dict[str, Dict[str, Any]] = {}
        self.camera_names: List[str] = []
        self.lidar_config: Optional[Dict[str, Any]] = None
        
        # Current scene tracking
        self.current_scene_token: Optional[str] = None
        self.current_sample_idx: int = 0
        
        # Vehicle state tracking for velocity/acceleration calculation
        self.previous_ego_state: Optional[Dict[str, Any]] = None
        self.previous_timestamp: Optional[float] = None

    @staticmethod
    def _vector3d_from_sequence(values: Optional[List[float]]) -> carla.Vector3D:
        if not values:
            return carla.Vector3D(0.0, 0.0, 0.0)
        return carla.Vector3D(float(values[0]), float(values[1]), float(values[2]))

    @staticmethod
    def _get_ego_trajectory_state(
        ego_vehicle: carla.Vehicle,
        trajectory_player: Optional[Any],
        trajectory_frame_idx: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        if trajectory_player is None or trajectory_frame_idx is None:
            return None
        return trajectory_player.get_actor_motion_state(ego_vehicle, trajectory_frame_idx)
        
    def initialize_dataset(self, camera_configs: List[Dict[str, Any]], lidar_config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize dataset structure and sensor definitions
        
        Args:
            camera_configs: List of camera configuration dictionaries
            lidar_config: Lidar configuration dictionary (optional)
            
        Returns:
            bool: True if successful
        """
        try:
            # Store camera configurations
            for cam_config in camera_configs:
                camera_id = cam_config['id']
                self.camera_configs[camera_id] = cam_config
                self.camera_names.append(camera_id)
            
            # Store lidar configuration if provided
            if lidar_config:
                self.lidar_config = lidar_config
            
            # Create directory structure for cameras and lidar
            sensor_names = self.camera_names.copy()
            if self.lidar_config:
                sensor_names.append(self.lidar_config['id'])
            self.image_processor.create_sample_directories(self.output_dir, sensor_names, self.split)
            
            # Create sensor definitions
            self._create_sensor_definitions()
            
            return True
            
        except Exception as e:
            print(f"Error initializing dataset: {e}")
            return False
    
    def create_scene(self, scene_id: str, map_name: str, weather: str, 
                    description: str = None) -> str:
        """
        Create scene entry
        
        Args:
            scene_id: Unique scene identifier
            map_name: CARLA map name
            weather: Weather condition
            description: Optional scene description
            
        Returns:
            str: Scene token
        """
        scene_token = self.token_manager.scene_token(scene_id)
        log_token = self.token_manager.log_token(scene_id)  # One log per scene
        
        # Create log entry (one per scene now)
        log_entry = {
            "token": log_token,
            "logfile": f"logs/{scene_id}.json",
            "vehicle": "carla_vehicle",
            "date_captured": time.strftime("%Y-%m-%d"),
            "location": map_name,
            "weather": weather
        }
        self.logs.append(log_entry)
        
        # Create scene entry
        scene_entry = {
            "token": scene_token,
            "name": scene_id,
            "description": description or f"CARLA scene {scene_id} on {map_name} with {weather} weather",
            "log_token": log_token,
            "nbr_samples": 0,  # Will be updated when scene is complete
            "first_sample_token": "",  # Will be set with first sample
            "last_sample_token": ""   # Will be updated with each sample
        }
        
        self.scenes.append(scene_entry)
        self.current_scene_token = scene_token
        self.current_scene_name = scene_id
        self.current_sample_idx = 0
        
        # Initialize CAN bus data storage for this scene
        self.can_bus_data_by_scene[scene_id] = {
            "pose": [],
            "ms_imu": [],
            "steeranglefeedback": [],
            "vehicle_monitor": [],
            "meta": {}  # Will be computed at save time
        }
        
        # Reset CAN bus simulator for new scene
        self.can_simulator.reset()
        
        # Reset annotation builder for new scene
        self.annotation_builder.reset()
        
        return scene_token
    
    def create_sample(self, timestamp: float, scene_token: str) -> str:
        """
        Create sample (keyframe) entry
        
        Args:
            timestamp: Timestamp in seconds
            scene_token: Parent scene token
            
        Returns:
            str: Sample token
        """
        self.current_sample_idx += 1
        sample_token = self.token_manager.sample_token(
            scene_token.split('_')[-1], self.current_sample_idx
        )
        
        # Convert timestamp to microseconds
        timestamp_us = int(timestamp * 1e6)
        
        # Create sample entry
        sample_entry = {
            "token": sample_token,
            "timestamp": timestamp_us,
            "scene_token": scene_token,
            "next": "",  # Will be updated with next sample
            "prev": ""   # Will be updated with previous sample
        }
        
        # Link with previous sample
        if len(self.samples) > 0:
            prev_sample = self.samples[-1]
            if prev_sample["scene_token"] == scene_token:
                sample_entry["prev"] = prev_sample["token"]
                prev_sample["next"] = sample_token
        
        self.samples.append(sample_entry)
        
        # Update scene with sample information
        scene = next(s for s in self.scenes if s["token"] == scene_token)
        if scene["first_sample_token"] == "":
            scene["first_sample_token"] = sample_token
        scene["last_sample_token"] = sample_token
        scene["nbr_samples"] = self.current_sample_idx
        
        return sample_token
    
    def create_ego_pose(
        self,
        ego_vehicle: carla.Vehicle,
        timestamp: float,
        trajectory_player: Optional[Any] = None,
        trajectory_frame_idx: Optional[int] = None,
    ) -> str:
        """
        Create ego pose entry
        
        Args:
            ego_vehicle: CARLA vehicle actor
            timestamp: Timestamp in seconds
            trajectory_player: Optional replay trajectory provider for ego state.
            trajectory_frame_idx: Frame index into the replay trajectory.
            
        Returns:
            str: Ego pose token
        """
        timestamp_us = int(timestamp * 1e6)
        # Use scene_name + sample_idx for unique ego pose tokens across scenes
        ego_pose_token = self.token_manager.ego_pose_token(self.current_scene_name, self.current_sample_idx)
        trajectory_state = self._get_ego_trajectory_state(
            ego_vehicle,
            trajectory_player,
            trajectory_frame_idx,
        )
        
        # Get vehicle state
        transform = ego_vehicle.get_transform()
        if trajectory_state is not None:
            velocity = self._vector3d_from_sequence(trajectory_state.get("selected_velocity"))
            angular_velocity = self._vector3d_from_sequence(trajectory_state.get("angular_velocity"))
            acceleration = CoordinateConverter.carla_to_nuscenes_velocity(
                self._vector3d_from_sequence(trajectory_state.get("acceleration"))
            )
        else:
            velocity = ego_vehicle.get_velocity()
            angular_velocity = ego_vehicle.get_angular_velocity()
            acceleration = None
        
        # If velocity is reported as zero but we know position changed, estimate velocity
        if trajectory_state is None and self.previous_ego_state is not None:
            dt = timestamp - self.previous_timestamp
            if dt > 0:
                # Calculate position change
                prev_transform = self.previous_ego_state["transform"]
                
                # Calculate velocity from position change if CARLA velocity is zero
                dx = transform.location.x - prev_transform.location.x
                dy = transform.location.y - prev_transform.location.y
                dz = transform.location.z - prev_transform.location.z
                
                position_velocity = carla.Vector3D(dx/dt, dy/dt, dz/dt)
                
                # Use position-based velocity if CARLA velocity is essentially zero
                speed_carla = (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
                speed_position = (position_velocity.x**2 + position_velocity.y**2 + position_velocity.z**2)**0.5
                
                if speed_carla < 0.1 and speed_position > 0.1:
                    # Use position-based velocity instead
                    velocity = position_velocity
        
        # Convert to nuScenes format
        pose_data = self.coordinate_converter.ego_pose_to_nuscenes(transform, velocity)
        
        # Calculate acceleration if we have previous state
        if acceleration is None:
            acceleration = self._calculate_ego_acceleration(velocity, timestamp)
        angular_acceleration = self._calculate_ego_angular_acceleration(angular_velocity, timestamp)
        
        # Create ego pose entry
        ego_pose_entry = {
            "token": ego_pose_token,
            "timestamp": timestamp_us,
            "translation": pose_data["translation"],
            "rotation": pose_data["rotation"],
            "velocity": pose_data.get("velocity", [0.0, 0.0, 0.0]),
            "acceleration": acceleration,
            "angular_velocity": self.coordinate_converter.carla_to_nuscenes_velocity(angular_velocity),
            "angular_acceleration": angular_acceleration
        }
        
        self.ego_poses.append(ego_pose_entry)
        
        # Store current state for next calculation
        self.previous_ego_state = {
            "velocity": velocity,
            "angular_velocity": angular_velocity,
            "transform": transform
        }
        self.previous_timestamp = timestamp
        
        return ego_pose_token
    
    def create_sample_data(self, sensor_id: str, filename: str, sample_token: str, 
                          ego_pose_token: str, timestamp: float) -> str:
        """
        Create sample data entry for camera or lidar
        
        Args:
            sensor_id: Sensor identifier (camera or lidar)
            filename: Relative path to sensor file
            sample_token: Parent sample token
            ego_pose_token: Associated ego pose token
            timestamp: Timestamp in seconds
            
        Returns:
            str: Sample data token
        """
        # Use scene_name for unique tokens across scenes
        sample_data_token = self.token_manager.sample_data_token(self.current_scene_name, sensor_id, self.current_sample_idx)
        sensor_token = self.token_manager.sensor_token(sensor_id)
        calibrated_sensor_token = self.token_manager.calibrated_sensor_token(sensor_id, self.rig_name)
        
        # Convert timestamp to microseconds
        timestamp_us = int(timestamp * 1e6)
        
        # Determine file format and dimensions based on sensor type
        if sensor_id == "LIDAR_TOP":
            fileformat = "pcd.bin"
            width = None  # Lidar doesn't have image dimensions
            height = None
        else:
            # Camera sensor
            fileformat = "jpg"
            # Get camera config for dimensions
            camera_config = self.camera_configs.get(sensor_id, {})
            width = camera_config.get('image_size_x', 800)
            height = camera_config.get('image_size_y', 600)
        
        # Create sample data entry
        sample_data_entry = {
            "token": sample_data_token,
            "sample_token": sample_token,
            "ego_pose_token": ego_pose_token,
            "calibrated_sensor_token": calibrated_sensor_token,
            "timestamp": timestamp_us,
            "fileformat": fileformat,
            "filename": filename,
            "is_key_frame": True,
            "next": "",
            "prev": ""
        }
        
        # Add dimensions for cameras only
        if width is not None and height is not None:
            sample_data_entry["width"] = width
            sample_data_entry["height"] = height
        
        # Link with previous sample_data from same sensor
        if sensor_id in self.sample_data_by_sensor:
            prev_sample_data = self.sample_data_by_sensor[sensor_id]
            sample_data_entry["prev"] = prev_sample_data["token"]
            prev_sample_data["next"] = sample_data_token
        
        # Store current sample_data for next linking
        self.sample_data_by_sensor[sensor_id] = sample_data_entry
        
        self.sample_data.append(sample_data_entry)
        
        return sample_data_token
    
    def add_can_bus_data(
        self,
        ego_vehicle: carla.Vehicle,
        timestamp: float,
        trajectory_player: Optional[Any] = None,
        trajectory_frame_idx: Optional[int] = None,
    ) -> None:
        """
        Add CAN bus data entry for current scene
        
        Args:
            ego_vehicle: CARLA vehicle actor
            timestamp: Timestamp in seconds
            trajectory_player: Optional replay trajectory provider for ego state.
            trajectory_frame_idx: Frame index into the replay trajectory.
        """
        if self.current_scene_name is None:
            return

        trajectory_state = self._get_ego_trajectory_state(
            ego_vehicle,
            trajectory_player,
            trajectory_frame_idx,
        )
            
        # Generate all CAN message types
        messages = self.can_simulator.generate_all_can_messages(
            ego_vehicle,
            timestamp,
            trajectory_state=trajectory_state,
        )
        
        # Append each message type to the appropriate list for this scene
        scene_can_data = self.can_bus_data_by_scene.get(self.current_scene_name)
        if scene_can_data:
            for msg_type in ["pose", "ms_imu", "steeranglefeedback", "vehicle_monitor"]:
                if msg_type in messages and msg_type in scene_can_data:
                    scene_can_data[msg_type].append(messages[msg_type])
    
    def create_annotations(self, world: carla.World, sample_token: str, ego_vehicle: carla.Vehicle, 
                          cameras: List[carla.Actor], lidar_data: Optional[Dict[str, Any]],
                          lidar_sensor: Optional[carla.Actor] = None,
                          camera_depth_data: Optional[Dict[str, Any]] = None,
                          frame_number: int = 0,
                          trajectory_player: Optional[Any] = None,
                          trajectory_frame_idx: Optional[int] = None) -> List[str]:
        """
        Create 3D bounding box annotations for all objects in the scene
        
        Args:
            world: CARLA world object
            sample_token: Sample token for these annotations
            ego_vehicle: Ego vehicle actor
            cameras: List of camera sensors
            lidar_data: Current lidar measurement data
            lidar_sensor: LiDAR sensor actor for coordinate transformation
            camera_depth_data: Per-camera decoded depth data for visibility
            frame_number: Current frame number for logging
            trajectory_player: Optional replay trajectory provider with recorded
                per-actor motion/context state.
            trajectory_frame_idx: Frame index into the replay trajectory.
            
        Returns:
            List[str]: List of annotation tokens created
        """
        return self.annotation_builder.process_scene_objects(
            world, sample_token, ego_vehicle, cameras, lidar_data, lidar_sensor,
            camera_depth_data=camera_depth_data,
            frame_number=frame_number,
            trajectory_player=trajectory_player,
            trajectory_frame_idx=trajectory_frame_idx,
        )
    
    def save_annotations(self) -> bool:
        """
        Save all annotation files to disk in proper nuScenes directory structure
        
        Returns:
            bool: True if successful
        """
        try:
            # Determine annotation directory based on split (nuScenes standard {version}-{split} format)
            if self.split in ["train", "val", "trainval"]:
                annotations_dir = self.output_dir / f"{self.version}-trainval"
            elif self.split == "test":
                annotations_dir = self.output_dir / f"{self.version}-test"
            elif self.split == "mini":
                annotations_dir = self.output_dir / f"{self.version}-mini"
            else:
                # Fallback to trainval if split not specified
                annotations_dir = self.output_dir / f"{self.version}-trainval"
                
            annotations_dir.mkdir(exist_ok=True)
            
            # Create separate directories for CAN bus and logs (as in original nuScenes)
            can_bus_dir = self.output_dir / "can_bus"
            logs_dir = self.output_dir / "logs"
            can_bus_dir.mkdir(exist_ok=True)
            logs_dir.mkdir(exist_ok=True)
            
            # Create map and log data
            self._create_map_and_log_data()
            
            # Save core annotation files in proper split directory
            annotation_files = {
                "scene.json": self.scenes,
                "sample.json": self.samples,
                "sample_data.json": self.sample_data,
                "ego_pose.json": self.ego_poses,
                "sensor.json": self.sensors,
                "calibrated_sensor.json": self.calibrated_sensors,
                "category.json": self.categories,
                "attribute.json": self.attributes,
                "visibility.json": self.visibility,
                "log.json": self.logs,
                "map.json": self._create_map_data(),
                "sample_annotation.json": self.annotation_builder.get_annotations(),
                "instance.json": self.annotation_builder.get_instances()
            }
            
            for filename, data in annotation_files.items():
                file_path = annotations_dir / filename
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
            
            # Save CAN bus data in separate folder (per-scene files)
            if self.can_bus_data_by_scene:
                self._save_can_bus_data(can_bus_dir)
            
            # Save log files in logs folder (individual log files)
            if self.logs:
                self._save_log_files(logs_dir)
            
            # Create additional required files
            additional_types = []
            additional_types_path = annotations_dir / "additional_types.json"
            with open(additional_types_path, 'w') as f:
                json.dump(additional_types, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error saving annotations: {e}")
            return False
    
    def _create_map_and_log_data(self) -> None:
        """Create map and log data entries"""
        # The logs are already created during scene creation
        # Just ensure we have a current log token for map creation
        if self.logs:
            self.current_log_token = self.logs[-1]["token"]
        else:
            # Fallback if no logs exist yet
            self.current_log_token = self.token_manager.generate_token()
            log_entry = {
                "token": self.current_log_token,
                "logfile": f"{self.rig_name}_{int(time.time())}",
                "vehicle": "carla_vehicle",
                "date_captured": time.strftime("%Y-%m-%d"),
                "location": getattr(self, 'location', 'Town01')
            }
            self.logs = [log_entry]
        
    def _create_map_data(self) -> List[Dict[str, Any]]:
        """Create map data structure grouping logs by location"""
        location_map = {}
        for log in self.logs:
            loc = log.get("location", "unknown")
            if loc not in location_map:
                location_map[loc] = []
            location_map[loc].append(log["token"])

        map_entries = []
        for loc, log_tokens in location_map.items():
            map_entries.append({
                "token": self.token_manager.generate_token(),
                "log_tokens": log_tokens,
                "category": "semantic_prior",
                "filename": "" 
            })
        
        return map_entries
        
    def _save_can_bus_data(self, can_bus_dir: Path) -> None:
        """
        Save CAN bus data as per-scene files matching nuScenes format.
        
        Creates files: {scene_name}_pose.json, {scene_name}_ms_imu.json,
        {scene_name}_steeranglefeedback.json, {scene_name}_vehicle_monitor.json,
        {scene_name}_meta.json
        
        Skips message types with empty data.
        """
        if not self.can_bus_data_by_scene:
            return
        
        message_types = ["pose", "ms_imu", "steeranglefeedback", "vehicle_monitor"]
        
        for scene_name, scene_data in self.can_bus_data_by_scene.items():
            # Compute meta statistics for this scene
            meta = {}
            for msg_type in message_types:
                entries = scene_data.get(msg_type, [])
                if entries:
                    utimes = [e.get("utime", 0) for e in entries]
                    if len(utimes) > 1:
                        duration_us = max(utimes) - min(utimes)
                        freq = len(utimes) / (duration_us / 1e6) if duration_us > 0 else 0
                    else:
                        freq = 0
                    meta[msg_type] = {
                        "count": len(entries),
                        "freq": round(freq, 2)
                    }
            
            # Save meta file if we have any data
            if meta:
                meta_path = can_bus_dir / f"{scene_name}_meta.json"
                with open(meta_path, 'w') as f:
                    json.dump(meta, f, indent=2)
            
            # Save each message type (skip empty ones)
            for msg_type in message_types:
                entries = scene_data.get(msg_type, [])
                if entries:  # Only save non-empty message types
                    file_path = can_bus_dir / f"{scene_name}_{msg_type}.json"
                    with open(file_path, 'w') as f:
                        json.dump(entries, f, indent=2)
            
    def _save_log_files(self, logs_dir: Path) -> None:
        """
        Save individual log files (matching nuScenes format).
        
        Each scene gets its own log file in the logs/ folder with weather
        and scene metadata.
        """
        if not self.logs:
            return
            
        for log_entry in self.logs:
            # Extract scene name from logfile path (logs/{scene_name}.json -> {scene_name})
            logfile = log_entry.get('logfile', '')
            if logfile.startswith('logs/'):
                scene_name = logfile[5:-5]  # Remove 'logs/' prefix and '.json' suffix
            else:
                scene_name = logfile
            
            log_filename = f"{scene_name}.json"
            log_file_path = logs_dir / log_filename
            
            # Find scene and count samples for this log
            scene_token = log_entry["token"]
            matching_scenes = [s for s in self.scenes if s.get("log_token") == scene_token]
            sample_count = sum(s.get("nbr_samples", 0) for s in matching_scenes)
            
            # Create log file content
            log_content = {
                "token": log_entry["token"],
                "vehicle": log_entry["vehicle"],
                "date_captured": log_entry["date_captured"],
                "location": log_entry["location"],
                "scenes": [s["token"] for s in matching_scenes],
                "samples": sample_count,
                "duration": sample_count * 0.5,  # 2 FPS = 0.5s per sample
                "weather": log_entry.get("weather", "")
            }
            
            with open(log_file_path, 'w') as f:
                json.dump(log_content, f, indent=2)
    
    def _create_sensor_definitions(self) -> None:
        """Create sensor and calibrated_sensor entries for cameras and lidar"""
        # Create camera sensor definitions
        for camera_id, cam_config in self.camera_configs.items():
            # Create sensor entry
            sensor_token = self.token_manager.sensor_token(camera_id)
            sensor_entry = {
                "token": sensor_token,
                "channel": camera_id,
                "modality": "camera"
            }
            self.sensors.append(sensor_entry)
            
            # Create calibrated sensor entry (unique per rig)
            calibrated_sensor_token = self.token_manager.calibrated_sensor_token(camera_id, self.rig_name)
            
            # Create transform for camera position
            camera_transform = carla.Transform(
                carla.Location(
                    x=cam_config['location'][0],
                    y=cam_config['location'][1], 
                    z=cam_config['location'][2]
                ),
                carla.Rotation(
                    pitch=cam_config['rotation'][0],
                    yaw=cam_config['rotation'][1],
                    roll=cam_config['rotation'][2]
                )
            )
            
            # Convert to nuScenes format
            camera_pose = self.coordinate_converter.camera_transform_to_nuscenes(camera_transform)
            
            # Apply basis change for camera (Optical -> Vehicle)
            # q_total = q_mount * q_basis
            q_mount = camera_pose["rotation"]
            q_basis = self.coordinate_converter.get_camera_basis_change_rotation()
            camera_pose["rotation"] = self.coordinate_converter.multiply_quaternions(q_mount, q_basis)
            
            # Compute camera intrinsics
            intrinsic = self.coordinate_converter.compute_camera_intrinsic(
                cam_config.get('image_size_x', 800), 
                cam_config.get('image_size_y', 600), 
                cam_config.get('fov', 90)
            )
            
            calibrated_sensor_entry = {
                "token": calibrated_sensor_token,
                "sensor_token": sensor_token,
                "translation": camera_pose["translation"],
                "rotation": camera_pose["rotation"],
                "camera_intrinsic": intrinsic
            }
            self.calibrated_sensors.append(calibrated_sensor_entry)
        
        # Create lidar sensor definition if present
        if self.lidar_config:
            lidar_id = self.lidar_config['id']
            
            # Create lidar sensor entry
            lidar_sensor_token = self.token_manager.sensor_token(lidar_id)
            lidar_sensor_entry = {
                "token": lidar_sensor_token,
                "channel": lidar_id,
                "modality": "lidar"
            }
            self.sensors.append(lidar_sensor_entry)
            
            # Create calibrated lidar sensor entry (unique per rig)
            lidar_calibrated_sensor_token = self.token_manager.calibrated_sensor_token(lidar_id, self.rig_name)
            
            # Create transform for lidar position
            lidar_transform = carla.Transform(
                carla.Location(
                    x=self.lidar_config['location'][0],
                    y=self.lidar_config['location'][1], 
                    z=self.lidar_config['location'][2]
                ),
                carla.Rotation(
                    pitch=self.lidar_config['rotation'][0],
                    yaw=self.lidar_config['rotation'][1],
                    roll=self.lidar_config['rotation'][2]
                )
            )
            
            # Convert to nuScenes format
            lidar_pose = self.coordinate_converter.camera_transform_to_nuscenes(lidar_transform)
            
            lidar_calibrated_sensor_entry = {
                "token": lidar_calibrated_sensor_token,
                "sensor_token": lidar_sensor_token,
                "translation": lidar_pose["translation"],
                "rotation": lidar_pose["rotation"],
                "camera_intrinsic": []  # Empty for lidar
            }
            self.calibrated_sensors.append(lidar_calibrated_sensor_entry)
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get summary information about the dataset"""
        total_can_entries = sum(
            len(scene_data.get("pose", []))
            for scene_data in self.can_bus_data_by_scene.values()
        )
        return {
            "rig_name": self.rig_name,
            "output_dir": str(self.output_dir),
            "num_scenes": len(self.scenes),
            "num_samples": len(self.samples),
            "num_sample_data": len(self.sample_data),
            "num_cameras": len(self.camera_names),
            "camera_names": self.camera_names,
            "total_can_entries": total_can_entries
        }

    def get_sample(self, sample_token: str) -> Optional[Dict[str, Any]]:
        """Return a sample entry by token."""
        return next((sample for sample in self.samples if sample["token"] == sample_token), None)

    def get_ego_pose(self, ego_pose_token: str) -> Optional[Dict[str, Any]]:
        """Return an ego pose entry by token."""
        return next((pose for pose in self.ego_poses if pose["token"] == ego_pose_token), None)

    def get_sample_annotations(self, sample_token: str) -> List[Dict[str, Any]]:
        """Return all annotations belonging to a sample."""
        return [ann for ann in self.annotation_builder.get_annotations() if ann["sample_token"] == sample_token]

    def get_sample_data_for_sample(self, sample_token: str) -> List[Dict[str, Any]]:
        """Return all sample_data entries belonging to a sample."""
        return [sample_data for sample_data in self.sample_data if sample_data["sample_token"] == sample_token]

    def get_sensor(self, sensor_id: str) -> Optional[Dict[str, Any]]:
        """Return a sensor entry by channel name."""
        sensor_token = self.token_manager.sensor_token(sensor_id)
        return next((sensor for sensor in self.sensors if sensor["token"] == sensor_token), None)

    def get_calibrated_sensor(self, sensor_id: str) -> Optional[Dict[str, Any]]:
        """Return a calibrated_sensor entry by channel name."""
        calibrated_sensor_token = self.token_manager.calibrated_sensor_token(sensor_id, self.rig_name)
        return next(
            (
                calibrated_sensor
                for calibrated_sensor in self.calibrated_sensors
                if calibrated_sensor["token"] == calibrated_sensor_token
            ),
            None,
        )

    def get_instances_by_tokens(self, instance_tokens: List[str]) -> List[Dict[str, Any]]:
        """Return instance entries for the provided instance tokens."""
        instance_token_set = set(instance_tokens)
        return [instance for instance in self.annotation_builder.get_instances() if instance["token"] in instance_token_set]
    
    def _calculate_ego_acceleration(self, current_velocity: carla.Vector3D, timestamp: float) -> List[float]:
        """Calculate ego vehicle acceleration from velocity change"""
        if self.previous_ego_state is None or self.previous_timestamp is None:
            return [0.0, 0.0, 0.0]
        
        dt = timestamp - self.previous_timestamp
        if dt <= 0:
            return [0.0, 0.0, 0.0]
        
        prev_vel = self.previous_ego_state["velocity"]
        
        # Calculate acceleration in CARLA coordinates then convert
        accel_carla = carla.Vector3D(
            (current_velocity.x - prev_vel.x) / dt,
            (current_velocity.y - prev_vel.y) / dt,
            (current_velocity.z - prev_vel.z) / dt
        )
        
        return self.coordinate_converter.carla_to_nuscenes_velocity(accel_carla)
    
    def _calculate_ego_angular_acceleration(self, current_angular_velocity: carla.Vector3D, timestamp: float) -> List[float]:
        """Calculate ego vehicle angular acceleration from angular velocity change"""
        if self.previous_ego_state is None or self.previous_timestamp is None:
            return [0.0, 0.0, 0.0]
        
        dt = timestamp - self.previous_timestamp
        if dt <= 0:
            return [0.0, 0.0, 0.0]
        
        prev_angular_vel = self.previous_ego_state["angular_velocity"]
        
        # Calculate angular acceleration in CARLA coordinates then convert
        angular_accel_carla = carla.Vector3D(
            (current_angular_velocity.x - prev_angular_vel.x) / dt,
            (current_angular_velocity.y - prev_angular_vel.y) / dt,
            (current_angular_velocity.z - prev_angular_vel.z) / dt
        )
        
        return self.coordinate_converter.carla_to_nuscenes_velocity(angular_accel_carla)