"""
Annotation builder for generating nuScenes-format annotation JSON.
Handles instance tracking across frames and temporal linking of annotations.
"""

import logging
import time
from typing import Dict, List, Any, Optional
import numpy as np
from .token_manager import TokenManager
from .detection import ObjectDetector, DetectedObject, LightHeadProxy
from .bbox import BoundingBoxBuilder, TrafficInfrastructureBBox
from .utils.visibility_utils import VisibilityCalculator
from .utils.attribute_utils import AttributeClassifier
from .utils.coordinate_utils import CoordinateConverter
from .nuscenes_standards import (
    CAT_CAR, CAT_TRAFFIC_LIGHT, CAT_TRAFFIC_SIGN, CAT_ADULT, CAT_CHILD,
    ATTR_VEHICLE_STOPPED, NuScenesStandards
)
import carla


class AnnotationBuilder:
    """Generate nuScenes annotation and instance data with temporal tracking."""
    
    def __init__(self, token_manager: TokenManager, max_detection_distance: float = 80.0) -> None:
        """
        Initialize annotation builder.
        
        Args:
            token_manager: Token generator for unique identifiers
            max_detection_distance: Maximum distance for object detection (kept for compatibility)
        """
        self.token_manager: TokenManager = token_manager
        self.max_detection_distance: float = max_detection_distance
        
        # Store generated annotations and instances
        self.annotations: List[Dict[str, Any]] = []
        self.instances: List[Dict[str, Any]] = []
        
        # Track instances for temporal linking
        self.instance_history: Dict[str, List[str]] = {}  # instance_token -> [annotation_tokens]
        
        # Map actor ID to instance token for consistent tracking across frames
        self.actor_instance_map: Dict[str, str] = {}
        
        # Helpers
        self.visibility_calculator = None # Initialized in process_scene_objects
        
        # Build category token -> name lookup
        self._category_names = {cat['token']: cat['name'] for cat in NuScenesStandards.get_categories()}
        
        # Frame counter for logging
        self._current_frame = 0
        self._profile_log_interval = 10
    
    def create_annotation(self, sample_token: str, 
                         category_token: str,
                         instance_token: str,
                         translation: List[float],
                         size: List[float],
                         rotation: List[float],
                         visibility_token: str,
                         attribute_token: Optional[str],
                         num_lidar_pts: int) -> str:
        """
        Create a real annotation entry in nuScenes format.
        
        Args:
            sample_token: Token of the sample this annotation belongs to
            category_token: Category token
            instance_token: Instance token
            translation: [x, y, z] in nuScenes coordinates
            size: [width, length, height]
            rotation: [w, x, y, z] quaternion in nuScenes coordinates
            visibility_token: Visibility token
            attribute_token: Attribute token, or None for static infrastructure
                             (will produce an empty attribute_tokens list)
            num_lidar_pts: Number of lidar points hitting the object
            
        Returns:
            Annotation token
        """
        # Generate annotation token
        annotation_token = self.token_manager.generate_token()
        
        # Link with previous annotation from same instance
        prev_annotation_token = ""
        if instance_token in self.instance_history and self.instance_history[instance_token]:
            prev_annotation_token = self.instance_history[instance_token][-1]
            # Update previous annotation's next field
            for ann in self.annotations:
                if ann["token"] == prev_annotation_token:
                    ann["next"] = annotation_token
                    break
        
        # Create annotation entry
        annotation_entry = {
            "token": annotation_token,
            "sample_token": sample_token,
            "instance_token": instance_token,
            "attribute_tokens": [attribute_token] if attribute_token is not None else [],
            "visibility_token": visibility_token,
            "translation": translation,
            "size": size,
            "rotation": rotation,
            "prev": prev_annotation_token,
            "next": "",
            "num_lidar_pts": num_lidar_pts,
            "num_radar_pts": 0
        }
        
        self.annotations.append(annotation_entry)
        
        # Track annotation for this instance
        if instance_token not in self.instance_history:
            self.instance_history[instance_token] = []
        self.instance_history[instance_token].append(annotation_token)
        
        # Update instance entry
        for inst in self.instances:
            if inst["token"] == instance_token:
                if inst["first_annotation_token"] == "":
                    inst["first_annotation_token"] = annotation_token
                inst["last_annotation_token"] = annotation_token
                inst["nbr_annotations"] += 1
                break
        
        return annotation_token
    
    def create_instance(self, category_token: str) -> str:
        """
        Create an instance entry in nuScenes format.
        
        Instances represent unique object identities tracked across frames.
        Each instance links to its first/last annotations and tracks count.
        
        Args:
            category_token: Category token for the instance
            
        Returns:
            Instance token
        """
        instance_token = self.token_manager.generate_token()
        
        instance_entry = {
            "token": instance_token,
            "category_token": category_token,
            "first_annotation_token": "",
            "last_annotation_token": "",
            "nbr_annotations": 0
        }
        
        self.instances.append(instance_entry)
        self.instance_history[instance_token] = []
        
        return instance_token
    
    def process_scene_objects(self, world: Any, sample_token: str, ego_vehicle: Any, 
                            cameras: List[Any], lidar_data: Optional[Any], 
                            lidar_sensor: Optional[Any] = None,
                            camera_depth_data: Optional[Dict[str, Any]] = None,
                            frame_number: int = 0,
                            trajectory_player: Optional[Any] = None,
                            trajectory_frame_idx: Optional[int] = None) -> List[str]:
        """
        Process scene objects and create annotations with instance tracking.
        
        Detects objects in proximity, calculates bounding boxes, counts LiDAR points,
        determines visibility, and creates nuScenes-format annotations. Instances are
        tracked across frames using CARLA actor IDs.
        
        Args:
            world: CARLA world object
            sample_token: Token of the sample
            ego_vehicle: Ego vehicle actor
            cameras: List of camera actors
            lidar_data: LiDAR measurement data for point counting
            lidar_sensor: LiDAR sensor actor for coordinate transformation
            camera_depth_data: Per-camera decoded depth data for visibility.
                Dict of camera_id -> {depth_map, transform, fx, fy, cx, cy, width, height}
            frame_number: Current frame number for logging
            trajectory_player: Optional replay trajectory provider with recorded
                per-actor motion/context state.
            trajectory_frame_idx: Frame index into the replay trajectory.
            
        Returns:
            List of annotation tokens created for this sample
        """
        logger = logging.getLogger('scenario_runner')
        self._current_frame = frame_number
        
        # Initialize or update visibility calculator
        # Must recreate if ego_vehicle or world changed (e.g., new scene or map reload)
        # Compare by actor ID since object references won't match across scenes
        if (self.visibility_calculator is None or 
            self.visibility_calculator.ego_vehicle_id != ego_vehicle.id or
            self.visibility_calculator.world != world):
            self.visibility_calculator = VisibilityCalculator(world, ego_vehicle)
            
        detection_start = time.perf_counter()
        # Step 1: Get objects in proximity using distance-based filtering
        objects_in_proximity = ObjectDetector.get_objects_of_interest(
            world, 
            ego_vehicle, 
            self.max_detection_distance
        )
        detection_ms = (time.perf_counter() - detection_start) * 1000.0

        lidar_prep_start = time.perf_counter()
        lidar_points_world = self._prepare_lidar_points_world(lidar_data, lidar_sensor)
        lidar_prep_ms = (time.perf_counter() - lidar_prep_start) * 1000.0
        
        # Get summary statistics
        summary = ObjectDetector.get_detection_summary(objects_in_proximity)
        
        # Log detection summary with readable category names
        if objects_in_proximity:
            readable_summary = {self._category_names.get(cat, cat[:8]): count for cat, count in summary.items()}
            summary_str = ", ".join([f"{count} {cat}" for cat, count in readable_summary.items()])
            logger.debug(f"[Frame {self._current_frame}] Found {len(objects_in_proximity)} objects: {summary_str}")
        else:
            logger.debug(f"[Frame {self._current_frame}] No objects within {self.max_detection_distance}m")
        
        # Step 2: For each object in proximity, calculate bounding box and count lidar points
        annotation_tokens = []
        lidar_count_ms = 0.0
        visibility_ms = 0.0
        object_loop_start = time.perf_counter()
        
        if objects_in_proximity:
            pass  # Removed verbose "Processing" log line
            
            for obj in objects_in_proximity:
                try:
                    actor = obj.actor
                    
                    # 1. Calculate Bounding Box
                    if isinstance(actor, LightHeadProxy):
                        # Each light head proxy carries its own pre-positioned local bbox.
                        bbox_info = BoundingBoxBuilder.get_actor_bbox_info(actor, source="head_proxy")
                    elif getattr(actor, 'source', None) == 'env_sign':
                        # Env-sign proxies carry the correct world transform and
                        # zero-centred bbox; bypass the spawned-actor infrastructure
                        # path which would apply wrong generic dimensions.
                        bbox_info = BoundingBoxBuilder.get_actor_bbox_info(actor, source="env_sign")
                    elif obj.category_token in (CAT_TRAFFIC_LIGHT, CAT_TRAFFIC_SIGN):
                        bbox_info = self._build_infrastructure_bbox(actor, obj.category_token)
                    else:
                        bbox_info = BoundingBoxBuilder.get_actor_bbox_info(actor)

                    bbox_corners = bbox_info.corners
                    extent = bbox_info.extent
                    bbox_transform = bbox_info.transform
                    location = bbox_transform.location
                    rotation = bbox_transform.rotation
                    
                    # 2. Count LiDAR points
                    num_lidar_pts = 0
                    if lidar_points_world is not None:
                        lidar_count_start = time.perf_counter()
                        num_lidar_pts = self._count_lidar_points_in_bbox(
                            lidar_points_world, bbox_corners
                        )
                        lidar_count_ms += (time.perf_counter() - lidar_count_start) * 1000.0
                    
                    # 3. Calculate Visibility
                    visibility_start = time.perf_counter()
                    visibility_token = self.visibility_calculator.calculate_visibility(
                            actor,
                            camera_depth_data=camera_depth_data,
                            category_token=obj.category_token,
                            bbox_extent=extent,
                            bbox_location=location,
                            bbox_transform=bbox_transform
                        )
                    visibility_ms += (time.perf_counter() - visibility_start) * 1000.0
                    
                    # Filter out fully occluded objects
                    if visibility_token == "0":
                        continue
                    
                    # 4. Determine Attribute
                    trajectory_state = None
                    if trajectory_player is not None and trajectory_frame_idx is not None:
                        trajectory_state = trajectory_player.get_actor_motion_state(
                            actor,
                            trajectory_frame_idx,
                        )
                    attribute_token = AttributeClassifier.get_attribute(
                        actor,
                        obj.category_token,
                        trajectory_state=trajectory_state,
                    )
                    
                    # 5. Convert to nuScenes coordinates
                    # Translation: [x, y, z]
                    nuscenes_translation = CoordinateConverter.carla_to_nuscenes_location(location)
                    
                    # Rotation: [w, x, y, z]
                    nuscenes_rotation = CoordinateConverter.carla_to_nuscenes_rotation(rotation)
                    
                    # Size: [width, length, height]
                    # CARLA extent is half-size. nuScenes wants full size.
                    # CARLA: x=forward (length/2), y=right (width/2), z=up (height/2)
                    # nuScenes: width, length, height
                    # So: width = 2*extent.y, length = 2*extent.x, height = 2*extent.z
                    nuscenes_size = [
                        float(extent.y * 2), # width
                        float(extent.x * 2), # length
                        float(extent.z * 2)  # height
                    ]
                    
                    # 6. Manage Instance Token (track same actor across frames)
                    actor_tracking_key = f"{getattr(actor, 'source', 'actor')}:{actor.id}"
                    if actor_tracking_key not in self.actor_instance_map:
                        instance_token = self.create_instance(obj.category_token)
                        self.actor_instance_map[actor_tracking_key] = instance_token
                    else:
                        instance_token = self.actor_instance_map[actor_tracking_key]
                    
                    # 7. Create Annotation
                    annotation_token = self.create_annotation(
                        sample_token,
                        obj.category_token,
                        instance_token,
                        nuscenes_translation,
                        nuscenes_size,
                        nuscenes_rotation,
                        visibility_token,
                        attribute_token,
                        num_lidar_pts
                    )
                    
                    annotation_tokens.append(annotation_token)
                    
                    # Improved debug log with readable names
                    cat_name = self._category_names.get(obj.category_token, "unknown")
                    logger.debug(
                        f"  + {cat_name} id={actor.id} dist={obj.distance:.1f}m "
                        f"lidar={num_lidar_pts} vis={visibility_token} "
                        f"size=[{nuscenes_size[0]:.2f},{nuscenes_size[1]:.2f},{nuscenes_size[2]:.2f}] "
                        f"bbox_source={bbox_info.source}"
                    )
                    
                except Exception as e:
                    logger.warning(f"[Frame {self._current_frame}] Error processing actor {obj.actor.id}: {e}")

        object_loop_ms = (time.perf_counter() - object_loop_start) * 1000.0
        if self._current_frame % self._profile_log_interval == 0:
            logger.info(
                f"[Frame {self._current_frame}] Annotation perf: "
                f"detect={detection_ms:.1f}ms lidar_prep={lidar_prep_ms:.1f}ms "
                f"loop={object_loop_ms:.1f}ms lidar_count={lidar_count_ms:.1f}ms "
                f"visibility={visibility_ms:.1f}ms objs={len(objects_in_proximity)} "
                f"anns={len(annotation_tokens)}"
            )
        
        return annotation_tokens

    def _build_infrastructure_bbox(self, actor: Any, category_token: str):
        """
        Construct bounding box for traffic lights/signs - mirrors debug drawer logic exactly.
        
        Returns:
            Shared bounding-box descriptor
        """
        logger = logging.getLogger('scenario_runner')
        bbox_info = TrafficInfrastructureBBox.get_annotation_bbox_info(actor, category_token)
        logger.debug(
            f"  [infra-bbox] {self._category_names.get(category_token, 'unknown')} "
            f"id={actor.id} source={bbox_info.source}"
        )
        return bbox_info

    def _corners_to_bbox_params(self, corners: List[carla.Location], actor: Any):
        """
        Convert 8 world-space corners to bbox parameters (extent, transform).
        """
        # Compute center from corners
        center = self._average_corner_location(corners)
        
        # Compute extent from corner spread
        corners_np = np.array([[c.x, c.y, c.z] for c in corners], dtype=np.float32)
        min_pt = corners_np.min(axis=0)
        max_pt = corners_np.max(axis=0)
        extent_vec = (max_pt - min_pt) / 2.0
        # Ensure minimum thickness
        extent_vec = np.maximum(extent_vec, np.array([0.05, 0.05, 0.1], dtype=np.float32))
        
        extent = carla.Vector3D(
            float(extent_vec[0]),
            float(extent_vec[1]),
            float(extent_vec[2])
        )
        
        # Use actor's rotation for orientation
        rotation = actor.get_transform().rotation
        bbox_transform = carla.Transform(center, rotation)
        
        return corners, extent, bbox_transform

    @staticmethod
    def _average_corner_location(corners: List[carla.Location]) -> carla.Location:
        if not corners:
            return carla.Location(0.0, 0.0, 0.0)
        count = float(len(corners))
        sum_x = sum(corner.x for corner in corners)
        sum_y = sum(corner.y for corner in corners)
        sum_z = sum(corner.z for corner in corners)
        return carla.Location(sum_x / count, sum_y / count, sum_z / count)

    def _should_force_visibility(self, category_token: str, distance: float, num_lidar_pts: int) -> bool:
        """Decide whether to override zero-visibility filtering for near objects."""
        near_ped_distance = 1.5  # meters
        lidar_recovery_distance = 2.5  # meters

        if category_token != CAT_PEDESTRIAN:
            return False

        if distance <= near_ped_distance:
            return True

        if num_lidar_pts > 0 and distance <= lidar_recovery_distance:
            return True

        return False
    
    def _prepare_lidar_points_world(
        self,
        lidar_data: Any,
        lidar_sensor: Optional[Any] = None,
    ) -> Optional[np.ndarray]:
        """Decode one LiDAR sample and transform its points to world space."""
        if lidar_data is None:
            return None

        measurement = lidar_data
        if isinstance(lidar_data, dict) and 'measurement' in lidar_data:
            measurement = lidar_data['measurement']

        if measurement is None:
            return None

        try:
            raw_data = np.frombuffer(measurement.raw_data, dtype=np.dtype('f4'))
            points = raw_data.reshape([-1, 4])
            points_xyz = points[:, :3]

            lidar_transform = None
            if hasattr(measurement, 'transform'):
                lidar_transform = measurement.transform
            elif lidar_sensor is not None:
                lidar_transform = lidar_sensor.get_transform()

            if lidar_transform is None:
                return points_xyz

            lidar_matrix = np.array(lidar_transform.get_matrix(), dtype=np.float32)
            points_hom = np.hstack(
                (points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float32))
            )
            points_world = (lidar_matrix @ points_hom.T).T
            return points_world[:, :3]
        except Exception as e:
            logger = logging.getLogger('scenario_runner')
            logger.debug(f"    Error preparing lidar points: {e}")
            return None

    def _count_lidar_points_in_bbox(self, points_world: np.ndarray, bbox_corners: List[carla.Location]) -> int:
        """
        Count the number of lidar points within a bounding box using optimized numpy operations.
        
        Args:
            points_world: LiDAR points in world coordinates with shape ``(N, 3)``
            bbox_corners: List of 8 corner locations defining the bounding box (in world coordinates)
            
        Returns:
            Number of lidar points inside the bounding box
        """
        if points_world is None or points_world.size == 0:
            return 0
            
        try:
            # Extract bounding box bounds
            corners_np = np.array([[c.x, c.y, c.z] for c in bbox_corners])
            min_bound = np.min(corners_np, axis=0)
            max_bound = np.max(corners_np, axis=0)
            
            # Vectorized check for points within bounds
            # (N, 3) >= (3,) -> (N, 3) boolean
            in_bounds = np.all((points_world >= min_bound) & (points_world <= max_bound), axis=1)
            
            return int(np.sum(in_bounds))
            
        except Exception as e:
            # Log error for debugging but return 0 gracefully
            logger = logging.getLogger('scenario_runner')
            logger.debug(f"    Error counting lidar points: {e}")
            return 0
    
    def get_annotations(self) -> List[Dict[str, Any]]:
        """
        Get all generated annotations.
        
        Returns:
            List of annotation dictionaries in nuScenes format
        """
        return self.annotations
    
    def get_instances(self) -> List[Dict[str, Any]]:
        """
        Get all generated instances.
        
        Returns:
            List of instance dictionaries in nuScenes format
        """
        return self.instances
    
    def reset(self) -> None:
        """Reset annotation builder tracking state for new scene.
        
        Clears tracking state but preserves accumulated annotations/instances.
        This ensures instance temporal linking is isolated per-scene while
        annotations from all scenes are saved together.
        """
        # Clear tracking state for new scene (so instances don't link across scenes)
        self.instance_history.clear()
        self.actor_instance_map.clear()
        # Reset visibility calculator so it gets recreated with new ego vehicle/world
        self.visibility_calculator = None
    
    def reset_all(self) -> None:
        """Fully reset annotation builder including all accumulated data.
        
        Use this when starting a completely new dataset, not between scenes.
        """
        self.annotations.clear()
        self.instances.clear()
        self.instance_history.clear()
        self.actor_instance_map.clear()
        self.visibility_calculator = None
