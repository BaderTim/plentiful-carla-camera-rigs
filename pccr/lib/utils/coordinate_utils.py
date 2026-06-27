"""
Coordinate system conversion utilities for CARLA to nuScenes format.
Handles transformations between different coordinate conventions.
"""

import math
import numpy as np
from typing import List, Dict, Any
import carla


class CoordinateConverter:
    """Convert between CARLA and nuScenes coordinate systems"""
    
    @staticmethod
    def carla_to_nuscenes_location(carla_location) -> List[float]:
        """
        Convert CARLA location to nuScenes [x, y, z]
        
        CARLA: x=forward, y=right, z=up (left-handed)
        nuScenes: x=forward, y=left, z=up (right-handed)
        """
        return [
            float(carla_location.x),
            float(-carla_location.y),  # Flip Y axis
            float(carla_location.z)
        ]
    
    @staticmethod
    def carla_to_nuscenes_rotation(carla_rotation) -> List[float]:
        """
        Convert CARLA rotation to nuScenes quaternion [w, x, y, z]
        
        Args:
            carla_rotation: CARLA Rotation object (pitch, yaw, roll in degrees)
            
        Returns:
            List[float]: Quaternion [w, x, y, z]
        """
        # Convert degrees to radians
        pitch = math.radians(carla_rotation.pitch)
        yaw = math.radians(carla_rotation.yaw)
        roll = math.radians(carla_rotation.roll)
        
        # Apply coordinate system conversion
        # CARLA uses left-handed, nuScenes uses right-handed
        # Flip yaw and pitch for coordinate system conversion
        yaw = -yaw
        pitch = -pitch
        
        # Convert Euler angles to quaternion
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return [float(w), float(x), float(y), float(z)]
    
    @staticmethod
    def carla_to_nuscenes_velocity(carla_velocity) -> List[float]:
        """
        Convert CARLA velocity to nuScenes format
        
        Args:
            carla_velocity: CARLA Vector3D velocity
            
        Returns:
            List[float]: Velocity [vx, vy, vz] in nuScenes coordinates
        """
        return [
            float(carla_velocity.x),
            float(-carla_velocity.y),  # Flip Y axis
            float(carla_velocity.z)
        ]
    
    @staticmethod
    def compute_camera_intrinsic(width: int, height: int, fov: float) -> List[List[float]]:
        """
        Compute camera intrinsic matrix from FOV
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            fov: Field of view in degrees
            
        Returns:
            List[List[float]]: 3x3 camera intrinsic matrix
        """
        fov_rad = math.radians(fov)
        focal_length = width / (2.0 * math.tan(fov_rad / 2.0))
        
        # Camera intrinsic matrix
        # [[fx,  0, cx],
        #  [ 0, fy, cy],
        #  [ 0,  0,  1]]
        intrinsic = [
            [focal_length, 0.0, width / 2.0],
            [0.0, focal_length, height / 2.0],
            [0.0, 0.0, 1.0]
        ]
        
        return intrinsic
    
    @staticmethod
    def ego_pose_to_nuscenes(carla_transform, carla_velocity=None) -> Dict[str, Any]:
        """
        Convert CARLA ego pose to nuScenes format
        
        Args:
            carla_transform: CARLA Transform object
            carla_velocity: CARLA Vector3D velocity (optional)
            
        Returns:
            Dict: nuScenes ego pose data
        """
        translation = CoordinateConverter.carla_to_nuscenes_location(carla_transform.location)
        rotation = CoordinateConverter.carla_to_nuscenes_rotation(carla_transform.rotation)
        
        pose_data = {
            "translation": translation,
            "rotation": rotation
        }
        
        if carla_velocity is not None:
            velocity = CoordinateConverter.carla_to_nuscenes_velocity(carla_velocity)
            pose_data["velocity"] = velocity
            # Calculate speed as magnitude of velocity vector
            pose_data["speed"] = float(math.sqrt(sum(v**2 for v in velocity)))
        
        return pose_data
    
    @staticmethod
    def camera_transform_to_nuscenes(carla_transform) -> Dict[str, Any]:
        """
        Convert camera transform relative to ego vehicle to nuScenes format
        
        Args:
            carla_transform: CARLA Transform of camera relative to vehicle
            
        Returns:
            Dict: Camera pose relative to ego vehicle
        """
        translation = CoordinateConverter.carla_to_nuscenes_location(carla_transform.location)
        rotation = CoordinateConverter.carla_to_nuscenes_rotation(carla_transform.rotation)
        
        return {
            "translation": translation,
            "rotation": rotation
        }
    
    @staticmethod
    def compute_relative_pose(current_pose: Dict, previous_pose: Dict) -> Dict[str, Any]:
        """
        Compute relative pose change between two poses
        
        Args:
            current_pose: Current ego pose in nuScenes format
            previous_pose: Previous ego pose in nuScenes format
            
        Returns:
            Dict: Relative pose change
        """
        if previous_pose is None:
            return {
                "translation": [0.0, 0.0, 0.0],
                "rotation": [0.0, 0.0, 0.0]  # Euler angles for relative rotation
            }
        
        # Calculate relative translation
        rel_translation = [
            current_pose["translation"][0] - previous_pose["translation"][0],
            current_pose["translation"][1] - previous_pose["translation"][1],
            current_pose["translation"][2] - previous_pose["translation"][2]
        ]
        
        # For relative rotation, we'd need more complex quaternion math
        # For now, return zero relative rotation
        rel_rotation = [0.0, 0.0, 0.0]
        
        return {
            "translation": rel_translation,
            "rotation": rel_rotation
        }
    
    @staticmethod
    def multiply_quaternions(q1: List[float], q2: List[float]) -> List[float]:
        """
        Multiply two quaternions q1 * q2
        q = [w, x, y, z]
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return [float(w), float(x), float(y), float(z)]

    @staticmethod
    def get_camera_basis_change_rotation() -> List[float]:
        """
        Get quaternion to rotate from Camera Optical Frame to CARLA/Vehicle Frame.
        
        Camera Optical: X-right, Y-down, Z-forward
        CARLA/Vehicle: X-forward, Y-right, Z-up (Left Handed) -> Converted to nuScenes: X-forward, Y-left, Z-up
        
        We need the rotation that transforms a point in Optical Frame to Vehicle Frame (in nuScenes coords).
        
        Optical Z (Forward) -> Vehicle X (Forward)
        Optical X (Right) -> Vehicle -Y (Right)
        Optical Y (Down) -> Vehicle -Z (Down)
        
        Rotation Matrix (Optical -> Vehicle):
        [[0, 0, 1],
         [-1, 0, 0],
         [0, -1, 0]]
         
        Quaternion [w, x, y, z]: [0.5, -0.5, 0.5, -0.5]
        """
        return [0.5, -0.5, 0.5, -0.5]