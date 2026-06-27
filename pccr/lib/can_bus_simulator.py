"""
CAN bus data simulator for CARLA vehicles.
Generates nuScenes-compatible CAN bus messages from CARLA vehicle state.

Supported message types:
- pose: Vehicle pose with position, orientation, velocity, acceleration
- ms_imu: IMU data (linear acceleration, orientation, rotation rate)
- steeranglefeedback: Steering wheel angle
- vehicle_monitor: Vehicle speed, yaw rate (partial - torque/cruise not available)
- meta: Statistics computed at save time

Not supported (CARLA doesn't provide):
- zoesensors: Renault Zoe specific sensors
- zoe_veh_info: Renault Zoe vehicle info
- route: Navigation route waypoints
"""

import time
import math
from typing import Dict, Any, Optional, List
import carla
from .utils.coordinate_utils import CoordinateConverter


class CANBusSimulator:
    """Simulate CAN bus data from CARLA vehicle state"""
    
    def __init__(self):
        self.previous_state = None
        self.previous_timestamp = None
        self.previous_angular_velocity = None

    @staticmethod
    def _vector3d_from_sequence(values: Optional[List[float]]) -> carla.Vector3D:
        if not values:
            return carla.Vector3D(0.0, 0.0, 0.0)
        return carla.Vector3D(float(values[0]), float(values[1]), float(values[2]))
        
    def generate_all_can_messages(
        self,
        vehicle: carla.Vehicle,
        timestamp: float,
        trajectory_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate all CAN bus message types from CARLA vehicle state.
        
        Args:
            vehicle: CARLA vehicle actor
            timestamp: Current timestamp in seconds
            trajectory_state: Optional recorded replay state for the ego vehicle.
            
        Returns:
            Dict with keys for each message type (pose, ms_imu, steeranglefeedback, vehicle_monitor)
        """
        # Get vehicle state
        transform = vehicle.get_transform()
        if trajectory_state is not None:
            velocity = self._vector3d_from_sequence(trajectory_state.get("selected_velocity"))
            angular_velocity = self._vector3d_from_sequence(trajectory_state.get("angular_velocity"))
            acceleration = self._vector3d_from_sequence(trajectory_state.get("acceleration"))
            control_throttle = float(trajectory_state.get("control_throttle", 0.0) or 0.0)
            control_steer = float(trajectory_state.get("control_steer", 0.0) or 0.0)
            control_brake = float(trajectory_state.get("control_brake", 0.0) or 0.0)
        else:
            velocity = vehicle.get_velocity()  # m/s
            angular_velocity = vehicle.get_angular_velocity()  # deg/s
            control = vehicle.get_control()
            control_throttle = float(getattr(control, "throttle", 0.0) or 0.0)
            control_steer = float(getattr(control, "steer", 0.0) or 0.0)
            control_brake = float(getattr(control, "brake", 0.0) or 0.0)

            # Calculate acceleration from velocity change
            acceleration = self._calculate_acceleration(velocity, timestamp)
        
        # Handle position-based velocity estimation for low speeds when replay state is absent.
        if trajectory_state is None and self.previous_state is not None:
            dt = timestamp - self.previous_timestamp
            if dt > 0:
                prev_pos = self.previous_state["position"]
                current_pos = transform.location
                dx = current_pos.x - prev_pos.x
                dy = current_pos.y - prev_pos.y
                dz = current_pos.z - prev_pos.z
                position_velocity = carla.Vector3D(dx/dt, dy/dt, dz/dt)
                
                speed_carla = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                speed_position = math.sqrt(position_velocity.x**2 + position_velocity.y**2 + position_velocity.z**2)
                
                if speed_carla < 0.1 and speed_position > 0.1:
                    velocity = position_velocity
        
        # Convert timestamp to microseconds (nuScenes format)
        utime = int(timestamp * 1e6)
        
        # Convert to nuScenes coordinates
        pos = CoordinateConverter.carla_to_nuscenes_location(transform.location)
        orientation = CoordinateConverter.carla_to_nuscenes_rotation(transform.rotation)
        vel = CoordinateConverter.carla_to_nuscenes_velocity(velocity)
        accel = CoordinateConverter.carla_to_nuscenes_velocity(acceleration)
        
        # Convert angular velocity from deg/s to rad/s
        angular_velocity_rad = carla.Vector3D(
            math.radians(angular_velocity.x),
            math.radians(angular_velocity.y), 
            math.radians(angular_velocity.z)
        )
        rotation_rate = CoordinateConverter.carla_to_nuscenes_velocity(angular_velocity_rad)
        
        # Calculate vehicle speed (m/s)
        vehicle_speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        # Yaw rate in rad/s (rotation around Z axis)
        # Negate for coordinate system conversion (CARLA left-handed to nuScenes right-handed)
        yaw_rate = -math.radians(angular_velocity.z)
        
        # Steering angle: CARLA control.steer is -1 to 1, convert to approximate degrees
        # Typical steering wheel range is about 450 degrees each way
        steer_angle_deg = control_steer * 450.0
        
        # Generate all message types
        messages = {
            "pose": {
                "accel": accel,
                "orientation": orientation,
                "pos": pos,
                "rotation_rate": rotation_rate,
                "utime": utime,
                "vel": vel
            },
            "ms_imu": {
                "linear_accel": accel,
                "q": orientation,
                "rotation_rate": rotation_rate,
                "utime": utime
            },
            "steeranglefeedback": {
                "value": steer_angle_deg,
                "utime": utime
            },
            "vehicle_monitor": {
                "vehicle_speed": vehicle_speed,
                "yaw_rate": yaw_rate,
                "utime": utime
            }
        }
        
        # Update previous state for next calculation
        self.previous_state = {
            "velocity": velocity,
            "position": transform.location,
            "timestamp": timestamp
        }
        self.previous_timestamp = timestamp
        self.previous_angular_velocity = angular_velocity
        
        return messages
    
    def generate_can_data(
        self,
        vehicle: carla.Vehicle,
        timestamp: float,
        trajectory_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate pose CAN bus data (legacy method for backward compatibility).
        
        Args:
            vehicle: CARLA vehicle actor
            timestamp: Current timestamp in seconds
            
        Returns:
            Dict: Pose CAN bus data in nuScenes format
        """
        messages = self.generate_all_can_messages(vehicle, timestamp, trajectory_state=trajectory_state)
        return messages["pose"]
    
    def _calculate_acceleration(self, current_velocity: carla.Vector3D, timestamp: float) -> carla.Vector3D:
        """Calculate acceleration from velocity change"""
        if self.previous_state is None or self.previous_timestamp is None:
            return carla.Vector3D(0.0, 0.0, 0.0)
        
        dt = timestamp - self.previous_timestamp
        if dt <= 0:
            return carla.Vector3D(0.0, 0.0, 0.0)
        
        prev_vel = self.previous_state["velocity"]
        
        accel_carla = carla.Vector3D(
            (current_velocity.x - prev_vel.x) / dt,
            (current_velocity.y - prev_vel.y) / dt,
            (current_velocity.z - prev_vel.z) / dt
        )
        
        return accel_carla
    
    def reset(self):
        """Reset simulator state for new scene"""
        self.previous_state = None
        self.previous_timestamp = None
        self.previous_angular_velocity = None