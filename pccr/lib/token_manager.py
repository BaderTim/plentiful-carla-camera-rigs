"""
Token management for nuScenes-compatible dataset.
Handles UUID generation and token linking.
"""

import uuid
from typing import Dict, Optional


class TokenManager:
    """Centralized UUID token generation and management"""
    
    def __init__(self):
        self.tokens = {}  # Store generated tokens by type and key
        
    def generate_token(self, prefix: str = "") -> str:
        """Generate unique 32-character hex token (nuScenes format)"""
        # Generate UUID and convert to 32-char hex string (no hyphens)
        return uuid.uuid4().hex
        
    def get_or_create_token(self, token_type: str, key: str) -> str:
        """Get existing token or create new one for a key"""
        cache_key = f"{token_type}_{key}"
        if cache_key not in self.tokens:
            self.tokens[cache_key] = self.generate_token()
        return self.tokens[cache_key]
        
    def scene_token(self, scene_id: str) -> str:
        """Get or create scene token"""
        return self.get_or_create_token("scene", scene_id)
        
    def sample_token(self, scene_id: str, sample_idx: int) -> str:
        """Get or create sample token"""
        return self.get_or_create_token("sample", f"{scene_id}_{sample_idx:06d}")
        
    def sensor_token(self, camera_id: str) -> str:
        """Get or create sensor token"""
        return self.get_or_create_token("sensor", camera_id)
        
    def calibrated_sensor_token(self, sensor_id: str, rig_name: str = "") -> str:
        """Get or create calibrated sensor token.
        
        Args:
            sensor_id: Sensor identifier (e.g., CAM_FRONT, LIDAR_TOP)
            rig_name: Rig name to ensure unique calibrations per rig
        """
        # Include rig_name to ensure unique calibrations per sensor rig
        key = f"{rig_name}_{sensor_id}" if rig_name else sensor_id
        return self.get_or_create_token("calibrated_sensor", key)
        
    def ego_pose_token(self, scene_id: str, sample_idx: int) -> str:
        """Get or create ego pose token.
        
        Args:
            scene_id: Scene identifier to ensure uniqueness across scenes
            sample_idx: Sample index within the scene
        """
        # Use scene_id + sample_idx to ensure unique ego poses per scene/sample
        return self.get_or_create_token("ego_pose", f"{scene_id}_{sample_idx:06d}")
        
    def sample_data_token(self, scene_id: str, sensor_id: str, sample_idx: int) -> str:
        """Get or create sample data token.
        
        Args:
            scene_id: Scene identifier to ensure uniqueness across scenes
            sensor_id: Sensor identifier (e.g., CAM_FRONT, LIDAR_TOP)
            sample_idx: Sample index within the scene
        """
        # Include scene_id to ensure unique tokens across scenes
        return self.get_or_create_token("sample_data", f"{scene_id}_{sensor_id}_{sample_idx:06d}")
        
    def log_token(self, scene_id: str) -> str:
        """Get or create log token based on scene_id (one log per scene)"""
        return self.get_or_create_token("log", scene_id)
        
    def clear_cache(self):
        """Clear all cached tokens"""
        self.tokens.clear()
        
    def get_all_tokens_by_type(self, token_type: str) -> Dict[str, str]:
        """Get all tokens of a specific type"""
        return {
            key.split(f"{token_type}_", 1)[1]: token 
            for key, token in self.tokens.items() 
            if key.startswith(f"{token_type}_")
        }