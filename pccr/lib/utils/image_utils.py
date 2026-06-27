"""
Image processing utilities for CARLA camera data.

Handles image format conversion (CARLA → JPEG/PNG), directory creation, and
standardised filename generation for the nuScenes output structure.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

import carla
import numpy as np
from PIL import Image


_logger = logging.getLogger("scenario_runner")


class ImageProcessor:
    """Convert and save CARLA camera images; generate dataset filenames.

    All methods are static — the class is used purely as a namespace.
    """

    # ------------------------------------------------------------------
    # Image conversion
    # ------------------------------------------------------------------

    @staticmethod
    def carla_image_to_rgb_array(carla_image: carla.Image) -> np.ndarray:
        """Convert a CARLA BGRA image into an (H, W, 3) RGB ``uint8`` array.

        Args:
            carla_image: Raw CARLA camera image (BGRA format).

        Returns:
            NumPy array with shape ``(height, width, 3)`` in RGB channel order.
        """
        array = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
        array = array.reshape((carla_image.height, carla_image.width, 4))
        return array[:, :, [2, 1, 0]]

    @staticmethod
    def carla_image_to_pil(carla_image: carla.Image) -> Image.Image:
        """Convert a CARLA image into a PIL ``RGB`` image.

        Args:
            carla_image: Raw CARLA camera image.

        Returns:
            PIL Image in ``"RGB"`` mode.
        """
        return Image.fromarray(ImageProcessor.carla_image_to_rgb_array(carla_image), "RGB")

    @staticmethod
    def carla_image_to_jpg(carla_image: carla.Image, output_path: Path, quality: int = 85) -> bool:
        """Save a CARLA camera image as a JPEG file.

        Args:
            carla_image: CARLA Image object.
            output_path: Destination file path (parent directories are created).
            quality: JPEG quality in the range ``[1, 100]``.

        Returns:
            ``True`` on success, ``False`` if an exception occurs.
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            pil_image = ImageProcessor.carla_image_to_pil(carla_image)
            pil_image.save(str(output_path), "JPEG", quality=quality)
            return True
        except Exception as exc:
            _logger.error("Error converting image to JPG: %s", exc)
            return False

    @staticmethod
    def carla_depth_to_png(carla_image: carla.Image, output_path: Path, max_depth: float = 1000.0) -> bool:
        """Convert a CARLA depth image to a 16-bit PNG for debugging.

        The CARLA depth sensor encodes depth as ``(R + G*256 + B*65536) / 16777215 * 1000 m``.
        This method decodes that into millimetres and writes a 16-bit greyscale PNG.

        Args:
            carla_image: Raw CARLA depth image.
            output_path: Destination file path (parent directories are created).
            max_depth: Maximum depth value to represent in the PNG (metres).

        Returns:
            ``True`` on success, ``False`` if an exception occurs.
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            array = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
            array = array.reshape((carla_image.height, carla_image.width, 4))

            b = array[:, :, 0].astype(np.float32)
            g = array[:, :, 1].astype(np.float32)
            r = array[:, :, 2].astype(np.float32)

            normalized = (r + g * 256.0 + b * 65536.0) / 16777215.0
            depth_m = np.clip(normalized * max_depth, 0.0, max_depth)
            depth_mm = np.round(depth_m * 1000.0).astype(np.uint16)

            pil_image = Image.fromarray(depth_mm, mode="I;16")
            pil_image.save(str(output_path), format="PNG")
            return True
        except Exception as exc:
            _logger.error("Error converting depth image to PNG: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Directory helpers
    # ------------------------------------------------------------------

    @staticmethod
    def create_sample_directories(
        base_path: Path, camera_names: List[str], split: str | None = None
    ) -> bool:
        """Create the ``samples/<sensor>/`` directory structure for a split.

        Args:
            base_path: Dataset root directory.
            camera_names: Sensor channel names (e.g. ``["CAM_FRONT", "LIDAR_TOP"]``).
            split: Unused; retained for API compatibility.

        Returns:
            ``True`` on success, ``False`` if an exception occurs.
        """
        try:
            samples_dir = base_path / "samples"
            samples_dir.mkdir(parents=True, exist_ok=True)
            for camera_name in camera_names:
                (samples_dir / camera_name).mkdir(exist_ok=True)
            return True
        except Exception as exc:
            _logger.error("Error creating directories: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Filename helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_image_filename(scene_id: str, sample_idx: int, extension: str = "jpg") -> str:
        """Return a standardised image filename.

        Format: ``{scene_id}_{sample_idx:06d}.{extension}``

        Args:
            scene_id: Scene identifier string.
            sample_idx: Zero-based sample index.
            extension: File extension without leading dot.

        Returns:
            Formatted filename string.
        """
        return f"{scene_id}_{sample_idx:06d}.{extension}"

    @staticmethod
    def get_capture_filename(capture_id: str, sensor_name: str, extension: str = "jpg") -> str:
        """Return a debug-capture filename.

        Args:
            capture_id: Unique capture identifier.
            sensor_name: Sensor channel name.
            extension: File extension without leading dot.

        Returns:
            Formatted filename string.
        """
        return f"{capture_id}__{sensor_name}.{extension}"

    @staticmethod
    def get_sample_data_filename(camera_name: str, scene_id: str, sample_idx: int) -> str:
        """Return the dataset-relative path for a camera sample data file.

        Args:
            camera_name: Sensor channel name.
            scene_id: Scene identifier string.
            sample_idx: Zero-based sample index.

        Returns:
            Relative path string: ``samples/{camera_name}/{filename}``.
        """
        filename = ImageProcessor.get_image_filename(scene_id, sample_idx)
        return f"samples/{camera_name}/{filename}"

    @staticmethod
    def get_depth_debug_relative_path(camera_name: str, scene_id: str, sample_idx: int) -> str:
        """Return the dataset-relative path for a debug depth output PNG.

        Args:
            camera_name: Sensor channel name.
            scene_id: Scene identifier string.
            sample_idx: Zero-based sample index.

        Returns:
            Relative path string: ``debug_depth/{camera_name}/{filename}``.
        """
        filename = ImageProcessor.get_image_filename(scene_id, sample_idx, extension="png")
        return f"debug_depth/{camera_name}/{filename}"

    # ------------------------------------------------------------------
    # Miscellaneous helpers
    # ------------------------------------------------------------------

    @staticmethod
    def validate_image_path(image_path: Path) -> bool:
        """Return ``True`` if *image_path* exists, is a non-empty regular file.

        Args:
            image_path: Absolute path to check.
        """
        try:
            return image_path.exists() and image_path.is_file() and image_path.stat().st_size > 0
        except Exception:
            return False

    @staticmethod
    def get_image_dimensions(carla_image: carla.Image) -> tuple[int, int]:
        """Return ``(width, height)`` for a CARLA image.

        Args:
            carla_image: CARLA Image object.
        """
        return (carla_image.width, carla_image.height)

    @staticmethod
    def estimate_storage_size(
        num_scenes: int,
        samples_per_scene: int,
        num_cameras: int,
        width: int = 800,
        height: int = 600,
        quality: int = 95,
    ) -> float:
        """Estimate dataset storage requirements in gigabytes.

        Uses a rough bits-per-pixel heuristic for JPEG at a given quality.

        Args:
            num_scenes: Number of scenes.
            samples_per_scene: Samples per scene.
            num_cameras: Cameras per sample.
            width: Image width in pixels.
            height: Image height in pixels.
            quality: JPEG quality setting.

        Returns:
            Estimated size in GB (includes 20 % overhead for metadata).
        """
        if quality >= 90:
            bits_per_pixel = 3.0
        elif quality >= 80:
            bits_per_pixel = 2.5
        else:
            bits_per_pixel = 2.0

        bytes_per_image = (width * height * bits_per_pixel) / 8
        total_images = num_scenes * samples_per_scene * num_cameras
        total_bytes = total_images * bytes_per_image
        return (total_bytes / (1024 ** 3)) * 1.2
