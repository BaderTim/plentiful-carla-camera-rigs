"""
Threaded disk-I/O workers for the scene-capture pipeline.

:class:`IOWorkersMixin` is mixed into :class:`~core.scene_runner.SceneRunner`
to provide bounded-queue worker threads that save sensor data asynchronously
as the simulation ticks.

Three worker channels are provided:

* **image** — RGB camera frames serialised to JPEG via
  :meth:`~lib.image_utils.ImageProcessor.carla_image_to_jpg`.
* **lidar** — LiDAR point clouds saved as raw float32 binary
  (5 columns: x, y, z, intensity, ring_index) matching the nuScenes format.
* **depth_debug** — Depth-camera frames saved as PNG for visual inspection;
  only active when ``capture_depth_debug`` is ``True``.
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import List

import numpy as np

from lib.utils.image_utils import ImageProcessor
from lib.utils.logging_utils import log_print


class IOWorkersMixin:
    """Mixin that adds bounded-queue threaded I/O workers to SceneRunner.

    Expects the host class to provide the following attributes on *self*
    (typically initialised in ``SceneRunner.__init__``):

    * ``image_save_queue`` — :class:`queue.Queue`
    * ``lidar_save_queue`` — :class:`queue.Queue`
    * ``depth_debug_queue`` — :class:`queue.Queue`
    * ``image_save_threads`` — :class:`list`
    * ``lidar_save_threads`` — :class:`list`
    * ``depth_debug_threads`` — :class:`list`
    * ``capture_depth_debug`` — :class:`bool`
    """

    # ------------------------------------------------------------------
    # Worker loops
    # ------------------------------------------------------------------

    def image_save_worker(self) -> None:
        """Consume (image, path) tuples from *image_save_queue* and write JPEG files.

        Terminates when a ``None`` sentinel is dequeued.
        """
        while True:
            try:
                item = self.image_save_queue.get(timeout=2)
                if item is None:
                    break
                image, image_path = item
                try:
                    ImageProcessor.carla_image_to_jpg(image, image_path, quality=85)
                except Exception as exc:
                    log_print(f"Warning: Failed to save image {image_path}: {exc}", "WARNING")
                finally:
                    self.image_save_queue.task_done()
            except queue.Empty:
                continue
            except Exception as exc:
                log_print(f"Error in image save worker: {exc}", "ERROR")
                break

    def lidar_save_worker(self) -> None:
        """Consume (measurement, path) tuples from *lidar_save_queue* and write PCD files.

        Converts CARLA's left-handed coordinate system (x=forward, y=right, z=up)
        to the nuScenes right-handed system (x=forward, y=left, z=up) by negating
        the y-axis.  Output is 5-column float32: ``[x, y, z, intensity, ring_index]``
        where *ring_index* is zero-padded.

        Terminates when a ``None`` sentinel is dequeued.
        """
        while True:
            try:
                item = self.lidar_save_queue.get(timeout=2)
                if item is None:
                    break
                lidar_data, lidar_path = item
                try:
                    lidar_path.parent.mkdir(parents=True, exist_ok=True)
                    raw = np.frombuffer(lidar_data.raw_data, dtype=np.dtype("f4"))
                    points = raw.reshape([-1, 4]).copy()  # x, y, z, intensity
                    points[:, 1] = -points[:, 1]          # CARLA → nuScenes y-flip
                    with_ring = np.zeros((points.shape[0], 5), dtype=np.float32)
                    with_ring[:, :4] = points.astype(np.float32)
                    with open(lidar_path, "wb") as f:
                        with_ring.tofile(f)
                except Exception as exc:
                    log_print(f"Warning: Failed to save lidar {lidar_path}: {exc}", "WARNING")
                finally:
                    self.lidar_save_queue.task_done()
            except queue.Empty:
                continue
            except Exception as exc:
                log_print(f"Error in lidar save worker: {exc}", "ERROR")
                break

    def depth_debug_save_worker(self) -> None:
        """Consume (depth_image, path) tuples from *depth_debug_queue* and write PNG files.

        Terminates when a ``None`` sentinel is dequeued.
        """
        while True:
            try:
                item = self.depth_debug_queue.get(timeout=2)
                if item is None:
                    break
                depth_image, depth_path = item
                try:
                    ImageProcessor.carla_depth_to_png(depth_image, depth_path)
                except Exception as exc:
                    log_print(f"Warning: Failed to save depth PNG {depth_path}: {exc}", "WARNING")
                finally:
                    self.depth_debug_queue.task_done()
            except queue.Empty:
                continue
            except Exception as exc:
                log_print(f"Error in depth debug save worker: {exc}", "ERROR")
                break

    # ------------------------------------------------------------------
    # Thread management — start
    # ------------------------------------------------------------------

    def start_image_save_threads(self, num_threads: int = 4) -> None:
        """Start *num_threads* image-saving worker threads.

        Any existing threads are stopped first.

        Args:
            num_threads: Number of parallel worker threads.
        """
        self.stop_image_save_threads()
        for _ in range(num_threads):
            t = threading.Thread(target=self.image_save_worker, daemon=True)
            t.start()
            self.image_save_threads.append(t)

    def start_lidar_save_threads(self, num_threads: int = 2) -> None:
        """Start *num_threads* LiDAR-saving worker threads.

        Args:
            num_threads: Number of parallel worker threads.
        """
        self.stop_lidar_save_threads()
        for _ in range(num_threads):
            t = threading.Thread(target=self.lidar_save_worker, daemon=True)
            t.start()
            self.lidar_save_threads.append(t)

    def start_depth_debug_threads(self, num_threads: int = 2) -> None:
        """Start depth-debug saving threads when ``capture_depth_debug`` is set.

        A no-op when ``self.capture_depth_debug`` is ``False``.

        Args:
            num_threads: Number of parallel worker threads.
        """
        self.stop_depth_debug_threads()
        if not self.capture_depth_debug:
            return
        for _ in range(num_threads):
            t = threading.Thread(target=self.depth_debug_save_worker, daemon=True)
            t.start()
            self.depth_debug_threads.append(t)

    # ------------------------------------------------------------------
    # Thread management — stop
    # ------------------------------------------------------------------

    def stop_image_save_threads(self) -> None:
        """Signal and join all image-saving threads; drain the queue."""
        for _ in self.image_save_threads:
            self.image_save_queue.put(None)
        for t in self.image_save_threads:
            if t.is_alive():
                t.join(timeout=2)
        self.image_save_threads.clear()
        _drain(self.image_save_queue)

    def stop_lidar_save_threads(self) -> None:
        """Signal and join all LiDAR-saving threads; drain the queue."""
        for _ in self.lidar_save_threads:
            self.lidar_save_queue.put(None)
        for t in self.lidar_save_threads:
            if t.is_alive():
                t.join(timeout=2)
        self.lidar_save_threads.clear()
        _drain(self.lidar_save_queue)

    def stop_depth_debug_threads(self) -> None:
        """Signal and join all depth-debug saving threads; drain the queue."""
        for _ in self.depth_debug_threads:
            self.depth_debug_queue.put(None)
        for t in self.depth_debug_threads:
            if t.is_alive():
                t.join(timeout=2)
        self.depth_debug_threads.clear()
        _drain(self.depth_debug_queue)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _drain(q: queue.Queue) -> None:
    """Discard all pending items from *q* without blocking."""
    while not q.empty():
        try:
            q.get_nowait()
        except queue.Empty:
            break
