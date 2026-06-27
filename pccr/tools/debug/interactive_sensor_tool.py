#!/usr/bin/env python3
"""
CARLA Interactive Sensor Positioning Tool.

Spawn a vehicle, attach cameras and a LiDAR sensor, then interactively
reposition them in real-time with keyboard controls.  Visual feedback
includes FOV triangles, directional arrows, and a yellow highlight for the
currently selected sensor.  The final configuration can be saved to JSON and
loaded back without restarting.

Controls
--------
Position adjustment::

    W / S   — forward / backward (X)
    A / D   — left / right (Y)
    Q / E   — up / down (Z)

Rotation (cameras only)::

    ← / →   — yaw
    ↑ / ↓   — pitch

Sensor selection::

    1 – 9, 0, -, =   — cameras 1 – 12
    L                 — LiDAR sensor

File / capture::

    Ctrl+S   — save configuration
    Ctrl+L   — load configuration
    C        — capture images from all cameras

Other::

    ESC   — quit

Usage::

    python debug/interactive_sensor_tool.py
    python debug/interactive_sensor_tool.py --vehicle vehicle.tesla.model3
    python debug/interactive_sensor_tool.py --host 192.168.1.10 --port 2000
    python debug/interactive_sensor_tool.py --config my_rig.json
"""

from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
from pathlib import Path
from queue import Queue
from typing import Any, Dict, List, Optional

import carla
import pygame

# Ensure project root is importable when run directly.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ---------------------------------------------------------------------------
# Default sensor configurations
# ---------------------------------------------------------------------------

DEFAULT_VEHICLE_BP = "vehicle.audi.tt"

DEFAULT_CAM_CONFIGS: List[Dict[str, Any]] = [
    {
        "id": "CAM_FRONT",
        "loc": carla.Location(x=0.24, y=0.0, z=1.34),
        "rot": carla.Rotation(pitch=18.0, yaw=0.0, roll=0.0),
        "fov": 120,
    },
    {
        "id": "CAM_BACK",
        "loc": carla.Location(x=-1.98, y=0.0, z=0.72),
        "rot": carla.Rotation(pitch=10.0, yaw=180.0, roll=0.0),
        "fov": 120,
    },
    {
        "id": "CAM_FRONT_LEFT",
        "loc": carla.Location(x=0.82, y=-0.96, z=0.74),
        "rot": carla.Rotation(pitch=0.0, yaw=-113.0, roll=0.0),
        "fov": 120,
    },
    {
        "id": "CAM_FRONT_RIGHT",
        "loc": carla.Location(x=0.84, y=0.96, z=0.84),
        "rot": carla.Rotation(pitch=0.0, yaw=114.5, roll=0.0),
        "fov": 120,
    },
]

DEFAULT_LIDAR_CONFIG: Dict[str, Any] = {
    "id": "LIDAR_TOP",
    "loc": carla.Location(x=1.7, y=0.0, z=1.6),
    "rot": carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
}

# Camera colours used when USE_MONO_COLOR is False.
_MULTI_COLORS = [
    carla.Color(0, 255, 0),     # green
    carla.Color(255, 0, 0),     # red
    carla.Color(0, 180, 255),   # blue
    carla.Color(255, 180, 0),   # orange
    carla.Color(200, 0, 200),   # purple
    carla.Color(0, 200, 100),   # teal
    carla.Color(255, 100, 100), # light red
    carla.Color(100, 255, 100), # light green
    carla.Color(100, 100, 255), # light blue
    carla.Color(255, 255, 100), # yellow
    carla.Color(255, 0, 255),   # magenta
    carla.Color(0, 255, 255),   # cyan
]

FOV_DISTANCE = 4.0
LIFE_TIME = 0.4
POSITION_STEP = 0.02
ROTATION_STEP = 0.5
USE_MONO_COLOR = True
MONO_COLOR = carla.Color(0, 0, 255)


# ---------------------------------------------------------------------------
# Tool class
# ---------------------------------------------------------------------------

class InteractiveSensorTool:
    """Real-time interactive sensor-positioning tool.

    Attributes:
        client: CARLA client.
        world: Current CARLA world.
        vehicle: Spawned test vehicle.
        cameras: List of ``[cfg_dict, carla.Actor]`` pairs — mutable so
            both the config and the actor can be updated simultaneously.
        lidar: Spawned LiDAR actor, or ``None`` before :meth:`spawn_lidar`.
        lidar_config: Mutable LiDAR config dict mirroring current position.
        selected_cam: Index into :attr:`cameras` for the selected camera.
        selected_sensor_type: One of ``"camera"`` or ``"lidar"``.
        output_dir: Directory for captured images.
        vehicle_bp: Blueprint ID of the vehicle to spawn.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 2000,
        vehicle_bp: str = DEFAULT_VEHICLE_BP,
        config_file: Optional[str] = None,
    ) -> None:
        """Connect to CARLA and initialise tool state.

        Args:
            host: CARLA simulator hostname.
            port: CARLA simulator port.
            vehicle_bp: Blueprint ID of the ego vehicle.
            config_file: JSON config to load on startup; ``None`` uses the
                built-in :data:`DEFAULT_CAM_CONFIGS`.
        """
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.vehicle: Optional[carla.Actor] = None
        self.cameras: List[List] = []
        self.lidar: Optional[carla.Actor] = None

        self.vehicle_bp = vehicle_bp
        self.cam_configs = [dict(c) for c in DEFAULT_CAM_CONFIGS]
        self.lidar_config = dict(DEFAULT_LIDAR_CONFIG)
        self.config_file = config_file or "sensor_config.json"

        self.selected_cam = 0
        self.selected_sensor_type = "camera"

        self.output_dir = "output_images"
        os.makedirs(self.output_dir, exist_ok=True)

        self.capture_queue: Queue = Queue()
        self.capture_pending = False
        self.show_triangles = True
        self.show_arrows = True

        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Interactive Sensor Tool — CARLA")
        self.clock = pygame.time.Clock()

    # ------------------------------------------------------------------
    # Spawn / destroy
    # ------------------------------------------------------------------

    def destroy_all_vehicles(self) -> None:
        """Destroy all existing vehicles in the CARLA world."""
        for actor in self.world.get_actors().filter("vehicle.*"):
            try:
                actor.destroy()
            except Exception:
                pass

    def spawn_vehicle(self) -> None:
        """Spawn the test vehicle at a consistent spawn point."""
        lib = self.world.get_blueprint_library()
        cands = lib.filter(self.vehicle_bp)
        bp = cands[0] if cands else lib.filter("vehicle.*")[0]
        print(f"Blueprint: {bp.id}")

        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points available.")
        sp = spawn_points[min(11, len(spawn_points) - 1)]
        self.vehicle = self.world.try_spawn_actor(bp, sp)
        if self.vehicle is None:
            self.vehicle = self.world.spawn_actor(bp, sp)
        print(f"Spawned vehicle id={self.vehicle.id} ({bp.id})")
        time.sleep(0.5)

        # Top-down spectator view.
        t = self.vehicle.get_transform()
        self.world.get_spectator().set_transform(
            carla.Transform(
                carla.Location(x=t.location.x, y=t.location.y, z=t.location.z + 12.0),
                carla.Rotation(pitch=-90.0),
            )
        )

    def spawn_cameras(self) -> None:
        """Spawn and attach all camera sensors to :attr:`vehicle`."""
        lib = self.world.get_blueprint_library()
        bp = lib.find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", "800")
        bp.set_attribute("image_size_y", "600")

        for cfg in self.cam_configs:
            bp.set_attribute("fov", str(cfg["fov"]))
            camera = self.world.spawn_actor(
                bp, carla.Transform(cfg["loc"], cfg["rot"]), attach_to=self.vehicle
            )

            def _cb(image: carla.Image, cam_id: str = cfg["id"]) -> None:
                if self.capture_pending:
                    path = os.path.join(self.output_dir, f"{cam_id}.png")
                    image.save_to_disk(path)
                    self.capture_queue.put((cam_id, path))

            camera.listen(_cb)
            self.cameras.append([cfg, camera])
            print(f"  Camera {cfg['id']} attached.")

    def spawn_lidar(self) -> None:
        """Spawn and attach the LiDAR sensor to :attr:`vehicle`."""
        lib = self.world.get_blueprint_library()
        bp = lib.find("sensor.lidar.ray_cast")
        bp.set_attribute("channels", "32")
        bp.set_attribute("range", "100")
        bp.set_attribute("points_per_second", "56000")
        bp.set_attribute("rotation_frequency", "10")
        self.lidar = self.world.spawn_actor(
            bp,
            carla.Transform(self.lidar_config["loc"], self.lidar_config["rot"]),
            attach_to=self.vehicle,
        )
        print(f"  LiDAR {self.lidar_config['id']} attached.")

    # ------------------------------------------------------------------
    # Debug visualisation
    # ------------------------------------------------------------------

    def _draw_fov_triangle(
        self,
        origin: carla.Location,
        rotation: carla.Rotation,
        color: carla.Color,
        fov_deg: float,
    ) -> None:
        """Draw a horizontal FOV triangle in the CARLA viewport."""
        yaw = math.radians(rotation.yaw)
        pitch = math.radians(rotation.pitch)
        half = math.radians(fov_deg / 2.0)
        cos_p = math.cos(pitch)
        sin_p = math.sin(pitch)
        for angle in (yaw - half, yaw + half):
            pt = carla.Location(
                x=origin.x + math.cos(angle) * FOV_DISTANCE * cos_p,
                y=origin.y + math.sin(angle) * FOV_DISTANCE * cos_p,
                z=origin.z + sin_p * FOV_DISTANCE,
            )
            self.world.debug.draw_line(
                origin, pt, thickness=0.02, color=color, life_time=LIFE_TIME
            )
        left = carla.Location(
            x=origin.x + math.cos(yaw - half) * FOV_DISTANCE * cos_p,
            y=origin.y + math.sin(yaw - half) * FOV_DISTANCE * cos_p,
            z=origin.z + sin_p * FOV_DISTANCE,
        )
        right = carla.Location(
            x=origin.x + math.cos(yaw + half) * FOV_DISTANCE * cos_p,
            y=origin.y + math.sin(yaw + half) * FOV_DISTANCE * cos_p,
            z=origin.z + sin_p * FOV_DISTANCE,
        )
        self.world.debug.draw_line(
            left, right, thickness=0.02, color=color, life_time=LIFE_TIME
        )

    def _draw_camera_debug(self) -> None:
        """Draw spheres, forward arrows, and FOV triangles for each camera."""
        for idx, (cfg, camera) in enumerate(self.cameras):
            t = camera.get_transform()
            color = MONO_COLOR if USE_MONO_COLOR else _MULTI_COLORS[idx % len(_MULTI_COLORS)]
            is_selected = (idx == self.selected_cam and self.selected_sensor_type == "camera")
            dot_color = carla.Color(255, 255, 0) if is_selected else color
            dot_size = 0.4 if is_selected else 0.25
            self.world.debug.draw_point(t.location, size=dot_size, color=dot_color, life_time=LIFE_TIME)
            if self.show_arrows:
                fv = t.rotation.get_forward_vector()
                end = carla.Location(
                    x=t.location.x + fv.x * 1.5,
                    y=t.location.y + fv.y * 1.5,
                    z=t.location.z + fv.z * 1.5,
                )
                self.world.debug.draw_arrow(
                    t.location, end, thickness=0.05, arrow_size=0.1,
                    color=color, life_time=LIFE_TIME,
                )
            if self.show_triangles:
                self._draw_fov_triangle(t.location, t.rotation, color, cfg["fov"])

    def _draw_lidar_debug(self) -> None:
        """Draw a sphere and radial lines for the LiDAR."""
        if self.lidar is None:
            return
        t = self.lidar.get_transform()
        color = carla.Color(255, 0, 0)
        is_selected = self.selected_sensor_type == "lidar"
        dot_color = carla.Color(255, 255, 0) if is_selected else color
        dot_size = 0.4 if is_selected else 0.25
        self.world.debug.draw_point(t.location, size=dot_size, color=dot_color, life_time=LIFE_TIME)
        if self.show_arrows:
            for i in range(8):
                a = math.radians(i * 45.0)
                pt = carla.Location(
                    x=t.location.x + math.cos(a) * 3.0,
                    y=t.location.y + math.sin(a) * 3.0,
                    z=t.location.z,
                )
                self.world.debug.draw_line(
                    t.location, pt, thickness=0.02, color=color, life_time=LIFE_TIME
                )

    # ------------------------------------------------------------------
    # Image capture
    # ------------------------------------------------------------------

    def _capture_images(self) -> None:
        """Trigger simultaneous image capture from all cameras."""
        if self.capture_pending:
            return
        print("Capturing images...")
        self.show_triangles = False
        self.show_arrows = False
        time.sleep(1.0)
        self.capture_pending = True
        while not self.capture_queue.empty():
            self.capture_queue.get()

        def finish() -> None:
            time.sleep(1.5)
            found: List = []
            for cfg, _ in self.cameras:
                fn = f"{cfg['id']}.png"
                fp = os.path.join(self.output_dir, fn)
                if os.path.exists(fp) and time.time() - os.path.getmtime(fp) < 3:
                    found.append(fn)
            if found:
                print(f"Captured {len(found)} images: {', '.join(found)}")
            else:
                print("No images captured.")
            self.capture_pending = False
            self.show_triangles = True
            self.show_arrows = True

        t = threading.Thread(target=finish, daemon=True)
        t.start()

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def _handle_input(self) -> bool:
        """Process pygame events and continuous key-press sensor adjustments.

        Returns:
            ``False`` when the user requests to quit; ``True`` otherwise.
        """
        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type != pygame.KEYDOWN:
                    continue
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_s and _ctrl_held():
                    self._save_config()
                elif event.key == pygame.K_l and _ctrl_held():
                    self._load_config()
                elif event.key == pygame.K_c:
                    self._capture_images()
                elif pygame.K_1 <= event.key <= pygame.K_9:
                    idx = event.key - pygame.K_1
                    if idx < len(self.cameras):
                        self.selected_cam = idx
                        self.selected_sensor_type = "camera"
                        print(f"Selected: {self.cameras[idx][0]['id']}")
                elif event.key == pygame.K_0 and 9 < len(self.cameras):
                    self.selected_cam = 9
                    self.selected_sensor_type = "camera"
                elif event.key == pygame.K_MINUS and 10 < len(self.cameras):
                    self.selected_cam = 10
                    self.selected_sensor_type = "camera"
                elif event.key == pygame.K_EQUALS and 11 < len(self.cameras):
                    self.selected_cam = 11
                    self.selected_sensor_type = "camera"
                elif event.key == pygame.K_l and self.lidar is not None:
                    self.selected_sensor_type = "lidar"
                    print(f"Selected: {self.lidar_config['id']}")
        except pygame.error:
            pass

        try:
            keys = pygame.key.get_pressed()
        except pygame.error:
            return True

        if self.selected_sensor_type == "camera" and self.cameras:
            cfg, camera = self.cameras[self.selected_cam]
            loc = cfg["loc"]
            rot = cfg["rot"]
            moved = rotated = False
            if keys[pygame.K_w]:   loc.x += POSITION_STEP; moved = True
            elif keys[pygame.K_s]: loc.x -= POSITION_STEP; moved = True
            elif keys[pygame.K_a]: loc.y -= POSITION_STEP; moved = True
            elif keys[pygame.K_d]: loc.y += POSITION_STEP; moved = True
            elif keys[pygame.K_q]: loc.z += POSITION_STEP; moved = True
            elif keys[pygame.K_e]: loc.z -= POSITION_STEP; moved = True
            elif keys[pygame.K_LEFT]:  rot.yaw -= ROTATION_STEP; rotated = True
            elif keys[pygame.K_RIGHT]: rot.yaw += ROTATION_STEP; rotated = True
            elif keys[pygame.K_UP]:    rot.pitch += ROTATION_STEP; rotated = True
            elif keys[pygame.K_DOWN]:  rot.pitch -= ROTATION_STEP; rotated = True
            if moved or rotated:
                camera.set_transform(carla.Transform(loc, rot))

        elif self.selected_sensor_type == "lidar" and self.lidar is not None:
            loc = self.lidar_config["loc"]
            moved = False
            if keys[pygame.K_w]:   loc.x += POSITION_STEP; moved = True
            elif keys[pygame.K_s]: loc.x -= POSITION_STEP; moved = True
            elif keys[pygame.K_a]: loc.y -= POSITION_STEP; moved = True
            elif keys[pygame.K_d]: loc.y += POSITION_STEP; moved = True
            elif keys[pygame.K_q]: loc.z += POSITION_STEP; moved = True
            elif keys[pygame.K_e]: loc.z -= POSITION_STEP; moved = True
            if moved:
                self.lidar.set_transform(
                    carla.Transform(loc, self.lidar_config["rot"])
                )

        return True

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def _save_config(self, filename: Optional[str] = None) -> None:
        """Save sensor configuration to a JSON file.

        Args:
            filename: Output path.  Defaults to :attr:`config_file`.
        """
        fn = filename or self.config_file
        data: Dict[str, Any] = {
            "vehicle_type": self.vehicle.type_id if self.vehicle else self.vehicle_bp,
            "cameras": [
                {
                    "id": cfg["id"],
                    "location": [
                        round(cfg["loc"].x, 3),
                        round(cfg["loc"].y, 3),
                        round(cfg["loc"].z, 3),
                    ],
                    "rotation": [
                        round(cfg["rot"].pitch, 3),
                        round(cfg["rot"].yaw, 3),
                        round(cfg["rot"].roll, 3),
                    ],
                    "fov": cfg["fov"],
                }
                for cfg, _ in self.cameras
            ],
            "lidar": {
                "id": self.lidar_config["id"],
                "location": [
                    round(self.lidar_config["loc"].x, 3),
                    round(self.lidar_config["loc"].y, 3),
                    round(self.lidar_config["loc"].z, 3),
                ],
                "rotation": [
                    round(self.lidar_config["rot"].pitch, 3),
                    round(self.lidar_config["rot"].yaw, 3),
                    round(self.lidar_config["rot"].roll, 3),
                ],
                "channels": 32,
                "range": 100.0,
                "points_per_second": 56000,
                "rotation_frequency": 10.0,
            },
        }
        with open(fn, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Configuration saved to {fn}")

    def _load_config(self, filename: Optional[str] = None) -> None:
        """Load sensor configuration from a JSON file.

        If FOV, camera count, or vehicle type changed, actors are respawned.
        Otherwise only transforms are updated without respawning.

        Args:
            filename: Input path.  Defaults to :attr:`config_file`.
        """
        fn = filename or self.config_file
        if not os.path.exists(fn):
            print(f"Config file not found: {fn}")
            return
        try:
            with open(fn) as f:
                data = json.load(f)
        except Exception as exc:
            print(f"Failed to load config: {exc}")
            return

        loaded_type = data.get("vehicle_type", self.vehicle_bp)
        current_type = self.vehicle.type_id if self.vehicle else self.vehicle_bp
        type_changed = loaded_type != current_type

        loaded_cams = data.get("cameras", [])
        fov_changed = any(
            i < len(self.cameras) and self.cameras[i][0]["fov"] != c.get("fov")
            for i, c in enumerate(loaded_cams)
        )
        count_changed = len(loaded_cams) != len(self.cameras)

        if type_changed or fov_changed or count_changed:
            print("Respawning sensors (vehicle/FOV/count changed)...")
            self._respawn_sensors(loaded_type, loaded_cams, data)
        else:
            # Only position/rotation changed — update transforms in-place.
            for i, cam_data in enumerate(loaded_cams):
                if i >= len(self.cameras):
                    break
                cfg, camera = self.cameras[i]
                cfg["loc"] = carla.Location(*cam_data["location"])
                cfg["rot"] = carla.Rotation(*cam_data["rotation"])
                if "fov" in cam_data:
                    cfg["fov"] = cam_data["fov"]
                camera.set_transform(carla.Transform(cfg["loc"], cfg["rot"]))
            if "lidar" in data and self.lidar:
                ld = data["lidar"]
                self.lidar_config["loc"] = carla.Location(*ld["location"])
                self.lidar_config["rot"] = carla.Rotation(*ld["rotation"])
                self.lidar.set_transform(
                    carla.Transform(self.lidar_config["loc"], self.lidar_config["rot"])
                )
        print(f"Configuration loaded from {fn}")

    def _respawn_sensors(
        self,
        new_vehicle_type: str,
        cam_data_list: List[Dict],
        data: Dict,
    ) -> None:
        """Destroy existing sensors (and optionally the vehicle) then respawn.

        Args:
            new_vehicle_type: Blueprint ID to use for the new vehicle.
            cam_data_list: Camera records from the loaded JSON.
            data: Full loaded JSON dict.
        """
        current_type = self.vehicle.type_id if self.vehicle else self.vehicle_bp
        type_changed = new_vehicle_type != current_type

        # Stop and destroy cameras.
        for _, cam in self.cameras:
            try: cam.stop()
            except Exception: pass
            try: cam.destroy()
            except Exception: pass
        self.cameras = []

        # Destroy LiDAR.
        if self.lidar is not None:
            try: self.lidar.destroy()
            except Exception: pass
            self.lidar = None

        if type_changed:
            self.vehicle_bp = new_vehicle_type
            if self.vehicle is not None:
                try: self.vehicle.destroy()
                except Exception: pass
                self.vehicle = None
            self.world.tick()
            time.sleep(0.5)
            self.spawn_vehicle()

        # Rebuild cam_configs from JSON.
        self.cam_configs = [
            {
                "id": c["id"],
                "loc": carla.Location(*c["location"]),
                "rot": carla.Rotation(*c["rotation"]),
                "fov": c.get("fov", 75),
            }
            for c in cam_data_list
        ]
        if "lidar" in data:
            ld = data["lidar"]
            self.lidar_config = {
                "id": ld["id"],
                "loc": carla.Location(*ld["location"]),
                "rot": carla.Rotation(*ld["rotation"]),
            }

        self.world.tick()
        time.sleep(0.3)
        self.spawn_cameras()
        if "lidar" in data:
            self.spawn_lidar()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Initialise and execute the interactive sensor-positioning loop."""
        print("Initialising Interactive Sensor Tool...")
        self.destroy_all_vehicles()
        self.spawn_vehicle()
        self.spawn_cameras()
        self.spawn_lidar()
        _print_help()

        running = True
        while running:
            self.clock.tick(30)
            running = self._handle_input()
            self._draw_camera_debug()
            self._draw_lidar_debug()
            time.sleep(0.05)

    def cleanup(self) -> None:
        """Stop and destroy all sensors and the vehicle, then quit pygame."""
        for _, cam in self.cameras:
            try: cam.stop()
            except Exception: pass
            try:
                if cam.is_alive: cam.destroy()
            except Exception: pass

        if self.lidar is not None:
            try:
                if self.lidar.is_alive: self.lidar.destroy()
            except Exception: pass

        if self.vehicle is not None:
            try:
                if self.vehicle.is_alive: self.vehicle.destroy()
            except Exception: pass

        self.cameras = []
        self.lidar = None
        self.vehicle = None

        try:
            self.world.tick()
        except Exception:
            pass
        pygame.quit()


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _ctrl_held() -> bool:
    """Return ``True`` if either Ctrl key is currently held."""
    keys = pygame.key.get_pressed()
    return bool(keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL])


def _print_help() -> None:
    """Print keyboard controls to stdout."""
    print(
        "\n"
        + "=" * 60
        + "\n"
        "INTERACTIVE SENSOR POSITIONING TOOL\n"
        + "=" * 60
        + "\n"
        "Position  W/S=forward/back  A/D=left/right  Q/E=up/down\n"
        "Rotation  ←/→=yaw  ↑/↓=pitch  (cameras only)\n"
        "Select    1-9,0,-,=  cameras   L=lidar\n"
        "Files     Ctrl+S  save   Ctrl+L  load\n"
        "Capture   C  capture images\n"
        "Quit      ESC\n"
        + "=" * 60
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Interactive sensor-positioning tool for CARLA.",
    )
    parser.add_argument("--host", default="localhost", help="CARLA host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument(
        "--vehicle",
        default=DEFAULT_VEHICLE_BP,
        dest="vehicle_bp",
        metavar="BLUEPRINT",
        help=f"Vehicle blueprint ID (default: {DEFAULT_VEHICLE_BP})",
    )
    parser.add_argument(
        "--config",
        default="sensor_config.json",
        help="JSON config file to load on startup / save to (default: sensor_config.json)",
    )
    args = parser.parse_args()

    tool = InteractiveSensorTool(
        host=args.host,
        port=args.port,
        vehicle_bp=args.vehicle_bp,
        config_file=args.config,
    )
    try:
        tool.run()
    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        tool.cleanup()


if __name__ == "__main__":
    main()
