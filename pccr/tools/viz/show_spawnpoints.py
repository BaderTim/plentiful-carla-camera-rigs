#!/usr/bin/env python3
"""
Draw CARLA spawn-point indices in the spectator window.

Each spawn point is annotated with its integer index and a direction arrow so
you can identify the correct spawn point for ``scenes.json``.

Usage::

    python3 tools/viz/show_spawnpoints.py
    python3 tools/viz/show_spawnpoints.py --host 192.168.1.10 --port 2001
    python3 tools/viz/show_spawnpoints.py --map Town03
    python3 tools/viz/show_spawnpoints.py --life-time 60
"""

from __future__ import annotations

import argparse
import sys

import carla


def show_spawnpoints(
    host: str = "localhost",
    port: int = 2000,
    map_name: str | None = None,
    life_time: float = 30.0,
    timeout: float = 30.0,
) -> None:
    """Connect to CARLA and annotate every map spawn point.

    Draws the index number and a forward-direction arrow for each spawn
    point directly in the CARLA spectator viewport.

    Args:
        host: CARLA simulator hostname.
        port: CARLA simulator port.
        map_name: If provided, load this map before drawing spawn points
            (e.g. ``"Town03"``).
        life_time: Duration in seconds that the debug annotations remain
            visible (default: 30 s).
        timeout: Client connection timeout in seconds.
    """
    client = carla.Client(host, port)
    client.set_timeout(timeout)
    world = client.get_world()

    if map_name and not world.get_map().name.endswith(map_name):
        print(f"Loading map: {map_name}")
        world = client.load_world(map_name)

    spawn_points = world.get_map().get_spawn_points()
    print(f"Map: {world.get_map().name}")
    print(f"Spawn points: {len(spawn_points)}")

    for i, sp in enumerate(spawn_points):
        world.debug.draw_string(
            sp.location,
            str(i),
            draw_shadow=True,
            color=carla.Color(255, 255, 0),
            life_time=life_time,
        )
        world.debug.draw_arrow(
            sp.location,
            sp.location + sp.get_forward_vector(),
            thickness=0.3,
            arrow_size=0.5,
            color=carla.Color(255, 255, 0),
            life_time=life_time,
        )

    print(
        f"Spawn points drawn for {life_time:.0f}s.  "
        "Open the CARLA spectator window to see them."
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Draw spawn-point indices in the CARLA spectator viewport.",
    )
    parser.add_argument(
        "--host", default="localhost",
        help="CARLA simulator host (default: localhost)",
    )
    parser.add_argument(
        "--port", type=int, default=2000,
        help="CARLA simulator port (default: 2000)",
    )
    parser.add_argument(
        "--map", dest="map_name", default=None,
        help="Load this map before drawing spawn points (e.g. Town03)",
    )
    parser.add_argument(
        "--life-time", type=float, default=30.0,
        help="How long (seconds) annotations remain visible (default: 30)",
    )
    parser.add_argument(
        "--timeout", type=float, default=30.0,
        help="Client connection timeout in seconds (default: 30)",
    )
    args = parser.parse_args()

    try:
        show_spawnpoints(
            host=args.host,
            port=args.port,
            map_name=args.map_name,
            life_time=args.life_time,
            timeout=args.timeout,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
