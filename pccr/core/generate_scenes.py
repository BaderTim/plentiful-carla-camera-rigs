#!/usr/bin/env python3
"""
Generate a randomised ``scenes.json`` configuration file.

Connects to a running CARLA server to query available spawn points per map,
then writes a JSON file with randomised scene parameters for each split
(train / val / test / mini).

Usage:

    python3 core/generate_scenes.py --output configs/scenes.json
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Ensure project root is importable when the script is run directly.
sys.path.insert(0, str(Path(__file__).parent.parent))

import carla

# ---------------------------------------------------------------------------
# Static configuration
# ---------------------------------------------------------------------------
RANDOM_SEED = 4
random.seed(RANDOM_SEED)

CARLA_HOST = "localhost"
CARLA_PORT = 2000

#: Scene-count and sampling parameters per split.
SPLITS = {
    "train": {
        "num_scenes": 80,
        "maps": ["Town01", "Town02", "Town04", "Town05", "Town10HD"],
        "weather": [
            "ClearNoon", "CloudyNoon", "WetNoon", "WetCloudyNoon",
            "SoftRainNoon", "MidRainyNoon", "HardRainNoon",
            "ClearSunset", "ClearSunset", "CloudySunset",
            "WetSunset", "SoftRainSunset", "MidRainSunset", "HardRainSunset",
        ],
        "traffic_density": [0.2, 0.4, 0.6, 0.8],
    },
    "val": {
        "num_scenes": 20,
        "maps": ["Town01", "Town10HD"],
        "weather": [
            "ClearNoon", "CloudyNoon", "WetNoon", "WetCloudyNoon",
            "SoftRainNoon", "MidRainyNoon", "HardRainNoon",
            "ClearSunset", "ClearSunset", "CloudySunset",
            "WetSunset", "SoftRainSunset", "MidRainSunset", "HardRainSunset",
        ],
        "traffic_density": [0.25, 0.45, 0.65],
    },
    "test": {
        "num_scenes": 40,
        "maps": ["Town03", "Town06", "Town07"],
        "weather": [
            "ClearNoon", "CloudyNoon", "WetNoon", "WetCloudyNoon",
            "SoftRainNoon", "MidRainyNoon", "HardRainNoon",
            "ClearSunset", "ClearSunset", "CloudySunset",
            "WetSunset", "SoftRainSunset", "MidRainSunset", "HardRainSunset",
        ],
        "traffic_density": [0.1, 0.35, 0.42, 0.69, 0.9],
    },
    "mini": {
        "num_scenes": 5,
        "maps": ["Town01", "Town10HD"],
        "weather": ["ClearNoon", "CloudyNoon", "MidRainyNoon", "ClearSunset"],
        "traffic_density": [0.3, 0.5, 0.7],
    },
}


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def query_spawn_point_counts(client: carla.Client) -> dict[str, int]:
    """Return ``{map_name: num_spawn_points}`` by loading each unique map once.

    Args:
        client: Connected CARLA client.

    Returns:
        Mapping from map name to available spawn-point count.
    """
    all_maps: set[str] = set()
    for cfg in SPLITS.values():
        all_maps.update(cfg["maps"])

    counts: dict[str, int] = {}
    for map_name in sorted(all_maps):
        print(f"Loading {map_name} to query spawn points...")
        world = client.load_world(map_name)
        spawn_points = world.get_map().get_spawn_points()
        counts[map_name] = len(spawn_points)
        print(f"  → {len(spawn_points)} spawn points available.")
    return counts


def generate_scenes_config(spawn_point_counts: dict[str, int]) -> dict:
    """Randomise scene parameters and return the full config dict.

    Args:
        spawn_point_counts: Map from map name to spawn-point count
            (from :func:`query_spawn_point_counts`).

    Returns:
        Config dict with ``"statistics"`` and ``"scenes"`` keys, ready for
        JSON serialisation.
    """
    stats: dict = {}
    scenes: dict = {split: [] for split in SPLITS}

    for split, cfg in SPLITS.items():
        counters: dict = defaultdict(Counter)
        for i in range(cfg["num_scenes"]):
            map_choice = random.choice(cfg["maps"])
            weather_choice = random.choice(cfg["weather"])
            density_choice = random.choice(cfg["traffic_density"])
            spawn_idx = random.randint(0, spawn_point_counts[map_choice] - 1)

            scenes[split].append(
                {
                    "id": f"{split}_{i + 1:02d}",
                    "map": map_choice,
                    "weather": weather_choice,
                    "traffic_density": density_choice,
                    "spawn_point": spawn_idx,
                }
            )
            counters["maps"][map_choice] += 1
            counters["weather"][weather_choice] += 1
            counters["traffic_density"][density_choice] += 1

        stats[split] = {
            "num_scenes": cfg["num_scenes"],
            "maps": dict(counters["maps"]),
            "weather": dict(counters["weather"]),
            "traffic_density": dict(counters["traffic_density"]),
        }

    return {"statistics": stats, "scenes": scenes}


def main(output_path: str) -> None:
    """Connect to CARLA, query spawn points, generate and write *output_path*.

    Args:
        output_path: Destination path for the scenes JSON file.
    """
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(30.0)

    spawn_point_counts = query_spawn_point_counts(client)
    result = generate_scenes_config(spawn_point_counts)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Finished! Scenes saved to {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CARLA scenes JSON.")
    parser.add_argument(
        "--output",
        default="configs/scenes.json",
        help="Path to output scenes JSON file.",
    )
    args = parser.parse_args()
    main(args.output)
