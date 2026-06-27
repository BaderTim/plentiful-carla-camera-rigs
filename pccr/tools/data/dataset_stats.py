#!/usr/bin/env python3
"""
Dataset statistics tool for nuScenes-format PCCR datasets.

Computes and visualises annotation, ego-motion, and object-speed statistics
across one or more v1.0-* splits found inside a nuScenes-style dataset root.

Usage::

    python3 tools/data/dataset_stats.py --dataset_path output/data/R1
    python3 tools/data/dataset_stats.py --dataset_path output/data/R1 --no_plots
    python3 tools/data/dataset_stats.py --dataset_path output/data/R1-c10 \\
        --output_dir output/stats/R1-c10
"""

from __future__ import annotations

import json
import math
import os
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Category / attribute display-name maps
# ---------------------------------------------------------------------------

CAT_NAME_MAP: Dict[str, str] = {
    "car": "Car",
    "truck": "Truck",
    "bus": "Bus",
    "motorcycle": "Motorcycle",
    "bicycle": "Bicycle",
    "adult": "Adult",
    "child": "Child",
    "traffic_light": "Traffic Light",
    "traffic_sign": "Traffic Sign",
}

ATTR_NAME_MAP: Dict[str, str] = {
    "stopped": "Stopped",
    "moving": "Moving",
    "parked": "Parked",
    "adult_standing": "Standing",
    "adult_moving": "Moving",
    "with_rider": "With Rider",
    "without_rider": "Without Rider",
}


# ---------------------------------------------------------------------------
# Name helpers
# ---------------------------------------------------------------------------

def clean_cat_name(name: str) -> str:
    """Return human-readable category name.

    Args:
        name: Raw category name from the dataset JSON.

    Returns:
        Display-friendly category name.
    """
    return CAT_NAME_MAP.get(name, name.split(".")[-1].replace("_", " ").title())


def clean_attr_name(name: str) -> str:
    """Return human-readable attribute name.

    Args:
        name: Raw attribute name from the dataset JSON.

    Returns:
        Display-friendly attribute name.
    """
    return ATTR_NAME_MAP.get(name, name.split(".")[-1].replace("_", " ").title())


# ---------------------------------------------------------------------------
# Console table formatter
# ---------------------------------------------------------------------------

def format_table(title: str, headers: List[str], rows: List[List]) -> None:
    """Print a fixed-width ASCII table to stdout.

    Args:
        title: Table heading line.
        headers: Column header labels.
        rows: Each row is a list of cell values (converted to ``str``).
    """
    print(f"\n--- {title} ---")
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)))

    header_str = " | ".join(f"{h:<{widths[i]}}" for i, h in enumerate(headers))
    print(header_str)
    print("-" * len(header_str))
    for row in rows:
        print(" | ".join(f"{str(val):<{widths[i]}}" for i, val in enumerate(row)))


# ---------------------------------------------------------------------------
# Core statistics computation
# ---------------------------------------------------------------------------

def compute_stats(
    version_dir: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Compute all statistics for a single v1.0-* split.

    Args:
        version_dir: Path to the split directory containing the ``*.json``
            table files (e.g. ``output/data/R1/v1.0-mini``).

    Returns:
        A 2-tuple of ``(stats, raw_data)`` where *stats* is a nested dict of
        aggregated metrics and *raw_data* contains the raw lists used to
        generate further plots.
    """
    version_dir = Path(version_dir)

    def load_json(filename: str) -> List[Dict]:
        path = version_dir / filename
        if not path.exists():
            return []
        with open(path) as f:
            return json.load(f)

    # Load all tables.
    scenes = load_json("scene.json")
    samples = load_json("sample.json")
    sample_data = load_json("sample_data.json")
    annotations = load_json("sample_annotation.json")
    instances = load_json("instance.json")
    categories = load_json("category.json")
    sensors = load_json("sensor.json")
    calibrated_sensors = load_json("calibrated_sensor.json")
    visibility = load_json("visibility.json")
    attributes = load_json("attribute.json")
    ego_poses = load_json("ego_pose.json")

    # Build lookup indices.
    inst_to_cat = {i["token"]: i["category_token"] for i in instances}
    cat_to_name = {c["token"]: c["name"] for c in categories}
    vis_to_level = {v["token"]: v.get("level", v.get("name", "unknown")) for v in visibility}
    attr_to_name = {a["token"]: a["name"] for a in attributes}
    sensor_to_ch = {s["token"]: s["channel"] for s in sensors}
    calib_to_sensor = {cs["token"]: cs["sensor_token"] for cs in calibrated_sensors}

    sample_ts = {s["token"]: s["timestamp"] for s in samples}
    ts_to_ego = {ep["timestamp"]: ep for ep in ego_poses}

    # --- Basic counts -------------------------------------------------------
    stats: Dict[str, Any] = {
        "overview": {
            "num_scenes": len(scenes),
            "num_samples": len(samples),
            "num_sample_data": len(sample_data),
            "num_annotations": len(annotations),
            "num_instances": len(instances),
            "num_categories": len(categories),
            "num_sensors": len(sensors),
        }
    }

    # --- Category distribution ----------------------------------------------
    cat_counts: Counter = Counter()
    for ann in annotations:
        cat_token = inst_to_cat.get(ann["instance_token"])
        cat_name = clean_cat_name(cat_to_name.get(cat_token, "unknown"))
        cat_counts[cat_name] += 1
    stats["categories"] = dict(cat_counts)

    # --- Visibility distribution ---------------------------------------------
    vis_counts = Counter(
        vis_to_level.get(ann["visibility_token"], "unknown") for ann in annotations
    )
    stats["visibility"] = dict(vis_counts)

    # --- Attribute distribution ---------------------------------------------
    attr_counts: Counter = Counter()
    for ann in annotations:
        for attr_token in ann.get("attribute_tokens", []):
            attr_name = clean_attr_name(attr_to_name.get(attr_token, "unknown"))
            attr_counts[attr_name] += 1
    stats["attributes"] = dict(attr_counts)

    # --- Sensor distribution ------------------------------------------------
    sensor_counts: Counter = Counter()
    for sd in sample_data:
        calib_token = sd["calibrated_sensor_token"]
        s_token = calib_to_sensor.get(calib_token)
        channel = sensor_to_ch.get(s_token, "unknown")
        sensor_counts[channel] += 1
    stats["sensors"] = dict(sensor_counts)

    # --- Ego speed ----------------------------------------------------------
    ego_speeds: List[float] = []
    for ep in ego_poses:
        if "velocity" in ep:
            vx, vy, vz = ep["velocity"]
            ego_speeds.append(math.sqrt(vx**2 + vy**2 + vz**2))

    if ego_speeds:
        stats["ego_speed"] = {
            "min": round(float(min(ego_speeds)), 2),
            "max": round(float(max(ego_speeds)), 2),
            "mean": round(float(sum(ego_speeds) / len(ego_speeds)), 2),
            "median": round(float(np.median(ego_speeds)), 2),
        }

    # --- Distance / size / speed per annotation ----------------------------
    distances: List[float] = []
    sizes_by_cat: defaultdict = defaultdict(list)
    ans_by_inst: defaultdict = defaultdict(list)
    all_volumes: List[float] = []
    all_cat_names: List[str] = []

    for ann in annotations:
        s_token = ann["sample_token"]
        ts = sample_ts.get(s_token)
        ego_pose = ts_to_ego.get(ts)

        if ego_pose:
            ex, ey, ez = ego_pose["translation"]
            ax, ay, az = ann["translation"]
            distances.append(math.sqrt((ax - ex) ** 2 + (ay - ey) ** 2 + (az - ez) ** 2))

        cat_token = inst_to_cat.get(ann["instance_token"])
        cat_name = clean_cat_name(cat_to_name.get(cat_token, "unknown"))
        sizes_by_cat[cat_name].append(ann["size"])

        w, lv, h = ann["size"]
        all_volumes.append(w * lv * h)
        all_cat_names.append(cat_name)
        ans_by_inst[ann["instance_token"]].append({"ts": ts, "pos": ann["translation"]})

    if distances:
        stats["distance_to_ego"] = {
            "min": round(float(min(distances)), 2),
            "max": round(float(max(distances)), 2),
            "mean": round(float(sum(distances) / len(distances)), 2),
            "p95": round(float(np.percentile(distances, 95)), 2),
        }

    # Size stats per category.
    stats["size_dist"] = {}
    for cat, sizes in sizes_by_cat.items():
        wv = [s[0] for s in sizes]
        lv = [s[1] for s in sizes]
        hv = [s[2] for s in sizes]
        stats["size_dist"][cat] = {
            "avg_wlh": [
                round(float(sum(wv) / len(wv)), 2),
                round(float(sum(lv) / len(lv)), 2),
                round(float(sum(hv) / len(hv)), 2),
            ],
            "count": len(sizes),
        }

    # Object speed from successive annotation positions.
    obj_speeds_by_cat: defaultdict = defaultdict(list)
    for inst_token, timeline in ans_by_inst.items():
        if len(timeline) < 2:
            continue
        timeline.sort(key=lambda x: x["ts"])
        cat_token = inst_to_cat.get(inst_token)
        cat_name = clean_cat_name(cat_to_name.get(cat_token, "unknown"))
        for i in range(len(timeline) - 1):
            t1, t2 = timeline[i]["ts"], timeline[i + 1]["ts"]
            p1, p2 = timeline[i]["pos"], timeline[i + 1]["pos"]
            dt = (t2 - t1) / 1e6
            if dt > 0:
                dist = math.sqrt(sum((p2[j] - p1[j]) ** 2 for j in range(3)))
                obj_speeds_by_cat[cat_name].append(dist / dt)

    stats["object_speeds"] = {}
    for cat, speeds in obj_speeds_by_cat.items():
        if speeds:
            stats["object_speeds"][cat] = {
                "max": round(float(max(speeds)), 2),
                "mean": round(float(sum(speeds) / len(speeds)), 2),
                "count": len(speeds),
            }

    # Aggregates in overview.
    if samples:
        stats["overview"]["avg_annotations_per_sample"] = round(
            len(annotations) / len(samples), 2
        )
    else:
        stats["overview"]["avg_annotations_per_sample"] = 0

    if ego_speeds:
        stats["overview"]["ego_speed_avg"] = round(float(sum(ego_speeds) / len(ego_speeds)), 2)
        stats["overview"]["ego_speed_min"] = round(float(min(ego_speeds)), 2)
        stats["overview"]["ego_speed_max"] = round(float(max(ego_speeds)), 2)

    # Temporal (scene durations).
    durations: List[float] = []
    for scene in scenes:
        first = next((s for s in samples if s["token"] == scene["first_sample_token"]), None)
        last = next((s for s in samples if s["token"] == scene["last_sample_token"]), None)
        if first and last:
            durations.append((last["timestamp"] - first["timestamp"]) / 1e6)

    if durations:
        stats["temporal"] = {
            "total_duration_sec": round(sum(durations), 2),
            "avg_scene_duration_sec": round(sum(durations) / len(durations), 2),
        }

    raw_data = {
        "ego_speeds": ego_speeds,
        "distances": distances,
        "volumes": all_volumes,
        "categories": all_cat_names,
        "sizes": [ann["size"] for ann in annotations],
        "sizes_by_cat": sizes_by_cat,
        "obj_speeds": [s for speeds in obj_speeds_by_cat.values() for s in speeds],
        "obj_speeds_by_cat": obj_speeds_by_cat,
        "vis_counts": vis_counts,
        "attr_counts": attr_counts,
        "sensor_counts": sensor_counts,
        "durations": durations,
    }
    return stats, raw_data


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------

def generate_plots(stats: Dict[str, Any], raw_data: Dict[str, Any], output_dir: str) -> None:
    """Generate CVPR-style publication plots and save them to *output_dir*.

    Args:
        stats: Aggregated statistics dict from :func:`compute_stats` or
            :func:`merge_results`.
        raw_data: Raw array dict from the same source.
        output_dir: Directory for output ``.png`` files.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.size": 26,
            "axes.titlesize": 32,
            "axes.labelsize": 32,
            "xtick.labelsize": 28,
            "ytick.labelsize": 28,
            "legend.fontsize": 28,
            "figure.titlesize": 36,
        }
    )

    def format_vol(x: float, _pos) -> str:
        if x == 0:
            return "0m³"
        val = f"{x:.4f}".rstrip("0").rstrip(".")
        return f"{val}m³"

    unique_cats = sorted(set(raw_data["categories"]))
    category_palette = dict(zip(unique_cats, sns.color_palette("husl", len(unique_cats))))

    # 1. Category distribution.
    plt.figure(figsize=(16, 9))
    cat_counts = Counter(raw_data["categories"])
    cats, counts = zip(*sorted(cat_counts.items(), key=lambda x: x[1], reverse=True))
    sns.barplot(x=list(counts), y=list(cats), hue=list(cats), palette=category_palette, legend=False)
    plt.title("Annotation Count per Category")
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "category_dist.png"), dpi=300)
    plt.close()

    # 2. Distance distributions.
    if raw_data["distances"]:
        plt.figure(figsize=(16, 9))
        sns.histplot(raw_data["distances"], bins=50, kde=True, color="steelblue")
        plt.title("Accumulated Distribution of Object Distances to Ego")
        plt.xlabel("Distance (m)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "distance_dist_accumulated.png"), dpi=300)
        plt.close()

        df_dist = pd.DataFrame({"Distance": raw_data["distances"], "Category": raw_data["categories"]})
        plt.figure(figsize=(16, 9))
        sns.kdeplot(data=df_dist, x="Distance", hue="Category", common_norm=False, fill=True, alpha=0.3, palette=category_palette)
        plt.title("Object Distances to Ego per Category")
        plt.xlabel("Distance (m)")
        plt.ylabel("Density")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "distance_dist_per_cat.png"), dpi=300)
        plt.close()

    # 3. Ego speed (km/h).
    if raw_data["ego_speeds"]:
        plt.figure(figsize=(16, 9))
        sns.histplot([s * 3.6 for s in raw_data["ego_speeds"]], bins=30, kde=True, color="steelblue")
        plt.title("Ego Vehicle Speed Distribution")
        plt.xlabel("Speed (km/h)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "ego_speed_dist.png"), dpi=300)
        plt.close()

    # 4. Bounding-box volume.
    if raw_data["volumes"]:
        plt.figure(figsize=(16, 9))
        ax = sns.histplot(raw_data["volumes"], bins=50, kde=True, color="steelblue", log_scale=True)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_vol))
        plt.title("Overview: Bounding Box Volume Distribution (Log-Scale)")
        plt.xlabel("Volume (m$^3$)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "volume_dist_overview.png"), dpi=300)
        plt.close()

        df_vol = pd.DataFrame({"Volume": raw_data["volumes"], "Category": raw_data["categories"]})
        plt.figure(figsize=(16, 9))
        ax = sns.boxplot(data=df_vol, x="Volume", y="Category", hue="Category", palette=category_palette, legend=False)
        plt.xscale("log")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_vol))
        plt.title("Bounding Box Volume per Category (Log-Scale)")
        plt.xlabel("Volume (m$^3$)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "volume_dist_per_cat.png"), dpi=300)
        plt.close()

    # 5. Object speed (km/h, capped at 150).
    if raw_data["obj_speeds"]:
        plt.figure(figsize=(16, 9))
        sns.histplot(
            [s * 3.6 for s in raw_data["obj_speeds"]],
            binrange=(0, 150), bins=50, kde=True, color="steelblue",
        )
        plt.title("Other Objects Speed Distribution (Limited to 150 km/h)")
        plt.xlabel("Speed (km/h)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "object_speed_dist.png"), dpi=300)
        plt.close()

    # 6. Width vs length scatter.
    if raw_data.get("sizes"):
        plt.figure(figsize=(16, 9))
        all_w = [s[0] for s in raw_data["sizes"]]
        all_l = [s[1] for s in raw_data["sizes"]]
        df = pd.DataFrame({"Width": all_w, "Length": all_l, "Category": raw_data["categories"]})
        if len(df) > 5000:
            df = df.sample(5000)
        sns.scatterplot(data=df, x="Width", y="Length", hue="Category", alpha=0.5, s=10, palette=category_palette)
        plt.title("Object Size Distribution (Width vs Length)")
        plt.xlabel("Width (m)")
        plt.ylabel("Length (m)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "size_scatter.png"), dpi=300)
        plt.close()

    print(f"Plots and stats saved to {output_dir}")


# ---------------------------------------------------------------------------
# Multi-split merge
# ---------------------------------------------------------------------------

def merge_results(
    all_results: List[Tuple[str, Dict[str, Any], Dict[str, Any]]],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Merge statistics from multiple splits into a single summary.

    Args:
        all_results: List of ``(split_name, stats, raw_data)`` tuples.

    Returns:
        A 2-tuple ``(summary_stats, merged_raw)`` or ``(None, None)`` when
        *all_results* is empty.
    """
    if not all_results:
        return None, None

    merged_raw: Dict[str, Any] = {
        "ego_speeds": [],
        "distances": [],
        "volumes": [],
        "categories": [],
        "sizes": [],
        "sizes_by_cat": defaultdict(list),
        "obj_speeds": [],
        "obj_speeds_by_cat": defaultdict(list),
        "vis_counts": Counter(),
        "attr_counts": Counter(),
        "sensor_counts": Counter(),
        "durations": [],
    }

    total_scenes = total_samples = total_sd = total_ann = total_inst = 0

    for _v, st, raw in all_results:
        for key in ("ego_speeds", "distances", "volumes", "categories", "sizes", "obj_speeds", "durations"):
            merged_raw[key].extend(raw[key])
        for counter in ("vis_counts", "attr_counts", "sensor_counts"):
            merged_raw[counter].update(raw[counter])
        for cat, speeds in raw["obj_speeds_by_cat"].items():
            merged_raw["obj_speeds_by_cat"][cat].extend(speeds)
        for cat, sizes in raw["sizes_by_cat"].items():
            merged_raw["sizes_by_cat"][cat].extend(sizes)

        ov = st["overview"]
        total_scenes += ov["num_scenes"]
        total_samples += ov["num_samples"]
        total_sd += ov["num_sample_data"]
        total_ann += ov["num_annotations"]
        total_inst += ov["num_instances"]

    summary_stats: Dict[str, Any] = {
        "overview": {
            "num_scenes": total_scenes,
            "num_samples": total_samples,
            "num_sample_data": total_sd,
            "num_annotations": total_ann,
            "num_instances": total_inst,
            "num_categories": len(merged_raw["sizes_by_cat"]),
            "num_sensors": len(merged_raw["sensor_counts"]),
            "avg_annotations_per_sample": round(total_ann / total_samples, 2) if total_samples > 0 else 0,
        },
        "categories": dict(Counter(merged_raw["categories"])),
        "visibility": dict(merged_raw["vis_counts"]),
        "attributes": dict(merged_raw["attr_counts"]),
        "sensors": dict(merged_raw["sensor_counts"]),
    }

    if merged_raw["ego_speeds"]:
        es = merged_raw["ego_speeds"]
        summary_stats["overview"].update(
            {
                "ego_speed_avg": round(float(sum(es) / len(es)), 2),
                "ego_speed_min": round(float(min(es)), 2),
                "ego_speed_max": round(float(max(es)), 2),
            }
        )
        summary_stats["ego_speed"] = {
            "min": round(float(min(es)), 2),
            "max": round(float(max(es)), 2),
            "mean": round(float(sum(es) / len(es)), 2),
            "median": round(float(np.median(es)), 2),
        }

    if merged_raw["distances"]:
        d = merged_raw["distances"]
        summary_stats["distance_to_ego"] = {
            "min": round(float(min(d)), 2),
            "max": round(float(max(d)), 2),
            "mean": round(float(sum(d) / len(d)), 2),
            "p95": round(float(np.percentile(d, 95)), 2),
        }

    summary_stats["size_dist"] = {}
    for cat, sizes in merged_raw["sizes_by_cat"].items():
        wv = [s[0] for s in sizes]
        lv = [s[1] for s in sizes]
        hv = [s[2] for s in sizes]
        summary_stats["size_dist"][cat] = {
            "avg_wlh": [
                round(float(sum(wv) / len(wv)), 2),
                round(float(sum(lv) / len(lv)), 2),
                round(float(sum(hv) / len(hv)), 2),
            ],
            "count": len(sizes),
        }

    summary_stats["object_speeds"] = {}
    for cat, speeds in merged_raw["obj_speeds_by_cat"].items():
        if speeds:
            summary_stats["object_speeds"][cat] = {
                "max": round(float(max(speeds)), 2),
                "mean": round(float(sum(speeds) / len(speeds)), 2),
                "count": len(speeds),
            }

    if merged_raw["durations"]:
        durs = merged_raw["durations"]
        summary_stats["temporal"] = {
            "total_duration_sec": round(sum(durs), 2),
            "avg_scene_duration_sec": round(sum(durs) / len(durs), 2),
        }

    return summary_stats, merged_raw


# ---------------------------------------------------------------------------
# Console print helpers
# ---------------------------------------------------------------------------

def print_stats(stats: Dict[str, Any]) -> None:
    """Print all statistics to stdout in human-readable table form.

    Args:
        stats: Aggregated statistics dict from :func:`compute_stats` or
            :func:`merge_results`.
    """
    if not stats:
        return

    ov = stats["overview"]
    overview_rows = [
        ["Scenes", ov["num_scenes"]],
        ["Samples (Key-frames)", ov["num_samples"]],
        ["Sample Data (Files)", ov["num_sample_data"]],
        ["Annotations", ov["num_annotations"]],
        ["Instances", ov["num_instances"]],
        ["Categories", ov["num_categories"]],
        ["Sensors", ov["num_sensors"]],
        ["Avg Annotations/Sample", ov["avg_annotations_per_sample"]],
    ]
    if "ego_speed_avg" in ov:
        overview_rows += [
            ["Ego Speed Mean (m/s)", ov["ego_speed_avg"]],
            ["Ego Speed Min (m/s)", ov["ego_speed_min"]],
            ["Ego Speed Max (m/s)", ov["ego_speed_max"]],
        ]
    format_table("Dataset Overview", ["Metric", "Value"], overview_rows)

    cat_rows = sorted([[k, v] for k, v in stats["categories"].items()], key=lambda x: x[1], reverse=True)
    format_table("Annotations per Category", ["Category", "Count"], cat_rows)

    if "distance_to_ego" in stats:
        d = stats["distance_to_ego"]
        format_table(
            "Distance to Ego (m)",
            ["Metric", "Value"],
            [["Min", d["min"]], ["Max", d["max"]], ["Mean", d["mean"]], ["p95", d["p95"]]],
        )

    if "size_dist" in stats:
        size_rows = [
            [cat, f"{s['avg_wlh'][0]}x{s['avg_wlh'][1]}x{s['avg_wlh'][2]}", s["count"]]
            for cat, s in stats["size_dist"].items()
        ]
        format_table("Average Object Sizes (WxLxH)", ["Category", "Avg Size", "Count"], sorted(size_rows))

    if "ego_speed" in stats:
        es = stats["ego_speed"]
        format_table(
            "Ego Vehicle Speed (m/s)",
            ["Metric", "Value"],
            [["Min", es["min"]], ["Max", es["max"]], ["Mean", es["mean"]], ["Median", es["median"]]],
        )

    if "object_speeds" in stats:
        speed_rows = [
            [cat, s["mean"], s["max"], s["count"]]
            for cat, s in stats["object_speeds"].items()
        ]
        format_table("Object Speeds (m/s)", ["Category", "Mean", "Max", "Samples"], sorted(speed_rows))

    sensor_rows = sorted([[k, v] for k, v in stats["sensors"].items()], key=lambda x: x[1], reverse=True)
    format_table("Sample Data per Sensor", ["Sensor", "Count"], sensor_rows)

    vis_rows = sorted([[k, v] for k, v in stats["visibility"].items()], key=lambda x: str(x[0]))
    format_table("Annotations per Visibility Level", ["Visibility", "Count"], vis_rows)

    if stats.get("attributes"):
        attr_rows = sorted([[k, v] for k, v in stats["attributes"].items()], key=lambda x: x[1], reverse=True)
        format_table("Annotations per Attribute", ["Attribute", "Count"], attr_rows)

    if "temporal" in stats:
        t = stats["temporal"]
        format_table(
            "Temporal Statistics",
            ["Metric", "Value"],
            [["Total Duration (s)", t["total_duration_sec"]], ["Avg Scene Duration (s)", t["avg_scene_duration_sec"]]],
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate statistics for nuScenes-format PCCR datasets."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="output/data/R1",
        help="Path to the dataset root directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/stats",
        help="Directory for JSON stats and plot images",
    )
    parser.add_argument("--no_plots", action="store_true", help="Disable plot generation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Analyzing dataset at: {args.dataset_path}")

    # Discover v1.0-* sub-directories (or treat root as a split dir).
    versions = [
        d
        for d in os.listdir(args.dataset_path)
        if os.path.isdir(Path(args.dataset_path) / d) and d.startswith("v1.0")
    ]
    if not versions:
        if (Path(args.dataset_path) / "sample.json").exists():
            versions = ["."]
        else:
            print(f"Error: No v1.0-* directories or nuScenes JSON files found in {args.dataset_path}")
            return

    all_results = []
    for version in sorted(versions):
        v_path = Path(args.dataset_path) / version
        v_name = version if version != "." else "default"
        print(f"\n>>> Processing split: {v_name}")

        stats, raw_data = compute_stats(v_path)
        all_results.append((v_name, stats, raw_data))

        split_out = os.path.join(args.output_dir, v_name)
        os.makedirs(split_out, exist_ok=True)
        with open(os.path.join(split_out, "stats.json"), "w") as f:
            json.dump(stats, f, indent=4)

        if not args.no_plots:
            generate_plots(stats, raw_data, split_out)

    if len(all_results) > 1:
        print(f"\n>>> Generating summary for {len(all_results)} splits")
        summary_stats, summary_raw = merge_results(all_results)
        if summary_stats:
            print_stats(summary_stats)
            summary_json = os.path.join(args.output_dir, "summary_stats.json")
            with open(summary_json, "w") as f:
                json.dump(summary_stats, f, indent=4)
            print(f"\nSummary statistics saved to {summary_json}")
            if not args.no_plots:
                generate_plots(summary_stats, summary_raw, args.output_dir)
    elif all_results:
        print_stats(all_results[0][1])
    else:
        print("No valid data processed.")


if __name__ == "__main__":
    main()
