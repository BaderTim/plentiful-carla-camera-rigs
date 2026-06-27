#!/usr/bin/env python3
"""
Merge multiple per-rig nuScenes datasets into a single BEVDet PKL file.

Assigns each scene to exactly one rig (round-robin), collects and enriches
sample infos with BEVDet-style adjacent annotations, rebuilds temporal
``prev`` pointers across the merged list, and writes the combined PKL.

Usage::

    python3 tools/data/create_data_mixed.py \\
        --root-dir /data \\
        --rigs R1 R1-c6 R1-c10 R1-f R1-r R1-t \\
        --version v1.0-mini \\
        --out-dir /data/R_mixed
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Ensure project root is importable when the script is run directly.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits as nuscenes_splits

from lib.data_converter import nuscenes_converter
from lib.data_converter.bevdet_annotations import (
    DETECTION_CLASSES,
    MAP_CATEGORY_TO_DETECTION,
    get_gt,
)


# ---------------------------------------------------------------------------
# Per-sample enrichment
# ---------------------------------------------------------------------------

def process_info_with_adj(
    nusc: NuScenes,
    info: dict,
    sample: dict,
) -> dict:
    """Enrich *info* in place with BEVDet-style adjacent annotation data.

    Computes per-annotation velocities, transforms boxes into ego coordinates,
    and attaches the scene token.

    Args:
        nusc: Loaded :class:`~nuscenes.NuScenes` object for the rig.
        info: Sample info dict from ``_fill_trainval_infos``.
        sample: Corresponding nuScenes sample record.

    Returns:
        The mutated *info* dict (for convenience).
    """
    ann_infos = []
    for ann_token in sample["anns"]:
        ann_info = nusc.get("sample_annotation", ann_token)
        velocity = nusc.box_velocity(ann_info["token"])
        if np.any(np.isnan(velocity)):
            velocity = np.zeros(3)
        ann_info["velocity"] = velocity
        ann_infos.append(ann_info)

    info["ann_infos"] = ann_infos
    info["ann_infos"] = get_gt(
        info,
        detection_classes=DETECTION_CLASSES,
        category_mapping=MAP_CATEGORY_TO_DETECTION,
    )
    info["scene_token"] = sample["scene_token"]
    return info


# ---------------------------------------------------------------------------
# Temporal pointer reconstruction
# ---------------------------------------------------------------------------

def rebuild_prev(infos: list[dict]) -> None:
    """Rewrite ``info["prev"]`` integer indices after cross-rig merging.

    After merging infos from multiple rigs, temporal ``prev`` pointers (which
    store sample tokens in the original ``_fill_trainval_infos``) would be
    stale.  This function converts them to integer list-indices within the
    merged list.

    Args:
        infos: Merged list of sample-info dicts (mutated in place).
    """
    token2idx: dict[str, int] = {info["token"]: i for i, info in enumerate(infos)}
    for info in infos:
        prev_token = info.get("prev_token", "")
        info["prev"] = token2idx.get(prev_token, -1) if prev_token else -1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments and produce the merged PKL output."""
    parser = argparse.ArgumentParser(description="Mixed Data converter arg parser")
    parser.add_argument(
        "--root-dir", type=str, default="/data", help="Root directory containing rig folders"
    )
    parser.add_argument(
        "--rigs", nargs="+", help="List of rig folder names (e.g. R1 R1-c6)"
    )
    parser.add_argument(
        "--version", type=str, default="v1.0-mini", help="Dataset version string"
    )
    parser.add_argument(
        "--out-dir", type=str, default="/data/R_mixed", help="Output directory"
    )
    parser.add_argument(
        "--info-prefix", type=str, default="nuscenes", help="PKL filename prefix"
    )
    parser.add_argument(
        "--no-lidar", action="store_true", help="Skip LiDAR-related processing"
    )
    args = parser.parse_args()

    if not args.rigs:
        args.rigs = ["R1", "R1-c6", "R1-c10", "R1-f", "R1-r", "R1-t"]
        print(f"Using default rigs: {args.rigs}")

    # ------------------------------------------------------------------
    # 1. Load per-rig NuScenes objects and collect scene lists.
    # ------------------------------------------------------------------
    all_rig_scenes: dict[str, list] = {}
    scene_to_rigs: dict[str, list[str]] = defaultdict(list)
    rig_nuscs: dict[str, NuScenes] = {}

    for rig in args.rigs:
        dataroot = os.path.join(args.root_dir, rig)
        if not os.path.exists(dataroot):
            print(f"Skipping {rig}, not found at {dataroot}")
            continue
        try:
            nusc = NuScenes(version=args.version, dataroot=dataroot, verbose=False)
            rig_nuscs[rig] = nusc
            avail_scenes = nuscenes_converter.get_available_scenes(nusc, no_lidar=args.no_lidar)
            all_rig_scenes[rig] = avail_scenes
            for s in avail_scenes:
                scene_to_rigs[s["name"]].append(rig)
        except Exception as exc:
            print(f"Error loading rig {rig}: {exc}")

    # ------------------------------------------------------------------
    # 2. Round-robin scene → rig assignment.
    # ------------------------------------------------------------------
    sorted_scene_names = sorted(scene_to_rigs.keys())
    scene_assignment: dict[str, str] = {}
    for i, name in enumerate(sorted_scene_names):
        options = scene_to_rigs[name]
        chosen = args.rigs[i % len(args.rigs)]
        if chosen not in options:
            chosen = options[0]
        scene_assignment[name] = chosen
        print(f"Scene {name} assigned to {chosen}")

    # ------------------------------------------------------------------
    # 3. Determine train / val split membership.
    # ------------------------------------------------------------------
    all_avail = sorted(scene_to_rigs.keys())
    version = args.version

    if "mini" in version:
        train_names = set(nuscenes_splits.mini_train)
        val_names = set(nuscenes_splits.mini_val)
    elif "test" in version:
        train_names = set(nuscenes_splits.test)
        val_names: set[str] = set()
    else:
        train_names = set(nuscenes_splits.train)
        val_names = set(nuscenes_splits.val)

    # Fallback: if none of the actual split names are present, use name hints.
    if not any(n in train_names for n in all_avail) and not any(n in val_names for n in all_avail):
        train_names = {n for n in all_avail if "train" in n.lower()}
        val_names = {n for n in all_avail if "val" in n.lower()}
        if not train_names and not val_names:
            train_names = set(all_avail)

    # ------------------------------------------------------------------
    # 4. Collect and enrich sample infos per rig.
    # ------------------------------------------------------------------
    merged_train_infos: list[dict] = []
    merged_val_infos: list[dict] = []

    for rig in args.rigs:
        if rig not in rig_nuscs:
            continue
        nusc = rig_nuscs[rig]

        assigned_tokens = {
            s["token"]
            for s in all_rig_scenes[rig]
            if scene_assignment[s["name"]] == rig
        }
        if not assigned_tokens:
            continue

        print(f"\n--- Processing {len(assigned_tokens)} scenes from rig {rig} ---")

        t_tokens = {
            s["token"] for s in all_rig_scenes[rig]
            if s["token"] in assigned_tokens and s["name"] in train_names
        }
        v_tokens = {
            s["token"] for s in all_rig_scenes[rig]
            if s["token"] in assigned_tokens and s["name"] in val_names
        }

        t_infos, v_infos = nuscenes_converter._fill_trainval_infos(
            nusc,
            t_tokens,
            v_tokens,
            test=("test" in version),
            no_lidar=args.no_lidar,
            root_path_prefix=rig,
        )

        # Filter to only the assigned scenes (avoids cross-contamination).
        t_infos = [
            info for info in t_infos
            if nusc.get("sample", info["token"])["scene_token"] in t_tokens
        ]
        v_infos = [
            info for info in v_infos
            if nusc.get("sample", info["token"])["scene_token"] in v_tokens
        ]

        # Enrich with adjacent annotation data.
        print(f"Adding adjacent annotation info for rig {rig}...")
        for info in t_infos:
            sample = nusc.get("sample", info["token"])
            process_info_with_adj(nusc, info, sample)
        for info in v_infos:
            sample = nusc.get("sample", info["token"])
            process_info_with_adj(nusc, info, sample)

        merged_train_infos.extend(t_infos)
        merged_val_infos.extend(v_infos)

    # ------------------------------------------------------------------
    # 5. Rebuild temporal pointers across the merged list.
    # ------------------------------------------------------------------
    print("Rebuilding temporal pointers...")
    rebuild_prev(merged_train_infos)
    rebuild_prev(merged_val_infos)

    # ------------------------------------------------------------------
    # 6. Persist.
    # ------------------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    metadata = {"version": version}

    def _dump(data: dict, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved → {path}")

    if "test" in version:
        _dump(
            {"infos": merged_train_infos, "metadata": metadata},
            os.path.join(args.out_dir, f"{args.info_prefix}_infos_test.pkl"),
        )
    elif "mini" in version:
        _dump(
            {"infos": merged_train_infos + merged_val_infos, "metadata": metadata},
            os.path.join(args.out_dir, f"{args.info_prefix}_infos_mini.pkl"),
        )
    else:
        _dump(
            {"infos": merged_train_infos, "metadata": metadata},
            os.path.join(args.out_dir, f"{args.info_prefix}_infos_train.pkl"),
        )
        _dump(
            {"infos": merged_val_infos, "metadata": metadata},
            os.path.join(args.out_dir, f"{args.info_prefix}_infos_val.pkl"),
        )

    print(
        f"\nMixed dataset: {len(merged_train_infos)} train, "
        f"{len(merged_val_infos)} val samples."
    )
    print(f"Output: {args.out_dir}")


if __name__ == "__main__":
    main()
