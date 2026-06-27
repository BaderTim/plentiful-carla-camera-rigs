#!/usr/bin/env python3
"""
Prune trajectory HDF5 files to flatten the ego-vehicle speed distribution.

Problem
-------
Large-scale CARLA recordings tend to produce many scenes where the ego vehicle
(firetruck) idles at spawn or quickly reaches a junction and stops, leading to
an abundance of near-zero-velocity scenes.  When these are used for training
downstream models, the skewed distribution causes overfitting to stationary
behaviour.

Solution
--------
This script reads every ``.h5`` trajectory file, derives a mean ego speed for
each file by differencing consecutive ``ego_vehicle/locations`` frames, bins
files by mean speed, and down-samples over-represented bins to level the
distribution.  Each split (``trainval``, ``test``, ``mini``) is processed
independently so cross-split ratios are preserved.

Survivors are hard-linked / copied to ``--output``.  An updated ``scenes.json``
is written into the output directory referencing only the kept scenes.  A
detailed log records per-file metrics, per-split before/after statistics, and
an overall summary.

Usage::

    python3 core/prune_trajectories.py \
        --input output/trajectories_old \
        --output ./output/trajectories_pruned \
        --scenes configs/scenes.json \
        --total trainval:80 \
        --total test:30 \
        --total mini:4 \
        --smoothness 1.0 

    # Inspect without writing anything:
    python3 core/prune_trajectories.py \
        --input  ./output/trajectories_old \
        --output ./output/trajectories_pruned \
        --scenes configs/scenes.json \
        --smoothness 0.5 \
        --dry-run
"""

import argparse
import json
import math
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Make sure the project root is importable when run directly.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.utils.logging_utils import log_print, setup_logging

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Speed (m/s) below which a frame is treated as "zero velocity".
DEFAULT_ZERO_THRESHOLD: float = 0.1

#: Number of equal-width velocity bins.
DEFAULT_N_BINS: int = 10

#: RNG seed used for reproducible down-sampling within each bin.
DEFAULT_SEED: int = 42

#: Known split prefixes (order matters for display).
KNOWN_SPLITS: List[str] = ["trainval", "test", "mini"]


# ---------------------------------------------------------------------------
# File analysis
# ---------------------------------------------------------------------------

def derive_speeds(locations: np.ndarray, simulation_fps: float) -> np.ndarray:
    """Return per-frame ego speeds (m/s) derived from consecutive location diffs.

    Args:
        locations: Shape ``(N, 3)`` float32 array of ``[x, y, z]`` positions.
        simulation_fps: Simulation frames per second (used to convert position
            delta to speed).

    Returns:
        Shape ``(N-1,)`` float32 array of speeds in metres per second.
        Returns an empty array when ``N < 2``.
    """
    if locations.shape[0] < 2:
        return np.array([], dtype=np.float32)
    deltas = np.diff(locations, axis=0)           # (N-1, 3)
    distances = np.linalg.norm(deltas, axis=1)    # (N-1,)
    return (distances * simulation_fps).astype(np.float32)


def analyse_file(path: Path, zero_threshold: float) -> Optional[Dict[str, Any]]:
    """Open a trajectory HDF5 file and compute ego-speed statistics.

    Args:
        path: Path to the ``.h5`` file.
        zero_threshold: Speed (m/s) for the zero-velocity frame test.

    Returns:
        A dict with the following keys::

            scene_id        str
            map_name        str
            weather         str
            traffic_density float
            spawn_point     int
            n_frames        int
            mean_speed      float   (m/s, averaged over N-1 inter-frame speeds)
            median_speed    float
            p25_speed       float
            p75_speed       float
            max_speed       float
            zero_frac       float   (fraction of frames below zero_threshold)
            speeds          np.ndarray   (N-1,)  — kept in memory for binning

        Returns ``None`` if the file cannot be read.
    """
    try:
        with h5py.File(path, "r") as f:
            meta = f["metadata"]
            scene_id: str = str(meta.attrs["scene_id"])
            map_name: str = str(meta.attrs.get("map_name", "unknown"))
            weather: str = str(meta.attrs.get("weather", "unknown"))
            traffic_density: float = float(meta.attrs.get("traffic_density", 0.0))
            spawn_point: int = int(meta.attrs.get("spawn_point", 0))
            simulation_fps: float = float(meta.attrs.get("simulation_fps", 10.0))

            locations: np.ndarray = f["ego_vehicle/locations"][:]  # (N, 3)
    except Exception as exc:
        log_print(f"  [ERROR] Could not read {path.name}: {exc}", level="ERROR")
        return None

    speeds = derive_speeds(locations, simulation_fps)
    n_frames = locations.shape[0]

    if speeds.size == 0:
        mean_speed = median_speed = p25_speed = p75_speed = max_speed = 0.0
        zero_frac = 1.0
    else:
        mean_speed = float(np.mean(speeds))
        median_speed = float(np.median(speeds))
        p25_speed = float(np.percentile(speeds, 25))
        p75_speed = float(np.percentile(speeds, 75))
        max_speed = float(np.max(speeds))
        zero_frac = float(np.mean(speeds < zero_threshold))

    return {
        "scene_id": scene_id,
        "map_name": map_name,
        "weather": weather,
        "traffic_density": traffic_density,
        "spawn_point": spawn_point,
        "n_frames": n_frames,
        "mean_speed": mean_speed,
        "median_speed": median_speed,
        "p25_speed": p25_speed,
        "p75_speed": p75_speed,
        "max_speed": max_speed,
        "zero_frac": zero_frac,
        "speeds": speeds,
    }


# ---------------------------------------------------------------------------
# Binning and pruning logic
# ---------------------------------------------------------------------------

def resolve_target(target_arg: str, bin_counts: List[int]) -> int:
    """Resolve the ``--target`` argument to an integer per-bin target count.

    Args:
        target_arg: One of ``"min"``, ``"median"``, or a string representation
            of a positive integer.
        bin_counts: List of per-bin file counts (length = n_bins).

    Returns:
        The resolved integer target count (≥ 1).
    """
    nonzero = [c for c in bin_counts if c > 0]
    if not nonzero:
        return 1
    if target_arg == "min":
        return max(1, min(nonzero))
    if target_arg == "median":
        return max(1, int(np.median(nonzero)))
    # Try to parse as integer
    try:
        return max(1, int(target_arg))
    except ValueError:
        raise ValueError(
            f"--target must be 'min', 'median', or a positive integer; got '{target_arg}'"
        )


def bin_and_prune(
    stats_list: List[Dict[str, Any]],
    n_bins: int,
    target_arg: str,
    rng: np.random.Generator,
    split_name: str,
    smoothness: float = 1.0,
    total_target: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], np.ndarray, np.ndarray, float, float]:
    """Assign each file to a speed bin and randomly prune over-filled bins.

    The global speed range used for binning is ``[0, max_mean_speed]`` across
    all files in *stats_list*.  Bins are equal-width.

    Each bin's keep target is derived from its *excess* above the mean of
    occupied bins.  Bins at or below the mean are left untouched; bins
    above it have their excess reduced by ``smoothness``::

        excess_b        = max(0, count_b - mean_occupied)
        per_bin_target[b] = max(floor_target,
                                round(count_b - smoothness * excess_b))

    This concentrates pruning on the peaks: a bin twice the mean loses
    twice as many files (in absolute terms) as a bin only slightly above
    the mean.  ``floor_target`` (resolved from ``--target``) sets the
    absolute minimum that any bin may reach.

    Args:
        stats_list: List of dicts from :func:`analyse_file`, one per file.
        n_bins: Number of equal-width bins.
        target_arg: Resolves the floor target via :func:`resolve_target`.
        rng: NumPy random generator for reproducible sampling.
        split_name: Used only for log messages.
        smoothness: Float in ``[0.0, 1.0]``.  ``0.0`` keeps all files;
            ``1.0`` pulls every bin above the mean down to the mean
            (floored at ``floor_target``).
        total_target: When provided, overrides the data-derived reference
            count with ``total_target / n_occupied_bins``.  This lets you
            say "keep at most N files from this split" rather than tuning
            per-bin numbers.  Files in bins already below the per-bin
            budget are always kept, so the actual output may exceed
            ``total_target`` when the distribution is sparse.

    Returns:
        A tuple of:
        - ``kept``   — dicts for files that survive pruning (status added)
        - ``pruned`` — dicts for files that are removed (status + reason added)
        - ``bin_counts_before`` — per-bin file counts before pruning (length n_bins)
        - ``bin_counts_after``  — per-bin file counts after pruning
        - ``global_min``  — lower edge of the speed range (always 0.0)
        - ``global_max``  — upper edge of the speed range
    """
    if not stats_list:
        empty = np.zeros(n_bins, dtype=int)
        return [], [], empty, empty.copy(), 0.0, 0.0

    mean_speeds = np.array([s["mean_speed"] for s in stats_list], dtype=float)
    global_min = 0.0
    global_max = float(mean_speeds.max()) if mean_speeds.max() > 0 else 1.0

    # Assign bin indices — clamp last file to bin n_bins-1 so max value falls in.
    bin_width = (global_max - global_min) / n_bins
    bin_indices = np.floor((mean_speeds - global_min) / bin_width).astype(int)
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Group files by bin
    bins: Dict[int, List[int]] = defaultdict(list)
    for idx, b in enumerate(bin_indices):
        bins[b].append(idx)

    bin_counts_before = np.array([len(bins.get(b, [])) for b in range(n_bins)], dtype=int)
    floor_target = resolve_target(target_arg, list(bin_counts_before))

    # Reference = mean count of occupied (non-empty) bins, or a
    # user-supplied per-bin budget derived from --total.
    occupied_counts = bin_counts_before[bin_counts_before > 0]
    n_occupied = len(occupied_counts)
    if total_target is not None:
        # User set a total scene count; distribute evenly across occupied bins.
        reference_count = total_target / n_occupied if n_occupied > 0 else 1.0
        if total_target > len(stats_list):
            log_print(
                f"  [{split_name}] WARNING: --total {total_target} exceeds "
                f"available files ({len(stats_list)}); no pruning will occur.",
                level="WARNING",
            )
    else:
        reference_count = float(np.mean(occupied_counts)) if n_occupied > 0 else 1.0

    # Per-bin targets: reduce only the excess above the reference.
    # smoothness=0 → no change; smoothness=1 → peaks pulled to reference.
    per_bin_targets = np.empty(n_bins, dtype=int)
    for _b in range(n_bins):
        _count = int(bin_counts_before[_b])
        _excess = max(0.0, _count - reference_count)
        per_bin_targets[_b] = max(floor_target, round(_count - smoothness * _excess))

    # When --total is set, rounding in per-bin targets may leave a deficit.
    # Top up by incrementing targets on the largest bins (greedy) until the
    # sum of capped targets reaches min(total_target, n_files).
    if total_target is not None:
        _budget = min(total_target, len(stats_list))
        _current = sum(
            min(int(per_bin_targets[_b]), int(bin_counts_before[_b]))
            for _b in range(n_bins)
        )
        # Distribute the deficit evenly: top up the smallest-target bins
        # first (round-robin, +1 at a time) so we restore balance rather
        # than dumping all remaining capacity back into the peak bin.
        _rounds = 0
        while _current < _budget and _rounds < n_bins * _budget:
            _rounds += 1
            # One pass: give +1 to each bin sorted by current target ASC
            _made_progress = False
            for _b in sorted(range(n_bins), key=lambda _b: per_bin_targets[_b]):
                if _current >= _budget:
                    break
                _cap = int(bin_counts_before[_b])
                if int(per_bin_targets[_b]) < _cap:
                    per_bin_targets[_b] += 1
                    _current += 1
                    _made_progress = True
            if not _made_progress:
                break  # all bins are at capacity

    # Cap each bin's target at its actual file count (empty bins cannot
    # contribute files even if they have a non-zero floor target).
    expected_total = int(sum(
        min(int(per_bin_targets[_b]), int(bin_counts_before[_b]))
        for _b in range(n_bins)
    ))
    ref_source = f"total={total_target}→{reference_count:.1f}/bin" if total_target is not None else f"ref_count={reference_count:.1f}"
    log_print(
        f"  [{split_name}] speed range=[{global_min:.2f}, {global_max:.2f}] m/s  "
        f"n_bins={n_bins}  floor_target={floor_target}  "
        f"{ref_source}  smoothness={smoothness:.2f}  expected_kept≈{expected_total}"
    )

    kept_indices: set = set()
    pruned_reasons: Dict[int, str] = {}

    for b in range(n_bins):
        file_indices = bins.get(b, [])
        lo = global_min + b * bin_width
        hi = global_min + (b + 1) * bin_width
        label = f"bin {b} [{lo:.3f}–{hi:.3f} m/s]"
        target = int(per_bin_targets[b])

        if len(file_indices) <= target:
            for fi in file_indices:
                kept_indices.add(fi)
        else:
            # Randomly keep `target` files, prune the rest
            chosen = rng.choice(file_indices, size=target, replace=False).tolist()
            for fi in file_indices:
                if fi in chosen:
                    kept_indices.add(fi)
                else:
                    pruned_reasons[fi] = f"over-represented in {label} (target={target})"

    kept = []
    pruned = []
    for idx, stats in enumerate(stats_list):
        stats = dict(stats)  # shallow copy to avoid mutating caller's data
        stats["bin_idx"] = int(bin_indices[idx])
        stats["bin_lo"] = global_min + bin_indices[idx] * bin_width
        stats["bin_hi"] = global_min + (bin_indices[idx] + 1) * bin_width
        if idx in kept_indices:
            stats["status"] = "kept"
            kept.append(stats)
        else:
            stats["status"] = "pruned"
            stats["reason"] = pruned_reasons.get(idx, "unknown")
            pruned.append(stats)

    bin_counts_after = np.zeros(n_bins, dtype=int)
    for s in kept:
        bin_counts_after[s["bin_idx"]] += 1

    return kept, pruned, bin_counts_before, bin_counts_after, global_min, global_max


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _speed_bar(
    bin_counts: np.ndarray,
    global_min: float = 0.0,
    bin_width: float = 1.0,
    max_bar: int = 30,
) -> List[str]:
    """Return ASCII bar-chart lines for a bin-count array with speed ranges."""
    peak = int(bin_counts.max()) if bin_counts.max() > 0 else 1
    lines = []
    for i, count in enumerate(bin_counts):
        lo = global_min + i * bin_width
        hi = global_min + (i + 1) * bin_width
        bar_len = int(round(count / peak * max_bar))
        bar = "#" * bar_len
        lines.append(
            f"    bin {i:2d} [{lo:5.2f}–{hi:5.2f} m/s]: {bar:<{max_bar}} {count:4d}"
        )
    return lines


def log_per_file_detail(
    all_stats: List[Dict[str, Any]],
    split_name: str,
    bin_width: float,
) -> None:
    """Write one detail line per file into the logger."""
    log_print(f"\n{'─' * 72}")
    log_print(f"  PER-FILE DETAIL  [{split_name.upper()}]")
    log_print(f"{'─' * 72}")

    all_stats_sorted = sorted(all_stats, key=lambda s: s["scene_id"])
    for s in all_stats_sorted:
        status = s["status"].upper()
        tag = f"[{'KEPT  ' if status == 'KEPT' else 'PRUNED'}]"
        core = (
            f"{s['scene_id']:<14}  "
            f"mean={s['mean_speed']:5.2f} m/s  "
            f"median={s['median_speed']:5.2f} m/s  "
            f"p25={s['p25_speed']:5.2f}  "
            f"p75={s['p75_speed']:5.2f}  "
            f"max={s['max_speed']:5.2f}  "
            f"zero={s['zero_frac'] * 100:5.1f}%  "
            f"bin={s['bin_idx']:2d} [{s['bin_lo']:5.2f}–{s['bin_hi']:5.2f} m/s]  "
            f"frames={s['n_frames']}"
        )
        if status == "PRUNED":
            reason = s.get("reason", "")
            log_print(f"  {tag} {core}  reason={reason}")
        else:
            log_print(f"  {tag} {core}")


def log_split_summary(
    split_name: str,
    all_stats: List[Dict[str, Any]],
    kept: List[Dict[str, Any]],
    pruned: List[Dict[str, Any]],
    bin_counts_before: np.ndarray,
    bin_counts_after: np.ndarray,
    global_min: float,
    global_max: float,
    n_bins: int,
) -> None:
    """Write the before/after summary block for one split."""
    log_print(f"\n{'=' * 72}")
    log_print(f"  SPLIT SUMMARY: {split_name.upper()}")
    log_print(f"{'=' * 72}")

    def _agg(lst: List[Dict[str, Any]], key: str) -> str:
        vals = [s[key] for s in lst]
        return f"{np.mean(vals):.3f}" if vals else "N/A"

    n_before = len(all_stats)
    n_after = len(kept)
    n_removed = len(pruned)
    pct = (n_removed / n_before * 100) if n_before else 0.0

    log_print(f"  Before : {n_before:4d} files  "
              f"mean_speed={_agg(all_stats, 'mean_speed'):>7} m/s  "
              f"zero_frac={float(np.mean([s['zero_frac'] for s in all_stats])) * 100:.1f}%"
              if all_stats else f"  Before : {n_before:4d} files")
    log_print(f"  After  : {n_after:4d} files  "
              f"mean_speed={_agg(kept, 'mean_speed'):>7} m/s  "
              f"zero_frac={float(np.mean([s['zero_frac'] for s in kept])) * 100:.1f}%"
              if kept else f"  After  : {n_after:4d} files")
    log_print(f"  Removed: {n_removed:4d} files ({pct:.1f}%)")

    log_print(f"\n  Speed range : [{global_min:.3f}, {global_max:.3f}] m/s  "
              f"(bin width = {(global_max - global_min) / n_bins if n_bins else 0:.3f} m/s)")
    bw = (global_max - global_min) / n_bins if n_bins and global_max > global_min else 1.0
    log_print("\n  Bin distribution BEFORE:")
    for line in _speed_bar(bin_counts_before, global_min, bw):
        log_print(line)
    log_print("\n  Bin distribution AFTER:")
    for line in _speed_bar(bin_counts_after, global_min, bw):
        log_print(line)

    if pruned:
        log_print(f"\n  Pruned scene IDs ({len(pruned)}):")
        ids = sorted(s["scene_id"] for s in pruned)
        # Format in rows of 8
        for i in range(0, len(ids), 8):
            log_print("    " + "  ".join(ids[i: i + 8]))


def log_overall_summary(
    split_results: Dict[str, Dict[str, Any]],
    elapsed: float,
    dry_run: bool,
    output_dir: str,
) -> None:
    """Write the final aggregated summary across all splits."""
    log_print(f"\n{'#' * 72}")
    log_print("  OVERALL SUMMARY")
    log_print(f"{'#' * 72}")

    total_before = sum(r["n_before"] for r in split_results.values())
    total_after = sum(r["n_after"] for r in split_results.values())
    total_removed = sum(r["n_removed"] for r in split_results.values())
    pct = (total_removed / total_before * 100) if total_before else 0.0

    log_print(f"  Total before : {total_before}")
    log_print(f"  Total after  : {total_after}")
    log_print(f"  Total removed: {total_removed} ({pct:.1f}%)")
    log_print("")

    for split_name, r in split_results.items():
        log_print(
            f"  {split_name:<10}  "
            f"before={r['n_before']:4d}  "
            f"after={r['n_after']:4d}  "
            f"removed={r['n_removed']:4d}"
        )

    log_print("")
    log_print(f"  Output directory : {output_dir}")
    log_print(f"  Dry-run mode     : {dry_run}")
    log_print(f"  Elapsed          : {elapsed:.1f}s ({elapsed / 60:.2f} min)")


# ---------------------------------------------------------------------------
# scenes.json helpers
# ---------------------------------------------------------------------------

def load_scenes_json(path: Path) -> Dict[str, Any]:
    """Load and return the scenes.json config.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed dict with ``"statistics"`` and ``"scenes"`` keys.
    """
    with open(path, "r") as f:
        return json.load(f)


def filter_scenes_json(
    original: Dict[str, Any],
    kept_ids_by_split: Dict[str, set],
) -> Dict[str, Any]:
    """Return a copy of *original* filtered to only the kept scene IDs.

    The ``statistics`` block is recomputed from the surviving scenes.

    The trajectory directory ``trainval`` combines the ``train`` and ``val``
    JSON splits; both are filtered against the same ``kept_ids`` set so the
    train/val distinction is preserved in the output.

    Args:
        original: Full scenes config as loaded from JSON.
        kept_ids_by_split: Mapping from split name to a set of kept scene IDs.
            May use directory names (``trainval``) or JSON split names
            (``train``, ``val``).

    Returns:
        Filtered config dict, ready for JSON serialisation.
    """
    from collections import Counter

    # Build a lookup that maps each JSON split name to the set of kept IDs.
    # The trajectory directory "trainval" covers both "train" and "val" JSON
    # splits, so alias them to the same kept-IDs set.
    _SPLIT_ALIASES: Dict[str, str] = {"train": "trainval", "val": "trainval"}

    def _kept_ids_for(json_split: str) -> set:
        if json_split in kept_ids_by_split:
            return kept_ids_by_split[json_split]
        alias = _SPLIT_ALIASES.get(json_split)
        if alias and alias in kept_ids_by_split:
            return kept_ids_by_split[alias]
        return set()

    new_scenes: Dict[str, List] = {}
    new_stats: Dict[str, Any] = {}

    for split, scene_list in original.get("scenes", {}).items():
        kept_ids = _kept_ids_for(split)
        filtered = [s for s in scene_list if s["id"] in kept_ids]
        new_scenes[split] = filtered

        # Recompute statistics
        maps_c: Counter = Counter()
        weather_c: Counter = Counter()
        density_c: Counter = Counter()
        for s in filtered:
            maps_c[s["map"]] += 1
            weather_c[s["weather"]] += 1
            density_c[str(s["traffic_density"])] += 1

        new_stats[split] = {
            "num_scenes": len(filtered),
            "maps": dict(maps_c),
            "weather": dict(weather_c),
            "traffic_density": dict(density_c),
        }

    return {"statistics": new_stats, "scenes": new_scenes}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pruning(
    input_dir: Path,
    output_dir: Path,
    scenes_path: Optional[Path],
    n_bins: int,
    zero_threshold: float,
    target_arg: str,
    seed: int,
    only_split: Optional[str],
    dry_run: bool,
    smoothness: float = 1.0,
    total_by_split: Optional[Dict[str, int]] = None,
) -> None:
    """End-to-end pruning pipeline.

    Args:
        input_dir: Root of the source trajectory directory tree.
        output_dir: Destination root (will be created).
        scenes_path: Path to the source ``scenes.json`` (may be ``None``).
        n_bins: Number of equal-width velocity bins.
        zero_threshold: Speed threshold for zero-velocity frame counting.
        target_arg: Per-bin target strategy string.
        seed: RNG seed for reproducible random down-sampling.
        only_split: If set, only process this split; skip all others.
        dry_run: When ``True``, analyse and log but do not write any files.
        smoothness: ``0.0`` = keep all files; ``1.0`` = maximally flat
            distribution.  Interpolates per-bin targets in between.
        total_by_split: Optional mapping of split name to desired total scene
            count.  For each split found in the key, overrides the data-derived
            reference count with ``total / n_occupied_bins``.  Build this dict
            from plain integers (applies to all splits) or ``split:N`` pairs.
            Splits not present in the mapping use the data-derived mean.
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Discover available splits
    if not input_dir.exists():
        log_print(f"Input directory not found: {input_dir}", level="ERROR")
        return

    split_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    if not split_dirs:
        log_print(f"No sub-directories found in {input_dir}", level="WARNING")
        return

    all_kept_ids: Dict[str, set] = {}
    split_results: Dict[str, Dict[str, Any]] = {}

    for split_dir in split_dirs:
        split_name = split_dir.name

        if only_split and split_name != only_split:
            log_print(f"Skipping split '{split_name}' (--split filter active).")
            continue

        h5_files = sorted(split_dir.glob("*.h5"))
        if not h5_files:
            log_print(f"  No .h5 files found in {split_dir}, skipping.", level="WARNING")
            continue

        log_print(f"\n{'─' * 72}")
        log_print(f"  Analysing split: {split_name}  ({len(h5_files)} files)")
        log_print(f"{'─' * 72}")

        # Analyse every file
        stats_list: List[Dict[str, Any]] = []
        for path in h5_files:
            result = analyse_file(path, zero_threshold)
            if result is not None:
                result["path"] = path
                stats_list.append(result)

        if not stats_list:
            log_print(f"  No readable .h5 files in {split_dir}.", level="WARNING")
            continue

        # Bin & prune
        split_total = (total_by_split or {}).get(split_name)
        kept, pruned, bin_counts_before, bin_counts_after, g_min, g_max = bin_and_prune(
            stats_list, n_bins, target_arg, rng, split_name,
            smoothness=smoothness, total_target=split_total,
        )

        # Per-file detail log
        all_combined = kept + pruned
        bin_width = (g_max - g_min) / n_bins if n_bins and g_max > g_min else 1.0
        log_per_file_detail(all_combined, split_name, bin_width)

        # Split summary log
        log_split_summary(
            split_name, all_combined, kept, pruned,
            bin_counts_before, bin_counts_after, g_min, g_max, n_bins,
        )

        # Copy kept files
        kept_ids: set = set()
        if not dry_run:
            split_out = output_dir / split_name
            split_out.mkdir(parents=True, exist_ok=True)
            for s in kept:
                src: Path = s["path"]
                dst = split_out / src.name
                shutil.copy2(src, dst)
                kept_ids.add(s["scene_id"])
            log_print(f"\n  Copied {len(kept)} files → {split_out}")
        else:
            for s in kept:
                kept_ids.add(s["scene_id"])

        all_kept_ids[split_name] = kept_ids
        split_results[split_name] = {
            "n_before": len(all_combined),
            "n_after": len(kept),
            "n_removed": len(pruned),
        }

    if not split_results:
        log_print("No splits were processed.", level="WARNING")
        return

    # Write updated scenes.json
    if scenes_path is not None and scenes_path.exists():
        original_scenes = load_scenes_json(scenes_path)
        filtered_scenes = filter_scenes_json(original_scenes, all_kept_ids)
        if not dry_run:
            out_scenes_path = output_dir / "scenes.json"
            with open(out_scenes_path, "w") as f:
                json.dump(filtered_scenes, f, indent=2)
            log_print(f"\n  Updated scenes.json written → {out_scenes_path}")
        else:
            # Log what would be written
            total_kept = sum(
                len(v) for v in filtered_scenes["scenes"].values()
            )
            log_print(
                f"\n  [DRY-RUN] Would write scenes.json with {total_kept} scenes."
            )
    elif scenes_path is not None:
        log_print(
            f"  scenes.json not found at {scenes_path} — skipping.", level="WARNING"
        )

    elapsed = time.time() - t0
    log_overall_summary(split_results, elapsed, dry_run, str(output_dir))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments and run the trajectory pruning pipeline."""
    parser = argparse.ArgumentParser(
        description="Prune trajectory HDF5 files to flatten the ego-speed distribution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True,
        help="Root directory containing trajectory split sub-directories "
             "(e.g. ./output/trajectories).",
    )
    parser.add_argument(
        "--output", default="./output/trajectories_pruned",
        help="Destination directory for pruned trajectory files.",
    )
    parser.add_argument(
        "--scenes", default=None,
        help="Path to scenes.json; a filtered copy is written to --output.",
    )
    parser.add_argument(
        "--n-bins", type=int, default=DEFAULT_N_BINS,
        help="Number of equal-width speed bins.",
    )
    parser.add_argument(
        "--zero-threshold", type=float, default=DEFAULT_ZERO_THRESHOLD,
        help="Speed (m/s) below which a frame is counted as zero-velocity.",
    )
    parser.add_argument(
        "--target", default="min",
        help="Per-bin target count after pruning: 'min' (smallest occupied bin), "
             "'median', or a positive integer.",
    )
    parser.add_argument(
        "--total", metavar="[SPLIT:]N", action="append", default=None,
        help="Target total scene count, either for all splits (e.g. --total 80) "
             "or per split (e.g. --total trainval:80 --total test:30).  "
             "May be repeated to set different values for each split.  "
             "Splits without an explicit value fall back to a global integer "
             "if one was given, otherwise use the data-derived mean.  "
             "Mutually compatible with --target (per-bin floor) and --smoothness.",
    )
    parser.add_argument(
        "--split", default=None, choices=["trainval", "test", "mini"],
        help="Process only this split (default: all splits found in --input).",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="Random seed for reproducible per-bin down-sampling.",
    )
    parser.add_argument(
        "--smoothness", type=float, default=1.0,
        help="Distribution smoothing strength in [0.0, 1.0].  "
             "0.0 keeps all files unchanged; 1.0 enforces the flattest "
             "possible distribution (current bin counts → flat target).  "
             "Values in between partially level the distribution.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Analyse and log without writing any files.",
    )
    args = parser.parse_args()

    if not (0.0 <= args.smoothness <= 1.0):
        print(f"Error: --smoothness must be in [0.0, 1.0], got {args.smoothness}")
        sys.exit(1)

    # Parse --total values into a per-split dict.
    # Accepts plain integers (applied to all splits) and "split:N" pairs.
    _KNOWN_SPLITS = ["trainval", "test", "mini"]
    _global_total: Optional[int] = None
    total_by_split: Dict[str, int] = {}
    for _item in (args.total or []):
        _s = str(_item)
        if ":" in _s:
            _split, _n = _s.split(":", 1)
            total_by_split[_split.strip()] = int(_n.strip())
        else:
            _global_total = int(_s)
    if _global_total is not None:
        for _sp in _KNOWN_SPLITS:
            total_by_split.setdefault(_sp, _global_total)

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    # Set up logging — always into the output directory even for dry-runs so
    # the log can be inspected afterwards.
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(str(output_dir), debug=False)

    log_print("=" * 72)
    log_print("  TRAJECTORY VELOCITY PRUNING")
    log_print("=" * 72)
    log_print(f"  Input            : {input_dir}")
    log_print(f"  Output           : {output_dir}")
    log_print(f"  Scenes JSON      : {args.scenes or '(none)'}")
    log_print(f"  Bins             : {args.n_bins}")
    log_print(f"  Zero threshold   : {args.zero_threshold} m/s")
    log_print(f"  Per-bin target   : {args.target}")
    _total_display = ", ".join(f"{k}:{v}" for k, v in sorted(total_by_split.items())) if total_by_split else "(auto)"
    log_print(f"  Total per split  : {_total_display}")
    log_print(f"  Smoothness       : {args.smoothness}")
    log_print(f"  Split filter     : {args.split or '(all)'}")
    log_print(f"  Seed             : {args.seed}")
    log_print(f"  Dry-run          : {args.dry_run}")

    run_pruning(
        input_dir=input_dir,
        output_dir=output_dir,
        scenes_path=Path(args.scenes) if args.scenes else None,
        n_bins=args.n_bins,
        zero_threshold=args.zero_threshold,
        target_arg=args.target,
        seed=args.seed,
        only_split=args.split,
        dry_run=args.dry_run,
        smoothness=args.smoothness,
        total_by_split=total_by_split or None,
    )


if __name__ == "__main__":
    main()
