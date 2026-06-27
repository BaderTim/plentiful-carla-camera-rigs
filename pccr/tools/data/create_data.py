#!/usr/bin/env python3
"""
Convert a captured nuScenes-format dataset to BEVDet-compatible PKL format.

Iterates over all rig directories inside ``--save-folder``, generates
``*_infos_*.pkl`` files with sweep information, builds the ground-truth
database (unless ``--no-lidar``), and appends BEVDet-style adjacent
annotation info.

Usage::

    python3 tools/data/create_data.py \
        --save-folder ./output/data \
        --version v1.0-mini
"""

import argparse
import sys
from os import path as osp
from pathlib import Path

# Ensure project root is importable when the script is run directly.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib.data_converter import nuscenes_converter
from lib.data_converter.bevdet_annotations import add_ann_adj_info
from lib.data_converter.create_gt_database import create_groundtruth_database


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def nuscenes_data_prep(
    data_path: str,
    info_prefix: str,
    version: str,
    dataset_name: str,
    out_dir: str,
    max_sweeps: int = 10,
    no_lidar: bool = False,
    root_path = None,
) -> None:
    """Prepare nuScenes-format PKL info files from a captured dataset.

    Creates ``*_infos_*.pkl`` and (optionally) ``*_dbinfos_*.pkl``.
    Also appends BEVDet-style adjacent annotation info (velocities, ego-
    coordinate boxes, scene tokens).

    Args:
        data_path: Physical path to the dataset root (rig directory).
        info_prefix: Short name used as prefix in output file names.
        version: nuScenes version string (e.g. ``"v1.0-mini"``).
        dataset_name: Dataset class name (``"NuScenesDataset"``).
        out_dir: Directory to write output PKL files into.
        max_sweeps: Number of consecutive LiDAR sweeps per sample.
        no_lidar: Skip all LiDAR-related processing when ``True``.
        root_path: Optional path prefix stored verbatim in PKL file paths.
    """
    nuscenes_converter.create_nuscenes_infos(
        data_path,
        info_prefix,
        version=version,
        max_sweeps=max_sweeps,
        out_dir=out_dir,
        no_lidar=no_lidar,
        root_path_prefix=root_path,
    )

    if version == "v1.0-mini":
        info_path = osp.join(out_dir, f"{info_prefix}_infos_mini.pkl")
        db_info_save_path = osp.join(out_dir, f"{info_prefix}_dbinfos_mini.pkl")
    elif version == "v1.0-test":
        info_path = osp.join(out_dir, f"{info_prefix}_infos_test.pkl")
        db_info_save_path = osp.join(out_dir, f"{info_prefix}_dbinfos_test.pkl")
    else:
        info_path = osp.join(out_dir, f"{info_prefix}_infos_train.pkl")
        db_info_save_path = osp.join(out_dir, f"{info_prefix}_dbinfos_train.pkl")

    if not no_lidar:
        create_groundtruth_database(
            dataset_name,
            data_path,
            info_prefix,
            info_path=info_path,
            db_info_save_path=db_info_save_path,
        )

    print("Adding annotation adjacent info...")
    add_ann_adj_info(data_path, info_prefix, version, out_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments and run data preparation for all rig directories."""
    parser = argparse.ArgumentParser(description="Data converter arg parser")
    parser.add_argument(
        "--save-folder",
        type=str,
        default="./output/data",
        help="Physical path of the dataset root folder",
    )
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="Path prefix to be stored verbatim in PKL file paths",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1.0-mini",
        help="Dataset version (e.g. v1.0-mini, v1.0-trainval, v1.0-test)",
    )
    parser.add_argument(
        "--max-sweeps",
        type=int,
        default=10,
        help="Number of LiDAR sweeps per sample",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Override output directory (defaults to each rig directory)",
    )
    parser.add_argument(
        "--no-lidar",
        action="store_true",
        help="Skip LiDAR-related processing",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker threads",
    )
    args = parser.parse_args()

    save_folder = Path(args.save_folder)

    def _has_version(d: Path, v: str) -> bool:
        if (d / v).exists():
            return True
        if v == "v1.0":
            return (d / "v1.0-trainval").exists() or (d / "v1.0-mini").exists()
        return False

    rig_dirs = [
        d for d in save_folder.iterdir()
        if d.is_dir() and _has_version(d, args.version)
    ]
    if not rig_dirs:
        if _has_version(save_folder, args.version):
            rig_dirs = [save_folder]
        else:
            print(
                f"No nuScenes datasets found in {args.save_folder} "
                f"with version {args.version}"
            )
            raise SystemExit(1)

    for rig_dir in rig_dirs:
        print(f"\n--- Processing rig: {rig_dir.name} ---")
        rig_root = str(rig_dir)
        info_prefix = rig_dir.name
        out_dir = args.out_dir if args.out_dir else rig_root

        if args.version == "v1.0":
            for sub_v in ("trainval", "test"):
                v_name = f"v1.0-{sub_v}"
                if not (rig_dir / v_name).exists():
                    continue
                print(f"  Sub-version: {v_name}")
                nuscenes_data_prep(
                    data_path=rig_root,
                    info_prefix=info_prefix,
                    version=v_name,
                    dataset_name="NuScenesDataset",
                    out_dir=out_dir,
                    max_sweeps=args.max_sweeps,
                    no_lidar=args.no_lidar,
                    root_path=args.root_path,
                )
        else:
            nuscenes_data_prep(
                data_path=rig_root,
                info_prefix=info_prefix,
                version=args.version,
                dataset_name="NuScenesDataset",
                out_dir=out_dir,
                max_sweeps=args.max_sweeps,
                no_lidar=args.no_lidar,
                root_path=args.root_path,
            )


if __name__ == "__main__":
    main()
