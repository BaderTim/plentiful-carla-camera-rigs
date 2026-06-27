#!/usr/bin/env python3
"""
CAN bus data verification and visualization tool.

Loads CAN bus data for a given scene and displays statistics and plots.

Usage::

    python3 tools/viz/visualize_canbus.py output/data/R1 --list
    python3 tools/viz/visualize_canbus.py output/data/R1 --scene mini_01
    python3 tools/viz/visualize_canbus.py output/data/R1 --scene mini_01 --plot
    python3 tools/viz/visualize_canbus.py output/data/R1 --scene mini_01 --route
    python3 tools/viz/visualize_canbus.py output/data/R1 --scene mini_01 --plot --save plots/
    python3 tools/viz/visualize_canbus.py output/data/R1 --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure project root is importable when run directly.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.gridspec import GridSpec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not found. Plotting disabled.")


class CANBusVisualizer:
    """Verify and visualize CAN bus data from a nuScenes-format dataset.

    Attributes:
        dataset_dir: Root dataset path.
        can_bus_dir: ``<dataset_dir>/can_bus`` subdirectory.
        message_types: Message type subkeys searched for each scene.
    """

    message_types = ["pose", "ms_imu", "steeranglefeedback", "vehicle_monitor"]

    def __init__(self, dataset_dir: str) -> None:
        """Initialise the visualizer, verifying that the can_bus folder exists.

        Args:
            dataset_dir: Path to the dataset root directory.

        Raises:
            FileNotFoundError: If ``<dataset_dir>/can_bus`` does not exist.
        """
        self.dataset_dir = Path(dataset_dir)
        self.can_bus_dir = self.dataset_dir / "can_bus"
        if not self.can_bus_dir.exists():
            raise FileNotFoundError(
                f"CAN bus directory not found: {self.can_bus_dir}"
            )

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def list_scenes(self) -> List[str]:
        """Return sorted names of all scenes that have a ``*_meta.json`` file."""
        scenes = {
            f.stem.replace("_meta", "")
            for f in self.can_bus_dir.glob("*_meta.json")
        }
        return sorted(scenes)

    def load_scene_data(self, scene_name: str) -> Dict[str, Any]:
        """Load all CAN bus JSON files for *scene_name*.

        Args:
            scene_name: Scene identifier (e.g. ``"mini_01"``).

        Returns:
            Dict keyed by message type (and ``"meta"``), each value being the
            parsed JSON content (list of records or dict for meta).
        """
        data: Dict[str, Any] = {}
        for mt in self.message_types:
            fp = self.can_bus_dir / f"{scene_name}_{mt}.json"
            data[mt] = json.loads(fp.read_text()) if fp.exists() else []
        meta_fp = self.can_bus_dir / f"{scene_name}_meta.json"
        data["meta"] = json.loads(meta_fp.read_text()) if meta_fp.exists() else {}
        return data

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify_data(self, scene_name: str) -> Dict[str, Any]:
        """Verify CAN bus data integrity and compute per-type statistics.

        Args:
            scene_name: Scene identifier.

        Returns:
            Dict with keys ``scene``, ``valid``, ``errors``, ``warnings``,
            and ``stats``.
        """
        data = self.load_scene_data(scene_name)
        results: Dict[str, Any] = {
            "scene": scene_name,
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {},
        }

        for mt in self.message_types:
            entries = data.get(mt, [])
            msg_stats: Dict[str, Any] = {
                "count": len(entries),
                "has_data": bool(entries),
            }
            if not entries:
                results["warnings"].append(f"{mt}: No data")
                results["stats"][mt] = msg_stats
                continue

            utimes = [e.get("utime", 0) for e in entries]
            if utimes != sorted(utimes):
                results["errors"].append(
                    f"{mt}: Timestamps not monotonically increasing"
                )
                results["valid"] = False

            if len(utimes) > 1:
                dts = np.diff(utimes) / 1e6
                msg_stats.update({
                    "duration_s": (max(utimes) - min(utimes)) / 1e6,
                    "avg_dt_ms": float(np.mean(dts) * 1000),
                    "min_dt_ms": float(np.min(dts) * 1000),
                    "max_dt_ms": float(np.max(dts) * 1000),
                    "std_dt_ms": float(np.std(dts) * 1000),
                    "freq_hz": float(1.0 / np.mean(dts)) if np.mean(dts) > 0 else 0.0,
                })
                if np.max(dts) > 1.0:  # > 1 s gap
                    results["warnings"].append(
                        f"{mt}: Large time gap ({np.max(dts)*1000:.1f} ms)"
                    )

            validator = {
                "pose": self._validate_pose,
                "ms_imu": self._validate_imu,
                "steeranglefeedback": self._validate_steering,
                "vehicle_monitor": self._validate_vehicle_monitor,
            }.get(mt)
            if validator:
                msg_stats.update(validator(entries))

            results["stats"][mt] = msg_stats

        # Cross-check against meta counts.
        meta = data.get("meta", {})
        for mt in self.message_types:
            if mt in meta:
                expected = meta[mt].get("count", 0)
                actual = results["stats"].get(mt, {}).get("count", 0)
                if expected != actual:
                    results["errors"].append(
                        f"{mt}: count mismatch (expected {expected}, got {actual})"
                    )
                    results["valid"] = False

        return results

    # ------------------------------------------------------------------
    # Per-type validators
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_pose(entries: List[Dict]) -> Dict[str, Any]:
        """Return position / velocity / acceleration statistics for pose data."""
        positions = np.array([e["pos"] for e in entries])
        velocities = np.array([e["vel"] for e in entries])
        accelerations = np.array([e["accel"] for e in entries])
        speeds = np.linalg.norm(velocities, axis=1)
        accels = np.linalg.norm(accelerations, axis=1)
        return {
            "pos_min": positions.min(axis=0).tolist(),
            "pos_max": positions.max(axis=0).tolist(),
            "total_distance_m": float(
                np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
            ),
            "speed_min_ms": float(speeds.min()),
            "speed_max_ms": float(speeds.max()),
            "speed_avg_ms": float(speeds.mean()),
            "accel_max_ms2": float(accels.max()),
            "accel_avg_ms2": float(accels.mean()),
        }

    @staticmethod
    def _validate_imu(entries: List[Dict]) -> Dict[str, Any]:
        """Return linear acceleration and rotation-rate statistics."""
        lin = np.array([e["linear_accel"] for e in entries])
        rot = np.array([e["rotation_rate"] for e in entries])
        return {
            "linear_accel_max": float(np.linalg.norm(lin, axis=1).max()),
            "linear_accel_avg": float(np.linalg.norm(lin, axis=1).mean()),
            "rotation_rate_max_rads": float(np.linalg.norm(rot, axis=1).max()),
            "rotation_rate_avg_rads": float(np.linalg.norm(rot, axis=1).mean()),
        }

    @staticmethod
    def _validate_steering(entries: List[Dict]) -> Dict[str, Any]:
        """Return steering angle statistics."""
        vals = np.array([e["value"] for e in entries])
        return {
            "steer_min_deg": float(vals.min()),
            "steer_max_deg": float(vals.max()),
            "steer_avg_deg": float(vals.mean()),
            "steer_std_deg": float(vals.std()),
        }

    @staticmethod
    def _validate_vehicle_monitor(entries: List[Dict]) -> Dict[str, Any]:
        """Return vehicle speed and yaw rate statistics."""
        speeds = np.array([e["vehicle_speed"] for e in entries])
        yaw = np.array([e["yaw_rate"] for e in entries])
        return {
            "vehicle_speed_min_ms": float(speeds.min()),
            "vehicle_speed_max_ms": float(speeds.max()),
            "vehicle_speed_avg_ms": float(speeds.mean()),
            "yaw_rate_min_rads": float(yaw.min()),
            "yaw_rate_max_rads": float(yaw.max()),
            "yaw_rate_avg_rads": float(np.abs(yaw).mean()),
        }

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_report(self, results: Dict[str, Any]) -> None:
        """Print a human-readable verification report to stdout.

        Args:
            results: Dict returned by :meth:`verify_data`.
        """
        print(f"\n{'='*60}")
        print(f"CAN Bus Report: {results['scene']}")
        print(f"{'='*60}")
        print(f"Status: {'✓ VALID' if results['valid'] else '✗ INVALID'}")

        if results["errors"]:
            print(f"\n{'Errors:':-^40}")
            for e in results["errors"]:
                print(f"  ✗ {e}")
        if results["warnings"]:
            print(f"\n{'Warnings:':-^40}")
            for w in results["warnings"]:
                print(f"  ⚠ {w}")

        print(f"\n{'Statistics:':-^40}")
        for mt, stats in results["stats"].items():
            print(f"\n  {mt}:")
            print(f"    Count: {stats.get('count', 0)}")
            if stats.get("count", 0) > 1:
                print(
                    f"    Duration: {stats.get('duration_s', 0):.2f}s  "
                    f"Freq: {stats.get('freq_hz', 0):.2f} Hz  "
                    f"Δt(avg/min/max): "
                    f"{stats.get('avg_dt_ms', 0):.1f}/"
                    f"{stats.get('min_dt_ms', 0):.1f}/"
                    f"{stats.get('max_dt_ms', 0):.1f} ms"
                )
            if mt == "pose" and stats.get("count", 0) > 0:
                print(
                    f"    Distance: {stats.get('total_distance_m', 0):.2f}m  "
                    f"Speed avg/max: "
                    f"{stats.get('speed_avg_ms', 0):.2f}/"
                    f"{stats.get('speed_max_ms', 0):.2f} m/s"
                )
            elif mt == "steeranglefeedback" and stats.get("count", 0) > 0:
                print(
                    f"    Steer: min={stats.get('steer_min_deg', 0):.1f}°  "
                    f"max={stats.get('steer_max_deg', 0):.1f}°"
                )
            elif mt == "vehicle_monitor" and stats.get("count", 0) > 0:
                print(
                    f"    Speed avg/max: "
                    f"{stats.get('vehicle_speed_avg_ms', 0):.2f}/"
                    f"{stats.get('vehicle_speed_max_ms', 0):.2f} m/s"
                )
        print(f"\n{'='*60}\n")

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _time_axis(entries: List[Dict]) -> np.ndarray:
        """Return seconds-from-start for *entries* ``utime`` fields."""
        if not entries:
            return np.array([])
        utimes = np.array([e["utime"] for e in entries])
        return (utimes - utimes[0]) / 1e6

    def plot_route(
        self, scene_name: str, save_path: Optional[str] = None
    ) -> None:
        """Render a standalone route map coloured by speed.

        Args:
            scene_name: Scene identifier.
            save_path: Directory to save the PNG, or ``None`` for interactive
                display.
        """
        if not HAS_MATPLOTLIB:
            print("Cannot plot: matplotlib not installed")
            return
        data = self.load_scene_data(scene_name)
        pose_data = data.get("pose", [])
        if not pose_data:
            print(f"No pose data for {scene_name}")
            return

        positions = np.array([e["pos"] for e in pose_data])
        velocities = np.array([e["vel"] for e in pose_data])
        x, y = positions[:, 0], positions[:, 1]
        speeds = np.linalg.norm(velocities, axis=1)

        fig, ax = plt.subplots(figsize=(10, 10))
        fig.suptitle(f"Route: {scene_name}", fontsize=14, fontweight="bold")

        pts = np.column_stack([x, y]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        norm = plt.Normalize(speeds.min(), speeds.max())
        lc = LineCollection(segs, cmap="plasma", norm=norm, linewidths=3)
        lc.set_array(speeds[:-1])
        ax.add_collection(lc)
        fig.colorbar(lc, ax=ax, label="Speed (m/s)", shrink=0.8)

        ax.scatter(x, y, c=speeds, cmap="plasma", norm=norm, s=50,
                   zorder=5, edgecolors="white", linewidths=0.5)
        ax.plot(x[0], y[0], "o", color="lime", markersize=15,
                markeredgecolor="black", markeredgewidth=2, label="Start", zorder=10)
        ax.plot(x[-1], y[-1], "s", color="red", markersize=15,
                markeredgecolor="black", markeredgewidth=2, label="End", zorder=10)

        arrow_idx = np.linspace(0, len(x) - 2, min(10, len(x) - 1), dtype=int)
        for i in arrow_idx:
            dx, dy = x[i + 1] - x[i], y[i + 1] - y[i]
            if np.hypot(dx, dy) > 0.1:
                ax.annotate(
                    "",
                    xy=(x[i + 1], y[i + 1]),
                    xytext=(x[i], y[i]),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
                    zorder=6,
                )

        total_dist = float(
            np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
        )
        stats_text = (
            f"Distance: {total_dist:.1f} m\n"
            f"Avg speed: {speeds.mean():.2f} m/s\n"
            f"Max speed: {speeds.max():.2f} m/s\n"
            f"Samples: {len(pose_data)}"
        )
        ax.text(
            0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        margin = max(x.max() - x.min(), y.max() - y.min()) * 0.1 or 1.0
        ax.set_xlim(x.min() - margin, x.max() + margin)
        ax.set_ylim(y.min() - margin, y.max() + margin)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect("equal")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3, linestyle="--")
        plt.tight_layout()

        if save_path:
            out = Path(save_path)
            out.mkdir(parents=True, exist_ok=True)
            fp = out / f"{scene_name}_route.png"
            plt.savefig(fp, dpi=150, bbox_inches="tight")
            print(f"Saved route plot: {fp}")
        else:
            plt.show()
        plt.close()

    def plot_scene(
        self, scene_name: str, save_path: Optional[str] = None
    ) -> None:
        """Render a multi-panel CAN bus overview for *scene_name*.

        Args:
            scene_name: Scene identifier.
            save_path: Directory to save the PNG, or ``None`` for interactive
                display.
        """
        if not HAS_MATPLOTLIB:
            print("Cannot plot: matplotlib not installed")
            return
        data = self.load_scene_data(scene_name)

        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(f"CAN Bus: {scene_name}", fontsize=14, fontweight="bold")
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        self._plot_trajectory(fig.add_subplot(gs[0, :2]), data.get("pose", []))
        self._plot_speed(
            fig.add_subplot(gs[0, 2]),
            data.get("pose", []),
            data.get("vehicle_monitor", []),
        )
        self._plot_velocity_components(
            fig.add_subplot(gs[1, 0]), data.get("pose", [])
        )
        self._plot_acceleration(
            fig.add_subplot(gs[1, 1]), data.get("pose", [])
        )
        self._plot_orientation(
            fig.add_subplot(gs[1, 2]), data.get("pose", [])
        )
        self._plot_steering(
            fig.add_subplot(gs[2, 0]), data.get("steeranglefeedback", [])
        )
        self._plot_yaw_rate(
            fig.add_subplot(gs[2, 1]), data.get("vehicle_monitor", [])
        )
        self._plot_imu_rotation(
            fig.add_subplot(gs[2, 2]), data.get("ms_imu", [])
        )
        plt.tight_layout()

        if save_path:
            out = Path(save_path)
            out.mkdir(parents=True, exist_ok=True)
            fp = out / f"{scene_name}_canbus.png"
            plt.savefig(fp, dpi=150, bbox_inches="tight")
            print(f"Saved canbus plot: {fp}")
        else:
            plt.show()
        plt.close()

    # subplot helpers ---------------------------------------------------------

    def _plot_trajectory(self, ax, pose_data):
        ax.set_title("Trajectory (BEV)")
        if not pose_data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            return
        pos = np.array([e["pos"] for e in pose_data])
        x, y = pos[:, 0], pos[:, 1]
        colors = np.linspace(0, 1, len(x))
        sc = ax.scatter(x, y, c=colors, cmap="viridis", s=20, alpha=0.7)
        ax.plot(x, y, "k-", alpha=0.3, linewidth=0.5)
        ax.plot(x[0], y[0], "go", markersize=10, label="Start")
        ax.plot(x[-1], y[-1], "ro", markersize=10, label="End")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect("equal")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax, label="Time")

    def _plot_speed(self, ax, pose_data, vm_data):
        ax.set_title("Speed Profile")
        if pose_data:
            t = self._time_axis(pose_data)
            speeds = np.linalg.norm([e["vel"] for e in pose_data], axis=1)
            ax.plot(t, speeds, "b-", label="Pose vel", linewidth=1.5)
        if vm_data:
            t = self._time_axis(vm_data)
            ax.plot(t, [e["vehicle_speed"] for e in vm_data], "r--",
                    label="VM speed", linewidth=1.5, alpha=0.7)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Speed (m/s)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_velocity_components(self, ax, pose_data):
        ax.set_title("Velocity Components")
        if not pose_data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            return
        t = self._time_axis(pose_data)
        vels = np.array([e["vel"] for e in pose_data])
        for i, (label, c) in enumerate(zip(["Vx", "Vy", "Vz"], "rgb")):
            ax.plot(t, vels[:, i], f"{c}-", label=label, linewidth=1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Velocity (m/s)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_acceleration(self, ax, pose_data):
        ax.set_title("Acceleration Magnitude")
        if not pose_data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            return
        t = self._time_axis(pose_data)
        mag = np.linalg.norm([e["accel"] for e in pose_data], axis=1)
        ax.plot(t, mag, "b-", linewidth=1)
        ax.fill_between(t, 0, mag, alpha=0.3)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Acceleration (m/s²)")
        ax.grid(True, alpha=0.3)

    def _plot_orientation(self, ax, pose_data):
        ax.set_title("Orientation (Quaternion WXYZ)")
        if not pose_data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            return
        t = self._time_axis(pose_data)
        ori = np.array([e["orientation"] for e in pose_data])
        for i, (lbl, c) in enumerate(zip("wxyz", "rgbm")):
            ax.plot(t, ori[:, i], f"{c}-", label=lbl, linewidth=1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Quaternion component")
        ax.legend(loc="upper right", ncol=2)
        ax.grid(True, alpha=0.3)

    def _plot_steering(self, ax, steer_data):
        ax.set_title("Steering Angle")
        if not steer_data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            return
        t = self._time_axis(steer_data)
        ax.plot(t, [e["value"] for e in steer_data], "b-", linewidth=1.5)
        ax.axhline(0, color="k", linestyle="--", alpha=0.3)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angle (°)")
        ax.grid(True, alpha=0.3)

    def _plot_yaw_rate(self, ax, vm_data):
        ax.set_title("Yaw Rate")
        if not vm_data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            return
        t = self._time_axis(vm_data)
        yaw_deg = np.degrees([e["yaw_rate"] for e in vm_data])
        ax.plot(t, yaw_deg, "b-", linewidth=1.5)
        ax.axhline(0, color="k", linestyle="--", alpha=0.3)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Yaw rate (°/s)")
        ax.grid(True, alpha=0.3)

    def _plot_imu_rotation(self, ax, imu_data):
        ax.set_title("IMU Rotation Rate")
        if not imu_data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            return
        t = self._time_axis(imu_data)
        rr = np.degrees([e["rotation_rate"] for e in imu_data])
        for i, (lbl, c) in enumerate(
            zip(["Roll rate", "Pitch rate", "Yaw rate"], "rgb")
        ):
            ax.plot(t, rr[:, i], f"{c}-", label=lbl, linewidth=1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Rate (°/s)")
        ax.legend()
        ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Verify and visualize CAN bus data from a nuScenes-format dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 tools/viz/visualize_canbus.py output/data/R1 --list
    python3 tools/viz/visualize_canbus.py output/data/R1 --scene mini_01
    python3 tools/viz/visualize_canbus.py output/data/R1 --scene mini_01 --plot
    python3 tools/viz/visualize_canbus.py output/data/R1 --scene mini_01 --route
    python3 tools/viz/visualize_canbus.py output/data/R1 --all
""",
    )
    parser.add_argument("dataset_dir", help="Path to dataset root (e.g. output/data/R1)")
    parser.add_argument("--scene", help="Scene name (e.g. mini_01)")
    parser.add_argument("--list", action="store_true", help="List available scenes")
    parser.add_argument("--all", action="store_true", help="Verify all scenes")
    parser.add_argument("--plot", action="store_true", help="Show all plots")
    parser.add_argument("--route", action="store_true", help="Show route map only")
    parser.add_argument("--save", help="Save plots to this directory")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    try:
        visualizer = CANBusVisualizer(args.dataset_dir)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    if args.list:
        scenes = visualizer.list_scenes()
        print(f"\nAvailable scenes in {args.dataset_dir}:")
        for s in scenes:
            print(f"  - {s}")
        print(f"\nTotal: {len(scenes)}")
        return

    if args.all:
        scenes = visualizer.list_scenes()
    elif args.scene:
        scenes = [args.scene]
    else:
        print("Error: specify --scene <name>, --list, or --all")
        parser.print_help()
        sys.exit(1)

    all_results = []
    for scene in scenes:
        results = visualizer.verify_data(scene)
        all_results.append(results)
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            visualizer.print_report(results)
        if args.plot or args.save:
            visualizer.plot_scene(scene, args.save)
        if args.route or args.save:
            visualizer.plot_route(scene, args.save)

    if len(scenes) > 1 and not args.json:
        valid = sum(1 for r in all_results if r["valid"])
        print(f"\n{'='*60}")
        print(f"Summary: {valid}/{len(scenes)} valid")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
