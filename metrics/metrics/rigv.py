import json
import sys

import numpy as np


def euler_to_rotmat(pitch, yaw, roll):
    p = np.radians(pitch)
    y = np.radians(yaw)
    r = np.radians(roll)

    ry = np.array(
        [
            [np.cos(p), 0, -np.sin(p)],
            [0, 1, 0],
            [np.sin(p), 0, np.cos(p)],
        ]
    )
    rz = np.array(
        [
            [np.cos(y), -np.sin(y), 0],
            [np.sin(y), np.cos(y), 0],
            [0, 0, 1],
        ]
    )
    rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(r), -np.sin(r)],
            [0, np.sin(r), np.cos(r)],
        ]
    )

    return rz @ ry @ rx


def rotation_distance(rot_a, rot_b):
    cos_theta = (np.trace(rot_a.T @ rot_b) - 1.0) / 2.0
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))


def compute_rigv(rig_json, lambdas=None, eps=1e-9):
    if lambdas is None:
        lambdas = {"t": 1.0, "r": 1.0, "f": 1.0}

    cameras = rig_json["cameras"]
    num_cameras = len(cameras)
    if num_cameras < 2:
        return {
            "RigV": 0.0,
            "TV": 0.0,
            "RV": 0.0,
            "FV": 0.0,
            "TV_norm": 0.0,
            "RV_norm": 0.0,
            "FV_norm": 0.0,
            "Dmax_rig": 0.0,
        }

    locations = np.array([camera["location"] for camera in cameras], dtype=np.float64)
    rotations = [euler_to_rotmat(*camera["rotation"]) for camera in cameras]
    fovs_deg = np.array([camera["fov"] for camera in cameras], dtype=np.float64)

    sum_dt = 0.0
    sum_dr = 0.0
    sum_df = 0.0
    dmax_rig = 0.0

    for i in range(num_cameras):
        for j in range(i + 1, num_cameras):
            dt = np.linalg.norm(locations[i] - locations[j])
            dr = rotation_distance(rotations[i], rotations[j])
            df = abs(fovs_deg[i] - fovs_deg[j])

            sum_dt += dt
            sum_dr += dr
            sum_df += df
            dmax_rig = max(dmax_rig, dt)

    num_pairs = num_cameras * (num_cameras - 1) / 2.0
    tv_raw = sum_dt / num_pairs
    rv_raw = sum_dr / num_pairs
    fv_raw = sum_df / num_pairs

    tv_norm = np.clip(tv_raw / (dmax_rig + eps), 0.0, 1.0)
    rv_norm = np.clip(rv_raw / np.pi, 0.0, 1.0)
    fv_norm = np.clip(fv_raw / 180.0, 0.0, 1.0)

    rigv = lambdas["t"] * tv_norm + lambdas["r"] * rv_norm + lambdas["f"] * fv_norm

    return {
        "RigV": rigv,
        "TV": tv_raw,
        "RV": rv_raw,
        "FV": fv_raw,
        "TV_norm": tv_norm,
        "RV_norm": rv_norm,
        "FV_norm": fv_norm,
        "Dmax_rig": dmax_rig,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m metrics.rigv rig.json [rig2.json ...]")
        raise SystemExit(1)

    for path in sys.argv[1:]:
        with open(path) as handle:
            rig = json.load(handle)
        metrics = compute_rigv(rig)

        print(f"{path}:")
        print(f"  Dmax_rig (max pairwise dt) = {metrics['Dmax_rig']:.4f} m")
        print(f"  TV  = {metrics['TV']:.4f} m   -> TV_norm = {metrics['TV_norm']:.4f}")
        print(f"  RV  = {metrics['RV']:.4f} rad -> RV_norm = {metrics['RV_norm']:.4f}")
        print(f"  FV  = {metrics['FV']:.4f} deg -> FV_norm = {metrics['FV_norm']:.4f}")
        print(f"  RigV(norm-weighted sum)   = {metrics['RigV']:.4f}\n")


if __name__ == "__main__":
    main()

