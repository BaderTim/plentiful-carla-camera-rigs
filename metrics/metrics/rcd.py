import json
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

from metrics.rigv import euler_to_rotmat, rotation_distance


def rig_contrastive_distance(rig_a, rig_b, alpha=0.9, lambdas=None):
    if lambdas is None:
        lambdas = {"t": 1.0, "r": 1.0, "f": 1.0}

    cameras_a = rig_a["cameras"]
    cameras_b = rig_b["cameras"]
    num_a, num_b = len(cameras_a), len(cameras_b)
    matched_count = min(num_a, num_b)
    max_count = max(num_a, num_b)

    locs_a = np.array([camera["location"] for camera in cameras_a])
    rots_a = [euler_to_rotmat(*camera["rotation"]) for camera in cameras_a]
    fovs_a = np.array([camera["fov"] for camera in cameras_a])

    locs_b = np.array([camera["location"] for camera in cameras_b])
    rots_b = [euler_to_rotmat(*camera["rotation"]) for camera in cameras_b]
    fovs_b = np.array([camera["fov"] for camera in cameras_b])

    cost = np.zeros((num_a, num_b))
    cost_t = np.zeros((num_a, num_b))
    cost_r = np.zeros((num_a, num_b))
    cost_f = np.zeros((num_a, num_b))

    for i in range(num_a):
        for j in range(num_b):
            dt = np.linalg.norm(locs_a[i] - locs_b[j])
            dr = rotation_distance(rots_a[i], rots_b[j])
            df = abs(fovs_a[i] - fovs_b[j])

            cost_t[i, j] = dt
            cost_r[i, j] = dr
            cost_f[i, j] = df
            cost[i, j] = lambdas["t"] * dt + lambdas["r"] * dr + lambdas["f"] * df

    row_ind, col_ind = linear_sum_assignment(cost)

    rcd_t = cost_t[row_ind, col_ind].mean() if matched_count > 0 else 0.0
    rcd_r = cost_r[row_ind, col_ind].mean() if matched_count > 0 else 0.0
    rcd_f = cost_f[row_ind, col_ind].mean() if matched_count > 0 else 0.0
    rcd_match = lambdas["t"] * rcd_t + lambdas["r"] * rcd_r + lambdas["f"] * rcd_f
    rcd_count = abs(num_a - num_b) / max_count if max_count > 0 else 0.0
    rcd_total = alpha * rcd_match + (1.0 - alpha) * rcd_count

    return {
        "RCD": rcd_total,
        "RCD_match": rcd_match,
        "RCD_count": rcd_count,
        "RCD_t": rcd_t,
        "RCD_r": rcd_r,
        "RCD_f": rcd_f,
    }


def load_rig(path):
    with open(path) as handle:
        return json.load(handle)


def main():
    if len(sys.argv) < 3:
        print("Usage: python -m metrics.rcd rig1.json rig2.json")
        raise SystemExit(1)

    rig_a = load_rig(sys.argv[1])
    rig_b = load_rig(sys.argv[2])
    result = rig_contrastive_distance(rig_a, rig_b)

    print(f"RCD({os.path.basename(sys.argv[1])}, {os.path.basename(sys.argv[2])}):")
    print(f"  RCD_t (Matched Translation) = {result['RCD_t']:.4f}")
    print(f"  RCD_r (Matched Rotation)    = {result['RCD_r']:.4f} rad")
    print(f"  RCD_f (Matched FoV)         = {result['RCD_f']:.4f} deg")
    print(f"  RCD_count (Penalty)         = {result['RCD_count']:.4f}")
    print(f"  --> Total RCD               = {result['RCD']:.4f}")


if __name__ == "__main__":
    main()
