import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

from metrics.rcd import rig_contrastive_distance
from utils.results import CONTROL_RIGS


def build_rigcd_dataset(results_df, rig_configs):
    rows = []
    for row in results_df.itertuples(index=False):
        if row.train_rig not in rig_configs or row.test_rig not in rig_configs:
            continue
        if row.mAP is None or pd.isna(row.mAP):
            continue

        rows.append(
            {
                "model": row.model,
                "train_rig": row.train_rig,
                "test_rig": row.test_rig,
                "mAP": float(row.mAP),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    baseline = (
        df[df["train_rig"] == df["test_rig"]]
        .set_index(["model", "train_rig"])["mAP"]
        .to_dict()
    )

    entries = []
    cache = {}
    for row in df.itertuples(index=False):
        base = baseline.get((row.model, row.train_rig))
        if base is None or base == 0:
            continue

        key = (row.train_rig, row.test_rig)
        if key not in cache:
            cache[key] = rig_contrastive_distance(
                rig_configs[row.train_rig],
                rig_configs[row.test_rig],
                alpha=0.9,
                lambdas={"t": 1.0, "r": 1.0, "f": 1.0},
            )
        rcd = cache[key]
        entries.append(
            {
                "model": row.model,
                "train_rig": row.train_rig,
                "test_rig": row.test_rig,
                "mAP": row.mAP,
                "baseline_mAP": base,
                "delta_map_rel": (base - row.mAP) / base,
                "rcd_t": rcd["RCD_t"],
                "rcd_r": rcd["RCD_r"],
                "rcd_f": rcd["RCD_f"],
                "rcd_count": rcd["RCD_count"],
            }
        )
    return pd.DataFrame(entries)


def control_test_split(df, control_rigs=CONTROL_RIGS):
    control = set(control_rigs)
    is_control = df["train_rig"].isin(control) & df["test_rig"].isin(control)
    return df[is_control].copy(), df[~is_control].copy()


def fit_rigcd_weights(train_df):
    if train_df.empty:
        raise ValueError("Cannot calibrate RigCD with an empty training set.")

    features = train_df[["rcd_t", "rcd_r", "rcd_f", "rcd_count"]].to_numpy(dtype=float)
    target = train_df["delta_map_rel"].to_numpy(dtype=float)

    def loss_fn(params):
        prediction = features @ params
        return np.sum((prediction - target) ** 2)

    result = minimize(
        loss_fn,
        x0=np.array([0.5, 0.5, 0.5, 0.5]),
        bounds=[(0.0, None), (0.0, None), (0.0, None), (0.0, None)],
        method="L-BFGS-B",
    )
    beta_t, beta_r, beta_f, beta_count = result.x

    avg_match_weight = (beta_t + beta_r + beta_f) / 3.0
    scale = avg_match_weight + beta_count
    if scale > 1e-9 and avg_match_weight > 1e-9:
        alpha = avg_match_weight / scale
        lambda_t = beta_t / avg_match_weight
        lambda_r = beta_r / avg_match_weight
        lambda_f = beta_f / avg_match_weight
    else:
        scale = max(float(scale), 1.0)
        alpha = 0.9
        lambda_t = lambda_r = lambda_f = 1.0

    return {
        "K": float(scale),
        "alpha": float(alpha),
        "lambda_t": float(lambda_t),
        "lambda_r": float(lambda_r),
        "lambda_f": float(lambda_f),
        "beta_t": float(beta_t),
        "beta_r": float(beta_r),
        "beta_f": float(beta_f),
        "beta_count": float(beta_count),
        "optimizer_success": bool(result.success),
        "optimizer_loss": float(result.fun),
    }


def predict_with_weights(df, weights):
    pred = (
        weights["K"]
        * (
            weights["alpha"]
            * (
                weights["lambda_t"] * df["rcd_t"]
                + weights["lambda_r"] * df["rcd_r"]
                + weights["lambda_f"] * df["rcd_f"]
            )
            + (1.0 - weights["alpha"]) * df["rcd_count"]
        )
    )
    return pred.astype(float)


def reliability(df, target_col="delta_map_rel", pred_col="prediction", n_boot=1000, seed=0):
    if len(df) < 2:
        return {"rho": None, "p_val": None, "ci": [None, None], "mae": None, "rmse": None}

    target = df[target_col].to_numpy(dtype=float)
    pred = df[pred_col].to_numpy(dtype=float)
    valid = np.isfinite(target) & np.isfinite(pred)
    target = target[valid]
    pred = pred[valid]
    if len(target) < 2:
        return {"rho": None, "p_val": None, "ci": [None, None], "mae": None, "rmse": None}

    rho, p_val = spearmanr(target, pred)
    error = pred - target

    rng = np.random.default_rng(seed)
    boot_stats = []
    indices = np.arange(len(target))
    for _ in range(n_boot):
        boot_idx = rng.choice(indices, size=len(indices), replace=True)
        if len(np.unique(target[boot_idx])) > 1 and len(np.unique(pred[boot_idx])) > 1:
            boot_rho, _ = spearmanr(target[boot_idx], pred[boot_idx])
            if np.isfinite(boot_rho):
                boot_stats.append(boot_rho)

    ci = [None, None]
    if boot_stats:
        ci = [float(np.percentile(boot_stats, 2.5)), float(np.percentile(boot_stats, 97.5))]

    return {
        "rho": float(rho) if np.isfinite(rho) else None,
        "p_val": float(p_val) if np.isfinite(p_val) else None,
        "ci": ci,
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
    }


def evaluate_calibration(train_df, test_df, weights):
    train_eval = train_df.copy()
    test_eval = test_df.copy()
    train_eval["prediction"] = predict_with_weights(train_eval, weights)
    test_eval["prediction"] = predict_with_weights(test_eval, weights) if not test_eval.empty else []
    return {
        "weights": weights,
        "train": reliability(train_eval),
        "test": reliability(test_eval) if not test_eval.empty else None,
        "train_df": train_eval,
        "test_df": test_eval,
    }


def prediction_matrix(df, value_col):
    matrix = df.pivot_table(index="train_rig", columns="test_rig", values=value_col, aggfunc="mean")
    return matrix


def ranking_matrix(df):
    ranked = df.copy()
    ranked["observed_rank"] = ranked.groupby(["model", "train_rig"])["delta_map_rel"].rank(
        method="average", ascending=False
    )
    ranked["predicted_rank"] = ranked.groupby(["model", "train_rig"])["prediction"].rank(
        method="average", ascending=False
    )
    ranked["rank_error"] = ranked["predicted_rank"] - ranked["observed_rank"]
    return ranked

