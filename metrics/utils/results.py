import json
import re
from pathlib import Path

import pandas as pd


CONTROL_RIGS = ("R1", "R1-c10", "R1-c6", "R1-f", "R1-r", "R1-t")


def normalize_rig_name(name):
    normalized = str(name).strip()
    if normalized.startswith("train_"):
        normalized = normalized[len("train_") :]
    if normalized.startswith("trained_on_"):
        normalized = normalized[len("trained_on_") :]
    if normalized.endswith(".json"):
        normalized = normalized[:-5]
    normalized = normalized.upper().replace("S", "R")
    for suffix in ("-C10", "-C6", "-F", "-R", "-T"):
        normalized = normalized.replace(suffix, suffix.lower())
    return normalized


def rig_sort_key(rig_name):
    normalized = normalize_rig_name(rig_name)
    match = re.fullmatch(r"R(\d+)(?:-(.*))?", normalized)
    if not match:
        return (float("inf"), normalized)

    suffix = match.group(2) or ""
    suffix_order = {
        "": 0,
        "c10": 1,
        "c6": 2,
        "f": 3,
        "r": 4,
        "t": 5,
    }
    return (int(match.group(1)), suffix_order.get(suffix, 99), suffix)


def sorted_rigs(rigs):
    return sorted({normalize_rig_name(rig) for rig in rigs}, key=rig_sort_key)


def load_standardized_results(results_root):
    results_root = Path(results_root)
    if not results_root.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_root}")

    rows = []
    for model_dir in sorted(path for path in results_root.iterdir() if path.is_dir()):
        model = model_dir.name
        for train_file in sorted(model_dir.glob("trained_on_*.json")):
            with train_file.open() as handle:
                payload = json.load(handle)

            train_rig = normalize_rig_name(payload.get("trained_on") or train_file.stem)
            for test_rig, result in payload.get("results", {}).items():
                metrics = result.get("metrics", {})
                rows.append(
                    {
                        "model": model,
                        "train_rig": train_rig,
                        "test_rig": normalize_rig_name(test_rig),
                        "mAP": metrics.get("mAP"),
                        "NDS": metrics.get("NDS"),
                        "source_log": result.get("source", {}).get("log"),
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["model", "train_rig", "test_rig"], key=lambda col: col.map(str))


def build_map_matrix(df, model):
    model_df = df[df["model"] == model]
    train_rigs = sorted_rigs(model_df["train_rig"])
    test_rigs = sorted_rigs(model_df["test_rig"])
    matrix = pd.DataFrame(index=train_rigs, columns=test_rigs, dtype=float)

    for row in model_df.itertuples(index=False):
        matrix.loc[row.train_rig, row.test_rig] = row.mAP
    return matrix.dropna(how="all", axis=0).dropna(how="all", axis=1)


def relative_map_change_matrix(map_matrix):
    relative = pd.DataFrame(index=map_matrix.index, columns=map_matrix.columns, dtype=float)
    for train_rig in map_matrix.index:
        if train_rig not in map_matrix.columns:
            continue
        baseline = map_matrix.loc[train_rig, train_rig]
        if pd.isna(baseline) or baseline == 0:
            continue
        relative.loc[train_rig] = (map_matrix.loc[train_rig] - baseline) / baseline * 100.0
    return relative.dropna(how="all", axis=0).dropna(how="all", axis=1)


def delta_map_rel_matrix(map_matrix):
    relative_change = relative_map_change_matrix(map_matrix)
    return -relative_change / 100.0


def iter_models(df):
    return sorted(df["model"].dropna().unique())

