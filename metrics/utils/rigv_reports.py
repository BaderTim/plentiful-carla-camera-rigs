import json
from pathlib import Path

import pandas as pd

from metrics.rigv import compute_rigv
from utils.results import sorted_rigs


def rigv_lambdas_from_weights(weights=None):
    if weights is None:
        return {"t": 1.0, "r": 1.0, "f": 1.0}
    return {
        "t": float(weights["lambda_t"]),
        "r": float(weights["lambda_r"]),
        "f": float(weights["lambda_f"]),
    }


def build_rigv_report(rig_configs, weights=None):
    lambdas = rigv_lambdas_from_weights(weights)
    rows = []
    for rig_name in sorted_rigs(rig_configs):
        metrics = compute_rigv(rig_configs[rig_name], lambdas=lambdas)
        rows.append(
            {
                "rig": rig_name,
                "lambda_t": lambdas["t"],
                "lambda_r": lambdas["r"],
                "lambda_f": lambdas["f"],
                **{key: float(value) for key, value in metrics.items()},
            }
        )
    return pd.DataFrame(rows)


def write_rigv_report(rig_configs, output_path, weights=None, label=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = build_rigv_report(rig_configs, weights=weights)
    report.to_csv(output_path.with_suffix(".csv"), index=False)

    payload = {
        "label": label,
        "lambdas": rigv_lambdas_from_weights(weights),
        "results": report.to_dict(orient="records"),
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return report
