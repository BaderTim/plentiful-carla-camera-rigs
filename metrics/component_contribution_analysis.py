#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd

from utils.plotting import ensure_dir, save_heatmap
from utils.results import iter_models, load_standardized_results, sorted_rigs
from utils.rig_configs import load_rig_configs
from utils.rigcd import (
    build_rigcd_dataset,
    control_test_split,
    evaluate_calibration,
    fit_rigcd_weights,
    prediction_matrix,
    reliability,
)


def ablation_weights(weights):
    return {
        "full": dict(weights),
        "without_lambda_t": {**weights, "lambda_t": 0.0},
        "without_lambda_r": {**weights, "lambda_r": 0.0},
        "without_lambda_f": {**weights, "lambda_f": 0.0},
        "without_count_alpha": {**weights, "alpha": 1.0},
    }


def evaluate_knockouts(test_df, weights):
    rows = []
    predictions = test_df.copy()
    baseline_rho = None
    for name, params in ablation_weights(weights).items():
        evaluation = evaluate_calibration(test_df, test_df, params)
        pred_col = f"prediction_{name}"
        predictions[pred_col] = evaluation["test_df"]["prediction"].to_numpy()
        metrics = reliability(predictions, pred_col=pred_col)
        if name == "full":
            baseline_rho = metrics["rho"]
        rows.append(
            {
                "component": name,
                "rho": metrics["rho"],
                "rho_drop": None if baseline_rho is None or metrics["rho"] is None else baseline_rho - metrics["rho"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
            }
        )
    return pd.DataFrame(rows), predictions


def run(results_root, rigs_root, output_root):
    output_dir = Path(output_root) / "component_contribution_analysis"
    ensure_dir(output_dir)

    results = load_standardized_results(results_root)
    required_rigs = sorted_rigs(set(results["train_rig"]) | set(results["test_rig"]))
    rig_configs = load_rig_configs(rigs_root, required_rigs)
    dataset = build_rigcd_dataset(results, rig_configs)

    summary = {}
    all_rows = []
    for model in iter_models(dataset):
        model_df = dataset[dataset["model"] == model]
        control_df, test_df = control_test_split(model_df)
        if control_df.empty or test_df.empty:
            print(f"[{model}] skipped: missing control or held-out pairs")
            continue

        weights = fit_rigcd_weights(control_df)
        knockout_df, predictions = evaluate_knockouts(test_df, weights)
        knockout_df.insert(0, "model", model)
        all_rows.append(knockout_df)

        slug = model.lower().replace("/", "_")
        knockout_df.to_csv(output_dir / f"{slug}_component_knockout.csv", index=False)
        predictions.to_csv(output_dir / f"{slug}_component_predictions.csv", index=False)

        heatmap_data = knockout_df.set_index("component")[["rho", "rho_drop", "mae", "rmse"]]
        save_heatmap(
            heatmap_data,
            output_dir / f"{slug}_component_knockout_heatmap.png",
            f"{model} Component Knockout",
            "Metric Value",
            cmap="RdBu_r",
            center=0,
            fmt=".3f",
        )
        summary[model] = {
            "weights": weights,
            "ablations": knockout_df.drop(columns=["model"]).to_dict(orient="records"),
        }
        print(f"[{model}] component knockout complete")

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        combined.to_csv(output_dir / "all_models_component_knockout.csv", index=False)

    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run RigCD component knockout analysis.")
    parser.add_argument("--results-root", default="results", help="Standardized results directory.")
    parser.add_argument("--rigs-root", default="rigs", help="Directory of rig JSON files.")
    parser.add_argument("--output-root", default="output", help="Output directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        run(args.results_root, args.rigs_root, args.output_root)
    except FileNotFoundError as error:
        raise SystemExit(str(error))


if __name__ == "__main__":
    main()
