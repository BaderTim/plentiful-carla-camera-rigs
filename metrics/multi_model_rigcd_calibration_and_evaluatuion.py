#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from utils.plotting import ensure_dir, save_heatmap
from utils.results import CONTROL_RIGS, iter_models, load_standardized_results, sorted_rigs
from utils.rig_configs import load_rig_configs
from utils.rigv_reports import write_rigv_report
from utils.rigcd import (
    build_rigcd_dataset,
    control_test_split,
    evaluate_calibration,
    fit_rigcd_weights,
    prediction_matrix,
    ranking_matrix,
)


def run(results_root, rigs_root, output_root):
    output_dir = Path(output_root) / "multi_model_rigcd_calibration_and_evaluation"
    ensure_dir(output_dir)

    results = load_standardized_results(results_root)
    required_rigs = sorted_rigs(set(results["train_rig"]) | set(results["test_rig"]))
    rig_configs = load_rig_configs(rigs_root, required_rigs)
    write_rigv_report(
        rig_configs,
        output_dir / "no_calibration_rigv",
        label="No calibration RigV",
    )
    dataset = build_rigcd_dataset(results, rig_configs)
    control_df, test_df = control_test_split(dataset)

    weights = fit_rigcd_weights(control_df)
    evaluation = evaluate_calibration(control_df, test_df, weights)
    train_df = evaluation["train_df"]
    test_df = evaluation["test_df"]

    train_df.to_csv(output_dir / "combined_control_predictions.csv", index=False)
    test_df.to_csv(output_dir / "combined_heldout_predictions.csv", index=False)

    with (output_dir / "combined_calibrated_weights.json").open("w") as handle:
        json.dump(weights, handle, indent=2, sort_keys=True)

    combined_rigv_base = output_dir / "combined_calibrated_rigv"
    write_rigv_report(
        rig_configs,
        combined_rigv_base,
        weights=weights,
        label="Combined calibrated RigV",
    )

    summary = {}
    for model in iter_models(dataset):
        model_train_df = train_df[train_df["model"] == model]
        model_test_df = test_df[test_df["model"] == model]
        model_evaluation = evaluate_calibration(model_train_df, model_test_df, weights)
        model_metrics = {
            "model": model,
            "control_rigs": list(CONTROL_RIGS),
            "num_control_pairs": int(len(model_train_df)),
            "num_heldout_pairs": int(len(model_test_df)),
            "calibrated_weights": str(output_dir / "combined_calibrated_weights.json"),
            "rigv_report": str(combined_rigv_base.with_suffix(".json")),
            "train": model_evaluation["train"],
            "test": model_evaluation["test"],
        }
        summary[model] = model_metrics
        with (output_dir / f"{model.lower().replace('/', '_')}_metrics.json").open("w") as handle:
            json.dump(model_metrics, handle, indent=2, sort_keys=True)

    with (output_dir / "summary_metrics.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    if not test_df.empty:
        observed = prediction_matrix(test_df, "delta_map_rel")
        predicted = prediction_matrix(test_df, "prediction")
        observed.to_csv(output_dir / "combined_heldout_observed_delta_matrix.csv")
        predicted.to_csv(output_dir / "combined_heldout_predicted_delta_matrix.csv")
        save_heatmap(
            observed * 100.0,
            output_dir / "combined_heldout_observed_delta_heatmap.png",
            "Combined Observed Relative mAP Drop",
            "Observed Drop (%)",
            cmap="YlOrRd",
            fmt=".1f",
            vmin=0,
        )
        save_heatmap(
            predicted * 100.0,
            output_dir / "combined_heldout_predicted_delta_heatmap.png",
            "Combined Predicted Relative mAP Drop",
            "Predicted Drop (%)",
            cmap="YlOrRd",
            fmt=".1f",
            vmin=0,
        )

        ranked = ranking_matrix(test_df)
        rank_error = prediction_matrix(ranked, "rank_error")
        rank_error.to_csv(output_dir / "combined_heldout_rank_error_matrix.csv")
        save_heatmap(
            rank_error,
            output_dir / "combined_heldout_rank_error_heatmap.png",
            "Combined Held-Out Ranking Error",
            "Predicted Rank - Observed Rank",
            cmap="RdBu_r",
            center=0,
            fmt=".1f",
        )

    print("Calibrated and evaluated combined multi-model RigCD.")


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate RigCD on all models combined.")
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
