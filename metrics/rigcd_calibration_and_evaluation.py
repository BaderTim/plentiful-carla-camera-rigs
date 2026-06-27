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


def model_slug(model):
    return model.lower().replace("/", "_")


def write_model_outputs(model, evaluation, output_dir, rig_configs):
    slug = model_slug(model)
    weights_path = output_dir / f"{slug}_calibrated_weights.json"
    metrics_path = output_dir / f"{slug}_metrics.json"
    train_csv = output_dir / f"{slug}_control_predictions.csv"
    test_csv = output_dir / f"{slug}_heldout_predictions.csv"

    train_df = evaluation["train_df"]
    test_df = evaluation["test_df"]
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    with weights_path.open("w") as handle:
        json.dump(evaluation["weights"], handle, indent=2, sort_keys=True)

    rigv_base = output_dir / f"{slug}_calibrated_rigv"
    write_rigv_report(
        rig_configs,
        rigv_base,
        weights=evaluation["weights"],
        label=f"{model} calibrated RigV",
    )

    metrics = {
        "model": model,
        "control_rigs": list(CONTROL_RIGS),
        "num_control_pairs": int(len(train_df)),
        "num_heldout_pairs": int(len(test_df)),
        "rigv_report": str(rigv_base.with_suffix(".json")),
        "train": evaluation["train"],
        "test": evaluation["test"],
    }
    with metrics_path.open("w") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)

    if not test_df.empty:
        observed = prediction_matrix(test_df, "delta_map_rel")
        predicted = prediction_matrix(test_df, "prediction")
        observed.to_csv(output_dir / f"{slug}_heldout_observed_delta_matrix.csv")
        predicted.to_csv(output_dir / f"{slug}_heldout_predicted_delta_matrix.csv")

        save_heatmap(
            observed * 100.0,
            output_dir / f"{slug}_heldout_observed_delta_heatmap.png",
            f"{model} Observed Relative mAP Drop",
            "Observed Drop (%)",
            cmap="YlOrRd",
            fmt=".1f",
            vmin=0,
        )
        save_heatmap(
            predicted * 100.0,
            output_dir / f"{slug}_heldout_predicted_delta_heatmap.png",
            f"{model} Predicted Relative mAP Drop",
            "Predicted Drop (%)",
            cmap="YlOrRd",
            fmt=".1f",
            vmin=0,
        )

        ranked = ranking_matrix(test_df)
        rank_error = prediction_matrix(ranked, "rank_error")
        rank_error.to_csv(output_dir / f"{slug}_heldout_rank_error_matrix.csv")
        save_heatmap(
            rank_error,
            output_dir / f"{slug}_heldout_rank_error_heatmap.png",
            f"{model} Held-Out Ranking Error",
            "Predicted Rank - Observed Rank",
            cmap="RdBu_r",
            center=0,
            fmt=".1f",
        )

    return metrics


def run(results_root, rigs_root, output_root):
    output_dir = Path(output_root) / "rigcd_calibration_and_evaluation"
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
    dataset.to_csv(output_dir / "all_rigcd_pairs.csv", index=False)

    summary = {}
    for model in iter_models(dataset):
        model_df = dataset[dataset["model"] == model]
        control_df, test_df = control_test_split(model_df)
        if control_df.empty:
            print(f"[{model}] skipped: no control pairs")
            continue
        weights = fit_rigcd_weights(control_df)
        evaluation = evaluate_calibration(control_df, test_df, weights)
        summary[model] = write_model_outputs(model, evaluation, output_dir, rig_configs)
        print(f"[{model}] calibrated and evaluated")

    with (output_dir / "summary_metrics.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate and evaluate RigCD per model.")
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
