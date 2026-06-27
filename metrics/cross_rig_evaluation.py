#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from utils.plotting import configure_plots, ensure_dir, map_cmap, relative_map_cmap, save_heatmap
from utils.results import build_map_matrix, iter_models, load_standardized_results, relative_map_change_matrix


def _model_slug(model):
    return model.lower().replace("/", "_")


def _save_combined_heatmap(matrices, path, panel_title_template, cmap, fmt, vmin, vmax, center=None):
    configure_plots()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    models = list(matrices)
    fig = plt.figure(figsize=(16, 13))
    grid = fig.add_gridspec(
        2,
        2,
        wspace=0.16,
        hspace=0.34,
        left=0.055,
        right=0.98,
        bottom=0.08,
        top=0.94,
    )
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    ]

    for idx, (ax, model) in enumerate(zip(axes, models)):
        matrix = matrices[model]
        sns.heatmap(
            matrix,
            ax=ax,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            center=center,
            vmin=vmin,
            vmax=vmax,
            linewidths=0.5,
            linecolor="#E0E0E0",
            cbar=False,
            annot_kws={"size": 10},
        )
        ax.set_title(panel_title_template.format(model=model))
        ax.set_xlabel("Evaluation Rig")
        ax.set_ylabel("Training Rig" if idx % 2 == 0 else "")
        ax.tick_params(axis="x", rotation=45)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
        ax.tick_params(axis="y", rotation=0)

    for ax in axes[len(models) :]:
        ax.axis("off")

    plt.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def run(results_root, output_root):
    output_dir = Path(output_root) / "cross_rig_evaluation"
    ensure_dir(output_dir)

    results = load_standardized_results(results_root)
    if results.empty:
        raise SystemExit(f"No standardized results found in {results_root}")

    absolute_matrices = {}
    relative_matrices = {}

    for model in iter_models(results):
        model_slug = _model_slug(model)
        map_matrix = build_map_matrix(results, model)
        rel_matrix = relative_map_change_matrix(map_matrix)
        absolute_matrices[model] = map_matrix * 100.0
        relative_matrices[model] = rel_matrix

        map_csv = output_dir / f"{model_slug}_absolute_map_matrix.csv"
        rel_csv = output_dir / f"{model_slug}_relative_map_change_matrix.csv"
        map_matrix.to_csv(map_csv)
        rel_matrix.to_csv(rel_csv)

        save_heatmap(
            absolute_matrices[model],
            output_dir / f"{model_slug}_absolute_map_heatmap.png",
            f"{model} Cross-Rig mAP",
            "mAP (%)",
            cmap=map_cmap(),
            fmt=".1f",
            vmin=0,
        )
        save_heatmap(
            relative_matrices[model],
            output_dir / f"{model_slug}_relative_map_change_heatmap.png",
            f"{model} Relative mAP Change",
            "Relative mAP Change (%)",
            cmap=relative_map_cmap(),
            center=0,
            fmt=".1f",
        )
        print(f"[{model}] wrote {map_csv} and {rel_csv}")

    absolute_vmax = max(matrix.max().max() for matrix in absolute_matrices.values())
    relative_absmax = max(abs(matrix).max().max() for matrix in relative_matrices.values())

    _save_combined_heatmap(
        absolute_matrices,
        output_dir / "all_models_absolute_map_heatmap.png",
        "{model} Cross-Rig mAP",
        cmap=map_cmap(),
        fmt=".1f",
        vmin=0,
        vmax=absolute_vmax,
    )
    _save_combined_heatmap(
        relative_matrices,
        output_dir / "all_models_relative_map_change_heatmap.png",
        "{model} Relative Cross-Rig mAP Change",
        cmap=relative_map_cmap(),
        fmt=".1f",
        vmin=-relative_absmax,
        vmax=relative_absmax,
        center=0,
    )
    print(f"[all models] wrote combined heatmaps to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate cross-rig mAP matrices and heatmaps.")
    parser.add_argument("--results-root", default="results", help="Standardized results directory.")
    parser.add_argument("--output-root", default="output", help="Output directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    run(args.results_root, args.output_root)


if __name__ == "__main__":
    main()
