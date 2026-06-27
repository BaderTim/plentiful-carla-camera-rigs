from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def configure_plots():
    sns.set_context("paper", font_scale=1.7)
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.labelsize": 22,
            "axes.titlesize": 25,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
        }
    )


def save_heatmap(
    matrix,
    path,
    title,
    cbar_label,
    cmap="viridis",
    center=None,
    fmt=".1f",
    vmin=None,
    vmax=None,
):
    configure_plots()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    width = max(10, 0.95 * len(matrix.columns) + 3.5)
    height = max(7, 0.8 * len(matrix.index) + 2.5)
    plt.figure(figsize=(width, height))
    ax = sns.heatmap(
        matrix,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        center=center,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        linecolor="#E0E0E0",
        cbar_kws={"label": cbar_label},
        annot_kws={"size": 15},
    )
    ax.set_title(title, fontsize=25, pad=14)
    ax.set_xlabel("Evaluation Rig", fontsize=22, labelpad=10)
    ax.set_ylabel("Training Rig", fontsize=22, labelpad=10)
    ax.tick_params(axis="both", labelsize=17)
    colorbar = ax.collections[0].colorbar
    if colorbar is not None:
        colorbar.ax.set_ylabel(cbar_label, fontsize=21, labelpad=14)
        colorbar.ax.tick_params(labelsize=16)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def map_cmap():
    return LinearSegmentedColormap.from_list("map_white_blue", ["#FFFFFF", "#5DA5DA"])


def relative_map_cmap():
    return LinearSegmentedColormap.from_list(
        "relative_map_soft_red_blue",
        ["#D6604D", "#FFFFFF", "#4393C3"],
    )
