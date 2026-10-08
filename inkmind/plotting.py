"""Shared chart styling for the Task 4/5 notebooks.

Model colours match the Task 3 architecture chart (TinyGPT blue, GPT-2 orange).
"""

import matplotlib.pyplot as plt

MODEL_COLORS = {"TinyGPT": "#2a78d6", "GPT-2": "#eb6834"}
INK = "#55554f"
GRID = "#e4e3dc"


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#b0b0a8")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, labelsize=9)
    ax.title.set_color("#1f1f1c")
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def plot_metric_vs_param(df, param, metrics, titles=None, by="model", xlabel=None, suptitle=None):
    """Small multiples: one panel per metric, one line per model (mean over
    prompts and seeds). Lines are direct-labelled at their right end."""
    titles = titles or metrics
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 3.8), squeeze=False)
    groups = list(df[by].unique())
    for ax, metric, title in zip(axes[0], metrics, titles):
        for name in groups:
            sub = df[df[by] == name].groupby(param)[metric].mean()
            color = MODEL_COLORS.get(name, "#2a78d6")
            ax.plot(sub.index, sub.values, color=color, linewidth=2, marker="o", markersize=7,
                    markeredgecolor="white", markeredgewidth=1.5, label=name)
            if len(groups) > 1:
                ax.annotate(name, (sub.index[-1], sub.values[-1]), xytext=(6, 0),
                            textcoords="offset points", va="center", fontsize=9, color=INK)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlabel or param)
        style_axes(ax)
    if len(groups) > 1:
        axes[0][0].legend(frameon=False, fontsize=9)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    return fig
