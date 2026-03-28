"""Utilities for persisting reports and plots."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_metrics(metrics: dict[str, float], output_path: Path) -> None:
    output_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_table(frame: pd.DataFrame, output_path: Path) -> None:
    frame.to_csv(output_path, index=False)


def plot_confusion_heatmap(confusion_table: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(confusion_table.values, cmap="Blues")
    ax.set_xticks(range(len(confusion_table.columns)))
    ax.set_xticklabels(confusion_table.columns)
    ax.set_yticks(range(len(confusion_table.index)))
    ax.set_yticklabels(confusion_table.index)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Normalized Confusion Matrix")
    for row_idx in range(confusion_table.shape[0]):
        for col_idx in range(confusion_table.shape[1]):
            ax.text(col_idx, row_idx, f"{confusion_table.iloc[row_idx, col_idx]:.2f}", ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_subgroup_bars(subgroup_metrics: pd.DataFrame, output_path: Path) -> None:
    if subgroup_metrics.empty:
        return
    top = subgroup_metrics.sort_values("macro_f1").groupby("subgroup").head(6)
    labels = [f"{row.subgroup}: {row.value}" for row in top.itertuples()]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(labels, top["macro_f1"])
    ax.set_xlabel("Macro-F1")
    ax.set_title("Subgroup Performance Snapshot")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
