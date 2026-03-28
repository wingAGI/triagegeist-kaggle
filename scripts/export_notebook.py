#!/usr/bin/env python
"""Export the notebook scaffold to a simple .ipynb file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "notebooks" / "triagegeist_submission.py",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=ROOT / "notebooks" / "triagegeist_submission.ipynb",
    )
    return parser.parse_args()


def flush_cell(cells: list[dict], cell_type: str | None, lines: list[str]) -> None:
    if cell_type is None or not lines:
        return
    if cell_type == "markdown":
        processed = []
        for line in lines:
            if line.startswith("# "):
                processed.append(line[2:])
            elif line.startswith("#"):
                processed.append(line[1:].lstrip())
            else:
                processed.append(line)
        cells.append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": processed,
            }
        )
        return

    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": lines,
        }
    )


def parse_py_to_ipynb(source: Path) -> dict:
    lines = source.read_text(encoding="utf-8").splitlines(keepends=True)
    cells: list[dict] = []
    current_type: str | None = None
    buffer: list[str] = []

    for line in lines:
        if line.startswith("# %% [markdown]"):
            flush_cell(cells, current_type, buffer)
            current_type = "markdown"
            buffer = []
            continue
        if line.startswith("# %%"):
            flush_cell(cells, current_type, buffer)
            current_type = "code"
            buffer = []
            continue
        if current_type is None:
            current_type = "code"
        buffer.append(line)

    flush_cell(cells, current_type, buffer)
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.13",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    args = parse_args()
    notebook = parse_py_to_ipynb(args.source)
    args.target.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.target}")


if __name__ == "__main__":
    main()
