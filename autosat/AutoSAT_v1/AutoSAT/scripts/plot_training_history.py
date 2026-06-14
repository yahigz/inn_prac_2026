#!/usr/bin/env python3
"""Plot AutoSAT training history with a 4-color task cycle.

The script reads iteration artifacts such as iter_0_result.json from a run
directory and writes one PNG per metric. Points are colored by iteration modulo
the cycle length so repeated tasks can be compared visually.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


DEFAULT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def _load_iteration_results(run_root: Path) -> list[dict]:
    result_files = sorted(
        run_root.glob("iter_*_result.json"),
        key=lambda path: int(re.search(r"iter_(\d+)_result\.json$", path.name).group(1)),
    )
    iterations: list[dict] = []
    for path in result_files:
        match = re.search(r"iter_(\d+)_result\.json$", path.name)
        if not match:
            continue
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        iterations.append({"iter": int(match.group(1)), "payload": payload, "path": path})
    return iterations


def _best_value(section: dict, metric: str) -> float | None:
    values = section.get(metric, {})
    if not values:
        return None
    numeric_values = [float(value) for value in values.values() if value is not None]
    if not numeric_values:
        return None
    return min(numeric_values)


def _series(iterations: list[dict], metric: str) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for item in iterations:
        payload = item["payload"]
        value = _best_value(payload, metric)
        if value is None:
            continue
        xs.append(item["iter"])
        ys.append(value)
    return xs, ys


def _cycle_color(index: int, cycle_length: int, colors: list[str]) -> str:
    return colors[index % cycle_length % len(colors)]


def _plot_metric(
    iterations: list[dict],
    metric: str,
    output_path: Path,
    cycle_length: int,
    colors: list[str],
    title: str,
) -> None:
    xs, ys = _series(iterations, metric)
    if not xs:
        raise ValueError(f"No values found for metric '{metric}' in {output_path.parent}")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(xs, ys, color="#9ca3af", linewidth=1.2, alpha=0.6, zorder=1)

    for x, y in zip(xs, ys):
        color = _cycle_color(x, cycle_length, colors)
        ax.scatter(x, y, s=44, color=color, edgecolors="white", linewidths=0.6, zorder=2)

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[i % len(colors)], markeredgecolor="white",
               markersize=8, label=f"function {i + 1}")
        for i in range(cycle_length)
    ]
    ax.legend(handles=legend_handles, title=f"Cycle {cycle_length}", ncols=min(cycle_length, 4))

    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot AutoSAT training history.")
    parser.add_argument("run_root", type=Path, help="Run folder containing iter_*_result.json files")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for PNG outputs")
    parser.add_argument("--cycle-length", type=int, default=4, help="Color cycle length for repeated tasks")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["time", "PAR-2"],
        help="Metrics to plot from each iter_*_result.json file",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    run_root = args.run_root.expanduser().resolve()
    if not run_root.exists() or not run_root.is_dir():
        raise FileNotFoundError(f"Run root does not exist: {run_root}")

    iterations = _load_iteration_results(run_root)
    if not iterations:
        raise FileNotFoundError(f"No iter_*_result.json files found under: {run_root}")

    output_dir = (args.output_dir or (run_root / "plots")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cycle_length = max(1, int(args.cycle_length))
    for metric in args.metrics:
        output_path = output_dir / f"{metric.replace('/', '_')}_history.png"
        _plot_metric(
            iterations=iterations,
            metric=metric,
            output_path=output_path,
            cycle_length=cycle_length,
            colors=DEFAULT_COLORS,
            title=f"{metric} history with {cycle_length}-cycle colors",
        )
        print(output_path)


if __name__ == "__main__":
    main()