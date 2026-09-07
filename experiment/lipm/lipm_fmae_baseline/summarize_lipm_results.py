#!/usr/bin/env python3
"""Summarize LiPM downstream logs with the same scaling as get_result scripts."""

import argparse
import ast
import re
from pathlib import Path

import numpy as np


NORMALIZE = [-1, 46.32390322689564, 45.28971041043597, 44.034358340136066,
             26.538070600693814, 35.473890427318786, 22.28958766646174,
             95.9036963]


METRICS_RE = re.compile(r"metrics (\{.*\})")


def read_metrics(path):
    path = Path(path)
    if not path.exists():
        return []
    text = path.read_text(errors="ignore")
    if "Training time" not in text:
        return []
    metrics = []
    for line in text.splitlines():
        match = METRICS_RE.search(line)
        if match:
            metrics.append(ast.literal_eval(match.group(1)))
    return metrics


def min_metric(path, key):
    metrics = read_metrics(path)
    if not metrics:
        return None
    values = [m[key] for m in metrics if key in m]
    return min(values) if values else None


def summarize_capacity(root, task_type):
    if task_type == "EV":
        brand_num_list = [1, 2, 3, 4, 5, 6]
        dataset_name = ["EV1", "EV2", "EV3", "EV4", "EV5", "EV6"]
    elif task_type == "lab":
        brand_num_list = [10, 11, 12, 13]
        dataset_name = ["MIT1", "THU", "MIT2", "KIT"]
    elif task_type == "BESS":
        brand_num_list = [7]
        dataset_name = ["BESS1"]
    else:
        raise ValueError(f"Unsupported capacity type: {task_type}")

    result = []
    lines = []
    for brand, name in zip(brand_num_list, dataset_name):
        rmse_list = []
        fold_values = []
        for fold in range(5):
            path = root / "capacity" / f"b{brand}" / f"f{fold}.txt"
            rmse = min_metric(path, "rmse")
            if rmse is not None:
                if brand <= 7:
                    rmse = rmse * 10000 / NORMALIZE[brand]
                else:
                    rmse = rmse * 100.0
                rmse_list.append(rmse)
            fold_values.append(rmse)
        avg = float(np.mean(rmse_list)) if rmse_list else float("nan")
        result.append(avg)
        lines.append((name, fold_values, avg))
    overall = float(np.mean(result)) if result else float("nan")
    return lines, overall


def summarize_ir(root):
    result = []
    lines = []
    for brand, name in zip([10], ["MIT1"]):
        rmse_list = []
        fold_values = []
        for fold in range(5):
            path = root / "IR" / f"b{brand}" / f"f{fold}.txt"
            rmse = min_metric(path, "rmse")
            if rmse is not None:
                rmse = rmse * 1000.0
                rmse_list.append(rmse)
            fold_values.append(rmse)
        avg = float(np.mean(rmse_list)) if rmse_list else float("nan")
        result.append(avg)
        lines.append((name, fold_values, avg))
    return lines, float(np.mean(result)) if result else float("nan")


def summarize_rul(root):
    result = []
    lines = []
    for brand, name in zip([10, 12, 13], ["MIT1", "MIT2", "KIT"]):
        rmse_list = []
        fold_values = []
        for fold in range(5):
            path = root / "RUL" / f"b{brand}" / f"f{fold}.txt"
            rmse = min_metric(path, "cell_rmse")
            if rmse is not None:
                rmse_list.append(rmse)
            fold_values.append(rmse)
        avg = float(np.mean(rmse_list)) if rmse_list else float("nan")
        result.append(avg)
        lines.append((name, fold_values, avg))
    return lines, float(np.mean(result)) if result else float("nan")


def print_block(title, lines, overall):
    print(title)
    for name, fold_values, avg in lines:
        values = " ".join("None" if v is None else f"{v:.9f}" for v in fold_values)
        print(f"{name} RMSE: {values} Average: {avg:.9f}")
    print(f"Overall average: {overall:.9f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="logs/lipm", type=Path)
    args = parser.parse_args()

    for task_type in ["EV", "BESS", "lab"]:
        lines, overall = summarize_capacity(args.root, task_type)
        print_block(f"Capacity {task_type}", lines, overall)
        print()

    lines, overall = summarize_ir(args.root)
    print_block("IR", lines, overall)
    print()

    lines, overall = summarize_rul(args.root)
    print_block("RUL", lines, overall)


if __name__ == "__main__":
    main()
