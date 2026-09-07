#!/usr/bin/env python3
import os
import re
import ast
import argparse
import numpy as np


METRICS_RE = re.compile(r"metrics (\{.*\})")


def file_to_string(filename):
    with open(filename, "r") as file:
        return file.read()


def get_log_root(log_root):
    if os.path.isabs(log_root):
        return log_root

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "../.."))
    candidates = [
        os.path.join(script_dir, log_root),
        os.path.join(repo_root, log_root),
        os.path.abspath(log_root),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path
    return os.path.join(script_dir, log_root)


def read_metrics(filename):
    if not os.path.exists(filename):
        return []

    res = file_to_string(filename)
    if "Training time" not in res:
        return []

    metrics = []
    for line in res.split("\n"):
        match = METRICS_RE.search(line)
        if match:
            metrics.append(ast.literal_eval(match.group(1)))
    return metrics


def min_metric(filename, metric_key):
    metrics = read_metrics(filename)
    if not metrics:
        return None

    values = []
    for metric in metrics:
        if metric_key in metric:
            values.append(metric[metric_key])
    if not values:
        return None
    return min(values)


def print_result(title, root, task, brand_num_list, dataset_name, metric_key, scale_func):
    result = []

    print("\n" + title)
    for b, n in zip(brand_num_list, dataset_name):
        metric_list = []
        print("{} RMSE: ".format(n), end=" ")
        for f in range(5):
            specify_name = os.path.join(root, task, "b{}".format(b), "f{}.txt".format(f))
            metric = min_metric(specify_name, metric_key)
            if metric is not None:
                metric = scale_func(metric, b)
                metric_list.append(metric)

            print(metric, end=" ")
        average = np.mean(metric_list) if metric_list else float("nan")
        print("Average:", average)
        result.append(average)

    overall = np.mean(result)
    print("Overall average: ", overall)
    return overall


def capacity_scale(rmse, brand_num):
    normalize = [-1, 46.32390322689564, 45.28971041043597, 44.034358340136066,
                 26.538070600693814, 35.473890427318786, 22.28958766646174,
                 95.9036963]
    if brand_num <= 7:
        return rmse * 10000 / normalize[brand_num]
    return rmse * 100.


def ir_scale(rmse, brand_num):
    return rmse * 1000.


def no_scale(rmse, brand_num):
    return rmse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_root", "--root", dest="log_root", type=str, default="logs/lipm")
    parser.add_argument("--task", type=str, default="all",
                        choices=["all", "capacity", "cell_capacity", "RUL", "IR"])
    args = parser.parse_args()

    root = get_log_root(args.log_root)
    summary = []

    if args.task in ["all", "capacity"]:
        summary.append(("system-level capacity", print_result(
            "system-level capacity",
            root,
            "capacity",
            [1, 2, 3, 4, 5, 6, 7],
            ["EV1", "EV2", "EV3", "EV4", "EV5", "EV6", "BESS1"],
            "rmse",
            capacity_scale,
        )))

    if args.task in ["all", "cell_capacity"]:
        summary.append(("cell-level capacity", print_result(
            "cell-level capacity",
            root,
            "capacity",
            [10, 11, 12, 13],
            ["MIT1", "THU", "MIT2", "KIT"],
            "rmse",
            capacity_scale,
        )))

    if args.task in ["all", "IR"]:
        summary.append(("IR", print_result(
            "IR",
            root,
            "IR",
            [10],
            ["MIT1"],
            "rmse",
            ir_scale,
        )))

    if args.task in ["all", "RUL"]:
        summary.append(("RUL", print_result(
            "RUL",
            root,
            "RUL",
            [10, 12, 13],
            ["MIT1", "MIT2", "KIT"],
            "cell_rmse",
            no_scale,
        )))

    print("\nSummary")
    for name, average in summary:
        print("{}: {}".format(name, average))
