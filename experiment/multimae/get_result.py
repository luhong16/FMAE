#!/usr/bin/env python3
import os
import argparse
import numpy as np


def file_to_string(filename):
    with open(filename, "r") as file:
        return file.read()


def get_log_root(log_root):
    if os.path.isabs(log_root):
        return log_root

    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, log_root),
        os.path.abspath(log_root),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path
    return os.path.join(script_dir, log_root)


def print_regression_result(title, path, brand_num_list, dataset_name, metric_key, scale_func):
    path = path + r"/f{fold_num}_b{brand_num}.txt"
    result = []

    print("\n" + title)
    for b, n in zip(brand_num_list, dataset_name):
        rmse_list = []
        print("{} RMSE: ".format(n), end=" ")
        for f in range(5):
            specify_name = path.format(fold_num=f, brand_num=b)
            res = file_to_string(specify_name)
            if "Training time" in res:
                rmse = None
                for line in reversed(res.split("\n")):
                    if metric_key in line:
                        rmse = scale_func(float(line.split()[-1]), b)
                        rmse_list.append(rmse)
                        break
            else:
                rmse = None

            print(rmse, end=" ")
        print("Average:", np.mean(rmse_list))
        result.append(np.mean(rmse_list))
    overall = np.mean(result)
    print("Overall average: ", overall)
    return overall


def print_anomaly_result(path):
    path = path + r"/f{fold_num}_b{brand_num}.txt"
    dataset_name = ["EV1", "EV2", "EV4", "EV5", "EV6"]
    brand_num_list = [1, 2, 4, 5, 6]
    result = []

    print("\nanomaly")
    for b, n in zip(brand_num_list, dataset_name):
        auc_list = []
        print("{} AUROC: ".format(n), end=" ")
        for f in range(5):
            specify_name = path.format(fold_num=f, brand_num=b)
            res = file_to_string(specify_name)
            if "Training time" in res:
                auc = None
                for line in reversed(res.split("\n")):
                    if "Max test set auroc score:" in line:
                        auc = float(line.split()[-1])
                        auc_list.append(auc)
                        break
            else:
                auc = None

            print(auc, end=" ")
        print("Average:", np.mean(auc_list))
        result.append(np.mean(auc_list))
    overall = np.mean(result)
    print("Overall average: ", overall)
    return overall


def print_capacity_result(ev_path, bess_path):
    normalize = [-1, 46.32390322689564, 45.28971041043597, 44.034358340136066,
                 26.538070600693814, 35.473890427318786, 22.28958766646174,
                 95.9036963]
    dataset_name = ["EV1", "EV2", "EV3", "EV4", "EV5", "EV6", "BESS1"]
    brand_num_list = [1, 2, 3, 4, 5, 6, 7]
    result = []

    print("\nsystem-level capacity")
    for b, n in zip(brand_num_list, dataset_name):
        if b == 7:
            path = bess_path + r"/f{fold_num}_b{brand_num}.txt"
        else:
            path = ev_path + r"/f{fold_num}_b{brand_num}.txt"

        rmse_list = []
        print("{} RMSE: ".format(n), end=" ")
        for f in range(5):
            specify_name = path.format(fold_num=f, brand_num=b)
            res = file_to_string(specify_name)
            if "Training time" in res:
                rmse = None
                for line in reversed(res.split("\n")):
                    if "Min test set RMSE:" in line:
                        rmse = float(line.split()[-1]) * 10000 / normalize[b]
                        rmse_list.append(rmse)
                        break
            else:
                rmse = None

            print(rmse, end=" ")
        print("Average:", np.mean(rmse_list))
        result.append(np.mean(rmse_list))
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
    parser.add_argument("--log_root", type=str, default="logs")
    parser.add_argument("--task", type=str, default="all",
                        choices=["all", "anomaly", "capacity", "cell_capacity", "RUL", "IR"])
    args = parser.parse_args()

    log_root = get_log_root(args.log_root)
    ckpt_model = "799_pretrain_multivit_half"
    summary = []

    if args.task in ["all", "anomaly"]:
        anomaly_path = os.path.join(
            log_root, "anomaly", "no", ckpt_model,
            "epoch10_blr5e-3_bsz32_ld0.65_wd0.01_dp0.1", "s5")
        summary.append(("anomaly", print_anomaly_result(anomaly_path)))

    if args.task in ["all", "capacity"]:
        capacity_ev_path = os.path.join(
            log_root, "capacity", "no", ckpt_model,
            "epoch20_blr8e-1_bsz32_ld0.8_wd0.01_dp0.0", "s5")
        capacity_bess_path = os.path.join(
            log_root, "capacity", "no", ckpt_model,
            "epoch20_blr8e-0_bsz32_ld0.8_wd0.01_dp0.0", "s5")
        summary.append(("system-level capacity", print_capacity_result(capacity_ev_path, capacity_bess_path)))

    if args.task in ["all", "cell_capacity"]:
        cell_capacity_path = os.path.join(
            log_root, "capacity", "max_min_volt_temp", ckpt_model,
            "epoch20_blr8e-1_bsz32_ld0.65_wd0.01_dp0.0", "s5")
        summary.append(("cell-level capacity", print_regression_result(
            "cell-level capacity",
            cell_capacity_path,
            [10, 11, 12, 13],
            ["MIT1", "THU", "MIT2", "KIT"],
            "Min test set RMSE:",
            capacity_scale,
        )))

    if args.task in ["all", "RUL"]:
        rul_path = os.path.join(
            log_root, "RUL", "max_min_volt_temp_cyclegap20", ckpt_model,
            "epoch100_blr1e-0_bsz32_ld0.5_wd0.005_dp0.0", "s5")
        summary.append(("RUL", print_regression_result(
            "RUL",
            rul_path,
            [10, 12, 13],
            ["MIT1", "MIT2", "KIT"],
            "Min test cell level RMSE:",
            no_scale,
        )))

    if args.task in ["all", "IR"]:
        ir_path = os.path.join(
            log_root, "IR", "max_min_volt_temp", ckpt_model,
            "epoch20_blr5e-1_bsz32_ld0.8_wd0.01_dp0.0", "s5")
        summary.append(("IR", print_regression_result(
            "IR",
            ir_path,
            [10],
            ["MIT1"],
            "Min test set RMSE:",
            ir_scale,
        )))

    print("\nSummary")
    for name, average in summary:
        print("{}: {}".format(name, average))
