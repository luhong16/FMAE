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


def get_model_config(model):
    key = model.lower().replace("-", "").replace("_", "")

    if key == "fmae":
        return {
            "name": "FMAE",
            "parts": [
                "capacity",
                "max_min_volt_temp_gaussian_noise_std0.005",
                "799_pretrain_vit_half_patch16",
                "epoch20_blr5e-2_bsz32_ld0.65_wd0.01_dp0.1",
                "s5",
            ],
        }

    if key in ["itransformer", "itrans"]:
        return {
            "name": "iTransformer",
            "parts": [
                "capacity",
                "use_volt_current_soc_with_noise",
                "iTransformer",
                "epoch20_blr1e-2_bsz32_ld1_wd0.05_dp0.1",
                "s5",
            ],
        }

    if key == "patchtst":
        return {
            "name": "PatchTST",
            "parts": [
                "capacity",
                "use_volt_current_soc_with_noise",
                "PatchTST",
                "epoch20_blr5e-3_bsz32_ld1_wd0.05_dp0.1",
                "s5",
            ],
        }

    if key in ["timemixer", "timemixer++", "timemixerpp", "timemixerplusplus"]:
        return {
            "name": "TimeMixer++",
            "parts": [
                "capacity",
                "use_volt_current_soc_with_noise",
                "TimeMixer++",
                "epoch20_blr1e-2_bsz32_ld1_wd0.05_dp0.1",
                "s5",
            ],
        }

    raise ValueError("Unknown model: {}".format(model))


def get_model_configs(model):
    if model.lower() == "all":
        return [
            get_model_config("FMAE"),
            get_model_config("iTransformer"),
            get_model_config("PatchTST"),
            get_model_config("TimeMixer++"),
        ]
    return [get_model_config(model)]


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


def capacity_scale(rmse, brand_num):
    return rmse * 100.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_root", type=str, default="logs")
    parser.add_argument("--model", type=str, default="FMAE",
                        help="FMAE, iTransformer, PatchTST, TimeMixer++, or all.")
    args = parser.parse_args()

    log_root = get_log_root(args.log_root)
    summary = []

    for config in get_model_configs(args.model):
        print("model: {}".format(config["name"]))
        cell_capacity_path = os.path.join(log_root, *config["parts"])
        average = print_regression_result(
            "cell-level capacity",
            cell_capacity_path,
            [10, 11, 12, 13],
            ["MIT1", "THU", "MIT2", "KIT"],
            "Min test set RMSE:",
            capacity_scale,
        )
        summary.append((config["name"], average))

    print("\nSummary")
    for name, average in summary:
        print("{} cell-level capacity: {}".format(name, average))
