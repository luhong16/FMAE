import random
import numpy as np
import torch
from main_baseline_utils import *
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as RFRegressor
from xgboost import XGBRegressor as xgb
from sklearn import metrics
from sklearn.metrics import auc
import os

def get_brand_num(downstream_task, data_type):
    if data_type == 'car':
        if downstream_task == 'anomaly' : return [1, 2, 4, 5, 6]
        elif downstream_task == 'capacity': return [1, 2, 3, 4, 5, 6]
        else: raise NotImplementedError
    elif data_type == 'lab':
        if downstream_task == 'capacity': return [10, 11, 12, 13]
        elif downstream_task == 'RUL': return [10, 12, 13]
        elif downstream_task == 'IR': return [10]
        elif downstream_task == 'NC_capacity': return [14]
        else: raise NotImplementedError
    elif data_type == 'storage':
        if downstream_task == 'capacity': return [7]
        else: raise NotImplementedError
    else: raise NotImplementedError

def create_features(downstream, data_type):
    brand_nums = get_brand_num(downstream_task = downstream, data_type = data_type)
    # print(brand_nums)
    points_list = [128, 256, 1280, 2560, 12800]
    num_folds = [0, 1, 2, 3, 4]
    results_path = f"./baselines/results"
    os.makedirs(results_path, exist_ok = True)
    write_logs("Starting downstream dataset creation ...", log_path = os.path.join(results_path, 'data_creation_logs.txt'))
    print('downstream:', downstream, '; data_type:', data_type, '; brands:', brand_nums)
    for brand_num in brand_nums:
        for fold in num_folds:
            if downstream != 'RUL':
                preprocess(downstream = downstream, brand_num = brand_num, fold_num = fold, 
                data_type = data_type, train = True, results_path = results_path)
                preprocess(downstream = downstream, brand_num = brand_num, fold_num = fold, 
                data_type = data_type, train = False, results_path = results_path)
                write_logs(f"{downstream}_brand{brand_num}_fold{fold}_train.pkl", log_path = os.path.join(results_path, 'data_creation_logs.txt'))
                write_logs(f"{downstream}_brand{brand_num}_fold{fold}_vali.pkl", log_path = os.path.join(results_path, 'data_creation_logs.txt'))
            else:
                for points in points_list:
                    print(f"{downstream}_brand{brand_num}_fold{fold}_points{points}")
                    preprocess(downstream = downstream, brand_num = brand_num, fold_num = fold, 
                    data_type = data_type, train = True, points = points, results_path = results_path)
                    preprocess(downstream = downstream, brand_num = brand_num, fold_num = fold, 
                    data_type = data_type, train = False, points = points, results_path = results_path)
                    write_logs(f"{downstream}_brand{brand_num}_fold{fold}_points{points}_train.pkl", log_path = os.path.join(results_path, 'data_creation_logs.txt'))
                    write_logs(f"{downstream}_brand{brand_num}_fold{fold}_points{points}_vali.pkl", log_path = os.path.join(results_path, 'data_creation_logs.txt'))
    return 


def main():
    create_features(downstream = 'anomaly', data_type = 'car')
    create_features(downstream = 'capacity', data_type = 'car')
    create_features(downstream = 'capacity', data_type = 'storage')
    create_features(downstream = 'capacity', data_type = 'lab')
    create_features(downstream = 'RUL', data_type = 'lab')
    create_features(downstream = 'IR', data_type = 'lab')
    create_features(downstream = 'NC_capacity', data_type = 'lab')
    

if __name__ == '__main__':
    main()