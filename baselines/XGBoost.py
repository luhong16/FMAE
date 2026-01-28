import random
import numpy as np
import torch
from main_baseline_utils import *
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as RFRegressor
from xgboost import XGBRegressor as xgb
from sklearn import metrics
from sklearn.metrics import auc
import argparse

def set_args():
    parser = argparse.ArgumentParser(description='XGBoost')
    parser.add_argument('--Brand', type = int, default = 3) # NC_capacity brand 14
    parser.add_argument('--Fold', type = int, default = 0)
    parser.add_argument('--Task', type = str, default = 'capacity') # NC_capacity
    parser.add_argument('--percent', type = float, default = 0.05) # 1.0 for NC_capacity
    parser.add_argument('--V_idx', type = float, default = 1)
    args = parser.parse_args()
    return args 

def main():
    # init
    seed = 5
    args = set_args()
    results_path = f"./baselines/results/{args.Task}"
    scores_path = f"./baselines/results/{args.Task}/diff_models_losses"
    os.makedirs(results_path, exist_ok = True), os.makedirs(scores_path, exist_ok = True)
    print(f"Evaluating {args.Task} baseline brand_{args.Brand} fold_{args.Fold} ...")
    # RF
    num_trees = [100] # searched num_trees = [6, 25, 50, 100, 250, 500, 1000]
    tree_depths = [9] # searched tree_depths = [3, 6, 7, 9, 11]
    etas = [0.2] # searched etas = [0.8, 0.5, 0.2]
    
    # GPU
    print(torch.cuda.device_count())
    device = torch.device('cpu')
    # model_names
    print("Running on {}".format(device))
    models = []
    for tree in num_trees:
        for depth in tree_depths:
            for eta in etas:
                models.append(f"XGBoost_t_{tree}_d_{depth}_eta_{eta}_train_{args.percent}")
    print(models, len(models))
    # more inits
    train_data = torch.load(f'./baselines/results/features/{args.Task}_brand{args.Brand}_fold{args.Fold}_train_dataKept_{args.percent}.pkl')
    vali_data = torch.load(f'./baselines/results/features/{args.Task}_brand{args.Brand}_fold{args.Fold}_vali_dataKept_1.0.pkl')
    
    print(train_data[0].shape, train_data[1].shape, vali_data[0].shape, vali_data[1].shape)
    if args.Brand != 14:
        X_max, X_min = np.max(train_data[0]), np.min(train_data[0])
        X_train, X_vali = min_max_norm(value = train_data[0], min = X_min, max = X_max), min_max_norm(value = vali_data[0], min = X_min, max = X_max)
        y_train, y_vali = train_data[1], vali_data[1]
    else:
        X_max, X_min = np.max(train_data[0][:, args.V_idx, :]), np.min(train_data[0][:, args.V_idx, :])
        X_train, X_vali = min_max_norm(value = train_data[0][:, args.V_idx, :], min = X_min, max = X_max), \
                            min_max_norm(value = vali_data[0][:, args.V_idx, :], min = X_min, max = X_max)
        # X_train, X_vali = train_data[0], vali_data[0]
        y_train, y_vali = train_data[1], vali_data[1]

    losses = []
    y_preds = []
    for tree in num_trees:
        for depth in tree_depths:
            for eta in etas:
                net = xgb(n_estimators = tree, max_depth = depth, eta = eta, random_state = seed).fit(X_train, y_train)
                train_loss, vali_loss= RMSE(net.predict(X_train), y_train), RMSE(net.predict(X_vali), y_vali)
                output = f"{args.Task}_brand{args.Brand}_fold{args.Fold}_XGB_trees{tree}_depth{depth}_eta{eta}_train_{args.percent} | train_loss: {train_loss} | vali_loss: {vali_loss}"
                print(output)
                losses.append(vali_loss)
                y_preds.append(net.predict(X_vali))
                # print(net.predict(X_vali))
    torch.save((models, losses, y_vali, y_preds), os.path.join(scores_path, f'Brand{args.Brand}_Fold{args.Fold}_train_{args.percent}_XGBoost_vali_losses.pkl'))
    print(models, len(models))
    print(losses, len(losses))
        
    return 
if __name__ == '__main__':
    main()