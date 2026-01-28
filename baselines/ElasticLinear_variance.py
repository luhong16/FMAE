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
    parser = argparse.ArgumentParser(description='ElasticLinear_variance')
    parser.add_argument('--Brand', type = int, default = 10)
    parser.add_argument('--Fold', type = int, default = 0)
    parser.add_argument('--Task', type = str, default = 'RUL')
    parser.add_argument('--Points', type = str, default = '128')
    parser.add_argument('--Predicting_log10', type = int, default = 1)
    args = parser.parse_args()
    return args 

def main():
    # init
    seed = 5
    args = set_args()
    if args.Predicting_log10: 
        results_path = f"./baselines/results/RUL_w_log"
        scores_path = f"./baselines/results/RUL_w_log/diff_models_losses"
        print(f"Evaluating RUL_w_log baseline variance brand_{args.Brand} fold_{args.Fold} points_{args.Points} ...")
    else:
        results_path = f"./baselines/results/RUL_wo_log"
        scores_path = f"./baselines/results/RUL_wo_log/diff_models_losses"
        print(f"Evaluating RUL_wo_log baseline variance brand_{args.Brand} fold_{args.Fold} points_{args.Points} ...")
        
    os.makedirs(results_path, exist_ok = True), os.makedirs(scores_path, exist_ok = True)
    # ElaticLinear
    l1_lambdas = [0] # searched l1 lambdas = [0, 0.2, 0.4, 0.6, 1.0]
    l2_lambdas = [0] # searched l2 lambdas = [0, 0.2, 0.4, 0.6, 1.0]
    lrs = np.array([0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01,0.009,0.008,0.007])
    print(torch.cuda.device_count())
    device = torch.device('cpu')
    # model_names
    print("Running on {}".format(device))
    models = []
    for l1 in l1_lambdas:
        for l2 in l2_lambdas:
            for lr in lrs:
                models.append(f"ElasticLinear_variance_l1_{l1}_l2_{l2}_lr_{lr}")
    print(models, len(models))
    # more inits
    train_data = torch.load(f'./baselines/results/features/{args.Task}_brand{args.Brand}_fold{args.Fold}_points{args.Points}_train_dataKept_1.0.pkl')
    vali_data = torch.load(f'./baselines/results/features/{args.Task}_brand{args.Brand}_fold{args.Fold}_points{args.Points}_vali_dataKept_1.0.pkl')
    print(train_data[0].shape, train_data[1].shape, vali_data[0].shape, vali_data[1].shape)
    X_max, X_min = np.max(train_data[0], axis = 0), np.min(train_data[0], axis = 0)
    # print("Before Norm:", train_data[0])
    # print(X_max, X_min)
    X_train, X_vali = min_max_norm(value = train_data[0], min = X_min, max = X_max), min_max_norm(value = vali_data[0], min = X_min, max = X_max)
    # whether to apply log10 on y
    if args.Predicting_log10: y_train, y_vali = np.log10(train_data[1][:, 2]), np.log10(vali_data[1][:, 2])
    else: y_train, y_vali = train_data[1][:, 2], vali_data[1][:, 2]
    # print("after Norm:", X_train)
    X_train, X_vali = X_train[:, 0].reshape(-1, 1), X_vali[:, 0].reshape(-1, 1)
    print("data stats:")
    print('X')
    print(X_train, X_vali)
    print(X_train.shape, X_vali.shape)
    print('y')
    print(y_train, y_vali)
    # exit()
    # exit()
    if not args.Predicting_log10:
        losses = []
        y_preds = []
        GTs = []
        y_trains = []
        for l1 in l1_lambdas:
            for l2 in l2_lambdas:
                for lr in lrs:
                    net = ElasticLinearRegression(d_in = X_train.shape[1], d_out = 1, l1_lambda = l1, l2_lambda = l2).to(device)
                    my_trainer = Trainer_Basic(net = net, epochs = 10000, lr = lr, bs = X_train.shape[0]  // 16, device = device, train_data = (X_train, y_train), vali_data = (X_vali, y_vali))
                    best_vali_loss, best_records = my_trainer.train()
                    vali_loss, GT, y_pred = my_trainer.pred()
                    output = f"{args.Task}_brand{args.Brand}_fold{args.Fold}_points{args.Points}_ElasticLinear_variance_l1_{l1}_l2_{l2}_lr_{lr} | " + best_records
                    print(output)
                    losses.append(vali_loss)
                    y_preds.append(y_pred)
                    GTs.append(GT)
                    y_trains.append(y_train)
        torch.save((models, losses, GTs, y_preds, y_trains), os.path.join(scores_path, f'Brand{args.Brand}_Fold{args.Fold}_points{args.Points}_ElasticLinear_variance_wo_log_vali_losses.pkl'))
        print(models, len(models))
        print(losses, len(losses))
        print([(GTs[idx], y_preds[idx]) for idx in range(len(GTs))])
    else:
        losses = []
        y_preds = []
        GTs = []
        y_trains = []
        for l1 in l1_lambdas:
            for l2 in l2_lambdas:
                for lr in lrs:
                    net = ElasticLinearRegression(d_in = X_train.shape[1], d_out = 1, l1_lambda = l1, l2_lambda = l2).to(device)
                    my_trainer = Trainer_Basic(net = net, epochs = 10000, lr = lr, bs = X_train.shape[0]  // 16, device = device, train_data = (X_train, y_train), vali_data = (X_vali, y_vali))
                    best_vali_loss, best_records = my_trainer.train()
                    vali_loss, GT, y_pred = my_trainer.pred_log10()
                    output = f"{args.Task}_brand{args.Brand}_fold{args.Fold}_points{args.Points}_ElasticLinear_variance_l1_{l1}_l2_{l2}_lr_{lr} | " + best_records
                    print(f'{output}, rmse_vali_cycles: {vali_loss}')
                    losses.append(vali_loss)
                    y_preds.append(y_pred)
                    GTs.append(GT)
                    y_trains.append(y_train)
        torch.save((models, losses, GTs, y_preds, y_trains), os.path.join(scores_path, f'Brand{args.Brand}_Fold{args.Fold}_points{args.Points}_ElasticLinear_variance_w_log_vali_losses.pkl'))
        print(models, len(models))
        print(losses, len(losses))
        print([(GTs[idx], y_preds[idx]) for idx in range(len(GTs))])
        
    return 
if __name__ == '__main__':
    main()