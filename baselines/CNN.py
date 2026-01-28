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
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn

class baseline_Conv1D(nn.Module):
    def __init__(self):
        super(baseline_Conv1D, self).__init__()
        
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels = 3, out_channels = 24, kernel_size = 6),
                               nn.ReLU(),
                               nn.BatchNorm1d(num_features = 24))
        self.pool1 = nn.MaxPool1d(kernel_size = 2, stride = 2)
        self.conv2 = nn.Sequential(nn.Conv1d(in_channels = 24, out_channels = 32, kernel_size = 3),
                               nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 3),
                               nn.ReLU(),
                               nn.BatchNorm1d(num_features = 32))
        self.pool2 = nn.MaxPool1d(kernel_size = 2, stride = 2)
        self.conv4 = nn.Sequential(nn.Conv1d(in_channels = 32, out_channels = 64, kernel_size = 3),
                               nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 3),
                               nn.ReLU(),
                               nn.BatchNorm1d(num_features = 64))
        self.pool3 = nn.MaxPool1d(kernel_size = 2, stride = 2)
        self.conv6 = nn.Sequential(nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 3),
                               nn.ReLU())
        self.conv7 = nn.Sequential(nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 3),
                               nn.ReLU(),
                               nn.BatchNorm1d(num_features = 64))
        self.pool4 = nn.MaxPool1d(kernel_size = 2, stride = 2)
        self.dense = nn.Sequential(nn.Flatten(),
                               nn.Linear(in_features = 256, out_features = 64),
                               nn.Dropout(p = 0.4))
        self.proj_head = nn.Linear(in_features = 64, out_features = 1)


    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x = self.pool1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.conv3(x)
        # print(x.shape)
        x = self.pool2(x)
        # print(x.shape)
        x = self.conv4(x)
        # print(x.shape)
        x = self.conv5(x)
        # print(x.shape)
        x = self.pool3(x)
        # print(x.shape)
        x = self.conv6(x)
        # print(x.shape)
        x = self.conv7(x)
        # print(x.shape)
        x = self.pool4(x)
        # print(x.shape)
        x = self.dense(x)
        # print(x.shape)
        x = self.proj_head(x)
        # print(x.shape)
        return x





def set_args():
    parser = argparse.ArgumentParser(description='conv1d')
    parser.add_argument('--Brand', type = int, default = 10)
    # parser.add_argument('--Fold', type = int, default = 0)
    parser.add_argument('--Task', type = str, default = 'IR')
    args = parser.parse_args()
    return args 

def main():
    # init
    seed = 5
    args = set_args()
    results_path = f"./baselines/results/{args.Task}"
    scores_path = f"./baselines/results/{args.Task}/diff_models_losses"
    create_path(results_path), create_path(scores_path)
    print(f"Evaluating {args.Task} baseline brand_{args.Brand} ...")
    # conv1d
    lrs = [0.1] # searched lrs = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001]
    folds = [0, 1, 2, 3, 4]
    # GPU
    print(torch.cuda.device_count())
    # device = torch.device('cpu')
    device = torch.device('cuda:4')
    # model_names
    print("Running on {}".format(device))
    models = []
    for lr in lrs:
        models.append(f"conv1d_lr_{lr}")
    print(models, len(models))
    for fold in folds:
        # more inits 
        train_data = torch.load(f'./baselines/results/features/{args.Task}_brand{args.Brand}_fold{fold}_train_dataKept_0.2.pkl')
        vali_data = torch.load(f'./baselines/results/features/{args.Task}_brand{args.Brand}_fold{fold}_vali_dataKept_1.0.pkl')
        print(train_data[0].shape, train_data[1].shape, vali_data[0].shape, vali_data[1].shape)
    
        X_train = train_data[0].transpose(0, 2, 1)
        X_vali = vali_data[0].transpose(0, 2, 1)
        y_train, y_vali = train_data[1], vali_data[1]
        losses = []
        for lr in lrs:
            net = baseline_Conv1D().to(device)
            my_trainer = Trainer_Basic(net = net, epochs = 10000, lr = lr, bs = X_train.shape[0] // 256, device = device, 
                                        train_data = (X_train, y_train), vali_data = (X_vali, y_vali), is_Elastic = False, E_S_patience = 40, lr_patience = 15)
            vali_loss, best_records = my_trainer.train()
            # vali_loss, _, _ = my_trainer.pred(v_min=y_min, v_max=y_max)
            output = f"{args.Task}_brand{args.Brand}_fold{fold}_conv1d_lr_{lr} | " + best_records
            print(output)
            # print(output, f'\nIR rmse: {vali_loss}')
            losses.append(vali_loss)

        torch.save((models, losses), os.path.join(scores_path, f'Brand{args.Brand}_Fold{fold}_conv1d_vali_losses.pkl'))
        print(models, len(models))
        print(losses, len(losses))

        
    return 
if __name__ == '__main__':
    main()