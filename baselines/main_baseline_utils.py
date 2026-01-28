import numpy as np
import torch
import datetime
import os
import math
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import Dataset, DataLoader
import random
import pandas as pd

from scipy.interpolate import interp1d
from scipy.stats import skew
import scipy


# some utils
def write_logs(content, log_path = './Baseline_logs.txt'):
    with open(log_path, 'a') as f: # with open 'w'/'a', will create a new txt file
        f.write(content + '\n')
def MSE(pred, true):
    return np.mean((pred - true) ** 2)
def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))
def min_max_norm(value, min, max):
    return (value - min) / (max - min)
def min_max_denorm(value, min, max):
    return value*(max - min) + min
def create_path(path):
    if not os.path.exists(path): os.mkdir(path)

def calculate_roc_auc_scores(y_train_label):
    return

# classes 
class EarlyStopping:
    """
    Early Stopping
    Save best only based on validation loss
    ReduceLROnPlateau version
    """
    def __init__(self, patience, verbose = True, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, curr_lr, min_lr, path = None):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        # if self.verbose:
        #     # print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        #     print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). ')
        # # torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        # self.val_loss_min = val_loss
        return



class Custom_Dataset(Dataset):
    """Small Custom Dataset, can be directly loaded into RAM"""
    def __init__(self, dataset):
        """
        params:
            dataset: a tuple (X, y)
        """
        assert dataset[1].shape[0] > 0, f'label is empty' 
        self.dataset = dataset
        self.length = dataset[0].shape[0]
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        return self.dataset[0][idx], self.dataset[1][idx]
    def get_length(self):
        return self.length

class ElasticLinearRegression(nn.Module):
    """
    with Elastic net regularization
    """
    def __init__(self, d_in, d_out, l1_lambda, l2_lambda):
        super().__init__()
        self.layer = nn.Linear(d_in, d_out)
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
    def forward(self, x):
        return self.layer(x)
    def l1_reg(self):
        l1_norm = self.layer.weight.abs().sum()
        return self.l1_lambda * l1_norm
    def l2_reg(self):
        l2_norm = self.layer.weight.pow(2).sum()
        return self.l2_lambda * l2_norm

class Trainer_Basic(object):
    def __init__(self, net, epochs, lr, bs, device, train_data, vali_data, is_Elastic = True, lr_patience =20, E_S_patience = 40):
        """
        params:
        """
        self.net = net
        self.num_epochs = epochs
        self.lr = lr
        self.bs = bs
        self.train_data = train_data
        self.vali_data = vali_data
        self.device = device
        self.is_Elastic = is_Elastic
        self.lr_patience = lr_patience
        self.E_S_patience = E_S_patience
        # torch.manual_seed(5)
        torch.manual_seed(7)
        
    def train(self, v_min=None, v_max=None):
        print("Start Training ...")
        self.train_dataloader = DataLoader(Custom_Dataset(dataset = self.train_data), batch_size = self.bs, shuffle = True)
        self.vali_dataloader = DataLoader(Custom_Dataset(dataset = self.vali_data), batch_size = self.bs)

        steps_per_epoch = math.ceil(len(self.train_dataloader))
        optimizer = AdamW(self.net.parameters(), lr = self.lr)
        early_stopping = EarlyStopping(patience = self.E_S_patience)
        criterion = nn.MSELoss()
        min_lr = self.lr / 10000
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer = optimizer, 
                                                mode = 'min',
                                                factor = 0.55,
                                                patience = self.lr_patience,
                                                min_lr = min_lr)
        best_records = ''
        best_vali_loss = np.inf
        for epoch in range(0, self.num_epochs):
            temp_losses = []
            self.net.train()
            for i, (X, y) in enumerate(self.train_dataloader):
                optimizer.zero_grad()
                X = X.float().to(self.device) # remember to cast X and y from np.double to torch.float for both train and validation
                y = y.float().to(self.device)
                outputs = self.net(X)
                if self.is_Elastic:
                    loss = torch.sqrt(criterion(outputs.view(-1), y)) + self.net.l1_reg() + self.net.l2_reg()
                else:
                    loss = torch.sqrt(criterion(outputs.view(-1), y))
                temp_losses.append(loss)
                # print("Epochs {}, Steps {}/{} | RMSELoss: {:.7f} | lr = {:.3e}".format(epoch + 1, i + 1, steps_per_epoch, loss, optimizer.param_groups[0]['lr']), end = "\r")
                loss.backward() # loss BP
                optimizer.step()
                # end for steps
            # train_loss = np.mean(temp_losses)
            train_loss = torch.stack(temp_losses).mean().item()
            if v_min is not None:
                vali_loss = self.vali(self.vali_dataloader, criterion)
            else:
                vali_loss = self.pred(v_min=v_min,v_max=v_max)[0]
            if vali_loss <= best_vali_loss: 
                best_records = f'train_loss: {train_loss} | vali_loss: {vali_loss}'
                best_vali_loss = vali_loss
            # print("Epochs {}/{} | Train_Loss: {:.7f} , Vali_Loss: {:.7f} | lr = {:.3e}"
            #     .format(epoch + 1, self.num_epochs, train_loss, vali_loss, optimizer.param_groups[0]['lr']))
            scheduler.step(metrics = vali_loss)
            early_stopping(val_loss = vali_loss, model = self.net, curr_lr = optimizer.param_groups[0]['lr'], min_lr = min_lr)
            if early_stopping.early_stop:
                # print('Early Stopping ...')
                break
            # end for epochs
        # print("{} Training Complete.".format(datetime.datetime.now()))
        return best_vali_loss, best_records
    def vali(self, dataloader, criterion):
        total_loss = []
        y_list = []
        y_pred_list = []
        self.net.eval()
        with torch.no_grad():
            for _, (X, y) in enumerate(dataloader):
                X = X.float().to(self.device)
                y = y.float().to(self.device)
                outputs= self.net(X)
                loss = torch.sqrt(criterion(outputs.view(-1), y))
                total_loss.append(loss)
                y_list.append(np.array([i for i in y.cpu().numpy()])), y_pred_list.append(np.array([i for i in outputs.cpu().numpy()]))
            # total_loss = np.average(total_loss)
            # total_loss =torch.stack(total_loss).mean().item()
            total_loss = RMSE(np.concatenate(y_list), np.concatenate(y_pred_list))
            print(total_loss)
        return total_loss
    def pred_log10(self):
        total_loss = []
        y_list = []
        y_pred_list = []
        self.net.eval()
        with torch.no_grad():
            for _, (X, y) in enumerate(self.vali_dataloader):
                X = X.float().to(self.device)
                y = y.float().to(self.device)
                outputs= self.net(X)
                loss = RMSE(np.array([10**i for i in outputs]), np.array([10**i for i in y]))
                total_loss.append(loss)
                y_list.append(np.array([10**i for i in y])), y_pred_list.append(np.array([10**i for i in outputs]))
            # total_loss = np.average(total_loss)
            # total_loss = np.mean(total_loss)

            total_loss = RMSE(np.concatenate(y_list), np.concatenate(y_pred_list))
            print(total_loss)
            # exit()
            
        return total_loss, np.concatenate(y_list), np.concatenate(y_pred_list)
    
    def pred(self, v_min=None, v_max=None):
        total_loss = []
        y_list = []
        y_pred_list = []
        self.net.eval()
        with torch.no_grad():
            for _, (X, y) in enumerate(self.vali_dataloader):
                X = X.float().to(self.device)
                y = y.float().to(self.device)
                outputs= self.net(X)
                if v_min == None and v_max == None:
                    loss = RMSE(np.array(outputs.cpu()), np.array(y.cpu()))
                    y_list.append(np.array(y.cpu()))
                    y_pred_list.append(np.array(outputs.cpu()))
                else:
                    loss = RMSE(np.array(min_max_denorm(outputs.cpu().numpy(), v_min, v_max)), np.array(min_max_denorm(y.cpu().numpy(), v_min, v_max)))
                    y_list.append(np.array(min_max_denorm(y.cpu().numpy(), v_min, v_max)))
                    y_pred_list.append(np.array(min_max_denorm(outputs.cpu().numpy(), v_min, v_max)))
                total_loss.append(loss)
                
            # total_loss = np.average(total_loss)
            total_loss = np.mean(total_loss)
        return total_loss, np.concatenate(y_list), np.concatenate(y_pred_list)


def preprocess(downstream, brand_num, fold_num, data_type, train, results_path, points = 128):
    """
    params:
    points: int; used only when downstream is RUL, num of datapoints selected for feature extraction
    """
    seed = 5
    random.seed(seed)
    np.random.seed(seed)
    features_path = os.path.join(results_path, 'features')
    create_path(features_path)
    
    if data_type == 'car':
        if downstream == 'anomaly':
            car_dict_dir = 'five_fold_utils_six_brand_all'
            train_percent = 0.2
        elif downstream == 'capacity':
            car_dict_dir = 'five_fold_utils_EV_capacity'
            train_percent = 0.05
        else:
            raise NotImplementedError
    elif data_type == 'lab':
        if downstream == 'capacity': 
            car_dict_dir = 'five_fold_utils_lab_capacity'
            train_percent = 0.05
        elif downstream == 'RUL': 
            # car_dict_dir = 'five_fold_utils_RUL'
            car_dict_dir = 'five_fold_utils_six_brand_RUL_NE'
            train_percent = 1.0
        elif downstream == 'IR': 
            car_dict_dir = 'five_fold_utils_IR'
            train_percent = 0.2 # 0.2
        elif downstream == 'NC_capacity': 
            car_dict_dir = 'five_fold_utils_nc_relaxation_capacity'
            train_percent = 1.0
        else: raise NotImplementedError
    elif data_type == 'storage':
        if downstream == 'capacity': 
            car_dict_dir = 'five_fold_utils_storage_capacity'
            train_percent = 0.05
        else: raise NotImplementedError
    else: raise NotImplementedError

    # checking whether dataset already exists
    if downstream != 'RUL':
        if train: dataset_name = f"{downstream}_brand{brand_num}_fold{fold_num}_train_dataKept_{train_percent}.pkl"
        else: dataset_name = f"{downstream}_brand{brand_num}_fold{fold_num}_vali_dataKept_1.0.pkl"
    else:
        if train: dataset_name = f"{downstream}_brand{brand_num}_fold{fold_num}_points{points}_train_dataKept_{train_percent}.pkl"
        else: dataset_name = f"{downstream}_brand{brand_num}_fold{fold_num}_points{points}_vali_dataKept_1.0.pkl"

    if os.path.exists(os.path.join(features_path, dataset_name)): 
        print(f"{dataset_name} already exist ...") # 
        return 
    # loading & shuffle
    ind_ood_car_dict = np.load(f'./five_fold_utils/{car_dict_dir}/ind_odd_dict{brand_num}.npz.npy', allow_pickle=True).item()

    all_car_dict = np.load(f'./five_fold_utils/{car_dict_dir}/all_car_dict.npz.npy', allow_pickle=True).item()
    random.shuffle(ind_ood_car_dict['ind_sorted'])
    random.shuffle(ind_ood_car_dict['ood_sorted'])
    for each_num in ind_ood_car_dict['ind_sorted'] + ind_ood_car_dict['ood_sorted']:
        random.shuffle(all_car_dict[each_num])
    ind_car_num_list = ind_ood_car_dict['ind_sorted']
    ood_car_num_list = ind_ood_car_dict['ood_sorted']
    # print(ood_car_num_list, len(ood_car_num_list))

    if downstream != 'anomaly':
        if train:
            car_number = ind_car_num_list[:int(fold_num * len(ind_car_num_list) / 5)] \
                        + ind_car_num_list[int((fold_num + 1) * len(ind_car_num_list) / 5):] \
                        + ood_car_num_list[:int(fold_num * len(ood_car_num_list) / 5)] \
                        + ood_car_num_list[int((fold_num + 1) * len(ood_car_num_list) / 5):]
        else:
            car_number = ind_car_num_list[int(fold_num * len(ind_car_num_list) / 5):int(
                                        (fold_num + 1) * len(ind_car_num_list) / 5)] \
                                    + ood_car_num_list[int(fold_num * len(ood_car_num_list) / 5):int(
                                        (fold_num + 1) * len(ood_car_num_list) / 5)]
    else:
        if train:
            car_number = ind_car_num_list[:int(fold_num * len(ind_car_num_list) / 5)] \
                        + ind_car_num_list[int((fold_num + 1) * len(ind_car_num_list) / 5):] \
                        + ood_car_num_list[int(fold_num * len(ood_car_num_list) / 5):int((fold_num + 1) * len(ood_car_num_list) / 5)]
        else:
            car_number = ind_car_num_list[int(fold_num * len(ind_car_num_list) / 5):int((fold_num + 1) * len(ind_car_num_list) / 5)]\
                        + ood_car_num_list[:int(fold_num * len(ood_car_num_list) / 5)] \
                        + ood_car_num_list[int((fold_num + 1) * len(ood_car_num_list) / 5):]
    
    X = []
    y = []
    meta_data = []
    for each_num in car_number:
        if downstream in ['anomaly', 'capacity', 'IR', 'NC_capacity']:  
            if train and data_type != 'storage':
                print(f'Before Brand {each_num} total num of snippets: {len(all_car_dict[each_num])}')
                pkls = all_car_dict[each_num][:int(len(all_car_dict[each_num]) * train_percent)]
                print(f'After Brand {each_num} train num of snippets: {len(pkls)}')
            else:
                pkls = all_car_dict[each_num]
            # pkls = all_car_dict[each_num]
        else:
            pkls = all_car_dict[each_num]
            # print(pkls)
            # exit()
        if downstream in ['anomaly', 'capacity', 'IR']:
            for pkl_all in pkls:
                each_pkl = pkl_all[0]

                each_pkl = "./" + each_pkl
                # print(each_pkl)
                # exit()
                train1 = torch.load(each_pkl)
                features = train1[0]
                # print(train1[1])
                # exit()
                if data_type == 'lab':
                    if train1[1]['discharge_segment'] > -1 and downstream == 'IR': continue
                # print(train1[1])
                # effe_len = train1[1]['effective_length']
                # if effe_len != 128: raise ValueError
                if downstream == 'anomaly' or downstream == 'IR': label = train1[1]['label']
                elif downstream == 'capacity': label = train1[1]['capacity']
                X.append(features)
                y.append(label)
                meta_data.append(train1[1])
                # print(label)
                # train1[0] is snippet，train1[1] is metadata
                # end for
        elif downstream in ['RUL']: # RUL
            cycle10_snippets = []
            # cycle10_snippets_name = []
            cycle100_snippets = []
            # cycle100_snippets_name = []
            has_label_flag = False
            has_capacity2_flag = False
            disc_capacity_2         = -1 # feature
            disc_capacity_1_100_max = -1 # feature
            for pkl_all in pkls:
                each_pkl = pkl_all[0]
                each_pkl = "./" + each_pkl
                train1 = torch.load(each_pkl)
                
                features = train1[0]
                metadata = train1[1]
                effe_len = metadata['effective_length']
                qd = features[ :effe_len, 2]
                cycle_qd = qd[-1]
                if metadata['discharge_segment'] >-1 and cycle_qd >= disc_capacity_1_100_max: 
                    disc_capacity_1_100_max = cycle_qd
                # print(metadata)
                if metadata['discharge_segment'] == 2 and cycle_qd >= disc_capacity_2: 
                    disc_capacity_2 = cycle_qd
                    has_capacity2_flag = True
                elif metadata['discharge_segment'] == 10: 
                    cycle10_snippets.append((features[ :effe_len, 0], features[ :effe_len, 2], each_pkl)) # voltage and qd
                    # cycle10_snippets_name.append(each_pkl)
                elif metadata['discharge_segment'] == 100: 
                    cycle100_snippets.append((features[ :effe_len, 0], features[ :effe_len, 2], each_pkl)) # voltage and qd
                    # cycle100_snippets_name.append(each_pkl)
                    if not has_label_flag:
                        y.append(metadata['label'])
                        has_label_flag = True
                else: 
                    print('Charging snippet')
                    continue
            if len(cycle100_snippets) > 0 and len(cycle100_snippets) > 0 and has_label_flag and has_capacity2_flag:
                # max_volt = 3.5
                # min_volt = 2.0
                # sorting list by pkl name
                cycle100_snippets.sort(key = lambda x: x[2])  # ascending, use reverse=True for descending
                cycle10_snippets.sort(key = lambda x: x[2])
                # cycle 10
                cycle10_volt_arr = np.concatenate([cycle10_snippets[i][0] for i in range(len(cycle10_snippets))]).reshape(-1, 1)
                cycle10_qd_arr = np.concatenate([cycle10_snippets[i][1] for i in range(len(cycle10_snippets))]).reshape(-1, 1)
                # cyc10_start_idx = np.where(cycle10_volt_arr >= max_volt)[0][-1]
                # cyc10_end_idx = np.where(cycle10_volt_arr <= min_volt)[0][0]
                # print(cyc10_start_idx, cyc10_end_idx)
                cyc10_end_idx = np.argmax(cycle10_qd_arr)
                cyc10_start_idx = np.argmax(cycle10_volt_arr[: cyc10_end_idx])
                assert cyc10_start_idx < len(cycle10_volt_arr)
                if cyc10_start_idx > cyc10_end_idx:
                    y.pop()
                    continue
                # print(cyc10_start_idx, cyc10_end_idx)
                # print(cycle10_volt_arr[cyc10_start_idx: cyc10_end_idx], cycle10_qd_arr[cyc10_start_idx: cyc10_end_idx])
                # exit()
                # cycle 100
                cycle100_volt_arr = np.concatenate([cycle100_snippets[i][0] for i in range(len(cycle100_snippets))]).reshape(-1, 1)
                cycle100_qd_arr = np.concatenate([cycle100_snippets[i][1] for i in range(len(cycle100_snippets))]).reshape(-1, 1)
                # print(cycle100_snippets)
                # cyc100_start_idx = np.where(cycle100_volt_arr >= max_volt)[0][-1]
                # cyc100_end_idx = np.where(cycle100_volt_arr <= min_volt)[0][0]
                cyc100_end_idx = np.argmax(cycle100_qd_arr)
                cyc100_start_idx = np.argmax(cycle100_volt_arr[: cyc100_end_idx])
                # print(cycle100_qd_arr, cycle100_volt_arr)
                if cyc100_start_idx > cyc100_end_idx:
                    y.pop()
                    print(cycle100_volt_arr, cycle100_qd_arr)
                    exit()
                    continue
                print(cyc100_start_idx, cyc100_end_idx)
                # print(cycle100_volt_arr[cyc100_start_idx: cyc100_end_idx], cycle100_qd_arr[cyc100_start_idx: cyc100_end_idx])
                # print(np.where(cycle100_volt_arr >= max_volt)[0], np.where(cycle100_volt_arr <= min_volt)[0])
                assert cyc100_start_idx < len(cycle100_volt_arr)
                # convert to df
                cycle10_data = np.concatenate((cycle10_volt_arr[cyc10_start_idx: cyc10_end_idx],  cycle10_qd_arr[cyc10_start_idx: cyc10_end_idx]), axis = 1)
                cycle100_data = np.concatenate((cycle100_volt_arr[cyc100_start_idx: cyc100_end_idx],  cycle100_qd_arr[cyc100_start_idx: cyc100_end_idx]), axis = 1)
                # convert to df
                disc_cyc10_df = pd.DataFrame(data = cycle10_data, columns = ['V', 'Qd'])
                disc_cyc100_df = pd.DataFrame(data = cycle100_data, columns = ['V', 'Qd'])
                # print(disc_cyc10_df)
                # # print(disc_cyc100_df)
                # exit()
                print(disc_cyc10_df, disc_cyc100_df)
                # exit()
                statistical_features = extract_features_RUL(disc_cyc10_df, disc_cyc100_df, disc_capacity_2, disc_capacity_1_100_max, points)
                X.append(statistical_features)
            else:
                if has_label_flag: y.pop() 
                continue
        elif downstream in ['NC_capacity']:
            for pkl_all in pkls:
                each_pkl = pkl_all[0]
                train1 = torch.load(each_pkl)
                features = train1[0]
                # print(train1[1])
                # effe_len = train1[1]['effective_length']
                # if effe_len != 128: raise ValueError
                label = train1[1]['label']
                voltages1 = features[:,1][:14] # 14 voltage points stored in the raw csv files
                voltages2 = features[:,0] # interpolated voltage points
                # print(voltages2)
                # exit()
                # statistical features
                v1_skewness = scipy.stats.skew(voltages1)
                v1_max = np.max(voltages1)
                v1_variance = np.var(voltages1)
                v2_skewness = scipy.stats.skew(voltages2)
                v2_max = np.max(voltages2)
                v2_variance = np.var(voltages2)
                # exit()
                X.append([[v1_variance, v1_skewness, v1_max], [v2_variance, v2_skewness, v2_max]])
                y.append(label)
                meta_data.append(train1[1])
    assert len(X) == len(y), f'Size not matching, Found X {len(X)}, y{len(y)}'
        # end for each_num
    X = np.array(X)
    y = np.array(y)
    # print(X.shape, y.shape)
    # feature extraction 
    if downstream == 'capacity': 
        X = X[:, :, 2]
        # print(X)
        for k in range(0, X.shape[0]):
            if data_type =='lab': X[k] = X[k]/100 * y[k]/100 
            else: X[k] = X[k]/100 * y[k]
        # print(X,X.shape, y)
        # exit()
    elif downstream == 'anomaly': X = X
    elif downstream == 'RUL': X = X
    elif downstream == 'IR': X = X[:, :, :3]
    elif downstream == 'NC_capacity': X = X
    else:raise NotImplementedError
    
    

    print(X.shape, y.shape)
    print(y, X[:4])
    # exit()
    torch.save((X, y, meta_data), os.path.join(features_path, dataset_name), pickle_protocol = 4)
    print(f"{dataset_name} creation complete ...")

def extract_features_RUL(disc_cyc10_df, disc_cyc100_df, disc_capacity_2, disc_capacity_1_100_max, points):
    """
    reproduction
    https://static-content.springer.com/esm/art%3A10.1038%2Fs41560-019-0356-8/MediaObjects/41560_2019_356_MOESM1_ESM.pdf
    discharge curve
    "Variance" model: 
        delta Q_100-10 features: Variance
    """
    disc_cyc10_df = disc_cyc10_df[disc_cyc10_df['Qd']!= 0]
    disc_cyc100_df = disc_cyc100_df[disc_cyc100_df['Qd']!= 0]
    # print("100_df ",disc_cyc100_df)

    upper_V = disc_cyc10_df['V'].iloc[0] if disc_cyc10_df['V'].iloc[0] > disc_cyc100_df['V'].iloc[0] else disc_cyc100_df['V'].iloc[0]
    lower_V = disc_cyc10_df['V'].iloc[-1] if disc_cyc10_df['V'].iloc[-1] < disc_cyc100_df['V'].iloc[-1] else disc_cyc100_df['V'].iloc[-1]
    # upper_V = disc_cyc10_df['V'].iloc[0] 
    # lower_V = disc_cyc10_df['V'].iloc[-1] 
    # print(disc_cyc10_df)
    # exit()

    # upper_V = 3.5
    # lower_V = 2.0

    # # print(upper_V, lower_V)
    # print(disc_cyc10_df['V'].iloc[0], disc_cyc10_df['V'].iloc[-1])
    # print(disc_cyc100_df['V'].iloc[0], disc_cyc100_df['V'].iloc[-1])
    # print("=======")

    step_size = (upper_V - lower_V)/points
    selected_V = [5] + [upper_V - step_size *i  for i in range(points)] + [0]
    # print(selected_V)

    Q_100 = [np.nan if disc_cyc100_df[(disc_cyc100_df['V'] <= selected_V[i-1]) & (disc_cyc100_df['V'] >= selected_V[i+1])]['Qd'].empty else \
            disc_cyc100_df[(disc_cyc100_df['V'] <= selected_V[i-1]) & (disc_cyc100_df['V'] >= selected_V[i + 1])]['Qd'].mean(axis = 0) for i in range(1, points+1)]
    Q_10 = [np.nan if disc_cyc10_df[(disc_cyc10_df['V'] <= selected_V[i-1]) & (disc_cyc10_df['V'] >= selected_V[i + 1])]['Qd'].empty else \
            disc_cyc10_df[(disc_cyc10_df['V'] <= selected_V[i-1]) & (disc_cyc10_df['V'] >= selected_V[i + 1])]['Qd'].mean(axis = 0) for i in range(1, points+1)]
    # assert not np.isnan(Q_100[0]), f"Q_100[0] is nan, {Q_100}"
    # assert not np.isnan(Q_10[0]), f"Q_10[0] is nan, {Q_10}"
    # assert not np.isnan(Q_100[-1]), f"Q_100[-1] is nan, {Q_100}"
    # assert not np.isnan(Q_10[-1]), f"Q_10[-1] is nan, {Q_10}"
    # interplate
    Q_100_interplated_df = pd.DataFrame(data = Q_100)
    Q_10_interplated_df = pd.DataFrame(data = Q_10)
    # print(Q_100_interplated_df)
    Q_100_interplated_df.interpolate(method = 'linear',inplace = True,limit_area = 'inside', limit_direction = 'forward')
    # print(Q_100_interplated_df)
    # exit()
    Q_100_interplated_df.interpolate(method = 'linear',inplace = True, limit_direction = 'both')
    Q_10_interplated_df.interpolate(method = 'linear',inplace = True, limit_area = 'inside', limit_direction = 'forward')
    Q_10_interplated_df.interpolate(method = 'linear',inplace = True, limit_direction = 'both')
    # Q_100_interplated_df.fillna(value = 0, inplace = True)
    # Q_10_interplated_df.fillna(value = 0, inplace = True)
    # print(Q_100_interplated_df)
    # print(Q_10_interplated_df)
    # return

    Q_100_interplated = Q_100_interplated_df.to_numpy().reshape(-1)
    Q_10_interplated = Q_10_interplated_df.to_numpy().reshape(-1)
    # print("Q_100: ",Q_100_interplated)
    # print("Q-10: ", Q_10_interplated)
    if any(np.isnan(Q_100_interplated)) & any(np.isnan(Q_10_interplated_df)):
        print(Q_100_interplated)
        print(Q_10_interplated)
        exit()
    # statics
    delta_Q_V = [Q_100_interplated[i] - Q_10_interplated[i] for i in range(points)]
    delta_Q_V = [np.nan if q_v >= 0 or q_v < -1 else q_v for q_v in delta_Q_V]
    fixed_delta_Q_V_df = pd.DataFrame(delta_Q_V).interpolate(method = 'linear', limit_direction = 'both')
    delta_Q_V = fixed_delta_Q_V_df.to_numpy().reshape(-1)
    # delta_Q_V =  (disc_cyc100_df['Qd'] - disc_cyc10_df['Qd']).to_numpy().reshape(-1)
    # import matplotlib.pyplot as plt
    # # plt.plot(delta_Q_V, [3.5 - (3.5-2.0)/len(delta_Q_V) * i for i in range(len(delta_Q_V))])
    # plt.plot(delta_Q_V, selected_V[1:-1])
    # print(delta_Q_V) 
    bar_delta_Q_V = sum(delta_Q_V) / points
    # minimum
    minimum = np.log10(np.abs(np.min(delta_Q_V)))
    # variance
    variance = np.log10(np.abs(np.sum([(delta_Q_V[i] - bar_delta_Q_V)**2 for i in range(points)])* (1 / points - 1)))
    # skewness
    skewness_numerator = np.sum([(delta_Q_V[i] - bar_delta_Q_V)**3 for i in range(points)])* (1 / points)
    skewness_denominator = np.sqrt(np.sum([(delta_Q_V[i] - bar_delta_Q_V)**2 for i in range(points)]))**3
    skewness = np.log10(np.abs(skewness_numerator/ skewness_denominator))
    # kurtosis
    kurtosis_numerator = np.sum([(delta_Q_V[i] - bar_delta_Q_V)**4 for i in range(points)])* (1 / points)
    kurtosis_denominator = (np.sum([(delta_Q_V[i] - bar_delta_Q_V)**2 for i in range(points)]) * (1 / points))**2
    kurtosis = np.log10(np.abs(kurtosis_numerator / kurtosis_denominator))
    # max_discharge_capacity - discharge_capacity 2
    delta_capacity = disc_capacity_1_100_max -  disc_capacity_2
    # print([variance, minimum, skewness, kurtosis, disc_capacity_2, delta_capacity])

    return [variance, minimum, skewness, kurtosis, disc_capacity_2, delta_capacity]



class Data:
    def __init__(self):
        self.time = 0
        self.current = 0
        self.voltage = []
        self.temperature = []
        self.SOC = 0

    def reset(self):
        self.delt = []
        self.record = []
        for i in range(self.voltage.shape[1]):
            self.count_num = self.voltage[:, i].tolist().count(0)
            if 1 <= self.count_num <= self.voltage.shape[0] * 0.2:
                for j in range(self.voltage.shape[0]):
                    if self.voltage[j, i] == 0:
                        self.delt.append(j)
            elif self.voltage.shape[0] * 0.2 <= self.count_num:
                self.record.append(i)
        self.delt = sorted(list(set(self.delt)))
        self.record = sorted(list(set(self.record)))
        # handling data with 0 values
        self.voltage = np.delete(self.voltage, self.delt, axis=0)
        self.time = np.delete(self.time, self.delt, axis=0)
        self.current = np.delete(self.current, self.delt, axis=0)
        self.SOC = np.delete(self.SOC, self.delt, axis=0)


# Consistency score
class Score:
    def __init__(self):
        self.score_V = 0
        self.score_T = 0
        self.score_R = 0
        self.score_Q = 0
        self.score_E = 0

        self.consistency_list_resistance = 0
        self.consistency_list_energy = 0
        self.consistency_list_capacity = 0

        self.final_score = 0


class Calculate:
    # temp consistency calculation
    def tConsistency(self, battery):
        temperature = battery.temperature
        if temperature.any() == 0:
            score_T = -1
        else:
            MeanT = np.mean(temperature, axis=1)  # Mean temperature curve
            MM_T = np.mean(MeanT)  # Mean of mean temperature curve
            CellTgroup = temperature  # Temperature array
            max_T = np.max(CellTgroup, axis=1)  # Maximum temperature curve
            min_T = np.min(CellTgroup, axis=1)  # Minimum temperature curve
            Tsum = 0
            for i in range(len(CellTgroup)):
                Tsum += np.power(max_T[i] - min_T[i], 2)

            S_T = np.sqrt(np.true_divide(Tsum, len(MeanT)))   # Root mean square error between maximum temperature curve and minimum temperature curve
            result_T = S_T
            Tincon_0 = 10  # Reference value for temperature consistency score of 0
            Tincon_100 = 0  # Reference value for temperature consistency score of 100
            tT = Tincon_0 - Tincon_100
            score_T = 0
            if result_T <= Tincon_100:
                score_T = 100
            else:
                if Tincon_100 < result_T <= 0.25 * tT + Tincon_100:
                    score_T = interp1d([Tincon_100, 0.25 * tT + Tincon_100], [100, 90])(result_T)
                else:
                    if 0.25 * tT + Tincon_100 < result_T <= 0.5 * tT + Tincon_100:
                        score_T = interp1d([0.25 * tT + Tincon_100, 0.5 * tT + Tincon_100], [90, 70])(result_T)
                    else:
                        if 0.5 * tT + Tincon_100 < result_T <= 0.75 * tT + Tincon_100:
                            score_T = interp1d([0.5 * tT + Tincon_100, 0.75 * tT + Tincon_100], [70, 40])(result_T)
                        else:
                            if 0.75 * tT + Tincon_100 < result_T <= Tincon_0:
                                score_T = interp1d([0.75 * tT + Tincon_100, Tincon_0], [40, 0])(result_T)
                            else:
                                if result_T > Tincon_0:
                                    score_T = 0
                                else:
                                    score_T = None
        return score_T

    # Internal resistance consistency

    def rConsistency(self, battery):
        Voltage = battery.voltage
        Current = battery.current
        SOC = battery.SOC
        if len(battery.record) == len(Voltage[0]):
            score_R = -1
            ResCell = -1
            print('rConsistency Error ...')
        else:
            ResCell = np.zeros(shape=(len(Voltage[0])))
            change = np.zeros(shape=(Voltage.shape[0]))
            for j in range(Voltage.shape[0] - 1):
                change[j] = abs(Current[j + 1] - Current[j])
            changeID = np.array(np.where(change > 10)[0]) # Current difference greater than 10
            # print(changeID)
            if len(changeID) >= 1 and (((SOC[changeID] >= 20).any())):
                if (changeID[0] == 0) and (SOC[0] >= 20):
                    CurrentchangeID = changeID[0]
                    for i in range(len(battery.voltage[0])):  # Internal resistance calculation: voltage difference at current switching point divided by current difference
                        ResCell[i] = abs((Voltage[CurrentchangeID + 2, i] - Voltage[CurrentchangeID, i]) / (
                                Current[CurrentchangeID + 2] - Current[CurrentchangeID])) * 1000
                else:
                    battery.R = []
                    ResCell2 = np.zeros((len(changeID), len(Voltage[0])))
                    for j in range(len(changeID)):
                        if (90 >= SOC[changeID[j]] >= 20):
                            CurrentchangeID = changeID[j]
                            for i in range(len(Voltage[0])):  # Internal resistance calculation: voltage difference at current switching point divided by current difference
                                ResCell2[j, i] = abs(
                                    (Voltage[CurrentchangeID + 1, i] - Voltage[CurrentchangeID - 1, i]) / (
                                            Current[CurrentchangeID + 1] - Current[CurrentchangeID])) * 1000
                    for i in range(ResCell2.shape[0]):
                        if ((np.array(ResCell2[i, :]) <= 10).all()) and ((np.array(ResCell2[i, :]) >= 0.4).any()):
                            battery.R.append(i)
                    if len(battery.R) == 0:
                        ResCell = None
                    else:
                        ResCell2 = ResCell2[battery.R, :]
                        ResCell = np.mean(ResCell2, axis=0)
                if ResCell is None:
                    score_R = -1
                    ResCell = 0
                else:
                    # print(battery.record, ResCell)
                    ResCell1 = np.delete(ResCell, battery.record)
                    Raverage = np.mean(ResCell1)  
                    # print(Raverage)
                    Rsum = 0
                    for i in range(len(ResCell1)):
                        Rsum = Rsum + np.power(ResCell1[i] - Raverage, 2)
                    S_R = np.sqrt(Rsum / len(ResCell1))  # Average internal resistance
                    result_R = S_R / Raverage  # Coefficient of variation of internal resistance standard deviation
                    Rincon_0 = 0.5  # Reference value for internal resistance consistency score of 0
                    Rincon_100 = 0  # Reference value for internal resistance consistency score of 100

                    tr = Rincon_0 - Rincon_100
                    # print(result_R)

                    if result_R <= Rincon_100:
                        score_R = 100
                    else:
                        if Rincon_100 < result_R <= 0.25 * tr + Rincon_100:
                            score_R = interp1d([Rincon_100, 0.25 * tr + Rincon_100], [100, 90])(result_R)
                        else:
                            if 0.25 * tr + Rincon_100 < result_R <= 0.5 * tr + Rincon_100:
                                score_R = interp1d([0.25 * tr + Rincon_100, 0.5 * tr + Rincon_100], [90, 70])(
                                    result_R)
                            else:
                                if 0.5 * tr + Rincon_100 < result_R <= 0.75 * tr + Rincon_100:
                                    score_R = interp1d([0.5 * tr + Rincon_100, 0.75 * tr + Rincon_100], [70, 40])(
                                        result_R)
                                else:
                                    if 0.75 * tr + Rincon_100 < result_R <= Rincon_0:
                                        score_R = interp1d([0.75 * tr + Rincon_100, Rincon_0], [40, 0])(result_R)
                                    else:
                                        if result_R > Rincon_0:
                                            score_R = 0
                                        else:
                                            score_R = -1
                    for j in range(len(battery.record)):
                        ResCell[battery.record[j]] = 999999.9
            else:
                score_R = -1
                ResCell = 0
        return score_R, ResCell

    ##Capacity and Charge Consistency
    def qConsistency(self, battery):
        Voltage = battery.voltage
        # print(len(battery.record), len(Voltage[0]))
        if len(battery.record) == len(Voltage[0]):
            score_Q = -1
            score_E = -1
            CapCell = -1
            E = -1
        else:
            Time = battery.time
            Time = [int(i) for i in Time]
            SOC = battery.SOC
            Current = battery.current
            Current_pt = Current
            battery.RC = []

            Time_pf = [i for i in range(int(Time[0]), int(Time[(-1)]) + 1)]  # Generate expanded time list from start to end
            ChrgID = [i for i in range(len(Time_pf))]  # Generate index
            Current_itp = interp1d(Time, Current_pt, kind='nearest')(Time_pf)  # Interpolate time and current
            AHint = 0
            for j in range(len(ChrgID) - 1):
                AHint += Current_itp[ChrgID[j]] * (Time_pf[ChrgID[(j + 1)]] - Time_pf[ChrgID[j]]) / 3600
            SOC_itp = interp1d(Time, SOC, kind='nearest')(Time_pf)  # Interpolate time and SOC
            dSOC = (SOC_itp[ChrgID[(-1)]] - SOC_itp[ChrgID[0]]) / 100  # SOC difference
            if battery.temperature.any() == 0:
                Capstd = AHint / dSOC
            else:
                Capstd = AHint / dSOC
                TemPre = np.mean(np.mean(battery.temperature))  # Mean temperature
                Capstd = Capstd * (1 - 0.02 * (TemPre - 25) / 10)  # Temperature correction
            # we have valid capstd here
            # handling cells with majority zeros in voltage and abnormal voltage
            # print(battery.voltage, battery.voltage.shape)
            battery.voltage = np.delete(battery.voltage, battery.record, axis=1)
            for i in range(len(battery.voltage[0])):
                if (max(battery.voltage[0, :]) > battery.voltage[-1, i]) or (
                        min(battery.voltage[-1, :]) < battery.voltage[100, i]):
                    ID = np.where(battery.voltage[0, :] == max(battery.voltage[0, :]))[0][0]
                    battery.RC.append(ID)
            battery.RC = sorted(list(set(battery.RC)))
            # print(battery.RC)
            if battery.voltage.shape[1] > 1: battery.voltage = np.delete(battery.voltage, battery.RC, axis=1)
            # print(battery.voltage)
            ChrgIDend = ChrgID[(-1)]  # End of charge point
            Voltage_pf = np.zeros(shape=(len(ChrgID), len(battery.voltage[0])))
            for i in range(len(battery.voltage[0])):
                f = interp1d(Time, battery.voltage[:, i], kind='slinear')
                Voltage_pf[:, i] = f(Time_pf)
            Voltage_pf = [list(i) for i in Voltage_pf]
            Voltage_pf = np.array(Voltage_pf)

            ChrgIDstr = ChrgID[0]  # Start of charge point
            minV = np.min(Voltage_pf[ChrgIDstr])  # Find minimum voltage
            minVID = np.where(Voltage_pf[ChrgIDstr] == minV)[0][0]  # Find the cell position corresponding to the minimum voltage value
            RDCID = np.zeros(shape=(len(battery.voltage[0])))  # Generate zero set
            RDC = np.zeros(shape=(len(battery.voltage[0])))  # Generate zero set
            for j in range(len(battery.voltage[0])):
                lim = np.where(Voltage_pf[:, minVID] >= Voltage_pf[0, j])[0]
                RDCID[j] = lim[(0)]
                RDC[j] = Current_itp[ChrgID[int(np.floor(RDCID[j]))]] * (
                        Time_pf[ChrgID[int(np.ceil(RDCID[j]))]] - Time_pf[ChrgID[int(np.floor(RDCID[j]))]]) * (
                                 RDCID[j] - int(np.floor(RDCID[j]))) / 3600
                for k in range(1, int(np.floor(RDCID[j])) - 1):  # Accumulate RDC
                    RDC[j] = RDC[j] + Current_itp[ChrgID[k]] * (
                            Time_pf[ChrgID[(k + 1)]] - Time_pf[ChrgID[k]]) / 3600
            maxV = np.max(Voltage_pf[ChrgIDend])  # Find maximum single cell voltage
            maxVID = np.where(Voltage_pf[ChrgIDend] == maxV)[0][0]  # Find cell with maximum single cell voltage
            RCCID = np.zeros(shape=(len(battery.voltage[0])))  # Create empty set for storing
            RCC = np.zeros(shape=(len(battery.voltage[0])))  # Create empty set for storing
            for j in range(len(battery.voltage[0])):  # Interpolate to find RCC
                lim = np.where(Voltage_pf[:, maxVID] <= Voltage_pf[-1, j])[0]
                RCCID[j] = lim[(-1)]
                RCC[j] = Current_itp[ChrgID[int(np.floor(RCCID[j]))]] * (Time_pf[ChrgID[int(np.ceil(RCCID[j]))]] -
                                                                         Time_pf[ChrgID[int(np.floor(RCCID[j]))]]) * (
                                 int(np.ceil(RCCID[j])) -
                                 RCCID[j]) / 3600
                for k in range(len(ChrgID) - int(np.ceil(RCCID[j])) - 1):  # Accumulate RCC
                    RCC[j] = RCC[j] + Current_itp[ChrgID[(int(np.ceil(RCCID[j])) + k)]] * (
                            Time_pf[ChrgID[(int(np.ceil(RCCID[j])) + k + 1)]] - Time_pf[
                        ChrgID[(int(np.ceil(RCCID[j])) + k)]]) / 3600
            CapCell = Capstd + RCC + RDC  # Cell capacity calculation
            Qaverage = np.mean(CapCell)  # Average single-cell capacity
            Qsum = 0
            for i in range(len(CapCell)):
                Qsum = Qsum + np.power(CapCell[i] - Qaverage, 2)

            S_Q = np.sqrt(Qsum / len(CapCell))  # Standard deviation of capacity
            result_Q1 = S_Q / Qaverage  # Coefficient of variation of capacity standard deviation
            result_Q2 = (np.max(CapCell) - np.min(CapCell)) / Qaverage  # Coefficient of variation of capacity range
            Qincon_0 = 0.05  # Threshold value for actual capacity consistency score of 0
            Qincon_100 = 0  # Threshold for actual capacity consistency score of 100
            tQ = Qincon_0 - Qincon_100
            score_Q = 0
            if result_Q1 <= Qincon_100:
                score_Q = 100
            elif result_Q1 > Qincon_100 and result_Q1 <= 0.25 * tQ + Qincon_100:
                score_Q = interp1d([Qincon_100, 0.25 * tQ + Qincon_100], [100, 90])(result_Q1)
            elif result_Q1 > 0.25 * tQ + Qincon_100 and result_Q1 <= 0.5 * tQ + Qincon_100:
                score_Q = interp1d([0.25 * tQ + Qincon_100, 0.5 * tQ + Qincon_100], [90, 70])(result_Q1)
            elif result_Q1 > 0.5 * tQ + Qincon_100 and result_Q1 <= 0.75 * tQ + Qincon_100:
                score_Q = interp1d([0.5 * tQ + Qincon_100, 0.75 * tQ + Qincon_100], [70, 40])(result_Q1)
            elif result_Q1 > 0.75 * tQ + Qincon_100 and result_Q1 <= Qincon_0:
                score_Q = interp1d([0.75 * tQ + Qincon_100, Qincon_0], [40, 0])(result_Q1)
            else:
                if result_Q1 > Qincon_0:
                    score_Q = 0
            kQ = 1
            if result_Q2 <= 0:
                kQ = 1
            elif result_Q2 > 0 and result_Q2 <= 0.25 * tQ:
                kQ = interp1d([0, 0.25 * tQ], [1, 0.9])(result_Q2)
            elif result_Q2 > 0.25 * tQ and result_Q2 <= 0.5 * tQ:
                kQ = interp1d([0.25 * tQ, 0.5 * tQ], [0.9, 0.7])(result_Q2)
            elif result_Q2 > 0.5 * tQ and result_Q2 <= 0.75 * tQ:
                kQ = interp1d([0.5 * tQ, 0.75 * tQ], [0.7, 0.4])(result_Q2)
            elif result_Q2 > 0.75 * tQ and result_Q2 <= tQ:
                kQ = interp1d([0.75 * tQ, tQ], [0.4, 0])(result_Q2)
            else:
                if result_Q2 > tQ:
                    kQ = 0
            E = Capstd + RDC  # Charge consistency
            Eaverage = np.mean(E)  # Mean charge
            Esum = 0
            for i in range(len(E)):
                Esum = Esum + np.power(E[i] - Eaverage, 2)

            S_E = np.sqrt(Esum / len(E))  # Charge standard deviation
            result_E1 = S_E / Eaverage  # Coefficient of variation of standard deviation
            result_E2 = (np.max(E) - np.min(E)) / Eaverage  # Coefficient of variation of range
            Eincon_0 = 0.05  # Threshold value for actual charge consistency score of 0
            Eincon_100 = 0  # Threshold for actual charge consistency score of 100
            tE = Eincon_0 - Eincon_100
            if result_E1 <= Eincon_100:
                score_E = 100
            elif result_E1 > Eincon_100 and result_E1 <= 0.25 * tE + Eincon_100:
                score_E = interp1d([Eincon_100, 0.25 * tE + Eincon_100], [100, 90])(result_E1)
            elif result_E1 > 0.25 * tE + Eincon_100 and result_E1 <= 0.5 * tE + Eincon_100:
                score_E = interp1d([0.25 * tE + Eincon_100, 0.5 * tE + Eincon_100], [90, 70])(result_E1)
            elif result_E1 > 0.5 * tE + Eincon_100 and result_E1 <= 0.75 * tE + Eincon_100:
                score_E = interp1d([0.5 * tE + Eincon_100, 0.75 * tE + Eincon_100], [70, 40])(result_E1)
            elif result_E1 > 0.75 * tE + Eincon_100 and result_E1 <= Eincon_0:
                score_E = interp1d([0.75 * tE + Eincon_100, Eincon_0], [40, 0])(result_E1)
            else:
                # if result_E1 > Eincon_0:
                #     score_E = 0
                score_E = 0
            if result_E2 <= 0:
                kE = 1
            elif result_E2 > 0 and result_E2 <= 0.25 * tE:
                kE = interp1d([0, 0.25 * tE], [1, 0.9])(result_E2)
            elif result_E2 > 0.25 * tE and result_E2 <= 0.5 * tE:
                kE = interp1d([0.25 * tE, 0.5 * tE], [0.9, 0.7])(result_E2)
            elif result_E2 > 0.5 * tE and result_E2 <= 0.75 * tE:
                kE = interp1d([0.5 * tE, 0.75 * tE], [0.7, 0.4])(result_E2)
            elif result_E2 > 0.75 * tE and result_E2 <= tE:
                kE = interp1d([0.75 * tE, tE], [0.4, 0])(result_E2)
            else:
                # if result_E2 > tE:
                #     kE = 0
                kE = 0

            score_E *= kE
            score_Q *= kQ
            battery.record = sorted(list(set(battery.record + battery.RC)))
            for c in range(len(battery.record)):
                CapCell = np.insert(CapCell, battery.record[c], 0, axis=0)
                E = np.insert(E, battery.record[c], 0, axis=0)
        return score_Q, score_E, CapCell, E

    # Voltage consistency
    def vConsistency(self, battery):
        voltage = battery.voltage  # Read previously processed voltage
        if len(battery.record) == len(voltage[0]):
            score_V = -1
        else:
            MeanV = np.mean(voltage, axis=1)  # Calculate mean voltage curve
            MM_V = np.mean(MeanV)  # Mean of mean voltage curve
            CellVgroup = voltage
            max_V = np.max(CellVgroup, axis=1)  # Calculate maximum voltage curve
            min_V = np.min(CellVgroup, axis=1)  # Calculate minimum voltage curve
            Vsum = 0
            for i in range(len(CellVgroup)):  # Accumulate squared difference between max and min voltage
                Vsum += np.power(max_V[i] - min_V[i], 2)

            S_V = np.sqrt(np.true_divide(Vsum, len(MeanV)))  # Range
            result_V = np.true_divide(S_V, MM_V)  # Coefficient of variation of range
            Vincon_0 = 0.025  # Reference value for voltage consistency score of 0
            Vincon_100 = 0  # Reference value for voltage consistency score of 100
            tV = Vincon_0 - Vincon_100
            score_V = 0
            if result_V <= Vincon_100:  # Evaluating
                score_V = 100
            else:
                if Vincon_100 < result_V <= 0.25 * tV + Vincon_100:
                    score_V = interp1d([Vincon_100, 0.25 * tV + Vincon_100], [100, 90])(result_V)
                else:
                    if 0.25 * tV + Vincon_100 < result_V <= 0.5 * tV + Vincon_100:
                        score_V = interp1d([0.25 * tV + Vincon_100, 0.5 * tV + Vincon_100], [90, 70])(result_V)
                    else:
                        if 0.5 * tV + Vincon_100 < result_V <= 0.75 * tV + Vincon_100:
                            score_V = interp1d([0.5 * tV + Vincon_100, 0.75 * tV + Vincon_100], [70, 40])(result_V)
                        else:
                            if 0.75 * tV + Vincon_100 < result_V <= Vincon_0:
                                score_V = interp1d([0.75 * tV + Vincon_100, Vincon_0], [40, 0])(result_V)
                            else:
                                if result_V > Vincon_0:
                                    score_V = 0
                                else:
                                    score_V = None
        return score_V


class Battery_Consistency(object):
    """
    Consistency calculation
    Calculate current charging segment

    time_loc: time column location (int)
    v_loc_str: starting column of voltage columns (int)
    v_loc_end: ending column of voltage columns (int)
    temp_loc_str: starting column of temperature columns (int)
    temp_loc_end: ending column of temperature columns (int)
    current_loc: current column location (int)
    soc_loc: SOC column location (int)
    mileage_loc: mileage column location (int)

    """

    def __init__(self, time_loc, v_loc_str, v_loc_end, temp_loc_str, temp_loc_end, current_loc, soc_loc, mileage_loc):
        """
        init
        """
        self.time_loc = time_loc
        self.v_loc_str = v_loc_str
        self.v_loc_end = v_loc_end + 1
        self.temp_loc_str = temp_loc_str
        self.temp_loc_end = temp_loc_end + 1
        self.current_loc = current_loc
        self.soc_loc = soc_loc
        self.mileage_loc = mileage_loc
        

    def loaddata(self, filepath): 
        """
        Loading data
        """
        data = filepath
        return (np.array(data), list(data.columns))

    def dataprocess(self, car_data, car_data_head):  
        """
        Data processing
        """
        data = Data()
        A = np.zeros(shape=(car_data.shape[0], car_data.shape[1] + 5))
        A[:, self.time_loc] = (car_data[:, self.time_loc] - min(car_data[:, self.time_loc]))  # time

        A[:, self.v_loc_str:self.v_loc_end] = car_data[:, self.v_loc_str:self.v_loc_end]  # cell's voltage
        A[:, self.temp_loc_str:self.temp_loc_end] = car_data[:, self.temp_loc_str:self.temp_loc_end]  # cell's temperature
        data.temperature = A[:, self.temp_loc_str:self.temp_loc_end].reshape(-1, 1) if self.temp_loc_end - self.temp_loc_str == 1 else A[:, self.temp_loc_str:self.temp_loc_end]
        A[:, self.current_loc] = -car_data[:, self.current_loc]  # cell's current
        A[:, self.soc_loc] = car_data[:, self.soc_loc]  # cell's SOC
        data.time = A[:, self.time_loc]
        data.current = A[:, self.current_loc]
        data.voltage = A[:, self.v_loc_str:self.v_loc_end].reshape(-1,1) if self.v_loc_end - self.v_loc_str == 1 else A[:, self.v_loc_str:self.v_loc_end]
        data.SOC = A[:, self.soc_loc]
        # Handling abonormal voltage
        data.delt = []
        data.record = []
        for i in range(data.voltage.shape[1]):
            data.count_num = data.voltage[:, i].tolist().count(0)
            if 1 <= data.count_num <= data.voltage.shape[0] * 0.2:
                for j in range(data.voltage.shape[0]):
                    if data.voltage[j, i] == 0:
                        data.delt.append(j)
            elif data.voltage.shape[0] * 0.2 <= data.count_num:
                data.record.append(i)
        data.delt = sorted(list(set(data.delt)))
        data.record = sorted(list(set(data.record)))
        # print(data.record)
        # Handling 0 values
        data.voltage = np.delete(data.voltage, data.delt, axis=0)
        data.time = np.delete(data.time, data.delt, axis=0)
        data.current = np.delete(data.current, data.delt, axis=0)
        data.SOC = np.delete(data.SOC, data.delt, axis=0)
        return data

    def scoreStatistics(self, battery, matrices = None):
        # print("in: ",weight_matrix_in)
        self.score = Score()
        calt = Calculate()
        self.score.score_T = calt.tConsistency(battery)
        self.score.score_R, self.score.consistency_list_resistance = calt.rConsistency(battery)
        self.score.score_Q, self.score.score_E, self.score.consistency_list_capacity, self.score.consistency_list_energy = calt.qConsistency(
            battery)

        self.score.score_V = calt.vConsistency(battery)

        score_matrix = [self.score.score_V, self.score.score_T, self.score.score_R, self.score.score_Q,
                        self.score.score_E]
        # print(score_matrix)
        if (np.array(score_matrix) == -1).all(): raise ValueError
        else:
            for i in range(len(score_matrix)):
                if score_matrix[i] == -1:
                    score_matrix[i] = 100
                else:
                    score_matrix[i] = score_matrix[i]
        weight_matrices = matrices
        if weight_matrices is None: raise NotImplementedError
        final_scores = []
        for weight_matrix in weight_matrices:
            final_score = np.sum(np.multiply(score_matrix, weight_matrix))
            final_scores.append(final_score)

        # return self.score
        # return self.score, self.score.score_T, self.score.score_R, self.score.score_Q, self.score.score_E, self.score.score_V, self.score.final_score, self.ResCell, self.CapCell, self.ECell
        return self.score, self.score.score_T, self.score.score_R, self.score.score_Q, self.score.score_E, self.score.score_V, final_scores
