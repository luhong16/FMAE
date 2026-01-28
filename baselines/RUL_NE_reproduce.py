import numpy as np
import os
import torch
import pandas as pd
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import math
import matplotlib.pyplot as plt

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
    def __init__(self, net, epochs, lr, bs, device, train_data, vali_data,\
        test_data, is_Elastic = True, lr_patience =20, E_S_patience = 40):
        """
        params:
        """
        self.net = net
        self.num_epochs = epochs
        self.lr = lr
        self.bs = bs
        self.train_data = train_data
        self.vali_data = vali_data
        self.test_data = test_data

        self.device = device
        self.is_Elastic = is_Elastic
        self.lr_patience = lr_patience
        self.E_S_patience = E_S_patience
        # torch.manual_seed(5)
        torch.manual_seed(7)
        
    def train(self):
        print("Start Training ...")

        self.train_dataloader = DataLoader(Custom_Dataset(dataset = self.train_data), batch_size = self.bs, shuffle = True)
        self.vali_dataloader = DataLoader(Custom_Dataset(dataset = self.vali_data), batch_size = len(Custom_Dataset(dataset = self.vali_data)))
        self.test_dataloader = DataLoader(Custom_Dataset(dataset = self.test_data), batch_size = len(Custom_Dataset(dataset = self.test_data)))

        steps_per_epoch = math.ceil(len(self.train_dataloader))
        optimizer = AdamW(self.net.parameters(), lr = self.lr)
        early_stopping = EarlyStopping(patience = self.E_S_patience)
        criterion = nn.MSELoss()
        min_lr = self.lr / 1000
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer = optimizer, 
                                                mode = 'min',
                                                factor = 0.1,
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
            vali_loss = self.vali(self.vali_dataloader, criterion)
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
                y_list.append(np.array([i for i in y])), y_pred_list.append(np.array([i for i in outputs]))
            # total_loss = np.average(total_loss)
            # total_loss =torch.stack(total_loss).mean().item()
            total_loss = RMSE(np.concatenate(y_list), np.concatenate(y_pred_list))
            # print(total_loss)
        return total_loss
    def pred_log10_vali(self):
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
            # print(total_loss)
            # exit()
            
        return total_loss, np.concatenate(y_list), np.concatenate(y_pred_list)
    
    def pred_log10_test(self):
        total_loss = []
        y_list = []
        y_pred_list = []
        self.net.eval()
        with torch.no_grad():
            for _, (X, y) in enumerate(self.test_dataloader):
                X = X.float().to(self.device)
                y = y.float().to(self.device)
                outputs= self.net(X)
                loss = RMSE(np.array([10**i for i in outputs]), np.array([10**i for i in y]))
                total_loss.append(loss)
                y_list.append(np.array([10**i for i in y])), y_pred_list.append(np.array([10**i for i in outputs]))
            # total_loss = np.average(total_loss)
            # total_loss = np.mean(total_loss)

            total_loss = RMSE(np.concatenate(y_list), np.concatenate(y_pred_list))
            # print(total_loss)
            # exit()
            
        return total_loss, np.concatenate(y_list), np.concatenate(y_pred_list)

def MSE(pred, true):
    return np.mean((pred - true) ** 2)
def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))
def min_max_norm(value, min, max):
    return (value - min) / (max - min)

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
    print([variance, minimum, skewness, kurtosis, disc_capacity_2, delta_capacity])

    return [variance, minimum, skewness, kurtosis, disc_capacity_2, delta_capacity]

def snippets_to_features(path, dict_name, dataset_path, points = 1280):
    """41 43 40"""
    if os.path.exists(f'{dataset_path}/{dict_name}_dataset_points_{points}.pkl'):
        return
    snippets = os.listdir(path)

    if not os.path.exists(f'{dataset_path}/{dict_name}_car_dict.pkl'):
        car_dict = {}
        for snippet in snippets:
            features, metadata = torch.load(os.path.join(path, snippet))
            # print(features[:2], metadata)
            car = metadata['car']
            if car not in car_dict:
                car_dict[car] = []
            car_dict[car].append(snippet)

        torch.save(car_dict, f'{dataset_path}/{dict_name}_car_dict.pkl')
    
    car_dict = torch.load(f'{dataset_path}/{dict_name}_car_dict.pkl')
    print(len(car_dict))
    # exit()
    X = []
    y = []
    for car, car_snippets in car_dict.items():
        cycle10_snippets = []
        # cycle10_snippets_name = []
        cycle100_snippets = []
        # cycle100_snippets_name = []
        has_label_flag = False
        has_capacity2_flag = False
        disc_capacity_2         = -1 # feature
        disc_capacity_1_100_max = -1 # feature
        car_snippets.sort()
        # print(car_snippets)
        # exit()
        for snippet in car_snippets:
            # each_pkl = pkl_all[0]
            train1 = torch.load(os.path.join(path, './'+snippet))
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
                cycle10_snippets.append((features[ :effe_len, 0], features[ :effe_len, 2], snippet)) # voltage and qd
                # cycle10_snippets_name.append(each_pkl)
            elif metadata['discharge_segment'] == 100: 
                cycle100_snippets.append((features[ :effe_len, 0], features[ :effe_len, 2], snippet)) # voltage and qd
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
            # else:
            #     if has_label_flag: y.pop() 
            #     continue
    assert len(X) == len(y), f'Size not matching, Found X {len(X)}, y{len(y)}'
    X = np.array(X)
    y = np.array(y)
    y = y[:, 2]
    torch.save((X,y), f'{dataset_path}/{dict_name}_dataset_points_{points}.pkl')
    print(X,y)
    # exit()

results_path = './baselines/results'
dataset_path = os.path.join(results_path, 'NE_RUL')
os.makedirs(results_path, exist_ok = True), os.makedirs(dataset_path, exist_ok = True)

train_path =  './data/lab_data/RUL_snippets/MIT_NE_fast_charging_comparison/discharging/train' 
primary_path = './data/lab_data/RUL_snippets/MIT_NE_fast_charging_comparison/discharging/test' 
secondary_path = './data/lab_data/RUL_snippets/MIT_NE_fast_charging_comparison/discharging/secondary_test' 

points_list = [128, 256, 512, 1024, 1280]
for points in points_list:
    snippets_to_features(train_path, 'NE_train', dataset_path, points)
    snippets_to_features(primary_path, 'NE_primary', dataset_path, points)
    snippets_to_features(secondary_path, 'NE_secondary', dataset_path, points)


    train_data = torch.load(f'{dataset_path}/NE_train_dataset_points_{points}.pkl')
    vali_data = torch.load(f'{dataset_path}/NE_primary_dataset_points_{points}.pkl')
    test_data = torch.load(f'{dataset_path}/NE_secondary_dataset_points_{points}.pkl')
    print(train_data[0].shape, train_data[1].shape, vali_data[0].shape, vali_data[1].shape)
    X_max, X_min =  np.max(train_data[0], axis = 0), \
                    np.min(train_data[0], axis = 0)
    X_train, X_vali, X_test = min_max_norm(value = train_data[0], min = X_min, max = X_max), \
                            min_max_norm(value = vali_data[0], min = X_min, max = X_max),\
                            min_max_norm(value = test_data[0], min = X_min, max = X_max),\
        # whether to apply log10 on y
    y_train, y_vali, y_test = np.log10(train_data[1][:]), \
                            np.log10(vali_data[1][:]), \
                            np.log10(test_data[1][:])


    l1_lambdas = [0]
    l2_lambdas = [0]
    lrs = np.array([0.005])
    device = torch.device('cpu')

    losses = []
    y_preds = []
    GTs = []

    primary_rmses = []
    secondary_rmses = []
    primary_MAPEs = []
    secondary_MAPEs = []


    for l1 in l1_lambdas:
        for l2 in l2_lambdas:
            for lr in lrs:
                net = ElasticLinearRegression(d_in = X_train.shape[1], d_out = 1, l1_lambda = l1, l2_lambda = l2).to(device)
                my_trainer = Trainer_Basic(net = net, epochs = 10000, lr = lr, bs = X_train.shape[0], device = device, \
                                    train_data = (X_train, y_train), vali_data = (X_vali, y_vali), test_data = (X_test, y_test))
                best_vali_loss, best_records = my_trainer.train()
                vali_loss, vali_GT, y_vali_pred = my_trainer.pred_log10_vali()
                test_loss, test_GT, y_test_pred = my_trainer.pred_log10_test()
                output = f"ElasticLinear_discharge_l1_{l1}_l2_{l2}_lr_{lr} | " + best_records
                print(f'{output}, rmse_primary_cycles: {vali_loss}, rmse_secondary_cycles: {test_loss}')

                vali_rmse = np.sqrt(np.mean((vali_GT - y_vali_pred)**2))
                test_rmse = np.sqrt(np.mean((test_GT - y_test_pred)**2))

                vali_mape = np.mean(np.abs(vali_GT - y_vali_pred)/vali_GT * 100)
                test_mape = np.mean(np.abs(test_GT - y_test_pred)/test_GT * 100)

                primary_rmses.append(vali_rmse)
                secondary_rmses.append(test_rmse)

                primary_MAPEs.append(vali_mape)
                secondary_MAPEs.append(test_mape)



    losses = []
    y_preds = []
    GTs = []
    for l1 in l1_lambdas:
        for l2 in l2_lambdas:
            for lr in lrs:
                net = ElasticLinearRegression(d_in = 1, d_out = 1, l1_lambda = l1, l2_lambda = l2).to(device)
                my_trainer = Trainer_Basic(net = net, epochs = 10000, lr = lr, bs = X_train.shape[0], device = device, \
                                    train_data = (X_train[:, 0].reshape(-1, 1), y_train), \
                                    vali_data = (X_vali[:, 0].reshape(-1, 1), y_vali), \
                                    test_data = (X_test[:, 0].reshape(-1, 1), y_test))
                best_vali_loss, best_records = my_trainer.train()
                vali_loss, vali_GT, y_vali_pred = my_trainer.pred_log10_vali()
                test_loss, test_GT, y_test_pred = my_trainer.pred_log10_test()
                output = f"ElasticLinear_variance_l1_{l1}_l2_{l2}_lr_{lr} | " + best_records
                print(f'{output}, rmse_primary_cycles: {vali_loss}, rmse_secondary_cycles: {test_loss}')

                vali_rmse = np.sqrt(np.mean((vali_GT - y_vali_pred)**2))
                test_rmse = np.sqrt(np.mean((test_GT - y_test_pred)**2))

                vali_mape = np.mean(np.abs(vali_GT - y_vali_pred)/vali_GT * 100)
                test_mape = np.mean(np.abs(test_GT - y_test_pred)/test_GT * 100)

                primary_rmses.append(vali_rmse)
                secondary_rmses.append(test_rmse)

                primary_MAPEs.append(vali_mape)
                secondary_MAPEs.append(test_mape)

    # print(primary_rmses)
    # exit()

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    models = ['Discharge', 'Variance']
    colors = ["#1CB3B0","#DADFEE", "#F1895E",'#CBDFDF', "#FCD924", "#FFE79D"]
    # colors = ['#CBDFDF', "#FCD924"]
    # draw

    # plt.rcParams['xtick.labelsize'] = 39  # For x-ticks
    # plt.rcParams['ytick.labelsize'] = 30  # For y-ticks


    tops = [150, 25, 220 ,14]

    axs[0, 0].bar(models, primary_rmses, capsize=5, alpha=1, color=colors, width = 0.4)
    axs[0, 0].set_title('NE Primary RMSE')
    axs[0, 0].set_ylim(top = tops[0])

    for idx, model in enumerate(models):
                axs[0, 0].text(
                    idx,
                    round(primary_rmses[idx]),
                    f'{round(primary_rmses[idx])}',
                    ha='center',
                    va='bottom',
                    fontsize=14
                )

    axs[0, 1].bar(models, primary_MAPEs, capsize=5, alpha=1, color=colors, width = 0.4)
    axs[0, 1].set_title('NE Primary MAPE')
    axs[0, 1].set_ylim(top = tops[1])
    for idx, model in enumerate(models):
                axs[0, 1].text(
                    idx,
                    round(primary_MAPEs[idx]),
                    f'{round(primary_MAPEs[idx])}',
                    ha='center',
                    va='bottom',
                    fontsize=14
                )



    axs[1, 0].bar(models, secondary_rmses, capsize=5, alpha=1, color=colors, width = 0.4)
    axs[1, 0].set_title('NE Secondary RMSE')
    for idx, model in enumerate(models):
                axs[1, 0].text(
                    idx,
                    round(secondary_rmses[idx]),
                    f'{round(secondary_rmses[idx])}',
                    ha='center',
                    va='bottom',
                    fontsize=14
                )
    axs[1, 0].set_ylim(top = tops[2])

    axs[1, 1].bar(models, secondary_MAPEs, capsize=5, alpha=1, color=colors, width = 0.4)
    axs[1, 1].set_title('NE Secondary MAPE')
    for idx, model in enumerate(models):
                axs[1, 1].text(
                    idx,
                    round(secondary_MAPEs[idx]+0.2),
                    f'{round(secondary_MAPEs[idx])}',
                    ha='center',
                    va='bottom',
                    fontsize=14
                )
    axs[1, 1].set_ylim(top = tops[3])

    for i in range(0, 2):
        for j in range(0,2):
            axs[i,j].set_xticklabels(models, fontsize=14)
            axs[i,j].set_yticklabels([int(num) for num in axs[i,j].get_yticks()], fontsize=14)

    # plt.tick_params(axis='x', which='major',length=0, labelsize=0)
    # plt.tick_params(axis='y', which='major',length =0, labelsize=0)
    # plt.tick_params(axis='both', labelsize=20)
    plt.tight_layout()

    plt.savefig(os.path.join(f'{dataset_path}', f'NE_RUL_results_reproduce_{points}.jpg'))
    plt.clf()

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    models = ['Discharge', 'Variance']
    colors = ["#1CB3B0","#DADFEE", "#F1895E",'#CBDFDF', "#FCD924", "#FFE79D"]
    # colors = ['#CBDFDF', "#FCD924"]
    # draw
    # original results from severson et al.
    primary_rmses = [91, 138]
    primary_MAPEs = [13.0, 14.7]
    secondary_rmses = [173, 196]
    secondary_MAPEs = [8.6, 11.4]



    axs[0, 0].bar(models, primary_rmses, capsize=5, alpha=1, color=colors, width = 0.4)
    axs[0, 0].set_title('NE Primary RMSE')
    axs[0, 0].set_ylim(top = tops[0])

    for idx, model in enumerate(models):
                axs[0, 0].text(
                    idx,
                    round(primary_rmses[idx]),
                    f'{round(primary_rmses[idx])}',
                    ha='center',
                    va='bottom',
                    fontsize=14
                )

    axs[0, 1].bar(models, primary_MAPEs, capsize=5, alpha=1, color=colors, width = 0.4)
    axs[0, 1].set_title('NE Primary MAPE')
    axs[0, 1].set_ylim(top = tops[1])
    for idx, model in enumerate(models):
                axs[0, 1].text(
                    idx,
                    round(primary_MAPEs[idx]),
                    f'{round(primary_MAPEs[idx])}',
                    ha='center',
                    va='bottom',
                    fontsize=14
                )



    axs[1, 0].bar(models, secondary_rmses, capsize=5, alpha=1, color=colors, width = 0.4)
    axs[1, 0].set_title('NE Secondary RMSE')
    for idx, model in enumerate(models):
                axs[1, 0].text(
                    idx,
                    round(secondary_rmses[idx]),
                    f'{round(secondary_rmses[idx])}',
                    ha='center',
                    va='bottom',
                    fontsize=14
                )
    axs[1, 0].set_ylim(top = tops[2])

    axs[1, 1].bar(models, secondary_MAPEs, capsize=5, alpha=1, color=colors, width = 0.4)
    axs[1, 1].set_title('NE Secondary MAPE')
    for idx, model in enumerate(models):
                axs[1, 1].text(
                    idx,
                    round(secondary_MAPEs[idx] +0.4),
                    f'{round(secondary_MAPEs[idx])}',
                    ha='center',
                    va='bottom',
                    fontsize=14
                )
    axs[1, 1].set_ylim(top = tops[3])

    for i in range(0, 2):
        for j in range(0,2):
            axs[i,j].set_xticklabels(models, fontsize=14)
            axs[i,j].set_yticklabels([int(num) for num in axs[i,j].get_yticks()], fontsize=14)

    # plt.tick_params(axis='x', which='major',length=0, labelsize=0)
    # plt.tick_params(axis='y', which='major',length =0, labelsize=0)
    plt.tight_layout()

    plt.savefig(os.path.join(f'{dataset_path}', f'NE_RUL_results_original_from_paper.jpg'))
