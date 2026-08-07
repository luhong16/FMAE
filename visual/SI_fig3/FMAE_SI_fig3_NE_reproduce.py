import numpy as np
import os
import torch
import pandas as pd
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from pathlib import Path

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

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = f'{Path(script_dir).parent}/data/SI_fig3'

# training
train_data = torch.load(f'{data_dir}/NE_train_dataset.pkl')
vali_data = torch.load(f'{data_dir}/NE_primary_dataset.pkl')
test_data = torch.load(f'{data_dir}/NE_secondary_dataset.pkl')

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
lrs = np.array([0.02])
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


fig, axs = plt.subplots(2, 2, figsize=(8, 8))
models = ['Discharge', 'Variance']
colors = ["#1CB3B0","#DADFEE", "#F1895E",'#CBDFDF', "#FCD924", "#FFE79D"]
# colors = ['#CBDFDF', "#FCD924"]
# draw

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

plt.savefig(os.path.join(script_dir , f'RUL_NE_reproduce.jpg'))
plt.clf()

fig, axs = plt.subplots(2, 2, figsize=(8, 8))
models = ['Discharge', 'Variance']
colors = ["#1CB3B0","#DADFEE", "#F1895E",'#CBDFDF', "#FCD924", "#FFE79D"]
# colors = ['#CBDFDF', "#FCD924"]
# draw

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

plt.savefig(os.path.join(script_dir, f'RUL_NE_original_from_the_article.jpg'))


# draw MAPE