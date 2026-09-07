import numpy as np
import os
import sys
import logging
from logging.handlers import RotatingFileHandler
import subprocess
from pathlib import Path
from LiPM.battery_model import BatteryNet
from LiPM.dataset.battery_dataset import load_pretrain_dataset, load_downstream_dataset
import torch
import joblib
import numpy as np


# stdout and stderr --> logger
class StdOutWrapper:
    def __init__(self, logger, level, original_stream):
        self.logger = logger
        self.level = level
        self.original_stream = original_stream

    def write(self, message):
        if message.strip():
            self.logger.log(self.level, message.strip())
        self.original_stream.write(message)

    def flush(self):
        self.original_stream.flush()

    def __getattr__(self, attr):
        return getattr(self.original_stream, attr)


def setup_logger(log_file_path):
    logger = logging.getLogger('stdout_stderr_logger')
    logger.setLevel(logging.DEBUG)

    # RotatingFileHandler
    file_handler = RotatingFileHandler(log_file_path, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("[%(asctime)s] - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # stdout, stderr
    sys.stdout = StdOutWrapper(logger, logging.INFO, sys.stdout)
    sys.stderr = StdOutWrapper(logger, logging.ERROR, sys.stderr)

    return logger

def make_save(save_path, name='test', append_log=False, idx=0):
    if append_log:
        if idx < 0:
            raise ValueError('idx must be specified when append_log is True')
        save_path = os.path.join(save_path, f'{name}_{idx}')
    else:
        i = 0
        while os.path.exists(os.path.join(save_path, f'{name}_{i}')):
            i += 1 
        save_path = os.path.join(save_path, f'{name}_{i}')
        os.makedirs(save_path)
    print(f'save path: {save_path}')

    log_file_path = os.path.join(save_path, 'log.txt')
    setup_logger(log_file_path)

    if not append_log:
        # backup code and log
        os.mkdir(os.path.join(save_path, 'model_save'))
        os.mkdir(os.path.join(save_path, 'code'))
        cur_path = Path(__file__).resolve()
        code_path = cur_path.parent
        cmd = f'cd {code_path} && find . -name \'*.py\' -exec cp --parents \{{}} {save_path}/code \;'
        subprocess.run(cmd, shell=True, check=True)

    return save_path


def load_model(model: BatteryNet, config):
    backbone_path = os.path.join(config.save_path, f'model_save/backbone_{config.ck_point}.pth')
    model.backbone.load_state_dict(torch.load(backbone_path))

    head_path = os.path.join(config.save_path, f'model_save/head_{config.ck_point}.pth')
    head_state_dict = torch.load(head_path)
    model.inter_head.load_state_dict(head_state_dict['inter'])
    model.Q_head.load_state_dict(head_state_dict['Q'])


def save_model(model: BatteryNet, config, iter_count):
    backbone_path = os.path.join(config.save_path, f'model_save/backbone_{iter_count+config.ck_point+1}.pth')
    torch.save(model.backbone.state_dict(), backbone_path)
    
    head_path = os.path.join(config.save_path, f'model_save/head_{iter_count+config.ck_point+1}.pth')
    heads = {
        'inter': model.inter_head.state_dict(),
        'Q': model.Q_head.state_dict()
    }
    torch.save(heads, head_path)
    print(f'save at: {iter_count+config.ck_point+1}')


def get_pretrain_data_config(config, is_test):
    data_names = [
        'HNEI',
        'ULPurdue',
        'SNLLFP',
        'HUST',
        'NASA',
        'CALCE',
        'MATR',
        'RWTH',
    ]

    # data_idx for pretrain
    all_data_idx = {
        'CALCE': [i for i in range(19)],  # 23 totally, 19 for pretrain
        'HNEI': [i for i in range(11)],  # 14 totally, 11 for pretrain
        'HUST': [i for i in range(70)],  # 77 totally, 70 for pretrain
        'MATR': [i for i in range(125)] + [i for i in range(140, 185)],  # 185 totally, 170 for pretrain
        # in MATR 140~184 no IR label
        'NASA': [i for i in range(5)] + [i for i in range(10, 24)], # 34 totally, 24 for pretrain
        # Data after B0030 is not suitable for downstream tasks (too few labels)
        'RWTH': [i for i in range(42)],  # 48 totally, 42 for pretrain
        'ULPurdue': [i for i in range(6)] + [i for i in range(10, 22)],  # 22 totally, 18 for pretrain
        'SNLLFP': [i for i in range(8)] + [i for i in range(11, 18)]  # 18 totally, 15 for pretrain
    }

    # for test, data_idx
    if is_test: 
        _test_num = 3
        all_data_idx = {
            'CALCE': [i for i in range(_test_num)],  # 23 totally
            'HNEI': [i for i in range(_test_num)],  # 14 totally
            'HUST': [i for i in range(_test_num)],  # 77 totally
            'MATR': [i for i in range(_test_num)],  # 185 totally
            'NASA': [i for i in range(_test_num)], # 34 totally
            'RWTH': [i for i in range(_test_num)],  # 48 totally
            'ULPurdue': [i for i in range(_test_num)],  # 22 totally
            'SNLLFP': [i for i in range(_test_num)]  # 18 totally
        }

    data_config = dict()
    data_idx = []
    if 'ir' in config.task:
        for name in data_names:
            data_idx.append(all_data_idx[name])
            data_config[name] = {
                'task': 'pretrain_ir',
                'patch_num': config.patch_num,
                'stride': -1,  # patch stride
                'pred_num': 0  
                }
            if name == 'NASA':
                data_config[name]['stride'] = 1
            elif name == 'HUST':
                data_config[name]['stride'] = 4
            elif name == 'HNEI':
                data_config[name]['stride'] = 1
            elif name == 'SNLLFP':
                data_config[name]['stride'] = 1
            elif name == 'ULPurdue':
                data_config[name]['stride'] = 1
    else:
        for name in data_names:
            data_idx.append(all_data_idx[name])
            data_config[name] = {
                'task': config.task,
                'seq_len': config.seq_len,
                'stride': -1,  # seq_stride
                'pred_len': config.pred_len 
                }
            # If it is pre-training of forecasting type, pred_len > 0, otherwise it is set to 0
            # forecasting is not used in our current work

            # The following "stride" setting is based on balancing the length of each dataset, 
            # which is len(dataset)
            if name == 'NASA':
                data_config[name]['stride'] = 32
            elif name == 'CALCE':
                data_config[name]['stride'] = 32
            elif name == 'HNEI':
                data_config[name]['stride'] = 16
            elif name == 'SNLLFP':
                data_config[name]['stride'] = 16
            elif name == 'ULPurdue':
                data_config[name]['stride'] = 16
            elif name == 'RWTH':
                data_config[name]['stride'] = 64

    return data_names, data_idx, data_config


def get_downstream_dataset(data_name, config, is_zero_shot=False):
    # return: train_datasets, test_datasets
    # train_datasets: [dataset1, dataset2, ...]
    # test_datasets: [dataset1, dataset2, ...]
    data_config = {
        'task': config.task,
        'stride': config.stride
    }
    if 'ir' in config.task:
        data_config['patch_num'] = config.patch_num
    else:
        data_config['seq_len'] = config.seq_len

    all_data_idx = {
        'CALCE': [19, 20, 21, 22],
        'HNEI': [11, 12, 13],
        'HUST': [i for i in range(70, 77)],
        'MATR': [i for i in range(125, 140)],
        'NASA': [i for i in range(5, 10)],
        'RWTH': [i for i in range(42, 48)],
        'ULPurdue': [i for i in range(6, 10)],
        'SNLLFP': [i for i in range(8, 11)]
    }
    _all_datasets = []
    for idx in all_data_idx[data_name]:
        _cur_datasets = load_downstream_dataset(data_name, [idx], data_config, is_cat=False, cache=config.cache)
        if len(_cur_datasets) > 0:
            _all_datasets.append(_cur_datasets)
    # _all_datasets: [[dataset1, dataset2], [dataset3, dataset4], [...], ...]

    if is_zero_shot:
        # split training set and test set by battery
        train_num = len(_all_datasets) * 3 // 4
        np.random.shuffle(_all_datasets)
        train_datasets = []
        test_datasets = []
        for i in range(len(_all_datasets)):
            if i < train_num:
                train_datasets.extend(_all_datasets[i])
            else:
                test_datasets.extend(_all_datasets[i])
        print('battery-split')
    else:
        # split by cycle
        train_datasets = []
        test_datasets = []
        for datasets in _all_datasets:
            train_datasets.extend(datasets[:len(datasets) * 3 // 4])
            test_datasets.extend(datasets[len(datasets) * 3 // 4:])
        
        print('cycle-split')
    
    return train_datasets, test_datasets


def get_pretrain_dataset(config, is_cat=True, is_test=False):
    # is_cat: whether to concatenate the datasets
    data_names, data_idx, data_config = get_pretrain_data_config(config, is_test)
    return load_pretrain_dataset(data_names, data_idx, data_config, config.cache, is_cat)


class LossRecord(object):
    def __init__(self, loss_names=['total', 'MAE', 'Q', 'MAE_mae', 'Q_mae']) -> None:
        self.loss_dict = {}
        for name in loss_names:
            self.loss_dict[name] = []
        self.count_nums = None
    
    def append(self, cur_loss_dict, count_num=None):
        for loss_name in cur_loss_dict:
            if loss_name not in self.loss_dict:
                raise ValueError(f'{loss_name} is not in loss_dict')
            self.loss_dict[loss_name].append(cur_loss_dict[loss_name])

        if count_num is not None:
            if self.count_nums is None:
                self.count_nums = []
            self.count_nums.append(count_num)
    
    def get_mean(self, show_iter=None):
        if show_iter is None:
            show_iter = len(self.loss_dict['total'])
        mean_loss_dict = {}
        if self.count_nums is not None:
            # use count_nums and show_iter to calculate the weighted mean
            def weighted_mean(loss_list, count_nums):
                return np.sum(np.array(loss_list) * np.array(count_nums)) / np.sum(count_nums)
            for loss_name in self.loss_dict:
                mean_loss_dict[loss_name] = weighted_mean(self.loss_dict[loss_name][-show_iter:], self.count_nums[-show_iter:])
        else:
            # calculate the mean of last show_iter losses
            for loss_name in self.loss_dict:
                mean_loss_dict[loss_name] = np.mean(self.loss_dict[loss_name][-show_iter:])

        return mean_loss_dict

    def print(self, epoch_count=None, iter_count=None, show_iter=None):
        if epoch_count is not None:
            print('epoch#', epoch_count, 'iter#', iter_count)
        if show_iter is None:
            show_iter = len(self.loss_dict['total'])

        mean_loss_dict = self.get_mean(show_iter)
        loss_names = sorted(mean_loss_dict.keys())
        for name in loss_names:
            print(f'{name}: {mean_loss_dict[name]:.6f}')
        print('-----------------------------------')
    
    def save(self, save_path, iter_idx=0):
        joblib.dump(self.loss_dict, os.path.join(save_path, f'loss_record_{iter_idx}.pkl'))

    def load(self, save_path, iter_idx=0):
        last_loss_dict = joblib.load(os.path.join(save_path, f'loss_record_{iter_idx}.pkl'))
        for loss_name in last_loss_dict:
            if loss_name not in self.loss_dict:
                print(f'{loss_name} is not in loss_dict')
                self.loss_dict[loss_name] = []
            self.loss_dict[loss_name] = last_loss_dict[loss_name]
        for loss_name in self.loss_dict:
            if len(self.loss_dict[loss_name]) == 0:
                print(f'{loss_name} is empty')