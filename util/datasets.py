# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL
import numpy as np
import torch
import random
import math
import pickle

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class Normalizer:
    def __init__(self, dfs=None, variable_length=False):
        print('dfs', np.array(dfs).shape)
        self.max_norm = 0
        self.min_norm = 0
        self.std = 0
        self.mean = 0
        res = []
        if dfs is not None:
            if variable_length:
                norm_length = min([len(df) for df in dfs])
                dfs = [df[0:norm_length] for df in dfs]
            res.extend(dfs)
            res = np.array(res)
            self.compute_min_max(res)
        else:
            print("df list not specified")

    def compute_min_max(self, res):
        column_max_all = np.max(res, axis=2)
        column_min_all = np.min(res, axis=2)
        column_std_all = np.std(res, axis=2)
        column_mean_all = np.mean(res, axis=2)
        self.max_norm = np.expand_dims(np.max(column_max_all, axis=0), -1)
        self.min_norm = np.expand_dims(np.min(column_min_all, axis=0), -1)
        self.std = np.expand_dims(np.mean(column_std_all, axis=0), -1)
        self.mean = np.expand_dims(np.mean(column_mean_all, axis=0), -1)

    def std_norm_df(self, df):
        return (df - self.mean) / np.maximum(1e-4, self.std)

    def norm_func(self, df):
        df_norm = df.copy()
        df_norm = (df_norm - self.mean) / np.maximum(np.maximum(1e-4, self.std), 0.075 * (self.max_norm - self.min_norm)) #0.075
        return df_norm


class PreprocessNormalizer:

    def __init__(self, dataset, metadata=None, normalizer_fn=None, label_normalizer=None, num_snippet=0, cycle_gap=None, car_dict=None, mileage=None, lab=None, cycle_info=None, downstream='', task='', seed=0, brand_num=None, **kwargs):
        self.dataset = dataset
        self.normalizer_fn = normalizer_fn
        self.metadata = metadata
        self.label_normalizer = label_normalizer # (mean, std)
        self.num_snippet = num_snippet # how many snippets are used in multi-snippet MAE
        self.car_dict = car_dict
        self.mileage = mileage
        self.lab = lab
        self.cycle_info = cycle_info # (current cycle,  remaining cycle RUL)
        self.cycle_gap = cycle_gap
        self.downstream = downstream
        self.task = task
        self.brand_num = brand_num
        self.rng = np.random.default_rng(seed)

        if self.downstream in ['pretrain', 'anomaly', 'capacity', 'RUL']:
            
            if self.downstream == 'anomaly':
                max_mileage_dict = pickle.load(open(r"normailze/max_mileage.pkl", 'rb'))
            for _, (all_snippets, max_mileage) in self.car_dict.items():
                for j in all_snippets:
                    if self.downstream == 'anomaly':
                        self.mileage[j] /= max_mileage_dict[_]
                    else:
                        self.mileage[j] /= max_mileage
                if self.downstream in ['capacity']:
                    self.car_dict[_][0].sort(key=lambda x: (self.mileage[x]))
        elif self.downstream in ['IR']:
            pass
        else:
            raise NotImplementedError

        if self.num_snippet > 0:
            self.idx = None
            self.build_idx()

    def __len__(self):
        if self.num_snippet == 0:
            return len(self.dataset[0])
        else:
            return len(self.idx)

    def build_idx(self):
        if self.downstream in ['pretrain']:
            self.shuffle_pretrain()
        elif self.downstream in ['RUL']:
            self.shuffle_finetune()
        else:
            raise NotImplementedError

    def shuffle_finetune(self):
        idx = []
        assert self.downstream == 'RUL'
        for (car, _, cycle), (all_snippets, max_mileage) in self.car_dict.items(): 
            for j in all_snippets:
                snippets = [j]
                for i in range(1, self.num_snippet):
                    pair = (car, _, cycle - i * self.cycle_gap)
                    if pair not in self.car_dict:
                        continue
                    assert len(self.car_dict[pair][0]) > 0
                    p = self.rng.integers(len(self.car_dict[pair][0]))
                    snippets.append(self.car_dict[pair][0][p])
                if len(snippets) != self.num_snippet:
                    continue
                idx.append(snippets)
        self.idx = np.array(idx) 

    # for each car, shuffle corresponding snippets, stack them
    def shuffle_pretrain(self):
        idx = []
        for _, (all_snippets, max_mileage) in self.car_dict.items(): 
            l = len(all_snippets) // self.num_snippet
            if l == 0:
                continue
            np_snippets = self.rng.permutation(all_snippets)
            idx.append(np.stack([np_snippets[i*l:(i+1)*l] for i in range(self.num_snippet)], axis=1))
        self.idx = np.concatenate(idx, axis = 0)

    def __getitem__(self, idx):
        if self.num_snippet == 0:
            df, label, car, brand = self.dataset[0][idx], self.dataset[1][idx], self.dataset[2][idx], self.dataset[3][idx]
           
            df = self.normalizer_fn(df)
            
            if self.task == "batterybrandmileage":
                normalize_mile = np.ones((1, df.shape[1]), dtype = df.dtype) * self.mileage[idx]
                df = np.concatenate((df, normalize_mile), axis = 0)
                
        else:
            assert self.normalizer_fn is not None
            _, label, car, brand = self.dataset[0][self.idx[idx, 0]], self.dataset[1][self.idx[idx, 0]], self.dataset[2][self.idx[idx, 0]], self.dataset[3][self.idx[idx, 0]]
            dfs = []
            for ids in self.idx[idx]:
                df = self.dataset[0][ids]
                assert self.dataset[2][ids] == car
                assert self.dataset[3][ids] == brand

                df = self.normalizer_fn(df)
                
                normalize_mile = np.ones((1, df.shape[1]), dtype = df.dtype) * self.mileage[ids]
                
                df = np.concatenate((df, normalize_mile), axis = 0)
                dfs.append(df)
            df = np.stack(dfs, axis = 0)
        
        if self.downstream in ['RUL']:
            if self.num_snippet == 0:
                cycle, rul = self.cycle_info[idx][0], self.cycle_info[idx][1]
            else:
                cycle, rul = self.cycle_info[self.idx[idx, 0]][0], self.cycle_info[self.idx[idx, 0]][1]
            return df, label, car, cycle, rul

        elif self.downstream == 'pretrain':
            if self.num_snippet == 0:
                lab = self.lab[idx]
            else:
                lab = self.lab[self.idx[idx, 0]]
            return df, label, car, lab

        else:
            return df, label, car


def load_dataset(fold_num, brand_num, same_normalizer=False, car_dict_dir='five_fold_utils_six_brand_all', downstream='', data_type=None, normalizer = None, dataset_fn=None, ind_ood_car_dict=None, all_car_dict=None, num_snippet=0, cycle_gap=0, data_percent=100, seed=0, normalizer_lab=None, task=''):
    
    TOTAL_BRAND_NUM = 14
    
    if data_type == 'pretrain':
        brand = []
        pretrain_list = []
        brand.append([])
        for i in range(1, 7):
            ind_ood_car_dict = np.load(f'./{car_dict_dir}/ind_odd_dict{i}.npz.npy', allow_pickle=True).item()
            brand.append(ind_ood_car_dict['ind_sorted'] + ind_ood_car_dict['ood_sorted'])
            pretrain_list += ind_ood_car_dict['ind_sorted'] + ind_ood_car_dict['ood_sorted']
        all_car_dict = np.load(f'./{car_dict_dir}/all_car_dict.npz.npy', allow_pickle=True).item()
        car_number = pretrain_list

    else:
        ind_car_num_list = ind_ood_car_dict['ind_sorted']
        ood_car_num_list = ind_ood_car_dict['ood_sorted']

        print('ind_car_num_list', ind_car_num_list)
        print('ood_car_num_list', ood_car_num_list)
        

        if downstream in ['anomaly']:
            if data_type == 'finetune_train' or data_type == "finetune_valid": 
                car_number = ind_car_num_list[:int(fold_num * len(ind_car_num_list) / 5)] + ind_car_num_list[
                                                int((fold_num + 1) * len(ind_car_num_list) / 5):] + ood_car_num_list[int(fold_num * len(ood_car_num_list) / 5):int((fold_num + 1) * len(ood_car_num_list) / 5)]
            elif data_type == 'finetune_test':
                car_number = ind_car_num_list[int(fold_num * len(ind_car_num_list) / 5):int((fold_num + 1) * len(ind_car_num_list) / 5)]\
                            + ood_car_num_list[:int(fold_num * len(ood_car_num_list) / 5)] + ood_car_num_list[int((fold_num + 1) * len(ood_car_num_list) / 5):]
            else:
                raise RuntimeError("No such data type")

        elif downstream in ['capacity', 'IR', 'RUL']:
            
            if data_type == 'finetune_train' or data_type == "finetune_valid": 
                
                car_number = ind_car_num_list[:int(fold_num * len(ind_car_num_list) / 5)] \
                            + ind_car_num_list[int((fold_num + 1) * len(ind_car_num_list) / 5):] \
                            + ood_car_num_list[:int(fold_num * len(ood_car_num_list) / 5)] \
                            + ood_car_num_list[int((fold_num + 1) * len(ood_car_num_list) / 5):]
                
            elif data_type == 'finetune_test':
                
                car_number = ind_car_num_list[int(fold_num * len(ind_car_num_list) / 5):int(
                                (fold_num + 1) * len(ind_car_num_list) / 5)] \
                            + ood_car_num_list[int(fold_num * len(ood_car_num_list) / 5):int(
                                (fold_num + 1) * len(ood_car_num_list) / 5)]
                
            else:
                raise RuntimeError("No such data type")
        
        else:
            raise NotImplementedError
        

    X = []
    y = []
    car_id = []
    brand_id = []
    mileage = []
    lab = [] # 1 is lab data, 0 is not lab data
    cycle_info = []
    metadata = []
    car_dict = {}

    print('car_number is ', data_type, car_number)
    print(len(car_number))
    
    for (___, each_num) in enumerate(car_number):
        print(___, len(all_car_dict[each_num]))
        not_first = False
        if downstream in ['anomaly', 'capacity', 'IR']:  
            if data_type == 'finetune_train':
                pkls = all_car_dict[each_num][:int(len(all_car_dict[each_num]) * 0.01 * data_percent)]
            elif data_type == 'finetune_valid':
                pkls = all_car_dict[each_num][int(len(all_car_dict[each_num]) * 0.01 * data_percent):]
                if len(pkls) == 0:
                    pkls = all_car_dict[each_num][-1:]
            elif data_type == 'finetune_test':
                pkls = all_car_dict[each_num]
            else:
                raise NotImplementedError
        elif downstream in ['pretrain', 'RUL']:    
            pkls = all_car_dict[each_num]
        else:
            raise NotImplementedError
        
        for pkl_all in pkls:
            each_pkl = pkl_all[0]
            train1 = torch.load(each_pkl) 
            
            if downstream in ['RUL']:
                if train1[1]['label'][1] < 100 - data_percent:
                    continue

                if train1[1]['label'][1] % 20 != 0:
                    continue
                
            if downstream == 'capacity' and float(train1[1]["capacity"]) == 0 and brand_num <= 7: #no capacity data
                assert NotImplementedError

            x = train1[0].transpose(1, 0).astype(np.float32)
            X.append(x) 
            
            metadata.append(train1[1])
            if downstream in ['anomaly', 'IR', 'pretrain']:
                if isinstance(train1[1]['label'], str):
                    assert downstream in ['anomaly', 'pretrain']
                    y.append(int(train1[1]['label'][0]))
                else:
                    y.append(train1[1]['label'])
            elif downstream in ['capacity']:
                if brand_num <= 7:
                    y.append(float(train1[1]['capacity']) / 100.)
                elif brand_num == 14:
                    y.append(float(train1[1]['label']) / 100.)
                else:
                    y.append(float(train1[1]['label'][0]) / 100.)
            elif downstream in ['RUL']:
                y.append(float(train1[1]['label'][2]) / 1000.)
            else:
                raise NotImplementedError

            if downstream == 'pretrain' or brand_num <= 7:
                mileage_cycle = float(train1[1]['mileage'])
            else:
                if train1[1]['charge_segment'] != -1:
                    mileage_cycle = train1[1]['charge_segment']
                elif train1[1]['discharge_segment'] != -1:
                    mileage_cycle = train1[1]['discharge_segment']
                else:
                    raise NotImplementedError

            if downstream in ['RUL']:
                cycle_info.append((train1[1]['label'][1], train1[1]['label'][2]))

            mileage.append(mileage_cycle)

            car = int(train1[1]['car'])
            car_id.append(car)
            
            if downstream == 'pretrain' or brand_num <= 7:
                lab.append(0)
            else:
                lab.append(1)

            if downstream == 'pretrain' or brand_num <= 7:
                charge = 'capacity' in train1[1] 
            else:
                charge = train1[1]['charge_segment'] != -1

            if downstream in ['pretrain', 'anomaly', 'capacity', 'RUL']:

                if downstream in ['pretrain', 'anomaly', 'capacity']:
                    pair = (car, charge)
                elif downstream in ['RUL']:
                    pair = (car, charge, mileage_cycle)
                    
                if pair not in car_dict:
                    car_dict[pair] = [[], 0.]
                car_dict[pair][0].append(len(y)-1)
                if downstream in ['pretrain', 'anomaly']:
                    car_dict[pair][1] = max(car_dict[pair][1], mileage_cycle)
                elif brand_num <= 6:
                    car_dict[pair][1] = 200000. 
                else:
                    car_dict[pair][1] = 1000.
                
            if data_type != 'pretrain':
                brand_id.append(brand_num)
            else:
                only_in_one_brand = False
                for i in range(1, 7):
                    if each_num in brand[i]:
                        brand_id.append(i)
                        assert only_in_one_brand is False
                        only_in_one_brand = True
                assert only_in_one_brand is True

    
    if normalizer is None:
        if same_normalizer:
            normalizer = Normalizer(dfs=X)
            
    if downstream in ['anomaly', 'pretrain']:
        y = np.stack(y).astype(np.int64)
    elif downstream in ['capacity', 'IR', 'RUL']:
        y = np.stack(y).astype(np.float32)
    else:
        raise NotImplementedError
    car_id = np.stack(car_id).astype(np.int64)
    brand_id = np.stack(brand_id).astype(np.int64)
    lab = np.stack(lab).astype(np.int64)

    label_normalizer = (np.mean(y), np.std(y))

    dataset = dataset_fn((X, y, car_id, brand_id), label_normalizer=label_normalizer, metadata=metadata, mileage=mileage, lab=lab, cycle_info=cycle_info, normalizer_fn=normalizer.norm_func if same_normalizer else [
                            None if normalizer[i] is None else normalizer[i].norm_func for i in range(len(normalizer))], num_snippet=num_snippet, cycle_gap=cycle_gap, car_dict=car_dict, downstream=downstream, seed=seed, task=task, brand_num=brand_num)
    
    return dataset, normalizer
