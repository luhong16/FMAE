import os
# /home/$user_name/...
# or /data/$user_name/..
__current_file_path__ = os.path.abspath(__file__)
__current_file_name__ = os.path.basename(__file__)  
# __code_path__ = __current_file_path__.replace(f'/dataset/{__current_file_name__}', '')
__user_name__ = __current_file_path__.split('/')[2]

from LiPM.dataset.MATR_dataset import load_MATR_dataset
from LiPM.dataset.NASA_dataset import load_NASA_dataset
from LiPM.dataset.CALCE_dataset import load_CALCE_dataset
from LiPM.dataset.HUST_dataset import load_HUST_dataset
from LiPM.dataset.RWTH_dataset import load_RWTH_dataset
from LiPM.dataset.HNEI_dataset import load_HNEI_dataset
from LiPM.dataset.ULPurdue_dataset import load_ULPurdue_dataset
from LiPM.dataset.SNLNCA_dataset import load_SNLNCA_dataset
from LiPM.dataset.SNLLFP_dataset import load_SNLLFP_dataset
from LiPM.dataset.SNLNMC_dataset import load_SNLNMC_dataset
from LiPM.dataset.THU_dataset import load_THU_dataset

import torch
from multiprocessing import Pool
from torch.utils.data import ConcatDataset


# save_data
save_dir_dict = {
    'MATR': f'/data/{__user_name__}/MATR/temp2/',
    'NASA': f'/data/{__user_name__}/NASA/temp2',
    'CALCE': f'/data/{__user_name__}/CALCE/temp2',
    'HUST': f'/data/{__user_name__}/HUST/temp2',
    'RWTH': f'/data/{__user_name__}/RWTH/temp2',
    'HNEI': f'/data/{__user_name__}/HNEI/temp2',
    'ULPurdue': f'/data/{__user_name__}/ULPurdue/temp2',
    'SNLLFP': f'/data/{__user_name__}/SNL_LFP/temp2',
    'SNLNCA': f'/data/{__user_name__}/SNL_NCA/temp2',
    'SNLNMC': f'/data/{__user_name__}/SNL_NMC/temp2',
    'THU': f'/data/{__user_name__}/THU/temp2',
}


load_dataset_dict = {
    'MATR': load_MATR_dataset,
    'NASA': load_NASA_dataset,
    'CALCE': load_CALCE_dataset,
    'HUST': load_HUST_dataset,
    'RWTH': load_RWTH_dataset,
    'HNEI': load_HNEI_dataset,
    'ULPurdue': load_ULPurdue_dataset,
    'SNLLFP': load_SNLLFP_dataset,
    'SNLNCA': load_SNLNCA_dataset,
    'SNLNMC': load_SNLNMC_dataset,
    'THU': load_THU_dataset,
}


all_data_idx = {
    'HUST': [i for i in range(77)],
    'NASA': [i for i in range(8)],
    'MATR': [i for i in range(185)],
    'CALCE': [i for i in range(23)],
    'RWTH': [i for i in range(23)],
    'HNEI': [i for i in range(14)],
    'ULPurdue': [i for i in range(22)],
    'SNLLFP': [i for i in range(30)],
    # 'SNLNCA': [i for i in range(24)],
    # 'SNLNMC': [i for i in range(29)],
    'THU': [i for i in range(347)]
}


ir_regression_tasks = ['ir_IRRegression', 'ir_QDRegression', 'ir_QCRegression', 'ir_CTimeRegression',
                       'ir_RULRegression', 'ir_ECRegression', 'ir_EDRegression' , 'ir_DeltaQc',
                       'ir_DeltaQd', 'ir_DeltaEc', 'ir_DeltaEd', 'ir_SOHRegression']
regression_task = ['IRRegression', 'QDRegression', 'QCRegression', 'CTimeRegression',
                       'RULRegression', 'ECRegression', 'EDRegression' , 'DeltaQc',
                       'DeltaQd', 'DeltaEc', 'DeltaEd', 'SOHRegression']

def load_dataset(data_idx, data_config, data_name, cache=False):
    dataset_list = []
    task = data_config['task']
    save_dir = save_dir_dict[data_name]
    for idx in data_idx:
        save_path = f'{save_dir}/{task}/dataset_{idx}.pth'
        if cache and os.path.exists(save_path):
            print(f'load {data_name}', idx)
            cur_datasets = torch.load(save_path)            
            reset_param(cur_datasets, data_config)
        else:
            if not os.path.exists( f'{save_dir}/{task}'):
                os.makedirs( f'{save_dir}/{task}')
            print(f'make dataset {data_name}', idx)
            cur_datasets = load_dataset_dict[data_name](idx, data_config)
            if cache:
                torch.save(cur_datasets, save_path, pickle_protocol=4)
        dataset_list.extend(cur_datasets)
    datasets = []
    for dataset in dataset_list:
        if len(dataset) > 0:
            datasets.append(dataset)
    return datasets


def reset_param(datasets, data_config):
    if data_config['task'] in ['pretrain_ir', 'forecast_ir']:
        # patched dataset
        for ds in datasets:
            ds.patch_num = data_config['patch_num']
            ds.stride = data_config['stride'] if  data_config['stride'] > 0 else data_config['patch_num']
            ds.pred_num = data_config['pred_num']
    elif data_config['task'] in ['pretrain', 'forecast']:
        for ds in datasets:
            ds.seq_len = data_config['seq_len']
            ds.pred_len = data_config['pred_len']
            ds.stride = data_config['stride'] if  data_config['stride'] > 0 else data_config['seq_len']
            if 'pred_var' in data_config:
                ds.pred_var = data_config['pred_var']  # forecast
    elif data_config['task'] in regression_task:
        for ds in datasets:
            ds.seq_len = data_config['seq_len']
            ds.stride = data_config['stride'] if  data_config['stride'] > 0 else data_config['seq_len']
    elif data_config['task'] in ir_regression_tasks:
        for ds in datasets:
            ds.patch_num = data_config['patch_num']
            ds.stride = data_config['stride'] if  data_config['stride'] > 0 else data_config['patch_num']
    elif data_config['task'] in ['AD']:
        pass

def load_pretrain_dataset(data_names=['MATR', 'NASA'], data_idx=[[1, 2, 3], [0, 1]], data_config=None, cache=False, is_cat=False): 
    dataset_list = []
    for i, name in enumerate(data_names):
        print('dataset:', name, 'begin')
        dataset_list += load_dataset(data_idx[i], data_config[name], name, cache)
        print('dataset:', name, 'end')
    print('totally', len(dataset_list), 'basic datasets')

    if is_cat:
        return ConcatDataset(dataset_list)
    return dataset_list


def load_downstream_dataset(data_name='NASA', data_idx=[0, 1], data_config=None, is_cat=True, cache=False):
    datasets = load_dataset(data_idx, data_config, data_name, cache)
    print('totally', len(datasets), 'basic datasets')

    if is_cat: 
        dataset = ConcatDataset(datasets)
        print('totally', len(dataset), 'samples')
        return dataset
    else:
        return datasets
    

def preprocess(data_name):
    data_idx = []
    data_config = dict()
    for name in data_name:
        data_idx.append(all_data_idx[name])
        data_config[name] = {
            'task': 'pretrain_ir',
            'patch_num': 16,
            'stride': -1,
            'pred_num': 0}
    dataset = load_pretrain_dataset(data_name, data_idx, data_config)
    print(f'{data_name}, sample_num: {len(dataset)}')  
    

# test
if __name__ == '__main__':
    data_name = [
        # 'HUST',
        # 'NASA',
        # 'CALCE',
        # 'MATR',
        # 'RWTH',
        # 'HNEI',
        # 'ULPurdue',
        # 'SNLLFP',  
        # 'SNLNMC',
    ]
    pool = Pool(48)

    for name in data_name:
        cur_data_idx = all_data_idx[name]
        for idx in cur_data_idx:
            data_idx = [[idx]]
            data_config = {name: {
                'task': 'pretrain_ir',
                'patch_num': 16,
                'stride': -1,
                'pred_num': 0
            }}
            pool.apply_async(load_pretrain_dataset, ([name], data_idx, data_config))

    pool.close()
    pool.join()
    print('ok')