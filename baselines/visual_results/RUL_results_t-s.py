import os
import numpy as np


def read_rul_logs_time_series_model(f_name):

    with open (f_name, 'r') as f:
        # records
        best_vali_loss = np.inf
        vali_GT = []
        # test_GT = []
        vali_pred = []
        # test_pred = []    
        # vali_flag = False
        # test_flag = False

        train_flag = False
        record_flag = False
        rmse_flag = False
        for line in f.readlines():
            if 'test_RMSE' in line:
                record_flag = True
                temp_vali_pred = []
                temp_vali_GT = []
                continue
            if record_flag and 'test_RMSE' not in line and 'Min test cell level RMSE' not in line:
                temp_list = line.split(' ')
                # print(temp_list)
                temp_vali_pred.append(float(temp_list[-1][:-1]))
                temp_vali_GT.append(float(temp_list[1]))
                # exit()
                continue
            if record_flag and 'Min test cell level RMSE' in line:
                record_flag = False
                temp_list = line.split(' ')
                temp_rmse = float(temp_list[-1][:-1])
                if temp_rmse <= best_vali_loss:
                    vali_GT = temp_vali_GT
                    vali_pred = temp_vali_pred
                    best_vali_loss = temp_rmse
                continue
            
        # print(best_vali_loss)
        # print(vali_pred)
    return vali_GT, vali_pred


def get_best_path(path_list, brand_list, fold_list):

    overall_RMSEs = []
    for path in path_list:
        brand_rmses = []
        for brand in brand_list:
            fold_rmses = []
            for fold in fold_list:
                log_path = os.path.join(path, f'f{fold}_b{brand}.txt')
                vali_GT, vali_pred= read_rul_logs_time_series_model(log_path)
                temp1 = np.array(vali_pred)
                temp2 = np.array(vali_GT)
                fold_rmses.append(np.sqrt(np.mean((temp1 - temp2)**2)))
            brand_rmses.append(np.mean(fold_rmses))
        print(brand_rmses)
        overall_RMSEs.append(np.mean(brand_rmses))
    
    for idx, path in enumerate(path_list):
        print(path, overall_RMSEs[idx])

    return path_list[np.argmin(overall_RMSEs)]



brands_domains = ['Lab']
lab_brands = [10, 12, 13]
folds = [0, 1, 2, 3, 4]

patchtst_rul_log_path = './logs/RUL/use_volt_current_soc/PatchTST'
itransformer_rul_log_path = './logs/RUL/use_volt_current_soc/iTransformer'

patchTST_paths = [os.path.join(patchtst_rul_log_path, f_name+'/s5') \
                  for f_name in os.listdir(patchtst_rul_log_path)]
itransformer_paths = [os.path.join(itransformer_rul_log_path, f_name+'/s5') \
                      for f_name in os.listdir(itransformer_rul_log_path)]

patchTST_path = get_best_path(patchTST_paths, brand_list=lab_brands, fold_list=folds)
itransformer_path = get_best_path(itransformer_paths, brand_list=lab_brands, fold_list=folds)