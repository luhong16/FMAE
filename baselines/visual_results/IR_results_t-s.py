import matplotlib.pyplot as plt
import os
import numpy as np

def get_best_IR_path(path_list, brand_list, fold_list):
    overall_IRs = []
    for path in path_list:
        # brand_rmses = []
        brand_IRs = []
        for brand in brand_list:
            fold_IRs = []
            for fold in fold_list:
                log_path = os.path.join(path, f'f{fold}_b{brand}.txt')
                with open (log_path, 'r') as f:
                    IR_error = np.inf
                    for line in f.readlines():
                        if 'Min test set RMSE' in line:
                            temp_min_rmse = float(line.split(' ')[-1])
                            if temp_min_rmse < IR_error:
                                IR_error = temp_min_rmse
                fold_IRs.append(IR_error)
            # if 'iTransformer/epoch20_blr5e-4' in path or 'PatchTST/epoch20_blr1e-2' in path:
            #     print(f'{brand} {path} avg auroc: {np.mean(aucs):.5f}', aucs)
            brand_IRs.append(np.mean(fold_IRs))
        overall_IRs.append(np.mean(brand_IRs))
        # print(brand_rmses)
        # overall_RMSEs.append(np.mean(brand_rmses))
    
    for idx, path in enumerate(path_list):
        print(path, overall_IRs[idx])

    return path_list[np.argmin(overall_IRs)], overall_IRs[np.argmin(overall_IRs)]*1000

patchtst_IR_log_path = './logs/IR/use_volt_current_soc/PatchTST'
itransformer_IR_log_path = './logs/IR/use_volt_current_soc/iTransformer'

patchTST_paths = [os.path.join(patchtst_IR_log_path, f_name+'/s5') \
                  for f_name in os.listdir(patchtst_IR_log_path)]
itransformer_paths = [os.path.join(itransformer_IR_log_path, f_name+'/s5') \
                      for f_name in os.listdir(itransformer_IR_log_path)]
_, patchTST_IR = get_best_IR_path(path_list=patchTST_paths, brand_list=[10], fold_list=[0,1,2,3,4])
_, iTransformer_IR = get_best_IR_path(path_list=itransformer_paths, brand_list=[10], fold_list=[0,1,2,3,4])
