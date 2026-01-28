import sys
sys.path.append('./baselines')
import pandas as pd
import os
import numpy as np
import torch
from main_baseline_utils import *
from sklearn import metrics
from sklearn.metrics import auc


path = "./baselines/results/capacity/diff_models_losses"
output_csv_path = "./baselines/results/capacity/output_csv"
os.makedirs(output_csv_path, exist_ok = True)

brands_list = [[1, 2, 3, 4, 5, 6], 
               [10, 11, 12, 13], 
               [7],
               [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13]]
brands_domains = ['Car', 
                  'Lab', 
                  'ES', 
                  'All']
folds = [0, 1, 2, 3, 4]
model_names = ['RF', 'XGBoost']
for idx, brands in enumerate(brands_list):
    domain = brands_domains[idx]
    has_models = None
    models = []
    brands_rmse = []
    num_models = []

    domain_GTs = []
    domain_preds = []
    for brand in brands:
        folds_rmse = []
        fold_GTs = []
        fold_preds = []
        for fold in folds:
            # loading results
            one_fold_rmse = []
            preds = []
            for model_name in model_names:
                results = torch.load(os.path.join(path, f'Brand{brand}_Fold{fold}_train_{0.05}_{model_name}_vali_losses.pkl'))
                # init models list
                if brand == brands[0] and fold == folds[0]:
                    for specific_model in results[0]: models.append(specific_model)
                    num_models.append(len(results[0]))
                # loading results
                for specific_result in results[1]: one_fold_rmse.append(specific_result)
                print(model_name, len(results))
                # exit()
                GT, y_pred = np.array(results[2]), np.array(results[3])
                # print(y_pred.shape)
                # print(GT.shape)
                preds.append(y_pred.reshape(y_pred.shape[0], -1))
                #end for model_name
            # appending one fold full result
            folds_rmse.append(one_fold_rmse)
            if brand == brands[0] and fold == folds[0]: models = np.array(models) # converting models to np array
            # updating GT, preds and brand_preds
            fold_GTs.append(GT)
            one_fold_preds = np.concatenate(preds, axis = 0)
            fold_preds.append(one_fold_preds)
            print(one_fold_preds.shape) # (4000, 24) for fold 1, 4000 models, 24 predicted value
            #end for fold
        brand_GTs = np.concatenate(fold_GTs)
        brand_preds = np.concatenate(fold_preds, axis = 1)
        print(brand_GTs.shape)
        print(brand_preds.shape)
        domain_GTs.append(brand_GTs)
        domain_preds.append(brand_preds)
        # print(brand_y, brand_y.shape)
        # print(brand_preds)
        # print(brand_model_preds.shape)
        avg_fold_rmse = np.mean(folds_rmse, axis = 0)
        folds_rmse.append(avg_fold_rmse), brands_rmse.append(avg_fold_rmse)
        df = pd.DataFrame(folds_rmse, columns = models) # tdo
        df.insert(loc = 0, column = "Folds", value = [fold + 1 for fold in folds] + ['avg'])
        df.to_csv(os.path.join(output_csv_path, f'Capacity_Brand_{brand}_baseline_results_5fold.csv'), encoding = "utf-8-sig", index = False)
        #end for brand        
    avg_rmse_all = np.mean(brands_rmse, axis = 0)
    brands_rmse.append(avg_rmse_all)
    df = pd.DataFrame(brands_rmse,  columns = models)
    df.insert(loc = 0, column = "Battery Brands", value = [brand for brand in brands] + ['avg'])
    df.to_csv(os.path.join(output_csv_path, f'Capacity_{domain}_baseline_results.csv'), encoding = "utf-8-sig", index = False)
    # exporting latex
    # print(num_models)
    lower_bound = []
    upper_bound = []
    # getting indcies
    for i, num in enumerate(num_models):
        if i == 0:
            lower_bound.append(0)
            upper_bound.append(num)
        else:
            lower_bound.append(upper_bound[i-1])
            upper_bound.append(upper_bound[i-1] + num)
    # getting latex 
    print(lower_bound, upper_bound)
    domain_indices = []
    domain_values = []
    topk = 1
    for i in range(len(lower_bound)):
        temp_avg_rmse = avg_rmse_all[lower_bound[i]:upper_bound[i]]
        indices = np.arange(0, topk)
        values = np.array(temp_avg_rmse[:topk])
        for j in range(topk, temp_avg_rmse.shape[0]):
            value = temp_avg_rmse[j]
            for k in range(0, topk):
                if values[k] > value:
                    old_values = values[k]
                    old_index = indices[k]
                    values[k] = value
                    indices[k] = j
                    if topk != 1:
                        max_index = np.argmax(values)
                        values[max_index] = old_values
                        indices[max_index] = old_index
                    break
        indices += lower_bound[i]
        domain_indices.append(indices)
        domain_values.append(values)
    domain_indices = np.array(domain_indices).reshape(-1)
    domain_values = np.array(domain_values).reshape(-1)
    print(domain_indices, domain_values)
    print("===========================")

    latex_model_names = models[domain_indices]
    latex_values = np.array(brands_rmse)[:, domain_indices]

    latex_columns = []
    for model in model_names:
        for i in range(topk):
            latex_columns.append(f"{model} {i+1}")

    latex_df = pd.DataFrame(data = latex_values, columns = latex_columns)
    latex_df.insert(loc = 0, column = "Battery Brands", value = [brand for brand in brands] + ['avg'])
    latex_topk_results = latex_df.to_latex(index = False, float_format = '%.4f', caption = f'Capacity {domain} Baseline results')
    # print(latex_df)
    print(latex_topk_results)
    print(latex_model_names)
    print("=========================")
    # print(domain_indicies)
    # exit()
    if idx == 3:
        # storing best results for visualization
        for i, preds_val in enumerate(domain_preds):
            curr_brand = brands[i]
            curr_GT = domain_GTs[i]
            curr_preds = preds_val[domain_indices, :]
            torch.save((curr_GT, curr_preds),
                    os.path.join(output_csv_path, f'Best_Capacity_{curr_brand}_train_{0.05}_prediction.pkl'))