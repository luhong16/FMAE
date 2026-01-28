import sys
sys.path.append('./baselines')
import pandas as pd
import os
import numpy as np
import torch
from main_baseline_utils import *
from sklearn import metrics
from sklearn.metrics import auc

path = "./baselines/results/anomaly/diff_models_results"
output_csv_path = "./baselines/results/capacity/output_csv"
os.makedirs(output_csv_path, exist_ok = True)

# pkls = os.listdir(path)
# print(pkls, len(pkls))

brands = [1, 2, 4, 5, 6]
folds = [0, 1, 2, 3, 4]

# snippet selection percentage
# percentages = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
#                 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1] # for complete hyperparameter search, enable here and line70 of Variation_evaluation.py
percentages = [0.05]

matrices = None
models = []
brands_auroc = []
for brand in brands:
    folds_auroc = []
    for fold in folds:
        # loading results
        results = torch.load(os.path.join(path, f'Brand{brand}_Fold{fold}_vali_results.pkl'))
        if matrices is None: 
            matrices = np.array(results[0]) # (# matrices, matrix_weights)
            for matrix in matrices:
                for p in percentages:
                    models.append(f'VE_{matrix[0]}_{matrix[1]}_{matrix[2]}_{matrix[3]}_{matrix[4]}_p{p}')
            models = np.array(models)
        
        y_scores_raw = np.array(results[1])
        y_scores = y_scores_raw.reshape(y_scores_raw.shape[0], -1)
        y_true = np.array(results[2])
        print(models.shape, y_scores.shape, y_true.shape)
        # getting aurocs
        aurocs = []
        for i in range(0, y_scores.shape[1]):
            temp_score = y_scores[:, i].reshape(-1)
            # print(temp_score)
            fpr, tpr, thresholds = metrics.roc_curve(y_true, temp_score, pos_label = '1')
            auroc = auc(fpr, tpr)
            aurocs.append(auroc)
        # appending all models fold auroc
        folds_auroc.append(aurocs)
        #end fold
    avg_auroc = np.mean(folds_auroc, axis = 0)
    folds_auroc.append(avg_auroc), brands_auroc.append(avg_auroc)
    df = pd.DataFrame(folds_auroc, columns = models)
    df.insert(loc = 0, column = "Folds", value = [fold + 1 for fold in folds] + ['avg'])
    df.to_csv(os.path.join(output_csv_path, f'Anomaly_Brand_{brand}_baseline_results_5fold.csv'), encoding = "utf-8-sig", index = False)
    #end brand
avg_auroc_all = np.mean(brands_auroc, axis = 0)
brands_auroc.append(avg_auroc_all)
df = pd.DataFrame(brands_auroc,  columns = models)
df.insert(loc = 0, column = "Battery Brands", value = [brand for brand in brands] + ['avg'])
df.to_csv(os.path.join(output_csv_path, f'Anomaly_baseline_results.csv'), encoding = "utf-8-sig", index = False)

topk = 1
overall_aurocs = np.array(brands_auroc)
print(overall_aurocs.shape)
# exit()
indices = np.arange(0, topk)
values = np.array(avg_auroc_all[:topk])

# print(avg_auroc_all, indices, values)
for i in range(topk, avg_auroc_all.shape[0]):
    value = avg_auroc_all[i]
    for j in range(0, topk):
        if values[j] <= value:
            old_values = values[j]
            old_index = indices[j]
            values[j] = value
            indices[j] = i
            if topk != 1:
                min_index = np.argmin(values)
                values[min_index] = old_values
                indices[min_index] = old_index
            break

print(indices, values)
print("=============")
latex_model_names = models[indices]
latex_values = np.array(brands_auroc)[:, indices]

latex_df = pd.DataFrame(data = latex_values, columns = [f'V_E {i+1}' for i in range(topk)])
latex_df.insert(loc = 0, column = "Battery Brands", value = [brand for brand in brands] + ['avg'])
latex_topk_results = latex_df.to_latex(index = False, float_format = '%.4f', caption = 'Anomaly Baseline results')
# print(latex_df)
print(latex_topk_results)
print(latex_model_names)


# storing best results for visualization
for brand in brands:
    for fold in folds:
        results = torch.load(os.path.join(path, f'Brand{brand}_Fold{fold}_vali_results.pkl'))
        y_scores_raw = np.array(results[1])
        # print(y_scores_raw.shape)
        y_scores = y_scores_raw.reshape(y_scores_raw.shape[0], -1)
        y_true = np.array(results[2])
        # print(y_scores.shape)
        # print(indices, y_scores[:, indices])
        torch.save((y_true, y_scores[:,indices].reshape(-1)), os.path.join(output_csv_path, f'Best_b{brand}_f{fold}_V_E_Results.pkl'))

