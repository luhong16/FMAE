import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn import metrics
from sklearn.metrics import auc
import torch


def read_logs(f_name):
    with open (f_name, 'r') as f:
        # records
        labels_list = None
        best_scores_list = None
        best_vali_loss = 0
        # flags
        test_start_flag = False
        test_end_flag = False
        
        for line in f.readlines():
            # determine whether to start reading logs
            if 'start(test)' in line: 
                test_start_flag = True
                temp_labels = []
                temp_scores = []
                continue
            # determing whether to stop reading logs
            if 'end(test)' in line:
                test_start_flag = False
                test_end_flag = True
                continue
            # determine whether to update best scores
            if test_end_flag and 'test_auroc' in line:
                temp_list = line.split(', ')
                vali_loss = None
                for s in temp_list:
                    if 'test_auroc' in s:
                        temp_s = s.split(' ')
                        # print(temp_s)
                        vali_loss = float(temp_s[1][10:])
                        # print(vali_loss)
                        # exit()
                        break
                # if vali_loss is best, updating score_list and label_list(redundant work but whatever)
                if vali_loss >= best_vali_loss:
                    best_vali_loss = vali_loss
                    labels_list = temp_labels
                    best_scores_list = temp_scores
                    # print("YEs")
                test_end_flag = False
                continue
            # reading logs
            if test_start_flag:
                temp_list = line.split(' ')
                temp_labels.append(temp_list[1])
                temp_scores.append(float(temp_list[3][:-1]))
                # print(temp_labels)
                # print(temp_scores)
                # exit()

            # print(line)
        # print(labels_list, best_scores_list)
        # exit()
    return labels_list, best_scores_list

def get_best_path(path_list, brand_list, fold_list):
    overall_aurocs = []
    for path in path_list:
        # brand_rmses = []
        model_aurocs = []
        for brand in brand_list:
            # fold_rmses = []
            fold_original_data = []
            for fold in fold_list:
                log_path = os.path.join(path, f'f{fold}_b{brand}.txt')
                labels_list, best_scores_list= read_logs(log_path)
                fpr, tpr, thresholds = metrics.roc_curve(labels_list, best_scores_list, pos_label = '1', drop_intermediate=False)
                fold_original_data.append((fpr, tpr))
            aucs = []
            for i, (fpr, tpr) in enumerate(fold_original_data):
                auc_score = auc(fpr, tpr)
                # print(f'Brand {brand} fold {i+1} auroc: {auc_score}')
                aucs.append(auc_score)
            # if 'iTransformer/epoch20_blr5e-4' in path or 'PatchTST/epoch20_blr1e-2' in path:
            #     print(f'{brand} {path} avg auroc: {np.mean(aucs):.5f}', aucs)
            model_aurocs.append(aucs)
        overall_aurocs.append(np.mean(model_aurocs))
        # print(brand_rmses)
        # overall_RMSEs.append(np.mean(brand_rmses))
    
    for idx, path in enumerate(path_list):
        print(path, overall_aurocs[idx])

    return path_list[np.argmax(overall_aurocs)]



brands = [1, 2, 4, 5, 6]
folds = [0, 1, 2, 3, 4]

patchtst_log_path = './logs/anomaly/no/PatchTST'
itransformer_log_path = './logs/anomaly/no/iTransformer'

patchTST_paths = [os.path.join(patchtst_log_path, f_name+'/s5') \
                  for f_name in os.listdir(patchtst_log_path)]
itransformer_paths = [os.path.join(itransformer_log_path, f_name+'/s5') \
                      for f_name in os.listdir(itransformer_log_path)]


patchTST_path = get_best_path(patchTST_paths, brand_list=brands, fold_list=folds)
itransformer_path = get_best_path(itransformer_paths, brand_list=brands, fold_list=folds)

