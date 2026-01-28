import random
import numpy as np
import torch
from main_baseline_utils import *
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as RFRegressor
from xgboost import XGBRegressor as xgb
from sklearn import metrics
from sklearn.metrics import auc
import argparse

def set_args():
    parser = argparse.ArgumentParser(description='Variation Evaluation')
    # self.score.score_V, self.score.score_T, self.score.score_R, self.score.score_Q, self.score.score_E [0.2286, 0.1242, 0.1774, 0.0999, 0.3699]
    # for snippet, T and R are changing, the rest remain perfect scores
    parser.add_argument('--Brand', type = int, default = 4)
    parser.add_argument('--Fold', type = int, default = 0)
    parser.add_argument('--score_V_weight', type = float, default = 0.2286)
    parser.add_argument('--score_T_weight', type = float, default = 0.1242)
    parser.add_argument('--score_R_weight', type = float, default = 0.1774)
    parser.add_argument('--score_Q_weight', type = float, default = 0.0999)
    parser.add_argument('--score_E_weight', type = float, default = 0.3699)
    parser.add_argument('--score_percentage', type = float, default = 0.1)
    args = parser.parse_args()
    return args 

def get_car_dict(X, y, meta_data_list):
    # getting variation evaluation scores 
    car_labels = []
    car_dict = {}
    for i in range(X.shape[0]):
        meta_data = meta_data_list[i]
        if meta_data['car'] not in car_dict: 
            car_dict[meta_data['car']] = [i]
            car_labels.append(y[i])
        else: car_dict[meta_data['car']].append(i)
    # print(len(car_dict), list(car_dict.keys()))
    return car_dict, car_labels

def Variation_evaluation(matrices, X, car_dict):
    # getting variation evaluation scores 
    car_scores = []
    for car, snippet_indices in car_dict.items():
        snippets_scores = []
        for idx in snippet_indices:
            features = X[idx, :, :] # (128,8) ["Volt", 'Current', 'SOC', 'Max_singla_volt', 'Min_single_volt', 'Max_temp', 'Min_temp', 'timestamp']
            # Calculatin consistency scores
            columns = ["Volt", 'Current', 'SOC', 'Max_singla_volt', 'Min_single_volt', 'Max_temp', 'Min_temp', 'timestamp']
            act = Battery_Consistency(time_loc=7, 
                                        v_loc_str = 3, 
                                        v_loc_end = 4, 
                                        temp_loc_str = 5,
                                        temp_loc_end = 6,
                                        current_loc= 1,
                                        soc_loc= 2,
                                        mileage_loc= None)
            data = act.dataprocess(features, columns)
            score, score_T, score_R, score_Q, score_E, score_V, final_scores = act.scoreStatistics(data, matrices = matrices)
            # print('score_T：',score_T, 'score_R：',score_R, 'score_Q：',score_Q, 'score_E：',score_E, 'score_V：',score_V, 'final_score：',final_scores)
            snippets_scores.append(final_scores)
            # end for pkl_all in pkls:
        V_Es_single_car_snippet_scores = np.array(snippets_scores) # shape = (# snippet, # matrix)
        single_car_scores = []
        for i in range(0, V_Es_single_car_snippet_scores.shape[1]):
            temp_scores_snippet = V_Es_single_car_snippet_scores[:, i]
            print(temp_scores_snippet)
            temp_scores_snippet = np.sort(temp_scores_snippet)[::-1] # reversing
            print(temp_scores_snippet)
            single_car_score = [np.mean(temp_scores_snippet[:max(1, int(temp_scores_snippet.shape[0] * percent))]) \
                                for percent in [0.05]] # searched snippet percentages = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                                                       # 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
            single_car_scores.append(single_car_score)
        
        car_scores.append(single_car_scores) 
        # snippets_scores.sort(reverse = True)
        # car_score = np.mean(snippets_scores[:max(1, int(len(snippets_scores) * score_percentage))])
        # print("car_score: ", car_score)
        # car_scores.append(car_score)
        # end for car, snippet_indices in car_dict.items()
    car_scores = np.array(car_scores) # shape = (# cars, # matrix, # percent)
    print(car_scores, car_scores.shape)
    return car_scores
def get_matrices():
    # self.score.score_V, self.score.score_T, self.score.score_R, self.score.score_Q, self.score.score_E
    # for snippet, T and R are changing, the rest remain perfect scores
    # original [0.2286, 0.1242, 0.1774, 0.0999, 0.3699]
    matrices = []
    
    matrices.append([0.2286, 0.4242, 0.1774, 0.0999, 0.0699]) # best hyperparameters
    ## searched hyperparameters
    # matrices.append([0.2286, 0.1242, 0.1774, 0.0999, 0.3699])
    # matrices.append([0.2286, 0.2242, 0.1774, 0.0999, 0.2699])
    # matrices.append([0.2286, 0.3242, 0.1774, 0.0999, 0.1699])
    # matrices.append([0.2286, 0.1242, 0.2774, 0.0999, 0.2699])
    # matrices.append([0.2286, 0.1242, 0.3774, 0.0999, 0.1699])
    # matrices.append([0.2286, 0.1242, 0.4774, 0.0999, 0.0699])
    # matrices.append([0.2286, 0.1742, 0.2274, 0.0999, 0.2699])
    # matrices.append([0.2286, 0.2242, 0.2774, 0.0999, 0.1699])
    # matrices.append([0.2286, 0.2742, 0.3274, 0.0999, 0.0699])
    
    return matrices

def main():
    # init
    args = set_args()
    results_path = f"./baselines/results/anomaly"
    scores_path = f"./baselines/results/anomaly/diff_models_results"
    os.makedirs(results_path, exist_ok = True), os.makedirs(scores_path, exist_ok = True)
    print(f"Evaluating Anomaly baseline brand_{args.Brand} fold_{args.Fold} ...")
    
    # more inits
    matrices = get_matrices()
    X, y, meta_data_list = torch.load(f'./baselines/results/features/anomaly_brand{args.Brand}_fold{args.Fold}_vali_dataKept_1.0.pkl')
    print(X.shape, y.shape)
    car_dict, y_true = get_car_dict(X = X, y = y, meta_data_list = meta_data_list)
    # getting scores
    y_score = Variation_evaluation(matrices = matrices, X = X, car_dict = car_dict)# shape = (# cars, # matrix, # percent)
    # todo
    # fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label = '1')
    # auroc = auc(fpr, tpr)
    torch.save((matrices, y_score, y_true), os.path.join(scores_path, f'Brand{args.Brand}_Fold{args.Fold}_vali_results.pkl'))


if __name__ == '__main__':
    main()