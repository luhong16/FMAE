import sys
sys.path.append('./baselines')
import pandas as pd
import os
import numpy as np
import torch
from main_baseline_utils import *
from sklearn import metrics
from sklearn.metrics import auc

brands_list = [[10, 12, 13]]
brands_domains = ['Lab']
folds = [0, 1, 2, 3, 4]

points_list = [128, 256, 1280, 2560, 12800] 
paths_list = ["./baselines/results/RUL_w_log/diff_models_losses"]
output_csv_path = "./baselines/results/RUL_w_log/output_csv"
os.makedirs(output_csv_path, exist_ok = True)


nmi_five_folds_discharge = [[],
                    [],
                    []]

nmi_five_folds_discharge_GT = [[],
                    [],
                    []]

nmi_five_folds_variance= [[],
                            [],
                            []]

nmi_five_folds_variance_GT = [[],
                            [],
                            []]


for idx, brands in enumerate(brands_list):
    domain = brands_domains[idx]
    has_models = None
    model_names = ['ElasticLinear_discharge', 'ElasticLinear_variance']
    models = []
    brands_rmse = []
    num_models = []

    domain_GTs = []
    domain_preds = []
    domain_y_train = []


    num_of_models = 0
    for i,brand in enumerate(brands):
        folds_rmse = []
        fold_GTs = []
        fold_preds = []
        fold_y_train = []

        for fold in folds:
            # loading results
            one_fold_rmse = []
            preds = []
            GTs = []
            y_trains = []

            for model_name in model_names:
                count = 0
                for path in paths_list:
                    for point in points_list:
                        results = torch.load(os.path.join(path, f'Brand{brand}_Fold{fold}_points{point}_{model_name}_w_log_vali_losses.pkl'))
                        # init models list
                        if brand == brands[0] and fold == folds[0]:
                            for specific_model in results[0]: 
                                if 'wo' in path: models.append('wo_norm_' + specific_model + f'_points_{point}')
                                else: models.append('w_norm_' + specific_model + f'_points_{point}')
                            count += len(results[0])
                        # loading results
                        for specific_result in results[1]: one_fold_rmse.append(specific_result)
                        print(model_name, brand, fold, len(results))
                        # exit()

                        GT, y_pred, y_train = np.array(results[2]), np.array(results[3]) , np.array(results[4])
                        # print(GT)
                        #end for point
                        # print(y_pred.shape)
                        # exit()
                        GTs.append(GT)
                        preds.append(y_pred)
                        y_trains.append(y_train)
                        # preds.append(y_pred.reshape(5, -1))
                        # print(y_pred.reshape(-1))
                    #end for path
                if brand == brands[0] and fold == folds[0]: num_models.append(count) # appending model count 
                #end for model_name
            # appending one fold full result
            folds_rmse.append(one_fold_rmse)
            # print(folds_rmse, len(folds_rmse))
            # exit()
            if brand == brands[0] and fold == folds[0]: models = np.array(models) # converting models to np array
            # updating GT, preds and brand_preds
            one_fold_GTs = np.concatenate(GTs, axis = 0)
            one_fold_preds = np.concatenate(preds, axis = 0)
            one_fold_y_trains = np.concatenate(y_trains, axis = 0)


            fold_GTs.append(one_fold_GTs)
            fold_preds.append(one_fold_preds)
            fold_y_train.append(one_fold_y_trains)

            print(one_fold_preds.shape) # (4000, 24) for fold 1, 4000 models, 24 predicted value
            num_of_models = one_fold_preds.shape[0]
            #end for fold
        brand_GTs = np.concatenate(fold_GTs,axis = 1)
        brand_preds = np.concatenate(fold_preds, axis = 1)
        brand_y_train = np.concatenate(fold_y_train, axis = 1)
        print(brand_GTs.shape)
        print(brand_preds.shape)
        domain_GTs.append(brand_GTs)
        domain_preds.append(brand_preds)
        domain_y_train.append(brand_y_train)
        # print(brand_y, brand_y.shape)
        # print(brand_preds)
        # print(brand_model_preds.shape)
        # print(folds_rmse[0][407])
        # exit()
        temp1 = []
        temp2 = []
        temp3 = []
        temp4 = []
        for fold in folds:
            temp1.append(folds_rmse[fold][51])
            temp2.append(folds_rmse[fold][79])


        nmi_five_folds_discharge[i].append(temp1) #
        nmi_five_folds_variance[i].append(temp2) 

        avg_fold_rmse = np.mean(folds_rmse, axis = 0)
        folds_rmse.append(avg_fold_rmse), brands_rmse.append(avg_fold_rmse)
        df = pd.DataFrame(folds_rmse, columns = models) # tdo
        df.insert(loc = 0, column = "Folds", value = [fold + 1 for fold in folds] + ['avg'])
        df.to_csv(os.path.join(output_csv_path, f'RUL_Brand_{brand}_baseline_results_5fold.csv'), encoding = "utf-8-sig", index = False)
        #end for brand        
    avg_rmse_all = np.mean(brands_rmse, axis = 0)
    # print(len(avg_rmse_all), avg_rmse_all)
    # exit()
    brands_rmse.append(avg_rmse_all)
    df = pd.DataFrame(brands_rmse,  columns = models)
    # print(df)
    # exit()
    df.insert(loc = 0, column = "Battery Brands", value = [brand for brand in brands] + ['avg'])
    df.to_csv(os.path.join(output_csv_path, f'RUL_{domain}_baseline_results.csv'), encoding = "utf-8-sig", index = False)
    # exporting latex
    # print(num_models)
    lower_bound = []
    upper_bound = []
    # getting indcies
    for i, num in enumerate(num_models):
        # print(num)
        # exit()
        if i == 0:
            lower_bound.append(0)
            upper_bound.append(num)
        else:
            lower_bound.append(upper_bound[i-1])
            upper_bound.append(upper_bound[i-1] + num)
    # getting latex 
    print(lower_bound, upper_bound)
    print(avg_rmse_all)
    # exit()
    domain_indices = []
    domain_values = []
    topk = 1
    for i in range(len(lower_bound)):
        temp_avg_rmse = avg_rmse_all[lower_bound[i]:upper_bound[i]]
        print(min(temp_avg_rmse))
        min_val = min(temp_avg_rmse)
        indices = [i for i, x in enumerate(temp_avg_rmse) if x == min_val][0]
        # exit()
        # indices = np.arange(0, topk)
        # values = np.array(temp_avg_rmse[:topk])
        # for j in range(topk, temp_avg_rmse.shape[0]):
        #     value = temp_avg_rmse[j]
        #     for k in range(0, topk):
        #         if values[k] > value:
        #             old_values = values[k]
        #             old_index = indices[k]
        #             values[k] = value
        #             indices[k] = j
        #             max_index = np.argmax(values)
        #             values[max_index] = old_values
        #             indices[max_index] = old_index
        #             break
        indices += lower_bound[i]
        domain_indices.append(indices)
        domain_values.append(min_val)
    domain_indices = np.array(domain_indices).flatten()
    domain_values = np.array(domain_values).flatten()
    print(domain_indices, domain_values)
    print("===========================")
    latex_model_names = models[domain_indices.astype(int)]
    latex_values = np.array(brands_rmse)[:, domain_indices.astype(int)]

    latex_columns = []
    for model in model_names:
        if 'discharge' in model: model = 'discharge'
        elif 'variance' in model: model = 'variance'
        else: raise NotImplementedError
        for i in range(topk):
            latex_columns.append(f"{model} {i+1}")

    latex_df = pd.DataFrame(data = latex_values, columns = latex_columns)
    latex_df.insert(loc = 0, column = "Battery Brands", value = [brand for brand in brands] + ['avg'])
    latex_topk_results = latex_df.to_latex(index = False, float_format = '%.4f', caption = f'RUL {domain} Baseline results')
    # print(latex_df)
    print(latex_topk_results)
    print(latex_model_names)
    print(domain_indices)
    # exit()

    print('==================')
    torch.save(nmi_five_folds_discharge,
                os.path.join(output_csv_path, f'RUL_five_folds_discharge_rmse.pkl'))
    torch.save(nmi_five_folds_variance,
                os.path.join(output_csv_path, f'RUL_five_folds_variance_rmse.pkl'))
    print(np.mean([np.mean(f) for f in nmi_five_folds_discharge])) #
    print(np.mean([np.mean(f) for f in nmi_five_folds_variance])) #
    # exit()
    # exit()
    # exit()

    # print(len(domain_preds[0]))
    for i, preds_val in enumerate(domain_preds):
        curr_brand = brands[i]
        # print(curr_brand)
        GT_val = domain_GTs[i]
        # print(domain_indices[0])
        # exit()
        # print(len(curr_preds))
        min_vals = 10000
        min_folds = None
        # for j in range(0, len(preds_val[:num_of_models])):
        #     curr_preds = preds_val[j, :]
        #     curr_GT = GT_val[j, :]
        #     f1 = np.mean((curr_GT[0:24] - curr_preds[0:24])**2)
        #     f2 = np.mean((curr_GT[24:49] - curr_preds[24:49])**2)
        #     f3 = np.mean((curr_GT[49:74] - curr_preds[49:74])**2)
        #     f4 = np.mean((curr_GT[74:99] - curr_preds[74:99])**2)
        #     f5 = np.mean((curr_GT[99:125] - curr_preds[99:125])**2)
        #     # print(f1)
        #     avgggg = np.sqrt([f1,f2,f3,f4,f5])
        #     print(avgggg, np.mean(avgggg))
        #     # print(np.mean(avgggg))
        #     if min_vals > np.mean(avgggg):
        #         min_vals = np.mean(avgggg)
        #         min_folds = avgggg
        
        # print(min_vals)
        # print(min_folds)
        # print(curr_preds)
        # print(models[domain_indices[0]])
        # exit()
        curr_preds = preds_val[domain_indices[0], :]
        curr_GT = domain_GTs[i][domain_indices[0], :]
        curr_y_train = domain_y_train[i][domain_indices[0], :]
        torch.save((curr_GT, curr_preds, curr_y_train),
                os.path.join(output_csv_path, f'Best_RUL_{curr_brand}__discharge_prediction.pkl'))
        curr_preds = preds_val[domain_indices[1], :]
        curr_GT = domain_GTs[i][domain_indices[1], :]
        curr_y_train = domain_y_train[i][domain_indices[1], :]
        torch.save((curr_GT, curr_preds, curr_y_train),
                os.path.join(output_csv_path, f'Best_RUL_{curr_brand}__variance_prediction.pkl'))

    exit()