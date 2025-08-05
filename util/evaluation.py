import numpy as np
from sklearn import metrics
from sklearn.metrics import auc
import pandas as pd
from tqdm import tqdm
import os
import os
from sklearn.metrics import confusion_matrix

def snippet_evaluate(predict, target):

    cm = confusion_matrix(target.astype(int), predict.astype(int), labels=[0, 1])
    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tp = cm[1, 1]
    precision = tp / (tp + fp) if tp + fp != 0 else 0
    recall = tp / (tp + fn) if tp + fn != 0 else 0
    false_rate = fp / (tn + fp) if tn + fp != 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    return f1, recall, false_rate, precision, accuracy

def find_best_percent(result, granularity_all=1000):
    """
    find threshold
    :param result: sorted result
    :param granularity_all: granularity_all
    """
    max_percent = 0
    best_n = 1
    print("threshold tuning start:")
    for n in range(1, 100):
        head_n = n / granularity_all
        data_length = max(round(len(result) * head_n), 1)
        count_dist = count_entries(result.loc[:data_length - 1], 'label')
        try:
            percent = count_dist['1'] / (count_dist['0'] + count_dist['1'])
        except KeyError:
            print("can't find n%,take 1%")
            percent = 0.01
        if percent > max_percent:
            max_percent = percent
            best_n = n
    print("top %d / %s is the highest, %s" % (granularity_all, best_n, max_percent))
    return best_n, max_percent, granularity_all

def count_entries(df, col_name):
    """
    count
    """
    count_dist = {'0': 0, '1': 0}
    col = df[col_name]
    for entry in col:
        if str(int(entry)) in count_dist.keys():
            count_dist[str(int(entry))] = count_dist[str(int(entry))] + 1
        else:
            count_dist[str(int(entry))] = 1
    return count_dist

def find_best_result(threshold_n, result, ind_car_num_list, ood_car_num_list, h = None):
    """
    find_best_result
    :param threshold_n: threshold
    :param result: sorted result
    :param dataframe_std: label
    """
    best_result, best_h, best_re, best_fa, best_f1, best_precision = None, 0, 0, 0, 0, 0
    best_auroc = 0
    if h is not None:
        train_result = charge_to_car(threshold_n, result, head_n=h)
        f1, recall, false_rate, precision, accuracy, auroc = calculate_stat(train_result, ind_car_num_list, ood_car_num_list)
        best_f1 = f1
        best_h = h
        best_re = recall
        best_fa = false_rate
        best_result = train_result
        best_auroc = auroc
    else:
        for h in range(50, 1000, 50):
            train_result = charge_to_car(threshold_n, result, head_n=h)
            f1, recall, false_rate, precision, accuracy, auroc = calculate_stat(train_result, ind_car_num_list, ood_car_num_list)
            if auroc >= best_auroc:
                best_f1 = f1
                best_h = h
                best_re = recall
                best_fa = false_rate
                best_result = train_result
                best_auroc = auroc
    return best_result, best_h, best_re, best_fa, best_f1, best_auroc

def charge_to_car(threshold_n, rec_result, head_n=92):
    """
    mapping from charge to car
    :param threshold_n: threshold
    :param rec_result: sorted result
    :param head_n: top %n
    :param gran: granularity
    """
    gran = 1000
    result = []
    for grp in rec_result.groupby('car'):
        temp = grp[1].values[:, -1].astype(float)
        idx = max(round(head_n / gran * len(temp)), 1)
        error = np.mean(temp[:idx])
        result.append([grp[0], int(error > threshold_n), error, threshold_n, grp[1].values[0, -2]])
    return pd.DataFrame(result, columns=['car', 'predict', 'error', 'threshold_n', 'label'])

def calculate_stat(dataframe, ind_car_num_list, ood_car_num_list):
    """
    calculated statistics
    :param dataframe_std:
    :param dataframe:
    :return:
    """


    _label = []
    for each_car, each_label in zip(dataframe['car'], dataframe['label']):
        if int(each_car) in ind_car_num_list:
            _label.append(0)
            assert each_label == 0
        elif int(each_car) in ood_car_num_list:
            _label.append(1)
            assert each_label == 1
        else:
            print('error', each_car)

    fpr, tpr, thresholds = metrics.roc_curve(_label, list(dataframe['error']), pos_label=1)
    auroc = auc(fpr, tpr)


    cm = confusion_matrix(dataframe['label'].astype(int), dataframe['predict'].astype(int))
    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tp = cm[1, 1]
    precision = tp / (tp + fp) if tp + fp != 0 else 0
    recall = tp / (tp + fn) if tp + fn != 0 else 0
    false_rate = fp / (tn + fp) if tn + fp != 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    return f1, recall, false_rate, precision, accuracy, auroc


def evaluation(train_res_csv, test_res_csv, brand_num, h = None, dir_prefix = 'five_fold_utils'):

    assert h is not None
    ind_ood_car_dict = np.load(os.path.join(dir_prefix, f'five_fold_utils_six_brand_all/ind_odd_dict{brand_num}.npz.npy'), allow_pickle=True).item()
    ind_car_num_list = ind_ood_car_dict['ind_sorted']
    ood_car_num_list = ind_ood_car_dict['ood_sorted'] 
    all_car_num_list = set(ind_car_num_list + ood_car_num_list)

    # best_result.to_csv(os.path.join(self.args.result_path, "train_res.csv")),
    
    rec_sorted_index = np.argsort(-test_res_csv[:, 2].astype(float))
    res = [test_res_csv[i][[1, 0, 2]] for i in rec_sorted_index]
    result = pd.DataFrame(res, columns=['car', 'label', 'rec_error'])
    result['car'] = result['car'].astype("int").astype("str")
    # result.to_csv(os.path.join(self.args.result_path, "test_segment_scores.csv"))
    test_result = charge_to_car(0, result, head_n=h)
    
    _score = list(test_result['error'])
    _label = []
    for each_car, each_label in zip(test_result['car'], test_result['label']):
        if int(each_car) in ind_car_num_list:
            _label.append(0)
            assert each_label == 0
        elif int(each_car) in ood_car_num_list:
            _label.append(1)
            assert each_label == 1
        else:
            raise NotImplementedError
    
    print('---start(test)---')
    score_sorted_index = np.argsort(np.array(_score).astype(float))
    for i in score_sorted_index:
        print(_label[i], test_result['car'][i], _score[i])
    print('---end(test)---')
    print('len(_score)', len(_score))
    fpr, tpr, thresholds = metrics.roc_curve(_label, _score, pos_label=1)
    AUC = auc(fpr, tpr)
    print('AUC', AUC)
    
    return AUC
