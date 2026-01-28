import os
import numpy as np



def get_best_path(path_list, brand_list, fold_list, normalizer, domain='EV'):

    record = [[] for _ in range(len(path_list))]
    overall_RMSEs = []
    for path_idx, path in enumerate(path_list):
        brand_rmses = []
        for brand in brand_list:
            fold_rmses = []
            for fold in fold_list:
                results = np.load(os.path.join(path, f'f{fold}_b{brand}/res.npy'))
                GTs = results[:,0] * 100 / normalizer[brand] * 100
                preds = results[:,2] * 100 / normalizer[brand] * 100
                fold_rmses.append(np.sqrt(np.mean((GTs - preds)**2)))
            brand_rmses.append(np.mean(fold_rmses))
        # print(brand_rmses)
        record[path_idx] = brand_rmses
        overall_RMSEs.append(np.mean(brand_rmses))

    for idx, path in enumerate(path_list):
        print(domain, path, overall_RMSEs[idx])
    
    print(f'Lowest rmse: {path_list[np.argmin(overall_RMSEs)]}, {record[np.argmin(overall_RMSEs)]}, {np.mean(record[np.argmin(overall_RMSEs)])}')

    return path_list[np.argmin(overall_RMSEs)]

normalizer = [0,46.32390322689564, 45.28971041043597, 44.034358340136066, 26.538070600693814, 35.473890427318786, 22.28958766646174
              , 95.9036963,0,0, 100, 100, 100, 100, 100]

normalizer2 = [0,46.32390322689564, 45.28971041043597, 44.034358340136066, 26.538070600693814, 35.473890427318786, 22.28958766646174
              , 95.9036963,0,0, 100, 100, 100, 100, 100]

patchTST_EV_BESS_res_path = './logs/capacity/no/PatchTST'
itransformer_EV_BESS_res_path = './logs/capacity/no/iTransformer'
patchTST_lab_res_path = './logs/capacity/use_volt_current_soc/PatchTST'
itransformer_lab_res_path = './logs/capacity/use_volt_current_soc/iTransformer'
# EV
print('EV results: ==========================')
patchTST_EV_paths = [os.path.join(patchTST_EV_BESS_res_path , f_name+'/s5') \
                  for f_name in os.listdir(patchTST_EV_BESS_res_path )]
itransformer_EV_paths = [os.path.join(itransformer_EV_BESS_res_path, f_name+'/s5') \
                      for f_name in os.listdir(itransformer_EV_BESS_res_path)]

patchTST_EV_path = get_best_path(patchTST_EV_paths, brand_list=[1,2,3,4,5,6], fold_list=[0,1,2,3,4], normalizer= normalizer, domain='EV')
itransformer_EV_path = get_best_path(itransformer_EV_paths, brand_list=[1,2,3,4,5,6], fold_list=[0,1,2,3,4], normalizer= normalizer, domain='EV')
# Lab

print('Lab results: ==========================')
patchTST_lab_paths = [os.path.join(patchTST_lab_res_path, f_name+'/s5') \
                  for f_name in os.listdir(patchTST_lab_res_path)]
itransformer_lab_paths = [os.path.join(itransformer_lab_res_path , f_name+'/s5') \
                      for f_name in os.listdir(itransformer_lab_res_path )]

patchTST_lab_path = get_best_path(patchTST_lab_paths, brand_list=[10, 11, 12, 13], fold_list=[0,1,2,3,4], normalizer= normalizer, domain='lab')
itransformer_lab_path = get_best_path(itransformer_lab_paths, brand_list=[10, 11, 12, 13], fold_list=[0,1,2,3,4], normalizer= normalizer, domain='lab')

# BESS
print('BESS results: ==========================')
patchTST_BESS_paths = [os.path.join(patchTST_EV_BESS_res_path , f_name+'/s5') \
                  for f_name in os.listdir(patchTST_EV_BESS_res_path )]
itransformer_BESS_paths = [os.path.join(itransformer_EV_BESS_res_path, f_name+'/s5') \
                      for f_name in os.listdir(itransformer_EV_BESS_res_path)]

patchTST_BESS_path = get_best_path(patchTST_BESS_paths, brand_list=[7], fold_list=[0,1,2,3,4], normalizer= normalizer, domain='BESS')
itransformer_BESS_path = get_best_path(itransformer_BESS_paths, brand_list=[7], fold_list=[0,1,2,3,4], normalizer= normalizer, domain='BESS')


