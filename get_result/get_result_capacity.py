
import os
import matplotlib.pyplot as plt

def file_to_string(filename):
    with open(filename, 'r') as file:
        return file.read()

import numpy as np
import argparse
if __name__ == "__main__":

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="logs/capacity/no/799_pretrain_vit_half_patch16/epoch20_blr5e-2_bsz32_ld0.5_wd0.005_dp0.0/s5")
    parser.add_argument("--type", type=str, default="EV")
    args = parser.parse_args()
    args.path = args.path + r"/f{fold_num}_b{brand_num}.txt"

    # Normalize to get SOH
    Normalize = [-1, 46.32390322689564, 45.28971041043597, 44.034358340136066, 26.538070600693814, 35.473890427318786, 22.28958766646174, 95.9036963]
        
    if args.type == 'EV':
        brand_num_list = [1,2,3,4,5,6]
        dataset_name = ["EV1", "EV2", "EV3", "EV4", "EV5", "EV6"]
    if args.type == 'lab':
        brand_num_list = [10, 11, 12, 13]
        dataset_name = ["MIT1", "THU", "MIT2", "KIT"]
    if args.type == 'BESS':
        brand_num_list = [7]
        dataset_name = ["BESS1"]
    if args.type == 'nc':
        brand_num_list = [14]
        dataset_name = [""]
        
    result = []
    for b, n in zip(brand_num_list, dataset_name):
        rmse_list = []
        print(f"{n} RMSE: ", end=' ')
        for f in range(5):
            
            specify_name = args.path.format(fold_num=f,brand_num=b)
            res = file_to_string(specify_name)
            if "Training time" in res:
                for line in reversed(res.split("\n")):
                    if "Min test set RMSE:" in line:
                        rmse = float(line.split()[-1])
                        if b <= 7:
                            rmse = rmse * 10000 / Normalize[b]
                        else:
                            rmse = rmse * 100.
                        rmse_list.append(rmse)
                        break
            else:
                rmse = None
            
            print(rmse, end=' ')
        print("Average:", np.mean(rmse_list))
        result.append(np.mean(rmse_list))
    print('Overall average: ', np.mean(result))