
import os

def file_to_string(filename):
    with open(filename, 'r') as file:
        return file.read()

import numpy as np
import argparse
if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="logs/IR/max_min_volt_temp/799_pretrain_vit_tiny_patch16_depth3/epoch20_blr8e-3_bsz32_ld0.65_wd0.05_dp0.1/s5")
    
    args = parser.parse_args()
    args.path = args.path + r"/f{fold_num}_b{brand_num}.txt"

    brand_num_list = [10]
    dataset_name = ["MIT1"]
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
                        rmse = float(line.split()[-1]) * 1000.
                        rmse_list.append(rmse)
                        break
            else:
                rmse = None
            
            print(rmse, end=' ')
        print("Average:", np.mean(rmse_list))
        result.append(np.mean(rmse_list))
    print('Overall average: ', np.mean(result))