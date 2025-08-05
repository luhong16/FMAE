
import os
import matplotlib.pyplot as plt

def file_to_string(filename):
    with open(filename, 'r') as file:
        return file.read()

import numpy as np
import argparse
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="logs/RUL/max_min_volt_temp_cyclegap20/799_pretrain_vit_half_patch16/epoch100_blr1e-2_bsz32_ld0.5_wd0.005_dp0.0/s5")
    
    args = parser.parse_args()
    args.path = args.path + r"/f{fold_num}_b{brand_num}.txt"

    brand_num_list = [10, 12, 13]
    dataset_name = ["MIT1", "MIT2", "KIT"]
    result = []
    for b, n in zip(brand_num_list, dataset_name):
        rmse_list = []
        print(f"{n} RMSE: ", end=' ')
        for f in range(5):
            
            specify_name = args.path.format(fold_num=f,brand_num=b)
            res = file_to_string(specify_name)
            if "Training time" in res:
                for line in reversed(res.split("\n")):
                    if "Min test cell level RMSE:" in line:
                        rmse = float(line.split()[-1])
                        rmse_list.append(rmse)
                        break
            else:
                rmse = None
            
            print(rmse, end=' ')
        print("Average:", np.mean(rmse_list))
        result.append(np.mean(rmse_list))
    print('Overall average: ', np.mean(result))
    