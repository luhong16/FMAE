
import os
import matplotlib.pyplot as plt

def file_to_string(filename):
    with open(filename, 'r') as file:
        return file.read()

import numpy as np
import argparse
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="logs/anomaly/no/799_pretrain_vit_half_patch16/epoch10_blr5e-3_bsz32_ld0.65_wd0.01_dp0.1/s5")
    args = parser.parse_args()
    args.path = args.path + r"/f{fold_num}_b{brand_num}.txt"

    dataset_name = ["EV1", "EV2", "EV4", "EV5", "EV6"]
    brand_num_list = [1,2,4,5,6]
    result = []
    
    for b, n in zip(brand_num_list, dataset_name):
        auc_list = []
        print(f"{n} AUROC: ", end=' ')
        for f in range(5):
            
            specify_name = args.path.format(fold_num=f,brand_num=b)
            res = file_to_string(specify_name)
            if "Training time" in res:
                for line in reversed(res.split("\n")):
                    if "Max test set auroc score:" in line:
                        auc = float(line.split()[-1])
                        auc_list.append(auc)
                        break
            else:
                auc = None
            
            print(auc, end=' ')
        print("Average:", np.mean(auc_list))
        result.append(np.mean(auc_list))
    print('Overall average: ', np.mean(result))
