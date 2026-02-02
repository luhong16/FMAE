This is the official code repository for our paper entitled **Multitask Battery Management With Flexible Pretraining**.

### Environment requirement

```
# system
Ubuntu 22.04

# basic environment
python 3.6.6

# torch version
torch==1.7.1+cu110
torchvision==0.8.2+cu110

# installing conda environments and packages by running the following commands
conda create --name fmae python=3.6.6
source activate fmae
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
cd FMAE
pip install -r requirements.txt
```

Environment setup may take 20 minutes to an hour, depending on your network conditions.

### Demo

We provide a demo that loads the pretrained model and finetunes it on the capacity estimation task using a small subset of EV data. This is intended to verify that your environment is correctly installed and the code executes as expected.

Instructions for downloading the full dataset, as well as performing pretraining and finetuning on the complete dataset, are provided below.

#### Run the demo

Execute the following command to start the finetuning process:
```
python -u main_finetune.py --finetune ./pretrained_model/mae_pretrainmae_vit_half_patch16_mpr0.5_mcr0.4_msn0_lrNone_blr0.00015_minlr0.0_weightdecay0.05_warmupepoch40_numsnippet5_pos_embed_dim12_d_pos_embed_dim8_d_typecombine_d_pad_typesoc_current_mileage_embed_epochs800_s0/checkpoint-799.pth --batch_size 32 --model vit_half_patch16 --epochs 20 --warmup_epochs 4 --blr 5e-2 --layer_decay 0.5 --fold_num 2 --brand_num -1 --weight_decay 0.005 --drop_path 0.0 --same_normalizer --mask_type no --pos_embed_dim 12 --seed 5  --downstream capacity --data_percent 100 --task batterybrandmileage --output_dir logs/demo > logs/demo/demo.txt 2>&1
```
It will take less than one minute.

#### Retrieve results

Execute the following command to read the results:
```
python get_result/get_result_capacity.py --path logs/demo --type demo
```

#### Expected Output

The expected results should look like the following (please note that values may vary slightly depending on your hardware and environment):

```
Demo RMSE:  1.489702393514389 Average: 1.489702393514389
Overall average:  1.489702393514389
```

### Preparation

#### Dataset download

We provide a python code to download dataset. Run

```
python thu_cloud_download.py -l https://cloud.tsinghua.edu.cn/d/713eb388382c49e585a6/ -s ./
```

, and then press 'y' and enter. 

You can also download `data.tar.gz.0/1/2/3/4` one by one, and `five_fold_utils` from the link in our paper (or at the end of this README).

Run

```
cat data.tar.gz.* | tar -xzvf -
```

to unzip them. 

Please make sure the structure is like the following. 


```
    |--data
    |--five_fold_utils
    |--script
    |--...
    main_pretrain.py
    main_finetune.py
    ...
```

#### Pretrained model

The pretrained model is in the `pretrained_model` folder. Of course, you can also pretrain from scratch. This part will be mentioned next.

#### Other information for five-fold cross-validation

`five_fold_utils` folder provides the path information for five-fold cross-validation.

`normalize` folder provides the normalization coefficients.

### FMAE pretraining (ours)

Run the following command to pretrain from scratch. 

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u -m torch.distributed.launch --nproc_per_node=8 main_pretrain.py --batch_size 32 --model mae_vit_half_patch16 --mask_patch_ratio 0.5 --mask_channel_ratio 0.4 --epochs 800 --warmup_epochs 40 --same_normalizer --blr 1.5e-4 --weight_decay 0.05 --num_snippet 5 --task batterybrandmileage --decoder_type combine --decoder_pad_type soc_current_mileage_embed --mask_snippet_num 0 --pos_embed_dim 12 --decoder_pos_embed_dim 8 --output_dir ./
```

### Finetuning

We provide several scripts named `finetune_{method_name}_{task_name}` (or `finetune_FMAE_{method_name}_{dataset_name}` for the capacity estimation task) in the `script` and `baselines/scripts` folder for finetuning.

#### FMAE finetuning (ours)

**Anomaly detection**: run the following script

```
bash script/finetune_FMAE_anomaly.sh
```

**Capacity estimation**: run the following scripts

```
bash script/finetune_FMAE_capacity_EV.sh
bash script/finetune_FMAE_capacity_lab.sh
bash script/finetune_FMAE_capacity_storage.sh
```

**RUL prediction**: run the following script

```
bash script/finetune_FMAE_RUL.sh
```

**IR estimation**: run the following script

```
bash script/finetune_FMAE_IR.sh
```

The result will be stored in the `logs` folder. 

#### LSTM

**Anomaly detection**: run the following script

```
bash script/finetune_LSTM_anomaly.sh
```

**Capacity estimation**: run the following scripts

```
bash script/finetune_LSTM_capacity_EV.sh
bash script/finetune_LSTM_capacity_lab.sh
bash script/finetune_LSTM_capacity_storage.sh
```

**RUL prediction**: run the following script

```
bash script/finetune_LSTM_RUL.sh
```

**IR estimation**: run the following script

```
bash script/finetune_LSTM_IR.sh
```

The result will be stored in the `logs` folder. 

#### PatchTST and iTransformer

**Anomaly detection**: run the following script

```
bash ./baselines/scripts/finetune_PatchTST_anomaly.sh
bash ./baselines/scripts/finetune_iTransformer_anomaly.sh
```

**Capacity estimation**: run the following scripts

```
bash ./baselines/scripts/finetune_PatchTST_capacity_EV.sh
bash ./baselines/scripts/finetune_PatchTST_capacity_lab.sh
bash ./baselines/scripts/finetune_PatchTST_capacity_storage.sh

bash ./baselines/scripts/finetune_iTransformer_capacity_EV.sh
bash ./baselines/scripts/finetune_iTransformer_capacity_lab.sh
bash ./baselines/scripts/finetune_iTransformer_capacity_storage.sh
```

**RUL prediction**: run the following script

```
bash ./baselines/scripts/finetune_PatchTST_RUL.sh
bash ./baselines/scripts/finetune_iTransformer_RUL.sh
```

**IR estimation**: run the following script

```
bash ./baselines/scripts/finetune_PatchTST_IR.sh
bash ./baselines/scripts/finetune_iTransformer_IR.sh
```

The result will be stored in the `logs` folder. 

#### Traditional methods 

To start with, run 

```
bash ./baselines/scripts/init_folder.sh
```

to create `baselines/result` folder.

For traditional methods, we need to obtain features from data snippets for downstream tasks, run the following script to generate features.

```
python ./baselines/data_preprocess.py
```

**Anomaly detection (Variation Evaluation)**, run the following script:

```
bash ./baselines/scripts/anomaly_traditional_methods.sh
```

**Capacity estimation (Random forest, XGBoost)**, run the following script:

```
bash ./baselines/scripts/capacity_traditional_methods.sh
```

**RUL prediction (discharge model, variance model)**, run the following script:

```
bash ./baselines/scripts/RUL_traditional_methods.sh
```

**IR estimation (CNN)**, run the following script:

```
bash ./baselines/scripts/IR_traditional_methods.sh
```

### Get results

To obtain the five-fold results on multiple datasets, we provide some codes, `get_result_{task_name}.py`, in the `get_result` folder. Run 

```
python get_result/get_result_{task_name}.py --path THE_PATH_OF_LOGS
```

, which shows the per-fold results and average results for each dataset in the specific task.

For the capacity estimation task, please add `--type {dataset_type}` to specify dataset types (EV, BESS or lab).

The following is an example of how to get the execution results of the previous scripts.

#### FMAE

**Anomaly detection**: run

```
python get_result/get_result_anomaly.py --path logs_gt2/anomaly/no/799_pretrain_vit_half_patch16/epoch10_blr5e-3_bsz32_ld0.65_wd0.01_dp0.1/s5
```

**Capacity estimation**: run

```
python get_result/get_result_capacity.py --path logs/capacity/no/799_pretrain_vit_half_patch16/epoch20_blr5e-2_bsz32_ld0.5_wd0.005_dp0.0/s5 --type EV

python get_result/get_result_capacity.py --path logs/capacity/no/799_pretrain_vit_half_patch16/epoch20_blr5e-1_bsz32_ld0.5_wd0.03_dp0.1/s5 --type BESS

python get_result/get_result_capacity.py --path logs/capacity/max_min_volt_temp/799_pretrain_vit_half_patch16/epoch20_blr5e-2_bsz32_ld0.65_wd0.01_dp0.1/s5 --type lab
```

**RUL prediction**: run

```
python get_result/get_result_RUL.py --path logs/RUL/max_min_volt_temp_cyclegap20/799_pretrain_vit_half_patch16/epoch100_blr1e-2_bsz32_ld0.5_wd0.005_dp0.0/s5
```

**IR estimation**: run

```
python get_result/get_result_IR.py --path logs/IR/max_min_volt_temp/799_pretrain_vit_half_patch16/epoch20_warmupepoch4_blr5e-3_bsz32_ld0.8_wd0.05_dp0.1/s5
```

Results of **LSTM** can be obtained by running the same command; just change the specified path.

For the **time-series models**, the following command may also be executed to obtain results, in addition to the one mentioned above:

```
python ./baselines/visual_results/anomaly_results_t-s.py
python ./baselines/visual_results/capacity_results_t-s.py
python ./baselines/visual_results/RUL_results_t-s.py
python ./baselines/visual_results/IR_results_t-s.py
```

For the **traditional methods**, run the following command to get the results:

```
python ./baselines/visual_results/anomaly_results_traditional_methods.py
python ./baselines/visual_results/capacity_results_traditional_methods.py
python ./baselines/visual_results/RUL_results_traditional_methods.py
python ./baselines/visual_results/IR_results_traditional_methods.py
```

### Data availability

The datasets are available at links below https://cloud.tsinghua.edu.cn/d/713eb388382c49e585a6/.

### Code reference

We use partial code from 

```
https://github.com/facebookresearch/mae
https://github.com/thuml/Time-Series-Library
https://github.com/962086838/Battery_fault_detection_NC_github
https://github.com/chenyifanthu/THU-Cloud-Downloader
```