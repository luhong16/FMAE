This is the official code repository for our paper entitled **Multitask Battery Management With Flexible Pretraining**.

# Environment requirement
```
# basic environment
python 3.6.6

# torch version
torch==1.7.1+cu110
torchvision==0.8.2+cu110

# installing packages
pip install -r requirement.txt
```
# Preparation
## Dataset download
Download `data` and `five_fold_utils` from the link in our paper (or at the end of this README) and unzip them. 
Please make sure the structure is like the following. 


```
    |--data
    	|--EV
            |--data
                |--...
            |--discharge_data
                |--...
        |--lab_data
            |--...
        |--BESS_data
            |--...
        |--NC_relaxation_datasets
        	|--...
    |--five_fold_utils
    	|--...
    |--script
    	|--...
    |--...
    main_pretrain.py
    main_finetune.py
    ...
```

## Pretrained model

The pretrained model is in the `pretrained_model` folder. Of course, you can also pretrain from scratch. This part will be mentioned next.

## Other information for five-fold cross-validation

`five_fold_utils` folder provides the path information for five-fold cross-validation.

`normalize` folder provides the normalization coefficients.


# FMAE pretraining (ours)

Run the following command to pretrain from scratch. 

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u -m torch.distributed.launch --nproc_per_node=8 main_pretrain.py --batch_size 32 --model mae_vit_half_patch16 --mask_patch_ratio 0.5 --mask_channel_ratio 0.4 --epochs 800 --warmup_epochs 40 --same_normalizer --blr 1.5e-4 --weight_decay 0.05 --num_snippet 5 --task batterybrandmileage --decoder_type combine --decoder_pad_type soc_current_mileage_embed --mask_snippet_num 0 --pos_embed_dim 12 --decoder_pos_embed_dim 8 --output_dir ./
```

# FMAE finetuning (ours)

We provide several scripts named `finetune_FMAE_{task_name}` (or `finetune_FMAE_{task_name}_{dataset_name}` for the capacity estimation task) in the `script` folder for finetuning.

For example, run

```
bash script/finetune_FMAE_anomaly.sh
```

The result will be stored in the `logs` folder.

# LSTM finetuning

We provide several scripts named `finetune_others_{task_name}` (or `finetune_FMAE_{task_name}_{dataset_name}` for the capacity estimation task) in the `script` folder for finetuning.

For example, run

```
bash script/finetune_others_anomaly.sh
```

The result will be stored in the`logs` folder.

# Data availability
The datasets are available at links below https://cloud.tsinghua.edu.cn/d/fca1245f527d479d82f5/.

# Code reference
We use partial code from 
```
https://github.com/facebookresearch/mae
https://github.com/962086838/Battery_fault_detection_NC_github
```
