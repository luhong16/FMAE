This directory contains the LiPM baseline experiment (Supplementary Table 17).

Since the pretrained model exceeds GitHub's single file size limit, please download the pretrained model from `https://cloud.tsinghua.edu.cn/d/9aceaee7626d437e82bd/`, or use the following command to download it:

```
python thu_cloud_download.py -l https://cloud.tsinghua.edu.cn/d/9aceaee7626d437e82bd/ -s ./pretrained_model/
```

### Pretrain

```bash
cd experiment/lipm
OUTPUT_DIR=pretrained_model \
bash lipm_fmae_baseline/pretrain_ev.sh
```

### Finetune

Run all downstream tasks:

```bash
cd experiment/lipm
bash lipm_fmae_baseline/run_all_tasks.sh pretrained_model/backbone-last.pth
```

### Get Results

Summarize results:

```bash
cd experiment/lipm
python get_result.py
```
