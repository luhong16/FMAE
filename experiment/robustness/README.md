This directory contains model robustness experiments (Supplementary Table 11).

### FMAE

```bash
cd experiment/robustness
bash finetune_FMAE_capacity_lab.sh
```

### iTransformer

```bash
cd experiment/robustness
bash finetune_iTransformer_capacity_lab_with_noise.sh
```

### PatchTST

```bash
cd experiment/robustness
bash finetune_PatchTST_capacity_lab_with_noise.sh
```

### TimeMixer++

```bash
cd experiment/robustness
bash finetune_TimeMixer++_capacity_lab_with_noise.sh
```

### Get Results

Summarize results of each model:

```bash
cd experiment/robustness
python get_result.py --model FMAE
python get_result.py --model iTransformer
python get_result.py --model PatchTST
python get_result.py --model TimeMixer++
```
