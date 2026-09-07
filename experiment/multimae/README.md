This directory contains Multi-modal MAE experiments (Supplementary Table 16).

### Pretrain

```bash
cd experiment/multimae
bash script/pretrain_MultiMAE.sh
```

### Finetune

Run all downstream tasks:

```bash
cd experiment/multimae
bash script/finetune_MultiMAE_all.sh
```

### Get Results

Summarize results:

```bash
cd experiment/multimae
python get_result.py
```
