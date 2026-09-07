This directory contains the multi-source pretrained FMAE experiment (Supplementary Table 19).

### Pretrain

```bash
cd experiment/multisource
bash pretrain_FMAE_ev_lab.sh
```

### Finetune

Run all downstream tasks:

```bash
cd experiment/multisource
bash finetune_FMAE_ev_lab.sh
```

### Get Results

Summarize results:

```bash
cd experiment/multisource
python get_result.py
```
