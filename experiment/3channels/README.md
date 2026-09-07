This directory contains the 3 channels pretrained FMAE experiment (Supplementary Table 18).

### Finetune

Run all downstream tasks:

```bash
cd experiment/3channels
bash finetune_FMAE_3channels.sh
```

### Get Results

Summarize results:

```bash
cd experiment/3channels
python get_result.py
```