This directory contains linear probing experiments (Supplementary Table 15).

### Finetune

Run all downstream tasks:

```bash
cd experiment/linearprobe
bash finetune_FMAE_linearprobe.sh
```

### Get Results

Summarize results:

```bash
cd experiment/linearprobe
python get_result.py
```
