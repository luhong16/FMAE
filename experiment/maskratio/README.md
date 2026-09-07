This directory contains FMAE mask-ratio ablation experiments (Supplementary Table 7).

### Low: patch masking ratio 0.2, channel masking ratio 0.2

Run all downstream tasks:

```bash
cd experiment/maskratio
bash finetune_FMAE_mpr0.2_mcr0.2.sh
```

### High: patch masking ratio 0.8, channel masking ratio 0.8

Run all downstream tasks:

```bash
cd experiment/maskratio
bash finetune_FMAE_mpr0.8_mcr0.8.sh
```

### Get Results

Summarize results of `Low: (0.2, 0.2)`:

```bash
cd experiment/maskratio
python get_result.py --mask_ratio 0.2
```

Summarize results of `High: (0.8, 0.8)`:

```bash
cd experiment/maskratio
python get_result.py --mask_ratio 0.8
```
