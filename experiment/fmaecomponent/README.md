This directory contains ablation studies on the key components of FMAE (Table 1).

### MAE

Run all downstream tasks:

```bash
cd experiment/fmaecomponent
bash finetune_MAE.sh
```

### MAE+channel tokens

Run all downstream tasks:

```bash
cd experiment/fmaecomponent
bash finetune_MAE_learn_channel_tokens.sh
```

### Get Results

Summarize results of MAE:

```bash
cd experiment/3channels
python get_result.py --component MAE
```

Summarize results of MAE+channel tokens:

```bash
cd experiment/3channels
python get_result.py --component MAE_learn_channel_tokens
```