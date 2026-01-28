mkdir -p ./baselines/results/IR
nohup python -u ./baselines/CNN.py >> ./baselines/results/IR/CNN_logs.txt 2>&1 &