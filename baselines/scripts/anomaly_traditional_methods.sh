mkdir -p ./baselines/results/anomaly
for brand_num in 1 2 4 5 6
do
for fold in 0 1 2 3 4
do
nohup python -u ./baselines/Variation_evaluation.py \
            --Brand $brand_num \
            --Fold $fold >> ./baselines/results/anomaly/Anomaly_Brand_${brand_num}_Fold_${fold}_V_E_logs.txt 2>&1 &

done
done