mkdir -p ./baselines/results/RUL_w_log
mkdir -p ./baselines/results/RUL_wo_log
for brand_num in 10 12 13
do
for fold in 0 1 2 3 4
do
for points in 128 256 1280 2560 12800
do
for Predicting_log10 in 1
do
nohup python -u ./baselines/ElasticLinear_variance.py \
            --Task RUL \
            --Brand $brand_num \
            --Fold $fold \
            --Points $points \
            --Predicting_log10 $Predicting_log10 >> ./baselines/results/RUL_w_log/RUL_Brand_${brand_num}_Fold_${fold}_Points_${points}_ElasticLinear_variance_logs.txt 2>&1 &

nohup python -u ./baselines/ElasticLinear_discharge.py \
            --Task RUL \
            --Brand $brand_num \
            --Fold $fold \
            --Points $points \
            --Predicting_log10 $Predicting_log10 >> ./baselines/results/RUL_w_log/RUL_Brand_${brand_num}_Fold_${fold}_Points_${points}_ElasticLinear_discharge_logs.txt 2>&1 &

done
done
done
done