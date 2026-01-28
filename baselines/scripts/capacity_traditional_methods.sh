# EV + lab
mkdir -p ./baselines/results/capacity
for percent in 0.05
do
for brand_num in 1 2 3 4 5 6 7 10 11 12 13
do
for fold in 0 1 2 3 4
do

nohup python -u ./baselines/RandomForest.py \
            --Task capacity \
            --percent $percent \
            --Brand $brand_num \
            --Fold $fold >> ./baselines/results/capacity/Capacity_Brand_${brand_num}_Fold_${fold}_percent_${percent}_RF_logs.txt 2>&1 &

# nohup python -u ./baselines/XGBoost.py \
#             --Task capacity \
#             --percent $percent \
#             --Brand $brand_num \
#             --Fold $fold >> ./baselines/results/capacity/Capacity_Brand_${brand_num}_Fold_${fold}_percent_${percent}_XGBoost_logs.txt 2>&1 &

done
done
done

# NC capacity
mkdir -p ./baselines/results/NC_capacity
for percent in 1.0
do
for brand_num in 14
do
for fold in 0 1 2 3 4
do

nohup python -u ./baselines/XGBoost.py \
            --Task NC_capacity \
            --percent $percent \
            --Brand $brand_num \
            --Fold $fold >> ./baselines/results/NC_capacity/NC_capacity_Brand_${brand_num}_Fold_${fold}_percent_${percent}_XGBoost_logs.txt 2>&1 &

done
done
done