modelname="LSTM"
usewandb=""
epoch="100"
warmupepoch="10"
blr="1e-2"
seed="5"
batchsize="32"
layerdecay="1"  
weightdecay="0.05" 
droppath="0.1" 
datapercent="60"
offset=4
mask_type="use_volt_current_soc"
h="--same_normalizer --mask_type ${mask_type}"
htail="${mask_type}"
downstream="RUL"
log_dir="logs/RUL/${htail}/${modelname}/epoch${epoch}_blr${blr}_bsz${batchsize}_ld${layerdecay}_wd${weightdecay}_dp${droppath}/s${seed}"

mkdir -p $log_dir


brandnum="10"
CUDA_VISIBLE_DEVICES=`expr 0 + ${offset}` nohup python -u main_finetune.py --batch_size ${batchsize} --model ${modelname} --epochs ${epoch} --warmup_epochs ${warmupepoch} --blr ${blr} --layer_decay ${layerdecay} --fold_num 0 --brand_num ${brandnum} --weight_decay ${weightdecay} --drop_path ${droppath} ${h} --seed ${seed} ${usewandb} --downstream ${downstream} --data_percent ${datapercent} --task batterybrandmileage > ${log_dir}/f0_b${brandnum}.txt 2>&1 &
CUDA_VISIBLE_DEVICES=`expr 1 + ${offset}` nohup python -u main_finetune.py --batch_size ${batchsize} --model ${modelname} --epochs ${epoch} --warmup_epochs ${warmupepoch} --blr ${blr} --layer_decay ${layerdecay} --fold_num 1 --brand_num ${brandnum} --weight_decay ${weightdecay} --drop_path ${droppath} ${h} --seed ${seed} ${usewandb} --downstream ${downstream} --data_percent ${datapercent} --task batterybrandmileage > ${log_dir}/f1_b${brandnum}.txt 2>&1 &
CUDA_VISIBLE_DEVICES=`expr 2 + ${offset}` nohup python -u main_finetune.py --batch_size ${batchsize} --model ${modelname} --epochs ${epoch} --warmup_epochs ${warmupepoch} --blr ${blr} --layer_decay ${layerdecay} --fold_num 2 --brand_num ${brandnum} --weight_decay ${weightdecay} --drop_path ${droppath} ${h} --seed ${seed} ${usewandb} --downstream ${downstream} --data_percent ${datapercent} --task batterybrandmileage > ${log_dir}/f2_b${brandnum}.txt 2>&1 &
CUDA_VISIBLE_DEVICES=`expr 3 + ${offset}` nohup python -u main_finetune.py --batch_size ${batchsize} --model ${modelname} --epochs ${epoch} --warmup_epochs ${warmupepoch} --blr ${blr} --layer_decay ${layerdecay} --fold_num 3 --brand_num ${brandnum} --weight_decay ${weightdecay} --drop_path ${droppath} ${h} --seed ${seed} ${usewandb} --downstream ${downstream} --data_percent ${datapercent} --task batterybrandmileage > ${log_dir}/f3_b${brandnum}.txt 2>&1 &
CUDA_VISIBLE_DEVICES=`expr 0 + ${offset}` nohup python -u main_finetune.py --batch_size ${batchsize} --model ${modelname} --epochs ${epoch} --warmup_epochs ${warmupepoch} --blr ${blr} --layer_decay ${layerdecay} --fold_num 4 --brand_num ${brandnum} --weight_decay ${weightdecay} --drop_path ${droppath} ${h} --seed ${seed} ${usewandb} --downstream ${downstream} --data_percent ${datapercent} --task batterybrandmileage > ${log_dir}/f4_b${brandnum}.txt 2>&1 &

brandnum="12"
CUDA_VISIBLE_DEVICES=`expr 0 + ${offset}` nohup python -u main_finetune.py --batch_size ${batchsize} --model ${modelname} --epochs ${epoch} --warmup_epochs ${warmupepoch} --blr ${blr} --layer_decay ${layerdecay} --fold_num 0 --brand_num ${brandnum} --weight_decay ${weightdecay} --drop_path ${droppath} ${h} --seed ${seed} ${usewandb} --downstream ${downstream} --data_percent ${datapercent} --task batterybrandmileage > ${log_dir}/f0_b${brandnum}.txt 2>&1 &
CUDA_VISIBLE_DEVICES=`expr 1 + ${offset}` nohup python -u main_finetune.py --batch_size ${batchsize} --model ${modelname} --epochs ${epoch} --warmup_epochs ${warmupepoch} --blr ${blr} --layer_decay ${layerdecay} --fold_num 1 --brand_num ${brandnum} --weight_decay ${weightdecay} --drop_path ${droppath} ${h} --seed ${seed} ${usewandb} --downstream ${downstream} --data_percent ${datapercent} --task batterybrandmileage > ${log_dir}/f1_b${brandnum}.txt 2>&1 &
CUDA_VISIBLE_DEVICES=`expr 2 + ${offset}` nohup python -u main_finetune.py --batch_size ${batchsize} --model ${modelname} --epochs ${epoch} --warmup_epochs ${warmupepoch} --blr ${blr} --layer_decay ${layerdecay} --fold_num 2 --brand_num ${brandnum} --weight_decay ${weightdecay} --drop_path ${droppath} ${h} --seed ${seed} ${usewandb} --downstream ${downstream} --data_percent ${datapercent} --task batterybrandmileage > ${log_dir}/f2_b${brandnum}.txt 2>&1 &
CUDA_VISIBLE_DEVICES=`expr 3 + ${offset}` nohup python -u main_finetune.py --batch_size ${batchsize} --model ${modelname} --epochs ${epoch} --warmup_epochs ${warmupepoch} --blr ${blr} --layer_decay ${layerdecay} --fold_num 3 --brand_num ${brandnum} --weight_decay ${weightdecay} --drop_path ${droppath} ${h} --seed ${seed} ${usewandb} --downstream ${downstream} --data_percent ${datapercent} --task batterybrandmileage > ${log_dir}/f3_b${brandnum}.txt 2>&1 &
CUDA_VISIBLE_DEVICES=`expr 2 + ${offset}` nohup python -u main_finetune.py --batch_size ${batchsize} --model ${modelname} --epochs ${epoch} --warmup_epochs ${warmupepoch} --blr ${blr} --layer_decay ${layerdecay} --fold_num 4 --brand_num ${brandnum} --weight_decay ${weightdecay} --drop_path ${droppath} ${h} --seed ${seed} ${usewandb} --downstream ${downstream} --data_percent ${datapercent} --task batterybrandmileage > ${log_dir}/f4_b${brandnum}.txt 2>&1 &

brandnum="13"
CUDA_VISIBLE_DEVICES=`expr 0 + ${offset}` nohup python -u main_finetune.py --batch_size ${batchsize} --model ${modelname} --epochs ${epoch} --warmup_epochs ${warmupepoch} --blr ${blr} --layer_decay ${layerdecay} --fold_num 0 --brand_num ${brandnum} --weight_decay ${weightdecay} --drop_path ${droppath} ${h} --seed ${seed} ${usewandb} --downstream ${downstream} --data_percent ${datapercent} --task batterybrandmileage > ${log_dir}/f0_b${brandnum}.txt 2>&1 &
CUDA_VISIBLE_DEVICES=`expr 1 + ${offset}` nohup python -u main_finetune.py --batch_size ${batchsize} --model ${modelname} --epochs ${epoch} --warmup_epochs ${warmupepoch} --blr ${blr} --layer_decay ${layerdecay} --fold_num 1 --brand_num ${brandnum} --weight_decay ${weightdecay} --drop_path ${droppath} ${h} --seed ${seed} ${usewandb} --downstream ${downstream} --data_percent ${datapercent} --task batterybrandmileage > ${log_dir}/f1_b${brandnum}.txt 2>&1 &
CUDA_VISIBLE_DEVICES=`expr 2 + ${offset}` nohup python -u main_finetune.py --batch_size ${batchsize} --model ${modelname} --epochs ${epoch} --warmup_epochs ${warmupepoch} --blr ${blr} --layer_decay ${layerdecay} --fold_num 2 --brand_num ${brandnum} --weight_decay ${weightdecay} --drop_path ${droppath} ${h} --seed ${seed} ${usewandb} --downstream ${downstream} --data_percent ${datapercent} --task batterybrandmileage > ${log_dir}/f2_b${brandnum}.txt 2>&1 &
CUDA_VISIBLE_DEVICES=`expr 3 + ${offset}` nohup python -u main_finetune.py --batch_size ${batchsize} --model ${modelname} --epochs ${epoch} --warmup_epochs ${warmupepoch} --blr ${blr} --layer_decay ${layerdecay} --fold_num 3 --brand_num ${brandnum} --weight_decay ${weightdecay} --drop_path ${droppath} ${h} --seed ${seed} ${usewandb} --downstream ${downstream} --data_percent ${datapercent} --task batterybrandmileage > ${log_dir}/f3_b${brandnum}.txt 2>&1 &
CUDA_VISIBLE_DEVICES=`expr 3 + ${offset}` nohup python -u main_finetune.py --batch_size ${batchsize} --model ${modelname} --epochs ${epoch} --warmup_epochs ${warmupepoch} --blr ${blr} --layer_decay ${layerdecay} --fold_num 4 --brand_num ${brandnum} --weight_decay ${weightdecay} --drop_path ${droppath} ${h} --seed ${seed} ${usewandb} --downstream ${downstream} --data_percent ${datapercent} --task batterybrandmileage > ${log_dir}/f4_b${brandnum}.txt 2>&1 &
