dirname="pretrained_model/mae_pretrainmae_vit_tiny_patch16_depth3_maskpatch0.5_maskchannel0.4_lrNone_blr0.00015_minlr0.0_weightdecay0.05_warmupepoch40_epochs800_s0same_normalizer_contexttype0"
ckpt="799"
modelname="vit_tiny_patch16_depth3"
usewandb=""
epoch="20"
warmupepoch="4"
blr="8e-3"
seed="5"
batchsize="32"
layerdecay="0.65" 
weightdecay="0.05" 
droppath="0.1" 
datapercent="20"
offset=0
mask_type="max_min_volt_temp"
h="--same_normalizer --mask_type ${mask_type}"
htail="${mask_type}"
downstream="IR"
log_dir="logs/IR/${htail}/${ckpt}_pretrain_${modelname}/epoch${epoch}_blr${blr}_bsz${batchsize}_ld${layerdecay}_wd${weightdecay}_dp${droppath}/s${seed}"


mkdir -p $log_dir

brandnum="10"
CUDA_VISIBLE_DEVICES=`expr 0 + ${offset}` nohup python -u main_finetune.py --finetune ./${dirname}/checkpoint-${ckpt}.pth --batch_size ${batchsize} --model ${modelname} --epochs ${epoch} --warmup_epochs ${warmupepoch} --blr ${blr} --layer_decay ${layerdecay} --fold_num 0 --brand_num ${brandnum} --weight_decay ${weightdecay} --drop_path ${droppath} ${h} --seed ${seed} ${usewandb} --downstream ${downstream} --data_percent ${datapercent} > ${log_dir}/f0_b${brandnum}.txt 2>&1 &
CUDA_VISIBLE_DEVICES=`expr 1 + ${offset}` nohup python -u main_finetune.py --finetune ./${dirname}/checkpoint-${ckpt}.pth --batch_size ${batchsize} --model ${modelname} --epochs ${epoch} --warmup_epochs ${warmupepoch} --blr ${blr} --layer_decay ${layerdecay} --fold_num 1 --brand_num ${brandnum} --weight_decay ${weightdecay} --drop_path ${droppath} ${h} --seed ${seed} ${usewandb} --downstream ${downstream} --data_percent ${datapercent} > ${log_dir}/f1_b${brandnum}.txt 2>&1 &
CUDA_VISIBLE_DEVICES=`expr 2 + ${offset}` nohup python -u main_finetune.py --finetune ./${dirname}/checkpoint-${ckpt}.pth --batch_size ${batchsize} --model ${modelname} --epochs ${epoch} --warmup_epochs ${warmupepoch} --blr ${blr} --layer_decay ${layerdecay} --fold_num 2 --brand_num ${brandnum} --weight_decay ${weightdecay} --drop_path ${droppath} ${h} --seed ${seed} ${usewandb} --downstream ${downstream} --data_percent ${datapercent} > ${log_dir}/f2_b${brandnum}.txt 2>&1 &
CUDA_VISIBLE_DEVICES=`expr 3 + ${offset}` nohup python -u main_finetune.py --finetune ./${dirname}/checkpoint-${ckpt}.pth --batch_size ${batchsize} --model ${modelname} --epochs ${epoch} --warmup_epochs ${warmupepoch} --blr ${blr} --layer_decay ${layerdecay} --fold_num 3 --brand_num ${brandnum} --weight_decay ${weightdecay} --drop_path ${droppath} ${h} --seed ${seed} ${usewandb} --downstream ${downstream} --data_percent ${datapercent} > ${log_dir}/f3_b${brandnum}.txt 2>&1 &
CUDA_VISIBLE_DEVICES=`expr 3 + ${offset}` nohup python -u main_finetune.py --finetune ./${dirname}/checkpoint-${ckpt}.pth --batch_size ${batchsize} --model ${modelname} --epochs ${epoch} --warmup_epochs ${warmupepoch} --blr ${blr} --layer_decay ${layerdecay} --fold_num 4 --brand_num ${brandnum} --weight_decay ${weightdecay} --drop_path ${droppath} ${h} --seed ${seed} ${usewandb} --downstream ${downstream} --data_percent ${datapercent} > ${log_dir}/f4_b${brandnum}.txt 2>&1 &
