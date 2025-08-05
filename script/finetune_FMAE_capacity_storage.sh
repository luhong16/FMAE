dirname="pretrained_model/mae_pretrainmae_vit_half_patch16_mpr0.5_mcr0.4_msn0_lrNone_blr0.00015_minlr0.0_weightdecay0.05_warmupepoch40_numsnippet5_pos_embed_dim12_d_pos_embed_dim8_d_typecombine_d_pad_typesoc_current_mileage_embed_epochs800_s0"
ckpt="799"
modelname="vit_half_patch16"
usewandb=""
epoch="20"
warmupepoch="4"
blr="5e-1"
seed="5"
batchsize="32"
layerdecay="0.5" 
weightdecay="0.03" 
droppath="0.1" 
datapercent="5"
offset=0
pos_embed_dim="12"
mask_type="no"
h="--same_normalizer --mask_type ${mask_type}"
htail="${mask_type}"
downstream="capacity"
log_dir="logs/capacity/${htail}/${ckpt}_pretrain_${modelname}/epoch${epoch}_blr${blr}_bsz${batchsize}_ld${layerdecay}_wd${weightdecay}_dp${droppath}/s${seed}"

mkdir -p $log_dir

brandnum="7"
CUDA_VISIBLE_DEVICES=`expr 0 + ${offset}` nohup python -u main_finetune.py --finetune ./${dirname}/checkpoint-${ckpt}.pth --batch_size ${batchsize} --model ${modelname} --epochs ${epoch} --warmup_epochs ${warmupepoch} --blr ${blr} --layer_decay ${layerdecay} --fold_num 0 --brand_num ${brandnum} --weight_decay ${weightdecay} --drop_path ${droppath} ${h} --pos_embed_dim ${pos_embed_dim} --seed ${seed} ${usewandb} --downstream ${downstream} --data_percent ${datapercent} --task batterybrandmileage --output_dir ${log_dir}/f0_b${brandnum} > ${log_dir}/f0_b${brandnum}.txt 2>&1 &
CUDA_VISIBLE_DEVICES=`expr 1 + ${offset}` nohup python -u main_finetune.py --finetune ./${dirname}/checkpoint-${ckpt}.pth --batch_size ${batchsize} --model ${modelname} --epochs ${epoch} --warmup_epochs ${warmupepoch} --blr ${blr} --layer_decay ${layerdecay} --fold_num 1 --brand_num ${brandnum} --weight_decay ${weightdecay} --drop_path ${droppath} ${h} --pos_embed_dim ${pos_embed_dim} --seed ${seed} ${usewandb} --downstream ${downstream} --data_percent ${datapercent} --task batterybrandmileage --output_dir ${log_dir}/f1_b${brandnum} > ${log_dir}/f1_b${brandnum}.txt 2>&1 &
CUDA_VISIBLE_DEVICES=`expr 2 + ${offset}` nohup python -u main_finetune.py --finetune ./${dirname}/checkpoint-${ckpt}.pth --batch_size ${batchsize} --model ${modelname} --epochs ${epoch} --warmup_epochs ${warmupepoch} --blr ${blr} --layer_decay ${layerdecay} --fold_num 2 --brand_num ${brandnum} --weight_decay ${weightdecay} --drop_path ${droppath} ${h} --pos_embed_dim ${pos_embed_dim} --seed ${seed} ${usewandb} --downstream ${downstream} --data_percent ${datapercent} --task batterybrandmileage --output_dir ${log_dir}/f2_b${brandnum} > ${log_dir}/f2_b${brandnum}.txt 2>&1 &
CUDA_VISIBLE_DEVICES=`expr 3 + ${offset}` nohup python -u main_finetune.py --finetune ./${dirname}/checkpoint-${ckpt}.pth --batch_size ${batchsize} --model ${modelname} --epochs ${epoch} --warmup_epochs ${warmupepoch} --blr ${blr} --layer_decay ${layerdecay} --fold_num 3 --brand_num ${brandnum} --weight_decay ${weightdecay} --drop_path ${droppath} ${h} --pos_embed_dim ${pos_embed_dim} --seed ${seed} ${usewandb} --downstream ${downstream} --data_percent ${datapercent} --task batterybrandmileage --output_dir ${log_dir}/f3_b${brandnum} > ${log_dir}/f3_b${brandnum}.txt 2>&1 &
CUDA_VISIBLE_DEVICES=`expr 3 + ${offset}` nohup python -u main_finetune.py --finetune ./${dirname}/checkpoint-${ckpt}.pth --batch_size ${batchsize} --model ${modelname} --epochs ${epoch} --warmup_epochs ${warmupepoch} --blr ${blr} --layer_decay ${layerdecay} --fold_num 4 --brand_num ${brandnum} --weight_decay ${weightdecay} --drop_path ${droppath} ${h} --pos_embed_dim ${pos_embed_dim} --seed ${seed} ${usewandb} --downstream ${downstream} --data_percent ${datapercent} --task batterybrandmileage --output_dir ${log_dir}/f4_b${brandnum} > ${log_dir}/f4_b${brandnum}.txt 2>&1 &
