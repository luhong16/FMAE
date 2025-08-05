# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import wandb
import socket
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import getpass
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import pickle
import timm
import random
import functools

assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import load_dataset, PreprocessNormalizer
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_vit, models_others

from engine_finetune import train_one_epoch, evaluate, get_evaluate_stats, evaluate_RUL
from util import tasks
from util.misc import setup_for_distributed


def get_args_parser():
    parser = argparse.ArgumentParser('FMAE', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--save_interval', default=10000, type=int)
    
    parser.add_argument('--task', type=str, default="batterybranda")
    parser.add_argument('--h', type=int, default=None, help='Choose largest h thousandths') 

    parser.add_argument("--downstream", type=str, default="anomaly")

    #wandb
    parser.add_argument("--use-wandb", action='store_true')
    parser.add_argument("--wandb-project", type=str, default="ai-x_project")
    parser.add_argument("--wandb-group", type=str, default="vit_finetune")
    parser.add_argument("--job-type", type=str, default="training-")
    parser.add_argument("--wandb-name", type=str, default="")
    parser.add_argument("--user-name", type=str, default="dl_project_")

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='Input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--SFT_blr', type=float, default=1e-6, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')
    
    parser.add_argument("--label_normalizer", action='store_true')
    
    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    parser.add_argument('--mask_type', default='no',
                        help='')
    
    parser.add_argument('--data_percent', default=20, type=int)

    parser.add_argument('--cycle_gap', default=0, type=int)
    parser.add_argument("--num_snippet", default=0, type=int) 
    parser.add_argument("--pos_embed_dim", default=-1, type=int) 
    parser.add_argument("--snippet_size", default=128, type=int) 

    # Dataset parameters
    parser.add_argument("--same_normalizer", action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')


    parser.add_argument('--fold_num', default=0, type=int)
    parser.add_argument('--brand_num', default=1, type=int)

    return parser


def main(args):

    args.distributed = False
    setup_for_distributed(True)
    if args.use_wandb and misc.is_main_process():
        run = wandb.init(config = args,
						project = args.wandb_project,
						group = args.wandb_group,
						entity = args.user_name,
						notes = socket.gethostname(),
						name = args.wandb_name,
						job_type = args.job_type)

        if args.wandb_name == '':
            wandb.run.name = args.output_dir

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cudnn.benchmark = True

    if args.downstream == 'anomaly':
        car_dict_dir = 'five_fold_utils_six_brand_all'
    elif args.downstream == 'capacity':
        if args.brand_num == 7:
            car_dict_dir = 'five_fold_utils_storage_capacity'
        elif args.brand_num in [10, 11, 12, 13]:
            car_dict_dir = 'five_fold_utils_lab_capacity'
        elif args.brand_num == 14:
            car_dict_dir = 'five_fold_utils_nc_relaxation_capacity'
        else:
            car_dict_dir = 'five_fold_utils_EV_capacity'
    elif args.downstream == 'IR':
        car_dict_dir = 'five_fold_utils_IR'
    elif args.downstream == 'RUL':
        car_dict_dir = 'five_fold_utils_RUL'
    else:
        raise NotImplementedError
    
    car_dict_dir = r"five_fold_utils/"+car_dict_dir

    if args.downstream in ['anomaly']:
        if args.finetune:
            normalizer = pickle.load(open(os.path.join(os.path.dirname(args.finetune), "norm.pkl"), 'rb'))
        else:
            normalizer = pickle.load(open(r"normailze/norm.pkl", 'rb'))
    elif args.downstream in ['capacity']:
        if args.brand_num in [7, 10, 11, 12, 13]:
            normalizer = None
        elif args.finetune:
            normalizer = pickle.load(open(os.path.join(os.path.dirname(args.finetune), "norm.pkl"), 'rb'))
        else:
            normalizer = pickle.load(open(r"normailze/norm.pkl", 'rb'))
    elif args.downstream in ['IR', 'RUL']:
        normalizer = None
    else:
        raise NotImplementedError
        
    ind_ood_car_dict = np.load(f'./{car_dict_dir}/ind_odd_dict{args.brand_num}.npz.npy', allow_pickle=True).item()
    all_car_dict = np.load(f'./{car_dict_dir}/all_car_dict.npz.npy', allow_pickle=True).item()
    
    random.shuffle(ind_ood_car_dict['ind_sorted'])
    random.shuffle(ind_ood_car_dict['ood_sorted'])
    for each_num in ind_ood_car_dict['ind_sorted'] + ind_ood_car_dict['ood_sorted']:
        random.shuffle(all_car_dict[each_num])
        
    dataset_train, _ = load_dataset(args.fold_num, brand_num=args.brand_num, same_normalizer=args.same_normalizer, car_dict_dir=car_dict_dir, downstream=args.downstream, data_type = 'finetune_train', normalizer=normalizer, dataset_fn=PreprocessNormalizer, ind_ood_car_dict=ind_ood_car_dict, all_car_dict=all_car_dict, num_snippet=args.num_snippet, cycle_gap=args.cycle_gap, data_percent=args.data_percent, seed=args.seed, task=args.task)
    normalizer = _
    dataset_test, _ = load_dataset(args.fold_num, brand_num=args.brand_num, same_normalizer=args.same_normalizer, car_dict_dir=car_dict_dir, downstream=args.downstream, data_type = 'finetune_test', normalizer=normalizer, dataset_fn=PreprocessNormalizer, ind_ood_car_dict=ind_ood_car_dict, all_car_dict=all_car_dict, num_snippet=args.num_snippet, cycle_gap=args.cycle_gap, data_percent=args.data_percent, seed=args.seed, task=args.task)
    
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    data_loader_valid = None
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    
    if args.task != "batterybrandmileage": 
        columns = ['volt', 'current', 'soc', 'max_single_volt', 'min_single_volt', 'max_temp', 'min_temp', 'timestamp']
    else:
        columns = ['volt', 'current', 'soc', 'max_single_volt', 'min_single_volt', 'max_temp', 'min_temp', 'timestamp', 'mileage']
        assert args.task == "batterybrandmileage"
    
    data_task = tasks.Task(task_name=args.task, columns=columns)

    kwargs = {}
    if args.pos_embed_dim != -1:
        kwargs["pos_embed_dim"] = args.pos_embed_dim
    kwargs["img_size"] = args.snippet_size
    kwargs["downstream"] = args.downstream
    
    if args.model.startswith("LSTM"):
        model = models_others.LSTMNet(input_dim=len(data_task.encoder), hidden_dim=args.snippet_size)
    else:
        model = models_vit.__dict__[args.model](
            in_chans=len(data_task.encoder), 
            num_classes=1,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
            num_snippet=args.num_snippet,
            **kwargs
        )

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
                
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
        trunc_normal_(model.head.weight, std=2e-5)
        
    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.model.startswith("LSTM"):
        optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        # build optimizer with layer-wise lr decay (lrd)
        param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
            no_weight_decay_list=model_without_ddp.no_weight_decay(),
            layer_decay=args.layer_decay
        )
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    if args.downstream in ['anomaly']:
        criterion = torch.nn.BCEWithLogitsLoss()
    elif args.downstream in ['capacity', 'IR', 'RUL']:
        criterion = torch.nn.MSELoss()
    else:
        raise NotImplementedError

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    #Test at first
    
    if args.downstream in ['anomaly']:
        if args.model.startswith("vit"):
            evaluate(data_loader_valid, data_loader_test, model, criterion, device, args, data_task)
    elif args.downstream in ['capacity', 'IR', 'RUL']:
        valid_res, valid_loss, _, _ = get_evaluate_stats(data_loader_valid, model, criterion, device, 'valid', label_normalizer=dataset_train.label_normalizer, args=args, data_task=data_task)
        test_res, test_loss, _, _ = get_evaluate_stats(data_loader_test, model, criterion, device, 'test', label_normalizer=dataset_train.label_normalizer, args=args, data_task=data_task)
    
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_valid_auroc = 0.0
    max_test_auroc = 0.0
    min_valid_RMSE = 1000.0
    min_test_RMSE = 1000.0
    min_valid_cell_level_RMSE = 1000.0
    min_test_cell_level_RMSE = 1000.0
    min_valid_cell_level_percentage_error = 1000.0
    min_test_cell_level_percentage_error = 1000.0
    for epoch in range(args.start_epoch, args.epochs):
        
        if args.downstream == 'RUL' and args.cycle_gap > 0:
            data_loader_train.dataset.shuffle_finetune()
        
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, label_normalizer=dataset_train.label_normalizer,
            args=args,
            data_task=data_task
        )

        if args.downstream in ['anomaly']:
            test_stats = evaluate(data_loader_valid, data_loader_test, model, criterion, device, args, data_task)
            
            print('test_auroc{:.3f}, test_loss {:.3f}'.format(
                    test_stats["test_auroc"], test_stats["test_loss"]))
            max_valid_auroc = max(max_valid_auroc, test_stats["valid_auroc"])
            max_test_auroc = max(max_test_auroc, test_stats["test_auroc"])
            print(f'Max test set auroc score: {max_test_auroc:.4f}')

            
            if args.use_wandb and misc.is_main_process():
                wandb.log({"train_loss": train_stats['loss'], 
                    "lr": train_stats['lr'], 
                    "test_auroc": test_stats['test_auroc'],
                    "test_loss": test_stats['test_loss']
                    }, 
                    step = epoch + 1)

        elif args.downstream in ['capacity', 'IR', 'RUL']:
            valid_res, valid_loss, _, _ = get_evaluate_stats(data_loader_valid, model, criterion, device, 'valid', label_normalizer=dataset_train.label_normalizer, args=args, data_task=data_task)
            test_res, test_loss, _, _ = get_evaluate_stats(data_loader_test, model, criterion, device, 'test', label_normalizer=dataset_train.label_normalizer, args=args, data_task=data_task)
            
            test_stats = {'test_RMSE': np.sqrt(test_loss), 'valid_RMSE': np.sqrt(valid_loss), 'test_loss': test_loss, 'valid_loss': valid_loss}
            min_valid_RMSE = min(min_valid_RMSE, test_stats['valid_RMSE'])
            if test_stats['test_RMSE'] <= min_test_RMSE and args.downstream in ['capacity']:
                print('update saved res')
                np.save(os.path.join(args.output_dir, "res.npy"), test_res)
            min_test_RMSE = min(min_test_RMSE, test_stats['test_RMSE'])
            print(f'Min test set RMSE: {min_test_RMSE:.9f}')

            print('test_RMSE {:.9f}, test_loss {:.9f}'.format(np.sqrt(test_loss), test_loss))

            if args.use_wandb and misc.is_main_process():
                wandb.log({"train_loss": train_stats['loss'], 
                    "lr": train_stats['lr'], 
                    "test_loss": test_stats['test_loss'],
                    "test_RMSE": test_stats['test_RMSE']
                    }, 
                    step = epoch + 1)
                
            if args.downstream in ['RUL']:
                
                valid_cell_level_rmse, valid_cell_level_percentage_error = evaluate_RUL(valid_res)
                test_cell_level_rmse, test_cell_level_percentage_error = evaluate_RUL(test_res)
                
                min_valid_cell_level_RMSE = min(min_valid_cell_level_RMSE, valid_cell_level_rmse)
                min_test_cell_level_RMSE = min(min_test_cell_level_RMSE, test_cell_level_rmse)
                min_valid_cell_level_percentage_error = min(min_valid_cell_level_percentage_error, valid_cell_level_percentage_error)
                min_test_cell_level_percentage_error = min(min_test_cell_level_percentage_error, test_cell_level_percentage_error)
                print(f'Min test cell level RMSE: {min_test_cell_level_RMSE:.9f}')
                print(f'Min test cell level percentage error: {min_test_cell_level_percentage_error:.9f}')

                print('test_cell_level_RMSE {:.9f}, test_cell_level_percentage_error {:.9f}'.format(
                    test_cell_level_rmse, test_cell_level_percentage_error))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    ckpt_str = args.finetune.replace('/', '_') 
    if len(ckpt_str.split('mask')) > 1:
        ckpt_str = ckpt_str.split('mask', 1)[1].replace('checkpoint-', '').replace('.pth', '') 
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
    