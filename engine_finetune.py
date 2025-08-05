# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import numpy as np
import math
import sys
from typing import Iterable, Optional

import torch

import util.evaluation
import util.misc as misc
import util.lr_sched as lr_sched

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    label_normalizer=None, args=None, data_task=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 200

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        samples = batch[0]
        targets = batch[1]
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        targets = targets.to(device, non_blocking=True)

        B = samples.shape[0]

        if args.mask_type in ["max_min_volt_temp", "use_volt_current_soc"]:
            if args.task == "batterybrandmileage":
                mask_channel = torch.tensor([0,0,0,1,1,1,1,0,0], dtype=torch.float32, device=device).unsqueeze(-1).repeat(B, 1, 1)
            else:
                mask_channel = torch.tensor([0,0,0,1,1,1,1,0], dtype=torch.float32, device=device).unsqueeze(-1).repeat(B, 1, 1)
        elif args.mask_type in ["max_min_volt", "use_volt_current_soc_temp"]:
            if args.task == "batterybrandmileage":
                mask_channel = torch.tensor([0,0,0,1,1,0,0,0,0], dtype=torch.float32, device=device).unsqueeze(-1).repeat(B, 1, 1)
            else:
                mask_channel = torch.tensor([0,0,0,1,1,0,0,0], dtype=torch.float32, device=device).unsqueeze(-1).repeat(B, 1, 1)
        elif args.mask_type in ["max_min_temp", "use_volt_current_soc_volt"]:
            if args.task == "batterybrandmileage":
                mask_channel = torch.tensor([0,0,0,0,0,1,1,0,0], dtype=torch.float32, device=device).unsqueeze(-1).repeat(B, 1, 1)
            else:
                mask_channel = torch.tensor([0,0,0,0,0,1,1,0], dtype=torch.float32, device=device).unsqueeze(-1).repeat(B, 1, 1)
        elif args.mask_type in ["max_min_temp_min_volt", "use_volt_current_soc_max_volt"]:
            if args.task == "batterybrandmileage":
                mask_channel = torch.tensor([0,0,0,0,1,1,1,0,0], dtype=torch.float32, device=device).unsqueeze(-1).repeat(B, 1, 1)
            else:
                mask_channel = torch.tensor([0,0,0,0,1,1,1,0], dtype=torch.float32, device=device).unsqueeze(-1).repeat(B, 1, 1)
        elif args.mask_type in ["all_volt"]:
            if args.task == "batterybrandmileage":
                mask_channel = torch.tensor([1,0,0,1,1,0,0,0,0], dtype=torch.float32, device=device).unsqueeze(-1).repeat(B, 1, 1)
            else:
                mask_channel = torch.tensor([1,0,0,1,1,0,0,0], dtype=torch.float32, device=device).unsqueeze(-1).repeat(B, 1, 1)
        elif args.mask_type in ["except_volt"]:
            if args.task == "batterybrandmileage":
                mask_channel = torch.tensor([0,1,1,1,1,1,1,1,1], dtype=torch.float32, device=device).unsqueeze(-1).repeat(B, 1, 1)
            else:
                mask_channel = torch.tensor([0,1,1,1,1,1,1,1], dtype=torch.float32, device=device).unsqueeze(-1).repeat(B, 1, 1)
        else:
            mask_channel = None
            
        if mask_channel is not None:
            mask_channel = data_task.encoder_filter(mask_channel)


        if args.num_snippet == 0:
            samples = samples.to(device, non_blocking=True)
            samples = data_task.encoder_filter(samples)
            if args.mask_type in ["use_volt_current_soc", "use_volt_current_soc_temp", "use_volt_current_soc_volt", "use_volt_current_soc_max_volt"]:
                samples = samples * (1 - mask_channel)
                mask_channel = None
            input = samples
        else:
            sample_list = []
            for i in range(args.num_snippet):
                samples_i = samples[:, i, :, :]
                samples_i = samples_i.to(device, non_blocking=True)
                samples_i = data_task.encoder_filter(samples_i)
                if args.mask_type in ["use_volt_current_soc", "use_volt_current_soc_temp", "use_volt_current_soc_volt", "use_volt_current_soc_max_volt"]:
                    samples_i = samples_i * (1 - mask_channel)
                sample_list.append(samples_i)
            input = sample_list.copy()
            if args.mask_type in ["use_volt_current_soc", "use_volt_current_soc_temp", "use_volt_current_soc_volt", "use_volt_current_soc_max_volt"]:
                mask_channel = None
        
        outputs = model(input, mask_channel).squeeze(-1)

        if args.label_normalizer:
            assert args.downstream in ['capacity']
            label_mean = torch.tensor(label_normalizer[0], device=targets.device)
            label_std = torch.tensor(label_normalizer[1], device=targets.device)
            loss = criterion(outputs, (targets.float() - label_mean) / label_std)
        else:
            loss = criterion(outputs, targets.float())

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def get_evaluate_stats(data_loader, model, criterion, device, header, label_normalizer=None, args=None, data_task=None):
    if data_loader is None:
        return None, 0, None, None
    
    metric_logger = misc.MetricLogger(delimiter="  ")

    # switch to evaluation mode
    model.eval()

    predict = []
    label = []
    logits = []
    car_id = []
    losses = []
    current_cycle = []
    rul_cycle = []
    epoch_loss = 0
    len_data = 0

    for batch in metric_logger.log_every(data_loader, 200, header):
        samples = batch[0]
        targets = batch[1]
        car = batch[2]
        
        targets = targets.to(device, non_blocking=True)

        B = samples.shape[0]

        if args.mask_type in ["max_min_volt_temp", "use_volt_current_soc"]:
            if args.task == "batterybrandmileage":
                mask_channel = torch.tensor([0,0,0,1,1,1,1,0,0], dtype=torch.float32, device=device).unsqueeze(-1).repeat(B, 1, 1)
            else:
                mask_channel = torch.tensor([0,0,0,1,1,1,1,0], dtype=torch.float32, device=device).unsqueeze(-1).repeat(B, 1, 1)
        elif args.mask_type in ["max_min_volt", "use_volt_current_soc_temp"]:
            if args.task == "batterybrandmileage":
                mask_channel = torch.tensor([0,0,0,1,1,0,0,0,0], dtype=torch.float32, device=device).unsqueeze(-1).repeat(B, 1, 1)
            else:
                mask_channel = torch.tensor([0,0,0,1,1,0,0,0], dtype=torch.float32, device=device).unsqueeze(-1).repeat(B, 1, 1)
        elif args.mask_type in ["max_min_temp", "use_volt_current_soc_volt"]:
            if args.task == "batterybrandmileage":
                mask_channel = torch.tensor([0,0,0,0,0,1,1,0,0], dtype=torch.float32, device=device).unsqueeze(-1).repeat(B, 1, 1)
            else:
                mask_channel = torch.tensor([0,0,0,0,0,1,1,0], dtype=torch.float32, device=device).unsqueeze(-1).repeat(B, 1, 1)
        elif args.mask_type in ["max_min_temp_min_volt", "use_volt_current_soc_max_volt"]:
            if args.task == "batterybrandmileage":
                mask_channel = torch.tensor([0,0,0,0,1,1,1,0,0], dtype=torch.float32, device=device).unsqueeze(-1).repeat(B, 1, 1)
            else:
                mask_channel = torch.tensor([0,0,0,0,1,1,1,0], dtype=torch.float32, device=device).unsqueeze(-1).repeat(B, 1, 1)
        elif args.mask_type in ["all_volt"]:
            if args.task == "batterybrandmileage":
                mask_channel = torch.tensor([1,0,0,1,1,0,0,0,0], dtype=torch.float32, device=device).unsqueeze(-1).repeat(B, 1, 1)
            else:
                mask_channel = torch.tensor([1,0,0,1,1,0,0,0], dtype=torch.float32, device=device).unsqueeze(-1).repeat(B, 1, 1)
        elif args.mask_type in ["except_volt"]:
            if args.task == "batterybrandmileage":
                mask_channel = torch.tensor([0,1,1,1,1,1,1,1,1], dtype=torch.float32, device=device).unsqueeze(-1).repeat(B, 1, 1)
            else:
                mask_channel = torch.tensor([0,1,1,1,1,1,1,1], dtype=torch.float32, device=device).unsqueeze(-1).repeat(B, 1, 1)
        else:
            mask_channel = None
            
        if mask_channel is not None:
            mask_channel = data_task.encoder_filter(mask_channel)

        if args.num_snippet == 0:
            samples = samples.to(device, non_blocking=True)
            samples = data_task.encoder_filter(samples)
            if args.mask_type in ["use_volt_current_soc", "use_volt_current_soc_temp", "use_volt_current_soc_volt", "use_volt_current_soc_max_volt"]:
                samples = samples * (1 - mask_channel)
                mask_channel = None
            input = samples
        else:
            sample_list = []
            for i in range(args.num_snippet):
                samples_i = samples[:, i, :, :]
                samples_i = samples_i.to(device, non_blocking=True)
                samples_i = data_task.encoder_filter(samples_i)
                if args.mask_type in ["use_volt_current_soc", "use_volt_current_soc_temp", "use_volt_current_soc_volt", "use_volt_current_soc_max_volt"]:
                    samples_i = samples_i * (1 - mask_channel)
                sample_list.append(samples_i)
            input = sample_list.copy()
            if args.mask_type in ["use_volt_current_soc", "use_volt_current_soc_temp", "use_volt_current_soc_volt", "use_volt_current_soc_max_volt"]:
                mask_channel = None

        output = model(input, mask_channel).squeeze(-1)
        
        if args.label_normalizer:
            assert False
            assert args.downstream in ['capacity']
            label_mean = torch.tensor(label_normalizer[0], device=targets.device)
            label_std = torch.tensor(label_normalizer[1], device=targets.device)
            loss = criterion(output * label_std + label_mean, targets.float())
        else:
            loss = criterion(output, targets.float())
            
        losses.append(loss.item())
        epoch_loss += loss.item() * len(samples)
        len_data += len(samples)

        logits.append(np.array(output.cpu()))
        car_id.append(np.array(car))
        predict.append(np.array(torch.sigmoid(output.cpu()) > 0.5))
        label.append(np.array(targets.cpu()))

        if args.downstream == 'RUL':
            current_cycle.append(np.array(batch[3]))
            rul_cycle.append(np.array(batch[4]))

        metric_logger.update(loss=loss.item())

    predict = np.concatenate(predict)
    label = np.concatenate(label)
    logits = np.concatenate(logits)
    car_id = np.concatenate(car_id)
    
    if args.downstream == 'RUL':
        current_cycle = np.concatenate(current_cycle)
        rul_cycle = np.concatenate(rul_cycle)
        return np.stack((label, car_id, logits, current_cycle, rul_cycle), axis=1), epoch_loss / len_data, predict, label
    else:
        return np.stack((label, car_id, logits), axis=1), epoch_loss / len_data, predict, label

@torch.no_grad()
def evaluate(data_loader_valid, data_loader_test, model, criterion, device, args, data_task):
    assert data_loader_valid is None
    valid_res, valid_loss, valid_predict, valid_label = get_evaluate_stats(data_loader_valid, model, criterion, device, 'valid', args=args, data_task=data_task)
    test_res, test_loss, test_predict, test_label = get_evaluate_stats(data_loader_test, model, criterion, device, 'test', args=args, data_task=data_task)
    
    AUC = util.evaluation.evaluation(valid_res, test_res, brand_num=args.brand_num, h=args.h)
    
    return {'test_auroc': AUC, 
            'test_loss': test_loss,
            'valid_auroc': 0, 
            'valid_loss': valid_loss
            }


@torch.no_grad()
def evaluate_RUL(all_res):
    if all_res is None:
        return 0., 0.
    answer = {}
    for res in all_res:
        rul = res[4]
        car = round(res[1])
        pred = res[2] * 1000.
        cycle = res[3]
        if car not in answer:
            answer[car] = [[], None]

        if answer[car][1] is None:
            answer[car][1] = rul + cycle
        else:
            assert answer[car][1] == rul + cycle

        answer[car][0].append((pred + cycle, cycle))
    
    square_error = []
    percentage_error = []
    threshold = 90
    for car, value in answer.items():
        preds = []
        for (pred, current) in value[0]:
            if current > threshold:
                preds.append(pred)
        pred = np.mean(preds)
        print(value[1], car, pred)
        square_error.append((pred - value[1]) ** 2)
        percentage_error.append(abs(pred - value[1]) / value[1])
    
    return np.sqrt(np.mean(square_error)), np.mean(percentage_error)
            
    