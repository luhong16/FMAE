# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
from multimae.criterion import MaskedMSELoss


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None, data_task=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 200

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    criterion = MaskedMSELoss(patch_size=16, stride=1, norm_pix=args.norm_pix_loss).to(device)

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        samples = batch[0]
        lab = batch[3]
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
        B = samples.shape[0]
        
        # unavaliable_channel = torch.tensor([0,0,0,1,1,1,1,0,0], dtype=torch.float32, device=device).repeat(B, 1)
        # unavaliable_channel = (unavaliable_channel * lab.to(device).unsqueeze(-1)).unsqueeze(-1)
        # unavaliable_channel = data_task.encoder_filter(unavaliable_channel).squeeze(-1)

        if args.num_snippet == 0:
            samples = samples.to(device, non_blocking=True)
            samples = data_task.encoder_filter(samples)
            # input, target = samples, samples
        else:
            # MultiMAE currently supports single snippet input in this adaptation
            # If num_snippet > 0, we might need to handle it differently or just take the first one for now
            # or adapt MultiMAE to handle multiple snippets (which would be multiple tasks or a larger sequence)
            # For now, let's assume num_snippet=0 or we take the first one.
            # But wait, the original code did:
            # sample_list = []
            # for i in range(args.num_snippet):
            #     samples_i = samples[:, i, :, :]
            #     ...
            #     sample_list.append(samples_i)
            # input, target = sample_list.copy(), sample_list.copy()
            
            # If we want to support num_snippet > 0, we need to know how MultiMAE should handle it.
            # Assuming for now we just process one sequence.
            samples = samples[:, 0, :, :] if samples.ndim == 4 else samples
            samples = samples.to(device, non_blocking=True)
            samples = data_task.encoder_filter(samples)

        # input_dict = {'battery': samples}
        input_dict = {}
        for i, idx in enumerate(data_task.encoder):
            channel_name = data_task.columns[idx]
            input_dict[channel_name] = samples[:, i:i+1, :]
        
        # Calculate num_encoded_tokens
        L = samples.shape[-1]
        patch_size = 16 # Hardcoded for now, should match model
        num_patches_per_channel = L // patch_size
        total_tokens = len(data_task.encoder) * num_patches_per_channel
        num_encoded_tokens = int(total_tokens * (1 - args.mask_patch_ratio))
        
        with torch.cuda.amp.autocast():
            # loss, _, _, _ = model(input, target, mask_patch_ratio=args.mask_patch_ratio, mask_channel_ratio=args.mask_channel_ratio, unavaliable_channel=unavaliable_channel)
            # print(model(input_dict, mask_inputs=True, num_encoded_tokens=num_encoded_tokens))
            preds, task_masks = model(input_dict, mask_inputs=True, num_encoded_tokens=num_encoded_tokens)
            # loss = criterion(preds['battery'], samples, task_masks['battery'])
            loss = 0
            for channel_name in input_dict.keys():
                loss += criterion(preds[channel_name], input_dict[channel_name], task_masks[channel_name])
            loss = loss / len(input_dict)
            '''
            if data_iter_step < 100:
                print(f"preds:{preds}, task_masks:{task_masks}")
                print(f"loss:{loss}, input_dict:{input_dict}")
            '''

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        grad_norm = loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        # print("grad_norm:", grad_norm)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
