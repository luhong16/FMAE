import os
from pathlib import Path
cur_file_path = Path(__file__).resolve()
import sys
sys.path.insert(0, str(cur_file_path.parent.parent))
from LiPM.battery_model import BatteryNet
from LiPM.utils import get_pretrain_dataset, make_save, load_model, save_model, LossRecord
from LiPM.default_config import load_setting
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import argparse


def batch_cuda(batch):
    for i in range(len(batch)):
        if isinstance(batch[i], torch.Tensor):
            batch[i] = batch[i].cuda()
    
    return batch

def pretrain(config):
    net = BatteryNet(config).cuda()

    if config.ck_point > 0:
        load_model(net, config)
    print('init model ok')
    opt = torch.optim.AdamW(net.parameters(), lr=config.lr, weight_decay=config.l2, betas=[0.9, 0.95])
    scheduler = CosineAnnealingWarmRestarts(opt, T_0=config.T_0)

    cache_pretrain_dataset_path = 'cache_path/pretrain_default_dataset.pt'
    if config.cache and os.path.exists(cache_pretrain_dataset_path):
        dataset = torch.load(cache_pretrain_dataset_path)
    else:
        dataset = get_pretrain_dataset(config)
        if config.cache:
            torch.save(dataset, cache_pretrain_dataset_path)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    data_iter =  iter(dataloader)
    print('dataset ok', len(dataset))

    iter_count = 0
    epoch_count = 0
    loss_record = LossRecord(['total', 'MMAE_mse', 'CIR_mse', 'MMAE_mae', 'CIR_mae'])

    net.train()
    while iter_count < config.max_iter:
        iter_count += 1
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch_count += 1
            if epoch_count >= config.max_epoch:
                print('reach max_iter')
                break
            dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        total_loss, loss_dict = net(batch_cuda(batch))
        opt.zero_grad()
        total_loss.backward()
        opt.step()

        loss_record.append(loss_dict)

        if iter_count % config.show_iter == config.show_iter-1:
            loss_record.print(epoch_count, iter_count, config.show_iter)
            save_iter = config.show_iter * config.save_iter
            if (iter_count % save_iter == save_iter - 1) and (config.log == 1):
                save_model(net, config, iter_count)
                print(f'save model at interation {iter_count}')
                print('++++++++++++++++++++')
            scheduler.step()
    if config.log == 1:
        loss_record.save(config.save_path, iter_count+config.ck_point+1)


parser = argparse.ArgumentParser()
parser.add_argument('--ck_point', type=int, default=-1)  
parser.add_argument('--cuda', type=str, default='1')
parser.add_argument('--last_idx', type=int, default=-1)
parser.add_argument('--log', type=int, default=0, help='1 for record log')
parser.add_argument('--save_path', type=str, default=None)  
parser.add_argument('--save_iter', type=int, default=10)  
parser.add_argument('--show_iter', type=int, default=200)
parser.add_argument('--default_setting', type=int, default=-1, help='-1 for No default setting')
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--l2', type=float, default=0.001)
parser.add_argument('--max_iter', type=int, default=50000)
parser.add_argument('--max_epoch', type=int, default=10)
parser.add_argument('--T_0', type=int, default=10, help='for CosineAnnealingWarmRestarts')
parser.add_argument('--weight_MAE', type=float, default=1.0, help='weight for MAE loss')
parser.add_argument('--weight_Q', type=float, default=1.0, help='weight for Q loss')
parser.add_argument('--batch_size', type=int, default=256)  
parser.add_argument('--cache', type=int, default=1, help='1 for cache, 0 for not cache')
parser.add_argument('--channel_ratio', type=float, default=0.3, help='ratio of channel_mask')
parser.add_argument('--patch_ratio', type=float, default=0.3, help='ratio of patch_mask')
parser.add_argument('--patch_len', type=int, default=64)
parser.add_argument('--patch_num', type=int, default=16)
parser.add_argument('--patch_stride', type=int, default=-1, help="-1 means stride == patch_len")

parser.add_argument('--d_model', type=int, default=512) 
parser.add_argument('--dp', type=float, default=0.3)
parser.add_argument('--n_head', type=int, default=8) 
parser.add_argument('--n_layer', type=int, default=12) 
parser.add_argument('--n_var', type=int, default=2)
parser.add_argument('--norm', type=str, default='rsm')
parser.add_argument('--pre_norm', type=int, default=1, help='1 for True, 0 for False')
parser.add_argument('--task', type=str, default='ir_pretrain')
parser.add_argument('--down_dim', type=int, default=256)
parser.add_argument('--down_n_head', type=int, default=8, help='number of head in ssl and downstream task')


def set_config(config):
    if config.default_setting != -1:
        load_setting(config, config.default_setting)
    config.emb_dim = config.d_model
    assert config.d_model % config.n_head == 0, 'd_model % n_head != 0'
    config.q_k_dim = config.d_model // config.n_head
    config.v_dim = config.q_k_dim


if __name__ == "__main__":
    config = parser.parse_args()
    set_config(config)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.cuda)

    save_path = 'save_path/LiPM/pretrain'
    if config.log not in [0, 1]:
        raise Exception('--log should be 1 for logging or 0 for not logging')
    if config.log == 1:
        name = f'{config.n_layer}_{config.n_head}_{config.d_model}_{config.weight_MAE}_{config.weight_Q}_{config.patch_ratio}_{config.channel_ratio}'
        if config.save_path is None or config.save_path == '':
            config.save_path = make_save(save_path, name, append_log=False)
        else:
            if config.ck_point == -1:
                raise Exception('--save_path should be empty when --ck_point == -1')
            config.save_path = make_save(config.save_path, name, append_log=True, idx=config.last_idx)
    
    print(config.__dict__)
    pretrain(config)