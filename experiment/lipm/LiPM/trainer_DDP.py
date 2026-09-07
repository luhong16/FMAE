import os
from pathlib import Path
cur_file_path = Path(__file__).resolve()
import sys
sys.path.insert(0, str(cur_file_path.parent.parent))
from LiPM.battery_model import BatteryNet
from LiPM.utils import get_pretrain_dataset, make_save, save_model, LossRecord
from LiPM.default_config import load_setting

import torch
import traceback
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    torch.manual_seed(3407)


def cleanup():
    dist.destroy_process_group()


def batch_cuda(batch):
    for i in range(len(batch)):
        if isinstance(batch[i], torch.Tensor):
            batch[i] = batch[i].cuda()
    
    return batch


def make_log(config):
    config.save_path = 'log/LiPM/pretrain'
    name = f'{config.n_layer}_{config.n_head}_{config.d_model}_{config.weight_MAE}_{config.weight_Q}_{config.patch_ratio}_{config.channel_ratio}'
    config.save_path = make_save(config.save_path, name, append_log=False)
    for k in config.__dict__:
        print(k, ':', config.__dict__[k])


def pretrain_func(rank, world_size, config):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)  
    torch.cuda.set_device(rank)  

    net = BatteryNet(config).cuda()
    net_ddp = DDP(net, device_ids=[rank], find_unused_parameters=False)

    if config.ck_point != -1:
        raise NotImplementedError('load model not implemented')
    print('cuda:', rank, 'load model ok')
    opt = torch.optim.AdamW(net_ddp.parameters(), lr=config.lr, weight_decay=config.l2, betas=[0.9, 0.95])
    scheduler = CosineAnnealingWarmRestarts(opt, T_0=config.T_0)

    cache_pretrain_dataset_path = 'cache/pretrain_default_dataset.pt'
    if config.cache and os.path.exists(cache_pretrain_dataset_path):
        dataset = torch.load(cache_pretrain_dataset_path)
    else:
        dataset = get_pretrain_dataset(config)
        if config.cache:
            torch.save(dataset, cache_pretrain_dataset_path)

    train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, sampler=train_sampler, pin_memory=True)
    data_iter =  iter(dataloader)
    print('rank', rank, 'dataset ok', len(dataset))
    
    iter_count = 0
    epoch_count = 0
    loss_record = LossRecord(['total', 'MAE', 'Q', 'MAE_mae', 'Q_mae'])

    net_ddp.train()
    while iter_count < config.max_iter:
        iter_count += 1
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch_count += 1
            if epoch_count >= config.max_epoch:
                break
            train_sampler.set_epoch(epoch_count)
            data_iter = iter(dataloader)
            batch = next(data_iter)
        total_loss, loss_dict = net_ddp(batch_cuda(batch))
        opt.zero_grad()
        total_loss.backward()
        opt.step()
        
        loss_record.append(loss_dict)

        if iter_count % config.show_iter == config.show_iter-1:
            if rank == 0:
                # print loss
                loss_record.print(epoch_count, iter_count, config.show_iter)

                save_iter = config.show_iter * config.save_iter
                if (iter_count % save_iter == save_iter - 1) and (config.log == 1):
                    save_model(net_ddp.module, config, iter_count)
                    print('++++++++++++++++++++')
        scheduler.step()
        dist.barrier() 
    
    if rank == 0 and config.log == 1:
        loss_record.save(config.save_path, iter_count+config.ck_point+1)


def pretrain(rank, world_size, config):
    if rank == 0 and config.log == 1:
        make_log(config)
        try:
            pretrain_func(rank, world_size, config)
        except Exception as e:
            traceback.print_exc()
            cleanup()
            raise
    else:
        pretrain_func(rank, world_size, config)

def run_demo(demo_fn, world_size, config):
    mp.spawn(demo_fn,
             args=(world_size, config),
             nprocs=world_size,  
             join=True)


parser = argparse.ArgumentParser()
# exp setting
parser.add_argument('--ck_point', type=int, default=-1)  
parser.add_argument('--cuda', type=str, default='1')
parser.add_argument('--last_idx', type=int, default=-1)
parser.add_argument('--log', type=int, default=0, help='1 for record log')
parser.add_argument('--save_path', type=str, default=None)  
parser.add_argument('--save_iter', type=int, default=10)  
parser.add_argument('--show_iter', type=int, default=200)
parser.add_argument('--default_setting', type=int, default=-1, help='-1 for No default setting')

# train
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--l2', type=float, default=0.001)
parser.add_argument('--max_iter', type=int, default=50000)
parser.add_argument('--max_epoch', type=int, default=10)
parser.add_argument('--T_0', type=int, default=10, help='for CosineAnnealingWarmRestarts')
parser.add_argument('--weight_MAE', type=float, default=1.0, help='weight for MAE loss')
parser.add_argument('--weight_Q', type=float, default=1.0, help='weight for Q loss')
# data
parser.add_argument('--batch_size', type=int, default=128)  # 96, 256
parser.add_argument('--cache', type=int, default=1, help='1 for cache, 0 for not cache')
parser.add_argument('--channel_ratio', type=float, default=0.3, help='ratio of channel_mask')
parser.add_argument('--patch_ratio', type=float, default=0.3, help='ratio of patch_mask')
parser.add_argument('--patch_len', type=int, default=64)
parser.add_argument('--patch_num', type=int, default=16)
parser.add_argument('--patch_stride', type=int, default=-1, help="-1 means stride == patch_len")

# model
## main
parser.add_argument('--d_model', type=int, default=512)  # 1024, 512
parser.add_argument('--dp', type=float, default=0.3)
parser.add_argument('--n_head', type=int, default=8) 
parser.add_argument('--n_layer', type=int, default=12)  # 24, 20
parser.add_argument('--n_var', type=int, default=2)
parser.add_argument('--norm', type=str, default='rsm')
parser.add_argument('--pre_norm', type=int, default=1, help='1 for True, 0 for False')
parser.add_argument('--task', type=str, default='ir_pretrain')
## ssl task
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
    n_gpus  = torch.cuda.device_count()
    world_size = n_gpus  

    save_path = '/log/Battery/pub'
    if config.log not in [0, 1]:
        raise Exception('--log should be 1 for logging or 0 for not logging')
    if config.log == 1:
        make_log(config)
    run_demo(pretrain, world_size, config)
    