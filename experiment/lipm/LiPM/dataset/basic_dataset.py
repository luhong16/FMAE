import torch
import numpy as np
from torch.utils.data import Dataset


EPS = 1e-8


def min_max_norm(data, axis=0):
    min_value = np.min(data, axis=axis, keepdims=True)
    max_value = np.max(data, axis=axis, keepdims=True)
    new_data = (data - min_value)/(max_value - min_value + EPS)
    return new_data 


def abs_norm(data, axis=0):
    max_value = np.max(data.abs(), axis=axis, keepdims=True)
    new_data = data / (max_value + EPS)
    return new_data


def norm_t(ts, patch_len, t_span, t0):
    ts = np.array(ts)
    if len(ts) == 0:
        return ts
    return ts - t0

def make_ipatch(datas, time, t_span, patch_len):
    new_data = [[]]
    new_t = [[]]
    i, j = 0, 1  
    while i < len(datas):
        while time[i] > time[0] + t_span*j:
            new_t[-1] = norm_t(new_t[-1], patch_len, t_span, t_span*(j-1))
            new_data.append([])
            new_t.append([])
            j += 1
        new_data[-1].append(datas[i])
        new_t[-1].append(time[i])
        i += 1
    
    _t_mask = torch.ones((len(new_data), patch_len), dtype=torch.long) 
    _patch_mask = torch.ones(len(new_data), dtype=torch.long)
    for i in range(len(new_data)):
        _t_mask[i][len(new_data[i]):] *= 0 
        _patch_mask[i] *= 0 if len(new_data[i]) == 0 else 1
        
        padding = np.zeros((patch_len - len(new_data[i]), datas.shape[-1]))
        if len(new_data[i]) == 0:
            new_data[i] = padding
            new_t[i] = padding[:, 0]
        else:
            new_data[i] = np.concatenate((new_data[i], padding))  
            new_t[i] = np.concatenate((new_t[i], padding[:, 0]))

    new_data = torch.tensor(np.array(new_data), dtype=torch.float32)  
    new_t = torch.tensor(np.array(new_t), dtype=torch.float32)

    return new_data, new_t, _t_mask, _patch_mask


class BasicPretrain(Dataset):
    def __init__(self, datas, seq_len, stride, pred_len=0, pred_var=None) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.stride = stride if stride > 0 else seq_len
        self.pred_len = pred_len
        norm_datas = datas
        self.datas = torch.from_numpy(norm_datas).type(torch.float32)  
        self.pred_var = pred_var  

    def __len__(self):
        return max(((len(self.datas) - self.seq_len - self.pred_len) // self.stride + 1, 0))
    
    def __getitem__(self, index):
        begin_idx = index * self.stride
        end_idx = begin_idx + self.seq_len

        if self.pred_len > 0:
            pred_begin = end_idx
            pred_end = pred_begin + self.pred_len

            if self.pred_var is None:
                return self.datas[begin_idx: end_idx, [0, 1]], self.datas[pred_begin: pred_end, [0, 1]]
            elif self.pred_var == 'v':
                return self.datas[begin_idx: end_idx, [0, 1]], self.datas[pred_begin: pred_end, 0]
            elif self.pred_var == 'c':
                return self.datas[begin_idx: end_idx, [0, 1]], self.datas[pred_begin: pred_end, 1]
            else:
                raise Exception('pred_var shoulde be v or c')
        return self.datas[begin_idx: end_idx, [0, 1]]
    

class Basic_ir(Dataset):
    def __init__(self, datas, time, patch_num, stride, pred_num, t_span) -> None:
        super().__init__()
        self.patch_len = 64
        self.patch_num = patch_num
        self.pred_num = pred_num
        self.stride = stride if stride != -1 else patch_num
        norm_datas = datas
        new_datas, self.time, self.t_mask, self.patch_mask = make_ipatch(norm_datas, time, t_span, self.patch_len)
        self.datas = new_datas  

    def __len__(self):
        return max(((len(self.time) - self.patch_num - self.pred_num) // self.stride + 1, 0))
    
    def __getitem__(self, index):
        begin_idx = index*self.stride
        end_idx = begin_idx + self.patch_num

        _t_mask = self.t_mask[begin_idx: end_idx]
        emb_t_mask = torch.ones((self.patch_num, 1), dtype=torch.long) 
        t_mask = torch.cat((emb_t_mask, _t_mask), dim=1)  
        patch_mask = self.patch_mask[begin_idx: end_idx]

        _time = self.time[begin_idx: end_idx]  
        time = torch.cat((torch.zeros((self.patch_num, 1)), _time), dim=1) 

        vc_data = self.datas[begin_idx: end_idx, :, [0, 1]] 
        _QdQc_data = self.datas[begin_idx: end_idx, :, [2, 3]]
        init_Q = _QdQc_data[0, 0, :]
        QdQc_data = _QdQc_data - init_Q

        if self.pred_num > 0:
            pred_begin = end_idx
            pred_end = pred_begin + self.pred_num
            pred_vc = self.datas[pred_begin: pred_end, :, [0, 1]]
            _time = self.time[pred_begin: pred_end]
            pred_time = torch.cat((torch.zeros((self.pred_num, 1)), _time), dim=1)

            _t_mask = self.t_mask[pred_begin: pred_end]
            _emb_mask = torch.ones((self.pred_num, 1), dtype=torch.long)
            pred_t_mask = torch.cat((_emb_mask, _t_mask), dim=1) 
            pred_patch_mask = self.patch_mask[pred_begin: pred_end]

            return vc_data, time, t_mask, patch_mask,\
                pred_vc, pred_time, pred_t_mask, pred_patch_mask
            
        return vc_data, QdQc_data, time, t_mask, patch_mask
    

class BasicReg(Dataset):
    def __init__(self, datas, target, seq_len, stride, trans_func) -> None:
        super(BasicReg, self).__init__()
        self.seq_len = seq_len
        self.stride = stride if stride > 0 else seq_len
        self.target = torch.tensor(trans_func(target), dtype=torch.float32) 
        norm_datas = datas 
        self.datas = torch.from_numpy(norm_datas[:, :4]).type(torch.float32)

    def __len__(self):
        return max(((len(self.datas) - self.seq_len) // self.stride + 1, 0))
    
    def __getitem__(self, index):
        begin_idx = index*self.stride
        end_idx = begin_idx + self.seq_len

        vc_data = self.datas[begin_idx: end_idx, [0, 1]]

        return vc_data, self.target
    

class BasicReg_ir(Dataset):
    def __init__(self, datas, target, time, patch_num, stride, t_span, trans_func) -> None:
        super(BasicReg_ir, self).__init__()
        self.patch_len = 64
        self.patch_num = patch_num
        self.stride = stride if stride != -1 else patch_num
        norm_datas = datas
        new_datas, self.time, self.t_mask, self.patch_mask = make_ipatch(norm_datas, time, t_span, self.patch_len)
        self.datas = new_datas  
        self.target = torch.tensor(trans_func(target), dtype=torch.float32)  

    def __len__(self):
        return max(((len(self.time) - self.patch_num) // self.stride + 1, 0))
    
    def __getitem__(self, index):
        begin_idx = index*self.stride
        end_idx = begin_idx + self.patch_num

        _t_mask = self.t_mask[begin_idx: end_idx]
        emb_mask = torch.ones((self.patch_num, 1), dtype=torch.long) 
        t_mask = torch.cat((emb_mask, _t_mask), dim=1)  
        patch_mask = self.patch_mask[begin_idx: end_idx]

        _time = self.time[begin_idx: end_idx]  
        time = torch.cat((torch.zeros((self.patch_num, 1)), _time), dim=1) 

        vc_data = self.datas[begin_idx: end_idx, :, [0, 1]]
            
        return vc_data, time, t_mask, patch_mask, self.target


class BasicDelta(Dataset):
    def __init__(self, datas, target, seq_len, stride) -> None:
        super(BasicDelta, self).__init__()
        self.seq_len = seq_len
        self.stride = stride if stride > 0 else seq_len
        norm_datas = datas
        if isinstance(target, int):
            self.target = torch.from_numpy(norm_datas[:, target]).type(torch.float32)
        elif isinstance(target, np.ndarray):
            self.target = torch.tensor(target)
        else:
            raise NotImplementedError(f'target type not supported: {type(target)}')
        self.datas = torch.from_numpy(norm_datas[:, [0, 1]]).type(torch.float32)

    def __len__(self):
        return max(((len(self.datas) - self.seq_len) // self.stride + 1, 0))
    
    def __getitem__(self, index):
        begin_idx = index*self.stride
        end_idx = begin_idx + self.seq_len

        vc_data = self.datas[begin_idx: end_idx]
        delta_target = self.target[end_idx-1] - self.target[begin_idx]
        if torch.isnan(delta_target):
            raise Exception(f'{delta_target}, {begin_idx}, {end_idx}, {self.target[end_idx-1]}, {self.target[begin_idx]}')
        return vc_data, delta_target
     

class BasicDelta_ir(Dataset):
    def __init__(self, datas, target, time, patch_num, stride, t_span) -> None:
        super(BasicDelta_ir, self).__init__()
        self.patch_len = 64
        self.patch_num = patch_num
        self.stride = stride if stride != -1 else patch_num
        norm_datas = datas
        if isinstance(target, np.ndarray):
            norm_datas = np.concatenate((norm_datas, target.reshape(-1, 1)), axis=1)
        
        new_datas, self.time, self.t_mask, self.patch_mask = make_ipatch(norm_datas, time, t_span, self.patch_len)
        self.datas = new_datas[:, :, :2]  
        if isinstance(target, int):
            self.target = new_datas[:, :, target]
        elif isinstance(target, np.ndarray):
            self.target = new_datas[:, :, -1]  #
        else:
            raise NotImplementedError(f'target type not supported: {type(target)}')

    def __len__(self):
        return max(((len(self.time) - self.patch_num) // self.stride + 1, 0))
    
    def get_delta_target(self, target_patch: torch.Tensor, t_mask: torch.Tensor, p_mask: torch.Tensor):
        i = 0
        while i < len(p_mask):
            if p_mask[i] == 1:
                begin_target = target_patch[i][0]
                break
            i += 1
        i = len(p_mask) - 1
        while i >= 0:
            if p_mask[i] == 1:
                end_target = target_patch[i][t_mask[i].sum().item()-2]
                break
            i -= 1
        return end_target - begin_target
            
    def __getitem__(self, index):
        begin_idx = index*self.stride
        end_idx = begin_idx + self.patch_num

        _t_mask = self.t_mask[begin_idx: end_idx]
        emb_mask = torch.ones((self.patch_num, 1), dtype=torch.long) 
        t_mask = torch.cat((emb_mask, _t_mask), dim=1)  
        p_mask = self.patch_mask[begin_idx: end_idx]

        _time = self.time[begin_idx: end_idx]  
        time = torch.cat((torch.zeros((self.patch_num, 1)), _time), dim=1) 

        delta_target = self.get_delta_target(self.target[begin_idx: end_idx], t_mask, p_mask)
        vc_data = self.datas[begin_idx: end_idx]
        return vc_data, time, t_mask, p_mask, delta_target


class BasicTime(Dataset):
    def __init__(self, datas, seq_len, stride, pred_len=0, pred_var=None) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.stride = stride if stride > 0 else seq_len
        self.pred_len = pred_len
        norm_datas = datas
        self.datas = torch.from_numpy(norm_datas).type(torch.float32)
        self.pred_var = pred_var  

    def __len__(self):
        return max(((len(self.datas) - self.seq_len - self.pred_len) // self.stride + 1, 0))
    
    def __getitem__(self, index):
        begin_idx = index * self.stride
        end_idx = begin_idx + self.seq_len

        data = self.datas[begin_idx: end_idx, [0, 1, -1]]

        if self.pred_len > 0:
            pred_begin = end_idx
            pred_end = pred_begin + self.pred_len

            pred_data = self.datas[pred_begin: pred_end, [0, 1, -1]]

            return data, pred_data

        Q_data = self.datas[begin_idx: end_idx, [2, 3]]
        return data, Q_data  
    

class RegTime(Dataset):
    def __init__(self, datas, target, seq_len, stride, trans_func) -> None:
        super(RegTime, self).__init__()
        self.seq_len = seq_len
        self.stride = stride if stride > 0 else seq_len
        self.target = torch.tensor(trans_func(target), dtype=torch.float32)  
        norm_datas = datas  
        self.datas = torch.from_numpy(norm_datas[:, [0, 1, -1]]).type(torch.float32)

    def __len__(self):
        return max(((len(self.datas) - self.seq_len) // self.stride + 1, 0))
    
    def __getitem__(self, index):
        begin_idx = index*self.stride
        end_idx = begin_idx + self.seq_len

        data = self.datas[begin_idx: end_idx]

        return data, self.target
    

class DeltaTime(Dataset):
    def __init__(self, datas, target, seq_len, stride) -> None:
        super(DeltaTime, self).__init__()
        self.seq_len = seq_len
        self.stride = stride if stride > 0 else seq_len
        norm_datas = datas
        if isinstance(target, int):
            self.target = torch.from_numpy(norm_datas[:, target]).type(torch.float32)
        elif isinstance(target, np.ndarray):
            self.target = torch.tensor(target)
        else:
            raise NotImplementedError(f'target type not supported: {type(target)}')
        self.datas = torch.from_numpy(norm_datas[:, [0, 1, -1]]).type(torch.float32)

    def __len__(self):
        return max(((len(self.datas) - self.seq_len) // self.stride + 1, 0))
    
    def __getitem__(self, index):
        begin_idx = index*self.stride
        end_idx = begin_idx + self.seq_len

        data = self.datas[begin_idx: end_idx]
        delta_target = self.target[end_idx-1] - self.target[begin_idx]
        if torch.isnan(delta_target):
            raise Exception(f'{delta_target}, {begin_idx}, {end_idx}, {self.target[end_idx-1]}, {self.target[begin_idx]}')
        return data, delta_target