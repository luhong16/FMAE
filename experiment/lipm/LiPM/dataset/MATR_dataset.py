import numpy as np
from LiPM.dataset.basic_dataset import BasicPretrain, Basic_ir, BasicReg, BasicReg_ir, BasicDelta, BasicDelta_ir, BasicTime, RegTime, DeltaTime
from LiPM.dataset.data_utils import load_data


class BasicMATRPretrain(BasicPretrain):
    def __init__(self, datas, seq_len, stride, pred_len, pred_var) -> None:
        super().__init__(datas, seq_len, stride, pred_len, pred_var)      


class BasicMATR_ir(Basic_ir):
    def __init__(self, datas, time, patch_num, stride, pred_num=0) -> None:
        super().__init__(datas, time, patch_num, stride, pred_num, t_span=t_span)  


def trans_IR(IR):
    return IR

def trans_QDQC(QD):
    return QD


def trans_SOH(SOH):
    return min((SOH, 1.))

def trans_RUL(RUL):
    return RUL / 100

t_span = 74.85


class MATRIRReg(BasicReg):
    def __init__(self, datas, IR, seq_len, stride) -> None:
        super().__init__(datas, IR, seq_len, stride, trans_IR)


class MATRIRReg_ir(BasicReg_ir):
    def __init__(self, datas, IR, time, patch_num, stride) -> None:
        super().__init__(datas, IR, time, patch_num, stride, t_span, trans_IR)


class MATRRULReg(BasicReg):
    def __init__(self, datas, RUL, seq_len, stride) -> None:
        super().__init__(datas, RUL, seq_len, stride, trans_RUL)

class MATRRULReg_ir(BasicReg_ir):
    def __init__(self, datas, RUL, time, patch_num, stride) -> None:
        super().__init__(datas, RUL, time, patch_num, stride, t_span, trans_RUL)


class MATRDeltaQcQd(BasicDelta):
    def __init__(self, datas, target, seq_len, stride) -> None:
        super().__init__(datas, target, seq_len, stride)


class MATRDeltaQcQd_ir(BasicDelta_ir):
    def __init__(self, datas, target, time, patch_num, stride) -> None:
        super().__init__(datas, target, time, patch_num, stride, t_span)


class MATRSOHReg(BasicReg):
    def __init__(self, datas, target, seq_len, stride) -> None:
        super().__init__(datas, target, seq_len, stride, trans_SOH)


class MATRSOHReg_ir(BasicReg_ir):
    def __init__(self, datas, target, time, patch_num, stride) -> None:
        super().__init__(datas, target, time, patch_num, stride, t_span, trans_SOH)


def load_MATR_dataset(data_idx, data_config):
    datas, labels = load_data('MATR', data_idx)
    return MATRDataset(datas, labels, data_config).datasets

class MATRDataset(object):
    def __init__(self, datas, labels, data_config) -> None:
        dataset_list = []
        pred_var = None
        if data_config['task'] in ['forecast', 'forecastLong']:
            pred_var = data_config['pred_var']
        
        if data_config['task'] in ['IRRegression', 'ir_IRRegression'] and np.all(labels[:, 0] == 0):
            pass
        else:
            for i, data in enumerate(datas):
                if data_config['task'] in ['pretrain', 'forecast', 'forecastLong']:
                    cur_dataset = BasicMATRPretrain(data[:, [1, 0, 3, 2]], data_config['seq_len'], data_config['stride'], data_config['pred_len'], pred_var)
                elif data_config['task'] in ['pretrain_ir', 'forecast_ir', 'forecastLong_ir']:
                    cur_dataset = BasicMATR_ir(data[:, [1, 0, 3, 2]], data[:, 5], data_config['patch_num'], data_config['stride'], data_config['pred_num'])
                elif data_config['task'] in ['IRRegression']:
                    cur_dataset = MATRIRReg(data[:, [1, 0, 3, 2]], labels[i][0], data_config['seq_len'], data_config['stride'])
                elif data_config['task'] in ['ir_IRRegression']:
                    cur_dataset = MATRIRReg_ir(data[:, [1, 0, 3, 2]], labels[i][0], data[:, 5], data_config['patch_num'], data_config['stride'])       
                elif data_config['task'] in ['DeltaQc']:
                    cur_dataset = MATRDeltaQcQd(data[:, [1, 0, 3, 2]], 3, data_config['seq_len'], data_config['stride'])
                elif data_config['task'] in ['ir_DeltaQc']:
                    cur_dataset = MATRDeltaQcQd_ir(data[:, [1, 0, 3, 2]], 3, data[:, 5], data_config['patch_num'], data_config['stride'])
                elif data_config['task'] in ['DeltaQd']:
                    cur_dataset = MATRDeltaQcQd(data[:, [1, 0, 3, 2]], 2, data_config['seq_len'], data_config['stride'])
                elif data_config['task'] in ['ir_DeltaQd']:
                    cur_dataset = MATRDeltaQcQd_ir(data[:, [1, 0, 3, 2]], 2, data[:, 5], data_config['patch_num'], data_config['stride'])
                elif data_config['task'] in ['SOHRegression']:
                    cur_dataset = MATRSOHReg(data[:, [1, 0, 3, 2]], labels[i][-1], data_config['seq_len'], data_config['stride'])
                elif data_config['task'] in ['ir_SOHRegression']:
                    cur_dataset = MATRSOHReg_ir(data[:, [1, 0, 3, 2]], labels[i][-1], data[:, 5], data_config['patch_num'], data_config['stride'])
                elif data_config['task'] in ['RULRegression']:
                    cur_dataset = MATRRULReg(data[:, [1, 0, 3, 2]], labels[i][-2], data_config['seq_len'], data_config['stride'])
                elif data_config['task'] in ['ir_RULRegression']:
                    cur_dataset = MATRRULReg_ir(data[:, [1, 0, 3, 2]], labels[i][-2], data[:, 5], data_config['patch_num'], data_config['stride'])
                # time in the input
                elif data_config['task'] == 'pretrain_time':
                    cur_dataset = BasicTime(data[:, [1, 0, 3, 4, 5]], data_config['seq_len'], data_config['stride'], data_config['pred_len'], pred_var)
                elif data_config['task'] == 'RULRegression_time':
                    cur_dataset = RegTime(data[:, [1, 0, 3, 4, 5]], labels[i][-2], data_config['seq_len'], data_config['stride'], trans_RUL)
                elif data_config['task'] == 'SOHRegression_time':
                    cur_dataset = RegTime(data[:, [1, 0, 3, 4, 5]], labels[i][-1], data_config['seq_len'], data_config['stride'], trans_SOH)
                elif data_config['task'] == 'IRRegression_time':
                    cur_dataset = RegTime(data[:, [1, 0, 3, 4, 5]], labels[i][0], data_config['seq_len'], data_config['stride'], trans_IR)
                elif data_config['task'] == 'DeltaQc_time':
                    cur_dataset = DeltaTime(data[:, [1, 0, 3, 4, 5]], 3, data_config['seq_len'], data_config['stride'])
                elif data_config['task'] == 'DeltaQd_time':
                    cur_dataset = DeltaTime(data[:, [1, 0, 3, 4, 5]], 2, data_config['seq_len'], data_config['stride'])
                else:
                    task = data_config['task']
                    raise ValueError(f'{task} not supported')
                dataset_list.append(cur_dataset)

        self.datasets = dataset_list
        print('basic dataset num:', len(self.datasets))
        
