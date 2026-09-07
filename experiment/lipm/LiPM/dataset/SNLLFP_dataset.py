from LiPM.dataset.basic_dataset import BasicPretrain, Basic_ir, BasicDelta, BasicReg_ir, BasicDelta_ir, BasicReg, BasicTime, RegTime, DeltaTime
from LiPM.dataset.data_utils import load_data


t_span = 160

class BasicSNLLFPPretrain(BasicPretrain):
    def __init__(self, datas, seq_len, stride, pred_len, pred_var) -> None:
        super().__init__(datas, seq_len, stride, pred_len, pred_var)


class BasicSNLLFP_ir(Basic_ir):
    # make patch first
    def __init__(self, datas, time, patch_num, stride, pred_num=0) -> None:
        super().__init__(datas, time, patch_num, stride, pred_num, t_span=t_span)  


def trans_SOH(SOH):
    return min((SOH, 1.))  # max soh is 100%

def trans_RUL(RUL):
    return RUL / 1000


def trans_QD_QC(Q):
    # Q = Q / 50
    return Q

def trans_EC_ED(E):
    # return (E - 80) / 60  # mean, std
    return E


class SNLLFPEDECReg(BasicReg):
    def __init__(self, datas, target, seq_len, stride) -> None:
        super().__init__(datas, target, seq_len, stride, trans_EC_ED)


class SNLLFPEDECReg_ir(BasicReg_ir):
    def __init__(self, datas, target, time, patch_num, stride) -> None:
        super().__init__(datas, target, time, patch_num, stride, t_span, trans_EC_ED)


class SNLLFPDelta(BasicDelta):
    def __init__(self, datas, target, seq_len, stride) -> None:
        super().__init__(datas, target, seq_len, stride)


class SNLLFPDelta_ir(BasicDelta_ir):
    def __init__(self, datas, target, time, patch_num, stride) -> None:
        super().__init__(datas, target, time, patch_num, stride, t_span)


class SNLLFPSOHReg(BasicReg):
    def __init__(self, datas, target, seq_len, stride) -> None:
        super().__init__(datas, target, seq_len, stride, trans_SOH)


class SNLLFPSOHReg_ir(BasicReg_ir):
    def __init__(self, datas, target, time, patch_num, stride) -> None:
        super().__init__(datas, target, time, patch_num, stride, t_span, trans_SOH)


class SNLLFPRULReg(BasicReg):
    def __init__(self, datas, target, seq_len, stride) -> None:
        super().__init__(datas, target, seq_len, stride, trans_RUL)


class SNLLFPRULReg_ir(BasicReg_ir):
    def __init__(self, datas, target, time, patch_num, stride) -> None:
        super().__init__(datas, target, time, patch_num, stride, t_span, trans_RUL)


def load_SNLLFP_dataset(data_idx=0, data_config=None):
    datas, labels = load_data('SNLLFP', data_idx)
    return SNLLFPDataset(datas, labels, data_config).datasets


class SNLLFPDataset(object):
    def __init__(self, datas, labels, data_config):
        dataset_list = []
        pred_var = None
        if data_config['task'] == 'forecast':
            pred_var = data_config['pred_var']
        for i, data in enumerate(datas):
            if data_config['task'] in ['pretrain', 'forecast']:
                cur_dataset = BasicSNLLFPPretrain(data[:, [1, 0, 3, 4]], data_config['seq_len'], data_config['stride'], data_config['pred_len'], pred_var)
            elif data_config['task'] in ['pretrain_ir', 'forecast_ir']:
                cur_dataset = BasicSNLLFP_ir(data[:, [1, 0, 3, 4]], data[:, 2], data_config['patch_num'], data_config['stride'], data_config['pred_num'])
            elif data_config['task'] == 'SOHRegression':
                cur_dataset = SNLLFPSOHReg(data[:, [1, 0, 3, 4]], labels[i, 4], data_config['seq_len'], data_config['stride'])
            elif data_config['task'] == 'ir_SOHRegression':
                cur_dataset = SNLLFPSOHReg_ir(data[:, [1, 0, 3, 4]], labels[i, 4], data[:, 2], data_config['patch_num'], data_config['stride'])
            elif data_config['task'] == 'RULRegression':
                if labels[i, 5] < 0:
                    continue
                cur_dataset = SNLLFPRULReg(data[:, [1, 0, 3, 4]], labels[i, 5], data_config['seq_len'], data_config['stride'])
            elif data_config['task'] == 'ir_RULRegression':
                if labels[i, 5] < 0:
                    continue
                cur_dataset = SNLLFPRULReg_ir(data[:, [1, 0, 3, 4]], labels[i, 5], data[:, 2], data_config['patch_num'], data_config['stride'])
            elif data_config['task'] =='DeltaQc':
                cur_dataset = SNLLFPDelta(data[:, [1, 0, 3, 4]], 3, data_config['seq_len'], data_config['stride'])
            elif data_config['task'] == 'ir_DeltaQc':
                cur_dataset = SNLLFPDelta_ir(data[:, [1, 0, 3, 4]], 3, data[:, 2], data_config['patch_num'], data_config['stride'])
            elif data_config['task'] == 'DeltaQd':
                cur_dataset = SNLLFPDelta(data[:, [1, 0, 3, 4]], 2, data_config['seq_len'], data_config['stride'])
            elif data_config['task'] == 'ir_DeltaQd':
                cur_dataset = SNLLFPDelta_ir(data[:, [1, 0, 3, 4]], 2, data[:, 2], data_config['patch_num'], data_config['stride'])
            # time in the input data
            elif data_config['task'] == 'pretrain_time':
                # [V, I, Qd, Qc, t]
                cur_dataset = BasicTime(data[:, [1, 0, 3, 4, 2]], data_config['seq_len'], data_config['stride'], data_config['pred_len'], pred_var)
            elif data_config['task'] == 'RULRegression_time':
                cur_dataset = RegTime(data[:, [1, 0, 3, 4, 2]], labels[i][5], data_config['seq_len'], data_config['stride'], trans_RUL)
            elif data_config['task'] == 'SOHRegression_time':
                cur_dataset = RegTime(data[:, [1, 0, 3, 4, 2]], labels[i][4], data_config['seq_len'], data_config['stride'], trans_SOH)
            elif data_config['task'] == 'DeltaQc_time':
                cur_dataset = DeltaTime(data[:, [1, 0, 3, 4, 2]], 3, data_config['seq_len'], data_config['stride'])
            elif data_config['task'] == 'DeltaQd_time':
                cur_dataset = DeltaTime(data[:, [1, 0, 3, 4, 2]], 2, data_config['seq_len'], data_config['stride'])
            else:
                raise NotImplementedError(f'{data_config["task"]} is not implemented')
            dataset_list.append(cur_dataset)

        self.datasets = dataset_list
        print('basic dataset num:', len(self.datasets))
    
