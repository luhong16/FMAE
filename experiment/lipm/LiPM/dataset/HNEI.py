from LiPM.dataset.basic_dataset import BasicPretrain, Basic_ir, BasicReg, BasicReg_ir, BasicDelta, BasicDelta_ir, BasicTime, RegTime, DeltaTime
from LiPM.dataset.data_utils import load_data

t_span = 460

class BasicHNEIPretrain(BasicPretrain):
    def __init__(self, datas, seq_len, stride, pred_len, pred_var) -> None:
        super().__init__(datas, seq_len, stride, pred_len, pred_var)


class BasicHNEI_ir(Basic_ir):
    def __init__(self, datas, time, patch_num, stride, pred_num=0) -> None:
        super().__init__(datas, time, patch_num, stride, pred_num, t_span=460)  


def trans_RUL(RUL):
    return RUL / 100 


def trans_SOH(soh):
    return soh


class HNETDeltaQCQD(BasicDelta):
    def __init__(self, datas, target, seq_len, stride) -> None:
        super().__init__(datas, target, seq_len, stride)


class HNEIDelatQCQD_ir(BasicDelta_ir):
    def __init__(self, datas, target, time, patch_num, stride) -> None:
        super().__init__(datas, target, time, patch_num, stride, t_span)


class HNEIRULReg(BasicReg):
    def __init__(self, datas, target, seq_len, stride):
        super().__init__(datas, target, seq_len, stride, trans_RUL)


class HNEIRULReg_ir(BasicReg_ir):
    def __init__(self, datas, target, time, patch_num, stride):
        super().__init__(datas, target, time, patch_num, stride, t_span, trans_RUL)


class HNEISOHReg(BasicReg):
    def __init__(self, datas, target, seq_len, stride) -> None:
        super().__init__(datas, target, seq_len, stride, trans_SOH)


class HNEISOHReg_ir(BasicReg_ir):
    def __init__(self, datas, target, time, patch_num, stride) -> None:
        super().__init__(datas, target, time, patch_num, stride, t_span, trans_SOH)


def load_HNEI_dataset(data_idx=0, data_config=None):
    datas, labels = load_data('HNEI', data_idx)
    return HNEIDataset(datas, labels, data_config).datasets


class HNEIDataset(object):
    def __init__(self, datas, labels, data_config):
        dataset_list = []
        pred_var = None
        if data_config['task'] == 'forecast':
            pred_var = data_config['pred_var']
        for i, data in enumerate(datas):
            if data_config['task'] in ['pretrain', 'forecast']:
                cur_dataset = BasicHNEIPretrain(data[:, [1, 0, 3, 4]], data_config['seq_len'], data_config['stride'], data_config['pred_len'], pred_var)
            elif data_config['task'] in ['pretrain_ir', 'forecast_ir']:
                cur_dataset = BasicHNEI_ir(data[:, [1, 0, 3, 4]], data[:, 2], data_config['patch_num'], data_config['stride'], data_config['pred_num'])
            elif data_config['task'] in ['RULRegression']:
                cur_dataset = HNEIRULReg(data[:, [1, 0, 3, 4]], labels[i][5], data_config['seq_len'], data_config['stride'])
            elif data_config['task'] in ['ir_RULRegression']:
                cur_dataset = HNEIRULReg_ir(data[:, [1, 0, 3, 4]], labels[i][5], data[:, 2], data_config['patch_num'], data_config['stride'])
            elif data_config['task'] in ['SOHRegression']:
                cur_dataset = HNEISOHReg(data[:, [1, 0, 3, 4]], labels[i][4], data_config['seq_len'], data_config['stride'])
            elif data_config['task'] in ['ir_SOHRegression']:
                cur_dataset = HNEISOHReg_ir(data[:, [1, 0, 3, 4]], labels[i][4], data[:, 2], data_config['patch_num'], data_config['stride'])
            elif data_config['task'] == 'DeltaQc':
                cur_dataset = HNETDeltaQCQD(data[:, [1, 0, 3, 4]], 3, data_config['seq_len'], data_config['stride'])
            elif data_config['task'] == 'ir_DeltaQc':
                cur_dataset = HNEIDelatQCQD_ir(data[:, [1, 0, 3, 4]], 3, data[:, 2], data_config['patch_num'], data_config['stride'])
            elif data_config['task'] == 'DeltaQd':
                cur_dataset = HNETDeltaQCQD(data[:, [1, 0, 3, 4]], 2, data_config['seq_len'], data_config['stride'])
            elif data_config['task'] == 'ir_DeltaQd':
                cur_dataset = HNEIDelatQCQD_ir(data[:, [1, 0, 3, 4]], 2, data[:, 2], data_config['patch_num'], data_config['stride'])

            elif data_config['task'] == 'pretrain_time':
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
                raise Exception(f'{data_config["task"]} is not implemented yet!')
            
            dataset_list.append(cur_dataset)

        self.datasets = dataset_list
        print('basic dataset num:', len(self.datasets))
    
