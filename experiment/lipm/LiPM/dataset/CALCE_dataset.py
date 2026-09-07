from LiPM.dataset.basic_dataset import BasicPretrain, Basic_ir, BasicReg, BasicReg_ir, BasicDelta_ir, BasicDelta, BasicTime, RegTime, DeltaTime
from LiPM.dataset.data_utils import load_data

# t_span = 320
t_span = 495

class BasicCALCEPretrain(BasicPretrain):
    def __init__(self, datas, seq_len, stride, pred_len, pred_var) -> None:
        super().__init__(datas, seq_len, stride, pred_len, pred_var)


class BasicCALCE_ir(Basic_ir):
    def __init__(self, datas, time, patch_num, stride, pred_num=0) -> None:
        super().__init__(datas, time, patch_num, stride, pred_num, t_span)  


def trans_IR(IR):
    return IR


def trans_QD_QC(Q):
    return Q

def trans_EC_ED(E):
    
    return E


class CALCEIRReg(BasicReg):
    def __init__(self, datas, target, seq_len, stride) -> None:
        super().__init__(datas, target, seq_len, stride, trans_IR)


class CALCEIRReg_ir(BasicReg_ir):
    def __init__(self, datas, target, time, patch_num, stride) -> None:
        super().__init__(datas, target, time, patch_num, stride, t_span, trans_IR)


class CALCEEDECReg(BasicReg):
    def __init__(self, datas, target, seq_len, stride) -> None:
        super().__init__(datas, target, seq_len, stride, trans_EC_ED)


class CALCEEDECReg_ir(BasicReg_ir):
    def __init__(self, datas, target, time, patch_num, stride) -> None:
        super().__init__(datas, target, time, patch_num, stride, t_span, trans_EC_ED)


class CALCEDelta(BasicDelta):
    def __init__(self, datas, target, seq_len, stride) -> None:
        super().__init__(datas, target, seq_len, stride)


class CALCEDelta_ir(BasicDelta_ir):
    def __init__(self, datas, target, time, patch_num, stride) -> None:
        super().__init__(datas, target, time, patch_num, stride, t_span)


def load_CALCE_dataset(data_idx=0, data_config=None):
    datas, labels = load_data('CALCE', data_idx)
    return CALCEDataset(datas, labels, data_config).datasets


class CALCEDataset(object):
    def __init__(self, datas, labels, data_config):
        dataset_list = []
        pred_var = None
        if data_config['task'] in ['forecast', 'forecastLong']:
            pred_var = data_config['pred_var']
        for i, data in enumerate(datas):
            if data_config['task'] in ['pretrain', 'forecast', 'forecastLong']:
                cur_dataset = BasicCALCEPretrain(data[:, [1, 0, 4, 3]], data_config['seq_len'], data_config['stride'], data_config['pred_len'], pred_var)
            elif data_config['task'] in ['pretrain_ir', 'forecast_ir', 'forecastLong_ir']:
                cur_dataset = BasicCALCE_ir(data[:, [1, 0, 4, 3]], data[:, 2], data_config['patch_num'], data_config['stride'], data_config['pred_num'])
            elif data_config['task'] == 'IRRegression':
                if labels[i][1] <= 0:
                    continue
                cur_dataset = CALCEIRReg(data[:, [1, 0, 4, 3]], labels[i][1], data_config['seq_len'], data_config['stride'])
            elif data_config['task'] == 'ir_IRRegression':
                if labels[i][1] <= 0:
                    continue
                cur_dataset = CALCEIRReg_ir(data[:, [1, 0, 4, 3]], labels[i][1], data[:, 2], data_config['patch_num'], data_config['stride'])
            elif data_config['task'] == 'DeltaQc':
                cur_dataset = CALCEDelta(data[:, [1, 0, 4, 3]], 3, data_config['seq_len'], data_config['stride'])
            elif data_config['task'] == 'ir_DeltaQc':
                cur_dataset = CALCEDelta_ir(data[:, [1, 0, 4, 3]], 3, data[:, 2], data_config['patch_num'], data_config['stride'])
            elif data_config['task'] == 'DeltaQd':
                cur_dataset = CALCEDelta(data[:, [1, 0, 4, 3]], 2, data_config['seq_len'], data_config['stride'])
            elif data_config['task'] == 'ir_DeltaQd':
                cur_dataset = CALCEDelta_ir(data[:, [1, 0, 4, 3]], 2, data[:, 2], data_config['patch_num'], data_config['stride'])                       
            elif data_config['task'] == 'DeltaEc':
                cur_dataset = CALCEDelta(data[:, [1, 0, 4, 3]], data[:, 5], data_config['seq_len'], data_config['stride'])
            elif data_config['task'] == 'ir_DeltaEc':
                cur_dataset = CALCEDelta_ir(data[:, [1, 0, 4, 3]], data[:, 5], data[:, 2], data_config['patch_num'], data_config['stride'])
            elif data_config['task'] == 'DeltaEd':
                cur_dataset = CALCEDelta(data[:, [1, 0, 4, 3]], data[:, 6], data_config['seq_len'], data_config['stride'])
            elif data_config['task'] == 'ir_DeltaEd':
                cur_dataset = CALCEDelta_ir(data[:, [1, 0, 4, 3]], data[:, 6], data[:, 2], data_config['patch_num'], data_config['stride'])
            elif data_config['task'] == 'pretrain_time':
                cur_dataset = BasicTime(data[:, [1, 0, 4, 3, 2]], data_config['seq_len'], data_config['stride'], data_config['pred_len'], pred_var)
            elif data_config['task'] == 'IRRegression_time':
                if labels[i][1] <= 0:
                    continue
                cur_dataset = RegTime(data[:, [1, 0, 4, 3, 2]], labels[i][1], data_config['seq_len'], data_config['stride'], trans_IR)
            elif data_config['task'] == 'DeltaQc_time':
                cur_dataset = DeltaTime(data[:, [1, 0, 4, 3, 2]], 3, data_config['seq_len'], data_config['stride'])
            elif data_config['task'] == 'DeltaQd_time':
                cur_dataset = DeltaTime(data[:, [1, 0, 4, 3, 2]], 2, data_config['seq_len'], data_config['stride'])
            elif data_config['task'] == 'DeltaEc_time':
                cur_dataset = DeltaTime(data[:, [1, 0, 4, 3, 2]], data[:, 5], data_config['seq_len'], data_config['stride'])
            elif data_config['task'] == 'DeltaEd_time':
                cur_dataset = DeltaTime(data[:, [1, 0, 4, 3, 2]], data[:, 6], data_config['seq_len'], data_config['stride'])
            else:
                raise NotImplementedError('not implemented task')

            dataset_list.append(cur_dataset)

        self.datasets = dataset_list
        print('basic dataset num:', len(self.datasets))
    
