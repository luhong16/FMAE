from LiPM.dataset.basic_dataset import BasicPretrain, Basic_ir, BasicReg, BasicReg_ir, BasicDelta, BasicDelta_ir, BasicTime, RegTime, DeltaTime
from LiPM.dataset.data_utils import load_data


t_span = 120

def trans_SOH(SOH):
    return min((SOH, 1.))  # max soh is 100%

def trans_RUL(RUL):
    return RUL / 10


class BasicNASAPretrain(BasicPretrain):
    def __init__(self, datas, seq_len, stride, pred_len, pred_var) -> None:
        super().__init__(datas, seq_len, stride, pred_len, pred_var)


class BasicNASA_ir(Basic_ir):
    # make patch first
    def __init__(self, datas, time, patch_num, stride, pred_num=0) -> None:
        super().__init__(datas, time, patch_num, stride, pred_num, t_span=t_span)  


class NASASOHReg(BasicReg):
    def __init__(self, datas, target, seq_len, stride) -> None:
        super().__init__(datas, target, seq_len, stride, trans_SOH)


class NASASOHReg_ir(BasicReg_ir):
    def __init__(self, datas, target, time, patch_num, stride) -> None:
        super().__init__(datas, target, time, patch_num, stride, t_span, trans_SOH)


class NASARULReg(BasicReg):
    def __init__(self, datas, target, seq_len, stride) -> None:
        super().__init__(datas, target, seq_len, stride, trans_RUL)


class NASARULReg_ir(BasicReg_ir):
    def __init__(self, datas, target, time, patch_num, stride) -> None:
        super().__init__(datas, target, time, patch_num, stride, t_span, trans_RUL)


class NASADeltaQCQD(BasicDelta):
    def __init__(self, datas, target, seq_len, stride) -> None:
        super().__init__(datas, target, seq_len, stride)


class NASADeltaQCQD_ir(BasicDelta_ir):
    def __init__(self, datas, target, time, patch_num, stride) -> None:
        super().__init__(datas, target, time, patch_num, stride, t_span)


data_names = ['B0005', 'B0006', 'B0007', 'B0018', 'B0025', 
      'B0026', 'B0027', 'B0028', 'B0029', 'B0030', 
      'B0031', 'B0032', 'B0033', 'B0034', 'B0036', 
      'B0038', 'B0039', 'B0040', 'B0041', 'B0042', 
      'B0043', 'B0044', 'B0045', 'B0046', 'B0047', 
      'B0048', 'B0049', 'B0050', 'B0051', 'B0052', 
      'B0053', 'B0054', 'B0055', 'B0056']

def load_NASA_dataset(data_idx=0, data_config=None):
    datas, labels = load_data('NASA', data_idx)
    return NASADataset(datas, labels, data_config).datasets


class NASADataset(object):
    def __init__(self, datas, labels, data_config):
        dataset_list = []
        pred_var = None
        if data_config['task'] in ['forecast', 'forecastLong']:
            pred_var = data_config['pred_var']
        for i, data in enumerate(datas):
            # data: (L, 3)
            if data_config['task'] in ['pretrain', 'forecast', 'forecastLong']:
                cur_dataset = BasicNASAPretrain(data[:, [1, 0, 3, 4]], data_config['seq_len'], data_config['stride'], data_config['pred_len'], pred_var)
            elif data_config['task'] in ['pretrain_ir', 'forecast_ir', 'forecastLong_ir']:
                cur_dataset = BasicNASA_ir(data[:, [1, 0, 3, 4]], data[:, 2], data_config['patch_num'], data_config['stride'], data_config['pred_num'])
            elif data_config['task'] == 'SOHRegression':
                if labels[i][1] <= 0:
                    continue
                cur_dataset = NASASOHReg(data[:, [1, 0, 3, 4]], labels[i][2], data_config['seq_len'], data_config['stride'])
            elif data_config['task'] == 'ir_SOHRegression':
                if labels[i][1] <= 0:
                    continue
                cur_dataset = NASASOHReg_ir(data[:, [1, 0, 3, 4]], labels[i][2], data[:, 2], data_config['patch_num'], data_config['stride'])
            elif data_config['task'] == 'RULRegression':
                if labels[i][1] <= 0:
                    continue
                cur_dataset = NASARULReg(data[:, [1, 0, 3, 4]], labels[i][3], data_config['seq_len'], data_config['stride'])
            elif data_config['task'] == 'ir_RULRegression':
                if labels[i][1]<= 0:
                    continue
                cur_dataset = NASARULReg_ir(data[:, [1, 0, 3, 4]], labels[i][3], data[:, 2], data_config['patch_num'], data_config['stride'])
            elif data_config['task'] == 'DeltaQc':
                cur_dataset = NASADeltaQCQD(data[:, [1, 0, 3, 4]], 3, data_config['seq_len'], data_config['stride'])
            elif data_config['task'] == 'ir_DeltaQc':
                cur_dataset = NASADeltaQCQD_ir(data[:, [1, 0, 3, 4]], 3, data[:, 2], data_config['patch_num'], data_config['stride'])
            elif data_config['task'] == 'DeltaQd':
                cur_dataset = NASADeltaQCQD(data[:, [1, 0, 3, 4]], 2, data_config['seq_len'], data_config['stride'])
            elif data_config['task'] == 'ir_DeltaQd':
                cur_dataset = NASADeltaQCQD_ir(data[:, [1, 0, 3, 4]], 2, data[:, 2], data_config['patch_num'], data_config['stride'])
            elif data_config['task'] == 'pretrain_time':
                cur_dataset = BasicTime(data[:, [1, 0, 3, 4, 2]], data_config['seq_len'], data_config['stride'], data_config['pred_len'], pred_var)
            elif data_config['task'] == 'RULRegression_time':
                if labels[i][1] <= 0:
                    continue
                cur_dataset = RegTime(data[:, [1, 0, 3, 4, 2]], labels[i][3], data_config['seq_len'], data_config['stride'], trans_RUL)
            elif data_config['task'] == 'SOHRegression_time':
                if labels[i][1] <= 0:
                    continue
                cur_dataset = RegTime(data[:, [1, 0, 3, 4, 2]], labels[i][2], data_config['seq_len'], data_config['stride'], trans_SOH)
            elif data_config['task'] == 'DeltaQc_time':
                cur_dataset = DeltaTime(data[:, [1, 0, 3, 4, 2]], 3, data_config['seq_len'], data_config['stride'])
            elif data_config['task'] == 'DeltaQd_time':
                cur_dataset = DeltaTime(data[:, [1, 0, 3, 4, 2]], 2, data_config['seq_len'], data_config['stride'])
            else:
                raise NotImplementedError(f'{data_config["task"]} is not implemented!')
            dataset_list.append(cur_dataset)

        self.datasets = dataset_list
        print('basic dataset num:', len(self.datasets))
    
