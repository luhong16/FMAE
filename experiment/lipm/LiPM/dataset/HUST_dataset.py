from LiPM.dataset.basic_dataset import BasicPretrain, Basic_ir, BasicReg, BasicReg_ir, BasicDelta, BasicDelta_ir, BasicTime, RegTime, DeltaTime
from LiPM.dataset.data_utils import load_data


t_span = 90


class BasicHUSTPretrain(BasicPretrain):
    def __init__(self, datas, seq_len, stride, pred_len, pred_var) -> None:
        super().__init__(datas, seq_len, stride, pred_len, pred_var)


class BasicHUST_ir(Basic_ir):
    def __init__(self, datas, time, patch_num, stride, pred_num=0) -> None:
        super().__init__(datas, time, patch_num, stride, pred_num, t_span)  


def trans_QD(QD):
    return QD


def trans_SOH(soh):
    return soh


def trans_RUL(RUL):
    return RUL/1000 


class HUSTQDReg(BasicReg):
    def __init__(self, datas, target, seq_len, stride) -> None:
        super().__init__(datas, target, seq_len, stride, trans_QD)


class HUSTQDReg_ir(BasicReg_ir):
    def __init__(self, datas, target, time, patch_num, stride) -> None:
        super().__init__(datas, target, time, patch_num, stride, t_span, trans_QD)


class HUSTRULReg(BasicReg):
    def __init__(self, datas, target, seq_len, stride) -> None:
        super().__init__(datas, target, seq_len, stride, trans_RUL)


class HUSTRULReg_ir(BasicReg_ir):
    def __init__(self, datas, target, time, patch_num, stride) -> None:
        super().__init__(datas, target, time, patch_num, stride, t_span, trans_RUL)


class HUSTDeltaQCQD(BasicDelta):
    def __init__(self, datas, target, seq_len, stride) -> None:
        super().__init__(datas, target, seq_len, stride)


class HUSTDelatQCQD_ir(BasicDelta_ir):
    def __init__(self, datas, target, time, patch_num, stride) -> None:
        super().__init__(datas, target, time, patch_num, stride, t_span)


class HUSTSOHReg(BasicReg):
    def __init__(self, datas, target, seq_len, stride) -> None:
        super().__init__(datas, target, seq_len, stride, trans_SOH)


class HUSTSOHReg_ir(BasicReg_ir):
    def __init__(self, datas, target, time, patch_num, stride) -> None:
        super().__init__(datas, target, time, patch_num, stride, t_span, trans_SOH)


def load_HUST_dataset(data_idx=0, data_config=None):
    datas, labels = load_data('HUST', data_idx)
    return HUSTDataset(datas, labels, data_config).datasets


class HUSTDataset(object):
    def __init__(self, datas, labels, data_config):
        dataset_list = []
        pred_var = None
        if data_config['task'] in ['forecast', 'forecastLong']:
            pred_var = data_config['pred_var']
        for i, data in enumerate(datas):
            # data: (L, 5)
            if data_config['task'] in ['pretrain', 'forecast', 'forecastLong']:
                cur_dataset = BasicHUSTPretrain(data[:, [1, 0, 3, 4]], data_config['seq_len'], data_config['stride'], data_config['pred_len'], pred_var)
            elif data_config['task'] in ['pretrain_ir', 'forecast_ir', 'forecastLong_ir']:
                cur_dataset = BasicHUST_ir(data[:, [1, 0, 3, 4]], data[:, 2], data_config['patch_num'], data_config['stride'], data_config['pred_num'])
            elif data_config['task'] in ['RULRegression']:
                cur_dataset = HUSTRULReg(data[:, [1, 0, 3, 4]], labels[i][4], data_config['seq_len'], data_config['stride'])
            elif data_config['task'] in ['ir_RULRegression']:
                cur_dataset = HUSTRULReg_ir(data[:, [1, 0, 3, 4]], labels[i][4], data[:, 2], data_config['patch_num'], data_config['stride'])
            elif data_config['task'] == 'SOHRegression':
                cur_dataset = HUSTSOHReg(data[:, [1, 0, 3, 4]], labels[i][3], data_config['seq_len'], data_config['stride'])
            elif data_config['task'] in ['ir_SOHRegression']:
                cur_dataset = HUSTSOHReg_ir(data[:, [1, 0, 3, 4]], labels[i][3], data[:, 2], data_config['patch_num'], data_config['stride'])   
            elif data_config['task'] == 'DeltaQc':
                cur_dataset = HUSTDeltaQCQD(data[:, [1, 0, 3, 4]], 3, data_config['seq_len'], data_config['stride'])
            elif data_config['task'] in ['ir_DeltaQc']:
                cur_dataset = HUSTDelatQCQD_ir(data[:, [1, 0, 3, 4]], 3, data[:, 2], data_config['patch_num'], data_config['stride'])
            elif data_config['task'] == 'DeltaQd':
                cur_dataset = HUSTDeltaQCQD(data[:, [1, 0, 3, 4]], 2, data_config['seq_len'], data_config['stride'])
            elif data_config['task'] in ['ir_DeltaQd']:
                cur_dataset = HUSTDelatQCQD_ir(data[:, [1, 0, 3, 4]], 2, data[:, 2], data_config['patch_num'], data_config['stride'])

            # time in input
            elif data_config['task'] == 'pretrain_time':
                cur_dataset = BasicTime(data[:, [1, 0, 3, 4, 2]], data_config['seq_len'], data_config['stride'], data_config['pred_len'], pred_var)
            elif data_config['task'] == 'RULRegression_time':
                cur_dataset = RegTime(data[:, [1, 0, 3, 4, 2]], labels[i][4], data_config['seq_len'], data_config['stride'], trans_RUL)
            elif data_config['task'] == 'SOHRegression_time':
                cur_dataset = RegTime(data[:, [1, 0, 3, 4, 2]], labels[i][3], data_config['seq_len'], data_config['stride'], trans_SOH)
            elif data_config['task'] == 'DeltaQc_time':
                cur_dataset = DeltaTime(data[:, [1, 0, 3, 4, 2]], 3, data_config['seq_len'], data_config['stride'])
            elif data_config['task'] == 'DeltaQd_time':
                cur_dataset = DeltaTime(data[:, [1, 0, 3, 4, 2]], 2, data_config['seq_len'], data_config['stride'])
            else:
                raise Exception(f'{data_config["task"]} is not implemented yet!')
                        
            dataset_list.append(cur_dataset)

        self.datasets = dataset_list
        print('basic dataset num:', len(self.datasets))
    
