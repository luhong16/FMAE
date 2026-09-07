from LiPM.dataset.basic_dataset import BasicPretrain, Basic_ir, BasicDelta, BasicDelta_ir, BasicReg, BasicReg_ir, RegTime, DeltaTime, BasicTime
from LiPM.dataset.data_utils import load_data


t_span = 132


def trans_RUL(RUL):
    return RUL / 100


def trans_SOH(SOH):
    return min((SOH, 1.))  # max soh is 100%


class BasicRWTHPretrain(BasicPretrain):
    def __init__(self, datas, seq_len, stride, pred_len, pred_var) -> None:
        super().__init__(datas, seq_len, stride, pred_len, pred_var)


class BasicRWTH_ir(Basic_ir):
    # make patch first
    def __init__(self, datas, time, patch_num, stride, pred_num=0) -> None:
        super().__init__(datas, time, patch_num, stride, pred_num, t_span)  


class RWTHDelta(BasicDelta):
    def __init__(self, datas, target, seq_len, stride) -> None:
        super().__init__(datas, target, seq_len, stride)


class RWTHDelta_ir(BasicDelta_ir):
    def __init__(self, datas, target, time, patch_num, stride) -> None:
        super().__init__(datas, target, time, patch_num, stride, t_span)


class RWTHRULReg(BasicReg):
    def __init__(self, datas, target, seq_len, stride) -> None:
        super().__init__(datas, target, seq_len, stride, trans_RUL)


class RWTHRULReg_ir(BasicReg_ir):
    def __init__(self, datas, target, time, patch_num, stride) -> None:
        super().__init__(datas, target, time, patch_num, stride, t_span, trans_RUL)


class RWTHSOHReg(BasicReg):
    def __init__(self, datas, target, seq_len, stride) -> None:
        super().__init__(datas, target, seq_len, stride, trans_SOH)


class RWTHSOHReg_ir(BasicReg_ir):
    def __init__(self, datas, target, time, patch_num, stride) -> None:
        super().__init__(datas, target, time, patch_num, stride, t_span, trans_SOH)


data_names = ['002', '003', '004', '005', '006', '007', '008', 
              '009', '010', '011', '012', '013', '014', '015', 
              '016', '017', '018', '019', '020', '021', '022', 
              '023', '024', '025', '026', '027', '028', '029', 
              '030', '031', '032', '033', '034', '035', '036', 
              '037', '038', '039', '040', '041', '042', '043', 
              '044', '045', '046', '047', '048', '049']

def load_RWTH_dataset(data_idx=0, data_config=None):
    datas, labels = load_data('RWTH', data_idx)
    return RWTHDataset(datas, labels, data_config).datasets


class RWTHDataset(object):
    def __init__(self, datas, labels, data_config):
        dataset_list = []
        pred_var = None
        if data_config['task'] in ['forecast', 'forecastLong']:
            pred_var = data_config['pred_var']
        for i, data in enumerate(datas):
            if data_config['task'] in ['pretrain', 'forecast', 'forecastLong']:
                cur_dataset = BasicRWTHPretrain(data[:, [1, 0, 4, 3]], data_config['seq_len'], data_config['stride'], data_config['pred_len'], pred_var)
            elif data_config['task'] in ['pretrain_ir', 'forecast_ir', 'forecastLong_ir']:
                cur_dataset = BasicRWTH_ir(data[:, [1, 0, 4, 3]], data[:, 2], data_config['patch_num'], data_config['stride'], data_config['pred_num'])
            elif data_config['task'] == 'DeltaQc':
                cur_dataset = RWTHDelta(data[:, [1, 0, 4, 3]], 3, data_config['seq_len'], data_config['stride'])
            elif data_config['task'] == 'ir_DeltaQc':
                cur_dataset = RWTHDelta_ir(data[:, [1, 0, 4, 3]], 3, data[:, 2], data_config['patch_num'], data_config['stride'])
            elif data_config['task'] == 'DeltaQd':
                cur_dataset = RWTHDelta(data[:, [1, 0, 4, 3]], 2, data_config['seq_len'], data_config['stride'])
            elif data_config['task'] == 'ir_DeltaQd':
                cur_dataset = RWTHDelta_ir(data[:, [1, 0, 4, 3]], 2, data[:, 2], data_config['patch_num'], data_config['stride'])                       
            elif data_config['task'] == 'RULRegression':
                cur_dataset = RWTHRULReg(data[:, [1, 0, 4, 3]], labels[i][2], data_config['seq_len'], data_config['stride'])
            elif data_config['task'] == 'ir_RULRegression':
                cur_dataset = RWTHRULReg_ir(data[:, [1, 0, 4, 3]], labels[i][2], data[:, 2], data_config['patch_num'], data_config['stride'])
            elif data_config['task'] == 'SOHRegression':
                cur_dataset = RWTHSOHReg(data[:, [1, 0, 4, 3]], labels[i][1], data_config['seq_len'], data_config['stride'])
            elif data_config['task'] == 'ir_SOHRegression':
                cur_dataset = RWTHSOHReg_ir(data[:, [1, 0, 4, 3]], labels[i][1], data[:, 2], data_config['patch_num'], data_config['stride'])
            # time in the input data
            elif data_config['task'] == 'pretrain_time':
                cur_dataset = BasicTime(data[:, [1, 0, 4, 3, 2]], data_config['seq_len'], data_config['stride'], data_config['pred_len'], pred_var)
            elif data_config['task'] == 'RULRegression_time':
                cur_dataset = RegTime(data[:, [1, 0, 4, 3, 2]], labels[i][2], data_config['seq_len'], data_config['stride'], trans_RUL)
            elif data_config['task'] == 'SOHRegression_time':
                cur_dataset = RegTime(data[:, [1, 0, 4, 3, 2]], labels[i][1], data_config['seq_len'], data_config['stride'], trans_SOH)
            elif data_config['task'] == 'DeltaQc_time':
                cur_dataset = DeltaTime(data[:, [1, 0, 4, 3, 2]], 3, data_config['seq_len'], data_config['stride'])
            elif data_config['task'] == 'DeltaQd_time':
                cur_dataset = DeltaTime(data[:, [1, 0, 4, 3, 2]], 2, data_config['seq_len'], data_config['stride'])
            else:
                raise NotImplementedError(f'{data_config["task"]} is not implemented!')

            dataset_list.append(cur_dataset)

        self.datasets = dataset_list
        print('basic dataset num:', len(self.datasets))
    
