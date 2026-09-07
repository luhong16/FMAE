from LiPM.dataset.basic_dataset import BasicPretrain, Basic_ir, BasicDelta, BasicDelta_ir, BasicReg, BasicReg_ir, BasicTime, RegTime, DeltaTime
from LiPM.dataset.data_utils import load_data


t_span = 900

class BasicULPurduePretrain(BasicPretrain):
    def __init__(self, datas, seq_len, stride, pred_len, pred_var) -> None:
        super().__init__(datas, seq_len, stride, pred_len, pred_var)


class BasicULPurdue_ir(Basic_ir):
    # make patch first
    def __init__(self, datas, time, patch_num, stride, pred_num=0) -> None:
        super().__init__(datas, time, patch_num, stride, pred_num, t_span=t_span)  


class ULPurdueDelta(BasicDelta):
    def __init__(self, datas, target, seq_len, stride) -> None:
        super().__init__(datas, target, seq_len, stride)


class ULPurdueDelta_ir(BasicDelta_ir):
    def __init__(self, datas, target, time, patch_num, stride) -> None:
        super().__init__(datas, target, time, patch_num, stride, t_span)


def trans_RUL(rul):
    return rul / 100


def trans_SOH(soh):
    return min(soh, 1.0)


class ULPurdueRULReg(BasicReg):
    def __init__(self, datas, target, seq_len, stride) -> None:
        super().__init__(datas, target, seq_len, stride, trans_RUL)


class ULPurdueRULReg_ir(BasicReg_ir):
    def __init__(self, datas, target, time, patch_num, stride) -> None:
        super().__init__(datas, target, time, patch_num, stride, t_span, trans_RUL)

class ULPurdueSOHReg(BasicReg):
    def __init__(self, datas, target, seq_len, stride) -> None:
        super().__init__(datas, target, seq_len, stride, trans_SOH)


class ULPurdueSOHReg_ir(BasicReg_ir):
    def __init__(self, datas, target, time, patch_num, stride) -> None:
        super().__init__(datas, target, time, patch_num, stride, t_span, trans_SOH)
    

# for downstream
data_names = ['CF10DPA_pouch_NCA_25C_0-100_1-1C_n', 'N10-EX9_18650_NCA_23C_0-100_0.5-0.5C_i', 'N10-NA7_18650_NCA_23C_0-100_0.5-0.5C_g', 
            'N10-OV8_18650_NCA_23C_0-100_0.5-0.5C_h', 'N15-EX4_18650_NCA_23C_0-100_0.5-0.5C_d', 'N15-NA10_18650_NCA_23C_0-100_0.5-0.5C_j', 
            'N15-OV3_18650_NCA_23C_0-100_0.5-0.5C_c', 'N20-EX2_18650_NCA_23C_0-100_0.5-0.5C_b', 'N20-NA5_18650_NCA_23C_0-100_0.5-0.5C_e', 
            'N20-NA6_18650_NCA_23C_0-100_0.5-0.5C_f', 'N20-OV1_18650_NCA_23C_0-100_0.5-0.5C_a', 'R10-EX6_18650_NCA_23C_2.5-96.5_0.5-0.5C_f', 
            'R10-NA11_18650_NCA_23C_2.5-96.5_0.5-0.5C_k', 'R10-OV5_18650_NCA_23C_2.5-96.5_0.5-0.5C_e', 'R15-EX4_18650_NCA_23C_2.5-96.5_0.5-0.5C_d', 
            'R15-NA10_18650_NCA_23C_2.5-96.5_0.5-0.5C_j', 'R15-OV3_18650_NCA_23C_2.5-96.5_0.5-0.5C_c', 'R20-EX2_18650_NCA_23C_2.5-96.5_0.5-0.5C_b', 
            'R20-NA7_18650_NCA_23C_2.5-96.5_0.5-0.5C_g', 'R20-NA8_18650_NCA_23C_2.5-96.5_0.5-0.5C_h', 'R20-NA9_18650_NCA_23C_2.5-96.5_0.5-0.5C_i', 
            'R20-OV1_18650_NCA_23C_2.5-96.5_0.5-0.5C_a']

def load_ULPurdue_dataset(data_idx=0, data_config=None):
    global t_span
    if data_idx == 0:
        t_span = 16
    else:
        t_span = 900
    datas, labels = load_data('ULPurdue', data_idx)
    return ULPurdueDataset(datas, labels, data_config).datasets


class ULPurdueDataset(object):
    def __init__(self, datas, labels, data_config):
        dataset_list = []
        pred_var = None
        if data_config['task'] == 'forecast':
            pred_var = data_config['pred_var']
        if data_config['task'] in ['RULRegression', 'ir_RULRegression'] and (labels.shape[1] < 6 or labels[0][5] < 0):
            pass
        else:
            for i, data in enumerate(datas):
                if data_config['task'] in ['pretrain', 'forecast']:
                    cur_dataset = BasicULPurduePretrain(data[:, [1, 0, 3, 4]], data_config['seq_len'], data_config['stride'], data_config['pred_len'], pred_var)
                elif data_config['task'] in ['pretrain_ir', 'forecast_ir']:
                    cur_dataset = BasicULPurdue_ir(data[:, [1, 0, 3, 4]], data[:, 2], data_config['patch_num'], data_config['stride'], data_config['pred_num'])
                elif data_config['task'] == 'DeltaQc':
                    cur_dataset = ULPurdueDelta(data[:, [1, 0, 3, 4]], 3, data_config['seq_len'], data_config['stride'])
                elif data_config['task'] == 'ir_DeltaQc':
                    cur_dataset = ULPurdueDelta_ir(data[:, [1, 0, 3, 4]], 3, data[:, 2], data_config['patch_num'], data_config['stride'])
                elif data_config['task'] == 'DeltaQd':
                    cur_dataset = ULPurdueDelta(data[:, [1, 0, 3, 4]], 2, data_config['seq_len'], data_config['stride'])
                elif data_config['task'] == 'ir_DeltaQd':
                    cur_dataset = ULPurdueDelta_ir(data[:, [1, 0, 3, 4]], 2, data[:, 2], data_config['patch_num'], data_config['stride'])
                elif data_config['task'] == 'RULRegression':
                    cur_dataset = ULPurdueRULReg(data[:, [1, 0, 3, 4]], labels[i][5], data_config['seq_len'], data_config['stride'])
                elif data_config['task'] == 'ir_RULRegression':
                    cur_dataset = ULPurdueRULReg_ir(data[:, [1, 0, 3, 4]], labels[i][5], data[:, 2], data_config['patch_num'], data_config['stride'])
                elif data_config['task'] == 'SOHRegression':
                    cur_dataset = ULPurdueSOHReg(data[:, [1, 0, 3, 4]], labels[i][4], data_config['seq_len'], data_config['stride'])
                elif data_config['task'] == 'ir_SOHRegression':
                    cur_dataset = ULPurdueSOHReg_ir(data[:, [1, 0, 3, 4]], labels[i][4], data[:, 2], data_config['patch_num'], data_config['stride'])
                # time in the input data
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
                    raise NotImplementedError(f'{data_config["task"]} is not supported!')
                dataset_list.append(cur_dataset)

        self.datasets = dataset_list
        print('basic dataset num:', len(self.datasets))
    
