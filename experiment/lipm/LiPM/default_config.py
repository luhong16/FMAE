def load_setting(config, setting_idx):
    setting_list = [setting_0, setting_1, setting_2, 
                    setting_3, setting_4, setting_5]
    setting_list[setting_idx](config)
    

def default_setting(config):
    # train
    config.lr = 1e-4
    config.l2 = 1e-3
    config.max_iter = 50000
    config.max_epoch = 10
    config.T_0 = 10
    config.weight_MAE = 1.0
    config.weight_Q = 1.0

    config.batch_size = 256
    config.channel_ratio = 0.3
    config.patch_ratio = 0.3
    config.patch_len = 64
    config.patch_num = 16
    config.patch_stride = -1

    config.dp = 0.3
    config.n_var = 2
    config.norm = 'rsm'
    config.pre_norm = 1 
    config.task = 'ir_pretrain' 

    config.down_dim = 256


def setting_0(config):
    config.d_model = 64
    config.n_head = 4
    config.n_layer = 2

def setting_1(config):
    config.d_model = 128
    config.n_head = 4
    config.n_layer = 3

def setting_2(config):
    config.d_model = 256
    config.n_head = 8
    config.n_layer = 6

def setting_3(config):
    config.d_model = 512
    config.n_head = 8
    config.n_layer = 12

def setting_4(config):
    config.d_model = 768
    config.n_head = 12
    config.n_layer = 18

def setting_5(config):
    config.d_model = 1024
    config.n_head = 16
    config.n_layer = 24

