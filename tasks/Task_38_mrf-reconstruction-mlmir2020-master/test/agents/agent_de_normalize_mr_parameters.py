import numpy as np

import torch.utils.data as data

ID_MAP_FF = 'FFmap'

ID_MAP_T1H2O = 'T1H2Omap'

ID_MAP_T1FAT = 'T1FATmap'

ID_MAP_B0 = 'B0map'

ID_MAP_B1 = 'B1map'

MR_PARAMS = (ID_MAP_FF, ID_MAP_T1H2O, ID_MAP_T1FAT, ID_MAP_B0, ID_MAP_B1)

def de_normalize(data: np.ndarray, minmax_tuple: tuple):
    return data * (minmax_tuple[1] - minmax_tuple[0]) + minmax_tuple[0]

def de_normalize_mr_parameters(data: np.ndarray, mr_param_ranges, mr_params=MR_PARAMS):
    data_de_normalized = data.copy()
    for idx, mr_param in enumerate(mr_params):
        if mr_param in mr_param_ranges:
             data_de_normalized[:, idx] = de_normalize(data[:, idx], mr_param_ranges[mr_param])
    return data_de_normalized
