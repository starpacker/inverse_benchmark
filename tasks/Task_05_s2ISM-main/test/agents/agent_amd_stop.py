import torch

def amd_stop(o_old, o_new, pre_flag: bool, flag: bool, stop, max_iter: int, threshold: float,
             tot: float, nz: int, k: int):
    int_f_old = (o_old[nz // 2]).sum()
    int_f_new = (o_new[nz // 2]).sum()
    d_int_f = (int_f_new - int_f_old) / tot
    int_bkg_old = o_old.sum() - int_f_old
    int_bkg_new = o_new.sum() - int_f_new
    d_int_bkg = (int_bkg_new - int_bkg_old) / tot

    if isinstance(stop, str) and stop == 'auto':
        if torch.abs(d_int_f) < threshold:
            if not pre_flag:
                flag = False
            else:
                pre_flag = False
        elif k == max_iter:
            flag = False
            print('Reached maximum number of iterations.')
    elif isinstance(stop, str) and stop == 'fixed':
        if k == max_iter:
            flag = False

    return pre_flag, flag, torch.Tensor([int_f_new, int_bkg_new]), torch.Tensor([d_int_f, d_int_bkg])
