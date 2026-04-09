import numpy as np

def PS4(Inp0, Inp1, Inp2, Inp3):
    '''
    Function to recover the phase information of a sample from four DHM in-axis acquisitions holograms
    '''
    inp0 = np.array(Inp0)
    inp1 = np.array(Inp1)
    inp2 = np.array(Inp2)
    inp3 = np.array(Inp3)

    # compensation process
    # U_obj ~ (I3-I1)j + (I2-I0)
    comp_phase = (inp3 - inp1) * 1j + (inp2 - inp0)

    return comp_phase
