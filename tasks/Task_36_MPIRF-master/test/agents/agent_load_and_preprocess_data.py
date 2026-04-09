import numpy as np

def load_and_preprocess_data(
    temperature=20.0,
    diameter=30e-9,
    mag_saturation=8e5,
    concentration=5e7,
    select_gradient_x=2.0,
    select_gradient_y=2.0,
    drive_freq_x=2500000.0 / 102.0,
    drive_freq_y=2500000.0 / 96.0,
    drive_amp_x=12e-3,
    drive_amp_y=12e-3,
    repetition_time=6.528e-4,
    sample_freq=2.5e6,
    delta_concentration=50e-3
):
    """
    Initializes the virtual phantom and scanner parameters.
    Returns a dictionary containing all configuration and the ground truth phantom.
    """
    
    # Constants
    PI = 3.1416
    KB = 1.3806488e-23
    TDT = 273.15
    U0 = 4.0 * PI * 1e-7

    # --- Phantom Calculation ---
    Tt = temperature + TDT
    volume = (diameter ** 3) * PI / 6.0
    m_core = mag_saturation
    mm = m_core * volume
    b_coeff = (U0 * mm) / (KB * Tt)

    # --- Scanner Calculation ---
    gx = select_gradient_x / U0
    gy = select_gradient_y / U0
    gg = np.array([[gx], [gy]])

    ay = drive_amp_x / U0
    ax = drive_amp_y / U0

    fn = round(repetition_time * sample_freq)
    
    # Spatial grid setup
    xmax = ax / gx
    ymax = ay / gy
    step = 1e-4

    x_sequence = np.arange(-xmax, xmax + step, step)
    y_sequence = np.arange(-ymax, ymax + step, step)
    xn = len(y_sequence)
    yn = len(x_sequence)

    # Time sequence
    t_sequence = np.arange(0, repetition_time + repetition_time / fn, repetition_time / fn)
    fn_len = len(t_sequence)

    # --- Drive Field Strength ---
    # X direction
    dh_x = ax * np.cos(2.0 * PI * drive_freq_x * t_sequence + PI / 2.0) * (-1.0)
    deri_dh_x = ax * np.sin(2.0 * PI * drive_freq_x * t_sequence + PI / 2.0) * 2.0 * PI * drive_freq_x
    
    # Y direction
    dh_y = ay * np.cos(2.0 * PI * drive_freq_y * t_sequence + PI / 2.0) * (-1.0)
    deri_dh_y = ay * np.sin(2.0 * PI * drive_freq_y * t_sequence + PI / 2.0) * 2.0 * PI * drive_freq_y

    dh = np.array([dh_x, dh_y])
    deri_dh = np.array([deri_dh_x, deri_dh_y])

    # --- Generate Ground Truth Phantom Image (P-Shape) ---
    c_img = np.zeros((xn, yn))
    # Coordinates for P shape based on legacy logic
    c_img[int(xn * (14 / 121)):int(xn * (105 / 121)), int(yn * (29 / 121)):int(yn * (90 / 121))] = 1.0
    c_img[int(xn * (29 / 121)):int(xn * (60 / 121)), int(yn * (44 / 121)):int(yn * (75 / 121))] = 0.0
    c_img[int(xn * (74 / 121)):int(xn * (105 / 121)), int(yn * (44 / 121)):int(yn * (90 / 121))] = 0.0
    phantom_image = c_img * concentration

    # --- Grid Coordinates for Field Calculation ---
    g_sc = np.zeros((xn, yn, 2))
    for i in range(xn):
        # Mapping legacy logic: y = (i) * (1e-4) * (-1) + Ymax
        y_pos = (i) * step * (-1) + ymax
        for j in range(yn):
            # Mapping legacy logic: x = (j) * (1e-4) - Xmax
            x_pos = (j) * step - xmax
            
            temp_field = gg * np.array([[x_pos], [y_pos]])
            g_sc[i, j, 0] = temp_field[0, 0]
            g_sc[i, j, 1] = temp_field[1, 0]

    config_data = {
        'xn': xn,
        'yn': yn,
        'fn': fn_len,
        'coil_sensitivity': 1.0,
        'mm': mm,
        'b_coeff': b_coeff,
        'deri_dh': deri_dh,
        'dh': dh,
        'g_sc': g_sc,
        'phantom_image': phantom_image,
        'delta_concentration': delta_concentration,
        'xmax': xmax,
        'ymax': ymax
    }

    return config_data, phantom_image
