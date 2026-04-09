import numpy as np

_MOL_CONST = {
    "N2": {
        "we": 2358.518, "wx": 14.2935, "wy": -0.00592949, "wz": -0.00024,
        "Be": 1.99826, "De": 5.774e-06, "alpha_e": 0.0173035, "beta_e": 1.55e-08,
        "gamma_e": -3.1536099e-05, "H0": 3e-12, "He": 1.8e-12,
        "v0": 2330.7, "mu": 0.31, "MW": 28.013, "m_reduced": 7.0015,
        "AG": 1.166, "AC1": 2.4158e-12, "GAMMA": 4.764, "Re": 2.074,
        "GAMMA_p": 2.066e-16, "ALPHA_p": 1.772e-16,
        "Const_Osc": 195853800.0, "Const_Raman": 0.24158, "G/A": 1.166
    }
}

_UNIV_CONST = {
    "h": 6.62607e-34, "h_bar": 1.05457e-34, "c": 29979000000.0,
    "k": 1.380649e-16, "k_": 0.695035, "hc/k": 1.44, "R": 8.314,
    "Const_D": 7.162e-7, "Const_N": 0.724
}

_CHI_NRS = {
    "SET 1": {"T0": 273.15, "P0": 1.01325, "SPECIES": {"N2": 7.7877, "CO2": 11.8, "CO": 12.3, "H2": 10.0206, "O2": 7.8538}}
}

def gaussian_line(w, w0, sigma):
    if sigma == 0: return np.zeros_like(w)
    return 2/sigma*(np.log(2)/np.pi)**0.5*np.exp(-4*np.log(2)*((w-w0)/sigma)**2)

def asym_Gaussian(w, w0, sigma, k, a_sigma, a_k, offset, power_factor=1.):
    response_low = np.exp(-abs((w[w <= w0]-w0)/(sigma-a_sigma))**(k-a_k))
    response_high = np.exp(-abs((w[w > w0]-w0)/(sigma+a_sigma))**(k+a_k))
    response = (np.append(response_low, response_high) + offset)**power_factor
    max_val = response.max()
    if max_val == 0: return response
    return np.nan_to_num(response/max_val)

def downsample(w, w_fine, spec_fine, mode='local_mean'):
    downsampled = []
    if mode == 'interp':
        downsampled = np.interp(w, w_fine, spec_fine)
    elif mode == 'local_mean':
        hw = int((w[1] - w[0])/(w_fine[1] - w_fine[0])/2)
        if hw < 1: hw = 1
        w_fine = np.array(w_fine)
        idx = np.searchsorted(w_fine, w)
        idx[idx >= len(w_fine)] = len(w_fine) - 1
        idx[idx < 0] = 0
        
        downsampled = []
        for i in idx:
            start = max(0, i - hw)
            end = min(len(spec_fine), i + hw + 1)
            if start >= end:
                downsampled.append(spec_fine[i])
            else:
                downsampled.append(np.mean(spec_fine[start:end]))
        downsampled = np.array(downsampled)
    return downsampled

class LineStrength:
    def __init__(self, species='N2'):
        self.mc_dict = _MOL_CONST[species]
        self.Const_D = _UNIV_CONST['Const_D']

    def int_corr(self, j, branch=0):
        mc = self.mc_dict
        if branch == 0:  # Q branch
            pt = j*(j+1)/(2*j-1)/(2*j+3)
            cd = 1-6*mc['Be']**2/mc['we']**2*j*(j+1)
        elif branch == -2:  # O branch
            pt = 3*j*(j-1)/2/(2*j+1)/(2*j-1)
            cd = (1 + 4*mc['Be']/mc['we']*mc['mu']*(2*j-1))**2
        elif branch == 2:  # S branch
            pt = 3*(j+1)*(j+2)/2/(2*j+1)/(2*j+3)
            cd = (1 - 4*mc['Be']/mc['we']*mc['mu']*(2*j+3))**2
        else: return 0, 1
        return pt, cd

    def term_values(self, v, j, mode='sum'):
        mc = self.mc_dict
        Bv = (mc['Be'] - mc['alpha_e']*(v+0.5) + mc['gamma_e']*(v+0.5)**2)
        Dv = mc['De'] + mc['beta_e']*(v+0.5)
        Hv = mc['H0'] + mc['He']*(v+0.5)
        Fv = Bv*j*(j+1) - Dv*j**2*(j+1)**2 + Hv*j**3*(j+1)**3
        
        Gv = (mc['we']*(v+0.5) - mc['wx']*(v+0.5)**2 + mc['wy']*(v+0.5)**3 + mc['wz']*(v+0.5)**4)
        
        if mode == 'sum': return Gv + Fv
        if mode == 'Gv': return Gv
        if mode == 'Fv': return Fv
        return 0

    def line_pos(self, v, j, branch=0):
        return self.term_values(v+1, j+branch) - self.term_values(v, j)

    def pop_factor(self, T, v, j, branch=0, del_Tv=0.0):
        def rho_v(v_):
            return np.exp(-1.44/(T + del_Tv)*self.term_values(v_, 0, mode='Gv'))
        def rho_r(v_, j_):
            gj = 3*(2+(-1)*(j_ % 2))
            return (2*j_ + 1)*gj*np.exp(-1.44/T*self.term_values(v_, j_, mode='Fv'))

        Qv = rho_v(np.arange(20)).sum()
        Qr = rho_r(v, np.arange(100)).sum()
        
        f_low = 1/Qv/Qr*rho_v(v)*rho_r(v, j)
        f_up = 1/Qv/Qr*rho_v(v+1)*rho_r(v+1, j+branch)
        return f_low - f_up

    def doppler_lw(self, T, nu_0=2300.):
        return nu_0*(T/self.mc_dict['MW'])**0.5*self.Const_D

def forward_operator(x_params):
    """
    Generates a synthetic CARS spectrum based on physical parameters.
    
    Args:
        x_params (dict): Dictionary containing:
            - 'nu': Wavenumber axis (array)
            - 'temperature': Gas temperature (K)
            - 'pressure': Pressure (bar)
            - 'x_mol': Mole fraction of resonant species
            - 'species': 'N2'
            - 'pump_lw': Pump laser linewidth
            - 'nu_shift': Spectral shift
            - 'nu_stretch': Spectral stretch
            - 'slit_params': list/tuple for slit function
            
    Returns:
        np.ndarray: Synthetic intensity spectrum I_as
    """
    # Unpack parameters
    nu_expt = x_params['nu']
    T = x_params['temperature']
    P = x_params['pressure']
    x_mol = x_params['x_mol']
    pump_lw = x_params['pump_lw']
    nu_shift = x_params.get('nu_shift', 0)
    nu_stretch = x_params.get('nu_stretch', 1.0)
    
    # Constants
    ls = LineStrength(x_params.get('species', 'N2'))
    Const_N = _UNIV_CONST["Const_N"]
    C_light = _UNIV_CONST["c"]
    
    # Grid for calculation (finer than experiment)
    nu_fine_grid_step = 0.05
    nu_expt_mod = nu_expt * nu_stretch + nu_shift
    start_nu = nu_expt_mod[0] - 10
    end_nu = nu_expt_mod[-1] + 10
    nu_s = np.arange(start_nu, end_nu, nu_fine_grid_step)
    
    # 1. Calculate Chi_RS (Resonant Susceptibility) - G-Matrix approximation logic
    # Simplified relaxation matrix parameters for N2
    fit_param_N2 = [0.0231, 1.67, 1.21, 0.1487] 
    
    js = 30 # rotational levels
    gamma_mat = np.zeros([js, js])
    
    # Relaxation Matrix Construction
    Ej = ls.term_values(0, np.arange(js), 'Fv')
    for _i in range(js):
        for _j in range(_i+1, js):
            del_E = Ej[_j] - Ej[_i]
            if abs(_i-_j) % 2 == 0:
                alpha, beta, sigma, m = fit_param_N2
                _term_1 = (1-np.exp(-m))/(1-np.exp(-m*T/295))*(295/T)**0.5
                _term_2 = ((1+1.5*1.44*Ej[_i]/T/sigma)/(1+1.5*1.44*Ej[_i]/T))**2
                gamma_ji = alpha*P/1.01325*_term_1*_term_2*np.exp(-beta*del_E*1.44/T)
                gamma_ij = gamma_ji*(2*_i+1)/(2*_j+1)*np.exp(del_E*1.44/T)
                gamma_mat[_j, _i], gamma_mat[_i, _j] = gamma_ji, gamma_ij
    for _i in range(js):
        gamma_mat[_i, _i] = -np.sum(gamma_mat[:, _i])

    # Compute Susceptibility
    chi_rs = np.zeros_like(nu_s, dtype='complex128')
    branches = (0, 2, -2) # Q, S, O branches
    vs = 2 # vibrational levels
    _js = np.arange(js)
    
    for _branch in branches:
        for _v in range(vs):
            nu_raman = ls.line_pos(_v, _js, branch=_branch)
            # Eigenvalue problem
            K_mat = np.diag(nu_raman) + gamma_mat*1j
            eigvals, eigvec = np.linalg.eig(K_mat)
            eigvec_inv = np.linalg.inv(eigvec)
            
            del_pop = ls.pop_factor(T, _v, _js, branch=_branch)
            
            # Raman cross section / transition amplitude
            pt_coeff, cd_coeff = np.zeros(js), np.zeros(js)
            for k_idx, k_val in enumerate(_js):
                pt, cd = ls.int_corr(k_val, _branch)
                pt_coeff[k_idx] = pt
                cd_coeff[k_idx] = cd
                
            pol_ratio = ls.mc_dict['G/A']
            Const_Raman = ls.mc_dict['Const_Raman']
            
            if _branch in (2, -2):
                d_sq = (Const_Raman**2*(4/45)*pt_coeff * pol_ratio**2*(_v+1)*cd_coeff)
            else:
                d_sq = Const_Raman**2*(1 + (4/45)*pt_coeff*pol_ratio**2)*(_v+1)*cd_coeff
            
            d = d_sq**0.5
            
            _term_l = d @ eigvec
            _term_r = eigvec_inv @ np.diag(del_pop) @ d
            _term = _term_l*_term_r
            
            for _j in _js:
                _term_b = ((-nu_s + np.real(eigvals[_j]))**2 + np.imag(eigvals[_j])**2)
                chi_rs += 1/2*_term[_j]*np.conj(-nu_s + eigvals[_j])/_term_b

    chi_rs = chi_rs/2/np.pi/C_light

    # 2. Non-resonant Background
    # Estimate Chi_NR
    chi_nrs_dict = _CHI_NRS["SET 1"]
    chi_val = chi_nrs_dict["SPECIES"]["N2"] * x_mol # Simplified: assuming mostly N2 or air-like
    chi_nrs_eff = chi_val * (P/chi_nrs_dict["P0"] * chi_nrs_dict["T0"]/T) * 1e-18
    
    num_density = P/T*Const_N
    chi_total = (x_mol * num_density * chi_rs + chi_nrs_eff) * 1e15 # Scale factor
    
    # 3. Pump Convolutions
    # Doppler (optional in sim, usually small for N2 CARS at high P, skipping for speed/robustness)
    
    # 4. Intensity
    I_as = np.abs(chi_total)**2
    
    # 5. Convolution with Pump Laser Linewidth
    if pump_lw > 0:
        pump_ls = gaussian_line(nu_s, (nu_s[0]+nu_s[-1])/2, pump_lw)
        # Kataoka convolution approximation
        chi_convol = np.convolve(chi_total, pump_ls, 'same')
        d_nu = nu_s[1] - nu_s[0]
        I_as = 0.5 * (I_as + d_nu * np.abs(chi_convol)**2)
        
    # 6. Instrument Function (Slit)
    # Using Gaussian slit for simplicity in forward operator
    slit_params = x_params.get('slit_params', [0.5, 2.0, 0, 0])
    # slit_params: sigma, k, a_sigma, a_k
    
    slit_func = asym_Gaussian(nu_s, (nu_s[0]+nu_s[-1])/2, 
                              sigma=slit_params[0], k=slit_params[1], 
                              a_sigma=slit_params[2], a_k=slit_params[3], offset=0)
    
    I_as = np.convolve(I_as, slit_func, 'same')
    
    # 7. Downsample to experimental grid
    I_final = downsample(nu_expt_mod, nu_s, I_as, mode='local_mean')
    
    # Normalize
    if I_final.max() > 0:
        I_final /= I_final.max()
        
    return I_final
