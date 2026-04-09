import sys
import os
import dill
import numpy as np
import traceback
from agent_forward_operator import forward_operator
from verification_utils import recursive_check

# --- HELPER INJECTION START ---
# We need to ensure that constants and classes relied upon by the pickled data or execution logic 
# are available in the global namespace if dill tries to deserialize them from a similar context.
# Based on the provided code, these are the constants and classes potentially needed.

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
# --- HELPER INJECTION END ---


def test_forward_operator():
    """
    Robust unit test for forward_operator using provided pickle data.
    """
    data_paths = ['/data/yjh/carspy-main_sandbox/run_code/std_data/standard_data_forward_operator.pkl']
    
    # 1. Strategy Identification
    outer_path = None
    inner_path = None
    
    for path in data_paths:
        if 'standard_data_forward_operator.pkl' in path:
            outer_path = path
        elif 'standard_data_parent_function_forward_operator' in path:
            inner_path = path

    if not outer_path:
        print("Error: standard_data_forward_operator.pkl not found in paths.")
        sys.exit(1)

    # 2. Execution Logic
    try:
        # Load Main Data
        print(f"Loading data from {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
            
        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output', None)

        print("Executing forward_operator...")
        
        # Scenario A: Simple Function Execution (Likely case based on provided code)
        if not inner_path:
            actual_result = forward_operator(*outer_args, **outer_kwargs)
            
            # If the function returns a callable (Factory pattern), we cannot verify it without inner data
            if callable(actual_result):
                print("Warning: forward_operator returned a callable, but no inner execution data provided.")
                print("Checking if expected output matches a callable signature (not possible with pickle usually).")
                # Fallback: Check if expected output is also a callable or object representation? 
                # Usually pickle stores the object itself.
                pass 
            
            check_against = expected_output

        # Scenario B: Factory Pattern (If inner data existed)
        else:
            print(f"Loading inner data from {inner_path}")
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            # Step 1: Create the operator
            operator_instance = forward_operator(*outer_args, **outer_kwargs)
            
            if not callable(operator_instance):
                print(f"Error: Expected callable from factory, got {type(operator_instance)}")
                sys.exit(1)
                
            # Step 2: Execute the operator
            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            check_against = inner_data.get('output', None)
            
            actual_result = operator_instance(*inner_args, **inner_kwargs)

        # 3. Verification
        print("Verifying results...")
        passed, msg = recursive_check(check_against, actual_result)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)

    except Exception as e:
        print(f"An error occurred during test execution: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_forward_operator()