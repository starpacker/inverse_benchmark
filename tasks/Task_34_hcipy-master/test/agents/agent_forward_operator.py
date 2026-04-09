import numpy as np

import copy

class Wavefront(object):
    def __init__(self, electric_field, wavelength=1):
        self.electric_field = electric_field.astype(complex)
        self.wavelength = wavelength
        
    @property
    def grid(self):
        return self.electric_field.grid
        
    @property
    def intensity(self):
        return np.abs(self.electric_field)**2
        
    @property
    def power(self):
        return self.intensity * self.grid.weights
        
    @property
    def total_power(self):
        return np.sum(self.power)
        
    @total_power.setter
    def total_power(self, p):
        self.electric_field *= np.sqrt(p / self.total_power)
        
    @property
    def wavenumber(self):
        return 2 * np.pi / self.wavelength
        
    def copy(self):
        return copy.deepcopy(self)

def forward_operator(actuator_delta, current_atmosphere_actuators, system_data):
    """
    The closed-loop step. Given a change in DM actuators, updates DM state, 
    simulates WFS sensing, and returns new slopes.
    
    Actually, for a simpler inversion logic (run_inversion), we will define this 
    as the function that computes the *Correction Command* given current state.
    
    But to adhere to the structure requested:
    Ideally 'forward_operator' simulates physics x -> y. 
    Here x could be DM actuators, y could be Science Image (for eval) or WFS Slopes (for loop).
    
    Let's make this function generate the SCIENCE IMAGE given the Atmosphere and DM state.
    This corresponds to the physical forward model of the imaging system.
    """
    dm = system_data['dm']
    atmosphere = system_data['atmosphere']
    prop = system_data['prop']
    aperture = system_data['aperture']
    wavelength_sci = system_data['wavelength_sci']
    telescope_diameter = system_data['telescope_diameter']
    
    # Set states
    # Note: DM and Atmosphere share object reference in load_data if not careful,
    # but here we assume they are distinct instances with passed actuator values.
    # To avoid side effects, we set them, compute, and assume caller manages state persistence or we do copy.
    # In run_inversion, we manage the state. Here we just use the objects as tools.
    
    # We need to temporarily set actuators to compute the image
    # But since we can't pass 'actuators' easily as arguments without modifying the objects inplace,
    # we assume the objects passed in system_data are already configured OR we configure them here.
    # Let's assume the caller configures the DM/Atmosphere objects before calling this, 
    # OR we pass the values. Let's pass values to be purely functional.
    
    if actuator_delta is not None:
        # This implies we are testing a perturbation. 
        # For simplicity, let's assume this function calculates the Science Image
        # based on the CURRENT state of dm and atmosphere stored in system_data.
        pass

    # Define Science Scene: Binary Star
    star_sep_lambda_D = 6
    star_sep_rad = star_sep_lambda_D * (wavelength_sci / telescope_diameter)
    
    # 1. Star 1 (On-axis)
    wf1 = Wavefront(aperture, wavelength_sci)
    wf1.total_power = 1.0
    wf1 = dm(atmosphere(wf1))
    psf1 = prop(wf1).power
    
    # 2. Star 2 (Off-axis)
    wf2 = Wavefront(aperture, wavelength_sci)
    wf2.total_power = 0.1
    k = wf2.wavenumber
    tilt_phase = k * (star_sep_rad * wf2.grid.x)
    wf2.electric_field *= np.exp(1j * tilt_phase)
    
    wf2 = dm(atmosphere(wf2))
    psf2 = prop(wf2).power
    
    return psf1 + psf2
