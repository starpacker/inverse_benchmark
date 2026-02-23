```python
import numpy as np
import torch
import matplotlib.pyplot as plt
import diffmetrology as dm
from matplotlib.image import imread
import os

# ==============================================================================
# 1. Load and Preprocess Data
# ==============================================================================

def load_and_preprocess_data(data_path, lens_name):
    print("Initialize a DiffMetrology object.")
    device = dm.init()
    origin_shift = np.array([0.0, 0.0, 0.0])
    DM = dm.DiffMetrology(
        calibration_path = data_path + '/calibration/',
        rotation_path = data_path + '/rotation_calibration/rotation.mat',
        lut_path = data_path + '/gamma_calibration/gammas.mat',
        origin_shift = origin_shift,
        scale=1.0,
        device=device
    )

    print("Crop the region of interst in the original images.")
    filmsize = np.array([768, 768])
    crop_offset = ((2048 - filmsize)/2).astype(int)
    for cam in DM.scene.cameras:
        cam.filmsize = filmsize
        cam.crop_offset = torch.Tensor(crop_offset).to(device)
    
    def crop(x):
        return x[..., crop_offset[0]:crop_offset[0]+filmsize[0], crop_offset[1]:crop_offset[1]+filmsize[1]]

    DM.test_setup()

    # Read measurements
    print(f"Loading lens file: {lens_name}")
    DM.scene.lensgroup.load_file('ThorLabs/' + lens_name + '.txt')

    print("Ground Truth Lens Parameters:")
    for i in range(len(DM.scene.lensgroup.surfaces)):
        print(f"Lens radius of curvature at surface[{i}]: {1.0/DM.scene.lensgroup.surfaces[i].c.item()}")
    print(DM.scene.lensgroup.surfaces[1].d)

    angle = 0.0
    Ts = np.array([70, 100, 110]) # period of the sinusoids
    t = 0

    # load data
    print("Loading measurement data...")
    data = np.load(data_path + '/measurement/' + lens_name + '/data_new.npz')
    imgs = data['imgs']
    refs = data['refs']
    imgs = crop(imgs)
    refs = crop(refs)
    del data

    # solve for ps and valid map
    print("Solving for intersections...")
    ps_cap, valid_cap, C = DM.solve_for_intersections(imgs, refs, Ts[t:])

    # set display pattern
    xs = [0]
    sinusoid_path = './camera_acquisitions/images/sinusoids/T=' + str(Ts[t])
    ims = [ np.mean(imread(sinusoid_path + '/' + str(x) + '.png'), axis=-1) for x in xs ] # use grayscale
    ims = np.array([ im/im.max() for im in ims ])
    ims = np.sum(ims, axis=0)
    DM.set_texture(ims)
    del ims
    
    # Set texture shift (hardcoded from original demo)
    DM.scene.screen.texture_shift = torch.Tensor([0.       , 1.1106231]).to(device) # LE1234-A

    print("Shift `origin` by an estimated value")
    origin = DM._compute_mount_geometry(C, verbose=True)
    DM.scene.lensgroup.origin = torch.Tensor(origin).to(device)
    DM.scene.lensgroup.update()
    print(origin)

    # Load real images for visualization/comparison
    print("Load real images for visualization")
    FR = dm.Fringe()
    a_cap, b_cap, psi_cap = FR.solve(imgs)
    imgs_sub = np.array([imgs[0,x,...] for x in xs])
    imgs_sub = imgs_sub - a_cap[:,0,...]
    imgs_sub = np.sum(imgs_sub, axis=0)
    imgs_sub = valid_cap * torch.Tensor(imgs_sub).to(device)
    I0 = valid_cap * len(xs) * (imgs_sub - imgs_sub.min().item()) / (imgs_sub.max().item() - imgs_sub.min().item())

    preprocessed_data = {
        'DM': DM,
        'ps_cap': ps_cap,
        'valid_cap': valid_cap,
        'angle': angle,
        'I0': I0,
        'device': device
    }
    return preprocessed_data

# ==============================================================================
# 2. Forward Operator
# ==============================================================================

def forward_model(DM, angle, valid_cap):
    """
    Ray tracing forward model.
    """
    ps = torch.stack(DM.trace(with_element=True, mask=valid_cap, angles=angle)[0])[..., 0:2]
    return ps

def render_model(DM, angle, valid_cap):
    """
    Image rendering forward model.
    """
    I = valid_cap*torch.stack(DM.render(with_element=True, angles=angle))
    I[torch.isnan(I)] = 0.0
    return I

# ==============================================================================
# 3. Run Inversion
# ==============================================================================

def run_inversion(preprocessed_data, maxit=5):
    DM = preprocessed_data['DM']
    ps_cap = preprocessed_data['ps_cap']
    valid_cap = preprocessed_data['valid_cap']
    angle = preprocessed_data['angle']
    device = preprocessed_data['device']

    print("Initialize lens parameters (perturbation).")
    DM.scene.lensgroup.surfaces[0].c = torch.Tensor([0.00]).to(device) # 1st surface curvature
    DM.scene.lensgroup.surfaces[1].c = torch.Tensor([0.00]).to(device) # 2nd surface curvature
    DM.scene.lensgroup.surfaces[1].d = torch.Tensor([3.00]).to(device) # lens thickness
    DM.scene.lensgroup.theta_x = torch.Tensor([0.00]).to(device) # lens X-tilt angle
    DM.scene.lensgroup.theta_y = torch.Tensor([0.00]).to(device) # lens Y-tilt angle
    DM.scene.lensgroup.update()

    print("Set optimization parameters.")
    diff_names = [
        'lensgroup.surfaces[0].c',
        'lensgroup.surfaces[1].c',
        'lensgroup.surfaces[1].d',
        'lensgroup.origin',
        'lensgroup.theta_x',
        'lensgroup.theta_y'
    ]

    def forward():
        return forward_model(DM, angle, valid_cap)

    def loss(ps):
        return torch.sum((ps[valid_cap,...] - ps_cap[valid_cap,...])**2, axis=-1).mean()

    def func_yref_y(ps):
        b = valid_cap[...,None] * (ps_cap - ps)
        b[torch.isnan(b)] = 0.0 # handle NaN ... otherwise LM won't work!
        return b

    # Optimize
    print(f"Running optimization with maxit={maxit}...")
    ls = DM.solve(diff_names, forward, loss, func_yref_y, option='Adam', R='I', maxit=maxit)
    
    return ls

# ==============================================================================
# 4. Evaluate Results
# ==============================================================================

def evaluate_results(preprocessed_data, save_string="result"):
    DM = preprocessed_data['DM']
    ps_cap = preprocessed_data['ps_cap']
    valid_cap = preprocessed_data['valid_cap']
    angle = preprocessed_data['angle']
    I0 = preprocessed_data['I0']
    
    print("Showing lens parameters:")
    for i in range(len(DM.scene.lensgroup.surfaces)):
        print(f"Lens radius of curvature at surface[{i}]: {1.0/DM.scene.lensgroup.surfaces[i].c.item()}")
    print(DM.scene.lensgroup.surfaces[1].d)

    print("Visualize status.")
    ps_current = forward_model(DM, angle, valid_cap)
    
    print("Showing spot diagrams at display.")
    DM.spot_diagram(ps_cap, ps_current, valid=valid_cap, angle=angle, with_grid=False)
    # plt.show()
    plt.savefig(f"{save_string}_spot_diagram.png")
    
    print("Showing images (measurement & modeled & |measurement - modeled|).")

    # Render images from parameters
    I = render_model(DM, angle, valid_cap)
    
    fig, axes = plt.subplots(2, 3)
    for i in range(2):
        im = axes[i,0].imshow(I0[i].cpu(), vmin=0, vmax=1, cmap='gray')
        axes[i,0].set_title(f"Camera {i+1}\nMeasurement")
        axes[i,0].set_xlabel('[pixel]')
        axes[i,0].set_ylabel('[pixel]')
        plt.colorbar(im, ax=axes[i,0])

        im = axes[i,1].imshow(I[i].cpu().detach(), vmin=0, vmax=1, cmap='gray')
        plt.colorbar(im, ax=axes[i,1])
        axes[i,1].set_title(f"Camera {i+1}\nModeled")
        axes[i,1].set_xlabel('[pixel]')
        axes[i,1].set_ylabel('[pixel]')
        
        im = axes[i,2].imshow(I0[i].cpu() - I[i].cpu().detach(), vmin=-1, vmax=1, cmap='coolwarm')
        plt.colorbar(im, ax=axes[i,2])
        axes[i,2].set_title(f"Camera {i+1}\nError")
        axes[i,2].set_xlabel('[pixel]')
        axes[i,2].set_ylabel('[pixel]')
        
        fig.suptitle(save_string)
        fig.savefig(f"{save_string}_images_{i}.jpg", bbox_inches='tight')
    # plt.show()
    
    # Print mean displacement error
    T = ps_current - ps_cap
    E = torch.sqrt(torch.sum(T[valid_cap, ...]**2, axis=-1)).mean()
    print("error = {} [um]".format(E*1e3))
    
    return E.item()

# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":
    data_path = './20210403'
    lens_name = 'LE1234-A'
    
    print("Step 1: Loading and preprocessing data...")
    preprocessed_data = load_and_preprocess_data(data_path, lens_name)
    
    print("Step 2: Initial Evaluation...")
    # visualize initial state
    evaluate_results(preprocessed_data, save_string="initial")
    
    print("Step 3: Running Inversion...")
    ls = run_inversion(preprocessed_data, maxit=5) # Reduced iterations as requested
    
    print("Step 4: Final Evaluation...")
    evaluate_results(preprocessed_data, save_string="optimized")
    
    print("OPTIMIZATION_FINISHED_SUCCESSFULLY")
```