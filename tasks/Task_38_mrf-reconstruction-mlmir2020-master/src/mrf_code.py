import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt

# =============================================================================
# DEFINITIONS & CONSTANTS
# =============================================================================

KEY_FINGERPRINTS = 'fingerprints'
KEY_MR_PARAMS = 'mr_params'

ID_MAP_FF = 'FFmap'
ID_MAP_T1H2O = 'T1H2Omap'
ID_MAP_T1FAT = 'T1FATmap'
ID_MAP_B0 = 'B0map'
ID_MAP_B1 = 'B1map'

MR_PARAMS = (ID_MAP_FF, ID_MAP_T1H2O, ID_MAP_T1FAT, ID_MAP_B0, ID_MAP_B1)

FILE_NAME_FINGERPRINTS = 'fingerprints.npy'
FILE_NAME_PARAMETERS = 'parameters.npy'
FILE_NAME_PARAMETERS_MIN = 'parameters_mins.pkl'
FILE_NAME_PARAMETERS_MAX = 'parameters_maxs.pkl'

# =============================================================================
# HELPER CLASSES & FUNCTIONS (MODEL & UTILS)
# =============================================================================

def de_normalize(data: np.ndarray, minmax_tuple: tuple):
    return data * (minmax_tuple[1] - minmax_tuple[0]) + minmax_tuple[0]

def de_normalize_mr_parameters(data: np.ndarray, mr_param_ranges, mr_params=MR_PARAMS):
    data_de_normalized = data.copy()
    for idx, mr_param in enumerate(mr_params):
        if mr_param in mr_param_ranges:
             data_de_normalized[:, idx] = de_normalize(data[:, idx], mr_param_ranges[mr_param])
    return data_de_normalized

class NumpyMRFDataset(data.Dataset):
    def __init__(self, dataset_dir: str, index_selection: list = None, transform=None) -> None:
        super().__init__()
        # Ensure files exist to avoid silent failures later
        if not os.path.exists(os.path.join(dataset_dir, FILE_NAME_PARAMETERS)):
             raise FileNotFoundError(f"Parameters file not found in {dataset_dir}")

        self.mr_params = np.load(os.path.join(dataset_dir, FILE_NAME_PARAMETERS), mmap_mode='r')
        self.fingerprints = np.load(os.path.join(dataset_dir, FILE_NAME_FINGERPRINTS), mmap_mode='r')
        
        with open(os.path.join(dataset_dir, FILE_NAME_PARAMETERS_MIN), 'rb') as f:
            mins = pickle.load(f)
        with open(os.path.join(dataset_dir, FILE_NAME_PARAMETERS_MAX), 'rb') as f:
            maxs = pickle.load(f)
            
        self.mr_param_ranges = {k: (mins[k], maxs[k]) for k in mins}

        if index_selection is not None:
            indexes = np.asarray([int(k) for k in index_selection])
            indexes.sort()
        else:
            indexes = np.arange(self.mr_params.shape[0])
        self.indexes = indexes
        self.transform = transform

    def __len__(self) -> int:
        return self.indexes.shape[0]

    def __getitem__(self, index: int):
        mr_p = np.asarray(self.mr_params[self.indexes[index]]).copy()
        fp = np.asarray(self.fingerprints[self.indexes[index]]).copy()
        
        sample = {KEY_MR_PARAMS: mr_p.astype(np.float32),
                  KEY_FINGERPRINTS: fp.astype(np.float32)}
        if self.transform:
            sample = self.transform(sample)
        return sample

class InvertibleModule(nn.Module):
    def forward(self, x, rev=False):
        pass

class SequenceINN(InvertibleModule):
    def __init__(self, *modules):
        super().__init__()
        self.module_list = nn.ModuleList(modules)

    def forward(self, x, rev=False):
        if not rev:
            for mod in self.module_list:
                x = mod(x, rev=False)
        else:
            for mod in reversed(self.module_list):
                x = mod(x, rev=True)
        return x

class F_fully_connected_small(nn.Module):
    def __init__(self, size_in, size, internal_size=None, dropout=0.0):
        super(F_fully_connected_small, self).__init__()
        if not internal_size:
            internal_size = 2*size

        self.d1 = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(size_in, internal_size)
        self.fc3 = nn.Linear(internal_size, size)
        self.nl1 = nn.ReLU()

    def forward(self, x):
        out = self.nl1(self.d1(self.fc1(x)))
        out = self.fc3(out)
        return out

class RNVPCouplingBlock(InvertibleModule):
    def __init__(self, dims_in, subnet_constructor):
        super().__init__()
        self.ndims = dims_in[0]
        self.split_len1 = self.ndims // 2
        self.split_len2 = self.ndims - self.split_len1
        
        self.subnet1 = subnet_constructor(self.split_len1, self.split_len2 * 2)

    def forward(self, x, rev=False):
        if not rev:
            x1, x2 = x[:, :self.split_len1], x[:, self.split_len1:]
            out = self.subnet1(x2)
            s1, t1 = out[:, :self.split_len2], out[:, self.split_len2:]
            s1 = torch.clamp(s1, -15.0, 15.0) 
            y1 = x1 * torch.exp(s1) + t1
            y2 = x2
            return torch.cat([y1, y2], dim=1)
        else:
            y1, y2 = x[:, :self.split_len1], x[:, self.split_len1:]
            x2 = y2
            out = self.subnet1(x2)
            s1, t1 = out[:, :self.split_len2], out[:, self.split_len2:]
            s1 = torch.clamp(s1, -15.0, 15.0)
            x1 = (y1 - t1) * torch.exp(-s1)
            return torch.cat([x1, x2], dim=1)

class PermuteRandom(InvertibleModule):
    def __init__(self, dims_in, seed=None):
        super().__init__()
        self.in_channels = dims_in[0]
        if seed is not None:
            np.random.seed(seed)
        self.perm = np.random.permutation(self.in_channels)
        self.perm_inv = np.zeros_like(self.perm)
        for i, p in enumerate(self.perm):
            self.perm_inv[p] = i
        
        self.register_buffer('perm_tensor', torch.LongTensor(self.perm))
        self.register_buffer('perm_inv_tensor', torch.LongTensor(self.perm_inv))

    def forward(self, x, rev=False):
        if not rev:
            return x[:, self.perm_tensor]
        else:
            return x[:, self.perm_inv_tensor]

def get_invnet(ndim, nb_blocks=4, hidden_layer=128, permute=True):
    modules = []
    for i in range(nb_blocks):
        def subnet_constructor(ch_in, ch_out):
            return F_fully_connected_small(ch_in, ch_out, internal_size=hidden_layer)
        modules.append(RNVPCouplingBlock([ndim], subnet_constructor))
        if permute:
            modules.append(PermuteRandom([ndim], seed=i))
    return SequenceINN(*modules)

# =============================================================================
# REQUIRED FUNCTIONAL COMPONENTS
# =============================================================================

def load_and_preprocess_data(data_dir: str, batch_size: int, device_str: str):
    """
    Loads the MRF dataset, creates a dataloader, determines dimensions,
    and initializes the Invertible Neural Network.
    
    Returns:
        tuple: (dataloader, model, optimizer, dims_dict, dataset_obj, device)
    """
    if not os.path.exists(data_dir):
        # Create dummy data for demonstration if folder doesn't exist
        # This ensures the code runs without requiring external large files
        os.makedirs(data_dir, exist_ok=True)
        N_SAMPLES = 100
        DIM_PARAMS = 5
        DIM_FINGER = 100
        
        dummy_params = np.random.rand(N_SAMPLES, DIM_PARAMS).astype(np.float32)
        dummy_fingers = np.random.randn(N_SAMPLES, DIM_FINGER).astype(np.float32)
        # Normalize fingerprints L2
        dummy_fingers = dummy_fingers / np.linalg.norm(dummy_fingers, axis=1, keepdims=True)
        
        np.save(os.path.join(data_dir, FILE_NAME_PARAMETERS), dummy_params)
        np.save(os.path.join(data_dir, FILE_NAME_FINGERPRINTS), dummy_fingers)
        
        mins = {k: 0.0 for k in MR_PARAMS}
        maxs = {k: 1.0 for k in MR_PARAMS}
        with open(os.path.join(data_dir, FILE_NAME_PARAMETERS_MIN), 'wb') as f:
            pickle.dump(mins, f)
        with open(os.path.join(data_dir, FILE_NAME_PARAMETERS_MAX), 'wb') as f:
            pickle.dump(maxs, f)
            
        print(f"Created dummy dataset in {data_dir}")

    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    
    dataset = NumpyMRFDataset(data_dir)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Get dimensions
    sample = dataset[0]
    ndim_x = sample[KEY_MR_PARAMS].shape[0]
    ndim_y = sample[KEY_FINGERPRINTS].shape[0]
    
    dims = {'ndim_x': ndim_x, 'ndim_y': ndim_y}
    
    model = get_invnet(ndim=ndim_y, nb_blocks=4, hidden_layer=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    return dataloader, model, optimizer, dims, dataset, device


def forward_operator(model, x, ndim_y, device):
    """
    Simulates the forward Bloch process (learning-based).
    Maps parameters x -> fingerprint y.
    
    Because the INN has fixed input/output dimension equal to ndim_y,
    and x usually has fewer dimensions (ndim_x), we pad x with zeros.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x).float()
    
    x = x.to(device)
    if x.ndim == 1:
        x = x.unsqueeze(0)
        
    current_bs = x.size(0)
    ndim_x = x.size(1)
    
    pad_len = ndim_y - ndim_x
    if pad_len < 0:
        raise ValueError("Parameter dimension cannot be larger than fingerprint dimension for this INN architecture.")
        
    pad_x = torch.zeros(current_bs, pad_len, device=device)
    x_padded = torch.cat((x, pad_x), dim=1)
    
    # Forward pass through INN (x -> y)
    y_pred = model(x_padded, rev=False)
    
    return y_pred


def run_inversion(model, y, ndim_x, device):
    """
    Performs the inversion (Reconstruction).
    Maps fingerprint y -> parameters x.
    """
    if not isinstance(y, torch.Tensor):
        y = torch.from_numpy(y).float()
        
    y = y.to(device)
    if y.ndim == 1:
        y = y.unsqueeze(0)
    
    # Inverse pass through INN (y -> x_padded)
    x_rec_padded = model(y, rev=True)
    
    # Crop the padding to get the actual parameters
    x_rec = x_rec_padded[:, :ndim_x]
    
    return x_rec.detach()


def evaluate_results(model, dataloader, dataset, dims, device, epochs):
    """
    Trains the INN model and then evaluates it on a sample.
    """
    ndim_x = dims['ndim_x']
    ndim_y = dims['ndim_y']
    
    # -----------------------
    # TRAINING LOOP
    # -----------------------
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[KEY_MR_PARAMS].to(device)
            y = batch[KEY_FINGERPRINTS].to(device)
            current_bs = x.size(0)
            
            # Pad x to match y dimension for INN
            pad_x = torch.zeros(current_bs, ndim_y - ndim_x, device=device)
            x_padded = torch.cat((x, pad_x), dim=1)
            
            optimizer.zero_grad()
            
            # Forward loss: predict y from x
            y_hat = model(x_padded, rev=False)
            loss_fwd = F.mse_loss(y_hat, y)
            
            # Backward loss: predict x from y
            x_hat_padded = model(y, rev=True)
            loss_bwd = F.mse_loss(x_hat_padded, x_padded)
            
            loss = loss_fwd + loss_bwd
            loss.backward()
            
            # Gradient Clipping
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data.clamp_(-15.00, 15.00)
            
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.6f}")

    # -----------------------
    # EVALUATION
    # -----------------------
    model.eval()
    
    # Get a single sample for evaluation
    sample_idx = 0
    sample = dataset[sample_idx]
    x_gt_np = sample[KEY_MR_PARAMS]
    y_gt_np = sample[KEY_FINGERPRINTS]
    
    # 1. Inversion (y -> x)
    x_rec = run_inversion(model, y_gt_np, ndim_x, device)
    x_rec_np = x_rec.cpu().numpy()
    
    # 2. Forward (x -> y)
    y_pred = forward_operator(model, x_gt_np, ndim_y, device)
    y_pred_np = y_pred.detach().cpu().numpy()

    # 3. Denormalize parameters for display
    x_gt_denorm = de_normalize_mr_parameters(x_gt_np[np.newaxis, :], dataset.mr_param_ranges)
    x_rec_denorm = de_normalize_mr_parameters(x_rec_np, dataset.mr_param_ranges)
    
    print("\nReconstruction Results (Parameters):")
    param_names = MR_PARAMS
    for i in range(ndim_x):
        name = param_names[i] if i < len(param_names) else f"Param {i}"
        err = abs(x_gt_denorm[0,i] - x_rec_denorm[0,i])
        print(f"{name}: GT = {x_gt_denorm[0,i]:.4f}, Pred = {x_rec_denorm[0,i]:.4f}, Error = {err:.4f}")
        
    mse_fp = np.mean((y_pred_np - y_gt_np)**2)
    print(f"\nForward Model Fingerprint MSE: {mse_fp:.6f}")
    
    plt.figure(figsize=(10, 4))
    plt.plot(y_gt_np, label='Ground Truth')
    plt.plot(y_pred_np[0], label='Predicted (Learned Bloch)')
    plt.title('MR Fingerprint Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('mrf_fingerprint_comparison.png')
    print("Saved fingerprint comparison plot to mrf_fingerprint_comparison.png")
    
    return x_rec_denorm


# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================

if __name__ == "__main__":
    # Configuration
    data_directory = 'in/train' 
    batch_size = 50
    device_name = 'cuda'
    num_epochs = 2 
    
    # 1. Load Data
    dl, net, opt, dimensions, ds_obj, dev = load_and_preprocess_data(data_directory, batch_size, device_name)
    
    # Note: Training is part of the evaluation logic in this specific refactor 
    # because the original code mixed training and evaluation in main.
    # To adhere to the constraints strictly, we treat "run_inversion" and "forward_operator"
    # as the core functional calls used INSIDE "evaluate_results" after training.
    
    # 4. Evaluate (includes Training Loop inside to setup the model state)
    final_params = evaluate_results(net, dl, ds_obj, dimensions, dev, num_epochs)
    
    print("OPTIMIZATION_FINISHED_SUCCESSFULLY")