import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, List, Optional, Union
from scipy.ndimage import binary_dilation
import scipy.io

def rescale_a(a):
    return a/2.15

def rescale_u(u):
    return u/0.028

def scale_back_a(a):
    return  a*2.15

def scale_back_u(u):
    return u*0.028

def plot_result(samples, output_dir, current_training_step):
    # samples: (n_samples, 2, size, size)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if isinstance(samples, torch.Tensor):
        samples = samples.cpu().numpy()
    elif isinstance(samples, list):
        samples = np.array(samples)
    elif isinstance(samples, np.ndarray):
        pass
    else:
        raise ValueError("Unsupported type for samples.")
    
    a_pred = samples[:, 0, :, :]
    u_pred = samples[:, 1, :, :]
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(a_pred[0, :, :], cmap='viridis')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(r'$a$')
    plt.subplot(1, 2, 2)
    plt.imshow(u_pred[0, :, :], cmap='viridis')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(r'$u$')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{current_training_step}.png'), bbox_inches="tight", pad_inches=0.1)
    plt.close()

def load_data(batch_size, return_dataset=False, rescale=True):
    dataset_path = '../DiffusionPDE_data/training/helmholtz'
    file_list = os.listdir(dataset_path)
    a_list = []
    u_list = []
    for file_name in file_list:
        data = scipy.io.loadmat(os.path.join(dataset_path, file_name))
        a_data = data['f_data']
        u_data = data['psi_data']
        a_list.append(a_data)
        u_list.append(u_data)
    a_GT = np.concatenate(a_list, axis=0)
    u_GT = np.concatenate(u_list, axis=0)


    if rescale: 
        a_GT = rescale_a(a_GT)
        u_GT = rescale_u(u_GT)
    dataset = TensorDataset(
    torch.tensor(a_GT, dtype=torch.get_default_dtype()),
    torch.tensor(u_GT, dtype=torch.get_default_dtype()),
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    if return_dataset:
        return dataset, dataloader
    return dataloader

def load_test_data(batch_size, return_dataset=False, rescale=True):
    dataset_path = '../DiffusionPDE_data/testing'
    file_name = 'helmholtz.mat'
    data = scipy.io.loadmat(os.path.join(dataset_path, file_name))
    a_GT = data['f_data']
    u_GT = data['psi_data']
    if rescale:
        a_GT = rescale_a(a_GT)
        u_GT = rescale_u(u_GT)
    dataset = TensorDataset(
    torch.tensor(a_GT, dtype=torch.get_default_dtype()),
    torch.tensor(u_GT, dtype=torch.get_default_dtype()),
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    if return_dataset:
        return dataset, dataloader
    return dataloader


import torch
from typing import Tuple, Optional, Union, List

@torch.no_grad()
def consistency_sample_cm(
    model,
    *,
    sigma_data: float = 0.5,
    device: str = "cuda",
    shape: Tuple[int,int,int,int] = (1, 2, 128, 128),
    z: Optional[torch.Tensor] = None,
    pred_x0_init: Optional[torch.Tensor] = None,
    return_intermediates: bool = False,
    return_cache: bool = False,
    t_list: Optional[Union[List[float], torch.Tensor]] = None,
    n_steps: int = 0,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    schedule: str = "power",
    use_trigflow_t0: bool = True, 
    use_seeded_z: bool = True,
    x_obs: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
):
    """
    Multi-step Consistency Model sampling with strict float64 precision 
    for all logic except the model forward pass.
    """
    device = torch.device(device)
    B, C, H, W = shape
    
    # Force double precision for logic
    calc_dtype = torch.float64

    # Ensure mask/obs are float64 if present
    if x_obs is not None:
        x_obs = x_obs.to(device=device, dtype=calc_dtype)
    if mask is not None:
        mask = mask.to(device=device, dtype=calc_dtype)

    # Build time schedule (Strict float64)
    if t_list is not None:
        t_all = torch.as_tensor(t_list, dtype=calc_dtype, device=device)
        # Recalculate sigma from t if needed, but usually we just need t_all
        # sigma = sigma_data * torch.tan(t_all) 
    else:
        # Step indices in float64
        i = torch.arange(n_steps + 1, device=device, dtype=calc_dtype)
        
        # Schedule calculation in float64
        if schedule == "karras":
            sigma = sigma_min * (sigma_max / sigma_min) ** (1 - (i / n_steps) ** rho)
        elif schedule == "power":
            lam = i / n_steps
            sigma = ((1 - lam) * sigma_max ** (1 / rho) + lam * sigma_min ** (1 / rho)) ** rho
        else:
            raise ValueError("Unknown schedule")
            
        t_all = torch.atan(sigma / sigma_data)

    # Init z (Strict float64)
    if z is None:
        if use_seeded_z:
            z = torch.randn(shape, generator=torch.Generator(device=device).manual_seed(123), device=device, dtype=calc_dtype)
        else:
            z = torch.randn(shape, device=device, dtype=calc_dtype)
    else:
        z = z.to(device=device, dtype=calc_dtype)

    # First pred at t0
    if pred_x0_init is None:
        t0 = t_all[0].expand(B)
        t0_exp = t0.view(B, 1, 1, 1)
        
        # --- Model Evaluation (Downcast -> Run -> Upcast) ---
        # The model likely expects float32. We cast inputs down, and cast output back up immediately.
        model_z = z.to(torch.float32)
        model_t = t0.to(torch.float32)
        F_pred = model(model_z, model_t, return_logvar=False).to(calc_dtype)
        # ----------------------------------------------------

        if use_trigflow_t0:
            pred_x0 = torch.cos(t0_exp)*z*sigma_data - sigma_data*torch.sin(t0_exp)*F_pred
        else:
            pred_x0 = -sigma_data * F_pred
    else:
        pred_x0 = pred_x0_init.to(device=device, dtype=calc_dtype)

    # 🔥 Project to observed data
    if x_obs is not None and mask is not None:
        pred_x0 = x_obs * mask + pred_x0 * (1 - mask)

    inter = [pred_x0.clone()] if return_intermediates else None

    # Iterate
    for k in range(1, len(t_all)):
        # Generate noise in float64
        if use_seeded_z:
            z = torch.randn(shape, generator=torch.Generator(device=device).manual_seed(123+k), device=device, dtype=calc_dtype)
        else:
            z = torch.randn(shape, device=device, dtype=calc_dtype)
            
        t = t_all[k].expand(B)
        t_exp = t.view(B, 1, 1, 1)

        # 1. Add noise (float64)
        x_t = torch.sin(t_exp) * z * sigma_data + torch.cos(t_exp) * pred_x0
        
        # 2. Model Call (Downcast -> Run -> Upcast)
        model_input = (x_t / sigma_data).to(torch.float32)
        model_t = t.to(torch.float32)
        F_t = model(model_input, model_t, return_logvar=False).to(calc_dtype)

        # 3. Denoise (float64)
        pred_x0 = torch.cos(t_exp) * x_t - torch.sin(t_exp) * sigma_data * F_t

        # 🔥 Project each step
        if x_obs is not None and mask is not None:
            pred_x0 = x_obs * mask + pred_x0 * (1 - mask)

        if return_intermediates:
            inter.append(pred_x0.clone())

    if return_cache:
        cache = {"z": z, "t_all": t_all}
        return (pred_x0, inter, cache) if return_intermediates else (pred_x0, cache)
    
    return (pred_x0, inter) if return_intermediates else pred_x0

@torch.no_grad()
def sample_dpm_solver(
    model,
    device,
    sigma_data=0.5,
    num_steps=35, 
    sigma_min=0.002,
    sigma_max=80.0,
    rho=7.0,
    schedule="karras",
    shape: Tuple[int,int,int,int] = (1, 2, 128, 128),
    use_seeded_z: bool = True,
    # NEW
    x_obs: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
):
    """
    2nd-order DPM-Solver sampler with optional projection onto observed samples.
    """
    # Schedule
    step_indices = torch.arange(num_steps, device=device)
    if schedule == "karras":
        sigma = sigma_min * (sigma_max / sigma_min) ** (1 - (step_indices /(num_steps -1)) ** rho)
    elif schedule == "power":
        lam = step_indices / (num_steps - 1)
        sigma = ((1 - lam) * sigma_max ** (1/rho) + lam * sigma_min ** (1/rho)) ** rho
    else:
        raise ValueError("Unknown schedule")
    timesteps = torch.atan(sigma / sigma_data)

    # Init x_t
    if use_seeded_z:
        x_t = torch.randn(shape, device=device, generator=torch.Generator(device=device).manual_seed(123)) * sigma_data
    else:
        x_t = torch.randn(shape, device=device) * sigma_data

    # 🔥 Project init
    if x_obs is not None and mask is not None:
        x_t = x_obs * mask + x_t * (1 - mask)

    # Loop
    for i in range(len(timesteps) - 1):
        s, t = timesteps[i], timesteps[i+1]

        # First model call
        F_s = model(x_t / sigma_data, s.repeat(1), return_logvar=False)
        eps_s = torch.sin(s) * x_t + torch.cos(s) * sigma_data * F_s

        x_t_euler = torch.cos(s - t) * x_t - torch.sin(s - t) * sigma_data * F_s

        # Second model call
        F_t = model(x_t_euler / sigma_data, t.repeat(1), return_logvar=False)
        eps_t = torch.sin(t) * x_t_euler + torch.cos(t) * sigma_data * F_t

        r_s = (torch.log(torch.tan(s)) - torch.log(torch.tan(timesteps[i+1]))) / \
              (torch.log(torch.tan(s)) - torch.log(torch.tan(t)))

        correction = (torch.sin(s - t) / (2 * r_s * torch.cos(s))) * (eps_t - eps_s)

        # Final update
        x_t = torch.cos(s - t) * x_t - torch.sin(s - t) * sigma_data * F_s - correction

        # 🔥 Project each step
        if x_obs is not None and mask is not None:
            x_t = x_obs * mask + x_t * (1 - mask)

    return x_t


@torch.no_grad()
def sample_dpm_solver_v2(
    model,
    device,
    sigma_data=0.5,
    num_steps=35,
    sigma_min=0.002,
    sigma_max=80.0,
    rho=7.0,
    schedule="karras",
    shape: Tuple[int, int, int, int] = (1, 2, 128, 128),
    use_seeded_z: bool = True,
    # conditioning
    x_obs: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
):
    """
    2nd-order DPM-Solver sampler with RePaint-style inpainting conditioning.
    Ensures the known region (mask==1) is always noised to the current noise level t.
    """
    B = shape[0]

    # ----------------------------
    # 1. Build sigma / time schedule
    # ----------------------------
    step_indices = torch.arange(num_steps, device=device)
    if schedule == "karras":
        sigma = sigma_min * (sigma_max / sigma_min) ** (1 - (step_indices / (num_steps - 1)) ** rho)
    elif schedule == "power":
        lam = step_indices / (num_steps - 1)
        sigma = ((1 - lam) * sigma_max ** (1 / rho) + lam * sigma_min ** (1 / rho)) ** rho
    else:
        raise ValueError(f"Unknown schedule {schedule}")
    timesteps = torch.atan(sigma / sigma_data)  # map σ→t

    # ----------------------------
    # 2. Helper: noisify known region to level t
    # ----------------------------
    def noisify_known_region(x_t_like, x_known, m, t_scalar, gen=None):
        """Replace known region with x_known noised to the same level t."""
        t = t_scalar.view(-1, 1, 1, 1)  # [B,1,1,1]
        if gen is None:
            z = torch.randn_like(x_t_like)
        else:
            z = torch.randn(x_t_like.shape, device=device, generator=gen)
        x_known_t = torch.cos(t) * x_known + torch.sin(t) * sigma_data * z
        return m * x_known_t + (1 - m) * x_t_like

    # ----------------------------
    # 3. Initialize x_t at the highest noise level
    # ----------------------------
    t0 = timesteps[0].expand(B)
    gen0 = torch.Generator(device=device).manual_seed(123) if use_seeded_z else None
    z0 = torch.randn(shape, device=device, generator=gen0) if use_seeded_z else torch.randn(shape, device=device)
    x_t = torch.sin(t0.view(B, 1, 1, 1)) * z0 * sigma_data

    # If we have observations, add them with matched noise
    if x_obs is not None and mask is not None:
        x_t = noisify_known_region(x_t, x_obs, mask, t0, gen0)

    # ----------------------------
    # 4. DPM-Solver loop
    # ----------------------------
    for i in range(len(timesteps) - 1):
        s, t = timesteps[i].expand(B), timesteps[i + 1].expand(B)

        # Optional: impose known region at level s before first call
        if x_obs is not None and mask is not None:
            gen_s = torch.Generator(device=device).manual_seed(1000 + i) if use_seeded_z else None
            x_t = noisify_known_region(x_t, x_obs, mask, s, gen_s)

        # --- First model call
        F_s = model(x_t / sigma_data, s, return_logvar=False)
        eps_s = torch.sin(s) * x_t + torch.cos(s) * sigma_data * F_s
        x_t_euler = torch.cos(s - t) * x_t - torch.sin(s - t) * sigma_data * F_s

        # --- Impose known region at level t for Euler proposal
        if x_obs is not None and mask is not None:
            gen_e = torch.Generator(device=device).manual_seed(2000 + i) if use_seeded_z else None
            x_t_euler = noisify_known_region(x_t_euler, x_obs, mask, t, gen_e)

        # --- Second model call
        F_t = model(x_t_euler / sigma_data, t, return_logvar=False)
        eps_t = torch.sin(t) * x_t_euler + torch.cos(t) * sigma_data * F_t

        # --- 2nd-order correction
        r_s = (torch.log(torch.tan(s)) - torch.log(torch.tan(timesteps[i + 1]))) / (
            torch.log(torch.tan(s)) - torch.log(torch.tan(t))
        )
        correction = (torch.sin(s - t) / (2 * r_s * torch.cos(s))) * (eps_t - eps_s)

        # --- Final update
        x_t = torch.cos(s - t) * x_t - torch.sin(s - t) * sigma_data * F_s - correction

        # --- Impose known region at level t for final state
        if x_obs is not None and mask is not None:
            gen_f = torch.Generator(device=device).manual_seed(3000 + i) if use_seeded_z else None
            x_t = noisify_known_region(x_t, x_obs, mask, t, gen_f)

    return x_t

def helmholtz_loss(u, a, return_residual=False):
    """
    Compute the Helmholtz loss given u and a.
    u: predicted solution, shape (B, H, W)
    a: predicted coefficient, shape (B, H, W)
    """

    h = 1 / (u.shape[-1]-1)
    u_padded = torch.nn.functional.pad(u, (1, 1, 1, 1), 'constant', 0)
    d2u = (u_padded[:, :-2, 1:-1] + u_padded[:, 2:, 1:-1] +
            u_padded[:, 1:-1, :-2] + u_padded[:, 1:-1, 2:] - 4 * u) / h**2
        

    helmholtz_residual = d2u + u - a
    # boundary residuals should be zero
    helmholtz_residual[:, 0, :] = 0
    helmholtz_residual[:, -1, :] = 0
    helmholtz_residual[:, :, 0] = 0
    helmholtz_residual[:, :, -1] = 0
    if return_residual:
        return helmholtz_residual

    return torch.mean(helmholtz_residual**2)

def calculate_h1_error(pred, target, h=1.0/127):
    """
    Calculates the H1 norm error between a prediction and a target.
    
    Args:
        pred (np.array): Your method's output (128x128).
        target (np.array): The ground truth solution (128x128).
        h (float): The grid spacing (distance between nodes). 
                   If domain is [0,1], h = 1/127.
    
    Returns:
        h1_error (float): The absolute H1 error.
        h1_rel_error (float): The relative H1 error (percentage).
    """
    # 1. Calculate the error field
    e = pred - target
    
    # 2. Calculate L2 Norm of the error
    # Sum of squares scaled by the area of a cell (h^2 for 2D)
    l2_error_sq = np.sum(e**2) * (h**2)
    
    # 3. Calculate Derivatives of the error
    # np.gradient uses 2nd order central differences for the interior
    # and 1st order differences for boundaries.
    grad_x = np.gradient(e, h, axis=1) # Derivative w.r.t x (columns)
    grad_y = np.gradient(e, h, axis=0) # Derivative w.r.t y (rows)
    
    # 4. Calculate L2 Norm of the gradients (The H1 Semi-Norm)
    # Norm of gradient vector at each point is grad_x^2 + grad_y^2
    h1_semi_norm_sq = np.sum(grad_x**2 + grad_y**2) * (h**2)
    
    # 5. Combine for H1 Norm
    h1_error = np.sqrt(l2_error_sq + h1_semi_norm_sq)
    
    # --- Optional: Relative Error ---
    # Usually more interpretable. We normalize by the H1 norm of the target.
    
    # Target norms
    target_grad_x = np.gradient(target, h, axis=1)
    target_grad_y = np.gradient(target, h, axis=0)
    target_h1_sq = (np.sum(target**2) + \
                    np.sum(target_grad_x**2 + target_grad_y**2)) * (h**2)
                    
    target_h1 = np.sqrt(target_h1_sq)
    
    return h1_error, h1_error / (target_h1 + 1e-12)


def calculate_semi_norm(pred, target, h=1.0/127):
    """
    Calculates the H1 semi-norm error between a prediction and a target.
    
    Args:
        pred (np.array): Your method's output (128x128).
        target (np.array): The ground truth solution (128x128).
        h (float): The grid spacing (distance between nodes). 
                   If domain is [0,1], h = 1/127.
    Returns:
        semi_norm_error (float): The absolute H1 semi-norm error.
        semi_norm_rel_error (float): The relative H1 semi-norm error (percentage).
    """

    # 1. Calculate the error field
    e = pred - target
    
    # 2. Calculate Derivatives of the error
    grad_x = np.gradient(e, h, axis=1) # Derivative w.r.t x (columns)
    grad_y = np.gradient(e, h, axis=0) # Derivative w.r.t y (rows)
    
    # 3. Calculate L2 Norm of the gradients (The H1 Semi-Norm)
    semi_norm_error_sq = np.sum(grad_x**2 + grad_y**2) * (h**2)
    
    semi_norm_error = np.sqrt(semi_norm_error_sq)
    
    # --- Optional: Relative Error ---
    target_grad_x = np.gradient(target, h, axis=1)
    target_grad_y = np.gradient(target, h, axis=0)
    target_semi_norm_sq = np.sum(target_grad_x**2 + target_grad_y**2) * (h**2)
    target_semi_norm = np.sqrt(target_semi_norm_sq)
    
    return semi_norm_error, semi_norm_error / (target_semi_norm + 1e-12)