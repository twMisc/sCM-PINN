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
    return a*0.2 - 1.5

def rescale_u(u):
    return 115*u - 0.9

def scale_back_a(a):
    return (a + 1.5) / 0.2

def scale_back_u(u):
    return (u + 0.9) / 115

def discrete_a(a):
    """Make a discrete a, >7.5 = 12, <7.5 = 3"""
    # a[a > 7.5] = 12
    # a[a <= 7.5] = 3
    # create a new tensor to avoid in-place operation
    if isinstance(a, np.ndarray):
        a = np.where(a > 7.5, 12., 3.)
    elif isinstance(a, torch.Tensor):   
        a = torch.where(a > 7.5, torch.tensor(12., device=a.device), torch.tensor(3., device=a.device))
    return a

def discrete_a_ste(a):
    """
    Forward: hard discretization
    Backward: pretend it's identity (grad flows as if continuous)
    """
    a_discrete = torch.where(a > 7.5, torch.tensor(12., device=a.device), torch.tensor(3., device=a.device))
    # STE: forward is discrete, backward is identity
    return (a_discrete - a).detach() + a


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

# def load_data(batch_size, return_dataset=False, rescale=True):
#     a_GT = np.load('dataset/a_GT.npy')
#     u_GT = np.load('dataset/u_GT.npy')
#     if rescale: 
#         a_GT = rescale_a(a_GT)
#         u_GT = rescale_u(u_GT)
#     dataset = TensorDataset(
#     torch.tensor(a_GT, dtype=torch.get_default_dtype()),
#     torch.tensor(u_GT, dtype=torch.get_default_dtype()),
#     )
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
#     if return_dataset:
#         return dataset, dataloader
#     return dataloader

# def load_test_data(batch_size, return_dataset=False, rescale=True):
#     data = scipy.io.loadmat('darcy.mat')
#     a_data = data['thresh_a_data']
#     u_data = data['thresh_p_data']
#     if rescale:
#         a_data = rescale_a(a_data)
#         u_data = rescale_u(u_data)
#     dataset = TensorDataset(
#     torch.tensor(a_data, dtype=torch.get_default_dtype()),
#     torch.tensor(u_data, dtype=torch.get_default_dtype()),
#     )
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
#     if return_dataset:
#         return dataset, dataloader
#     return dataloader

def load_data(batch_size, return_dataset=False, rescale=True):
    dataset_path = '../DiffusionPDE_data/training/darcy'
    file_list = os.listdir(dataset_path)
    a_list = []
    u_list = []
    for file_name in file_list:
        data = scipy.io.loadmat(os.path.join(dataset_path, file_name))
        a_data = data['thresh_a_data']
        u_data = data['thresh_p_data']
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
    file_name = 'darcy.mat'
    data = scipy.io.loadmat(os.path.join(dataset_path, file_name))
    a_GT = data['thresh_a_data']
    u_GT = data['thresh_p_data']
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
    # NEW
    x_obs: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
):
    """
    Multi-step Consistency Model sampling with optional projection onto observed samples.
    """
    device = torch.device(device)
    B, C, H, W = shape

    # Build time schedule
    if t_list is not None:
        t_all = torch.as_tensor(t_list, dtype=torch.float32, device=device)
    else:
        i = torch.arange(n_steps + 1, device=device, dtype=torch.float64)
        if schedule == "karras":
            sigma = sigma_min * (sigma_max / sigma_min) ** (1 - (i / n_steps) ** rho)
        elif schedule == "power":
            lam = i / n_steps
            sigma = ((1 - lam) * sigma_max ** (1 / rho) + lam * sigma_min ** (1 / rho)) ** rho
        else:
            raise ValueError("Unknown schedule")
        t_all = torch.atan(sigma / sigma_data)
        t_all = t_all.to(torch.float32)

    # Init z
    if z is None:
        if use_seeded_z:
            z = torch.randn(shape, generator=torch.Generator(device=device).manual_seed(123), device=device)
        else:
            z = torch.randn(shape, device=device)

    # First pred at t0
    if pred_x0_init is None:
        t0 = t_all[0].expand(B)
        t0_exp = t0.view(B, 1, 1, 1)
        if use_trigflow_t0:
            pred_x0 = torch.cos(t0_exp)*z*sigma_data - sigma_data*torch.sin(t0_exp)*model(z, t0, return_logvar=False)
        else:
            pred_x0 = -sigma_data * model(z, t0, return_logvar=False)
    else:
        pred_x0 = pred_x0_init

    # 🔥 Project to observed data if mask provided
    if x_obs is not None and mask is not None:
        pred_x0 = x_obs * mask + pred_x0 * (1 - mask)

    inter = [pred_x0.clone()] if return_intermediates else None

    # Iterate
    for k in range(1, len(t_all)):
        if use_seeded_z:
            z = torch.randn(shape, generator=torch.Generator(device=device).manual_seed(123+k), device=device)
        else:
            z = torch.randn(shape, device=device)
        t = t_all[k].expand(B)
        t_exp = t.view(B, 1, 1, 1)

        x_t = torch.sin(t_exp) * z * sigma_data + torch.cos(t_exp) * pred_x0
        F_t = model(x_t / sigma_data, t, return_logvar=False)
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


def expand_mask_numpy(mask: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """
    Expand 1's in a binary mask to include their neighbors, NumPy version.
    
    Parameters
    ----------
    mask : np.ndarray
        Binary mask of shape (B, H, W) for 2D or (B, D, H, W) for 3D.
        Values should be 0 or 1.
    connectivity : int
        For 2D: 4 or 8
        For 3D: 6 or 26

    Returns
    -------
    np.ndarray
        Expanded mask, same dtype and shape as input.
    """
    ndim = mask.ndim - 1  # exclude batch dimension

    if ndim == 2:
        if connectivity == 4:
            struct = np.array([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]], dtype=bool)
        elif connectivity == 8:
            struct = np.ones((3, 3), dtype=bool)
        else:
            raise ValueError("For 2D, connectivity must be 4 or 8")

    elif ndim == 3:
        if connectivity == 6:
            struct = np.zeros((3, 3, 3), dtype=bool)
            struct[1, 1, :] = True
            struct[1, :, 1] = True
            struct[:, 1, 1] = True
        elif connectivity == 26:
            struct = np.ones((3, 3, 3), dtype=bool)
        else:
            raise ValueError("For 3D, connectivity must be 6 or 26")
    else:
        raise ValueError("Only 2D and 3D masks are supported")

    expanded = np.zeros_like(mask, dtype=mask.dtype)
    for b in range(mask.shape[0]):
        expanded[b] = binary_dilation(mask[b].astype(bool), structure=struct).astype(mask.dtype)

    return expanded


def expand_mask_torch(mask: torch.Tensor,
                      connectivity: int = 8) -> torch.Tensor:
    """
    Expand 1's in a binary mask to include their neighbors, GPU-optimized for PyTorch.

    Parameters
    ----------
    mask : torch.Tensor
        Binary mask of shape (B, H, W) for 2D or (B, D, H, W) for 3D.
        Values should be 0 or 1.
    connectivity : int
        For 2D: 4 or 8
        For 3D: 6 or 26

    Returns
    -------
    torch.Tensor
        Expanded mask, same dtype, device, and shape as input.
    """
    ndim = mask.ndim - 1  # exclude batch dimension

    if ndim == 2:
        if connectivity == 4:
            struct = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                                  dtype=mask.dtype,
                                  device=mask.device)
        elif connectivity == 8:
            struct = torch.ones((3, 3), dtype=mask.dtype, device=mask.device)
        else:
            raise ValueError("For 2D, connectivity must be 4 or 8")

        kernel = struct.unsqueeze(0).unsqueeze(0)  # (1,1,3,3)
        mask_t = mask.unsqueeze(1).float()  # (B,1,H,W)
        neighbor_count = F.conv2d(mask_t, kernel, padding=1)

    elif ndim == 3:
        if connectivity == 6:
            struct = torch.zeros((3, 3, 3),
                                 dtype=mask.dtype,
                                 device=mask.device)
            struct[1, 1, :] = 1
            struct[1, :, 1] = 1
            struct[:, 1, 1] = 1
        elif connectivity == 26:
            struct = torch.ones((3, 3, 3),
                                dtype=mask.dtype,
                                device=mask.device)
        else:
            raise ValueError("For 3D, connectivity must be 6 or 26")

        kernel = struct.unsqueeze(0).unsqueeze(0)  # (1,1,3,3,3)
        mask_t = mask.unsqueeze(1).float()  # (B,1,D,H,W)
        neighbor_count = F.conv3d(mask_t, kernel, padding=1)

    else:
        raise ValueError("Only 2D and 3D masks are supported")

    # Any voxel with at least one neighbor (including itself) becomes 1
    expanded = (neighbor_count > 0).float()

    return expanded.squeeze(1).type_as(mask)


def FDM_Darcy_loss(u, a, D=1.0, use_mask=True, output_mask=False, tol=1e-12):
    """
    Finite Difference Method PDE residual loss for Darcy flow.
    Works with PyTorch tensors or NumPy arrays.
    
    Args:
        u: solution tensor/array [B, H, W]
        a: coefficient tensor/array [B, H, W]
        D: domain length (default=1.0)
        use_mask: whether to use regular/irregular masking
        output_mask: return mask along with loss
        tol: tolerance for mask

    Returns:
        loss (and mask if output_mask=True)
    """

    is_torch = isinstance(u, torch.Tensor)
    xp = torch if is_torch else np

    batchsize, H, W = u.shape
    dx = D / (H - 1)

    # Interior points
    u_center = u[:, 1:-1, 1:-1]
    u_ip1 = u[:, 2:, 1:-1]
    u_im1 = u[:, :-2, 1:-1]
    u_jp1 = u[:, 1:-1, 2:]
    u_jm1 = u[:, 1:-1, :-2]

    a_center = a[:, 1:-1, 1:-1]
    a_ip1 = a[:, 2:, 1:-1]
    a_im1 = a[:, :-2, 1:-1]
    a_jp1 = a[:, 1:-1, 2:]
    a_jm1 = a[:, 1:-1, :-2]

    if use_mask:
        mask = ((xp.abs(a_center - a_ip1) < tol) &
                (xp.abs(a_center - a_im1) < tol) &
                (xp.abs(a_center - a_jp1) < tol) &
                (xp.abs(a_center - a_jm1) < tol))
        
        mask = mask.astype(u.dtype) if not is_torch else mask.float()
        irregular_mask = 1 - mask

        if is_torch:
            irregular_mask_expanded = expand_mask_torch(irregular_mask, connectivity=8)
        else:
            irregular_mask_expanded = expand_mask_numpy(irregular_mask, connectivity=8)

        regular_mask_extended = 1 - irregular_mask_expanded
        mask = regular_mask_extended.astype(u.dtype) if not is_torch else regular_mask_extended.float()

    # Compute averaged a at half-points
    a_x_plus = 0.5 * (a_center + a_ip1)
    a_x_minus = 0.5 * (a_center + a_im1)
    a_y_plus = 0.5 * (a_center + a_jp1)
    a_y_minus = 0.5 * (a_center + a_jm1)

    # Flux differences (discrete divergence)
    flux_x = a_x_plus * (u_ip1 - u_center) - a_x_minus * (u_center - u_im1)
    flux_y = a_y_plus * (u_jp1 - u_center) - a_y_minus * (u_center - u_jm1)

    Du = -(flux_x + flux_y) / dx**2

    # Boundary conditions (Dirichlet=0)
    loss = xp.zeros_like(u)
    loss[:, 0, :] = u[:, 0, :]
    loss[:, -1, :] = u[:, -1, :]
    loss[:, :, 0] = u[:, :, 0]
    loss[:, :, -1] = u[:, :, -1]
    if use_mask:
        loss[:, 1:-1, 1:-1] = xp.abs(Du - 1) * mask
    else:
        loss[:, 1:-1, 1:-1] = xp.abs(Du - 1)

    if output_mask:
        return loss, mask
    return loss

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