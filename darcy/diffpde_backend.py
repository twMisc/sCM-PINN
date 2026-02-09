import torch
import pickle
import numpy as np
import torch.nn.functional as F

def get_darcy_loss(a, u, a_GT, u_GT, a_mask, u_mask, device=torch.device('cuda')):
    """
    Computes Darcy Loss. 
    Robust to input shapes (B, H, W) or (B, 1, H, W).
    """
    # 1. Force inputs to (Batch, 1, H, W)
    if u.dim() == 3: u = u.unsqueeze(1)
    if a.dim() == 3: a = a.unsqueeze(1)
    if u_GT.dim() == 3: u_GT = u_GT.unsqueeze(1)
    if a_GT.dim() == 3: a_GT = a_GT.unsqueeze(1)
    
    # Check to ensure we didn't accidentally stack batch into channels
    if u.shape[1] > 1 and u.shape[0] == 1:
        u = u.permute(1, 0, 2, 3)
        a = a.permute(1, 0, 2, 3)
        u_GT = u_GT.permute(1, 0, 2, 3)
        a_GT = a_GT.permute(1, 0, 2, 3)

    # 2. Define Kernels
    deriv_x = torch.tensor([[-1, 0, 1]], dtype=torch.float64, device=device).view(1, 1, 1, 3) / 2
    deriv_y = torch.tensor([[-1], [0], [1]], dtype=torch.float64, device=device).view(1, 1, 3, 1) / 2
    
    # 3. Compute Gradients
    grad_x_next_x = F.conv2d(u, deriv_x, padding=(0, 1))
    grad_x_next_y = F.conv2d(u, deriv_y, padding=(1, 0))
    
    # 4. Multiply by Permeability 'a'
    grad_x_next_x = a * grad_x_next_x
    grad_x_next_y = a * grad_x_next_y
    
    # 5. Compute Divergence
    result = F.conv2d(grad_x_next_x, deriv_x, padding=(0, 1)) + \
             F.conv2d(grad_x_next_y, deriv_y, padding=(1, 0))
    
    # 6. PDE Residual (forcing term +1)
    pde_loss = result + 1.0 
    
    # 7. Observation Loss
    observation_loss_a = (a - a_GT) * a_mask 
    observation_loss_u = (u - u_GT) * u_mask
    
    return pde_loss, observation_loss_a, observation_loss_u

def load_pickle_model(model_path, device):
    with open(model_path, 'rb') as f:
        net = pickle.load(f)['ema'].to(device)
    return net

def sample_guided_diffusion(net, x_obs, mask, num_steps=32, rho=7, 
                            zeta_obs_a=0.8, zeta_obs_u=0.0, zeta_pde=1.0, 
                            sigma_min=0.002, sigma_max=80, device='cuda'):
    """
    Heun sampler for DARCY FLOW with physics-guided gradient updates.
    Returns: (Batch_Size, 2, 128, 128)
    """
    batch_size = x_obs.shape[0]
    resolution = x_obs.shape[2]
    
    # Slice to keep (B, 1, H, W) structure
    a_GT = x_obs[:, 0:1, :, :]
    u_GT = x_obs[:, 1:2, :, :]
    known_index_a = mask[:, 0:1, :, :]
    known_index_u = mask[:, 1:2, :, :]

    # Latents & Schedule
    latents = torch.randn([batch_size, net.img_channels, resolution, resolution], device=device)
    class_labels = None
    if net.label_dim:
        class_labels = torch.eye(net.label_dim, device=device)[torch.randint(net.label_dim, size=[batch_size], device=device)]

    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    sigma_t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    sigma_t_steps = torch.cat([net.round_sigma(sigma_t_steps), torch.zeros_like(sigma_t_steps[:1])])

    x_next = latents.to(torch.float64) * sigma_t_steps[0]

    for i, (sigma_t_cur, sigma_t_next) in enumerate(zip(sigma_t_steps[:-1], sigma_t_steps[1:])):
        x_cur = x_next.detach().clone()
        x_cur.requires_grad = True
        sigma_t = net.round_sigma(sigma_t_cur)
        
        # 1. Heun Step
        x_N = net(x_cur, sigma_t, class_labels=class_labels).to(torch.float64)
        d_cur = (x_cur - x_N) / sigma_t
        x_next = x_cur + (sigma_t_next - sigma_t) * d_cur
        
        if i < num_steps - 1:
            x_N = net(x_next, sigma_t_next, class_labels=class_labels).to(torch.float64)
            d_prime = (x_next - x_N) / sigma_t_next
            x_next = x_cur + (sigma_t_next - sigma_t) * (0.5 * d_cur + 0.5 * d_prime)
        
        # 2. Gradient Guidance
        if (zeta_obs_a > 0 or zeta_obs_u > 0 or zeta_pde > 0):
            
            # Use slicing [:, 0:1, :, :] to preserve (B, 1, H, W)
            a_N = x_N[:, 0:1, :, :]
            u_N = x_N[:, 1:2, :, :]
            
            # Apply Scaling
            a_N = ((a_N + 1.5) / 0.2).to(torch.float64)
            u_N = ((u_N + 0.9) / 115).to(torch.float64)            

            pde_loss, obs_loss_a, obs_loss_u = get_darcy_loss(
                a_N, u_N, a_GT, u_GT, known_index_a, known_index_u)

            # Sum over Batch using p=2, dim=1
            L_pde = pde_loss.view(batch_size, -1).norm(p=2, dim=1).sum() / (resolution ** 2)
            L_obs_a = obs_loss_a.view(batch_size, -1).norm(p=2, dim=1).sum()
            L_obs_u = obs_loss_u.view(batch_size, -1).norm(p=2, dim=1).sum()
            
            grad_x_cur_obs_a = torch.autograd.grad(outputs=L_obs_a, inputs=x_cur, retain_graph=True)[0]
            grad_x_cur_obs_u = torch.autograd.grad(outputs=L_obs_u, inputs=x_cur, retain_graph=True)[0]
            grad_x_cur_pde = torch.autograd.grad(outputs=L_pde, inputs=x_cur)[0]
            
            if i <= 0.8 * num_steps:
                x_next = x_next - zeta_obs_a * grad_x_cur_obs_a - zeta_obs_u * grad_x_cur_obs_u
            else:
                x_next = x_next - 0.1 * (zeta_obs_a * grad_x_cur_obs_a + zeta_obs_u * grad_x_cur_obs_u) - zeta_pde * grad_x_cur_pde

    # Final Output Construction
    x_final = x_next
    
    # Use slicing [:, 0:1, :, :] to keep (B, 1, H, W)
    a_final = x_final[:, 0:1, :, :]
    u_final = x_final[:, 1:2, :, :]
    
    # Apply Scaling
    a_final = ((a_final + 1.5) / 0.2).to(torch.float64)
    
    # Apply Thresholding (In-place works fine here)
    a_final[a_final > 7.5] = 12.0
    a_final[a_final <= 7.5] = 3.0
    
    u_final = ((u_final + 0.9) / 115).to(torch.float64)

    # Concatenate along channel dim (dim 1)
    # Input shapes are (B, 1, H, W) -> Output shape is (B, 2, H, W)
    return torch.cat([a_final, u_final], dim=1).float()