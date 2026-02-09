import torch
import pickle
import numpy as np

def get_helmholtz_loss(a, u, a_GT, u_GT, a_mask, u_mask):
    """
    Calculates PDE and Observation losses.
    Expects inputs (a, u) to be in Physics Space (scaled).
    """
    S = u.size(2)
    h = 1 / (S - 1)
    
    # Handle dimensions (B, 1, S, S)
    a = a.view(-1, 1, S, S)
    u = u.view(-1, 1, S, S)
    a_GT = a_GT.view(-1, 1, S, S)
    u_GT = u_GT.view(-1, 1, S, S)

    # 1. PDE Loss Calculation
    u_padded = torch.nn.functional.pad(u, (1, 1, 1, 1), 'constant', 0)
    d2u = (u_padded[:, :, :-2, 1:-1] + u_padded[:, :, 2:, 1:-1] +
           u_padded[:, :, 1:-1, :-2] + u_padded[:, :, 1:-1, 2:] - 4 * u[:, :, :, :]) / h**2
    
    pde_loss = d2u + u - a
    
    # Apply Zero Boundary Conditions
    pde_loss[:, :, 0, :] = 0
    pde_loss[:, :, -1, :] = 0
    pde_loss[:, :, :, 0] = 0
    pde_loss[:, :, :, -1] = 0
    
    # 2. Observation Loss
    observation_loss_a = (a - a_GT) * a_mask 
    observation_loss_u = (u - u_GT) * u_mask
    
    return pde_loss, observation_loss_a, observation_loss_u

def load_pickle_model(model_path, device):
    """Loads the pre-trained .pkl model."""
    with open(model_path, 'rb') as f:
        net = pickle.load(f)['ema'].to(device)
    return net

def sample_guided_diffusion(net, x_obs, mask, num_steps=32, rho=7, 
                            zeta_obs_a=0.8, zeta_obs_u=0.0, zeta_pde=1.0, 
                            sigma_min=0.002, sigma_max=80, device='cuda'):
    """
    Heun sampler with physics-guided gradient updates.
    Uses exact "slow" gradient calculation to match reference script.
    """
    batch_size = x_obs.shape[0]
    resolution = x_obs.shape[2]
    
    # Split Ground Truth (Physics Space)
    a_GT = x_obs[:, 0:1, :, :]
    u_GT = x_obs[:, 1:2, :, :]
    
    # Split Mask
    known_index_a = mask[:, 0:1, :, :]
    known_index_u = mask[:, 1:2, :, :]

    # Prepare Latents
    latents = torch.randn([batch_size, net.img_channels, resolution, resolution], device=device)
    class_labels = None
    if net.label_dim:
        class_labels = torch.eye(net.label_dim, device=device)[torch.randint(net.label_dim, size=[batch_size], device=device)]

    # Prepare Sigma Schedule
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    sigma_t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    sigma_t_steps = torch.cat([net.round_sigma(sigma_t_steps), torch.zeros_like(sigma_t_steps[:1])])

    # Initialize x
    x_next = latents.to(torch.float64) * sigma_t_steps[0]

    # Sampling Loop
    for i, (sigma_t_cur, sigma_t_next) in enumerate(zip(sigma_t_steps[:-1], sigma_t_steps[1:])):
        x_cur = x_next.detach().clone()
        x_cur.requires_grad = True
        sigma_t = net.round_sigma(sigma_t_cur)
        
        # --- 1. Heun Euler Step ---
        x_N = net(x_cur, sigma_t, class_labels=class_labels).to(torch.float64)
        d_cur = (x_cur - x_N) / sigma_t
        x_next = x_cur + (sigma_t_next - sigma_t) * d_cur
        
        # --- 2. Heun 2nd Order Correction ---
        if i < num_steps - 1:
            x_N = net(x_next, sigma_t_next, class_labels=class_labels).to(torch.float64)
            d_prime = (x_next - x_N) / sigma_t_next
            x_next = x_cur + (sigma_t_next - sigma_t) * (0.5 * d_cur + 0.5 * d_prime)
        
        # --- 3. Gradient Guidance (Exact Replication) ---
        if (zeta_obs_a > 0 or zeta_obs_u > 0 or zeta_pde > 0):
            
            # Scale network output to Physics Space 
            a_N = (x_N[:, 0:1, :, :] * 2.15).to(torch.float64)
            u_N = (x_N[:, 1:2, :, :] * 0.028).to(torch.float64)
            
            pde_loss, obs_loss_a, obs_loss_u = get_helmholtz_loss(
                a_N, u_N, a_GT, u_GT, known_index_a, known_index_u)

            # Norm calculation matching reference
            L_pde = torch.norm(pde_loss, 2) / (127 ** 2)
            L_obs_a = torch.norm(obs_loss_a, 2)
            L_obs_u = torch.norm(obs_loss_u, 2)

            # Exact separate gradient calculation
            # We use retain_graph=True for the first two calls
            grad_x_cur_obs_a = torch.autograd.grad(outputs=L_obs_a, inputs=x_cur, retain_graph=True)[0]
            grad_x_cur_obs_u = torch.autograd.grad(outputs=L_obs_u, inputs=x_cur, retain_graph=True)[0]
            grad_x_cur_pde = torch.autograd.grad(outputs=L_pde, inputs=x_cur)[0]
            
            # Exact update logic
            if i <= 0.8 * num_steps:
                x_next = x_next - zeta_obs_a * grad_x_cur_obs_a - zeta_obs_u * grad_x_cur_obs_u
            else:
                x_next = x_next - 0.1 * (zeta_obs_a * grad_x_cur_obs_a + zeta_obs_u * grad_x_cur_obs_u) - zeta_pde * grad_x_cur_pde

    # Final Post-Process to Physics Space
    x_final = x_next
    a_final = x_final[:, 0:1, :, :] * 2.15
    u_final = x_final[:, 1:2, :, :] * 0.028
    
    # Return stack (B, 2, 128, 128)
    return torch.cat([a_final, u_final], dim=1).float()