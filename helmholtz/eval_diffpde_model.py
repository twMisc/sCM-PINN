# %%
import torch
import scipy
import pickle

import tqdm

def get_helmholtz_loss(a, u, a_GT, u_GT, a_mask, u_mask, device=torch.device('cuda')):
    """Return the loss of the Helmholtz equation and the observation loss."""
    S = u.size(2)
    h = 1 / (S - 1)
    a = a.view(1, 1, S, S)
    u_padded = torch.nn.functional.pad(u, (1, 1, 1, 1), 'constant', 0)
    d2u = (u_padded[:, :, :-2, 1:-1] + u_padded[:, :, 2:, 1:-1] +
           u_padded[:, :, 1:-1, :-2] + u_padded[:, :, 1:-1, 2:] - 4 * u[:, :, :, :]) / h**2
    pde_loss = d2u + u - a
    pde_loss = pde_loss.squeeze()
    pde_loss[0, :] = 0
    pde_loss[-1, :] = 0
    pde_loss[:, 0] = 0
    pde_loss[:, -1] = 0
    
    a_GT = a_GT.view(1, 1, S, S)
    u_GT = u_GT.view(1, 1, S, S)
    observation_loss_a = (a - a_GT).squeeze()
    observation_loss_a = observation_loss_a * a_mask  
    observation_loss_u = (u - u_GT).squeeze()
    observation_loss_u = observation_loss_u * u_mask
    
    return pde_loss, observation_loss_a, observation_loss_u
# %%
config = {
    'data': { 'datapath': '../DiffusionPDE_data/testing/helmholtz.mat', 'offset': 0 },
    'test': { 'pre-trained': '../DiffusionPDE_data/pretrained-models/pretrained-helmholtz.pkl', 'iterations': 128 },
    'generate': {
        'seed': 0,
        'device': 'cuda', 
        'batch_size': 1,
        'sigma_min': 0.002,
        'sigma_max': 80,
        'rho': 7,
        'zeta_obs_a': 0.8,
        'zeta_obs_u': 0,
        'zeta_pde': 1
    }
}
num_steps = config['test']['iterations']
# %%
datapath = config['data']['datapath']
device = config['generate']['device']

data = scipy.io.loadmat(datapath)
all_a_GT = data['f_data']
all_u_GT = data['psi_data']

batch_size = config['generate']['batch_size']
seed = config['generate']['seed']
torch.manual_seed(seed)

network_pkl = config['test']['pre-trained']
f = open(network_pkl, 'rb')
net = pickle.load(f)['ema'].to(device)

sigma_min = config['generate']['sigma_min']
sigma_max = config['generate']['sigma_max']
sigma_min = max(sigma_min, net.sigma_min)
sigma_max = min(sigma_max, net.sigma_max)

rho = config['generate']['rho']

# Handle Zeta Logic
zeta_obs_a = config['generate']['zeta_obs_a']
zeta_obs_u = config['generate']['zeta_obs_u']

# %%
resolution = net.img_resolution
h = 1.0 / (resolution - 1)
# %%
offset = 0
a_GT = torch.tensor(all_a_GT[offset, :, :], dtype=torch.float64, device=device)
u_GT = torch.tensor(all_u_GT[offset, :, :], dtype=torch.float64, device=device)
# Prepare Latents
latents = torch.randn([batch_size, net.img_channels, resolution, resolution], device=device)
class_labels = None
if net.label_dim:
    class_labels = torch.eye(net.label_dim, device=device)[torch.randint(net.label_dim, size=[batch_size], device=device)]

# Prepare Steps
step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
sigma_t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
sigma_t_steps = torch.cat([net.round_sigma(sigma_t_steps), torch.zeros_like(sigma_t_steps[:1])])

# Initialize x
x_next = latents.to(torch.float64) * sigma_t_steps[0]

# Masks
# if use_full_a:
known_index_a = torch.ones((resolution, resolution), dtype=torch.float32).to(device)
# else:
#     known_index_a = random_index(500, resolution, seed=1, device=device)
    
# known_index_u = random_index(500, resolution, seed=0, device=device)

# Pre-calculate Normalized GT for Hard Constraint
# Network normalization logic reversed: a_net = (a_real * 0.2) - 1.5
# a_GT_norm = (a_GT * 0.2) - 1.5

iterator = tqdm.tqdm(list(enumerate(zip(sigma_t_steps[:-1], sigma_t_steps[1:]))), unit='step', leave=False)
zeta_pde = config['generate']['zeta_pde']
known_index_u = torch.zeros((resolution, resolution), dtype=torch.float32).to(device)

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
        L_pde = pde_loss.view(batch_size, -1).norm(p=2, dim=1).sum() / (128 ** 2)
        L_obs_a = obs_loss_a.view(batch_size, -1).norm(p=2, dim=1).sum()
        L_obs_u = obs_loss_u.view(batch_size, -1).norm(p=2, dim=1).sum()
        
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
            
            
# Post-process
x_final = x_next
a_final_ts = x_final[:,0,:,:].unsqueeze(0)
u_final_ts = x_final[:,1,:,:].unsqueeze(0)
a_final_ts = ((a_final_ts*2.15).to(torch.float64))
u_final_ts = ((u_final_ts*0.028).to(torch.float64))

# %%
import matplotlib.pyplot as plt
with torch.no_grad():
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.imshow(a_final_ts[0, 0].cpu().numpy(), cmap='viridis')
    plt.colorbar()
    plt.title('Predicted a')
    plt.subplot(1,2,2)
    plt.imshow(u_final_ts[0, 0].cpu().numpy(), cmap='viridis')
    plt.colorbar()
    plt.title('Predicted u')
    plt.show()
# %%
with torch.no_grad():
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.imshow(a_GT.cpu().numpy(), cmap='viridis')
    plt.colorbar()
    plt.title('Ground Truth a')
    plt.subplot(1,2,2)
    plt.imshow(u_GT.cpu().numpy(), cmap='viridis')
    plt.colorbar()
    plt.title('Ground Truth u')
    plt.show()
# %%
# calculate the errors: relative L2 and H1 norm
from utils import calculate_h1_error
import numpy as np
true_u = u_GT.cpu().numpy()
pred_u = u_final_ts[0,0].detach().cpu().numpy()
l2_error_u = np.linalg.norm(pred_u - true_u) / (np.linalg.norm(true_u) + 1e-12)
h1_norm_u, h1_error_u = calculate_h1_error(pred_u, true_u, h=h)
print(f'Relative L2 Error of u: {l2_error_u:.4f}')
print(f'H1 Norm of u: {h1_norm_u:.4f}, Relative H1 Error of u: {h1_error_u:.4f}')
# %%