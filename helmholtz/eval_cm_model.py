# %%
import numpy as np 
import torch
import matplotlib.pyplot as plt
from networks_util import create_model, load_model_state, create_sep_model
from utils import consistency_sample_cm, helmholtz_loss
from utils import rescale_a, rescale_u, scale_back_a, scale_back_u
# %%
# torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = create_model().to(device)
# model = load_model_state(model, './helmholtz-output/consistency/model_epoch/model_epoch_8.pth')
model = create_sep_model().to(device)
model = load_model_state(model,'./helmholtz-output/consistency-fdm/model_epoch/model_epoch_2.pth')
model.eval()
# %%
pred = consistency_sample_cm(model, use_seeded_z=False, t_list=[np.pi/2, 1.1], sigma_min=1e-8)

# %%
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.imshow(pred[0,0].cpu().numpy(), cmap='viridis')
plt.colorbar()
plt.title('Predicted a')
plt.subplot(1,2,2)
plt.imshow(pred[0,1].cpu().numpy(), cmap='viridis')
plt.colorbar()
plt.title('Predicted u')
plt.show()
# %%
u_sample = scale_back_u(pred[:,1,:,:])
a_sample = scale_back_a(pred[:,0,:,:])
# a_sample = discrete_a(a_sample)
# %%
# compute the residual using FDM_Darcy_loss
residual = helmholtz_loss(u_sample, a_sample, return_residual=True)
# %%
plt.figure(figsize=(6,6))
plt.title('Residual of the Helmholtz Equation')
plt.imshow(np.abs(residual.cpu())[0][1:-1, 1:-1], cmap='viridis')
plt.colorbar()
plt.show()
# %%
# log scale the residual for better visualization
plt.figure(figsize=(6,6))
plt.title('Log-Scaled Residual of the Helmholtz Equation')
plt.imshow(np.log10(np.abs(residual.cpu())[0][1:-1, 1:-1] + 1e-12), cmap='viridis')
plt.colorbar()
plt.show()
# %%
# check forward problem
from utils import load_test_data
dataset, dataloader = load_test_data(batch_size=1, return_dataset=True, rescale=False)
# %%
ind = np.random.randint(0, len(dataset))
real_data = dataset[ind:ind+1]
a_real = (real_data[0].numpy())
u_real = (real_data[1].numpy())
# %%
mask = torch.ones((1, 2, 128, 128)).to(device)
# mask second channel zeros
mask[:,1,:,:] = 0
a_real = torch.tensor(a_real).to(device)#.double()
u_real = torch.tensor(u_real).to(device)#.double()
x_obs = torch.stack([rescale_a(a_real), rescale_u(u_real)], dim=1)  # (B, 2, 128, 128)
pred = consistency_sample_cm(model, x_obs=x_obs, mask=mask, use_seeded_z=False, n_steps=64, schedule='power', sigma_min=1e-8)
# %%
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.imshow(pred[0,0].cpu().numpy(), cmap='viridis')
plt.colorbar(fraction=0.046, pad=0.04)
plt.title('Predicted a (Forward Problem)')
plt.subplot(1,2,2)
plt.imshow(pred[0,1].cpu().numpy(), cmap='viridis')
plt.colorbar(fraction=0.046, pad=0.04)
plt.title('Predicted u (Forward Problem)')
plt.show()
# %%
# scale back pred
a_pred = scale_back_a(pred[:,0,:,:])
u_pred = scale_back_u(pred[:,1,:,:])
# %%
plt.imshow(torch.abs(u_pred - u_real)[0].cpu(), cmap='viridis') 
plt.colorbar(fraction=0.046, pad=0.04)
# %%
# calculate relative error
print('Relative L2 Error in u:', torch.norm(u_pred[0] - u_real[0], 2) / torch.norm(u_real[0], 2)       )
# %%
plt.imshow(u_pred[0].cpu(), cmap='viridis')
plt.colorbar()
# %%
plt.imshow(u_real[0].cpu(), cmap='viridis')
plt.colorbar()
# %%
# check real data loss
residual_real = helmholtz_loss(u_real[0:1], a_real[0:1], return_residual=True)**2
plt.imshow((np.abs(residual_real.cpu())[0][1:-1, 1:-1]), cmap='viridis')
plt.colorbar(fraction=0.046, pad=0.04)

# %%
# calculate H1 norm and H1 error of the generated solution
from utils import calculate_h1_error

h1_error, h1_error_rel = calculate_h1_error(u_pred.cpu().numpy()[0], u_real.cpu().numpy()[0])
print('H1 Error:', h1_error)
print('Relative H1 Error:', h1_error_rel)
h1_norm, _ = calculate_h1_error(u_pred.cpu().numpy()[0], np.zeros_like(u_pred.cpu().numpy()[0]))
print('H1 Norm of the predicted solution:', h1_norm)
h1_norm_real, _ = calculate_h1_error(u_real.cpu().numpy()[0], np.zeros_like(u_real.cpu().numpy()[0]))
print('H1 Norm of the real solution:', h1_norm_real)
print('Relative L2 Error in u:', torch.norm(u_pred[0] - u_real[0], 2).item() / torch.norm(u_real[0], 2).item()       )

# %%
@torch.no_grad()
def find_golden_noise(
    model,
    x_obs,
    mask,
    sigma_data=0.5,
    sigma_max=80.0,
    device="cuda",
    candidates=64, # How many random noises to test
    shape=(1, 2, 128, 128)
):
    """
    Finds the 'Golden Noise' z that naturally aligns best with the unmasked observations.
    No gradients, just forward pass selection.
    """
    B, C, H, W = shape
    
    # 1. Generate a batch of candidates
    # We expand the batch dimension to 'candidates'
    z_candidates = torch.randn((candidates, C, H, W), device=device)
    
    # 2. Setup inputs for the "Peek" step (at sigma_max)
    sigma_tensor = torch.full((candidates,), sigma_max, device=device)
    t = torch.atan(sigma_tensor / sigma_data)
    
    # Precompute Trig terms
    t_in = t.view(candidates, 1, 1, 1)
    cos_t = torch.cos(t_in)
    sin_t = torch.sin(t_in)
    
    # Map noise to CM state (Geometric Projection)
    # At t=0 (sigma=max), x ~ z * sigma_max * cos(t) roughly, but let's use exact formula
    # x_t = cos(t)*x0 + sin(t)*sigma_data*eps. Here x0=0 (pure noise assumption for init)
    # Actually, simpler: Initialize x_t like the sampler does
    x_t = z_candidates * sigma_max 
    
    # Convert to CM input scaling
    # x_in = x_t * cos(t)  (Using the Heun/CM bridge logic we established)
    x_in = x_t * cos_t 

    # 3. Run Model (Batched Forward Pass)
    F = model(x_in / sigma_data, t, return_logvar=False)
    
    # 4. Reconstruct x0
    pred_x0 = cos_t * x_in - sin_t * sigma_data * F
    
    # 5. Calculate Error against Observable Data
    # We only care about how well it matches the *unmasked* (known) part.
    # Note: mask usually is 1 for Keep, 0 for Drop (or vice versa). 
    # Let's assume 'mask' is 1 where we have data (x_obs).
    
    # Expand x_obs and mask to match batch size
    x_obs_batch = x_obs.repeat(candidates, 1, 1, 1)
    mask_batch = mask.repeat(candidates, 1, 1, 1)
    
    # Error = MSE(pred, obs) * mask
    diff = (pred_x0 - x_obs_batch) * mask_batch
    
    # Sum error per sample
    # flatten to (B, -1) then sum squares
    errors = (diff ** 2).view(candidates, -1).sum(dim=1)
    
    # 6. Pick the Winner
    best_idx = torch.argmin(errors)
    best_z = z_candidates[best_idx].unsqueeze(0) # Keep shape (1, C, H, W)
    
    print(f"Golden Noise Found: Best Error {errors[best_idx]:.4f} vs Worst {errors.max():.4f}")
    
    return best_z

# 1. Find the best starting point
golden_z = find_golden_noise(
    model, 
    x_obs=x_obs, 
    mask=mask, 
    candidates=64  # Higher = better stability, more VRAM
)

# 2. Run your existing sampler with this specific z
# Note: Use use_seeded_z=False so it uses the 'z' you passed
pred = consistency_sample_cm(
    model, 
    x_obs=x_obs, 
    mask=mask, 
    use_seeded_z=False,
    n_steps=64, 
    schedule='power', 
    sigma_min=1e-8,
    device=device
)
# %%
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.imshow(pred[0,0].cpu().numpy(), cmap='viridis')
plt.colorbar(fraction=0.046, pad=0.04)
plt.title('Predicted a (Forward Problem)')
plt.subplot(1,2,2)
plt.imshow(pred[0,1].cpu().numpy(), cmap='viridis')
plt.colorbar(fraction=0.046, pad=0.04)
plt.title('Predicted u (Forward Problem)')
plt.show()
# %%
# scale back pred
a_pred = scale_back_a(pred[:,0,:,:])
u_pred = scale_back_u(pred[:,1,:,:])
# %%
plt.imshow(torch.abs(u_pred - u_real)[0].cpu(), cmap='viridis') 
plt.colorbar(fraction=0.046, pad=0.04)
# %%
# calculate relative error
print('Relative L2 Error in u:', torch.norm(u_pred[0] - u_real[0], 2) / torch.norm(u_real[0], 2)       )
# %%
plt.imshow(u_pred[0].cpu(), cmap='viridis')
plt.colorbar()
# %%
plt.imshow(u_real[0].cpu(), cmap='viridis')
plt.colorbar()
# %%
# # check real data loss
# residual_real = helmholtz_loss(u_real[0:1], a_real[0:1], return_residual=True)**2
# plt.imshow((np.abs(residual_real.cpu())[0][1:-1, 1:-1]), cmap='viridis')
# plt.colorbar(fraction=0.046, pad=0.04)

# %%
# calculate H1 norm and H1 error of the generated solution
from utils import calculate_h1_error

h1_error, h1_error_rel = calculate_h1_error(u_pred.cpu().numpy()[0], u_real.cpu().numpy()[0])
print('H1 Error:', h1_error)
print('Relative H1 Error:', h1_error_rel)
h1_norm, _ = calculate_h1_error(u_pred.cpu().numpy()[0], np.zeros_like(u_pred.cpu().numpy()[0]))
print('H1 Norm of the predicted solution:', h1_norm)
h1_norm_real, _ = calculate_h1_error(u_real.cpu().numpy()[0], np.zeros_like(u_real.cpu().numpy()[0]))
print('H1 Norm of the real solution:', h1_norm_real)
print('Relative L2 Error in u:', torch.norm(u_pred[0] - u_real[0], 2).item() / torch.norm(u_real[0], 2).item()       )

# %%
