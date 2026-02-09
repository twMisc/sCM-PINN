# %%
import numpy as np 
import torch
import matplotlib.pyplot as plt
from networks_util import create_model, load_model_state, create_sep_model
from utils import consistency_sample_cm, poisson_loss
from utils import rescale_a, rescale_u, scale_back_a, scale_back_u
# %%
# torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_model().to(device)
model = load_model_state(model, './poisson-diffpde-output/consistency/model_epoch/model_epoch_8.pth')
# model = create_sep_model().to(device)
# model = load_model_state(model,'./poisson-diffpde-output/consistency-fdm/model_epoch/model_epoch_2.pth')
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
residual = poisson_loss(u_sample, a_sample, return_residual=True)
# %%
plt.figure(figsize=(6,6))
plt.title('Residual of the Poisson Equation')
plt.imshow(np.abs(residual.cpu())[0][1:-1, 1:-1], cmap='viridis')
plt.colorbar()
plt.show()
# %%
# log scale the residual for better visualization
plt.figure(figsize=(6,6))
plt.title('Log-Scaled Residual of the Poisson Equation')
plt.imshow(np.log10(np.abs(residual.cpu())[0][1:-1, 1:-1] + 1e-12), cmap='viridis')
plt.colorbar()
plt.show()
# %%
# check forward problem
from utils import load_test_data
dataset, dataloader = load_test_data(batch_size=1, return_dataset=True, rescale=False)
# %%
real_data = dataset[0:1]
a_real = (real_data[0].numpy())
u_real = (real_data[1].numpy())
# %%
mask = torch.ones((1, 2, 128, 128)).to(device)
# mask second channel zeros
mask[:,1,:,:] = 0
a_real = torch.tensor(a_real).to(device)#.double()
u_real = torch.tensor(u_real).to(device)#.double()
x_obs = torch.stack([rescale_a(a_real), rescale_u(u_real)], dim=1)  # (B, 2, 128, 128)
pred = consistency_sample_cm(model, x_obs=x_obs, mask=mask, use_seeded_z=False, n_steps=128, schedule='power', sigma_min=1e-4)
# %%
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.imshow(pred[0,0].cpu().numpy(), cmap='viridis')
plt.colorbar()
plt.title('Predicted a (Forward Problem)')
plt.subplot(1,2,2)
plt.imshow(pred[0,1].cpu().numpy(), cmap='viridis')
plt.colorbar()
plt.title('Predicted u (Forward Problem)')
plt.show()
# %%
# scale back pred
a_pred = scale_back_a(pred[:,0,:,:])
u_pred = scale_back_u(pred[:,1,:,:])
# %%
plt.imshow(torch.abs(u_pred - u_real)[0].cpu(), cmap='viridis') 
plt.colorbar()
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
residual_real = poisson_loss(u_real[0:1], a_real[0:1], return_residual=True)**2
plt.imshow((np.abs(residual_real.cpu())[0][1:-1, 1:-1]), cmap='viridis')
plt.colorbar()

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
