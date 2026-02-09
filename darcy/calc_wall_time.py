"""Calculate the wall time of the sampling process for sCM-PINN vs DiffusionPDE."""

import os
import csv
import click
import numpy as np 
import torch
import random
import matplotlib.pyplot as plt
from utils import load_test_data, rescale_a, sample_dpm_solver, consistency_sample_cm, scale_back_a, scale_back_u, rescale_u, rescale_a, FDM_Darcy_loss, Energy_Darcy_loss,calculate_h1_error, discrete_a
from networks_util import create_model, create_sep_model, load_model_state
from diffpde_backend import load_pickle_model, sample_guided_diffusion
from eval_task_error import generate_mask_for_batch
import time

print("Setting random seeds for reproducibility...")
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# load models
print("Loading models...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = '../DiffusionPDE_data/pretrained-models/pretrained-darcy.pkl'
diffpde_model = load_pickle_model(model_path, device)
diffpde_model.eval()


model_path = './darcy_redo_output/sCM/consistency-fdm-sep-nomask-uniform/model_epoch/model_epoch_1.pth'
model = create_sep_model()
model = load_model_state(model, model_path)
model = model.to(device)
model.eval()

print("Models loaded successfully. Loading test data...")
# Load test data
batch_size = 1
total_samples = 1
dataset, dataloader = load_test_data(batch_size, return_dataset=True, rescale=False)
inds = np.random.choice(len(dataset), total_samples, replace=False)
real_data = dataset[inds]
a_data = real_data[0].numpy()
u_data = real_data[1].numpy()
mask = generate_mask_for_batch('forward', batch_size, device, 0)
i=0
a_batch = a_data[i:i+batch_size]
u_batch = u_data[i:i+batch_size]
a_batch = torch.tensor(a_batch).to(device)
u_batch = torch.tensor(u_batch).to(device)
x_obs = torch.stack([rescale_a(a_batch), rescale_u(u_batch)], dim=1).to(torch.float32)  # (B, 2, 128, 128)
x_obs_raw = torch.stack([a_batch, u_batch], dim=1)
rho = 7
zeta_obs_a = 5.0
zeta_obs_u = 0.0
zeta_pde = 100.0
num_steps = 32


print("Test data loaded successfully. Starting DiffusionPDE sampling...")
# DiffusionPDE sampling
# calculate wall time
start_time = time.time()
with torch.enable_grad():
    pred = sample_guided_diffusion(
        diffpde_model, 
        x_obs=x_obs_raw, 
        mask=mask, 
        num_steps=num_steps, 
        rho=rho,
        zeta_obs_a=zeta_obs_a, 
        zeta_obs_u=zeta_obs_u, 
        zeta_pde=zeta_pde,
        device=device
    )
end_time = time.time()
diffpde_wall_time = end_time - start_time
print(f"DiffusionPDE sampling wall time: {diffpde_wall_time:.4f} seconds")

print("Starting sCM-PINN sampling...")
# sCM-PINN sampling
cm_steps = 64
# calculate wall time
start_time = time.time()
with torch.no_grad():
    pred = consistency_sample_cm(model, sigma_data=0.5, device=device, shape=(batch_size, 2, 128, 128), schedule='power', return_intermediates=False, use_seeded_z=False, x_obs=x_obs, mask=mask, n_steps=cm_steps)
end_time = time.time()
scm_wall_time = end_time - start_time
print(f"sCM-PINN sampling wall time: {scm_wall_time:.4f} seconds")
