# %%
# load mat files and check dataset
import os 
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import torch
# %%
dataset_path = '../DiffusionPDE_data/training/poisson/'
file_list = os.listdir(dataset_path)
print(f"Total number of files: {len(file_list)}")
# %%
# load a sample file
sample_file = os.path.join(dataset_path, file_list[0])
data = scipy.io.loadmat(sample_file)
# %%
# check keys in the mat file
print(data.keys())
# %%
# extract a and u
a = data['f_data']  # coefficient
u = data['phi_data']  # solution
# %%
# check shapes
print(f"a shape: {a.shape}, u shape: {u.shape}")
# %%
# visualize a and u
ind = np.random.randint(0, a.shape[0])
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(a[ind], cmap='viridis')
plt.title('Coefficient a')
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(u[ind], cmap='viridis')
plt.title('Solution u')
plt.colorbar()
plt.show()
# %%
from utils import load_data
dataset, dataloader = load_data(return_dataset=True, batch_size=4, rescale=False)
# %%
from utils import poisson_loss
# compute poisson loss for a sample
a_sample = dataset[ind+1][0].unsqueeze(0)
u_sample = dataset[ind+1][1].unsqueeze(0)
poisson_residual = poisson_loss(u_sample, a_sample, return_residual=True)**2
# %%
plt.figure(figsize=(6, 5))
plt.imshow(poisson_residual[0].numpy(), cmap='viridis')
plt.title('Poisson Residual from poisson_loss')
plt.colorbar()
# %%
