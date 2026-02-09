import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LogNorm, ListedColormap, BoundaryNorm
import os
from utils import FDM_Darcy_loss

# --- 2. Updated Plotting Script for .npy files ---
def plot_comparison(model_files, save_name="darcy_uncond_comparison.pdf"):
    """
    model_files: Dict { 
        'Model Name': {
            'a': 'path/to/samples_a.npy',
            'u': 'path/to/samples_u.npy'
        } 
    }
    """
    models = list(model_files.keys())
    num_models = len(models)
    
    # Figure setup: N rows x 3 columns
    fig, axes = plt.subplots(num_models, 3, figsize=(10, 3 * num_models), constrained_layout=True)
    if num_models == 1: axes = axes[np.newaxis, :]
        
    # Visualization Settings
    # We define boundaries. Value 3 goes to bin [0, 7.5], Value 12 goes to bin [7.5, 15]
    cmap_a = ListedColormap(["#440154", "#fde725"]) # Dark Purple & Bright Yellow (Viridis extremes)
    bounds_a = [0, 7.5, 15]
    norm_a = BoundaryNorm(bounds_a, cmap_a.N)    
    res_norm = LogNorm(vmin=1e-2, vmax=1e3) # Requested Log Scale

    for i, model_name in enumerate(models):
        paths = model_files[model_name]
        path_a = paths['a']
        path_u = paths['u']
        
        try:
            # Load .npy files directly
            # .npy usually loads the array directly. 
            # If your script saved lists, wrap in np.array just in case, 
            # but usually np.save/load handles arrays.
            a_batch = np.load(path_a)
            u_batch = np.load(path_u)
            
            # Select the FIRST sample
            # Expected shape: (B, C, H, W) or (B, H, W) or (H, W)
            # We want (H, W)
            
            # Take a random sample (fixed seed for reproducibility)
            np.random.seed(0)
            idx = np.random.randint(0, a_batch.shape[0])
            a_sample = a_batch[idx]
            u_sample = u_batch[idx]
            
            # Remove channel dim if present (e.g., if shape is (1, 128, 128))
            if a_sample.ndim == 3: a_sample = a_sample.squeeze()
            if u_sample.ndim == 3: u_sample = u_sample.squeeze()
                
        except Exception as e:
            print(f"Error loading {model_name} from {paths}: {e}")
            a_sample = np.zeros((128, 128))
            u_sample = np.zeros((128, 128))

        # Calculate Residual using your FDM function
        # FDM function expects (B, H, W) -> add dims to make (1, 128, 128)
        a_in = a_sample[None, ...]
        u_in = u_sample[None, ...]
        
        residual_map = FDM_Darcy_loss(u_in, a_in, D=1.0, use_mask=False)
        residual_plot = residual_map[0] # Squeeze back to 2D for plotting

        # Plot Coefficient
        im_a = axes[i, 0].imshow(a_sample, cmap=cmap_a, norm=norm_a, origin='lower')
        axes[i, 0].set_ylabel(model_name, fontsize=12, fontweight='bold')
        if i == 0: axes[i, 0].set_title(r"Coefficient $\mathbf{a}$")
        axes[i, 0].set_xticks([]); axes[i, 0].set_yticks([])

        # Plot Solution
        im_u = axes[i, 1].imshow(u_sample, cmap='viridis', origin='lower')
        if i == 0: axes[i, 1].set_title(r"Solution $\mathbf{u}$")
        axes[i, 1].set_xticks([]); axes[i, 1].set_yticks([])

        # Plot Residual
        im_r = axes[i, 2].imshow(residual_plot, cmap='coolwarm', norm=res_norm, origin='lower')
        if i == 0: axes[i, 2].set_title(r"PDE Residual $|\mathcal{R}|$")
        axes[i, 2].set_xticks([]); axes[i, 2].set_yticks([])

    # Add Colorbars (One per column)
    cbar_a = fig.colorbar(im_a, ax=axes[:, 0], location='bottom', fraction=0.05, pad=0.05, ticks=[3.75, 11.25])
    cbar_a.ax.set_xticklabels(['3', '12']) # Set text labels manually
    cbar_a.set_label("Permeability")
    
    cbar_u = fig.colorbar(im_u, ax=axes[:, 1], location='bottom', fraction=0.05, pad=0.05)
    cbar_u.set_label("Pressure")

    cbar_r = fig.colorbar(im_r, ax=axes[:, 2], location='bottom', fraction=0.05, pad=0.05)
    cbar_r.set_label(r"Residual Error ($10^i$)")

    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {save_name}")

if __name__ == "__main__":
    # Updated paths based on your screenshot structure
    files = {
        "DiffusionPDE": {
            "a": "figures_uncond/DiffusionPDE/steps_32/uncond_samples_a_diffpde.npy",
            "u": "figures_uncond/DiffusionPDE/steps_32/uncond_samples_u_diffpde.npy"
        },
        "sCM (Stage 1)": {
            "a": "figures_uncond/sCM-Base/steps_2/uncond_samples_a_cm.npy",
            "u": "figures_uncond/sCM-Base/steps_2/uncond_samples_u_cm.npy"
        },
        "sCM-PINN (Ours)": {
            "a": "figures_uncond/sCM-PINN/steps_2/uncond_samples_a_cm.npy",
            "u": "figures_uncond/sCM-PINN/steps_2/uncond_samples_u_cm.npy"
        }
    }
    
    plot_comparison(files)