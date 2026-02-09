# %%
import os
import numpy as np 
import torch
import matplotlib.pyplot as plt
from networks_util import create_model, load_model_state, create_sep_model
from utils import consistency_sample_cm, FDM_Darcy_loss
from utils import rescale_a, rescale_u, scale_back_a, scale_back_u, discrete_a, load_test_data, heun_sample_cm, sample_dpm_solver_cm, calculate_h1_error

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)    
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_sep = create_sep_model().to(device)
model_sep = load_model_state(model_sep,'./darcy_redo_output/sCM/consistency-fdm-sep-nomask-uniform/model_epoch/model_epoch_1.pth')
model_sep.eval()
model_no_sep = create_model().to(device)
model_no_sep = load_model_state(model_no_sep, './darcy_redo_output/sCM/consistency-no-sep-net/model_epoch/model_epoch_1.pth')
model_no_sep.eval()

n_samples = 128
dataset, dl = load_test_data(1, return_dataset=True, rescale=False)
a_datas, u_datas = dataset[:n_samples]

pred_no_sep = []
pred_sep = []
with torch.no_grad():
    for i in range(n_samples):
        # Sample with non-sep model
        a_sample_no_sep = consistency_sample_cm(model_no_sep, use_seeded_z=False, t_list=[np.pi/2, 1.1])
        pred_no_sep.append(a_sample_no_sep.cpu())
        # Sample with sep model
        a_sample_sep = consistency_sample_cm(model_sep, use_seeded_z=False, t_list=[np.pi/2, 1.1])
        pred_sep.append(a_sample_sep.cpu())
pred_no_sep = torch.cat(pred_no_sep, dim=0)
pred_sep = torch.cat(pred_sep, dim=0)

a_samples_no_sep = scale_back_a(pred_no_sep[:,0,:,:]).cpu().numpy()
a_samples_no_sep = discrete_a(a_samples_no_sep)
a_samples_sep = scale_back_a(pred_sep[:,0,:,:]).cpu().numpy()
a_samples_sep = discrete_a(a_samples_sep)
# %%
# Assuming data is flattened numpy arrays
# gt_pixels: Ground Truth (mostly 3s and 12s)
# fail_pixels: No Freeze (Failure case, likely clustered around 12)
# ours_pixels: Frozen Decoder (Should match GT)
gt_pixels = a_datas.flatten().numpy()
fail_pixels = a_samples_no_sep.flatten()
ours_pixels = a_samples_sep.flatten()
plt.figure(figsize=(8, 5))

# Define bins to cover the range [0, 15] clearly
bins = np.linspace(0, 15, 60) # High resolution bins to show sharpness

# Plot Ground Truth (Filled Gray)
plt.hist(gt_pixels, bins=bins, density=True, alpha=0.3, color='gray', 
         label='Ground Truth', edgecolor='none')

# Plot Failure Case (Red Step)
plt.hist(fail_pixels, bins=bins, density=True, histtype='step', linewidth=2, 
         color='red', label='Joint Training (w/o Freeze)')

# Plot Ours (Blue Step)
plt.hist(ours_pixels, bins=bins, density=True, histtype='step', linewidth=2, 
         color='blue', linestyle='--', label='sCM-PINN (Frozen Decoder)')

plt.title("Coefficient Mode Collapse Analysis")
plt.xlabel("Permeability Value $a(x)$")
plt.ylabel("Density (Log Scale)")
plt.yscale('log') # Log scale is often great for seeing the "missing" mode at 3
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
# %%
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def plot_ablation_figure(pixels_gt, pixels_fail, pixels_ours, 
                            maps_gt, maps_fail, maps_ours):
    
    # Increase figure width slightly to accommodate spacing
    fig = plt.figure(figsize=(13, 5))
    
    # Increased wspace from 0.2 -> 0.35 to prevent text clipping
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.3], wspace=0.35)
    
    # --- LEFT PANEL: Histogram ---
    ax_hist = fig.add_subplot(gs[0])
    
    # Custom bins to create two wide "buckets" centered at 3 and 12
    # This captures values like 2.9-3.1 into the "3" bin and 11.9-12.1 into "12"
    # We use a slight range width (e.g., +/- 1.0) for visual thickness
    bins = [2.0, 4.0, 11.0, 13.0] 
    
    # Plot GT (Filled Gray)
    # weights=np.ones... ensures normalized density summing to 1 for valid comparison
    ax_hist.hist(pixels_gt, bins=bins, density=True, color='gray', alpha=0.4, 
                 label='Ground Truth', edgecolor='none')
    
    # Plot Failure (Red Line) - step histogram looks cleaner for comparison
    ax_hist.hist(pixels_fail, bins=bins, density=True, histtype='step', 
                 color='#D62728', linewidth=3, label='Joint Training (No Freeze)')
    
    # Plot Ours (Blue Dashed)
    ax_hist.hist(pixels_ours, bins=bins, density=True, histtype='step', 
                 color='#1F77B4', linewidth=3, linestyle='--', 
                 label='sCM-PINN (Frozen)')
    
    ax_hist.set_yscale('log')
    ax_hist.set_title("Coefficient Distribution (Log Scale)", fontsize=12, pad=10)
    
    # STRICTLY set x-ticks to only 3 and 12
    ax_hist.set_xticks([3, 12])
    ax_hist.set_xticklabels(['3', '12'], fontsize=11, fontweight='bold')
    ax_hist.set_xlabel("Permeability Value $a(x)$")
    
    ax_hist.set_ylabel("Pixel Density")
    ax_hist.grid(True, axis='y', ls="-", alpha=0.2)
    ax_hist.legend(loc='upper left', fontsize=10, framealpha=0.95)
    
    # --- RIGHT PANEL: 3x3 Grid ---
    gs_imgs = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs[1], 
                                               hspace=0.05, wspace=0.05)
    
    rows = [maps_gt, maps_fail, maps_ours]
    row_labels = ["Ground Truth", "Joint Training\n(Failure)", "sCM-PINN\n(Ours)"]
    
    vmin, vmax = 0, 13
    
    for i in range(3): # Rows
        for j in range(3): # Cols
            ax = fig.add_subplot(gs_imgs[i, j])
            im = ax.imshow(rows[i][j], cmap='viridis')
            ax.axis('off')
            
            # Add Row Labels on the left of the first column
            if j == 0:
                # x coordinate moved further left (-0.25) to avoid overlap
                ax.text(-0.25, 0.5, row_labels[i], transform=ax.transAxes, 
                        va='center', ha='right', fontsize=11, fontweight='bold')

    # Add a colorbar for context (optional, but helpful for "3 vs 12")
    # cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
    # fig.colorbar(im, cax=cbar_ax, label='Permeability $a(x)$')
    
    return fig
# Usage:
# fig = plot_ablation_figure(all_gt_pixels, all_fail_pixels, all_ours_pixels,
#                            sample_gt_batch, sample_fail_batch, sample_ours_batch)
# plt.show()
fig = plot_ablation_figure(gt_pixels, fail_pixels, ours_pixels,
                    a_datas[:3].numpy(), a_samples_no_sep[:3], a_samples_sep[:3])
fig.show()
os.makedirs('./figures/adaptation', exist_ok=True)
fig.savefig('./figures/adaptation/ablation_figure.eps', dpi=300, bbox_inches='tight')
fig.savefig('./figures/adaptation/ablation_figure.png', dpi=300, bbox_inches='tight')
fig.savefig('./figures/adaptation/ablation_figure.pdf', dpi=300, bbox_inches='tight')

# %%
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def plot_ablation_vertical(pixels_gt, pixels_fail, pixels_ours, 
                                 maps_gt, maps_fail, maps_ours):
    
    # Vertical format: 6 inches wide, 8 inches tall
    fig = plt.figure(figsize=(6, 8)) 
    
    # Grid: Top (Histogram) vs Bottom (Images)
    # height_ratios=[1, 3] gives images more room
    gs = gridspec.GridSpec(2, 1, height_ratios=[0.8, 3], hspace=0.15)
    
    # --- TOP PANEL: Discrete Bar Chart ---
    ax_hist = fig.add_subplot(gs[0])
    
    # 1. Calculate Densities manually for the two modes
    # We count how many pixels are closer to 3 vs 12
    def get_densities(pixels):
        count_3 = np.sum((pixels > 1) & (pixels < 5))
        count_12 = np.sum((pixels > 10) & (pixels < 14))
        total = count_3 + count_12 + 1e-6
        return [count_3 / total, count_12 / total]

    dens_gt = get_densities(pixels_gt)
    dens_fail = get_densities(pixels_fail)
    dens_ours = get_densities(pixels_ours)
    
    # 2. Plotting at discrete positions x=[0, 1]
    x_pos = np.array([0.25, 0.75])
    width = 0.25 # Width of each bar
    
    # Ground Truth (Center)
    ax_hist.bar(x_pos, dens_gt, width=width, color='gray', alpha=0.4, 
                label='Ground Truth')
    
    # Failure (Left offset)
    # We use 'step' look by plotting edges or just using fill=False
    ax_hist.bar(x_pos - 0.05, dens_fail, width=width, fill=False, 
                edgecolor='#D62728', linewidth=2, label='Joint Training')
    
    # Ours (Right offset)
    ax_hist.bar(x_pos + 0.05, dens_ours, width=width, fill=False, 
                edgecolor='#1F77B4', linewidth=2, linestyle='--', label='sCM-PINN')
    
    # 3. Formatting to remove whitespace
    ax_hist.set_yscale('log')
    ax_hist.set_ylim(0.01, 5.0) # Adjust based on your actual density peaks
    
    # Set X-ticks to just the two relevant categories
    ax_hist.set_xticks([0.25, 0.75])
    ax_hist.set_xticklabels(['Low Permeability\n($a=3$)', 'High Permeability\n($a=12$)'], 
                            fontsize=10, fontweight='bold')
    
    ax_hist.set_ylabel("Density (Log Scale)")
    ax_hist.set_title("Coefficient Distribution Analysis", fontsize=11, pad=5)
    
    # Place legend in the upper center (now plenty of room)
    ax_hist.legend(loc='upper center', ncol=3, fontsize=9, frameon=False)
    ax_hist.grid(True, axis='y', alpha=0.2)
    
    # --- BOTTOM PANEL: 3x3 Grid ---
    gs_imgs = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs[1], 
                                               hspace=0.05, wspace=0.05)
    
    rows = [maps_gt, maps_fail, maps_ours]
    row_labels = ["Ground Truth", "Joint Training", "sCM-PINN (Ours)"]
    
    for i in range(3):
        for j in range(3):
            ax = fig.add_subplot(gs_imgs[i, j])
            ax.imshow(rows[i][j], cmap='viridis')
            ax.axis('off')
            
            # Add labels rotated on the left side to save horizontal space
            if j == 0:
                ax.text(-0.15, 0.5, row_labels[i], transform=ax.transAxes, 
                        va='center', ha='center', fontsize=10, rotation=90, fontweight='bold')

    return fig

os.makedirs('./figures/adaptation', exist_ok=True)
fig_vertical = plot_ablation_vertical(gt_pixels, fail_pixels, ours_pixels,
                    a_datas[:3].numpy(), a_samples_no_sep[:3], a_samples_sep[:3])
fig_vertical.savefig('./figures/adaptation/ablation_figure_vertical.eps', dpi=300, bbox_inches='tight')
fig_vertical.savefig('./figures/adaptation/ablation_figure_vertical.png', dpi=300, bbox_inches='tight')
fig_vertical.savefig('./figures/adaptation/ablation_figure_vertical.pdf', dpi=300, bbox_inches='tight')
fig_vertical.show()
# %%

