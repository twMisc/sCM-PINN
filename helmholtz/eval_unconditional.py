"""
Evaluate unconditional sampling quality using PDE Residuals.
"""

import os
import csv
import click
import numpy as np 
import torch
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import scale_back_a, scale_back_u
from networks_util import create_model, create_sep_model, load_model_state
from diffpde_backend import load_pickle_model, sample_guided_diffusion

from utils import consistency_sample_cm, sample_dpm_solver, helmholtz_loss



def save_results(save_path, save_name, results: dict):
    file_path = os.path.join(save_path, save_name)
    file_exists = os.path.exists(file_path)

    with open(file_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)

@click.command()
@click.option('--device', default='cuda', help='Device to use for sampling.')
@click.option('--batch_size', default=16, help='Batch size for sampling.')
@click.option('--total_samples', default=1024, help='Total number of samples to generate.')
@click.option('--model_type', default='diffusion', type=click.Choice(['cm', 'diffusion', 'diffpde']), help='Type of model.')
@click.option('--network_type', default='unet', type=click.Choice(['unet', 'sep_unet']), help='Type of network architecture.')
@click.option('--model_path', required=True, help='Path to the trained model checkpoint.')
@click.option('--save_path', required=True, help='Path to save the evaluation results.')
@click.option('--save_name', default='uncond_error.csv', help='Filename to save the evaluation results.')
@click.option('--seed', help='Optional random seed', default=None, type=int)
@click.option('--num_steps', default=35, help='Number of steps for sampling (diffusion/diffpde).')
@click.option('--cm_steps', default=2, help='Number of steps for consistency sampling.')
@click.option('--figure_dir', default='./figures', help='Directory to save figures.')
@click.option('--network_name', default=None, help='Optional name of the network.')
@click.option('--rho', default=7.0, help='Rho parameter for schedule.')
def main(device, batch_size, total_samples, model_path, save_path, save_name, seed, model_type, network_type, num_steps, cm_steps, figure_dir, network_name, rho):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    # --- Load Model ---
    if model_type == 'diffpde':
        model = load_pickle_model(model_path, device)
        print(f"Loaded Pickle Model from {model_path}")
    elif network_type == 'unet':
        model = create_model()
        model = load_model_state(model, model_path)
        model.to(device)
        model.eval()
    else:
        model = create_sep_model()
        model = load_model_state(model, model_path)
        model.to(device)
        model.eval()

    if model_type != 'diffpde':
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"torch.compile not available: {e}")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print(f"Evaluating Unconditional Sampling using {model_type} model.")
    
    # Initialize lists for new metrics
    all_pde_residuals = []
    all_rel_residuals = []
    all_norm_residuals = []
    
    all_samples_a = []
    all_samples_u = []
    with torch.no_grad():
        for i in tqdm(range(0, total_samples, batch_size)):
            current_batch_size = min(batch_size, total_samples - i)
            
            # --- Sampling Logic ---
            if model_type == 'diffpde':
                # Unconditional for DiffPDE: Mask = 0
                dummy_x = torch.randn(current_batch_size, 2, 128, 128).to(device)
                mask = torch.zeros((current_batch_size, 2, 128, 128), dtype=torch.float32).to(device)
                
                with torch.enable_grad():
                    pred = sample_guided_diffusion(
                        model, 
                        x_obs=dummy_x, 
                        mask=mask, 
                        num_steps=num_steps, 
                        rho=rho,
                        zeta_obs_a=0.0, 
                        zeta_obs_u=0.0, 
                        zeta_pde=0.0,
                        device=device
                    )
                # DiffPDE output is usually directly in physical space
                a_samples_raw = pred[:, 0:1, :, :].cpu().numpy()
                u_samples_raw = pred[:, 1:2, :, :].cpu().numpy()        
                
                # Assuming DiffPDE output doesn't need scaling back if sample_guided_diffusion handles it
                a_samples = a_samples_raw
                u_samples = u_samples_raw

            elif model_type == 'cm':
                # CM Unconditional
                t0 = np.pi/2
                t1 = 1.1
                times = [t0] if cm_steps == 1 else [t0, t1]
                
                pred = consistency_sample_cm(
                    model, 
                    sigma_data=0.5, 
                    device=device, 
                    shape=(current_batch_size, 2, 128, 128), 
                    t_list=times if cm_steps <=2 else None,
                    n_steps=cm_steps if cm_steps > 2 else None,
                    schedule='power', 
                    return_intermediates=False, 
                    use_seeded_z=False, 
                    x_obs=None, 
                    mask=None, 
                    sigma_min=1e-8
                )
                
                a_samples = scale_back_a(pred[:, 0:1, :, :].cpu().numpy())
                u_samples = scale_back_u(pred[:, 1:2, :, :].cpu().numpy())

            elif model_type == 'diffusion':
                # DPM Solver Unconditional
                pred = sample_dpm_solver(
                    model, 
                    device=device, 
                    sigma_data=0.5, 
                    num_steps=num_steps, 
                    shape=(current_batch_size, 2, 128, 128), 
                    schedule='power', 
                    use_seeded_z=False, 
                    x_obs=None, 
                    mask=None
                )
                a_samples = scale_back_a(pred[:, 0:1, :, :].cpu().numpy())
                u_samples = scale_back_u(pred[:, 1:2, :, :].cpu().numpy())

            # --- CRITICAL FIX: Ensure 3D Shape (B, H, W) ---
            # scale_back functions return (B, 1, H, W) because we passed sliced input 0:1
            # We must squeeze the channel dimension before passing to helmholtz_loss
            
            a_samples_3d = a_samples.squeeze(1) # (B, 128, 128)
            u_samples_3d = u_samples.squeeze(1) # (B, 128, 128)
            
            a_tensor = torch.from_numpy(a_samples_3d).to(device)
            u_tensor = torch.from_numpy(u_samples_3d).to(device)
            
            # --- Compute PDE Residual ---
            # Now inputs are 3D, matching helmholtz_loss expectations
            residual_map = helmholtz_loss(u_tensor, a_tensor, return_residual=True)
            
            # L2 norm per sample
            batch_residuals = torch.norm(residual_map.reshape(current_batch_size, -1), p=2, dim=1)
            all_pde_residuals.extend(batch_residuals.cpu().numpy())
            batch_a_norms = torch.norm(a_tensor.reshape(current_batch_size, -1), p=2, dim=1)
            batch_rel = batch_residuals / (batch_a_norms + 1e-12)
            all_rel_residuals.extend(batch_rel.cpu().numpy())
            h = 1.0 / (u_tensor.shape[-1] - 1)
            batch_norm_res = batch_residuals * (h**2)
            all_norm_residuals.extend(batch_norm_res.cpu().numpy())

            all_samples_a.append(a_samples)
            all_samples_u.append(u_samples)

    saved_samples_a = np.concatenate(all_samples_a, axis=0)
    saved_samples_u = np.concatenate(all_samples_u, axis=0)

    # --- Statistics ---
    # 1. Raw Residuals
    all_pde_residuals = np.array(all_pde_residuals)
    mean_res = np.mean(all_pde_residuals)
    median_res = np.median(all_pde_residuals)
    std_res = np.std(all_pde_residuals)

    # 2. Relative Residuals
    all_rel_residuals = np.array(all_rel_residuals)
    mean_rel = np.mean(all_rel_residuals)
    median_rel = np.median(all_rel_residuals)
    std_rel = np.std(all_rel_residuals)

    # 3. Normalized Residuals
    all_norm_residuals = np.array(all_norm_residuals)
    mean_norm_res = np.mean(all_norm_residuals)
    median_norm_res = np.median(all_norm_residuals)
    std_norm_res = np.std(all_norm_residuals)
    print(f"\nResults for {model_type} ({network_type}):")
    print(f"Mean PDE Residual Norm: {mean_res:.6f}")
    print(f"Median PDE Residual Norm: {median_res:.6f}")
    print(f"Std PDE Residual Norm: {std_res:.6f}")

    print(f"\nResults for {model_type} ({network_type}):")
    print("-" * 30)
    print(f"Raw PDE Residual   | Mean: {mean_res:.4e} | Median: {median_res:.4e} | Std: {std_res:.4e}")
    print(f"Relative Residual  | Mean: {mean_rel:.4e} | Median: {median_rel:.4e} | Std: {std_rel:.4e}")
    print(f"Norm. Res (x h^2)  | Mean: {mean_norm_res:.4e} | Median: {median_norm_res:.4e} | Std: {std_norm_res:.4e}")
    print("-" * 30)

    results = {
        'model_type': model_type,
        'network_type': network_type,
        'network_name': network_name if network_name is not None else os.path.basename(model_path),
        'num_steps': num_steps if model_type == 'diffusion' or model_type == 'diffpde' else cm_steps,
        
        # Original Metrics
        'mean_pde_residual': mean_res,
        'median_pde_residual': median_res,
        'std_pde_residual': std_res,
        
        # New Metrics
        'mean_rel_residual': mean_rel,
        'median_rel_residual': median_rel,
        'std_rel_residual': std_rel,
        
        'mean_norm_residual': mean_norm_res,
        'median_norm_residual': median_norm_res,
        'std_norm_residual': std_norm_res
    }
    save_results(save_path, save_name, results)
    print(f"Results saved to {os.path.join(save_path, save_name)}")

    # --- Visualization ---
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    
    idx = np.random.randint(0, len(saved_samples_a))
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title(f'Generated Coeff a (Uncond)')
    plt.imshow(saved_samples_a[idx, 0], cmap='viridis')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.title(f'Generated Sol u (Uncond)')
    plt.imshow(saved_samples_u[idx, 0], cmap='viridis')
    plt.colorbar()
    
    plot_name = f"sample_uncond_{model_type}.png"
    plt.savefig(os.path.join(figure_dir, plot_name))
    plt.close()
    print(f"Sample plot saved to {os.path.join(figure_dir, plot_name)}")

    # save the samples for potential future analysis
    np.save(os.path.join(figure_dir, f"uncond_samples_a_{model_type}.npy"), saved_samples_a)
    np.save(os.path.join(figure_dir, f"uncond_samples_u_{model_type}.npy"), saved_samples_u)
    print(f"Sample arrays saved to {figure_dir}")

if __name__ == '__main__':
    main()