"""
Evaluate the forward/inverse/reconstruction problems on the generated samples given the observed data.
We will evaluate the MSE and PDE Error following the distillation paper.
"""

import os
import csv
import click
import numpy as np 
import torch
import random
import matplotlib.pyplot as plt
from utils import load_test_data, rescale_a, sample_dpm_solver, consistency_sample_cm, scale_back_a, scale_back_u, rescale_u, rescale_a, FDM_Darcy_loss, calculate_h1_error, discrete_a
from networks_util import create_model, create_sep_model, load_model_state
from diffpde_backend import load_pickle_model, sample_guided_diffusion

def create_random_mask(mask_percent=0.75, shape=(1, 2, 128, 128), verbose=False):
    """ Create a random mask with the given percentage of masked pixels. 
    Each channel will have independent masks.
    """

    assert len(shape) == 4, "Shape must be (batch_size, channels, height, width)"
    batch_size, channels, height, width = shape
    num_pixels = height * width
    num_masked = int(mask_percent * num_pixels)

    mask = np.ones(shape, dtype=np.float32)
    for b in range(batch_size):
        for c in range(channels):
            masked_indices = np.random.choice(num_pixels, num_masked, replace=False)
            mask[b, c].flat[masked_indices] = 0.0
    if verbose:
        print(f"Created random mask with {mask_percent*100}% masked pixels.")
        print(f"Mask shape: {mask.shape}, Number of masked pixels per channel: {num_masked}, Number of observed pixels per channel: {num_pixels - num_masked}")
    return torch.tensor(mask)

def test_random_mask(mask_percent=0.75):
    mask = create_random_mask(mask_percent=mask_percent, shape=(1, 2, 128, 128))
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(mask[0, 0, :, :], cmap='gray')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(mask[0, 1, :, :], cmap='gray')
    plt.colorbar()
    plt.savefig('random_mask.png')
    plt.close()
    
def generate_mask_for_batch(problem_type, batch_size, device, recon_mask_percent=0.75):
    """ Generate mask for the batch based on the problem type."""
    # create random mask if needed
    if problem_type == 'reconstruction':
        mask = create_random_mask(mask_percent=recon_mask_percent, shape=(batch_size, 2, 128, 128), verbose=True).to(device)
        # print(f"Using random mask with {recon_mask_percent*100}% masked pixels for reconstruction problem.")
    elif problem_type == 'forward':
        mask = torch.ones((batch_size, 2, 128, 128), dtype=torch.float32).to(device)  # no masking
        # mask for a channel
        mask[:, 0, :, :] = 1.0
        mask[:, 1, :, :] = 0.0
        # print("Using forward problem mask (observe 'a' channel only).")
    elif problem_type == 'inverse':
        mask = torch.ones((batch_size, 2, 128, 128), dtype=torch.float32).to(device)  # no masking
        # mask for u channel
        mask[:, 0, :, :] = 0.0
        mask[:, 1, :, :] = 1.0
        # print("Using inverse problem mask (observe 'u' channel only).")
    else:
        raise ValueError(f"Unknown problem_type: {problem_type}")
    return mask

def save_results(save_path, save_name, results: dict):
    """
    Save evaluation results to a CSV file. If file exists, append as a new row.

    Args:
        save_path (str): Directory to save results
        save_name (str): Filename for results (e.g. "sampling_error.csv")
        results (dict): Dictionary of results {metric_name: value}
    """
    file_path = os.path.join(save_path, save_name)
    file_exists = os.path.exists(file_path)

    with open(file_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results.keys()))

        if not file_exists:
            writer.writeheader()  # write header only once

        writer.writerow(results)

@click.command()
@click.option('--device', default='cuda', help='Device to use for sampling (e.g., "cuda" or "cpu").')
@click.option('--batch_size', default=16, help='Batch size for sampling.')
@click.option('--total_samples', default=1024, help='Total number of samples to generate for evaluation.')
@click.option('--model_type', default='diffusion', type=click.Choice(['cm', 'diffusion', 'diffpde']), help='Type of model: cm, diffusion, or diffpde.')
@click.option('--network_type', default='unet', type=click.Choice(['unet', 'sep_unet']), help='Type of network architecture.')
@click.option('--model_path', required=True, help='Path to the trained model checkpoint.')
@click.option('--save_path', required=True, help='Path to save the evaluation results.')
@click.option('--save_name', default='task_error.csv', help='Filename to save the evaluation results.')
@click.option('--seed', help='Optional random seed for reproducibility', default=None, type=int)
@click.option('--num_steps', default=35, help='Number of steps for sampling (only for diffusion models).')
@click.option('--cm_steps', default=2, help='Number of steps for consistency model sampling (only for CM models).')
@click.option('--use_pde_mask', is_flag=True, help='Whether to use PDE mask for pde error calculation.')
@click.option('--problem_type', default='forward', type=click.Choice(['forward', 'inverse', 'reconstruction']), help='Type of problem to evaluate.')
@click.option('--recon_mask_percent', default=0.75, help='Percentage of pixels to mask for reconstruction problem (only if problem_type is "reconstruction").')
@click.option('--figure_dir', default='./figures', help='Directory to save figures.')
@click.option('--figure_name', default='task_sample.png', help='Filename to save the figure.')
@click.option('--network_name', default=None, help='Optional name of the network (for logging purposes).')
@click.option('--mask_mse_eval', is_flag=True, help='Whether to evaluate MSE only on the non-obeserved pixels for reconstruction problem.')
@click.option('--zeta_obs_a', default=0.8, help='Guidance weight for observation a.')
@click.option('--zeta_obs_u', default=0.0, help='Guidance weight for observation u.')
@click.option('--zeta_pde', default=1.0, help='Guidance weight for PDE residual.')
@click.option('--rho', default=7.0, help='Rho parameter for schedule.')
def main(device, batch_size, total_samples, model_path, save_path, save_name, seed, model_type, network_type, num_steps, cm_steps, use_pde_mask, problem_type, recon_mask_percent, figure_dir, figure_name, network_name, mask_mse_eval, zeta_obs_a, zeta_obs_u, zeta_pde, rho):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    dataset, dataloader = load_test_data(batch_size, return_dataset=True, rescale=False)

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

    # get real samples from dataset
    inds = np.random.choice(len(dataset), total_samples, replace=False)
    real_data = dataset[inds]
    a_data = real_data[0].numpy()
    u_data = real_data[1].numpy()
    print(f"Real data shapes: a: {a_data.shape}, u: {u_data.shape}")

    print(f"Evaluating {problem_type} problem using {model_type} model with {network_type} network.")
    # get generated samples
    gen_a_samples = []
    gen_u_samples = []
    all_masks = []
    with torch.no_grad():
        for i in range(0, total_samples, batch_size):
            # generate mask for the batch
            current_batch_size = min(batch_size, total_samples - i)
            mask = generate_mask_for_batch(problem_type, current_batch_size, device, recon_mask_percent)
            all_masks.append(mask.cpu().numpy())
            a_batch = a_data[i:i+current_batch_size]
            u_batch = u_data[i:i+current_batch_size]
            a_batch = torch.tensor(a_batch).to(device)
            u_batch = torch.tensor(u_batch).to(device)
            x_obs = torch.stack([rescale_a(a_batch), rescale_u(u_batch)], dim=1).to(torch.float32)  # (B, 2, 128, 128)
            if model_type == 'diffpde':
                x_obs_raw = torch.stack([a_batch, u_batch], dim=1)
                assert x_obs_raw.shape == (current_batch_size, 2, 128, 128)
                with torch.enable_grad():
                    pred = sample_guided_diffusion(
                        model, 
                        x_obs=x_obs_raw, 
                        mask=mask, 
                        num_steps=num_steps, 
                        rho=rho,
                        zeta_obs_a=zeta_obs_a, 
                        zeta_obs_u=zeta_obs_u, 
                        zeta_pde=zeta_pde,
                        device=device
                    )
                    assert pred.shape == (current_batch_size, 2, 128, 128)
                
                # The guided sampler returns Physics space directly. 
                # We assume the rest of the script expects raw values for 'gen_a_samples' appending
                # We skip the scale_back functions below for this type
                a_samples = pred[:, 0:1, :, :].cpu().numpy()
                u_samples = pred[:, 1:2, :, :].cpu().numpy()        

            elif model_type == 'cm':
                # for cm, we use a pre-determined time step for 1 or 2 steps
                t0 = np.arctan(80 / 0.5)
                t1 = 1.1
                if cm_steps == 1:
                    times = [t0]
                elif cm_steps == 2:
                    times = [t0, t1]

                if cm_steps > 2:
                    pred = consistency_sample_cm(model, sigma_data=0.5, device=device, shape=(batch_size, 2, 128, 128), schedule='power', return_intermediates=False, use_seeded_z=False, x_obs=x_obs, mask=mask, n_steps=cm_steps)
                else:
                    pred = consistency_sample_cm(model, sigma_data=0.5, device=device, shape=(batch_size, 2, 128, 128), t_list=times, return_intermediates=False, use_seeded_z=False, x_obs=x_obs, mask=mask)
            elif model_type == 'diffusion':
                pred = sample_dpm_solver(model, device=device, sigma_data=0.5, num_steps=num_steps, shape=(batch_size, 2, 128, 128), schedule='power', use_seeded_z=False, x_obs=x_obs, mask=mask)

            if model_type != 'diffpde':
                # Original scaling back logic   
                a_samples = scale_back_a(pred[:, 0:1, :, :].cpu().numpy())
                a_samples = discrete_a(a_samples)  # discretize 'a' after scaling back
                u_samples = scale_back_u(pred[:, 1:2, :, :].cpu().numpy())      
                
            # print(f"DEBUG: Batch Size: {current_batch_size}, Pred Shape: {pred.shape}")
            gen_a_samples.append(a_samples)
            gen_u_samples.append(u_samples)
            print(f"Generated {i + a_samples.shape[0]} / {total_samples} samples", end='\r')

    # [Pre-processing] Discretize 'a' for Error Rate calculation
    # Since 'a' is discrete (3 or 12), we snap the continuous output to the nearest valid value.
    # The midpoint is 7.5.
    # gen_a_discrete = np.where(gen_a_samples > 7.5, 12.0, 3.0)

    # calculate metrics
    gen_a_samples = np.concatenate(gen_a_samples, axis=0)[:total_samples].reshape(-1, a_data.shape[1], a_data.shape[2])
    gen_u_samples = np.concatenate(gen_u_samples, axis=0)[:total_samples].reshape(-1, u_data.shape[1], u_data.shape[2])
    print(f"\nGenerated samples shapes: a: {gen_a_samples.shape}, u: {gen_u_samples.shape}")
    if problem_type == 'reconstruction' and mask_mse_eval:
        all_masks = np.concatenate(all_masks, axis=0)[:total_samples].reshape(-1, 2, 128, 128)
        a_mask = all_masks[:, 0:1, :, :]
        u_mask = all_masks[:, 1:2, :, :]
        # reverse mask: only evaluate on the unobserved pixels
        a_mask = 1.0 - a_mask
        u_mask = 1.0 - u_mask
        
        # --- 1. Error Rate for a (Masked) ---
        # Calculate percentage of mismatched pixels in the unobserved region
        mismatches = (gen_a_samples != a_data)
        # Apply mask: metrics only count if pixel is unobserved (mask = 1)
        err_a = np.sum(mismatches * a_mask) / np.sum(a_mask)

        # --- 2. Relative L2 for u (Masked) ---
        # Relative L2 = ||(u_gen - u_real) * mask|| / ||u_real * mask||
        diff_norm_sq = np.sum(((gen_u_samples - u_data) * u_mask) ** 2)
        true_norm_sq = np.sum((u_data * u_mask) ** 2)
        err_u = np.sqrt(diff_norm_sq / (true_norm_sq + 1e-12))

        print(f"Mask Error Rate a: {err_a:.4f}, Mask Rel L2 u: {err_u:.4f}")
    else:
        # --- 1. Relative L2 for a (Global) ---
        # We calculate the relative error per sample, then average
        flatten_gen = gen_a_samples.reshape(total_samples, -1)

        # for this example we calculate error rate instead of rel L2 for 'a'
        mismatches = (gen_a_samples != a_data)
        err_a = np.sum(mismatches) / (total_samples * a_data.shape[1] * a_data.shape[2])

        # --- 2. Relative L2 for u (Global) ---
        # We calculate the relative error per sample, then average
        flatten_gen = gen_u_samples.reshape(total_samples, -1)
        flatten_real = u_data.reshape(total_samples, -1)
        
        diff_norms = np.linalg.norm(flatten_gen - flatten_real, axis=1)
        true_norms = np.linalg.norm(flatten_real, axis=1)
        
        # Add epsilon to avoid division by zero
        err_u = np.mean(diff_norms / (true_norms + 1e-12))
        
        print(f"Error rate a: {err_a:.4f}, Rel L2 u: {err_u:.4f}")

        # --- NEW: H1 Error Calculation ---
        h1_errors = []
        rel_h1_errors = []
        h1_norms = []
        
        # Iterate to calculate H1 for each sample (H1 is sample-wise)
        for j in range(gen_u_samples.shape[0]):
            # h=1/127 based on 128 grid size
            h1, rel_h1 = calculate_h1_error(gen_u_samples[j], u_data[j], h=1.0/127)
            h1_norm, _ = calculate_h1_error(gen_u_samples[j], np.zeros_like(gen_u_samples[j]), h=1.0/127)
            h1_norms.append(h1_norm)
            h1_errors.append(h1)
            rel_h1_errors.append(rel_h1)
        
        mean_h1 = np.mean(h1_errors)
        mean_rel_h1 = np.mean(rel_h1_errors)
        mean_h1_norm = np.mean(h1_norms)
        print(f"Mean H1 Error: {mean_h1:.4f}, Mean Rel H1 Error: {mean_rel_h1:.4f}, Mean H1 Norm: {mean_h1_norm:.4f}")

    # ... plot one generated sample ...
    # plot one generated sample vs real sample
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    sample_idx = np.random.randint(0, total_samples)
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    plt.title('Real a')
    plt.imshow(a_data[sample_idx, :, :], cmap='viridis')
    plt.colorbar()
    plt.subplot(2, 2, 2)
    plt.title('Generated a')
    plt.imshow(gen_a_samples[sample_idx, :, :], cmap='viridis')
    plt.colorbar()
    plt.subplot(2, 2, 3)
    plt.title('Real u')
    plt.imshow(u_data[sample_idx, :, :], cmap='viridis')
    plt.colorbar()
    plt.subplot(2, 2, 4)
    plt.title('Generated u')
    plt.imshow(gen_u_samples[sample_idx, :, :], cmap='viridis')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, figure_name))
    plt.close()
    
    # compute PDE residual error
    if use_pde_mask:
        pde_residual = FDM_Darcy_loss(gen_u_samples, gen_a_samples, use_mask=True, output_mask=False)
    else:
        pde_residual = FDM_Darcy_loss(gen_u_samples, gen_a_samples, use_mask=False, output_mask=False)
    pde_error = np.mean(pde_residual**2)
    print(f"PDE Residual MSE: {pde_error}")

    # calculate energy loss
    energy_loss = Energy_Darcy_loss(gen_u_samples, gen_a_samples)
    print(f"Energy Loss: {energy_loss}")

    # Save results
    results = {
        'model_type': model_type,
        'network_type': network_type,
        'network_name': network_name if network_name is not None else os.path.basename(model_path),
        'num_steps': num_steps if model_type == 'diffusion' or model_type == 'diffpde' else cm_steps,
        'problem_type': problem_type,
        'error_rate_a': err_a, 
        'rel_l2_u': err_u,    
        'pde_residual_mse': pde_error,
        'energy_loss': energy_loss,
        'mean_h1_error': mean_h1,
        'mean_rel_h1_error': mean_rel_h1,
        'mean_h1_norm': mean_h1_norm
    }
    save_results(save_path, save_name, results)
    print(f"Results saved to {os.path.join(save_path, save_name)}")

    # save the generated samples and real samples for future analysis
    np.savez(os.path.join(figure_dir, f"generated_samples_{model_type}_{network_type}_{problem_type}.npz"), gen_a=gen_a_samples, gen_u=gen_u_samples, real_a=a_data, real_u=u_data)
    print(f"Generated and real samples saved to {os.path.join(figure_dir, f'generated_samples_{model_type}_{network_type}_{problem_type}.npz')}")

if __name__ == '__main__':
    main()