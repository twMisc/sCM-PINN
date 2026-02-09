import os 
import torch
import click
import copy
import random
import numpy as np

from tqdm import tqdm

from utils import scale_back_u, scale_back_a, plot_result, load_data, consistency_sample_cm
from networks_util import is_compiled, create_model, EMA
from networks_util import load_model_state 

# Enable cuDNN autotuner for conv layers
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
MY_PATH_TO_TEACHER = './helmholtz-output/diffusion/model_epoch/model_epoch_10.pth'  # Path to pretrained diffusion model


def train_loop(
    model,
    optimizer,
    dataloader,
    epochs,
    device,
    save_interval=5000,
    output_path="./sCM/consistency",
    sigma_data=0.5,
    P_mean=-1.2,
    P_std=1.2,
    use_amp=True,
    use_ema=True,
    teacher_model=None,
    use_pde_loss=False
):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    def model_wrapper(scaled_x_t, t):
        pred, logvar = model(scaled_x_t, t.flatten(), return_logvar=True)
        return pred, logvar

    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)  # AMP scaler
    ema = EMA(model, decay=0.9999) if use_ema else None

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(dataloader)
    )
    step = 0
    for epoch in range(epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}", ascii=True)
        for batch in progress_bar:
            optimizer.zero_grad()
            model.train()
            a, u = batch
            batch = torch.stack([a, u], dim=1)  # Shape: (B, 2, 128, 128)
            x0 = batch.to(device)

            # Sample noise from log-normal distribution
            sigma = torch.randn(x0.shape[0], device=x0.device).reshape(-1, 1, 1, 1)
            sigma = (sigma * P_std + P_mean).exp()  # Sample from proposal distribution
            t = torch.arctan(sigma / sigma_data)  # Convert to t using arctan

            # Generate z and x_t
            z = torch.randn_like(x0) * sigma_data
            x_t = torch.cos(t) * x0 + torch.sin(t) * z
            if teacher_model is None:
                # For consistency TRAINING
                # Estimate dx_t/dt (For consistency TRAINING)
                dxt_dt = torch.cos(t) * z - torch.sin(t) * x0
            else:
                # For consistency DISTILLATION
                # (model_pretrained is assumed to output v-predictions)
                with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
                    with torch.no_grad():
                        pretrain_pred = teacher_model(x_t / sigma_data, t.flatten(), return_logvar=False)
                        dxt_dt = sigma_data * pretrain_pred

            # Next we have to calculate g and F_theta. We can do this simultaneously with torch.func.jvp
            # This doesn't match the paper because I think the paper had a typo
            v_x = torch.cos(t) * torch.sin(t) * dxt_dt / sigma_data
            v_t = torch.cos(t) * torch.sin(t)
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
                F_theta, F_theta_grad, logvar = torch.func.jvp(
                    model_wrapper, 
                    (x_t / sigma_data, t),
                    (v_x, v_t),
                    has_aux=True
                )
            logvar = logvar.view(-1, 1, 1, 1)
            F_theta_grad = F_theta_grad.detach()
            F_theta_minus = F_theta.detach()

            # warmup for first 3000 steps
            r = min(1.0, step / 3000)
            # Calculate gradient g using JVP rearrangement
            g = -torch.cos(t) * torch.cos(t) * (sigma_data * F_theta_minus - dxt_dt)
            # Note that F_theta_grad is already multiplied by sin(t) cos(t) from the tangents. Doing it early helps with stability.
            second_term = -r * (torch.cos(t) * torch.sin(t) * x_t + sigma_data * F_theta_grad)
            g = g + second_term
            
            # Tangent normalization
            g_norm = torch.linalg.vector_norm(g, dim=(1, 2, 3), keepdim=True)
            g_norm = g_norm * np.sqrt(g_norm.numel() / g.numel())  # Multiplying by sqrt(numel(g_norm) / numel(g)) ensures that the norm is invariant to the spatial dimensions.
            g = g / (g_norm + 0.1)  # 0.1 is the constant c, can be modified but 0.1 was used in the paper
            
            # Tangent clipping (Only use this OR normalization)
            # g = torch.clamp(g, min=-1, max=1)
            
            # Calculate loss with adaptive weighting
            # According to the paper, weight should be one over the dimensionality of the data
            weight = 1 / np.prod(x0.shape[1:])
            loss = (weight / torch.exp(logvar)) * torch.square(F_theta - F_theta_minus - g) + logvar
            loss = loss.mean()

            # ----------------------
            # Backward
            # ----------------------
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if step<3000:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if ema:
                ema.update(model)

            progress_bar.set_postfix({"loss": loss.item(), "grad_norm": grad_norm.item()})
            step += 1

            # ----------------------
            # Periodic sampling & saving
            # ----------------------
            if (step + 1) % save_interval == 0:
                model.eval()
                active_model = ema.ema_model if ema else model
                t0 = np.arctan(80 / 0.5)
                t1 = 1.1
                times = [t0, t1]

                pred2, inter2 = consistency_sample_cm(
                    active_model,
                    sigma_data=0.5,
                    device=device,
                    shape=(1, 2, 128, 128),
                    t_list=times,
                    return_intermediates=True,
                )
                pred = inter2[0]  # Get the intermediate result at t0
                pred[:, 0, :, :] = scale_back_a(pred[:, 0, :, :])
                pred[:, 1, :, :] = scale_back_u(pred[:, 1, :, :])
                samples_dir = os.path.join(output_path, "samples_one_step")
                os.makedirs(samples_dir, exist_ok=True)
                plot_result(pred, samples_dir, step + 1)

                pred = inter2[-1]  # Get the final result at t1
                pred[:, 0, :, :] = scale_back_a(pred[:, 0, :, :])
                pred[:, 1, :, :] = scale_back_u(pred[:, 1, :, :])
                samples_dir = os.path.join(output_path, "samples_two_step")
                os.makedirs(samples_dir, exist_ok=True)
                plot_result(pred, samples_dir, step + 1)

                checkpoint_dir = os.path.join(output_path, "checkpoints")
                os.makedirs(checkpoint_dir, exist_ok=True)
                if is_compiled(model):
                    state_dict = model._orig_mod.state_dict()
                else:
                    state_dict = model.state_dict()
                torch.save(
                    {
                        "epoch": epoch,
                        "step": step,
                        "model_state_dict": state_dict,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "loss": loss.item(),
                        "ema_state_dict": ema.ema_model.state_dict() if ema else None,
                    },
                    os.path.join(checkpoint_dir, f"checkpoint_{step+1}.pth"),
                )
        # end of epoch, also save model
        model_dir = os.path.join(output_path, 'model_epoch')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if is_compiled(model):
            state_dict = model._orig_mod.state_dict()
        else:
            state_dict = model.state_dict()
        torch.save(
        {
            "epoch": epoch,
            "step": step,
            "model_state_dict": state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": loss.item(),
            "ema_state_dict": ema.ema_model.state_dict() if ema else None,
        }, os.path.join(model_dir, f"model_epoch_{epoch+1}.pth")
        )


@click.command()
@click.option('--device', default='cuda', help='Device to use for training (e.g., "cuda" or "cpu").')
@click.option('--epochs', default=8, help='Number of training epochs.')
@click.option('--batch_size', default=2, help='Batch size for training.')
@click.option('--lr', default=1e-4, help='Learning rate for the optimizer.')
@click.option('--save_interval', default=5000, help='Interval (in steps) to save model checkpoints and samples.')
@click.option('--save_path', default='./helmholtz-output/consistency', help='Path to save model checkpoints and samples.')
@click.option('--seed', help='Optional random seed for reproducibility', default=None, type=int)
@click.option('--amp/--no-amp', default=True, help='Enable/disable AMP (bfloat16 autocast).')
@click.option('--ema/--no-ema', default=False, help='Enable/disable EMA model tracking.')
@click.option('--teacher_path', default=MY_PATH_TO_TEACHER, help='Path to the pretrained teacher model for distillation.', type=str)
@click.option('--teacher_ema/--no-teacher_ema', default=False, help='Use EMA weights for the teacher model if available.')
@click.option('--distillation/--no-distillation', default=False, help='Enable/disable consistency distillation training.')
@click.option('--propose_mean', default=-1.2, help='Mean of the proposal distribution for sigma.')
@click.option('--propose_std', default=1.2, help='Standard deviation of the proposal distribution for sigma.')
@click.option('--use_pde_loss/--no-use_pde_loss', default=False, help='Enable/disable physics-informed loss term.')
def main(device, epochs, batch_size, lr, save_interval, save_path, seed, amp, ema, teacher_path, teacher_ema, distillation, propose_mean, propose_std, use_pde_loss):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    dataloader = load_data(batch_size)
    model = create_model().to(device)
    teacher_model = None
    if teacher_path is not None:
        try:
            model = load_model_state(model, teacher_path, key="model_state_dict", device=device)
            print(f"Loaded model from {teacher_path}")
        except Exception as e:
            print(f"Failed to load model from {teacher_path}: {e}")
        if distillation:
            teacher_model = copy.deepcopy(model).to(device)
            if teacher_ema and 'ema_state_dict' in torch.load(teacher_path):
                teacher_model.load_state_dict(torch.load(teacher_path)['ema_state_dict'])
            teacher_model.eval()
            for param in teacher_model.parameters():
                param.requires_grad = False


    # Compile model if supported (PyTorch 2.0+)
    try:
        model = torch.compile(model)
        if teacher_model is not None:
            teacher_model = torch.compile(teacher_model)
    except Exception as e:
        print(f"torch.compile not available: {e}")


    optimizer = torch.optim.RAdam(model.parameters(), lr=lr)

    if teacher_model is None:
        print("Training with Consistency Training")
    else:
        print("Training with Consistency Distillation")
    train_loop(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        epochs=epochs,
        device=device,
        save_interval=save_interval,
        output_path=save_path,
        use_amp=amp,
        use_ema=ema,
        teacher_model=teacher_model,
        P_mean=propose_mean,
        P_std=propose_std,
        use_pde_loss=use_pde_loss
    )


if __name__ == '__main__':
    main()