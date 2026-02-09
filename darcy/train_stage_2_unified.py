import os
import sys
from typing import Optional
import yaml
import click
import torch
import torch.nn.functional as F
import numpy as np
import random

from tqdm import tqdm

from utils import (
    scale_back_u,
    scale_back_a,
    plot_result,
    load_data,
    consistency_sample_cm,
    discrete_a,
    discrete_a_ste,
    FDM_Darcy_loss,
    Energy_Darcy_loss,
    DarcyEnergyLossFixedA,
)
from networks_util import (
    is_compiled,
    create_sep_model,
    create_model,
    EMA,
    load_model_state,
    LossWeightManager,
)

# Enable cuDNN autotuner for conv layers
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("highest")

MY_PATH_TO_STAGE_1 = "./darcy_redo_output/sCM/consistency/model_epoch/model_epoch_8.pth"


# ---------------------------------------------------------------------
# Helper: physics loss backend (FDM vs Ritz), with optional STE for a
# ---------------------------------------------------------------------
def compute_physics_loss(
    pred_x0: torch.Tensor,
    physics_type: str,
    use_a_ste: bool,
    mask_fdm: bool = True,
):
    """
    pred_x0: (B, 2, H, W), in *scaled* space.
    physics_type: "fdm" or "ritz"
    """
    # Separate channels and scale back
    u_samples = scale_back_u(pred_x0[:, 1, :, :])
    a_samples = scale_back_a(pred_x0[:, 0, :, :])

    if use_a_ste:
        a_samples_discrete = discrete_a_ste(a_samples)
    else:
        a_samples_discrete = discrete_a(a_samples)

    if physics_type == "fdm":
        # Residual-based loss; we square and average later
        res = FDM_Darcy_loss(u_samples, a_samples_discrete, output_mask=False, use_mask=mask_fdm)
        loss = torch.square(res)
    elif physics_type == "ritz":
        # Energy_Darcy_loss is assumed to already average over batch;
        # but we still treat it as a tensor and allow .mean() for uniformity.
        loss = Energy_Darcy_loss(u_samples, a_samples_discrete)
    elif physics_type == 'ritz_v2':
        loss_fn = DarcyEnergyLossFixedA()
        loss = loss_fn(u_samples, a_samples_discrete)
    elif physics_type == "fdm+ritz_v2":
        res = FDM_Darcy_loss(u_samples, a_samples_discrete, output_mask=False, use_mask=mask_fdm)
        loss_fdm = torch.square(res).mean()
        loss_ritz = DarcyEnergyLossFixedA()(u_samples, a_samples_discrete)
        loss = loss_fdm + loss_ritz
    else:
        raise ValueError(f"Unknown physics_type: {physics_type}")

    return loss

def get_weight_value(weight, learnable: bool):
    if learnable:
        return torch.sigmoid(weight)
    else:
        return weight   # use raw constant directly


# ---------------------------------------------------------------------
# Unified train loop
# ---------------------------------------------------------------------
def train_loop(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader,
    epochs: int,
    device: str,
    save_interval: int,
    output_path: str,
    sigma_data: float = 0.5,
    P_mean: float = -1.2,
    P_std: float = 1.2,
    use_amp: bool = True,
    use_ema: bool = True,
    teacher_model: Optional[torch.nn.Module] = None,
    loss_weights: Optional[LossWeightManager] = None,
    loss_weight_optimizer: Optional[torch.optim.Optimizer] = None,
    use_sep_net: bool = True,
    use_ct: bool = True,
    use_phy_rand: bool = True,
    use_phy_1step: bool = True,
    use_phy_2step: bool = True,
    physics_type: str = "fdm",
    use_a_ste: bool = False,
    use_ct_t1: bool = False,
    mask_fdm: bool = True,
    t1_strategy: str = "fixed",  # options: "fixed", "uniform", "normal"
    t1_mean: float = 1.1,        # The center point (previously hardcoded 1.1)
    t1_width: float = 0.2,       # Half-width for uniform or std-dev for normal
):
    """
    Unified stage-2 trainer supporting:
    - sep vs non-sep network
    - CT loss (optionally only u-channel for sep net)
    - physics loss at random time / 1-step / 2-step
    - physics backend: FDM residual vs Ritz energy
    - learnable balancing coefficients via LossWeightManager
    """
    if os.path.exists(output_path):
            user_input = input(f"[WARN]: Output path '{output_path}' already exists.\nDo you want to overwrite/append? [y/N]: ").strip().lower()
            if user_input not in ['y', 'yes']:
                print("[INFO] Aborting training to prevent overwriting.")
                sys.exit()
    else:
        os.makedirs(output_path)
        print(f"[INFO] Created output directory at '{output_path}'.")

    # Wrapper for torch.func.jvp
    def model_wrapper(scaled_x_t, t):
        pred, logvar = model(scaled_x_t, t.flatten(), return_logvar=True)
        return pred, logvar

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = EMA(model, decay=0.9999) if use_ema else None

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(dataloader)
    )

    # Loss weights (if any)
    if loss_weights is not None:
        w_loss_ct = loss_weights["ct"]
        w_loss_rand = loss_weights["rand"]
        w_loss_one = loss_weights["one_step"]
        w_loss_two = loss_weights["two_step"]
        w_loss_t1_ct = loss_weights['t1_ct']
    else:
        # default scalar 1.0 (no balancing)
        w_loss_ct = torch.tensor(1.0, device=device)
        w_loss_rand = torch.tensor(1.0, device=device)
        w_loss_one = torch.tensor(1.0, device=device)
        w_loss_two = torch.tensor(1.0, device=device)
        w_loss_t1_ct = torch.tensor(1.0, device=device)

    global_step = 0
    state_dict = None  # will be filled after first step

    for epoch in range(epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}", ascii=True)

        for batch in progress_bar:
            model.train()
            a, u = batch
            x0 = torch.stack([a, u], dim=1).to(device)  # (B, 2, H, W)

            # --------------------------------------------------
            # Sample sigma, t, z and x_t (TrigFlow parameterization)
            # --------------------------------------------------
            sigma = torch.randn(x0.shape[0], device=x0.device).reshape(-1, 1, 1, 1)
            sigma = (sigma * P_std + P_mean).exp()  # log-normal
            t = torch.arctan(sigma / sigma_data)

            z = torch.randn_like(x0) * sigma_data
            x_t = torch.cos(t) * x0 + torch.sin(t) * z

            # --------------------------------------------------
            # dxt/dt (training vs distillation)
            # --------------------------------------------------
            if teacher_model is None:
                # Consistency training
                dxt_dt = torch.cos(t) * z - torch.sin(t) * x0
            else:
                # Consistency distillation (teacher outputs v-pred)
                with torch.no_grad():
                    pretrain_pred = teacher_model(
                        x_t / sigma_data, t.flatten(), return_logvar=False
                    )
                    dxt_dt = sigma_data * pretrain_pred

            # --------------------------------------------------
            # JVP to get F_theta and its directional derivative
            # --------------------------------------------------
            v_x = torch.cos(t) * torch.sin(t) * dxt_dt / sigma_data
            v_t = torch.cos(t) * torch.sin(t)

            F_theta, F_theta_grad, logvar = torch.func.jvp(
                model_wrapper,
                (x_t / sigma_data, t),
                (v_x, v_t),
                has_aux=True,
            )

            logvar = logvar.view(-1, 1, 1, 1)
            F_theta_grad = F_theta_grad.detach()
            F_theta_minus = F_theta.detach()

            # --------------------------------------------------
            # Construct tangent g 
            # --------------------------------------------------
            r = 1.0  # no warmup for now, but easy to add

            g = -torch.cos(t) * torch.cos(t) * (sigma_data * F_theta_minus - dxt_dt)
            second_term = -r * (torch.cos(t) * torch.sin(t) * x_t + sigma_data * F_theta_grad)
            g = g + second_term

            # Tangent normalization 
            g_norm = torch.linalg.vector_norm(g, dim=(1, 2, 3), keepdim=True)
            g_norm = g_norm * np.sqrt(g_norm.numel() / g.numel())
            g = g / (g_norm + 0.1)

            # --------------------------------------------------
            # CT Loss
            # sep net: only use u-channel; base net: both channels
            # --------------------------------------------------
            loss_ct = None
            if use_ct:
                if use_sep_net:
                    # Use only u-channel (index 1)
                    loss_ct = torch.square(
                        F_theta[:, 1, :, :] - F_theta_minus[:, 1, :, :] - g[:, 1, :, :]
                    )
                else:
                    # Use all channels
                    loss_ct = torch.square(F_theta - F_theta_minus - g)

            # --------------------------------------------------
            # Physics loss @ random t (from current F_theta prediction)
            # --------------------------------------------------
            loss_phy_rand = None
            pred_x0_rand = torch.cos(t) * x_t - torch.sin(t) * sigma_data * F_theta
            if use_phy_rand:
                loss_phy_rand = compute_physics_loss(
                    pred_x0_rand, physics_type=physics_type, use_a_ste=use_a_ste, mask_fdm=mask_fdm
                )

            # --------------------------------------------------
            # 1. Generate t1 (Shared for Phy-2step and CT-t1)
            # --------------------------------------------------
            # We generate t1 if either 2-step physics or t1-CT is enabled
            t1 = None
            t1_exp = None
            
            if use_phy_2step or use_ct_t1:
                B = x0.shape[0]
                
                if t1_strategy == "fixed":
                    t1 = t1_mean * torch.ones(B, device=device)
                    
                elif t1_strategy == "uniform":
                    # Uniformly sample in [mean - width, mean + width]
                    # Clamped to avoid 0 or > pi/2 (pure noise)
                    low = max(1e-3, t1_mean - t1_width)
                    high = min(np.pi / 2.0 - 1e-3, t1_mean + t1_width)
                    t1 = torch.rand(B, device=device) * (high - low) + low
                    
                elif t1_strategy == "normal":
                    # Gaussian sampling around mean
                    t1 = torch.randn(B, device=device) * t1_width + t1_mean
                    # Clamp strictly to valid trigonometric range
                    t1 = torch.clamp(t1, min=1e-3, max=np.pi / 2.0 - 1e-3)
                
                else:
                    raise ValueError(f"Unknown t1_strategy: {t1_strategy}")

                t1_exp = t1.view(-1, 1, 1, 1)

            # --------------------------------------------------
            # 2. Physics loss via 1-step and 2-step sampling
            # --------------------------------------------------
            loss_phy_1step = None
            loss_phy_2step = None

            if use_phy_1step or use_phy_2step:
                # 1-step sampling (from pure noise t0 = pi/2)
                z0 = torch.randn_like(x0)
                t0 = (np.pi / 2.0) * torch.ones(x0.shape[0], device=device)
                model_out_t0 = model(z0, t0, return_logvar=False)
                pred_x0_1step = -sigma_data * model_out_t0
                pred_x0_1step = pred_x0_1step.view_as(x0)

                if use_phy_1step:
                    loss_phy_1step = compute_physics_loss(
                        pred_x0_1step, physics_type=physics_type, use_a_ste=use_a_ste, mask_fdm=mask_fdm
                    )

                # 2-step sampling (using the t1 generated above)
                if use_phy_2step:
                    # We assume t1 was generated above because check (use_phy_2step or use_ct_t1) covers it.
                    z1 = torch.randn_like(x0)
                    
                    # Add noise back to reach state x_t1
                    x_t1 = (
                        torch.sin(t1_exp) * z1 * sigma_data
                        + torch.cos(t1_exp) * pred_x0_1step
                    )

                    pred_x0_2step = torch.cos(t1_exp) * x_t1 - torch.sin(t1_exp) * sigma_data * model(
                        x_t1 / sigma_data, t1, return_logvar=False
                    )
                    pred_x0_2step = pred_x0_2step.view_as(x0)

                    loss_phy_2step = compute_physics_loss(
                        pred_x0_2step, physics_type=physics_type, use_a_ste=use_a_ste, mask_fdm=mask_fdm
                    )
            # --------------------------------------------------
            # Optional CT loss at t1 = 1.1 (same as 2-step time)
            # --------------------------------------------------
            loss_ct_t1 = None
            if use_ct_t1:
                # We need F_theta_t1, F_theta_minus_t1, and g_t1 at this t1
                # so we recompute them here.

                # Build x_t1 again (already computed above!)
                # Now compute dxt_dt_t1 for CT construction
                z1_for_ct = torch.randn_like(x0)  # noise for consistency relation
                x_t1_for_ct = torch.cos(t1_exp) * x0 + torch.sin(t1_exp) * z1_for_ct

                if teacher_model is None:
                    dxt_dt_t1 = torch.cos(t1_exp) * z1_for_ct - torch.sin(t1_exp) * x0
                else:
                    with torch.no_grad():
                        teacher_out_t1 = teacher_model(x_t1_for_ct/sigma_data, t1.flatten(), return_logvar=False)
                        dxt_dt_t1 = sigma_data * teacher_out_t1

                # Compute JVP at t1
                v_x_t1 = torch.cos(t1_exp) * torch.sin(t1_exp) * dxt_dt_t1 / sigma_data
                v_t_t1 = torch.cos(t1_exp) * torch.sin(t1_exp)

                F_theta_t1, F_theta_grad_t1, logvar_t1 = torch.func.jvp(
                    model_wrapper,
                    (x_t1_for_ct / sigma_data, t1_exp),
                    (v_x_t1, v_t_t1),
                    has_aux=True,
                )
                F_theta_minus_t1 = F_theta_t1.detach()
                F_theta_grad_t1 = F_theta_grad_t1.detach()

                # Build g_t1
                g_t1 = (
                    -torch.cos(t1_exp) * torch.cos(t1_exp) * (sigma_data * F_theta_minus_t1 - dxt_dt_t1)
                    - torch.cos(t1_exp) * torch.sin(t1_exp) * x_t1_for_ct
                    - sigma_data * F_theta_grad_t1
                )

                # sep-net vs base-net logic
                if use_sep_net:
                    loss_ct_t1 = torch.square(
                        F_theta_t1[:, 1, :, :] - F_theta_minus_t1[:, 1, :, :] - g_t1[:, 1, :, :]
                    )
                else:
                    loss_ct_t1 = torch.square(F_theta_t1 - F_theta_minus_t1 - g_t1)


            # --------------------------------------------------
            # Combine losses with learnable balancing coefficients
            # (sigmoid -> (0, 1) scaling)
            # --------------------------------------------------

            # second arg = whether weights are learnable
            w_ct   = get_weight_value(w_loss_ct,   learnable=(loss_weights is not None))
            w_rand = get_weight_value(w_loss_rand, learnable=(loss_weights is not None))
            w_one  = get_weight_value(w_loss_one,  learnable=(loss_weights is not None))
            w_two  = get_weight_value(w_loss_two,  learnable=(loss_weights is not None))
            w_t1_ct = get_weight_value(w_loss_t1_ct, learnable=(loss_weights is not None))
            
            total_loss = 0.0

            if use_ct and loss_ct is not None:
                total_loss = total_loss + w_ct * loss_ct.mean()

            if use_phy_rand and loss_phy_rand is not None:
                total_loss = total_loss + w_rand * loss_phy_rand.mean()

            if use_phy_1step and loss_phy_1step is not None:
                total_loss = total_loss + w_one * loss_phy_1step.mean()

            if use_phy_2step and loss_phy_2step is not None:
                total_loss = total_loss + w_two * loss_phy_2step.mean()

            if use_ct_t1 and loss_ct_t1 is not None:
                total_loss = total_loss + w_t1_ct * loss_ct_t1.mean()


            # --------------------------------------------------
            # Backward pass
            # --------------------------------------------------
            optimizer.zero_grad(set_to_none=True)
            if loss_weight_optimizer is not None:
                loss_weight_optimizer.zero_grad(set_to_none=True)

            scaler.scale(total_loss).backward()

            # Unscale grads
            scaler.unscale_(optimizer)
            if loss_weight_optimizer is not None:
                scaler.unscale_(loss_weight_optimizer)

            # Flip grads for loss weights (gradient ascent on weights)
            if loss_weights is not None:
                for p in loss_weights.parameters():
                    if p.grad is not None:
                        p.grad.mul_(-1)

            # Clip only model parameters
            clip_val = 10.0 if global_step < 3000 else 1.0
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=clip_val
            )

            # Update model
            scaler.step(optimizer)

            # Update loss weights (if separate optimizer)
            if loss_weight_optimizer is not None:
                scaler.step(loss_weight_optimizer)

            scaler.update()
            scheduler.step()

            if ema:
                ema.update(model)

            progress_bar.set_postfix(
                {
                    "loss": float(total_loss.item()),
                    "grad_norm": float(grad_norm.item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
                }
            )
            global_step += 1

            # --------------------------------------------------
            # Periodic sampling & saving
            # --------------------------------------------------
            if (global_step + 1) % save_interval == 0:
                model.eval()
                active_model = ema.ema_model if ema else model
                t0 = np.pi / 2.0
                t1 = 1.1
                times = [t0, t1]

                with torch.no_grad():
                    pred2, inter2 = consistency_sample_cm(
                        active_model,
                        sigma_data=sigma_data,
                        device=device,
                        shape=(1, 2, x0.shape[-2], x0.shape[-1]),
                        t_list=times,
                        return_intermediates=True,
                        use_trigflow_t0=False,
                    )

                    # one-step sample (at t0)
                    pred = inter2[0]
                    pred[:, 0, :, :] = scale_back_a(pred[:, 0, :, :])
                    pred[:, 0, :, :] = discrete_a(pred[:, 0, :, :])
                    pred[:, 1, :, :] = scale_back_u(pred[:, 1, :, :])

                    samples_dir = os.path.join(output_path, "samples_one_step")
                    os.makedirs(samples_dir, exist_ok=True)
                    plot_result(pred, samples_dir, global_step + 1)

                    # two-step sample (final)
                    pred = inter2[-1]
                    pred[:, 0, :, :] = scale_back_a(pred[:, 0, :, :])
                    pred[:, 0, :, :] = discrete_a(pred[:, 0, :, :])
                    pred[:, 1, :, :] = scale_back_u(pred[:, 1, :, :])

                    samples_dir = os.path.join(output_path, "samples_two_step")
                    os.makedirs(samples_dir, exist_ok=True)
                    plot_result(pred, samples_dir, global_step + 1)

                    checkpoint_dir = os.path.join(output_path, "checkpoints")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    if is_compiled(model):
                        state_dict = model._orig_mod.state_dict()
                    else:
                        state_dict = model.state_dict()

                    torch.save(
                        {
                            "epoch": epoch,
                            "step": global_step,
                            "model_state_dict": state_dict,
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scaler_state_dict": scaler.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "loss": float(total_loss.item()),
                            "ema_state_dict": ema.ema_model.state_dict() if ema else None,
                        },
                        os.path.join(checkpoint_dir, f"checkpoint_{global_step+1}.pth"),
                    )

        # end of epoch: also save model_epoch
        model_dir = os.path.join(output_path, "model_epoch")
        os.makedirs(model_dir, exist_ok=True)

        if state_dict is None:
            # first epoch, no periodic checkpoint yet
            if is_compiled(model):
                state_dict = model._orig_mod.state_dict()
            else:
                state_dict = model.state_dict()

        torch.save(
            {
                "epoch": epoch,
                "step": global_step,
                "model_state_dict": state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": float(total_loss.item()),
                "ema_state_dict": ema.ema_model.state_dict() if ema else None,
            },
            os.path.join(model_dir, f"model_epoch_{epoch+1}.pth"),
        )


# ---------------------------------------------------------------------
# Unified CLI
# ---------------------------------------------------------------------
@click.command()
@click.option("--device", default="cuda", help='Device, e.g. "cuda" or "cpu".')
@click.option("--epochs", default=1, type=int)
@click.option("--batch_size", default=2, type=int)
@click.option("--lr", default=1e-4, type=float)
@click.option(
    "--save_interval",
    default=2500,
    type=int,
    help="Steps between checkpoint/sample saves.",
)
@click.option(
    "--save_path",
    default="./darcy_redo_output/sCM/consistency-unified",
    help="Output directory.",
)
@click.option("--seed", default=None, type=int, help="Random seed.")
@click.option("--amp/--no-amp", default=False, help="Enable/disable AMP.")
@click.option("--ema/--no-ema", default=False, help="Enable/disable EMA.")
@click.option(
    "--use_sep_net/--no-use_sep_net",
    default=True,
    help="Use sep network (and u-channel-only CT loss).",
)
@click.option(
    "--pretrained_path",
    default=MY_PATH_TO_STAGE_1,
    type=str,
    help="Path to stage-1 pretrained model for student.",
)
@click.option(
    "--distillation/--no-distillation",
    default=False,
    help="Use teacher model for consistency distillation.",
)
@click.option(
    "--teacher_path",
    default=None,
    type=str,
    help="Path to teacher model checkpoint.",
)
@click.option(
    "--teacher_ema/--no-teacher_ema",
    default=False,
    help="Use EMA weights from teacher checkpoint if available.",
)
@click.option("--propose_mean", default=-1.2, type=float)
@click.option("--propose_std", default=1.2, type=float)
@click.option(
    "--use_loss_weight/--no-use_loss_weight",
    default=True,
    help="Enable learnable balancing coefficients.",
)
@click.option(
    "--automatic_init_loss_weight/--no-automatic_init_loss_weight",
    default=False,
    help="(Placeholder) automatic init of loss weights.",
)
@click.option(
    "--use_sep_optimizer/--no-use_sep_optimizer",
    default=True,
    help="Use separate optimizer for loss weights.",
)
@click.option(
    "--physics_type",
    type=click.Choice(["fdm", "ritz"]),
    default="fdm",
    help="Physics backend: FDM residual or Ritz energy.",
)
@click.option("--use_ct/--no-use_ct", default=True, help="Use CT loss.")
@click.option(
    "--use_phy_rand/--no-use_phy_rand",
    default=True,
    help="Use physics loss at random t.",
)
@click.option(
    "--use_phy_1step/--no-use_phy_1step",
    default=True,
    help="Use 1-step physics loss (t0 = pi/2).",
)
@click.option(
    "--use_phy_2step/--no-use_phy_2step",
    default=True,
    help="Use 2-step physics loss (t0 -> t1).",
)
@click.option(
    "--use_a_ste/--no-use_a_ste",
    default=False,
    help="Use STE (discrete_a_ste) for a.",
)
@click.option("--use_ct_t1/--no-use_ct_t1", default=False, help="Add extra CT loss at t1 = 1.1")
@click.option("--mask_fdm/--no-mask_fdm", default=True, help="Mask FDM loss near interface.")
@click.option(
    "--t1_strategy",
    type=click.Choice(["fixed", "uniform", "normal"]),
    default="fixed",
    help="Strategy for selecting t1 in 2-step physics loss.",
)
@click.option(
    "--t1_mean",
    default=1.1,
    type=float,
    help="Mean value for t1 when using uniform or normal strategy.",
)
@click.option(
    "--t1_width",
    default=0.2,
    type=float,
    help="Width (half-width for uniform, std-dev for normal) for t1.",
)
@click.option(
    "--sep_optimizer_lr",
    default=1e-3,
    type=float,
    help="Learning rate for loss weight optimizer if using separate optimizer.",
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    default=None,
    help="Path to YAML config file. Values here override code defaults.",
)
def main(
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    save_interval: int,
    save_path: str,
    seed: Optional[int],
    amp: bool,
    ema: bool,
    use_sep_net: bool,
    pretrained_path: str,
    distillation: bool,
    teacher_path: Optional[str],
    teacher_ema: bool,
    propose_mean: float,
    propose_std: float,
    use_loss_weight: bool,
    automatic_init_loss_weight: bool,
    use_sep_optimizer: bool,
    physics_type: str,
    use_ct: bool,
    use_phy_rand: bool,
    use_phy_1step: bool,
    use_phy_2step: bool,
    use_a_ste: bool,
    use_ct_t1: bool,
    mask_fdm: bool,
    t1_strategy: str,
    t1_mean: float,
    t1_width: float,
    sep_optimizer_lr: float,
    config: Optional[str]
):
    # ----------------------
    # Load config (if any)
    # ----------------------
    if config is not None:
        with open(config, "r") as f:
            cfg = yaml.safe_load(f) or {}
        print(f"[INFO] Loaded config from {config}")

        # Config values override CLI defaults
        device       = cfg.get("device", device)
        epochs       = cfg.get("epochs", epochs)
        batch_size   = cfg.get("batch_size", batch_size)
        lr           = cfg.get("lr", lr)
        save_interval = cfg.get("save_interval", save_interval)
        save_path    = cfg.get("save_path", save_path)
        seed         = cfg.get("seed", seed)

        amp          = cfg.get("amp", amp)
        ema          = cfg.get("ema", ema)
        use_sep_net  = cfg.get("use_sep_net", use_sep_net)

        pretrained_path  = cfg.get("pretrained_path", pretrained_path)
        distillation     = cfg.get("distillation", distillation)
        teacher_path     = cfg.get("teacher_path", teacher_path)
        teacher_ema      = cfg.get("teacher_ema", teacher_ema)

        propose_mean = cfg.get("propose_mean", propose_mean)
        propose_std  = cfg.get("propose_std", propose_std)

        use_loss_weight              = cfg.get("use_loss_weight", use_loss_weight)
        automatic_init_loss_weight   = cfg.get("automatic_init_loss_weight", automatic_init_loss_weight)
        use_sep_optimizer            = cfg.get("use_sep_optimizer", use_sep_optimizer)

        physics_type  = cfg.get("physics_type", physics_type)
        use_ct        = cfg.get("use_ct", use_ct)
        use_phy_rand  = cfg.get("use_phy_rand", use_phy_rand)
        use_phy_1step = cfg.get("use_phy_1step", use_phy_1step)
        use_phy_2step = cfg.get("use_phy_2step", use_phy_2step)
        use_a_ste     = cfg.get("use_a_ste", use_a_ste)
        use_ct_t1     = cfg.get("use_ct_t1", use_ct_t1)
        mask_fdm      = cfg.get("mask_fdm", mask_fdm)
        t1_strategy  = cfg.get("t1_strategy", t1_strategy)
        t1_mean      = cfg.get("t1_mean", t1_mean)
        t1_width     = cfg.get("t1_width", t1_width)
        sep_optimizer_lr = cfg.get("sep_optimizer_lr", sep_optimizer_lr)

    # ----------------------
    # Seeding
    # ----------------------
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        print(f"[INFO] Random seed set to {seed}")

    # ----------------------
    # Data
    # ----------------------
    dataloader = load_data(batch_size)

    # ----------------------
    # Student model
    # ----------------------
    teacher_model = None

    if use_sep_net:
        # sep model can optionally load pre-trained weights internally
        try:
            model = create_sep_model(pre_trained_path=pretrained_path).to(device)
            print(f"[INFO] Loaded sep model from {pretrained_path}")
        except Exception as e:
            print(f"[WARN] Error loading sep model from {pretrained_path}: {e}")
            print("[WARN] Proceeding with randomly initialized sep model.")
            model = create_sep_model().to(device)
    else:
        model = create_model().to(device)
        if pretrained_path is not None:
            try:
                model = load_model_state(
                    model, pretrained_path, key="model_state_dict", device=device
                )
                print(f"[INFO] Loaded base model from {pretrained_path}")
            except Exception as e:
                print(f"[WARN] Error loading base model: {e}")
                print("[WARN] Proceeding with randomly initialized model.")

    # ----------------------
    # Teacher model (distillation)
    # ----------------------
    if distillation:
        if teacher_path is None:
            raise ValueError("Teacher model path must be provided for distillation.")
        try:
            teacher_model = create_model().to(device)
            teacher_model = load_model_state(
                teacher_model, teacher_path, key="model_state_dict", device=device
            )
            print(f"[INFO] Loaded teacher model from {teacher_path}")
            if teacher_ema and teacher_model is not None:
                if hasattr(teacher_model, "ema_model") and teacher_model.ema_model is not None:
                    teacher_model = teacher_model.ema_model
                    print("[INFO] Using EMA weights for the teacher model.")
                else:
                    print("[WARN] EMA weights not found; using standard weights.")
        except Exception as e:
            print(f"[ERROR] Error loading teacher model: {e}")
            raise

    # ----------------------
    # torch.compile (if available)
    # ----------------------
    try:
        model = torch.compile(model)
        if teacher_model is not None:
            teacher_model = torch.compile(teacher_model)
        print("[INFO] Using torch.compile()")
    except Exception as e:
        print(f"[WARN] torch.compile not available or failed: {e}")

    # ----------------------
    # Loss weights setup
    # ----------------------
    loss_weights = None
    loss_weight_optimizer = None

    if use_loss_weight:
        if automatic_init_loss_weight:
            print("[WARN] Automatic init of loss weights not implemented; using defaults.")
        # Different initial balancing for FDM vs Ritz
        if physics_type == "fdm":
            param_dict = {
                "ct": 10.0,
                "rand": -10.0,
                "one_step": -10.0,
                "two_step": -10.0,
                't1_ct': 10.0,
            }
        else:  # physics_type == "ritz"
            param_dict = {
                "ct": 10.0,
                "rand": -10.0,
                "one_step": -10.0,
                "two_step": -10.0,
                't1_ct': 10.0,
            }

        loss_weights = LossWeightManager(param_dict).to(device)

    # ----------------------
    # Optimizer setup
    # ----------------------
    if use_sep_optimizer:
        print("[INFO] Using separate optimizers for model and loss weights.")
        if use_sep_net and hasattr(model, "unet") and hasattr(model.unet, "decoder_trainable"):
            model_params = model.unet.decoder_trainable.parameters()
        else:
            model_params = model.parameters()

        optimizer = torch.optim.RAdam(model_params, lr=lr)

        if loss_weights is not None:
            loss_weight_optimizer = torch.optim.SGD(loss_weights.parameters(), lr=sep_optimizer_lr)
    else:
        print("[INFO] Using single optimizer for model + loss weights.")
        if use_sep_net and hasattr(model, "unet") and hasattr(model.unet, "decoder_trainable"):
            trainable_params = list(model.unet.decoder_trainable.parameters())
        else:
            trainable_params = list(model.parameters())

        if loss_weights is not None:
            trainable_params += list(loss_weights.parameters())

        optimizer = torch.optim.RAdam(trainable_params, lr=lr)
        loss_weight_optimizer = None

    if teacher_model is None:
        print("[INFO] Training with Consistency Training.")
    else:
        print("[INFO] Training with Consistency Distillation.")

    # ----------------------
    # Call unified train loop
    # ----------------------
    train_loop(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        epochs=epochs,
        device=device,
        save_interval=save_interval,
        output_path=save_path,
        sigma_data=0.5,
        P_mean=propose_mean,
        P_std=propose_std,
        use_amp=amp,
        use_ema=ema,
        teacher_model=teacher_model,
        loss_weights=loss_weights,
        loss_weight_optimizer=loss_weight_optimizer,
        use_sep_net=use_sep_net,
        use_ct=use_ct,
        use_phy_rand=use_phy_rand,
        use_phy_1step=use_phy_1step,
        use_phy_2step=use_phy_2step,
        physics_type=physics_type,
        use_a_ste=use_a_ste,
        use_ct_t1=use_ct_t1,
        t1_strategy=t1_strategy,
        t1_mean=t1_mean,
        t1_width=t1_width,
        mask_fdm=mask_fdm,
    )


if __name__ == "__main__":
    main()
