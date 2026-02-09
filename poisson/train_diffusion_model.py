import os 
import torch
import click
import random
import numpy as np

from tqdm import tqdm

from utils import scale_back_u, scale_back_a, plot_result, load_data, sample_dpm_solver
from networks_util import is_compiled, create_model, EMA

# Enable cuDNN autotuner for conv layers
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


def train_loop(
    model,
    optimizer,
    dataloader,
    epochs,
    device,
    save_interval=5000,
    output_path="./sCM/diffusion",
    sigma_data=0.5,
    P_mean=-1.2,
    P_std=1.2,
    use_amp=True,
    use_ema=True,
    use_ritz=False,
):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

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
            images = torch.stack([a, u], dim=1).to(device)

            # ----------------------
            # Forward + Loss
            # ----------------------
            sigma = torch.randn(images.shape[0], device=images.device).reshape(-1, 1, 1, 1)
            sigma = (sigma * P_std + P_mean).exp()
            t = torch.arctan(sigma / sigma_data)

            z = torch.randn_like(images) * sigma_data
            x_t = torch.cos(t) * images + torch.sin(t) * z

            with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
                pred_v_t, logvar = model(x_t / sigma_data, t.flatten(), return_logvar=True)
                pred_v_t = pred_v_t * sigma_data
                logvar = logvar.view(-1, 1, 1, 1)

                v_t = torch.cos(t) * z - torch.sin(t) * images
                # weight = 0.5
                # according to paper, weight should be one over the dimesionality of the data
                weight = 1 / np.prod(images.shape[1:])
                loss = (weight / torch.exp(logvar)) * ((pred_v_t - v_t) / sigma_data) ** 2 + logvar
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
                pred = sample_dpm_solver(active_model, device, num_steps=35, schedule="power")
                pred[:, 0, :, :] = scale_back_a(pred[:, 0, :, :])
                pred[:, 1, :, :] = scale_back_u(pred[:, 1, :, :])

                samples_dir = os.path.join(output_path, "samples")
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
        # state_dict is not saved correctly if model is compiled
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
@click.option('--epochs', default=10, help='Number of training epochs.')
@click.option('--batch_size', default=16, help='Batch size for training.')
@click.option('--lr', default=1e-3, help='Learning rate for the optimizer.')
@click.option('--save_interval', default=5000, help='Interval (in steps) to save model checkpoints and samples.')
@click.option('--save_path', default='./poisson-diffpde-output/diffusion', help='Path to save model checkpoints and samples.')
@click.option('--seed', help='Optional random seed for reproducibility', default=None, type=int)
@click.option('--amp/--no-amp', default=True, help='Enable/disable AMP (bfloat16 autocast).')
@click.option('--ema/--no-ema', default=False, help='Enable/disable EMA model tracking.')
def main(device, epochs, batch_size, lr, save_interval, save_path, seed, amp, ema):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    dataloader = load_data(batch_size)
    model = create_model().to(device)

    # Compile model if supported (PyTorch 2.0+)
    try:
        model = torch.compile(model)
    except Exception as e:
        print(f"torch.compile not available: {e}")

    optimizer = torch.optim.RAdam(model.parameters(), lr=lr)

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
    )


if __name__ == '__main__':
    main()