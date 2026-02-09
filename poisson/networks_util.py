import torch
import torch.nn as nn
from collections import OrderedDict
from networks import SongUNet, SongUNetEncoder, SongUNetDecoder, SplitSongUNet
import copy

def count_parameters(model, requires_grad=True):
    """Count the number of parameters in the model."""
    if not requires_grad:
        return sum(p.numel() for p in model.parameters())
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def copy_weights(pretrained_model, model):
    """Copy the weights from the pretrained model to the unet."""
    for name, param in pretrained_model.named_parameters():
        if name in model.state_dict():
            model.state_dict()[name].copy_(param)
        else:
            print(f"Skipping {name} as it is not in the model.")

class LogVarLayer(nn.Module):
    def __init__(self, in_channels):
        super(LogVarLayer, self).__init__()
        self.linear = nn.Linear(in_channels, 1)

    def forward(self, x):
        return self.linear(x)
    
class ModifiedUnet(nn.Module):
    def __init__(self, unet, logvar_dim=512):
        super(ModifiedUnet, self).__init__()
        self.unet = unet
        self.logvar_layer = LogVarLayer(logvar_dim)

    def forward(self, x, t, return_logvar=False):
        if not return_logvar:
            pred = self.unet(x, t, class_labels=None, return_emb=return_logvar)
        else:
            pred, emb = self.unet(x, t, class_labels=None, return_emb=return_logvar)
        if return_logvar:
            logvar = self.logvar_layer(emb)
            return pred, logvar
        return pred
    
def load_model_state(model, checkpoint_path, key="model_state_dict", device="cpu", return_checkpoint=False):
    """
    Load model weights robustly, handling both normal and torch.compile ('_orig_mod.') checkpoints.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = checkpoint[key] if key in checkpoint else checkpoint

    # If keys start with "_orig_mod.", strip them
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        new_state = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                new_state[k[len("_orig_mod."):]] = v
            else:
                new_state[k] = v
        state_dict = new_state

    # Load into the model
    model.load_state_dict(state_dict, strict=True)
    if return_checkpoint:
        return model, checkpoint
    return model

def is_compiled(model):
    return hasattr(model, "_orig_mod")

def create_model():
    unet = SongUNet(128, 2, 2, embedding_type='positional', encoder_type='standard', decoder_type='standard', channel_mult_noise=1, resample_filter=[1,1], model_channels=128, channel_mult=[2,2,2], augment_dim=9)
    model = ModifiedUnet(unet , unet.map_layer1.out_features)
    return model

class EMA:
    """Exponential Moving Average for model parameters."""
    def __init__(self, model, decay=0.99):
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for p_ema, p in zip(self.ema_model.parameters(), model.parameters()):
            p_ema.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)

def create_sep_model(pre_trained_path=None):
    unet = SongUNet(128,
                2,
                2,
                embedding_type='positional',
                encoder_type='standard',
                decoder_type='standard',
                channel_mult_noise=1,
                resample_filter=[1, 1],
                model_channels=128,
                channel_mult=[2, 2, 2],
                augment_dim=9)
    model = ModifiedUnet(unet)
    if pre_trained_path is not None:
        try:
            print(f"Loading pre-trained model from {pre_trained_path}")
            model = load_model_state(model, pre_trained_path, key="model_state_dict")
        except Exception as e:
            print(f"Failed to load pre-trained model: {e}")
    unet = model.unet

    encoder = SongUNetEncoder(unet)
    for params in encoder.parameters():
        params.requires_grad = False
    decoder_1 = SongUNetDecoder(unet.dec)
    decoder_2 = SongUNetDecoder(copy.deepcopy(unet.dec))
    model = SplitSongUNet(encoder, decoder_1, decoder_2, frozen=False)
    model = ModifiedUnet(model, logvar_dim=unet.map_layer1.out_features)
    for params in model.unet.parameters():
        params.requires_grad = False
    for params in model.unet.decoder_trainable.parameters():
        params.requires_grad = True
    return model

class LossWeightManager(nn.Module):
    def __init__(self, param_dict, device=None):
        """
        param_dict: dict
            Dictionary of {name: init_value}.
            Example: {"ct": 10.0, "rand": -10.0, "one_step": -10.0, "two_step": -10.0}
        device: torch.device or None
            Device to place the parameters on.
        """
        super().__init__()
        self.device = device or torch.device("cpu")

        # register parameters dynamically
        for name, init_val in param_dict.items():
            param = nn.Parameter(torch.tensor(init_val, dtype=torch.float32, device=self.device))
            self.register_parameter(f"w_loss_{name}", param)

    def forward(self):
        """
        Optional: return a dict of all loss weights
        """
        return {name: param for name, param in self.named_parameters()}

    def __getitem__(self, key):
        """Allow dictionary-like access: manager['ct']"""
        return getattr(self, f"w_loss_{key}")

    def parameters_list(self):
        """Convenience: get list of all nn.Parameter"""
        return list(self.parameters())
