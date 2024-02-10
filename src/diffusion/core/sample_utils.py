import os
from collections import OrderedDict
from pathlib import Path
import yaml
import torch

from diffusion.custom_modules.models.base_model import BaseDenoiser
import diffusion.custom_modules.models as models

def load_config_from_dir(model_dir: Path) -> dict:
    with open(model_dir / "config.yaml", "rb") as f:
        cfg = yaml.load(f, yaml.FullLoader)
    return cfg


def load_model_from_dir(model_dir: Path) -> tuple[dict, BaseDenoiser | None]:
    trimmed_prefix = "AAA_trimmed_"

    model_dir = Path(model_dir)
    cfg = load_config_from_dir(model_dir)
    ckpt_paths = sorted(list(model_dir.glob("*.ckpt")))
    ckpt_path = ckpt_paths[0]
    is_ckpt_path_trimmed = "trimmed" in ckpt_path.name
    if len(ckpt_paths) > 1 and is_ckpt_path_trimmed:
        outdated = ckpt_path.name.replace(trimmed_prefix, "") != ckpt_paths[1].name
        if outdated:
            print(f"trimmed checkpoint is outdated: trimmed={ckpt_path.name}, found={ckpt_paths[1].name}")
            is_ckpt_path_trimmed = False
            os.remove(ckpt_path)  # prevent it from coming up again
            ckpt_path = ckpt_paths[1]

    print(f"using ckpt: {ckpt_path}")
    model_class = getattr(models, cfg["model"]["type"])

    ckpt = torch.load(ckpt_path)
    state_dict = ckpt["state_dict"]
    if "pretrained" in cfg["model"]["kwargs"]:
        print(f"model kwargs contains pretrained, replacing it with None")
        cfg["model"]["kwargs"]["pretrained"] = None
    if "encoder_weights" in cfg["model"]["kwargs"]:
        print(f"model kwargs contains encoder_weights, replacing it with None")
        cfg["model"]["kwargs"]["encoder_weights"] = None

    if is_ckpt_path_trimmed:
        print("trimmed ckpt found")
        model_state_dict = ckpt["state_dict"]
    elif any(k.startswith("ema_") for k in state_dict.keys()):
        print("ema weights found, loading ema weights")
        model_state_dict = OrderedDict([
            (k, v)
            for k, v in state_dict.items()
            if k.startswith("ema_model.")
        ])
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(model_state_dict, prefix="ema_model.module.")
    else:
        print("ema weights not found, loading model")
        model_state_dict = OrderedDict([
            (k, v)
            for k, v in state_dict.items()
            if k.startswith("model.")
        ])
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(model_state_dict, prefix="model.")
    model = model_class(**cfg["model"]["kwargs"])
    load_status = model.load_state_dict(model_state_dict)
    print(load_status)
    model = model.eval()

    # trim down checkpoint and save the trimmed version, probably can shrink ckpt sizes by like 50%
    if is_ckpt_path_trimmed:
        print("loaded ckpt is trimmed so no need to trim it again")
    else:
        ckpt = {"state_dict": model_state_dict}
        trimmed_ckpt_path = ckpt_path.parent / f"{trimmed_prefix}{ckpt_path.name}"
        torch.save(ckpt, trimmed_ckpt_path)
        print(f"saved trimmed ckpt path: {trimmed_ckpt_path}")
    
    return cfg, model