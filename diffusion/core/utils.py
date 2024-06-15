import numpy as np
from pathlib import Path
import yaml
import hydra
from omegaconf import OmegaConf
from pathlib import Path
import torch
import torch.nn as nn
from collections import OrderedDict

from diffusion.envs.constants import MODEL_OUT_DIR

def load_config_from_dir(model_dir: str | Path) -> dict:
    with open(model_dir / "config.yaml", "rb") as f:
        cfg = yaml.load(f, yaml.FullLoader)
    return cfg


def load_model_from_dir(
        model_dir: str | Path,
) -> tuple[dict, nn.Module]:
    model_dir = Path(model_dir)
    cfg = load_config_from_dir(model_dir)
    model_state_dict = get_state_dict_from_dir(model_dir)
    model = hydra.utils.instantiate(cfg["model"])
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(model_state_dict, prefix="model.")
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(model_state_dict, prefix="ema_model.module.")
    load_status = model.load_state_dict(model_state_dict)
    print(f"{load_status=}")
    model = model.eval()
    return model


def get_state_dict_from_dir(
        model_dir: str | Path
):
    model_dir = Path(model_dir)
    ckpt_paths = sorted(list(model_dir.glob("*.ckpt")))
    ckpt_path = ckpt_paths[-1]

    print(f"using last ckpt: {ckpt_path}")
    ckpt = torch.load(ckpt_path)
    model_state_dict = ckpt["state_dict"]
    return model_state_dict

