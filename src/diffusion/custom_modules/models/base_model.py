from dataclasses import dataclass
from abc import ABC
import torch


@dataclass
class DenoiserOutput:
    pred: torch.Tensor


class BaseDenoiser(ABC, torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)

    def get_name(self) -> str:
        """

        :return: str, name of the model to be logged on wandb
        """
        pass

    def predict(self, img: torch.Tensor, timesteps: torch.Tensor) -> DenoiserOutput:
        """

        :param img: torch.Tensor: (b, c, h, w)
        :param timesteps: torch.Tensor: (b, )
        :return: DenoiserOutput:
            - pred: (b, c, h, w)
        """
        pass
