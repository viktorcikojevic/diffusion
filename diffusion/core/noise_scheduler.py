"""
Code ported from here: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
"""
import math
import numpy as np
import torch

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas).astype(np.float32)


class DDPMScheduler():
    
    
    def __init__(
            self,    
            num_diffusion_timesteps: int,
            scheduler_type: str = "linear",
            img_key: str = "img",
            img_noised_key: str = "img_noised",
            noise_output_key: str = "noise",
            timesteps_key: str = "timesteps",
    ) -> None:    
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.scheduler_type = scheduler_type    
        
        # keys
        self.img_key = img_key
        self.img_noised_key = img_noised_key
        self.noise_output_key = noise_output_key
        self.timesteps_key = timesteps_key
        
        # get bets
        self.beta_t = get_named_beta_schedule(scheduler_type, num_diffusion_timesteps).astype(np.float32)
        self.alpha_t = 1 - self.beta_t
        self.alpha_t_bar = np.cumprod(self.alpha_t)
        self.sigma_t = np.sqrt(self.beta_t)
        
        # convert to torch tensors
        self.beta_t = torch.from_numpy(self.beta_t)
        self.alpha_t = torch.from_numpy(self.alpha_t)
        self.alpha_t_bar = torch.from_numpy(self.alpha_t_bar)
        self.alpha_t_bar_sqrt = self.alpha_t_bar.sqrt()
        self.one_minus_alpha_t_bar_sqrt = (1-self.alpha_t_bar).sqrt()
        # send to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._send_to_device(device)
        
    def _send_to_device(self, device: torch.device):
        self.beta_t = self.beta_t.to(device)
        self.alpha_t = self.alpha_t.to(device)
        self.alpha_t_bar = self.alpha_t_bar.to(device)
        self.alpha_t_bar_sqrt = self.alpha_t_bar_sqrt.to(device)
        self.one_minus_alpha_t_bar_sqrt = self.one_minus_alpha_t_bar_sqrt.to(device)    
    
    def sample_forward_process(self, batch: dict) -> dict:
        
        
        # sample timesteps
        num_timesteps = batch[self.img_key].shape[0]
        timesteps = torch.randint(0, self.num_diffusion_timesteps, (num_timesteps,), device=batch[self.img_key].device)
        batch[self.timesteps_key] = timesteps
        
        # sample noise
        uniform_noise = torch.randn_like(batch[self.img_key], device=batch[self.img_key].device)
        batch[self.noise_output_key] = uniform_noise
        
        c1 = self.alpha_t_bar_sqrt[timesteps].view(-1, 1, 1, 1)
        c2 = self.one_minus_alpha_t_bar_sqrt[timesteps].view(-1, 1, 1, 1)
        
        # create img_noised
        img_noised = c1 * batch[self.img_key] + c2 * uniform_noise
        batch[self.img_noised_key] = img_noised
            
        return batch