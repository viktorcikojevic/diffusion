import torch
import numpy as np


class DDPMScheduler():
    
    def __init__(
            self,    
            num_diffusion_timesteps: int,
            noise_scheduler: str = "linear"
    ) -> None:
    
        assert noise_scheduler in ["linear", "cosine"], f"unknown noise scheduler: {noise_scheduler}"
    
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.noise_scheduler = noise_scheduler    
        self._get_named_beta_schedule(noise_scheduler)
        
        # convert to torch tensors
        self.beta_t = torch.from_numpy(self.beta_t)
        self.alpha_t = torch.from_numpy(self.alpha_t)
        self.alpha_t_bar = torch.from_numpy(self.alpha_t_bar)
        
    
    
    def _get_named_beta_schedule(self, name: str) -> np.ndarray:
        """
        Get a pre-defined beta schedule for the given name.
        """
        if self.noise_scheduler == "linear":
            scale = 1000 / self.num_diffusion_timesteps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            self.beta_t = np.linspace(
                beta_start, beta_end, self.num_diffusion_timesteps, dtype=np.float64
            )
            self.alpha_t = 1 - self.beta_t
            self.alpha_t_bar = np.cumprod(self.alpha_t)
            self.sigma_t = np.sqrt(self.beta_t)
        else:
            raise NotImplementedError(f"unknown beta schedule: {self.noise_scheduler}")
        
    
    def generate_denoising_data(self, img: torch.Tensor) -> torch.Tensor:
        """
        Generate noise and timesteps for the given image
        """
        B, C, H, W = img.shape
        img_device = img.device
        
        noise = torch.randn(B, C, H, W).to(img_device)
        # generate random timesteps from 1 to num_diffusion_timesteps, B of those
        timesteps = torch.randint(1, self.num_diffusion_timesteps + 1, (B,))
        
        alpha_t_bar = self.alpha_t_bar[timesteps - 1].to(img_device)
        while len(alpha_t_bar.shape) < len(img.shape):
            alpha_t_bar = alpha_t_bar[..., None]
        img_noised = (torch.sqrt(alpha_t_bar) * img + torch.sqrt(1 - alpha_t_bar) * noise).float()
        timesteps = timesteps.to(img_device)
        
        return img_noised, noise, timesteps