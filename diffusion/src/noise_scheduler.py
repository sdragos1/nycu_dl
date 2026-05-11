import torch


class LinearNoiseScheduler:
    def __init__(self, beta_start: float, beta_end: float, total_steps: int, device: torch.device):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.total_steps = total_steps

        self.beta = torch.linspace(self.beta_start, self.beta_end, self.total_steps).to(device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0).to(device)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar).to(device)
        self.minus_sqrt_alpha_bar = torch.sqrt(1 - self.alpha_bar).to(device)

    def noise(self, images_t: torch.Tensor, timesteps_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        eps = torch.randn_like(images_t)
        sqrt_alpha_bar = self.sqrt_alpha_bar[timesteps_t].view(-1, 1, 1, 1)
        minus_sqrt_alpha_bar = self.minus_sqrt_alpha_bar[timesteps_t].view(-1, 1, 1, 1)
        return sqrt_alpha_bar * images_t + minus_sqrt_alpha_bar * eps, eps
