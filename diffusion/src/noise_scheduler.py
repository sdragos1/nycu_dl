import torch


class NoiseScheduler:
    def __init__(self, beta_start: float, beta_end: float, total_steps: int):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.total_steps = total_steps

        self.beta = torch.linspace(self.beta_start, self.beta_end, self.total_steps)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.minus_sqrt_alpha_bar = 1 - self.sqrt_alpha_bar

    def noise(self, image: torch.Tensor, t: int) -> torch.Tensor:
        assert t < self.total_steps

        eps = torch.randn_like(image)
        sqrt_alpha_bar = self.sqrt_alpha_bar[t]
        minus_sqrt_alpha_bar = self.minus_sqrt_alpha_bar[t]
        return sqrt_alpha_bar * image + minus_sqrt_alpha_bar * eps
