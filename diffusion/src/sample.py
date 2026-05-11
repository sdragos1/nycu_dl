import torch
from torch import nn

from src.noise_scheduler import LinearNoiseScheduler


@torch.no_grad()
def sample(model: nn.Module, labels_t: torch.Tensor,
           noise_scheduler: LinearNoiseScheduler, device: torch.device) -> torch.Tensor:
    x = torch.randn((labels_t.size(0), 3, 64, 64)).to(device)

    for t in reversed(range(noise_scheduler.total_steps)):
        timestamps_t = torch.full((labels_t.size(0),), t, device=device, dtype=torch.long)

        pred = model(x, timestamps_t, labels_t)
        alpha = noise_scheduler.alpha[t]

        sigma = torch.sqrt(noise_scheduler.beta[t])
        minus_sqrt_alpha_bar = noise_scheduler.minus_sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        mean = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / minus_sqrt_alpha_bar) * pred)

        if t > 0:
            noise = torch.randn_like(x)
            x = mean + sigma * noise
        else:
            x = mean
    return x
