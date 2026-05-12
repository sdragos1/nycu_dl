import os
import hydra
import torch
from omegaconf import DictConfig
from torchvision.utils import save_image, make_grid

from src.ddpm.unet import UNet
from src.iclevr_dataset import test_data_loaders, ICLEVRDataset
from src.noise_scheduler import LinearNoiseScheduler
from src.utils import get_device
from src.evaluator import Evaluation
from src.sample import sample

@torch.no_grad()
def generate_process_image(model, label_t, noise_scheduler, device, save_path):
    x = torch.randn((1, 3, 64, 64)).to(device)
    
    save_steps = torch.linspace(noise_scheduler.total_steps - 1, 0, 10, dtype=torch.long)
    process_images = []
    
    for t in reversed(range(noise_scheduler.total_steps)):
        timestamps_t = torch.full((1,), t, device=device, dtype=torch.long)

        pred = model(x, timestamps_t, label_t)
        alpha = noise_scheduler.alpha[t]

        sigma = torch.sqrt(noise_scheduler.beta[t])
        minus_sqrt_alpha_bar = noise_scheduler.minus_sqrt_alpha_bar[t]
        mean = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / minus_sqrt_alpha_bar) * pred)

        if t > 0:
            noise = torch.randn_like(x)
            x = mean + sigma * noise
        else:
            x = mean
            
        if t in save_steps:
            process_images.append(((x.clone().detach().cpu() + 1) / 2).clamp(0, 1))
            
    grid = make_grid(torch.cat(process_images, dim=0), nrow=10)
    save_image(grid, save_path)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def infer(cfg: DictConfig) -> None:
    device = get_device()
    
    os.makedirs("images/test", exist_ok=True)
    os.makedirs("images/new_test", exist_ok=True)

    noise_scheduler = LinearNoiseScheduler(
        beta_start=cfg.train.beta_start,
        beta_end=cfg.train.beta_end,
        total_steps=cfg.train.num_timesteps,
        device=device
    )

    model_path = cfg.train.resume_ckpt if cfg.train.resume_ckpt else "./best_model.pt"

    model = UNet(cfg.model).to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    if 'model' in state_dict:
        model.load_state_dict(state_dict['model'])
    else:
        model.load_state_dict(state_dict)
    model.eval()
    
    evaluator = Evaluation(device=device)

    print("Generating denoising process image...")
    objects = ICLEVRDataset(root=cfg.train.data_root or None).objects
    num_classes = len(objects)
    
    special_prompt = ["red sphere", "cyan cylinder", "cyan cube"]
    special_label = torch.zeros(num_classes, dtype=torch.float32)
    for label in special_prompt:
        special_label[objects[label]] = 1.0
    special_label = special_label.unsqueeze(0).to(device)
    generate_process_image(model, special_label, noise_scheduler, device, "images/denoising_process.png")
    print("Denoising process image saved to images/denoising_process.png")

    test_loader, new_test_loader = test_data_loaders(batch_size=8, root=cfg.train.data_root)
    test_loader.collate_fn = lambda x: x
    new_test_loader.collate_fn = lambda x: x

    for loader, name in [(test_loader, "test"), (new_test_loader, "new_test")]:
        print(f"Evaluating {name} dataset...")
        all_samples = []
        all_labels_t = []
        
        idx = 0
        for batch_labels in loader:
            batch_size = len(batch_labels)
            labels_t = torch.zeros((batch_size, num_classes), dtype=torch.float32)
            for b in range(batch_size):
                for label in batch_labels[b]:
                    labels_t[b, objects[label]] = 1.0
                    
            labels_t = labels_t.to(device)
            with torch.no_grad():
                samples = sample(model, labels_t, noise_scheduler, device)
                
            samples = samples.clamp(min=-1, max=1)
            
            all_samples.append(samples)
            all_labels_t.append(labels_t)
            
            samples_img = (samples + 1) / 2
            for b in range(batch_size):
                save_path = f"images/{name}/{idx}.png"
                save_image(samples_img[b], save_path)
                idx += 1
                
        full_samples = torch.cat(all_samples, dim=0)
        full_labels_t = torch.cat(all_labels_t, dim=0)
        
        acc = evaluator.eval(full_samples, full_labels_t)
        print(f"Accuracy for {name}: {acc * 100:.2f}%")
        
        full_samples_img = (full_samples + 1) / 2
        grid = make_grid(full_samples_img, nrow=8)
        save_image(grid, f"images/{name}_grid.png")
        print(f"Saved {name} grid to images/{name}_grid.png")

if __name__ == '__main__':
    infer()
