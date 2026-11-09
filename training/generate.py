"""Sample a short video from a trained MotionLab research checkpoint."""
import argparse

import imageio.v2 as imageio
import numpy as np
import torch

from training.model import TinyVideoDiffusion
from training.train import get_device


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/latest.pt")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", default="generated.mp4")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = get_device(args.device)
    state = torch.load(args.checkpoint, map_location=device)
    config = state.get("config", {})
    frames = config.get("frames", 8)
    size = config.get("size", 32)

    model = TinyVideoDiffusion().to(device)
    model.load_state_dict(state["model"])
    model.eval()

    total_steps = 1000
    betas = torch.linspace(1e-4, 0.02, total_steps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    stride = max(total_steps // max(args.steps, 1), 1)
    schedule = list(range(total_steps - 1, -1, -stride))[:args.steps]

    sample = torch.randn(1, 3, frames, size, size, device=device)
    prompt = [args.prompt]
    for step in schedule:
        t = torch.tensor([step], device=device, dtype=torch.long)
        predicted_noise = model(sample, t, prompt)
        alpha = alphas[step]
        alpha_bar = alpha_bars[step]
        beta = betas[step]
        sample = (sample - beta / torch.sqrt(1.0 - alpha_bar) * predicted_noise) / torch.sqrt(alpha)
        if step > 0:
            sample = sample + torch.sqrt(beta) * torch.randn_like(sample)

    sample = sample.clamp(-1, 1)[0].permute(1, 2, 3, 0).cpu().numpy()
    output_frames = [((frame + 1) * 127.5).clip(0, 255).astype(np.uint8)
                     for frame in sample]
    imageio.mimsave(args.output, output_frames, fps=4)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
