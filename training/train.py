"""Train the tiny video diffusion baseline from random initialization."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from training.dataset import LocalVideoDataset
from training.model import TinyVideoDiffusion


def get_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", default="")
    args = parser.parse_args()

    device = get_device(args.device)
    dataset = LocalVideoDataset(args.data, frames=args.frames, size=args.size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, pin_memory=(device.type == "cuda"))
    model = TinyVideoDiffusion().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start_epoch = 0
    global_step = 0

    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state.get("epoch", 0)
        global_step = state.get("global_step", 0)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Linear beta schedule: intentionally simple baseline for experimentation.
    betas = torch.linspace(1e-4, 0.02, 1000, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    print(f"Device: {device}; training clips: {len(dataset)}; parameters: "
          f"{sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        for videos, prompts in loader:
            videos = videos.to(device)
            batch = videos.shape[0]
            t = torch.randint(0, len(betas), (batch,), device=device)
            noise = torch.randn_like(videos)
            a_bar = alpha_bars[t].view(batch, 1, 1, 1, 1)
            noisy = a_bar.sqrt() * videos + (1.0 - a_bar).sqrt() * noise

            prediction = model(noisy, t, list(prompts))
            loss = F.mse_loss(prediction, noise)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            global_step += 1

        average = total_loss / max(len(loader), 1)
        metrics = {"epoch": epoch + 1, "global_step": global_step,
                   "train_noise_mse": average, "device": str(device)}
        print(json.dumps(metrics))
        checkpoint = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1, "global_step": global_step,
            "config": {"frames": args.frames, "size": args.size,
                       "base_channels": 32, "from_scratch": True},
        }
        torch.save(checkpoint, output_dir / "latest.pt")

    print(f"Training run finished. Checkpoint: {output_dir / 'latest.pt'}")


if __name__ == "__main__":
    main()
