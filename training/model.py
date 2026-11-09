"""Small text-conditioned 3D diffusion network, initialized from random weights.

This is a research baseline, not a production video model. It operates on tiny
clips (default 8 x 32 x 32) to make the training pipeline testable.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F


class TextConditioner(nn.Module):
    """Simple hashed-token embedding; deliberately avoids pretrained text models."""

    def __init__(self, vocab_size: int = 4096, dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.vocab_size = vocab_size

    def tokenize(self, prompts, device):
        rows = []
        for prompt in prompts:
            tokens = prompt.lower().split()[:32] or ["empty"]
            ids = [sum((i + 1) * ord(ch) for i, ch in enumerate(token)) % self.vocab_size
                   for token in tokens]
            rows.append(ids)
        width = max(len(row) for row in rows)
        ids = torch.zeros(len(rows), width, dtype=torch.long, device=device)
        mask = torch.zeros(len(rows), width, dtype=torch.float32, device=device)
        for i, row in enumerate(rows):
            ids[i, :len(row)] = torch.tensor(row, device=device)
            mask[i, :len(row)] = 1.0
        return ids, mask

    def forward(self, prompts, device):
        ids, mask = self.tokenize(prompts, device)
        vectors = self.embedding(ids)
        return (vectors * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    scale = math.log(10000) / max(half - 1, 1)
    frequencies = torch.exp(torch.arange(half, device=timesteps.device) * -scale)
    angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    result = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if dim % 2:
        result = F.pad(result, (0, 1))
    return result


class ResBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int):
        super().__init__()
        groups_in = 8 if in_channels % 8 == 0 else 1
        groups_out = 8 if out_channels % 8 == 0 else 1
        self.norm1 = nn.GroupNorm(groups_in, in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.condition = nn.Linear(cond_dim, out_channels)
        self.norm2 = nn.GroupNorm(groups_out, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, condition):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.condition(condition)[:, :, None, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return (h + self.skip(x)) / math.sqrt(2.0)


class TinyVideoDiffusion(nn.Module):
    """A compact 3D U-Net that predicts diffusion noise in video clips."""

    def __init__(self, base_channels: int = 32, cond_dim: int = 128):
        super().__init__()
        self.text = TextConditioner(dim=cond_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim * 2), nn.SiLU(),
            nn.Linear(cond_dim * 2, cond_dim),
        )
        self.input = nn.Conv3d(3, base_channels, 3, padding=1)
        self.enc1 = ResBlock3D(base_channels, base_channels, cond_dim)
        self.down = nn.Conv3d(base_channels, base_channels * 2, 3,
                              stride=(1, 2, 2), padding=1)
        self.mid = ResBlock3D(base_channels * 2, base_channels * 2, cond_dim)
        self.up_project = nn.Conv3d(base_channels * 2, base_channels, 3, padding=1)
        self.dec1 = ResBlock3D(base_channels * 2, base_channels, cond_dim)
        self.output_norm = nn.GroupNorm(8 if base_channels % 8 == 0 else 1, base_channels)
        self.output = nn.Conv3d(base_channels, 3, 3, padding=1)

    def forward(self, noisy_video, timesteps, prompts):
        text_condition = self.text(prompts, noisy_video.device)
        time_condition = self.time_mlp(timestep_embedding(timesteps, text_condition.shape[-1]))
        condition = text_condition + time_condition

        x0 = self.input(noisy_video)
        skip = self.enc1(x0, condition)
        x = self.down(skip)
        x = self.mid(x, condition)
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = self.up_project(x)
        x = self.dec1(torch.cat([x, skip], dim=1), condition)
        return self.output(F.silu(self.output_norm(x)))
