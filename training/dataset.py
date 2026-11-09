"""Local video dataset reader.

Expected layout:
  data/videos/example.mp4
  data/captions/example.txt

Use only videos you own or are licensed to use. No dataset is downloaded.
"""
from pathlib import Path
import random

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class LocalVideoDataset(Dataset):
    def __init__(self, root, frames=8, size=32):
        self.root = Path(root)
        self.frames = frames
        self.size = size
        video_dir = self.root / "videos"
        caption_dir = self.root / "captions"
        if not video_dir.exists() or not caption_dir.exists():
            raise FileNotFoundError(
                f"Expected {video_dir} and {caption_dir}. See DATASET.md for the layout."
            )
        extensions = {".mp4", ".mov", ".webm", ".avi", ".mkv"}
        self.items = []
        for video in sorted(video_dir.iterdir()):
            caption = caption_dir / f"{video.stem}.txt"
            if video.suffix.lower() in extensions and caption.exists():
                self.items.append((video, caption))
        if not self.items:
            raise RuntimeError("No matching video + caption pairs found in the dataset folders.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        video_path, caption_path = self.items[index]
        frames = [np.asarray(frame) for frame in iio.imiter(video_path)]
        if len(frames) < 1:
            raise RuntimeError(f"No readable frames in {video_path}")
        if len(frames) >= self.frames:
            indices = np.linspace(0, len(frames) - 1, self.frames).round().astype(int)
        else:
            indices = list(range(len(frames)))
            indices += [len(frames) - 1] * (self.frames - len(frames))

        clip = []
        for frame_index in indices:
            frame = Image.fromarray(frames[int(frame_index)]).convert("RGB")
            frame = frame.resize((self.size, self.size), Image.Resampling.BICUBIC)
            clip.append(np.asarray(frame, dtype=np.float32) / 127.5 - 1.0)

        # [T,H,W,C] -> [C,T,H,W]
        video_tensor = torch.from_numpy(np.stack(clip)).permute(3, 0, 1, 2).contiguous()
        caption = caption_path.read_text(encoding="utf-8").strip() or "a video"
        return video_tensor, caption
