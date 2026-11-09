# Preparing a small video dataset

The first research run uses videos stored locally. The repository intentionally
does not download a dataset automatically.

## Folder layout

```text
data/
  videos/
    clip_001.mp4
    clip_002.mp4
  captions/
    clip_001.txt
    clip_002.txt
```

Each caption file contains a short plain-language description of its matching
video. Filenames must match exactly (apart from the extension).

## First experiment

- Start with 5–20 short clips that you own or have permission to use.
- Keep clips simple and visually consistent.
- The baseline samples 8 frames per clip and resizes them to 32 x 32 pixels.
- Use a small dataset only to verify that the pipeline runs. It is not enough
  to train a useful general-purpose video generator.

Do not use private videos or copyrighted material unless you have the necessary
rights. Keep the dataset on your own machine; do not commit large videos to Git.

## Important limitations

This baseline uses a tiny 3D convolutional denoiser and a simple hashed-word
text conditioner. It is trained from random initialization. It is a learning
experiment, not a state-of-the-art text-to-video system. Its output may look
like noise or unstable patterns until trained extensively, and even then its
quality will be very limited. A serious model needs much larger datasets,
careful evaluation, and substantial accelerator memory and compute.
