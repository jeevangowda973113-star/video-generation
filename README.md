# MotionLab — AI Video Studio

An early, cost-conscious foundation for a video-generation platform and an original AI video-model research project.

## Current status

The repository contains:
- Responsive text-to-video and image-to-video frontend prototype
- Prompt input, image validation and local image preview
- A small 3D diffusion model architecture initialized from random weights
- Local video/caption dataset loader
- Training loop with checkpoint saving and resume support
- A basic sampler that exports a short MP4
- Dataset and hardware limitations documented before any compute spending

**Important:** The frontend is not connected to the research model yet. The training code is an educational baseline, not a commercial-quality generator. It may produce poor or noisy results, especially with a tiny dataset and limited compute.

## Our from-scratch training plan

1. Verify that the tiny model and data pipeline run locally.
2. Run a small experiment on a few videos that we own or are licensed to use.
3. Track training loss, sample quality, memory use and training time.
4. Improve the architecture and conditioning only after measuring the baseline.
5. Connect model inference to the web app after it can generate repeatable samples.
6. Estimate the real GPU and storage budget before scaling.

"From scratch" means the model weights start randomly initialized. This prototype does not use pretrained video-model weights or a paid generation API. It is not training a large commercial model; that would require much more data and accelerator compute.

## Prepare the environment

Install Python 3.10 or newer. Install a PyTorch build suitable for your operating system and hardware by following the official selector at https://pytorch.org/get-started/locally/.

Then, from the repository root:

```bash
python -m venv .venv
```

Activate the environment, then install the remaining dependencies:

```bash
pip install -r training/requirements.txt
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

## Prepare training videos

Follow [DATASET.md](DATASET.md). Each video needs a matching caption text file. Keep your dataset local; large videos and model checkpoints are ignored by Git.

## Start a tiny training experiment

```bash
python -m training.train --data data --epochs 1 --batch-size 1 --frames 8 --size 32
```

The first run saves `checkpoints/latest.pt`. Continue a run with:

```bash
python -m training.train --data data --epochs 2 --resume checkpoints/latest.pt
```

The `--epochs` value is the total target epoch count, not the number of extra epochs.

## Generate a test clip

After training:

```bash
python -m training.generate --checkpoint checkpoints/latest.pt --prompt "a red ball rolling across a table" --output generated.mp4
```

The output is a very small, low-resolution research sample. Do not expect production-quality video.

## Hardware and cost safety

- The code automatically selects CUDA, Apple MPS, or CPU when available.
- Standard Windows setups with AMD graphics commonly fall back to CPU; AMD GPU acceleration depends on a compatible ROCm/Linux setup and supported hardware.
- A CPU run can be very slow. Begin with the one-epoch smoke test before increasing resolution, clip length or dataset size.
- No paid GPU provider is configured. Do not add paid compute without an explicit budget decision.
- Training from random initialization at commercial video quality is far beyond this tiny prototype and can require substantial GPU clusters, data storage and engineering.

## Deploying the website

The current frontend is static HTML and can be imported into Vercel with no build command and no output directory. The training scripts do not run inside a static Vercel deployment. Model training and inference need a separate machine or compute service; we will not connect paid infrastructure by default.

Never expose model credentials in browser code.
