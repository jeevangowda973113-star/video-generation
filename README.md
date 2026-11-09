# MotionLab — AI Video Studio

An early, cost-conscious foundation for an open-source AI video generation platform.

## Current status

This commit contains a responsive frontend prototype:
- Text-to-video and image-to-video request forms
- Prompt input, image validation and local image preview
- Aspect ratio, duration and camera-motion controls
- Session-only activity list
- Clear notice that generation is not connected yet

**Important:** This version does not generate AI videos. It does not call a model API, start a GPU, or create paid compute usage.

## Run locally

Open `index.html` in a browser, or serve the repository with any static web server.

## Deploy

The project is static HTML and can be imported into Vercel with no build command and no output directory. It can also be hosted on GitHub Pages.

## Planned architecture

1. Keep the frontend static and inexpensive.
2. Add a backend job queue with request limits and explicit cost controls.
3. Integrate an open-source video model only after testing its hardware requirements, license, and actual per-video compute cost.
4. Add temporary object storage and automatic cleanup before public user uploads are enabled.

Never expose model credentials in browser code. Do not enable paid GPU providers without explicit budget approval.
