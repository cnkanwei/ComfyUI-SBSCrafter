# ComfyUI-SBSCrafter

High-quality 2D → 3D Side-by-Side (SBS) conversion for ComfyUI, implementing the
**StereoCrafter** approach ([Zhao et al., 2024](https://github.com/TencentARC/StereoCrafter)):

```
monocular depth → forward warp (disocclusion holes) → SVD diffusion inpainting
→ feathered fusion → SBS / TB / anaglyph output
```

What sets this apart from simple pixel-shift SBS nodes: disocclusions (the areas
revealed behind a foreground subject) are **hallucinated by a video diffusion
model** instead of stretch-filled, so object edges look real. A small-hole
splitter keeps diffusion away from subject interiors (faces stay pixel-faithful),
and everything is batch-chunked so hundreds of 1080p frames fit a 16 GB GPU.

## Quick start

Three nodes: `LoadImage → (any depth node) → 2D → 3D Stereo, all-in-one → SaveImage`.
Connect an **SVD Inpaint Loader** to its `svd_pipe` input for diffusion-quality
hole filling (recommended); without it, holes get a fast classical fill.
The granular nodes below are for power users who want to swap the inpainting
model, control each eye, or preview intermediates.

## Nodes

| Node | Role |
|------|------|
| **2D → 3D Stereo, all-in-one** | image + depth (+ optional `svd_pipe`) → finished stereo. Internally: depth refine → warp (`keep_original_left`, auto-convergence) → inpaint → blend → combine. |
| **Depth → Stereo Warp** | image + depth → left/right warped eyes, hole masks, warped depth. Softmax splatting (SoftSplat-style, depth-weighted bilinear). `keep_original_left/right` keeps one eye pixel-exact and halves inpaint cost — recommended. Resolution-independent strength via `disparity_percent`, `auto_convergence` for automatic zero-parallax. |
| **SVD Inpaint Loader / SVD Inpaint** | StereoCrafter's Stable Video Diffusion inpainting. Spatial tiling (128 px overlap), 23-frame / 3-overlap temporal chunking with generated-tail context, single-image mode (repeat 8 + average). `small_hole_px` fills thin interior cracks sharply at full resolution and excludes them from the diffusion mask. |
| **Stereo Blend** | composites inpainted content into the warped eye: distance feather (smoothstep), depth-adaptive radius, Reinhard-style ring color matching, temporal Gaussian smoothing of blend weights. |
| **Depth Refine** | guided filter (He et al.) that snaps depth edges to image edges — removes edge halos/ghosting before warping. |
| **Simple Hole Inpaint** | classical OpenCV Telea/NS fill, a fast zero-dependency fallback. |
| **Stereo Combine** | full/half SBS, full/half TB, cross-eye, red-cyan anaglyph. |

## Install

1. Clone into `ComfyUI/custom_nodes/`, `pip install -r requirements.txt`
   (into ComfyUI's Python).
2. Download the diffusion weights (once, ~4.3 GB total) into `ComfyUI/models/diffusers/`:
   - `svd-xt-1-1-base/` — VAE + image encoder + scheduler + feature extractor of
     [stable-video-diffusion-img2vid-xt-1-1](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1)
     (the official repo is gated; fp16 mirrors exist on the HF hub)
   - `stereocrafter-unet/` — the inpainting UNet from
     [TencentARC/StereoCrafter](https://huggingface.co/TencentARC/StereoCrafter)
3. A depth model. Recommended: Depth Anything V3
   ([ComfyUI-DepthAnythingV3](https://github.com/PozzettiAndrea/ComfyUI-DepthAnythingV3)) —
   its **DA3-Streaming** node gives temporally consistent depth for videos of any
   length with bounded VRAM.

## Workflows (`workflows/`)

Simple (all-in-one node — start here):

- `image_sbs_simple.json` — LoadImage → DA3 depth → **all-in-one** → Save.
- `video_sbs_simple.json` — same idea for video with DA3-Streaming depth,
  audio carried through (`normalize_depth` OFF — the depth is already
  normalized over the whole clip).

Advanced (granular nodes — swappable inpaint, per-eye control, previews):

- `image_sbs.json` — LoadImage → DA3 depth → Depth Refine → Warp
  (`keep_original_left`) → SVD Inpaint → Blend → Combine → Save.
- `video_sbs_da3streaming.json` — video path with DA3-Streaming depth
  (VIDEO-type plumbing via core `CreateVideo` / `GetVideoComponents`), audio
  carried through. Depth is estimated at ~540 px short side and snapped back to
  full-res edges by Depth Refine.

Video tips: keep `normalize_depth` **off** when the depth source already
normalizes the whole clip (DA3-Streaming does); `temporal_sigma 3` in the blend;
for 1000+ frame clips render in overlapping segments (RAM, not VRAM, is the
limit — all frames stay in system memory as float32).

## Quality tooling

`tests/analyze_sbs.py <full_sbs.png>` measures horizontal/vertical disparity,
foreground pop-out, cardboarding, and writes a disparity heatmap. Targets:
vertical p95 < 1 px, foreground pop-out 1.5–3 % of eye width.
`tests/test_nodes.py` is the unit suite.

## Performance (RTX 5070 Ti 16 GB, fp16)

- Image 2560 px: ~10 s/eye (only one eye needs inpainting with `keep_original_*`)
- Video 1080×1920: ~2.3 s/frame end-to-end including DA3-Streaming depth

## License — noncommercial

- Original code in this repository:
  **[PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/)**
  (see `LICENSE`) — use, modify and share freely for any noncommercial purpose;
  commercial use is not permitted.
- `svd_pipeline.py` is derived from
  [TencentARC/StereoCrafter](https://github.com/TencentARC/StereoCrafter)
  (Copyright (C) 2024 THL A29 Limited) under the StereoCrafter license
  (`third_party/StereoCrafter-LICENSE.txt`): **academic, research and education
  use only**. The StereoCrafter UNet and SVD weights carry their own licenses
  (Tencent / Stability AI).

In short: this project is free for personal, research and educational use, and
**not for commercial use** in any part.

## Credits

- [StereoCrafter](https://github.com/TencentARC/StereoCrafter) (Zhao et al.,
  2024) — the pipeline design and the SVD inpainting model this pack implements.
- [Stable Video Diffusion](https://stability.ai/) — base video model.
- [SoftSplat](https://github.com/sniklaus/softmax-splatting) (Niklaus & Liu,
  2020) — softmax splatting idea used by the forward warp.
- [Depth Anything V3](https://github.com/ByteDance-Seed/depth-anything-3) /
  [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything) —
  recommended depth sources.
- E. Reinhard et al., *Color Transfer between Images* (2001) — color matching.
- K. He et al., *Guided Image Filtering* (2010) — depth refine.
