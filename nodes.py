"""
ComfyUI-SBSCrafter
Depth-based 2D -> Side-by-Side 3D conversion nodes implementing the
StereoCrafter approach (Zhao et al., 2024): monocular depth -> forward
warp (disocclusion holes) -> Stable Video Diffusion inpainting -> edge
feathered fusion -> SBS/TB output.

Pipeline:
    depth  ->  SBSC_DepthStereoWarp  ->  SBSC_SVDInpaint (or any inpaint)
           ->  SBSC_StereoBlend      ->  SBSC_StereoCombine  ->  SBS image/video
"""

import numpy as np
import torch
import torch.nn.functional as F

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


# ----------------------------------------------------------------------------
# tensor helpers  (ComfyUI IMAGE = [B,H,W,C] float 0..1 ; MASK = [B,H,W] 0..1)
# ----------------------------------------------------------------------------
def img_to_bchw(img):
    # [B,H,W,C] -> [B,C,H,W]
    return img.permute(0, 3, 1, 2).contiguous()


def bchw_to_img(t):
    # [B,C,H,W] -> [B,H,W,C]
    return t.permute(0, 2, 3, 1).contiguous()


def mask_to_b1hw(mask):
    if mask.dim() == 3:          # [B,H,W]
        return mask.unsqueeze(1)
    if mask.dim() == 4 and mask.shape[-1] == 1:  # [B,H,W,1]
        return mask.permute(0, 3, 1, 2)
    return mask


def depth_to_b1hw(depth, normalize=True):
    """Accept an IMAGE (grayscale-as-RGB) or MASK depth, return [B,1,H,W] in 0..1."""
    if depth.dim() == 4 and depth.shape[-1] in (1, 3):     # IMAGE layout
        d = depth[..., 0].unsqueeze(1)
    elif depth.dim() == 3:                                  # MASK layout
        d = depth.unsqueeze(1)
    else:
        d = depth
    d = d.float()
    if normalize:
        B = d.shape[0]
        flat = d.view(B, -1)
        lo = flat.min(dim=1)[0].view(B, 1, 1, 1)
        hi = flat.max(dim=1)[0].view(B, 1, 1, 1)
        d = (d - lo) / (hi - lo + 1e-6)
    return d.clamp(0.0, 1.0)


def _device_of(t, prefer_cuda=True):
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return t.device


def _auto_chunk(H, W, budget_px=16_000_000):
    """Frames per GPU sub-batch so B*H*W stays under a pixel budget.
    Long 1080p+ videos would otherwise put the whole batch on the GPU at once
    (150 frames x 1080p peaked at ~27GB and OOM'd a 16GB card)."""
    return max(1, int(budget_px // max(1, int(H) * int(W))))


def _chunk_ranges(total, chunk):
    for s in range(0, total, chunk):
        yield s, min(total, s + chunk)


# ----------------------------------------------------------------------------
# forward warp via softmax splatting  (bilinear + depth-weighted occlusion,
# softmax splatting (SoftSplat, Niklaus & Liu 2020))
# ----------------------------------------------------------------------------
def warp_one_eye(img_bchw, depth_b1hw, shift_x, imp_temp=8.0):
    """
    Horizontal forward-warp of image + depth by a per-pixel shift.

    Each source pixel splats bilinearly onto its two nearest destination
    columns; contributions are weighted by exp(imp_temp * depth) so nearer
    surfaces win where several pixels overlap (occlusion). Destination columns
    that receive nothing are true disocclusion holes.

    img_bchw   : [B,C,H,W]   depth_b1hw : [B,1,H,W] in 0..1
    returns warped_img [B,C,H,W], hole_mask [B,1,H,W], warped_depth [B,1,H,W]
    """
    B, C, H, W = img_bchw.shape
    dev = img_bchw.device
    xs = torch.arange(W, device=dev).view(1, 1, 1, W).float()
    tgt = xs + shift_x                                   # [B,1,H,W] float dest x
    x0 = torch.floor(tgt)
    w1 = tgt - x0
    w0 = 1.0 - w1
    x0l = x0.long()
    x1l = x0l + 1

    imp = torch.exp(imp_temp * depth_b1hw.clamp(0.0, 1.0))
    v0 = ((x0l >= 0) & (x0l < W)).float()
    v1 = ((x1l >= 0) & (x1l < W)).float()
    cw0 = w0 * imp * v0                                  # contribution weights
    cw1 = w1 * imp * v1
    x0c = x0l.clamp(0, W - 1)
    x1c = x1l.clamp(0, W - 1)

    numer = torch.zeros(B, C, H, W, device=dev, dtype=img_bchw.dtype)
    denom = torch.zeros(B, 1, H, W, device=dev, dtype=img_bchw.dtype)
    dnum = torch.zeros(B, 1, H, W, device=dev, dtype=img_bchw.dtype)
    numer.scatter_add_(3, x0c.expand(B, C, H, W), img_bchw * cw0.expand(B, C, H, W))
    numer.scatter_add_(3, x1c.expand(B, C, H, W), img_bchw * cw1.expand(B, C, H, W))
    denom.scatter_add_(3, x0c, cw0)
    denom.scatter_add_(3, x1c, cw1)
    dnum.scatter_add_(3, x0c, depth_b1hw * cw0)
    dnum.scatter_add_(3, x1c, depth_b1hw * cw1)

    eps = 1e-6
    hole = (denom < eps).float()
    warped_img = numer / (denom + eps)
    warped_depth = dnum / (denom + eps)
    return warped_img, hole, warped_depth


# ----------------------------------------------------------------------------
# Node 1 : depth -> stereo warp (+ holes)
# ----------------------------------------------------------------------------
class SBSC_DepthStereoWarp:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "depth": ("IMAGE",),
                "max_disparity_px": ("INT", {"default": 40, "min": 0, "max": 512}),
                "convergence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "synthesis": (["both_eyes", "keep_original_left", "keep_original_right"],),
                "normalize_depth": ("BOOLEAN", {"default": True}),
                "invert_depth": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                # If > 0, overrides max_disparity_px with this percent of image width
                # (resolution-independent stereo strength; comfortable ~1.5-3.0).
                "disparity_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                # If on, convergence is set per image to the median scene depth
                # (mid-scene sits on screen; nearer pops out, farther recedes).
                "auto_convergence": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "MASK", "IMAGE", "IMAGE")
    RETURN_NAMES = ("left_warped", "left_hole", "right_warped", "right_hole",
                    "left_wdepth", "right_wdepth")
    FUNCTION = "warp"
    CATEGORY = "SBSCrafter"

    def warp(self, image, depth, max_disparity_px, convergence, synthesis,
             normalize_depth, invert_depth, disparity_percent=0.0,
             auto_convergence=False):
        out_dev = image.device
        dev = _device_of(image)
        img_all = img_to_bchw(image)                       # stays on out_dev (CPU)
        d_all = depth_to_b1hw(depth, normalize=normalize_depth)
        if invert_depth:
            d_all = 1.0 - d_all
        # match depth to image resolution
        if d_all.shape[-2:] != img_all.shape[-2:]:
            d_all = F.interpolate(d_all, size=img_all.shape[-2:], mode="nearest")

        if disparity_percent and disparity_percent > 0:
            disp = float(img_all.shape[3]) * disparity_percent / 100.0
        else:
            disp = float(max_disparity_px)
        if auto_convergence:
            # zero-parallax plane at the median scene depth. ONE value for the whole
            # batch: per-frame medians would make convergence jitter across video frames.
            convergence = float(d_all.median())

        B, _, H, W = img_all.shape
        outs = ([], [], [], [], [], [])
        for s, e in _chunk_ranges(B, _auto_chunk(H, W)):
            img = img_all[s:e].to(dev)
            d = d_all[s:e].to(dev)
            # signed shift; foreground (large depth) moves most,
            # convergence = zero-parallax plane
            base = (d - convergence)

            def zero_eye(x):  # original as this eye: no warp, no holes
                return x, torch.zeros_like(d), d.clone()

            if synthesis == "keep_original_left":
                lw, lh, ld = zero_eye(img)
                rw, rh, rd = warp_one_eye(img, d, -disp * base)
            elif synthesis == "keep_original_right":
                lw, lh, ld = warp_one_eye(img, d, +disp * base)
                rw, rh, rd = zero_eye(img)
            else:  # both_eyes, each half disparity in opposite directions
                lw, lh, ld = warp_one_eye(img, d, +0.5 * disp * base)
                rw, rh, rd = warp_one_eye(img, d, -0.5 * disp * base)

            for lst, t in zip(outs, (lw, lh, rw, rh, ld, rd)):
                lst.append(t.to(out_dev))

        lw, lh, rw, rh, ld, rd = (torch.cat(l, dim=0) for l in outs)
        return (bchw_to_img(lw), lh.squeeze(1),
                bchw_to_img(rw), rh.squeeze(1),
                bchw_to_img(ld.repeat(1, 3, 1, 1)),
                bchw_to_img(rd.repeat(1, 3, 1, 1)))


# ----------------------------------------------------------------------------
# fusion: composite inpainted content into the warped frame with a soft,
# depth-aware transition around the disocclusion holes.
#
# Built from textbook pieces:
#   * distance-to-hole feathering with a smoothstep falloff
#   * feather radius modulated by warped depth (nearer content -> wider feather)
#   * Reinhard-style per-channel color matching (Reinhard et al. 2001) of the
#     inpainted content against a ring of clean pixels around the holes
#   * optional temporal smoothing of the blend weights (1-D Gaussian conv along
#     the frame axis) so hole boundaries do not shimmer in video
# ----------------------------------------------------------------------------
def _dist_to_hole(hole_b1hw):
    """Per-pixel Euclidean distance (px) to the nearest hole pixel."""
    B = hole_b1hw.shape[0]
    out = torch.zeros_like(hole_b1hw)
    m = hole_b1hw.detach().cpu().numpy()
    for b in range(B):
        not_hole = (m[b, 0] <= 0.5).astype(np.uint8)
        if _HAS_CV2:
            d = cv2.distanceTransform(not_hole * 255, cv2.DIST_L2, 5)
        else:
            from scipy import ndimage
            d = ndimage.distance_transform_edt(not_hole > 0).astype(np.float32)
        out[b, 0] = torch.from_numpy(np.asarray(d, np.float32)).to(hole_b1hw.device)
    return out


def _smoothstep(t):
    t = t.clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _ring_color_match(clean_bchw, inpaint_bchw, hole_b1hw, ring_px=24):
    """Shift the inpainted frame's per-channel mean/std toward the clean frame.
    Statistics come from a ring of valid pixels around the holes — the region
    the inpainted content has to seamlessly continue into."""
    B = clean_bchw.shape[0]
    k = 2 * ring_px + 1
    ring = (F.max_pool2d(hole_b1hw, k, stride=1, padding=ring_px) - hole_b1hw).clamp(0, 1)
    out = inpaint_bchw.clone()
    for b in range(B):
        sel = ring[b, 0] > 0.5
        if int(sel.sum()) < 64:
            continue
        for c in range(clean_bchw.shape[1]):
            ref = clean_bchw[b, c][sel]
            src = inpaint_bchw[b, c][sel]
            s_std = src.std()
            if float(s_std) < 1e-5:
                continue
            gain = (ref.std() / (s_std + 1e-8)).clamp(0.5, 2.0)
            out[b, c] = (inpaint_bchw[b, c] - src.mean()) * gain + ref.mean()
    return out.clamp(0.0, 1.0)


def _temporal_gauss_smooth(w_b1hw, sigma):
    """Smooth blend weights along the frame axis with a normalized Gaussian."""
    B = w_b1hw.shape[0]
    if B == 1 or sigma <= 0:
        return w_b1hw
    r = max(1, int(round(3 * sigma)))
    x = torch.arange(-r, r + 1, dtype=torch.float32, device=w_b1hw.device)
    kern = torch.exp(-0.5 * (x / sigma) ** 2)
    kern = (kern / kern.sum()).view(1, 1, -1)
    flat = w_b1hw.permute(1, 2, 3, 0).reshape(-1, 1, B)      # [H*W, 1, B]
    flat = F.pad(flat, (r, r), mode="replicate")
    sm = F.conv1d(flat, kern, padding=0)                     # [H*W, 1, B]
    return sm.reshape(*w_b1hw.shape[1:], B).permute(3, 0, 1, 2).contiguous()


# ----------------------------------------------------------------------------
# Node 2 : fusion node
# ----------------------------------------------------------------------------
class SBSC_StereoBlend:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "warped_image": ("IMAGE",),
                "inpainted_image": ("IMAGE",),
                "hole_mask": ("MASK",),
                "feather_px": ("INT", {"default": 24, "min": 1, "max": 256}),
                "depth_adaptive": ("BOOLEAN", {"default": True}),
                "temporal_sigma": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 8.0, "step": 0.5}),
                "color_match": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "warped_depth": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("blended",)
    FUNCTION = "blend"
    CATEGORY = "SBSCrafter"

    def blend(self, warped_image, inpainted_image, hole_mask, feather_px,
              depth_adaptive, temporal_sigma, color_match, warped_depth=None):
        out_dev = warped_image.device
        dev = _device_of(warped_image)
        warped_all = img_to_bchw(warped_image)
        inpaint_all = img_to_bchw(inpainted_image)
        hole_all = mask_to_b1hw(hole_mask).float()

        B, _, H, W = warped_all.shape
        # the temporal kernel reaches round(3*sigma) frames each way; carry that
        # margin around each sub-batch so results match a full-batch pass exactly
        t_radius = max(1, int(round(3 * temporal_sigma))) if (B > 1 and temporal_sigma > 0) else 0

        outs = []
        for s, e in _chunk_ranges(B, _auto_chunk(H, W)):
            s2, e2 = max(0, s - t_radius), min(B, e + t_radius)
            warped = warped_all[s2:e2].to(dev)
            inpaint = inpaint_all[s2:e2].to(dev)
            hole = hole_all[s2:e2].to(dev)

            if inpaint.shape[-2:] != warped.shape[-2:]:
                inpaint = F.interpolate(inpaint, size=warped.shape[-2:], mode="bilinear", align_corners=False)
            if hole.shape[-2:] != warped.shape[-2:]:
                hole = F.interpolate(hole, size=warped.shape[-2:], mode="nearest")

            radius = torch.full_like(hole, float(feather_px))
            if depth_adaptive and warped_depth is not None:
                wd = depth_to_b1hw(warped_depth[s2:e2], normalize=False).to(dev)
                if wd.shape[-2:] != warped.shape[-2:]:
                    wd = F.interpolate(wd, size=warped.shape[-2:], mode="nearest")
                # nearer content (larger depth value) gets a wider feather
                radius = radius * (0.5 + wd.clamp(0.0, 1.0))

            dist = _dist_to_hole(hole)
            w = _smoothstep(1.0 - dist / (radius + 1e-6))
            w = _temporal_gauss_smooth(w, temporal_sigma)
            # holes always take the inpainted content in full
            w = torch.where(hole > 0.5, torch.ones_like(w), w)

            if color_match:
                inpaint = _ring_color_match(warped, inpaint, hole)

            blended = warped * (1.0 - w) + inpaint * w
            keep0 = s - s2
            outs.append(blended[keep0:keep0 + (e - s)].clamp(0.0, 1.0).to(out_dev))

        return (bchw_to_img(torch.cat(outs, dim=0)),)


# ----------------------------------------------------------------------------
# Node 3 : combine L/R into a stereo layout
# ----------------------------------------------------------------------------
class SBSC_StereoCombine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "left": ("IMAGE",),
                "right": ("IMAGE",),
                "layout": (["full_sbs", "half_sbs", "full_tb", "half_tb",
                            "anaglyph_rc", "cross_eye"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("stereo",)
    FUNCTION = "combine"
    CATEGORY = "SBSCrafter"

    def combine(self, left, right, layout):
        l = img_to_bchw(left)
        r = img_to_bchw(right)
        if r.shape != l.shape:
            r = F.interpolate(r, size=l.shape[-2:], mode="bilinear", align_corners=False)
        B, C, H, W = l.shape

        if layout == "cross_eye":            # swap eyes, full width
            out = torch.cat([r, l], dim=3)
        elif layout == "full_sbs":
            out = torch.cat([l, r], dim=3)
        elif layout == "half_sbs":
            lh = F.interpolate(l, size=(H, W // 2), mode="bilinear", align_corners=False)
            rh = F.interpolate(r, size=(H, W // 2), mode="bilinear", align_corners=False)
            out = torch.cat([lh, rh], dim=3)
        elif layout == "full_tb":
            out = torch.cat([l, r], dim=2)
        elif layout == "half_tb":
            lh = F.interpolate(l, size=(H // 2, W), mode="bilinear", align_corners=False)
            rh = F.interpolate(r, size=(H // 2, W), mode="bilinear", align_corners=False)
            out = torch.cat([lh, rh], dim=2)
        elif layout == "anaglyph_rc":        # red-cyan
            out = torch.stack([l[:, 0], r[:, 1], r[:, 2]], dim=1)
        else:
            out = torch.cat([l, r], dim=3)
        return (bchw_to_img(out.clamp(0.0, 1.0)),)


# ----------------------------------------------------------------------------
# Depth refine: guided filter — snaps depth edges to image edges, removing the
# halo/ghost band where the depth silhouette doesn't match the RGB silhouette.
# ----------------------------------------------------------------------------
def _box_filter(x, r):
    """Separable box mean with reflect padding. x: [B,C,H,W], radius r."""
    k = 2 * r + 1
    x = F.pad(x, (r, r, r, r), mode="reflect")
    x = F.avg_pool2d(x, kernel_size=(k, 1), stride=1)
    x = F.avg_pool2d(x, kernel_size=(1, k), stride=1)
    return x


def guided_filter(guide, src, radius, eps):
    """Classic guided filter (He et al.). guide/src: [B,1,H,W] float."""
    mean_I = _box_filter(guide, radius)
    mean_p = _box_filter(src, radius)
    mean_Ip = _box_filter(guide * src, radius)
    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = _box_filter(guide * guide, radius) - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = _box_filter(a, radius)
    mean_b = _box_filter(b, radius)
    return mean_a * guide + mean_b


class SBSC_DepthRefine:
    """Edge-aware depth cleanup before warping:
    guided-filters the depth with the RGB image as guide so depth silhouettes
    snap to image silhouettes (fixes halo/ghosting at object edges)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "depth": ("IMAGE",),
                "radius": ("INT", {"default": 8, "min": 1, "max": 64}),
                "eps": ("FLOAT", {"default": 0.0001, "min": 0.000001, "max": 0.1, "step": 0.0001}),
                "iterations": ("INT", {"default": 2, "min": 1, "max": 5}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("refined_depth",)
    FUNCTION = "refine"
    CATEGORY = "SBSCrafter"

    def refine(self, image, depth, radius, eps, iterations):
        out_dev = depth.device
        dev = _device_of(image)
        img_all = img_to_bchw(image).float()
        d_all = depth_to_b1hw(depth, normalize=False).float()
        if d_all.shape[-2:] != img_all.shape[-2:]:
            d_all = F.interpolate(d_all, size=img_all.shape[-2:], mode="bilinear",
                                  align_corners=False)
        B, _, H, W = img_all.shape
        outs = []
        for s, e in _chunk_ranges(B, _auto_chunk(H, W)):
            img = img_all[s:e].to(dev)
            d = d_all[s:e].to(dev)
            # luminance guide
            guide = (0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3])
            for _ in range(iterations):
                d = guided_filter(guide, d, radius, eps)
            outs.append(d.clamp(0.0, 1.0).to(out_dev))
        d = torch.cat(outs, dim=0)
        return (bchw_to_img(d.repeat(1, 3, 1, 1)),)


# ----------------------------------------------------------------------------
# SVD / StereoCrafter diffusion inpaint (the StereoCrafter hole filler)
# ----------------------------------------------------------------------------
def _blend_h(a, b, overlap_size):
    weight_b = (torch.arange(overlap_size).view(1, 1, 1, -1) / overlap_size).to(b.device)
    b[:, :, :, :overlap_size] = (1 - weight_b) * a[:, :, :, -overlap_size:] + weight_b * b[:, :, :, :overlap_size]
    return b


def _blend_v(a, b, overlap_size):
    weight_b = (torch.arange(overlap_size).view(1, 1, -1, 1) / overlap_size).to(b.device)
    b[:, :, :overlap_size, :] = (1 - weight_b) * a[:, :, -overlap_size:, :] + weight_b * b[:, :, :overlap_size, :]
    return b


def _spatial_tiled_process(cond_frames, mask_frames, process_func, tile_num,
                           spatial_n_compress=8, **kargs):
    """StereoCrafter's overlapping-tile SVD process (128px overlap, latent-space blending)."""
    height, width = cond_frames.shape[2], cond_frames.shape[3]
    tile_overlap = (128, 128)
    tile_size = (int((height + tile_overlap[0] * (tile_num - 1)) / tile_num),
                 int((width + tile_overlap[1] * (tile_num - 1)) / tile_num))
    tile_stride = ((tile_size[0] - tile_overlap[0]), (tile_size[1] - tile_overlap[1]))

    cols = []
    for i in range(tile_num):
        rows = []
        for j in range(tile_num):
            cond_tile = cond_frames[:, :, i*tile_stride[0]:i*tile_stride[0]+tile_size[0],
                                    j*tile_stride[1]:j*tile_stride[1]+tile_size[1]]
            mask_tile = mask_frames[:, :, i*tile_stride[0]:i*tile_stride[0]+tile_size[0],
                                    j*tile_stride[1]:j*tile_stride[1]+tile_size[1]]
            tile = process_func(
                frames=cond_tile, frames_mask=mask_tile,
                height=cond_tile.shape[2], width=cond_tile.shape[3],
                num_frames=len(cond_tile), output_type="latent", **kargs,
            ).frames[0]
            rows.append(tile)
        cols.append(rows)

    latent_stride = (tile_stride[0] // spatial_n_compress, tile_stride[1] // spatial_n_compress)
    latent_overlap = (tile_overlap[0] // spatial_n_compress, tile_overlap[1] // spatial_n_compress)

    results_cols = []
    for i, rows in enumerate(cols):
        results_rows = []
        for j, tile in enumerate(rows):
            if i > 0:
                tile = _blend_v(cols[i-1][j], tile, latent_overlap[0])
            if j > 0:
                tile = _blend_h(rows[j-1], tile, latent_overlap[1])
            results_rows.append(tile)
        results_cols.append(results_rows)

    pixels = []
    for i, rows in enumerate(results_cols):
        for j, tile in enumerate(rows):
            if i < len(results_cols) - 1:
                tile = tile[:, :, :latent_stride[0], :]
            if j < len(rows) - 1:
                tile = tile[:, :, :, :latent_stride[1]]
            rows[j] = tile
        pixels.append(torch.cat(rows, dim=3))
    return torch.cat(pixels, dim=2)


class SBSC_SVDInpaintLoader:
    """Loads the StereoCrafter SVD inpainting pipeline from
    models/diffusers/svd-xt-1-1-base (vae+image_encoder+scheduler+feature_extractor)
    and models/diffusers/stereocrafter-unet (the inpainting UNet)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cpu_offload": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("SBSC_SVD_PIPE",)
    RETURN_NAMES = ("svd_pipe",)
    FUNCTION = "load"
    CATEGORY = "SBSCrafter"

    def load(self, cpu_offload):
        import os
        import folder_paths
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
        from diffusers.models import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
        from diffusers.schedulers import EulerDiscreteScheduler
        from .svd_pipeline import StableVideoDiffusionInpaintingPipeline

        base = os.path.join(folder_paths.models_dir, "diffusers", "svd-xt-1-1-base")
        unet_dir = os.path.join(folder_paths.models_dir, "diffusers", "stereocrafter-unet")
        for p in (base, unet_dir):
            if not os.path.isdir(p):
                raise RuntimeError(f"Missing model folder: {p} (see README for download instructions)")

        dtype = torch.float16
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            base, subfolder="image_encoder", variant="fp16", torch_dtype=dtype)
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            base, subfolder="vae", variant="fp16", torch_dtype=dtype)
        unet = UNetSpatioTemporalConditionModel.from_pretrained(unet_dir, torch_dtype=dtype)
        scheduler = EulerDiscreteScheduler.from_pretrained(base, subfolder="scheduler")
        feature_extractor = CLIPImageProcessor.from_pretrained(base, subfolder="feature_extractor")

        for m in (image_encoder, vae, unet):
            m.requires_grad_(False)

        pipe = StableVideoDiffusionInpaintingPipeline(
            vae=vae, image_encoder=image_encoder, unet=unet,
            scheduler=scheduler, feature_extractor=feature_extractor)
        if cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to("cuda")
        return (pipe,)


class SBSC_SVDInpaint:
    """StereoCrafter-style hole filling: runs SVD inpainting on the warped frames
    at reduced resolution (cap long side, snap to /128 grid, optional tiling),
    then upscales back. For a single image it repeats it as 8 frames and averages
    (StereoCrafter's single-image mode)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "svd_pipe": ("SBSC_SVD_PIPE",),
                "warped_image": ("IMAGE",),
                "hole_mask": ("MASK",),
                "resolution_limit": ("INT", {"default": 1024, "min": 256, "max": 1536, "step": 128}),
                "tile_num": ("INT", {"default": 1, "min": 1, "max": 3}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 50}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**31}),
            },
            "optional": {
                # Holes narrower than this (px) are interior stretch-cracks, not real
                # disocclusions: they are filled sharply at FULL resolution (Telea)
                # and excluded from the SVD mask, so diffusion content never lands on
                # subject interiors (faces!). 0 disables the split.
                "small_hole_px": ("INT", {"default": 8, "min": 0, "max": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("inpainted", "effective_mask")
    FUNCTION = "run"
    CATEGORY = "SBSCrafter"

    def run(self, svd_pipe, warped_image, hole_mask, resolution_limit, tile_num, steps, seed,
            small_hole_px=8):
        out_dev = warped_image.device
        frames = img_to_bchw(warped_image).float()          # (f,c,H,W) 0..1
        mask = mask_to_b1hw(hole_mask).float()              # (f,1,H,W)
        f, c, H, W = frames.shape

        # --- split holes: thin interior cracks vs wide disocclusion bands ---
        if small_hole_px > 0 and _HAS_CV2:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (small_hole_px, small_hole_px))
            wide_np = np.empty((f, H, W), np.float32)
            frames_np = frames.cpu().numpy()
            mask_np = mask.cpu().numpy()[:, 0]
            for i in range(f):
                m8 = (mask_np[i] > 0.5).astype(np.uint8) * 255
                wide = cv2.morphologyEx(m8, cv2.MORPH_OPEN, k)
                wide_np[i] = wide.astype(np.float32) / 255.0
                # sharp full-res fill for ALL holes (base layer); SVD later
                # overwrites only the wide bands
                a = (np.clip(frames_np[i].transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
                filled = cv2.inpaint(a, (m8 > 0).astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
                frames_np[i] = filled.astype(np.float32).transpose(2, 0, 1) / 255.0
            frames = torch.from_numpy(frames_np)
            mask = torch.from_numpy(wide_np).unsqueeze(1)   # SVD sees wide bands only

        eff_mask = mask[:, 0].clone()                       # what the blend should treat as holes
        if float(mask.sum()) == 0:
            # no wide disocclusion bands at all — the sharp full-res fill is final
            return (bchw_to_img(frames.clamp(0.0, 1.0)).to(out_dev), eff_mask.to(out_dev))

        # working size: cap the long side, snap to /128
        scale = min(1.0, resolution_limit / max(H, W))
        th = max(256, int(H * scale) // 128 * 128)
        tw = max(256, int(W * scale) // 128 * 128)

        frames_lr = F.interpolate(frames, size=(th, tw), mode="bilinear", align_corners=False)
        # preserve holes when downscaling: any coverage > 5% counts as hole
        mask_lr = (F.interpolate(mask, size=(th, tw), mode="bilinear", align_corners=False) > 0.05).float()

        # CPU generator works for tensors on any device in diffusers' randn_tensor
        generator = torch.Generator(device="cpu").manual_seed(seed)
        # keep float32: the pipeline's VAE handles its own dtype (force_upcast etc.)
        frames_lr = frames_lr.float()
        mask_lr = mask_lr.float()

        from .svd_pipeline import tensor2vid

        def run_svd(cf, cm):
            latents = _spatial_tiled_process(
                cf, cm, svd_pipe, tile_num,
                spatial_n_compress=8,
                min_guidance_scale=1.01, max_guidance_scale=1.01,
                decode_chunk_size=2, fps=7, motion_bucket_id=127,
                noise_aug_strength=0.0, num_inference_steps=steps,
                generator=generator,
            ).unsqueeze(0)
            if latents.dtype == torch.float16:
                svd_pipe.vae.to(dtype=torch.float16)
            vid = svd_pipe.decode_latents(latents, num_frames=latents.shape[1], decode_chunk_size=2)
            return tensor2vid(vid, svd_pipe.image_processor, output_type="pt")[0].cpu()  # (f,c,h,w)

        infer_for_image = f == 1
        if infer_for_image:
            # single-image mode: duplicate to 8 frames, average the outputs
            generated = run_svd(frames_lr.repeat(8, 1, 1, 1),
                                mask_lr.repeat(8, 1, 1, 1)).mean(dim=0, keepdim=True)
        elif frames_lr.shape[0] <= 23:
            generated = run_svd(frames_lr, mask_lr)
        else:
            # video mode: 23-frame chunks, 3-frame overlap (StereoCrafter scheme). The previous
            # chunk's GENERATED tail frames (with their original masks) lead the
            # next chunk as temporal context, then get trimmed from its output.
            CHUNK, OVERLAP = 23, 3
            outs = []
            cached_gen = None
            cached_mask = None
            pos = 0
            while pos < frames_lr.shape[0]:
                take = CHUNK if cached_gen is None else CHUNK - OVERLAP
                cur_f = frames_lr[pos:pos + take]
                cur_m = mask_lr[pos:pos + take]
                if cached_gen is None:
                    in_f, in_m, trim = cur_f, cur_m, 0
                else:
                    in_f = torch.cat([cached_gen[-OVERLAP:], cur_f], dim=0)
                    in_m = torch.cat([cached_mask[-OVERLAP:], cur_m], dim=0)
                    trim = OVERLAP
                gen = run_svd(in_f, in_m)
                cached_gen, cached_mask = gen, in_m
                outs.append(gen[trim:])
                pos += take
            generated = torch.cat(outs, dim=0)
            assert generated.shape[0] == frames_lr.shape[0], \
                f"chunking mismatch: {generated.shape[0]} vs {frames_lr.shape[0]}"

        svd_up = F.interpolate(generated.float().cpu(), size=(H, W), mode="bilinear", align_corners=False)

        # composite: SVD content only inside the (feathered) wide bands; everywhere
        # else keep the sharp full-res base — diffusion never touches the subject.
        feather = F.max_pool2d(mask, kernel_size=9, stride=1, padding=4)   # dilate 4px
        feather = _box_filter(feather, 3).clamp(0.0, 1.0)                  # soften edge
        out = svd_up * feather + frames * (1.0 - feather)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return (bchw_to_img(out.clamp(0.0, 1.0)).to(out_dev), eff_mask.to(out_dev))


# ----------------------------------------------------------------------------
# Node 4 : quick classical hole-fill (zero-dependency inpaint for the holes)
#          Use this to get an end-to-end result now; swap for a diffusion
#          inpaint node when you want maximum quality on large disocclusions.
# ----------------------------------------------------------------------------
class SBSC_SimpleInpaint:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "warped_image": ("IMAGE",),
                "hole_mask": ("MASK",),
                "method": (["telea", "ns"],),
                "radius": ("INT", {"default": 4, "min": 1, "max": 32}),
                "dilate": ("INT", {"default": 2, "min": 0, "max": 16}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("inpainted",)
    FUNCTION = "run"
    CATEGORY = "SBSCrafter"

    def run(self, warped_image, hole_mask, method, radius, dilate):
        if not _HAS_CV2:
            raise RuntimeError("SBSC_SimpleInpaint needs opencv (cv2).")
        flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
        img = warped_image.detach().cpu().numpy()
        hole = mask_to_b1hw(hole_mask).detach().cpu().numpy()[:, 0]
        B = img.shape[0]
        out = np.empty_like(img)
        for b in range(B):
            a = (np.clip(img[b], 0, 1) * 255).astype(np.uint8)
            m = (hole[b] > 0.5).astype(np.uint8) * 255
            if dilate > 0:
                k = np.ones((dilate * 2 + 1, dilate * 2 + 1), np.uint8)
                m = cv2.dilate(m, k)
            res = cv2.inpaint(a, m, radius, flag)
            out[b] = res.astype(np.float32) / 255.0
        return (torch.from_numpy(out).to(warped_image.device),)


NODE_CLASS_MAPPINGS = {
    "SBSC_DepthStereoWarp": SBSC_DepthStereoWarp,
    "SBSC_DepthRefine": SBSC_DepthRefine,
    "SBSC_SimpleInpaint": SBSC_SimpleInpaint,
    "SBSC_SVDInpaintLoader": SBSC_SVDInpaintLoader,
    "SBSC_SVDInpaint": SBSC_SVDInpaint,
    "SBSC_StereoBlend": SBSC_StereoBlend,
    "SBSC_StereoCombine": SBSC_StereoCombine,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SBSC_DepthStereoWarp": "Depth → Stereo Warp (SBSCrafter)",
    "SBSC_DepthRefine": "Depth Refine, edge-aware (SBSCrafter)",
    "SBSC_SimpleInpaint": "Simple Hole Inpaint (SBSCrafter)",
    "SBSC_SVDInpaintLoader": "SVD Inpaint Loader (SBSCrafter)",
    "SBSC_SVDInpaint": "SVD Inpaint (SBSCrafter)",
    "SBSC_StereoBlend": "Stereo Blend (SBSCrafter)",
    "SBSC_StereoCombine": "Stereo Combine SBS/TB (SBSCrafter)",
}
