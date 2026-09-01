"""Unit tests for SBSCrafter nodes. Run:
  python tests/test_nodes.py
Verifies warp holes, blend fusion (holes==inpaint), temporal-margin chunking
equivalence, and combine layouts.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import nodes as N

torch.manual_seed(0)
B, H, W = 2, 64, 96
img = torch.rand(B, H, W, 3) * 0.2
img[:, 20:44, 30:60, :] = 0.9
depth = torch.zeros(B, H, W, 3)
depth[:, 20:44, 30:60, :] = 1.0
depth += 0.1

warp = N.SBSC_DepthStereoWarp()
lw, lh, rw, rh, ld, rd = warp.warp(img, depth, 30, 0.3, "both_eyes", True, False)
assert lw.shape == img.shape and lh.shape == (B, H, W)
assert 0.0 < float(lh.mean()) < 0.3, f"hole frac {float(lh.mean())}"
print("warp ok, hole_frac", round(float(lh.mean()), 4))

# keep_original modes: kept eye must equal the input exactly, zero holes
for syn, kept_img, kept_hole in [("keep_original_left", 0, 1), ("keep_original_right", 2, 3)]:
    outs = warp.warp(img, depth, 30, 0.3, syn, True, False)
    assert torch.allclose(outs[kept_img], img), syn
    assert float(outs[kept_hole].sum()) == 0.0, syn
print("keep_original ok (kept eye exact, zero holes)")

inpaint = lw.clone().permute(0, 3, 1, 2)
inpaint = torch.where(lh.unsqueeze(1) > 0.5, torch.full_like(inpaint, 0.5), inpaint).permute(0, 2, 3, 1)
blend = N.SBSC_StereoBlend()
(out,) = blend.blend(lw, inpaint, lh, 24, True, 0.0, True, warped_depth=ld)
assert out.shape == img.shape and not torch.isnan(out).any()
hb = lh.unsqueeze(-1) > 0.5
assert float((out[hb.expand_as(out)] - inpaint[hb.expand_as(out)]).abs().max()) < 1e-3
print("blend ok (holes == inpaint)")

# chunked == unchunked (with temporal smoothing active)
BB = 30
wi = torch.rand(BB, 48, 64, 3); ii = torch.rand(BB, 48, 64, 3)
hm = (torch.rand(BB, 48, 64) > 0.9).float()
wd = torch.rand(BB, 48, 64, 3)
orig = N._auto_chunk
try:
    N._auto_chunk = lambda H, W, budget_px=16_000_000: BB
    (ref,) = blend.blend(wi, ii, hm, 24, True, 3.0, True, warped_depth=wd)
    N._auto_chunk = lambda H, W, budget_px=16_000_000: 5
    (chk,) = blend.blend(wi, ii, hm, 24, True, 3.0, True, warped_depth=wd)
finally:
    N._auto_chunk = orig
assert torch.allclose(ref, chk, atol=1e-5), float((ref - chk).abs().max())
print("blend chunked == unchunked ok")

# all-in-one node, Telea fallback path (no SVD models needed)
conv = N.SBSC_Convert()
(sbs,) = conv.convert(img, depth, 2.5, "keep_original_left", "full_sbs",
                      True, False, True, 0.0, 42)
assert sbs.shape == (B, H, W * 2, 3) and not torch.isnan(sbs).any()
assert torch.allclose(sbs[:, :, :W, :], img)   # left eye = untouched original
print("convert (all-in-one) ok, left eye exact")

# particle depth fix: bright dots over a far background get background depth
pimg = torch.full((1, 128, 160, 3), 0.3)
pd = torch.full((1, 128, 160, 3), 0.2)
for (y, x) in [(20, 30), (60, 100), (100, 50)]:
    pimg[0, y:y+3, x:x+3, :] = 1.0     # bright 3px flakes
    pd[0, y-6:y+9, x-6:x+9, :] = 0.9   # depth model paints a big near blob
pfx = N.SBSC_ParticleDepthFix()
fixed, pmask = pfx.fix(pimg, pd, 4, 0.2, "bright")
assert pmask.sum() > 0, "no particles detected"
sel = pmask[0] > 0.5
assert float(fixed[0, ..., 0][sel].mean()) < 0.4, "particle depth not pushed to background"
protect = torch.ones(1, 128, 160)      # protect everything -> nothing changes
fixed2, pmask2 = pfx.fix(pimg, pd, 4, 0.2, "bright", protect_mask=protect)
assert float(pmask2.sum()) == 0 and torch.allclose(fixed2[..., 0], pd[..., 0])
print("particle depth fix ok (detection, bg push, protect_mask)")

comb = N.SBSC_StereoCombine()
for lay in ["full_sbs", "half_sbs", "full_tb", "half_tb", "anaglyph_rc", "cross_eye"]:
    (s,) = comb.combine(lw, rw, lay)
    assert s.shape[0] == B
print("combine ok")
print("\nALL TESTS PASSED")
