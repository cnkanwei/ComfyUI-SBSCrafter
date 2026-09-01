"""Quality-regression tool: measure the real stereo in an SBS output.
Run:  python_embeded\python.exe ...\tests\analyze_sbs.py <path_to_full_sbs.png>

Reports, via optical flow between the two eyes:
  - foreground vs background horizontal disparity (sign = pop-out/recede)
  - subject internal depth range (cardboard check)
  - vertical disparity (should be ~0 for comfort)
and writes <name>_disp.png (left eye | disparity heatmap).

Use it as a before/after metric when changing depth model, disparity, or inpaint.
Good targets:  |vertical| p95 < ~1px ; foreground pop-out ~1.5-3% of eye width ;
subject-internal range clearly > 0 (not a flat card) ; fg and bg different sign/level.
"""
import cv2, numpy as np, sys, os

path = sys.argv[1]
img = cv2.imread(path)
H, W = img.shape[:2]
half = W // 2
L, R = img[:, :half], img[:, half:half*2]
print(f"file: {os.path.basename(path)}  full {W}x{H}  each eye {half}x{H}")

Lg = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY)
Rg = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY)
flow = cv2.calcOpticalFlowFarneback(Lg, Rg, None, 0.5, 4, 41, 5, 7, 1.5, 0)
dx, dy = flow[..., 0], flow[..., 1]

dark = Lg < 110
gx = cv2.Sobel(Lg, cv2.CV_32F, 1, 0, 3); gy = cv2.Sobel(Lg, cv2.CV_32F, 0, 1, 3)
edge = np.sqrt(gx*gx + gy*gy)
fg = cv2.medianBlur(((dark | (edge > np.percentile(edge, 92))).astype(np.uint8))*255, 9) > 0
bg = ~cv2.dilate(fg.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)

def stats(name, a):
    print(f"  {name}: median {np.median(a):+.2f}px  p5..p95 {np.percentile(a,5):+.2f}..{np.percentile(a,95):+.2f}  std {a.std():.2f}")

print("horizontal disparity (left-eye -> right-eye, +right):")
stats("foreground", dx[fg]); stats("background", dx[bg])
print(f"  fg-vs-bg separation: {np.median(dx[fg]) - np.median(dx[bg]):+.2f}px  ({100*abs(np.median(dx[fg])-np.median(dx[bg]))/half:.2f}% of eye width)")
print(f"  subject internal depth range: {np.percentile(dx[fg],95)-np.percentile(dx[fg],5):.2f}px")
print(f"  vertical disparity (want ~0): median {np.median(dy):+.3f}  p95|dy| {np.percentile(np.abs(dy),95):.2f}px")

d = dx.copy(); lo, hi = np.percentile(d, 2), np.percentile(d, 98)
heat = cv2.applyColorMap((np.clip((d-lo)/(hi-lo+1e-6), 0, 1)*255).astype(np.uint8), cv2.COLORMAP_JET)
heat[fg] = (heat[fg]*0.7 + 255*0.3).astype(np.uint8)
outp = path.replace(".png", "_disp.png")
cv2.imwrite(outp, np.hstack([L, heat]))
print("saved:", outp)
