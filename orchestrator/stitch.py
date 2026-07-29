"""
stitch.py -- Vertical 1x2 frame stitcher for the VandeBharat camera rig.

Two IDS GV-5040 area-scan frames (native 1456 x 1088 px) are mounted one
ABOVE the other and share a 15% vertical overlap band. This script fuses
the two frames into a single tall composite (~1456 x 2013 px) that is then
fed to the YOLO component/defect detector.

WHY OVERLAP + WHICH METHOD
--------------------------
An overlap band is what makes a seamless join possible: the same strip of
the scene is seen by BOTH cameras, so we can (a) align them and (b) cross-
fade one into the other so no hard seam line cuts through a defect.

Two ways to stitch:
  1. CALIBRATED (default here) -- cameras are rigidly mounted, so the
     overlap is a KNOWN, FIXED number of pixels. We simply place the frames
     at that fixed offset and alpha-blend the shared band. Robust even on
     bare, low-texture metal where feature matching fails.
  2. FEATURE-BASED (optional refine) -- detect ORB keypoints inside the
     overlap band of both frames, match them, and nudge the offset to
     correct small mechanical drift. Only trust it when enough matches are
     found; otherwise fall back to the calibrated offset.

HOW THE STITCH WORKS (top-to-bottom)
------------------------------------
    row 0            +------------------+
                     |                  |
                     |    TOP frame     |   <- rows [0 .. H)
                     |                  |
    H - overlap ---> |==== overlap ====|   <- SHARED band, alpha-blended
    H           ---> |==== band    ====|      top fades out, bottom fades in
                     |                  |
                     |   BOTTOM frame   |   <- rows [H-overlap .. 2H-overlap)
                     |                  |
    2H-overlap ----> +------------------+

In the overlap band, output = top*(1-a) + bottom*a, where a ramps linearly
0 -> 1 down the band. Above the band = pure top; below = pure bottom.
"""

import argparse

import cv2
import numpy as np


# 15% overlap, as decided for the VandeBharat rig (GSD 0.15 mm/px).
DEFAULT_OVERLAP_RATIO = 0.15


def _refine_offset_with_features(top, bottom, overlap_px, search=40):
    """Optionally refine the vertical/horizontal offset using ORB features.

    Looks only inside the overlap band (bottom strip of `top`, top strip of
    `bottom`). Returns (dy, dx) pixel correction to add to the nominal
    offset, or (0, 0) if not enough reliable matches were found.

    `dy` > 0 means the true overlap is a little SMALLER than nominal (frames
    sit further apart); `dx` corrects sideways mechanical misalignment.
    """
    h = top.shape[0]

    # Crop the shared band from each frame (grayscale for feature detection).
    top_band = cv2.cvtColor(top[h - overlap_px:h], cv2.COLOR_BGR2GRAY)
    bot_band = cv2.cvtColor(bottom[0:overlap_px], cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=1500)
    k1, d1 = orb.detectAndCompute(top_band, None)
    k2, d2 = orb.detectAndCompute(bot_band, None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return 0, 0  # low-texture metal -> trust the calibrated offset

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(d1, d2)
    if len(matches) < 8:
        return 0, 0

    # Keep the strongest matches, take the MEDIAN shift (robust to outliers).
    matches = sorted(matches, key=lambda m: m.distance)[:60]
    shifts = np.array([
        (k2[m.trainIdx].pt[1] - k1[m.queryIdx].pt[1],   # vertical
         k2[m.trainIdx].pt[0] - k1[m.queryIdx].pt[0])    # horizontal
        for m in matches
    ])
    dy, dx = np.median(shifts, axis=0)

    # Reject implausible corrections (mount is rigid; drift is small).
    if abs(dy) > search or abs(dx) > search:
        return 0, 0
    return int(round(dy)), int(round(dx))


def stitch_vertical(top, bottom, overlap_ratio=DEFAULT_OVERLAP_RATIO,
                    use_features=False):
    """Stitch two frames into one vertical composite.

    top, bottom   : BGR images, SAME width and height (e.g. 1456 x 1088).
    overlap_ratio : fraction of frame height that the two cameras share.
    use_features  : if True, refine the fixed offset with ORB matching.

    Returns the composite BGR image (height = 2*H - overlap_px).
    """
    if top.shape != bottom.shape:
        raise ValueError(
            f"frames must match in size: {top.shape} vs {bottom.shape}")

    h, w = top.shape[:2]
    overlap_px = int(round(overlap_ratio * h))          # 0.15 * 1088 = 163
    dy = dx = 0

    if use_features:
        # Nudge the nominal offset to absorb small mechanical drift.
        dy, dx = _refine_offset_with_features(top, bottom, overlap_px)
        overlap_px = max(1, overlap_px - dy)

    out_h = 2 * h - overlap_px                           # 2*1088 - 163 = 2013
    composite = np.zeros((out_h, w, 3), dtype=np.uint8)

    # 1) Lay down the TOP frame from row 0.
    composite[0:h] = top

    # 2) Lay down the BOTTOM frame starting where the overlap begins.
    #    Rows [h-overlap_px .. out_h) belong to the bottom frame.
    start = h - overlap_px
    composite[start:start + h] = bottom

    # 3) Alpha-blend the shared band so the seam disappears.
    #    Alpha ramps 0 -> 1 down the band: top fades out, bottom fades in.
    alpha = np.linspace(0.0, 1.0, overlap_px, dtype=np.float32)[:, None, None]
    top_band = top[h - overlap_px:h].astype(np.float32)
    bot_band = bottom[0:overlap_px].astype(np.float32)
    blended = top_band * (1.0 - alpha) + bot_band * alpha
    composite[start:h] = np.clip(blended, 0, 255).astype(np.uint8)

    # 4) Apply any sideways (dx) correction by rolling the bottom half.
    #    (Kept simple; for large dx use a full affine warp instead.)
    if dx:
        composite[h:] = np.roll(composite[h:], dx, axis=1)

    return composite


def main():
    ap = argparse.ArgumentParser(
        description="Vertically stitch two overlapping camera frames.")
    ap.add_argument("top", help="path to the TOP frame image")
    ap.add_argument("bottom", help="path to the BOTTOM frame image")
    ap.add_argument("-o", "--out", default="stitched.png",
                    help="output composite path")
    ap.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP_RATIO,
                    help="overlap fraction of frame height (default 0.15)")
    ap.add_argument("--features", action="store_true",
                    help="refine offset with ORB feature matching")
    args = ap.parse_args()

    top = cv2.imread(args.top)
    bottom = cv2.imread(args.bottom)
    if top is None or bottom is None:
        raise SystemExit("could not read one of the input frames")

    composite = stitch_vertical(top, bottom, args.overlap, args.features)
    cv2.imwrite(args.out, composite)
    print(f"stitched {top.shape[1]}x{top.shape[0]} + "
          f"{bottom.shape[1]}x{bottom.shape[0]} -> "
          f"{composite.shape[1]}x{composite.shape[0]}  ({args.out})")


if __name__ == "__main__":
    main()
