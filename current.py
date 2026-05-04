"""
Golden Ratio Detector for Logos
Checks if a logo image has shapes close to the golden ratio (phi = 1.618).

Usage:
    python golden_ratio_detector.py logo.png
    python golden_ratio_detector.py logo.png --tolerance 0.05
    python golden_ratio_detector.py logo.png --no-vis
"""

import os
import argparse
from itertools import combinations

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# golden ratio values
PHI = 1.618
PHI_SQ = PHI * PHI    # 2.618
PHI_CB = PHI * PHI * PHI    # 4.236

# settings
MIN_AREA_RATIO = 0.005    # ignore shapes smaller than 0.5% of image
MAX_SIZE = 1000           # resize big images to this


# helpers

def is_near(value, target, tol):
    # check if value is within tolerance of target
    return target * (1 - tol) <= value <= target * (1 + tol)


def safe_ratio(a, b):
    # always returns bigger/smaller, never crashes
    lo = min(a, b)
    if lo == 0:
        return 0.0
    return max(a, b) / lo


def deviation(value, target=PHI):
    # how far off as a percent
    if target == 0:
        return 100.0
    return abs(value - target) / target * 100


def check_extended(ratio, tol):
    # check phi, phi^2, phi^3 (area comparisons)
    for target, name in [(PHI, "phi"), (PHI_SQ, "phi^2"), (PHI_CB, "phi^3")]:
        if is_near(ratio, target, tol):
            return True, name
    return False, None


# image preparation

def preprocess(path):
    # load image and prepare it for analysis
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cant read: {path}")

    # handle PNG transparency by putting white behind it
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        bgr = img[:, :, :3].astype(np.float32)
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        white = np.full_like(bgr, 255.0)
        img = (bgr * alpha + white * (1 - alpha)).astype(np.uint8)

    # resize keeping aspect ratio
    longest = max(img.shape[:2])
    if longest > MAX_SIZE:
        scale = MAX_SIZE / longest
        nw = int(img.shape[1] * scale)
        nh = int(img.shape[0] * scale)
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    # grayscale and blur for circle detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # make binary mask using corners as background reference
    binary = make_mask(img)
    return img, blur, binary


def make_mask(img):
    # detect background color from corners then make black/white mask
    h, w = img.shape[:2]
    pad = max(2, min(h, w) // 50)

    corners = np.array([
        img[pad, pad],
        img[pad, w - 1 - pad],
        img[h - 1 - pad, pad],
        img[h - 1 - pad, w - 1 - pad],
    ], dtype=np.float32)
    bg = np.median(corners, axis=0)

    # distance from background per pixel
    diff = img.astype(np.float32) - bg
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    dist_u8 = np.clip(dist, 0, 255).astype(np.uint8)

    # otsu picks threshold automatically
    _, mask = cv2.threshold(dist_u8, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # close small gaps
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def get_contours(mask, area):
    # find shape outlines, biggest first
    # RETR_LIST gets inner contours too (needed for frame logos)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST,
                                    cv2.CHAIN_APPROX_SIMPLE)
    min_a = area * MIN_AREA_RATIO
    contours = [c for c in contours if cv2.contourArea(c) >= min_a]
    contours.sort(key=cv2.contourArea, reverse=True)
    return contours


# heuristic A - bounding boxes

def check_boxes(contours, tol):
    if not contours:
        return {"eligible": False, "triggered": False,
                "ratio": 0, "deviation": 100, "boxes": []}

    boxes = []

    # global box around all shapes
    pts = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(pts)
    r = safe_ratio(w, h)
    boxes.append(("Global", (x, y, w, h), r, is_near(r, PHI, tol)))

    # boxes for top 3 contours
    for i, c in enumerate(contours[:3]):
        bx, by, bw, bh = cv2.boundingRect(c)
        r = safe_ratio(bw, bh)
        boxes.append((f"C{i+1}", (bx, by, bw, bh), r, is_near(r, PHI, tol)))

    triggered = any(b[3] for b in boxes)
    hits = [b[2] for b in boxes if b[3]]
    if hits:
        report = min(hits, key=lambda r: deviation(r))
    else:
        report = min((b[2] for b in boxes), key=lambda r: deviation(r))

    return {
        "eligible": True,
        "triggered": triggered,
        "ratio": round(report, 4),
        "deviation": round(deviation(report), 2),
        "boxes": boxes,
    }


# heuristic B - area ratios

def check_areas(contours, tol):
    if len(contours) < 2:
        return {"eligible": False, "triggered": False,
                "ratio": 0, "deviation": 100,
                "best_pair": None, "matched": None}

    areas = [cv2.contourArea(c) for c in contours[:5]]
    best_r, best_d, best_pair, best_target = 0, 100, None, None
    triggered = False

    for i in range(len(areas)):
        for j in range(i + 1, len(areas)):
            r = safe_ratio(areas[i], areas[j])
            hit, label = check_extended(r, tol)

            target = {"phi": PHI, "phi^2": PHI_SQ,
                      "phi^3": PHI_CB}.get(label, PHI)
            d = deviation(r, target)

            # prefer hits, then lowest deviation
            better = (hit and not triggered) or \
                     (hit == triggered and d < best_d)
            if better:
                best_r, best_d = r, d
                best_pair, best_target = (i, j), label
                if hit:
                    triggered = True

    return {
        "eligible": True,
        "triggered": triggered,
        "ratio": round(best_r, 4),
        "deviation": round(best_d, 2),
        "best_pair": best_pair,
        "matched": best_target,
    }


# heuristic C - centroid distances

def check_centroids(contours, tol):
    if len(contours) < 3:
        return {"eligible": False, "triggered": False,
                "ratio": 0, "deviation": 100, "centroids": []}

    centroids = []
    for c in contours[:3]:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroids.append((cx, cy))

    if len(centroids) < 3:
        return {"eligible": False, "triggered": False,
                "ratio": 0, "deviation": 100, "centroids": centroids}

    # 3 distances between 3 points
    dists = [np.linalg.norm(np.array(centroids[i]) - np.array(centroids[j]))
             for i, j in combinations(range(3), 2)]

    best_r, best_d = 0, 100
    for i in range(len(dists)):
        for j in range(i + 1, len(dists)):
            r = safe_ratio(dists[i], dists[j])
            d = deviation(r)
            if d < best_d:
                best_d, best_r = d, r

    return {
        "eligible": True,
        "triggered": is_near(best_r, PHI, tol),
        "ratio": round(best_r, 4),
        "deviation": round(best_d, 2),
        "centroids": centroids,
    }


# heuristic D - circles

def check_circles(blur, tol):
    h, w = blur.shape[:2]
    short = min(h, w)

    found = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(short // 8, 20),
        param1=100,
        param2=60,
        minRadius=max(short // 50, 8),
        maxRadius=short // 2,
    )

    if found is None or len(found[0]) < 2:
        circles = ([] if found is None
                   else [tuple(map(int, c)) for c in found[0]])
        return {"eligible": False, "triggered": False,
                "ratio": 0, "deviation": 100, "circles": circles}

    circles = [tuple(map(int, c)) for c in found[0]]

    best_r, best_d = 0, 100
    for (_, _, r1), (_, _, r2) in combinations(circles, 2):
        r = safe_ratio(r1, r2)
        d = deviation(r)
        if d < best_d:
            best_d, best_r = d, r

    return {
        "eligible": True,
        "triggered": is_near(best_r, PHI, tol),
        "ratio": round(best_r, 4),
        "deviation": round(best_d, 2),
        "circles": circles,
    }


# scoring

def score(a, b, c, d):
    # only count heuristics that actually ran
    all_res = [a, b, c, d]
    eligible = [r for r in all_res if r.get("eligible")]
    triggered = [r for r in eligible if r.get("triggered")]

    total = len(eligible)
    matches = len(triggered)
    frac = matches / total if total else 0

    if frac >= 0.75:
        label = "GOLDEN"
    elif frac >= 0.25:
        label = "POSSIBLY GOLDEN"
    else:
        label = "NOT GOLDEN"

    return {"matches": matches, "eligible": total,
            "frac": round(frac, 3), "label": label}


# visualization

def make_title(name, res):
    if res and res.get("eligible"):
        status = "GOLDEN" if res["triggered"] else "not golden"
        return (f"{name}\nRatio = {res['ratio']} | {status} "
                f"(dev {res['deviation']}%)")
    return f"{name}\n(not enough data)"


def visualize(img, contours, a, b, c, d, score_info, name, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Golden Ratio Analysis - {name}",
                 fontsize=18, fontweight="bold")

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    thick = max(2, min(h, w) // 250)

    # original
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Original Logo", fontsize=12)
    axes[0, 0].axis("off")

    # boxes
    img_b = rgb.copy()
    for n, (x, y, bw, bh), ratio, hit in a.get("boxes", []):
        col = (0, 200, 0) if hit else (220, 120, 0)
        cv2.rectangle(img_b, (x, y), (x + bw, y + bh), col, thick)
    axes[0, 1].imshow(img_b)
    for n, (x, y, bw, bh), ratio, hit in a.get("boxes", []):
        col_m = "#00C800" if hit else "#DC7800"
        if y > 18:
            ty, va = y - 4, "bottom"
        else:
            ty, va = y + bh + 4, "top"
        axes[0, 1].annotate(
            f"{ratio:.2f}", xy=(x + bw / 2, ty),
            ha="center", va=va, fontsize=9, color=col_m, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor=col_m, alpha=0.9),
        )
    axes[0, 1].set_title(make_title("A: Bounding Boxes", a), fontsize=11)
    axes[0, 1].axis("off")

    # circles
    img_c = rgb.copy()
    for (cx, cy, r) in d.get("circles", []):
        cv2.circle(img_c, (cx, cy), r, (0, 120, 255), thick)
        cv2.circle(img_c, (cx, cy), 3, (0, 120, 255), -1)
    axes[0, 2].imshow(img_c)
    axes[0, 2].set_title(make_title("D: Circle Radii", d), fontsize=11)
    axes[0, 2].axis("off")

    # areas
    img_a = rgb.copy()
    if b.get("best_pair") and len(contours) > max(b["best_pair"]):
        i, j = b["best_pair"]
        cv2.drawContours(img_a, [contours[i]], -1, (0, 200, 0), thick + 1)
        cv2.drawContours(img_a, [contours[j]], -1, (0, 120, 255), thick + 1)
    title_b = "B: Area Proportions"
    if b.get("matched"):
        title_b += f" (matched {b['matched']})"
    axes[1, 0].imshow(img_a)
    axes[1, 0].set_title(make_title(title_b, b), fontsize=11)
    axes[1, 0].axis("off")

    # centroids
    img_cn = rgb.copy()
    cents = c.get("centroids", [])
    dot = max(5, min(h, w) // 100)
    for cx, cy in cents:
        cv2.circle(img_cn, (cx, cy), dot, (255, 0, 0), -1)
    for i, j in combinations(range(len(cents)), 2):
        cv2.line(img_cn, cents[i], cents[j], (255, 200, 0), thick)
    axes[1, 1].imshow(img_cn)
    axes[1, 1].set_title(make_title("C: Centroid Distances", c), fontsize=11)
    axes[1, 1].axis("off")

    # summary
    axes[1, 2].axis("off")
    summary = (
        f"Score: {score_info['matches']} / {score_info['eligible']}  "
        f"({int(score_info['frac']*100)}%)\n\n"
        f"Classification:\n{score_info['label']}\n\n"
        f"A (boxes):     {'HIT' if a['triggered'] else '-'}   "
        f"ratio {a['ratio']}\n"
        f"B (areas):     {'HIT' if b['triggered'] else '-'}   "
        f"ratio {b['ratio']}\n"
        f"C (centroids): {'HIT' if c['triggered'] else '-'}   "
        f"ratio {c['ratio']}\n"
        f"D (circles):   {'HIT' if d['triggered'] else '-'}   "
        f"ratio {d['ratio']}"
    )
    axes[1, 2].text(0.02, 0.95, summary, fontsize=13, va="top",
                    family="monospace")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# main

def main():
    parser = argparse.ArgumentParser(description="Golden Ratio Detector")
    parser.add_argument("image", help="Path to logo image")
    parser.add_argument("--tolerance", type=float, default=0.05,
                        help="Tolerance around phi (default 0.05)")
    parser.add_argument("--no-vis", action="store_true",
                        help="Skip saving visualisation")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"Error: file not found - {args.image}")
        return

    name = os.path.splitext(os.path.basename(args.image))[0]
    print(f"Analysing: {args.image}")

    # load and prepare
    img, blur, mask = preprocess(args.image)
    h, w = img.shape[:2]
    contours = get_contours(mask, h * w)

    # run heuristics
    a = check_boxes(contours, args.tolerance)
    b = check_areas(contours, args.tolerance)
    c = check_centroids(contours, args.tolerance)
    d = check_circles(blur, args.tolerance)

    # score
    s = score(a, b, c, d)

    # print results
    print(f"\nResult: {s['label']} (Score {s['matches']}/{s['eligible']})")
    print(f"  A boxes:      {'HIT' if a['triggered'] else 'miss'} "
          f"ratio={a['ratio']}")
    print(f"  B areas:      {'HIT' if b['triggered'] else 'miss'} "
          f"ratio={b['ratio']}")
    print(f"  C centroids:  {'HIT' if c['triggered'] else 'miss'} "
          f"ratio={c['ratio']}")
    print(f"  D circles:    {'HIT' if d['triggered'] else 'miss'} "
          f"ratio={d['ratio']}")

    # save visualization
    if not args.no_vis:
        out = f"{name}_analysis.png"
        visualize(img, contours, a, b, c, d, s, name, out)
        print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()