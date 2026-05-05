"""
Runs golden_ratio_detector.py over a folder structure of logos and writes
results to a CSV file for later analysis and plotting.


The script imports the existing detector
"""

import os
import csv
import argparse
from pathlib import Path

# import the existing detector
import golden_ratio_detector as detector


# accepted image file extensions
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

# default tolerance values used for the sensitivity sweep
SWEEP_TOLERANCES = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]

# CSV column order - kept stable so the downstream plotting script knows it
CSV_COLUMNS = [
    "filename", "subset", "tolerance",
    "A_eligible", "A_triggered", "A_ratio", "A_deviation",
    "B_eligible", "B_triggered", "B_ratio", "B_deviation", "B_matched",
    "C_eligible", "C_triggered", "C_ratio", "C_deviation",
    "D_eligible", "D_triggered", "D_ratio", "D_deviation",
    "eligible_count", "hit_count", "frac", "label",
]


def find_logos(root):
    """iterate the dataset folder and return"""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {root}")

    logos = []
    # each immediate daughter is treated as a subset
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        for path in sorted(sub.iterdir()):
            if path.suffix.lower() in IMAGE_EXTS:
                logos.append((sub.name, path))
    return logos


def classify(eligible, hit):
    """
    Fewer than two eligible heuristics is not enough information to make
    a golden logo
    """
    if eligible < 2:
        return "INSUFFICIENT DATA", 0.0
    frac = hit / eligible
    if frac >= 1.0:
        label = "GOLDEN"
    elif frac >= 0.5:
        label = "PARTIALLY GOLDEN"
    elif frac > 0:
        label = "WEAK GOLDEN INDICATION"
    else:
        label = "NOT GOLDEN"
    return label, round(frac, 3)


def analyse_one(path, tolerance, save_vis, vis_dir):
    """Run all four heuristics on a single image and return a dict row."""
    img, blur, mask = detector.preprocess(str(path))
    h, w = img.shape[:2]
    contours = detector.get_contours(mask, h * w)

    a = detector.check_boxes(contours, tolerance)
    b = detector.check_areas(contours, tolerance)
    c = detector.check_centroids(contours, tolerance)
    d = detector.check_circles(blur, tolerance)

    eligible = sum(1 for r in (a, b, c, d) if r.get("eligible"))
    hit = sum(1 for r in (a, b, c, d) if r.get("triggered"))
    label, frac = classify(eligible, hit)

    # save annotated visualisation if requested
    if save_vis and vis_dir is not None:
        vis_dir.mkdir(parents=True, exist_ok=True)
        out_path = vis_dir / f"{path.stem}_t{int(tolerance*1000):03d}.png"
        score_info = {
            "matches": hit, "eligible": eligible,
            "frac": frac, "label": label,
        }
        detector.visualize(
            img, contours, a, b, c, d, score_info,
            path.stem, str(out_path),
        )

    return {
        "filename": path.name,
        "tolerance": tolerance,
        "A_eligible": int(a.get("eligible", False)),
        "A_triggered": int(a.get("triggered", False)),
        "A_ratio": a.get("ratio", 0),
        "A_deviation": a.get("deviation", 100),
        "B_eligible": int(b.get("eligible", False)),
        "B_triggered": int(b.get("triggered", False)),
        "B_ratio": b.get("ratio", 0),
        "B_deviation": b.get("deviation", 100),
        "B_matched": b.get("matched") or "",
        "C_eligible": int(c.get("eligible", False)),
        "C_triggered": int(c.get("triggered", False)),
        "C_ratio": c.get("ratio", 0),
        "C_deviation": c.get("deviation", 100),
        "D_eligible": int(d.get("eligible", False)),
        "D_triggered": int(d.get("triggered", False)),
        "D_ratio": d.get("ratio", 0),
        "D_deviation": d.get("deviation", 100),
        "eligible_count": eligible,
        "hit_count": hit,
        "frac": frac,
        "label": label,
    }


def print_summary(rows, tolerances):
    """Per-subset summary printed to console at the end of a run."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    subsets = sorted({r["subset"] for r in rows})

    for tol in tolerances:
        tol_rows = [r for r in rows if abs(r["tolerance"] - tol) < 1e-9]
        if not tol_rows:
            continue

        print(f"\nTolerance = {tol:.0%}")
        print("-" * 70)

        for sub in subsets:
            sub_rows = [r for r in tol_rows if r["subset"] == sub]
            if not sub_rows:
                continue

            n = len(sub_rows)
            golden = sum(1 for r in sub_rows if r["label"] == "GOLDEN")
            partial = sum(1 for r in sub_rows if r["label"] == "PARTIALLY GOLDEN")
            weak = sum(1 for r in sub_rows if r["label"] == "WEAK GOLDEN INDICATION")
            not_g = sum(1 for r in sub_rows if r["label"] == "NOT GOLDEN")
            insuf = sum(1 for r in sub_rows if r["label"] == "INSUFFICIENT DATA")

            # per-heuristic flag rates among eligible logos only
            def rate(prefix):
                elig = [r for r in sub_rows if r[f"{prefix}_eligible"]]
                if not elig:
                    return "n/a"
                hits = sum(1 for r in elig if r[f"{prefix}_triggered"])
                return f"{hits}/{len(elig)} ({hits/len(elig)*100:.0f}%)"

            print(f"  {sub:<20} n={n:>3}  "
                  f"GOLDEN={golden}  PARTIAL={partial}  "
                  f"WEAK={weak}  NOT={not_g}  INSUF={insuf}")
            print(f"  {'':<20} A: {rate('A')}   B: {rate('B')}   "
                  f"C: {rate('C')}   D: {rate('D')}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch golden ratio detector with CSV output"
    )
    parser.add_argument("dataset", help="Path to dataset folder containing subset subfolders")
    parser.add_argument("--output", "-o", default="results.csv",
                        help="Path for CSV output (default: results.csv)")
    parser.add_argument("--tolerance", "-t", type=float, default=0.05,
                        help="Tolerance around phi (default: 0.05)")
    parser.add_argument("--tolerance-sweep", action="store_true",
                        help="Run across multiple tolerance values for sensitivity analysis")
    parser.add_argument("--no-vis", action="store_true",
                        help="Skip saving annotated visualisations (faster)")
    parser.add_argument("--vis-dir", default="output_visuals",
                        help="Folder for annotated visualisations (default: output_visuals)")
    args = parser.parse_args()

    # discover logos
    print(f"Scanning {args.dataset} ...")
    logos = find_logos(args.dataset)
    if not logos:
        print("No images found. Check the dataset folder structure.")
        return

    print(f"Found {len(logos)} images across "
          f"{len({s for s, _ in logos})} subsets.")

    # decide which tolerance values to use
    tolerances = SWEEP_TOLERANCES if args.tolerance_sweep else [args.tolerance]
    if args.tolerance_sweep:
        print(f"Sensitivity sweep: tolerances = "
              f"{[f'{t:.0%}' for t in tolerances]}")

    # only save visualisations on the primary tolerance to avoid duplicates
    primary_tol = args.tolerance if not args.tolerance_sweep else 0.05

    rows = []
    total_runs = len(logos) * len(tolerances)
    run_idx = 0

    for tol in tolerances:
        vis_dir = Path(args.vis_dir) if (not args.no_vis and tol == primary_tol) else None

        for subset, path in logos:
            run_idx += 1
            print(f"  [{run_idx}/{total_runs}] {subset}/{path.name} @ tol={tol:.0%}",
                  end="", flush=True)
            try:
                row = analyse_one(path, tol, save_vis=(vis_dir is not None), vis_dir=vis_dir)
                row["subset"] = subset
                rows.append(row)
                print(f"  -> {row['label']} ({row['hit_count']}/{row['eligible_count']})")
            except Exception as exc:
                print(f"  -> ERROR: {exc}")

    # write CSV
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})
    print(f"\nResults written to: {args.output}")

    # console summary
    print_summary(rows, tolerances)


if __name__ == "__main__":
    main()