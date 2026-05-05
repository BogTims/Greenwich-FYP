"""
Plot Results from batch_run.py
Reads a results CSV produced by batch_run.py and generates the charts
"""

import os
import csv
import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# colours kept consistent across all charts
SUBSET_COLOURS = {
    "claimed_phi": "#C8102E",      # red - the ones designers claim use phi
    "general":     "#1F77B4",      # blue - control popular logos
    "ai_controls": "#2CA02C",      # green - synthetic phi-perfect logos
}

LABEL_COLOURS = {
    "GOLDEN":                  "#1B7F1B",
    "PARTIALLY GOLDEN":        "#74C476",
    "WEAK GOLDEN INDICATION":  "#FDD0A2",
    "NOT GOLDEN":              "#C8102E",
    "INSUFFICIENT DATA":       "#999999",
}

LABEL_ORDER = [
    "GOLDEN", "PARTIALLY GOLDEN", "WEAK GOLDEN INDICATION",
    "NOT GOLDEN", "INSUFFICIENT DATA",
]


def load_csv(path):
    """Load the CSV into a list of dicts with numeric coercion."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # coerce numeric fields
            for key in ("tolerance", "A_ratio", "A_deviation", "B_ratio",
                        "B_deviation", "C_ratio", "C_deviation",
                        "D_ratio", "D_deviation", "frac"):
                try:
                    r[key] = float(r.get(key, 0) or 0)
                except ValueError:
                    r[key] = 0.0
            for key in ("A_eligible", "A_triggered", "B_eligible", "B_triggered",
                        "C_eligible", "C_triggered", "D_eligible", "D_triggered",
                        "eligible_count", "hit_count"):
                try:
                    r[key] = int(r.get(key, 0) or 0)
                except ValueError:
                    r[key] = 0
            rows.append(r)
    return rows


def get_subsets(rows):
    """Return subset names in a stable order."""
    found = sorted({r["subset"] for r in rows})
    # put the canonical names first if they exist
    preferred = ["claimed_phi", "general", "ai_controls"]
    ordered = [p for p in preferred if p in found]
    extras = [s for s in found if s not in preferred]
    return ordered + extras


def colour_for(subset, idx):
    """Lookup colour for a subset, falling back to a tab10 default."""
    if subset in SUBSET_COLOURS:
        return SUBSET_COLOURS[subset]
    return plt.cm.tab10(idx % 10)


# -------------------flag rates

def plot_flag_rates(rows, out_path, tolerance):
    """Per-heuristic hit rate (among eligible) by subset, single tolerance."""
    rows = [r for r in rows if abs(r["tolerance"] - tolerance) < 1e-9]
    if not rows:
        print(f"  skip flag_rates: no rows at tolerance {tolerance}")
        return

    subsets = get_subsets(rows)
    heuristics = ["A", "B", "C", "D"]
    heuristic_names = ["A: Bounding Box", "B: Areas",
                       "C: Centroids", "D: Circles"]

    # rate[subset][heuristic] = (hits, eligible)
    rate = {s: {} for s in subsets}
    for s in subsets:
        sub_rows = [r for r in rows if r["subset"] == s]
        for h in heuristics:
            elig = [r for r in sub_rows if r[f"{h}_eligible"]]
            hits = sum(1 for r in elig if r[f"{h}_triggered"])
            rate[s][h] = (hits, len(elig))

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(heuristics))
    width = 0.8 / max(len(subsets), 1)

    for i, s in enumerate(subsets):
        values = []
        labels = []
        for h in heuristics:
            hits, elig = rate[s][h]
            pct = (hits / elig * 100) if elig else 0
            values.append(pct)
            labels.append(f"{hits}/{elig}" if elig else "n/a")

        offset = (i - (len(subsets) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width,
                      label=s.replace("_", " "),
                      color=colour_for(s, i), edgecolor="black", linewidth=0.5)
        # annotate each bar
        for bar, lbl in zip(bars, labels):
            h_bar = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h_bar + 1,
                    lbl, ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(heuristic_names)
    ax.set_ylabel("Hit rate among eligible logos (%)")
    ax.set_ylim(0, 110)
    ax.set_title(f"Per-heuristic hit rates by subset  "
                 f"(tolerance = {tolerance:.0%})")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ------------------------classification distribution

def plot_classification(rows, out_path, tolerance):
    """Stacked bar of classification labels per subset."""
    rows = [r for r in rows if abs(r["tolerance"] - tolerance) < 1e-9]
    if not rows:
        return

    subsets = get_subsets(rows)
    counts = {s: {lbl: 0 for lbl in LABEL_ORDER} for s in subsets}
    for r in rows:
        if r["label"] in counts[r["subset"]]:
            counts[r["subset"]][r["label"]] += 1

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(subsets))
    bottom = np.zeros(len(subsets))

    for lbl in LABEL_ORDER:
        vals = np.array([counts[s][lbl] for s in subsets])
        if vals.sum() == 0:
            continue
        bars = ax.bar(x, vals, bottom=bottom,
                      label=lbl, color=LABEL_COLOURS[lbl],
                      edgecolor="black", linewidth=0.5)
        # annotate counts inside bars
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_y() + bar.get_height()/2,
                        str(int(v)), ha="center", va="center",
                        fontsize=9, fontweight="bold")
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", " ") for s in subsets])
    ax.set_ylabel("Number of logos")
    ax.set_title(f"Classification distribution by subset  "
                 f"(tolerance = {tolerance:.0%})")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------sensitivity curve

def plot_sensitivity(rows, out_path):
    """Flag rate vs tolerance, one line per subset.

    'Flagged' = labelled GOLDEN or PARTIALLY GOLDEN.
    Only produced if the CSV contains more than one tolerance value.
    """
    tolerances = sorted({r["tolerance"] for r in rows})
    if len(tolerances) < 2:
        print("  skip sensitivity: only one tolerance in CSV")
        return

    subsets = get_subsets(rows)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, s in enumerate(subsets):
        xs, ys = [], []
        for tol in tolerances:
            sub_rows = [r for r in rows
                        if r["subset"] == s
                        and abs(r["tolerance"] - tol) < 1e-9]
            if not sub_rows:
                continue
            flagged = sum(1 for r in sub_rows
                          if r["label"] in ("GOLDEN", "PARTIALLY GOLDEN"))
            xs.append(tol * 100)
            ys.append(flagged / len(sub_rows) * 100)

        ax.plot(xs, ys, marker="o", linewidth=2,
                label=s.replace("_", " "),
                color=colour_for(s, i))

    ax.set_xlabel("Tolerance window around phi (%)")
    ax.set_ylabel("Logos flagged GOLDEN or PARTIALLY GOLDEN (%)")
    ax.set_title("Sensitivity of detection rate to tolerance window")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


#-------------------------eligibility

def plot_eligibility(rows, out_path, tolerance):
    """Eligibility rate of each heuristic per subset.

    Eligibility tells us how often each heuristic could even be applied
    - critical context for heuristic D which is often skipped.
    """
    rows = [r for r in rows if abs(r["tolerance"] - tolerance) < 1e-9]
    if not rows:
        return

    subsets = get_subsets(rows)
    heuristics = ["A", "B", "C", "D"]
    heuristic_names = ["A: Bounding Box", "B: Areas",
                       "C: Centroids", "D: Circles"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(heuristics))
    width = 0.8 / max(len(subsets), 1)

    for i, s in enumerate(subsets):
        sub_rows = [r for r in rows if r["subset"] == s]
        n = len(sub_rows)
        if n == 0:
            continue
        values = []
        for h in heuristics:
            elig = sum(1 for r in sub_rows if r[f"{h}_eligible"])
            values.append(elig / n * 100)

        offset = (i - (len(subsets) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width,
                      label=f"{s.replace('_', ' ')} (n={n})",
                      color=colour_for(s, i), edgecolor="black", linewidth=0.5)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{v:.0f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(heuristic_names)
    ax.set_ylabel("Eligibility rate (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Heuristic eligibility rate by subset")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# --------------------- main ---------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot results from batch_run.py")
    parser.add_argument("csv", help="Path to results CSV from batch_run.py")
    parser.add_argument("--output-dir", "-o", default="charts",
                        help="Folder for output charts (default: charts)")
    parser.add_argument("--tolerance", "-t", type=float, default=0.05,
                        help="Primary tolerance for non-sweep charts (default 0.05)")
    args = parser.parse_args()

    rows = load_csv(args.csv)
    if not rows:
        print("CSV is empty.")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tolerances_in_csv = sorted({r["tolerance"] for r in rows})
    print(f"Loaded {len(rows)} rows.  "
          f"Tolerances present: {[f'{t:.0%}' for t in tolerances_in_csv]}")

    # pick a tolerance present in the CSV closest to the requested one
    primary = min(tolerances_in_csv, key=lambda t: abs(t - args.tolerance))
    if abs(primary - args.tolerance) > 1e-9:
        print(f"  using closest available tolerance: {primary:.0%}")

    print("\nGenerating charts:")
    plot_flag_rates(rows, out_dir / "flag_rates.png", primary)
    plot_classification(rows, out_dir / "classification_distribution.png", primary)
    plot_eligibility(rows, out_dir / "heuristic_eligibility.png", primary)
    plot_sensitivity(rows, out_dir / "sensitivity_curve.png")

    print("\nDone.")


if __name__ == "__main__":
    main()