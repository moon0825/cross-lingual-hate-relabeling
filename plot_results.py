#!/usr/bin/env python3
"""Draw summary plots using only released numeric results.

These plots facilitate inspection and do not recreate the manuscript page layout.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
NAMES = {"exaone": "EXAONE", "gemma4": "Gemma", "kcelectra": "KcELECTRA", "xlmr": "XLM-R"}


def read(name):
    with (DATA / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def forest(rows, labels, lower, upper, output):
    values = np.array([float(row["mean"]) for row in rows]) * 100
    lo = np.array([float(row[lower]) for row in rows]) * 100
    hi = np.array([float(row[upper]) for row in rows]) * 100
    fig, ax = plt.subplots(figsize=(9, 1.5 + .36 * len(rows)), layout="constrained")
    y = np.arange(len(rows))
    ax.errorbar(values, y, xerr=[values - lo, hi - values], fmt="o", color="#176B87",
                ecolor="#7196A3", capsize=3, markersize=4)
    ax.axvline(0, color="#888888", linestyle="--", linewidth=.8)
    ax.set(yticks=y, yticklabels=labels, xlabel="Macro-F1 difference (percentage points)")
    ax.invert_yaxis()
    ax.set_title("Paired mean difference and pointwise 95% confidence interval", fontsize=11, pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output.with_suffix(".pdf"))
    fig.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "plots")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    main_rows = [row for row in read("paired_contrasts.csv") if row["kind"] == "relabel_minus_source"]
    labels = [f'{NAMES[r["translator"]]} translation / {NAMES[r["model"]]} / {r["selection"]} / '
              f'{NAMES[r["label"]]} relabel' for r in main_rows]
    forest(main_rows, labels, "ci_low", "ci_high", args.output / "main_relabeling")
    kold_rows = read("kold_contrasts.csv")
    labels = [f'{NAMES[r["translator"]]} translation / {NAMES[r["model"]]} / '
              f'{NAMES[r["label_source"]]} relabel' for r in kold_rows]
    forest(kold_rows, labels, "ci95_lower", "ci95_upper", args.output / "kold_relabeling")
    rows = read("annotation_audit_summary.csv")
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    x = np.arange(len(rows))
    for offset, key, label, color in ((-.24, "source_agreement", "Source labels", "#8B96A3"),
                                     (0, "exaone_agreement", "EXAONE relabels", "#1672AC"),
                                     (.24, "gemma_agreement", "Gemma relabels", "#D16A1B")):
        ax.bar(x + offset, [100 * int(r[key]) / int(r["n"]) for r in rows],
               width=.23, label=label, color=color)
    ax.set(xticks=x, xticklabels=[r["rater"] for r in rows], ylabel="Agreement (%)", ylim=(0, 100))
    ax.tick_params(axis="x", labelrotation=30)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(.5, 1.13))
    fig.savefig(args.output / "shared_300_agreement.pdf")
    fig.savefig(args.output / "shared_300_agreement.png", dpi=300)
    plt.close(fig)
    print("Saved three numeric summary plots in PDF and PNG formats.")


if __name__ == "__main__":
    main()
