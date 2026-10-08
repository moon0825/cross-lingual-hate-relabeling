#!/usr/bin/env python3
"""Reproduce all 44 contrasts from run_metrics.csv (Python, numpy, scipy).

Usage: python reproduce_statistics.py
Writes CSV to stdout. Does not require source text or evaluation labels.
"""
import csv
import itertools
from pathlib import Path
import sys

import numpy as np
from scipy import stats


def calculate(path):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    score = {(x["translator"], x["model"], x["selection"], x["label"], int(x["repeat"])):
             float(x["macro_f1"]) for x in rows}
    t_values = ["exaone", "gemma4"]
    m_values = ["kcelectra", "xlmr"]
    s_values = ["retrieved", "random"]
    l_values = ["source", "exaone", "gemma4"]
    expected = set(itertools.product(t_values, m_values, s_values, l_values, range(1, 24)))
    if len(rows) != 552 or set(score) != expected:
        raise ValueError("Expected exactly 552 unique completed conditions")
    results = []
    for t, m in itertools.product(t_values, m_values):
        def vector(s, l):
            return np.array([score[t, m, s, l, r] for r in range(1, 24)])
        def add(kind, a, s="", l=""):
            mean, sd = float(a.mean()), float(a.std(ddof=1))
            lo, hi = stats.t.interval(.95, 22, loc=mean, scale=sd / np.sqrt(23))
            results.append(dict(translator=t, model=m, kind=kind, selection=s, label=l,
                                n=23, mean=mean, sd=sd, ci_low=lo, ci_high=hi,
                                p_raw=float(stats.ttest_1samp(a, 0).pvalue)))
        for s in s_values:
            for l in l_values[1:]:
                add("relabel_minus_source", vector(s, l)-vector(s, "source"), s, l)
            add("gemma_minus_exaone", vector(s, "gemma4")-vector(s, "exaone"), s)
        for l in l_values[1:]:
            add("selection_by_label_interaction",
                vector("retrieved", l)-vector("retrieved", "source")
                -vector("random", l)+vector("random", "source"), l=l)
        for l in l_values:
            add("retrieved_minus_random", vector("retrieved", l)-vector("random", l), l=l)
    p = np.array([x["p_raw"] for x in results])
    order = np.argsort(p)
    adjusted = np.empty(44)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(44, 0, -1)))
    for row, value in zip(results, adjusted):
        row["p_holm_44"] = float(value)
    return results


if __name__ == "__main__":
    values = calculate(Path(__file__).with_name("run_metrics.csv"))
    writer = csv.DictWriter(sys.stdout, fieldnames=list(values[0]))
    writer.writeheader()
    writer.writerows(values)
