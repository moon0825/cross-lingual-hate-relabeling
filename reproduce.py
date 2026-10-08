#!/usr/bin/env python3
"""Recompute published paired statistics from released run-level metrics.

Run from any directory: python /path/to/repository/reproduce.py
No source texts, participant responses, credentials, or model downloads are used.
"""
import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def read_csv(name):
    with (DATA / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def paired_summary(values):
    values = np.asarray(values, dtype=float)
    if len(values) != 23 or not np.isfinite(values).all():
        raise ValueError("A paired contrast must contain 23 finite differences")
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    low, high = stats.t.interval(.95, 22, loc=mean, scale=sd / np.sqrt(23))
    return dict(n=23, mean=mean, sd=sd, ci_low=float(low), ci_high=float(high),
                p_raw=float(stats.ttest_1samp(values, 0).pvalue))


def holm(rows, key):
    p = np.asarray([row["p_raw"] for row in rows])
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(
        p[order] * np.arange(len(p), 0, -1)))
    for row, value in zip(rows, adjusted):
        row[key] = float(value)


def controls():
    rows = read_csv("control_run_metrics.csv")
    score = {(row["model"], row["arm"], int(row["repeat"])):
             float(row["macro_f1"]) for row in rows}
    expected = set(itertools.product(("kcelectra", "xlmr"), ("S", "L", "W", "P"), range(1, 24)))
    if len(rows) != 184 or set(score) != expected:
        raise ValueError("Expected 184 unique control runs")
    results = []
    for model in ("kcelectra", "xlmr"):
        for contrast in ("L-P", "L-W", "L-S", "P-S", "W-S"):
            first, second = contrast.split("-")
            differences = [score[model, first, repeat] - score[model, second, repeat]
                           for repeat in range(1, 24)]
            primary = contrast in ("L-P", "L-W")
            row = dict(model=model, contrast=contrast, primary=primary, **paired_summary(differences))
            # The six secondary contrasts were reported descriptively, without tests.
            if not primary:
                row["p_raw"] = ""
            row["p_holm_4"] = ""
            results.append(row)
    holm([row for row in results if row["primary"]], "p_holm_4")
    return results


def kold():
    rows = read_csv("kold_cell_metrics.csv")
    score = {(row["translator"], row["model"], row["label_source"], int(row["repeat_seed"])):
             float(row["macro_f1"]) for row in rows}
    seeds = range(20260901, 20260924)
    expected = set(itertools.product(("exaone", "gemma4"), ("kcelectra", "xlmr"),
                                     ("source", "exaone", "gemma4"), seeds))
    if len(rows) != 276 or set(score) != expected or any(int(row["n"]) != 40129 for row in rows):
        raise ValueError("Expected 276 unique KOLD results, each on 40,129 comments")
    results = []
    for translator, model, label in itertools.product(("exaone", "gemma4"), ("kcelectra", "xlmr"),
                                                      ("exaone", "gemma4")):
        differences = [score[translator, model, label, seed] - score[translator, model, "source", seed]
                       for seed in seeds]
        row = dict(translator=translator, model=model, label_source=label,
                   id=f"{translator}_{model}_{label}_minus_source", reference="source",
                   **paired_summary(differences))
        row["ci95_lower"] = row.pop("ci_low")
        row["ci95_upper"] = row.pop("ci_high")
        row["t_test_defined"] = True
        for seed, value in zip(seeds, differences):
            row[f"difference_s{seed}"] = value
        results.append(row)
    holm(results, "p_holm_8")
    for row in results:
        row["positive_after_adjustment"] = row["mean"] > 0 and row["p_holm_8"] < .05
        row["negative_after_adjustment"] = row["mean"] < 0 and row["p_holm_8"] < .05
    return results


def compare(actual, expected, keys):
    key = lambda row: tuple(str(row[column]) for column in keys)
    left, right = {key(row): row for row in actual}, {key(row): row for row in expected}
    if len(left) != len(actual) or len(right) != len(expected) or left.keys() != right.keys():
        raise AssertionError("Contrast identities or duplicate rows differ")
    maximum_error, count = 0.0, 0
    for identity, reference in right.items():
        result = left[identity]
        for column, expected_value in reference.items():
            observed = result[column]
            try:
                target = float(expected_value)
            except (ValueError, TypeError):
                if str(observed) != expected_value:
                    raise AssertionError(f"{identity}: {column} differs")
            else:
                difference = abs(float(observed) - target)
                if not np.isfinite(difference) or difference > 1e-12:
                    raise AssertionError(f"{identity}: {column} differs by {difference}")
                maximum_error = max(maximum_error, difference)
                count += 1
    return dict(rows=len(actual), numeric_values_compared=count, max_absolute_difference=maximum_error,
                tolerance=1e-12, status="PASS")


def verify_audit():
    payload = json.loads((DATA / "annotation_audit_statistics.json").read_text())
    assert (payload["human_raters"], payload["llm_raters"], payload["n_unique_texts"],
            payload["judgments_per_rater"]) == (4, 5, 300, 300)
    summary = {row["rater"]: row for row in read_csv("annotation_audit_summary.csv")}
    assert len(summary) == len(payload["raters"]) == 9
    for rater in payload["raters"]:
        row = summary[rater["label"]]
        assert row["family"] == rater["family"]
        assert int(row["n"]) == rater["n"] == sum(rater["label_counts"].values()) == 300
        for key, field in (("hate", "hate"), ("non-hate", "non_hate"), ("unclear", "unclear")):
            assert int(row[field]) == rater["label_counts"].get(key, 0)
        for key, field in (("source", "source_agreement"), ("exaone", "exaone_agreement"),
                           ("gemma4", "gemma_agreement")):
            assert int(row[field]) == rater["agreement_counts"][key]
        for key, prefix in (("exaone", "exaone"), ("gemma4", "gemma")):
            gain = (rater["agreement_counts"][key] - rater["agreement_counts"]["source"]) / 3
            assert abs(gain - float(row[prefix + "_gain_pp"])) < 1e-12
            assert abs(gain - rater["contrasts"][key]["gain_pp"]) < 1e-12
    return dict(status="PASS", humans=4, llm_judges=5, shared_translations=300,
                scope="Aggregate arithmetic consistency only; item-level bootstrap is not rerun")


def verify_checksums():
    count = 0
    for line in (ROOT / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", 1)
        path = ROOT / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise AssertionError(f"Checksum mismatch: {name}")
        count += 1
    return dict(status="PASS", files=count)


def write_csv(path, rows, columns):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs", help="Directory for reconstructed results")
    args = parser.parse_args()
    report = {"checksums": verify_checksums()}
    spec = importlib.util.spec_from_file_location("main_statistics", DATA / "reproduce_statistics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    families = [
        ("main", "paired_contrasts.csv", module.calculate(DATA / "run_metrics.csv"),
         ("translator", "model", "kind", "selection", "label")),
        ("controls", "control_paired_contrasts.csv", controls(), ("model", "contrast")),
        ("kold", "kold_contrasts.csv", kold(), ("translator", "model", "label_source")),
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    for name, filename, computed, keys in families:
        reference = read_csv(filename)
        report[name] = compare(computed, reference, keys)
        write_csv(args.output / filename, computed, list(reference[0]))
    report["audit_aggregate_arithmetic"] = verify_audit()
    report["status"] = "PASS"
    report["holm_family_sizes"] = {"main": 44, "control_primary": 4, "kold": 8}
    (args.output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
