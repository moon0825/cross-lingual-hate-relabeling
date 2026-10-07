# Supporting statistics

**English-to-Korean data augmentation for hate-speech detection: Separating the effects of sentence retrieval and relabeling**

This repository provides run-level results and aggregate annotation-audit
statistics accompanying the manuscript.

## Files

- `data/run_metrics.csv`: 552 main-experiment runs across 23 paired repeats.
- `data/paired_contrasts.csv`: 44 paired contrasts with Holm-adjusted p values.
- `data/control_*.csv`: label-distribution and shuffled-label control results
  for retrieved EXAONE translations; 184 condition-by-repeat result rows.
- `data/kold_*.csv`: external evaluation on 40,129 KOLD comments, including
  276 condition-by-repeat results and eight paired contrasts.
- `data/annotation_audit_*`: aggregate results from four human raters and
  five LLM judges, each judging the same 300 translations. These are 300 texts
  in total, not 300 different texts for each judge.
- `data/native_threshold_metrics.csv`: recorded native-task threshold metrics.
- `data/diagnostic_tables/`: target-group recall, label transitions, and timing.
- `data/model_metadata.json` and `data/kold_dataset_metadata.json`: model
  versions and the public KOLD release used in the study.

## Reading the results

Condition keys `exaone` and `gemma4` identify the translator or label source
named by the column; `source` denotes inherited corpus labels. Classifier keys
`kcelectra` and `xlmr` denote KcELECTRA and XLM-R. `retrieved` and `random`
identify the two sentence-selection conditions.

The control arms are S (source labels), L (EXAONE relabels), W (source-label
training weighted to match corpus-specific relabeled class counts), and
P (shuffled EXAONE relabels). The manuscript and supplement describe the
matching and shuffling procedures.

F1, recall, accuracy, and agreement proportions use the 0–1 scale unless a
field explicitly specifies percent or percentage points. In
`annotation_audit_summary.csv`, the three `*_agreement` columns are counts out
of `n` (300), not proportions. MCC and Cohen's kappa retain their native scale.
Audit fields ending in `_gain_pp` are percentage-point
differences; JSON fields named `agreement_percent` are percentages. Main,
control-primary, and KOLD contrast families use separate Holm corrections
over 44, four, and eight comparisons, respectively. Confidence intervals in
the paired performance contrasts are pointwise 95% intervals.

## Data scope and privacy

This is a statistical-results release. Corpus texts, individual annotation
responses, original record-ID assignments, and training/plotting code are
not included. Human raters have pseudonymous identifiers. All numerical
results and condition labels are preserved; private annotation-file
fingerprints and internal approval metadata have been removed.

The files support inspection of reported aggregate results. Recomputing the
annotation bootstrap or retraining classifiers also requires the corresponding
item-level inputs and code described in the manuscript and supplement.

## Licensing

A separate reuse license has not yet been assigned. Public availability should
not be interpreted as a CC BY or software-license grant. The original datasets
and models remain subject to their respective providers' terms.
