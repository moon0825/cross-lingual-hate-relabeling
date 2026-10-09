# Supporting statistics and reproduction code

**English-to-Korean data augmentation for hate-speech detection: Separating the effects of sentence retrieval and relabeling**

This repository provides the reported run-level results, aggregate annotation-audit
statistics, ID-only training assignments, and executable numerical-analysis code.
It contains four human raters and five LLM judges assessing the same 300 translations.

## Quick start

Use Python 3.12 or later in an isolated environment:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py
python plot_results.py
```

On Windows, use `.venv\Scripts\activate` to activate the environment.
The commands need no credentials, source-text files, model weights, or GPU.
`reproduce.py` checks release checksums, reconstructs 44 main contrasts, ten
control contrasts (four primary tests and six descriptive comparisons), and
eight KOLD contrasts. The updated release also reconstructs 14 post-hoc target
L-minus-W contrasts with Holm adjustment, four-rater nominal alpha, panel
agreement gains, and aggregate error-turnover counts. It compares reconstructed entries with the released
tables using an absolute numerical tolerance of 1e-12 and writes the results
and a verification report to `outputs/`.

`plot_results.py` draws three summary plots from released numbers: main
relabeling effects, KOLD relabeling effects, and shared-300 label agreement.
These inspection plots use a portable font and do not reproduce the manuscript's
page layout. Each is saved as a vector PDF and a 300-dpi PNG under `outputs/plots/`.
Both commands accept `--output PATH` to select another output directory.

The original standalone main-analysis executable is also supplied unchanged:
`python data/reproduce_statistics.py` writes its 44 contrasts as CSV to standard output.

## Files

- `data/run_metrics.csv`: 552 main-experiment runs across 23 paired repeats.
- `data/paired_contrasts.csv`: 44 paired contrasts with Holm-adjusted p values.
- `data/control_*.csv`: label-distribution and shuffled-label controls for
  retrieved EXAONE translations; 184 condition-by-repeat result rows.
- `data/kold_*.csv`: external evaluation on 40,129 KOLD comments, including
  276 condition-by-repeat results and eight paired contrasts.
- `data/annotation_audit_*`: aggregate results from four human raters and five
  LLM judges, all judging the same 300 translations in total.
- `data/native_threshold_metrics.csv`: recorded native-task threshold metrics.
- `data/diagnostic_tables/`: target-group recall, label transitions, and timing.
- `data/model_metadata.json`, `data/kold_dataset_metadata.json`: model versions
  and the public KOLD release used in the study.
- `data/execution_metadata.json`: public numeric seeds, sampling-role definitions,
  exclusions, and execution-order descriptions. Internal paths and private-input
  fingerprints are excluded.
- `reproducibility/`: original source row IDs and their request/retention order,
  together with a schema and checksum manifest. See its README for identifier semantics.
- `reproduce.py`, `data/reproduce_statistics.py`, `plot_results.py`: executable
  numerical verification and summary plotting.
- `SHA256SUMS.txt`: checksums of the distributed files, excluding this checksum file.

## Reading the results

Condition keys `exaone` and `gemma4` identify the translator or label source named
by the column; `source` denotes inherited corpus labels. Classifier keys
`kcelectra` and `xlmr` denote KcELECTRA and XLM-R. `retrieved` and `random`
identify the two sentence-selection conditions.

The control arms are S (source labels), L (EXAONE relabels), W (source-label
training weighted to match corpus-specific relabeled class counts), and
P (shuffled EXAONE relabels). The manuscript and supplement describe these procedures.
The six secondary control comparisons retain blank p-value fields, as reported;
the reproduction does not add hypothesis tests to them.

F1, recall, accuracy, and agreement proportions use the 0–1 scale unless a field
explicitly specifies percent or percentage points. In `annotation_audit_summary.csv`,
the three `*_agreement` columns are counts out of `n` (300), not proportions.
MCC and Cohen's kappa retain their native scale. Fields ending in `_gain_pp` are
percentage-point differences; JSON fields named `agreement_percent` are percentages.
Main, control-primary, KOLD, and post-hoc target families use separate Holm
corrections over 44, four, eight, and 14 comparisons. Performance-contrast confidence intervals are pointwise 95% intervals.

## Scope and privacy

Corpus utterances, translations, individual annotation responses or rationales,
and identifying participant information are not distributed here. Human raters use
pseudonymous identifiers in aggregate summaries. Original row IDs are provided
for training-source assignment reconstruction, not for human-response linkage.
Private annotation-file fingerprints and internal approval metadata are excluded.

The code reproduces reported contrasts from run-level metrics. It does not retrain
classifiers, regenerate translations/relabels, or rerun the item-level annotation
bootstrap. These steps require the underlying datasets, original model environment,
and item-level inputs described in the manuscript and supplement. The human point estimates are reconstructed from released category counts,
six pairwise agreement counts, and class-conditioned cross-tabulations.
Nominal alpha treats non-hate, hate, and unclear as distinct categories.
The 95% intervals are reproduced by taking percentiles of the saved 10,000
verified bootstrap estimates; the private item-row resampling itself cannot
be rerun from this release. Those estimates were generated with shared
translator-stratified item indices and all four raters kept fixed. Bootstrap
CSV estimates have 12 significant digits, and quantile comparisons use
an absolute tolerance of 1e-9; other numeric checks retain 1e-12.
Agreement with a fixed panel is not independent gold-label accuracy. Obtaining
the original source corpora remains subject to their distributors' access conditions;
this release does not grant additional access to them.

## Licensing

A separate reuse license has not yet been assigned. Public availability should
not be interpreted as a CC BY or software-license grant. The original datasets
and models remain subject to their respective providers' terms.

## Updated post-hoc analysis files

- `data/human_reinforcement_summary.json`: fixed-panel alpha, agreement gains and 36 aggregate class cross-tabs.
- `data/human_bootstrap_estimates.csv`: the saved 10,000 jointly sampled alpha and panel-gain estimates, with no item identities or responses.
- `data/target_recall_reproduction.csv`: all model-by-arm-by-repeat-by-target recalls, checked against the original public target table.
- `data/target_lw_repeat_differences.csv`: 322 paired L-minus-W recall differences.
- `data/target_lw_contrasts.csv`: 14 paired t22 summaries, pointwise 95% intervals and Holm14 adjusted p values.
- `data/target_lw_turnover_by_repeat.csv`: 414 aggregate four-way error tables.
- `data/target_lw_turnover_summary.csv`: 18 descriptive summaries across the paired repeats.
- `data/target_lw_item_turnover_histograms.csv`: ID-free frequency bins of correctness counts across 23 repeats, separately within each model and stratum. They reconstruct the unique-ever-recovered/error counts without releasing evaluation IDs.
- `reproduce_reinforcement.py`: standalone public-input verification of these additions.

Target recall uses the original overlapping target-positive masks restricted to human binary hate.
Denominators are gender 232, age 102, race or region of origin 893, religion 46,
politics 1,790, occupation 365, and disability 99. The 23 paired repeats vary the
training/native-sampling bundle while the gold corpus and input pools stay fixed.
Turnover is descriptive: repeated observations on the same comments are not
independent samples. Overall non-hate correctness is available; positive-only
target masks do not estimate target-specific false-positive rates. These are
post-hoc diagnostics, not new-corpus confirmation or a fairness guarantee.

The release contains no manuscript PDF, publisher full text, private input
fingerprints, new evaluation row IDs, or individual human-response matrix.
Existing ID-only training assignments retain their previous public scope.
