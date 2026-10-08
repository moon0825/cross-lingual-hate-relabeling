# ID-only training assignments

These files contain original source row identifiers and ordering only. They do
not include source texts, translations, training/model labels, evaluation data,
or links to individual human responses.

- `requested_assignments.csv`: 23 repeats × two selection methods × 2,000
  requested source rows before translation exclusions; common to both translators.
- `retained_assignments.csv`: actual translated-source rows used for training,
  by translator, selection, and repeat. Each assignment is shared across all
  three label conditions and both classifiers.
- `native_training_ids.csv`: 200 K-HATERS training rows per repeat, common to
  all paired conditions.
- `export_manifest.json`: exact schemas, counts, and checksums of these three files.

`repeat` is 1–23, corresponding to the ordered `repeat_seeds` in
`../data/execution_metadata.json`. All order columns are zero-based.
`selection_order` records the original position among 2,000 requests and can
have gaps after exclusions. `retained_translation_order` is contiguous among
retained translations. `training_frame_order` is the contiguous position after
the 200 native examples, before seeded DataLoader shuffling. The original
engine's canonical position is `selection_order + 200`.

The assignment key `exaone40` refers to the same EXAONE translator called
`exaone` in the statistical tables; `gemma4` has the same meaning in both.

`source_corpus` and `source_row_id` jointly identify a record in the original
CuratedHS or MetaHate source table. K-HATERS `split_row_id` is its zero-based
position in the original released split; only its training split is included.
These are original corpus row identifiers, not newly assigned anonymized IDs.
The corpus versions and preparation are described in the manuscript and supplement.
Access to the original datasets remains subject to their distribution terms.

The original assignment export was checked against all 552 frozen training
inputs. This public copy preserves all three exported CSVs byte for byte;
publishing identifiers does not modify the experimental assignments.
