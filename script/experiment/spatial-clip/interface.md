# E2 experiment interface

`run.py` exposes `qualify`, `extract`, `qualify-peripheral`, `derive`, `preflight`,
`train`, `validate`, `freeze`, `test`, `report`, and sequential `all`. See README
for checkout-relative commands and the fixed scientific protocol.

Output root: `VINED_E2_OUTPUT_DIR` or `output/experiment/spatial-clip/e2`, confined
to the spatial-clip experiment directory. Immutable JSON records refuse changed
identities; feature archives and scientific generation publishers retain their
normal overwrite protection. Partial stages are not complete generations.

- `qualification.json`, `qualification-panel/`: training-only image provenance,
  exact prepared RGB hashes, lossless previews, fresh/archived features and errors.
- `peripheral-qualification.json`: fixed visible peripheral common-pass check.
- `source-bundle.json`, `features/{A,B,G}.npz`: independently verified source
  generations; all observations/statuses remain represented.
- `derived/{arm}.json`: source/alignment/dataset IDs and exact-reference matching.
  `alignment/` and `datasets/` contain standalone unpadded generations.
- `preflight.json`, `runs/{arm}/resolved.json`, `initial.pt`: full shared settings,
  reference support, anatomical mapping/baseline, initialization hashes/shapes,
  active/registered counts and observed batch support.
- `runs/{arm}/training/`: production checkpoints and histories.
  `runs/{arm}/result.json` binds the validation-selected checkpoint and resources.
- `evaluation/{val,test}/{arm}/`: hashed per-trial NPZ predictions,
  identity/subregion index, setup/provenance and metrics. Prediction arrays retain
  original bins, physical bin centers/edges, observed counts, predicted log
  expected counts and eligible/primary masks.
- `{val,test}-comparison.json`: exact identity checks and named B−A/B−G/G−A
  comparisons, each with native and supplementary parent results.
- `frozen.json`: validation-only decision and fixed model/provenance/code hashes;
  required unchanged before any test inference.
- `{val,test}-report.md`, `training-validation-curves.png`: readable diagnostics.
  `completion.json` is written after all three trained arms and final comparisons.

Each time bin is one temporal token. Counts are spike counts per original 1/60 s
bin; predictions are log expected counts. Primary reporting selects original
indices 11..145 where present; model optimization/selection use 11..T−1.
Subregion order and baseline are shared exactly across arms. Test payloads never
select checkpoints, parameters, support, or feature variants.
