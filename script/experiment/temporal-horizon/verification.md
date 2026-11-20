# E0 manual readiness verification

Preflight ran against generation `01e5d3830786d45e1aa6f074d3e6c08fb0e476165f93b11a94d605edefb7995c`, session `0cad7ea8-8e6c-4ad1-a5c5-53fbb2df1a63`. Artifacts are under `output/experiment/temporal-horizon/`. Training was not started.

## Configuration and population

Both production configurations resolve to strict encoding, D=0, S=12, visual-only `vision-clip` input and unfiltered `all_recorded` native subregion targets. Their sole configuration difference is `training.temporal_context.bins` (6 versus 12). Parameter names/shapes match: **957,325 parameters** per model. Both use seed 42, identical optimizer/batches, an initial 50-epoch budget, and full-support validation-loss checkpoint selection.

All **280 recorded neurons** partition into the same **13 groups**, without omitted or duplicated source columns: APN, CA1, DG-mo, DG-sg, MB, VISpm1, VISpm2/3, VISpm4, VISpm5, VISpm6a, bsc, fp and or. Groups containing only one or two neurons remain present. Runtime integer target sums were checked against original neuron arrays across every split. Production full-support training means match the E0 support means exactly.

## Original temporal support

The median of the 218 training trial lengths is **146 complete bins**. Validation has 31 trials and test has 63 trials.

| Split | Full S=12 positions | Primary positions | Primary cells |
|---|---:|---:|---:|
| Train | 123845 | 23841 | 309933 |
| Validation | 20448 | 3468 | 45084 |
| Test | 33773 | 6730 | 87490 |

Every trial's production context view has target indices exactly `11..T-1` for both H values. The first supervised target reads H6 indices 6..11 or H12 indices 0..11; last-target slices end at the target without accessing future bins or another trial. Encoding receives visual features, positions, masks and session identity without neural targets. The existing independent-window implementation remains unchanged.

Stored representative boundaries include training trials 1 (161 bins) and 2 (91 bins), validation trial 3 (485 bins), and test trial 10 (3631 bins). Primary intervals stop at min(T,146): t=145 is included where present, t=146 excluded. Flooring a fractional median excludes bins extending beyond its endpoint. Trial IDs, physical coordinates, arrays and original validity remain intact.

## Direct execution and artifacts

Bounded CPU inference with initialized, untrained models ran on real test trials 45 (75 bins) and 179 (148 bins). Both returned `[1,T,13]` log-count tensors. H6 prediction availability began at bin 5, H12 at bin 11, and target eligibility began at bin 11 for both. Primary slices were `[11,75)` and `[11,146)`.

Native primary/full/tail metrics, W=1/6/12 diagnostics, parent reconstruction and paired metrics executed on those inference outputs. The two trials contributed 2587 native primary cells and 1592 parent primary cells. Pairing each collection with itself produced exactly zero primary NLL difference. Trial 45 had no tail and remained explicitly counted as lacking tail support. `manual-inference.json` serialized and read back both arms' summaries. These outputs demonstrate readiness only, not trained E0 performance.

Code inspection confirms interval slicing precedes smoothing; parent predictions sum intensities before taking logs; observed parent counts and constants sum the same groups. Undefined diagnostics retain reasons/counts and no activity threshold filters identities. The existing Ruff check passed for the new sources. No tests, temporary test scripts or CI infrastructure were added. Historical experiment source, checkpoints and outputs were not modified.

## Execution limits

There is no implementation blocker to launching the arms using [README.md](README.md). Full training, selected-checkpoint restoration through the E0 evaluation command, and the final two-trained-model comparison remain unexecuted because training is explicitly excluded from this task. Their production API boundaries and persisted schemas were inspected. Actual curves, selected epochs and comparative performance require those runs. No winner or trained-model baseline was selected.
