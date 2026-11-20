# E0 controlled temporal horizon comparison

From the checkout root, use the project Python environment. This isolated experiment compares fresh strict visual encoding at H=6 and H=12, D=0, with S=12 and all recorded anatomical subregions. It does not consume historical checkpoints. Production architecture, alignment, aggregation and general defaults are unchanged.

## Commands

Preflight resolves **both** arms, checks the real persisted generation and source-neuron partition, constructs both models without optimization, and records matching parameter dimensions:

```powershell
.venv\Scripts\python.exe script/experiment/temporal-horizon/run.py preflight
```

Training is an explicit action, separate for each arm:

```powershell
.venv\Scripts\python.exe script/experiment/temporal-horizon/run.py train --arm H6
.venv\Scripts\python.exe script/experiment/temporal-horizon/run.py train --arm H12
```

After both runs finish:

```powershell
.venv\Scripts\python.exe script/experiment/temporal-horizon/run.py evaluate --arm H6 --device cuda
.venv\Scripts\python.exe script/experiment/temporal-horizon/run.py evaluate --arm H12 --device cuda
.venv\Scripts\python.exe script/experiment/temporal-horizon/run.py compare
```

Use `--device cpu` for CPU inference. On Linux substitute `.venv/bin/python`.
Commands never implicitly train another arm or choose a winning model. Repeated stages create unique attempts; completed manifests select the current explicit attempt and verify every artifact hash. A running/abandoned stage lock fails visibly. Resume is not exposed by this bounded experiment wrapper.

## Configuration and identity

`configs/experiment.json` declares the generation, expected content ID, session, fixed E0 policy and isolated output root. `configs/training.json` contains one shared production override object. The initial budget is 50 epochs, seed 42, batch size 1, four-step gradient accumulation, AdamW at 0.0002, no scheduler, validation each epoch, and minimum validation-loss checkpoint selection. The two arms differ only in `training.temporal_context.bins` and their output identity. D=0 is the production default and can be absent from resolved context fields; it is explicit in experiment metadata.

To change the epoch budget, edit shared `training_overrides.training.num_epochs` before preflight. Once an output root is bound, configuration changes require a new `output_root` beneath `output/experiment/temporal-horizon/`. Point to an alternate experiment JSON with `--experiment-config PATH` on every command. All scientific inputs use production resolution; unsupported fields or incompatible policies fail instead of being silently overridden. Historical experiment directories are never output destinations.

## Temporal and metric contract

Encoding reads exactly `t-H+1..t` within the same trial. The existing independent-window model implementation remains authoritative; no new attention mask is used. Both loss and validation selection use all real target bins `t>=11`. H6's available predictions at bins 5..10 do not contribute.

The median uses **training split original sequence lengths only**. Primary reporting scores complete bins `11<=t<min(T,floor(median_train_length))`. A fractional median never admits a bin extending beyond its endpoint. For median 146 this includes t=145 and excludes t=146. Full eligible support and the disjoint tail starting at `max(11,floor(median))` are supplementary. No scientific arrays, physical coordinates, trial membership or validity masks are changed.

The constant baseline is prescribed, not selected: each native subregion's mean count over **full eligible training S=12 support**. This same baseline initializes output biases and supplies training validation BPS and offline NLL. It is not a median-restricted baseline. Likelihood uses `exp(logp)-y*logp` (Poisson NLL, `log_input=True`, `full=False`); baseline means are floored at 1e-8 for likelihood only. NLL improvement is baseline minus model loss.

For each trial and native channel, W=1/6/12 diagnostics use trailing mean counts. Each W uses every complete endpoint inside its own declared interval (the first endpoint is interval start + W−1); smoothing cannot cross trial, median or tail boundaries. Pearson correlation is undefined for fewer than two endpoints or a constant series. Variance ratio is `var(prediction)/var(target)` with population variances; constant targets are undefined. All anatomical channels, including low-activity groups, stay present. Parent counts, predicted intensities and training constants sum native groups; log outputs are never summed.

NLL summaries include global target-cell weighting and equal-trial means. Diagnostic summaries include endpoint-cell weighting and equal-trial means of defined channel scores. Population standard deviations, quartiles, defined/undefined contributor counts and reasons are explicit. Fractions improved against the constant and paired fractions favoring each arm use only trials with support; ties and missing trials remain counted. Paired NLL differences are H12 minus H6, so negative favors H12 numerically. Paired diagnostic differences use contributors defined in both arms.

Supplementary full/tail/parent results never select a primary winner. These short runs identify exploratory relative differences, not converged performance or statistical superiority.

## Artifacts

Outputs live only under `output/experiment/temporal-horizon/` by default:

- `manifest.json`: immutable controlled experiment configuration, support, baseline and dataset identity.
- `preflight.json`: resolved parameter shapes/count, source hashes, native/source mapping, counts and representative trial boundaries. `runs/H6/resolved.json` and `runs/H12/resolved.json` preserve full effective configurations.
- `runs/H*/train-attempts/<id>/`: production checkpoints, configuration/provenance, history and selection records; `result.json` binds the selected checkpoint hash/epoch and curves to the resolved run. Source hashes are recorded per training attempt.
- `runs/H*/evaluate-attempts/<id>/`: inference provenance, native and supplementary parent metrics, plus `predictions/index.json` and numbered NPZ files. Each NPZ retains original `[T,G]` int64 `observed` counts and floating `log_expected`, original `[T]` bin indices, physical centers/edges in seconds, eligible/primary masks. G denotes ordered native subregions; the index retains session, original trial/sample identity, channel identities and metadata. Arrays contain all original time positions; only masks define scored support.
- `comparison/compare-attempts/<id>/`: `comparison.json`, readable `report.md`, both full training histories and `training-validation-curves.png`. Pairing rejects changed checkpoints, different configuration except H, dataset/split/mapping/baseline mismatches, changed implementation sources and unequal observed records. Comparison performs no inference.

`train.json`, `evaluate.json` and `compare.json` point to completed attempt directories. Failed attempts remain inspectable and cannot be consumed as completed results. No model superiority is selected automatically. See [verification.md](verification.md) for the manual readiness evidence and execution limits.
