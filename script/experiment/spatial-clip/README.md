# E2 spatial CLIP representation comparison

Run from the checkout root with the existing local environment:

```powershell
.venv\Scripts\python.exe script/experiment/spatial-clip/run.py all
```

The fixed reference session, replay, source, alignment, and E0 dataset identities
are declared in `common.py`. Outputs default to
`output/experiment/spatial-clip/e2`. Existing generations are never replaced.
Set `VINED_E2_OUTPUT_DIR` to a fresh directory under
`output/experiment/spatial-clip` for a separate execution.

The individual stages are:

```powershell
.venv\Scripts\python.exe script/experiment/spatial-clip/run.py qualify
.venv\Scripts\python.exe script/experiment/spatial-clip/run.py extract
.venv\Scripts\python.exe script/experiment/spatial-clip/run.py qualify-peripheral
.venv\Scripts\python.exe script/experiment/spatial-clip/run.py derive
.venv\Scripts\python.exe script/experiment/spatial-clip/run.py preflight
.venv\Scripts\python.exe script/experiment/spatial-clip/run.py train
.venv\Scripts\python.exe script/experiment/spatial-clip/run.py validate
.venv\Scripts\python.exe script/experiment/spatial-clip/run.py freeze
.venv\Scripts\python.exe script/experiment/spatial-clip/run.py test
.venv\Scripts\python.exe script/experiment/spatial-clip/run.py report
```

`derive`, `train`, `validate`, and `test` accept `--arm A|B|G`; omitting it
processes one arm at a time in A/B/G order. Call `validate` without `--arm` to
assemble the paired validation comparison before `freeze`. Inspect qualification
images/results and preflight before training when running stages manually.
`all` freezes all three predeclared arms and their validation-selected checkpoints
before final test inference; it performs no test-based tuning or automatic
superiority declaration.

A/B/G are extracted in one frozen CLIP vision pass with float32 weights, CUDA
float16 autocast, SDPA, batches of 64 and a 512-image bounded cache. A is 768
projected pooled CLS features. B is 4096 flattened raw final-layer quadrant means.
G is the normalized global raw patch mean repeated four times. Each section is
normalized as one vector. Full patch states are not persisted. The training-only
qualification panel covers consecutive images, repeated prepared pixels, the
64-image boundary, stationary/closed-loop/freeze/blank states. The fixed visible
peripheral supplement is trial 83, observation 5; its initial panel's largest
azimuth member is explicitly identified as zero contrast.

Derived generations copy E0's brackets and weights, neural arrays, all complete
60 Hz bins, original identities and persisted split membership. Publication fails
if any of the 312 eligible reference trials is lost. Neural equality checks are
mechanical provenance checks, including test payload equality; they do not select
representations or settings.

Preflight copies every shape-compatible state from an untrained seed-42 A
reference to each arm and copies the complete B initial state to G. It records
initial tensor hashes, shapes, unmatched width-dependent keys, registered and
actual encoding-gradient parameter counts, real batch masks, and strict windows.
Training uses the unchanged E0 H12 50-epoch protocol and full S12 support; no
trained checkpoint initializes an E2 arm. The optional visual decoder remains.

`val-comparison.json` and `test-comparison.json` include native/parent primary,
full, and tail metrics; cell-weighted/equal-trial NLL and constant-baseline gains;
W=1/6/12 correlations/variance ratios; paired trial/channel differences; undefined
support and all subregions. Pair names are B-minus-A, B-minus-G, G-minus-A;
negative NLL differences favor the first arm. Parent predictions sum intensities.
`frozen.json` binds code, provenance, checkpoints, and the validation decision.
`training-validation-curves.png` shows full-support losses and selected epochs.

The grid/global contrast is the main spatial control. G has lower effective
input rank, and both patch representations contain contextualized information.
A/B differ in representation family and active capacity. Bin-center interpolation
can use a later source bracket, so the experiment is offline aligned. E0's prior
use of the same test membership prevents an independent confirmatory test claim.
These single-seed short runs do not establish convergence or superiority from
tiny global NLL differences.
