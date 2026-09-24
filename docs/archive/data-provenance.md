# Reproducible visual datasets and caches

Implemented 2026-09-19 for D12, D13, and P05. Existing replay corrections are retained. No full session regeneration was performed for this change.

## Feature provenance

`prepare_visual_stim.py --clip-revision REVISION` resolves a Hub branch/tag once per invocation to an immutable commit SHA. Model weights and processor assets load from that same snapshot. The numeric feature archive adds a JSON-string `provenance` field and `clip_revision`, with downloaded artifact SHA-256 hashes, image processor settings, package versions, extraction source hashes, replay manifest hash, and sampling/normalization details. For an exact repeat, pass the recorded 40-character SHA instead of `main`.

## Trial splits

`prepare_data.py --split-seed 42` derives a deterministic seed from the EID and supplied seed. Sessions processed earlier in the invocation do not affect the result. Connected overlapping half-open neural windows are assigned together, preventing shared physical observations from crossing train/validation/test splits. Requested ratios are 70/10/20 percent of interval groups; trial ratios can differ. Fewer than three groups is an error rather than an empty split.

Every aligned row carries `trial_id`, `intervals`, `split`, and `provenance_id`. The session's `provenance.json` records split membership and algorithm, filtering settings, rejected IDs, neuron order, visual provenance and hash, and preprocessing source hashes. Firing-rate selection remains session-level QC; this change does not implement training-only neuron selection or fix the separate D08 cluster mapping issue.

After a successful preparation invocation, `split_independence.json` compares the configured `data/train_eids.txt` and `data/test_eids.txt` groups. Shared EIDs and subjects are reported separately; missing prepared sessions are listed as unverified. Within-session trial splits do not establish independence between subjects or task blocks. Use the report to assess the intended research protocol rather than inferring independence from disjoint trial IDs.

Aligned datasets publish from a temporary directory and refuse to replace an existing `<eid>_aligned`. Use a fresh data root when rebuilding.

## Cache generations

Cache layout is now:

```text
datasets/ibl_mm/
  <eid>.manifest.json
  generations/<generation-id>/<eid>_<split>_<original-trial-id>.npy
```

Multi-session cache roots keep their existing `ibl_mm_<count>` names. Each manifest lists every split's exact filenames, original IDs, array shapes, and file hashes, plus preprocessing options/source/package versions and aligned dataset file hashes. Readers reject mismatched requested padding/sorting settings, changed inputs, changed preprocessing, and legacy caches. Verification includes hashing aligned files at loader initialization and sample files on first access; this adds I/O overhead.

A complete generation is built before publishing session manifests with atomic file replacement. Interrupted builds leave unreferenced files; readers never scan these or old loose `.npy` files. Existing generations remain intact for in-flight readers. Publication is atomic per EID, not a transaction across all sessions. Concurrent complete writers use last-published manifest semantics. There is no automatic deletion of older generations.

Rebuild in order: extract features with revision metadata, prepare aligned datasets in a fresh root, then run cache creation. Keep `VINED_VISUAL_DIR` pointed at the intended feature directory. Existing training CLI cache roots remain unchanged.

## Local dependency management and verification

Install using the matching platform constraints file, run `pip check`, and use the existing offline environment check as documented in [environment setup](environment.md). Review package upgrades manually and update constraints deliberately. Feature archives continue to record the exact CLIP revision, artifact hashes, and package versions.

Local verification passed syntax checks and the existing offline environment check (imports, video/dataset round trips, attention/model execution, and CLI help). The existing environment still contains leftover `timm`, `torchvision`, and an editable `vined` distribution; those packages were not removed. Follow the clean-environment installation instructions when rebuilding it.

No test code was created or modified. Extraction with a real CLIP snapshot remains unverified.
