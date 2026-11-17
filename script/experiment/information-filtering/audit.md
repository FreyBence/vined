# Information filtering audit

## Summary

Experiment 2 preserves recorded units, Allen IDs, physical bins, and source-column
partitions. Original int64 counts permit exact full-region ground truth. CLIP
features are available per bin; contrast is optional persisted trial metadata,
not a substitute for visual features. Existing stages provide reusable training,
checkpoint, support, metrics and plotting infrastructure.

## Findings

### A01 — Train-only screening and filtered restoration are absent

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Pipeline

**Location:** `src/trainer/targets.py`, `src/trainer/setup.py`.

**Finding:** Anatomical grouping preserves all neurons but cannot consume a fixed
train-derived subset. No inexpensive out-of-fold information scoring exists.

**Expected:** Delay-independent selection, recorded source identities and
checkpoint-bound filtered counts, without changing visual data or architecture.

**Suggested disposition:** Wrap / centralize

### A02 — Common-target filtered predictions and validation comparison are absent

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Data integrity

**Location:** `src/evaluation/setup.py`, Experiment 2 evaluation/comparison scripts.

**Finding:** Existing evaluation restores all groups and restricts inference to
test. Summing filtered predictions alone would compare incomplete populations.

**Expected:** Validation inference, full original regional observations, excluded
train-mean contributions, and separate native diagnostics and paired weighting.

**Suggested disposition:** Wrap / centralize

## Conforming areas

- H=12/S=24 maintains Experiment 1/2 common support, including D=0 and D=6.
- Subregions partition parent areas by recorded Allen labels; no invented layers.
- Existing production trainer supplies fresh initialization, best validation
  checkpoint, latest optimizer state and epoch-boundary resume.
- Existing offline metrics use training means and explicit valid-bin support.
