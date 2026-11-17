# Subregion aggregation audit

## Summary

The regional experiment provides reusable controlled runs and saved predictions
for D=0 and D=6. Its sum-count contract permits exact additive reconstruction.
The implementation initially lacks recorded-subregion runtime targets and a
comparison across granularities.

## Findings

### A01 — Subregion targets are unsupported

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Pipeline

**Location:** `src/trainer/targets.py`, `src/trainer/setup.py`,
`src/multi_modal/mm.py`, `src/evaluation/setup.py`.

**Finding:** Only neuron and parent-region targets are accepted. Original Allen
anatomy survives dataset publication, but the regional transform merges layers.

**Expected:** Preserve the selected neurons and refine existing parents by
recorded anatomical labels, with sum-count outputs and checkpoint-bound mapping.

**Suggested disposition:** Wrap / centralize

### A02 — Existing comparison requires identical target granularity

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Data integrity

**Location:** `script/experiment/target-delay/common.py`, `compare.py`, `analysis.py`.

**Finding:** Existing comparisons match native targets directly. Experiment 1
uses H=12/S=24 from the wider delay sweep; recomputing support for D=0/6 alone
would broaden observations. Native losses with different channel definitions
are not directly comparable.

**Expected:** Reconstruct parent intensities/counts, retain original support,
verify original neuron/trial/bin provenance, and compare identical parent targets.

**Suggested disposition:** Wrap / centralize

## Conforming areas

- Source validation retains stable unit identities and original Allen anatomy;
  no additional firing-rate filtering is needed.
- Runtime parent targets are int64 summed counts with explicit source columns
  and neuron counts; summing refined counts has an exact inverse relationship.
- Existing linear regional heads and Poisson objective support variable target
  widths without changing the transformer.
- Production training supplies independent initialization, validation selection,
  checkpoint history and resumable state. Experiment stages retain provenance.
