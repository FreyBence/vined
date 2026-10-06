# Alignment audit

## Summary

Alignment is partially implemented through `src/prepare_data.py`,
`utils.ibl_data_utils`, and `utils.visual_data`. The active path is a legacy
joint preparation/dataset pipeline, not the stimulus-bounded alignment boundary
required by `spec.md`. Current visual generations cannot enter that path;
prepared neural generations are not consumed. Timing, coverage, population
identity, and output provenance require migration before this path satisfies
the accepted design.

This audit uses source inspection and the public visual-features and neural-data
interfaces. No session execution was performed. Alignment has no active
`architecture.md` or `interface.md`. Findings concern alignment and its observed
handoffs; dependency internals and archives are outside scope.

## Findings

### A01 — Current prepared modality contracts are not consumed

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Interface

**Location:**  
`src/prepare_data.py:60–65`; `src/utils/visual_data.py:16–22`;
`src/utils/ibl_data_utils.py::prepare_data`, `bin_spiking_data`.

**Finding:**  
The preparation entry point requires the obsolete top-level visual `provenance`
array. Current feature archives instead expose definition, observation records,
features, and a verified manifest through `FeatureArtifactReader`; the legacy
loader explicitly rejects them. Regenerating current features does not fix this
failure. On the neural side, the path loads a source population and independently
bins spikes rather than consuming `NeuralCounts`/`CountWindow` and their explicit
edges, validity, and source identity. Neither current prepared modality boundary
is integrated into alignment.

**Expected:**  
Specification §§2–5 and `docs/dependencies.md` require alignment to consume
already prepared visual and neural representations through its two declared
upstream interfaces.

**Suggested disposition:**  
Wrap / centralize — retain usable source functionality behind the upstream
boundaries and adapt alignment to the current prepared representations.

---

### A02 — Alignment retains a fixed pre-stimulus window and ignores stimulus offset

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Pipeline

**Location:**  
`src/prepare_data.py:40–47,148–151`;
`src/utils/ibl_data_utils.py::bin_spiking_data`, `bin_behaviors`,
`get_spike_data_per_interval`.

**Finding:**  
Both modalities use `[stimOn - 0.5, stimOn + 1.5)` and a fixed 100-bin sequence.
The first visual query is `stimOn - 0.49`. No alignment code uses `stimOff`.
Consequently pre-stimulus neural observations remain in retained trials,
post-offset observations may remain, and longer stimuli are truncated at an
unrelated endpoint. Neural allocation uses `ceil(interval_len / binsize)` while
visual binning requires an integer-sized fixed window; neither derives complete
bins from the actual stimulus duration. Visual centers are independently
constructed rather than checked against supplied neural edges.

**Expected:**  
Specification §§6–10 and 26 require onset-anchored complete neural bins,
`T = floor((stimOff - stimOn) / bin_size)`, centers from those bins, and exclusion
of the trailing partial interval.

**Suggested disposition:**  
Refactor.

---

### A03 — Incomplete visual streams are accepted and missing observations can be bridged

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Data integrity

**Location:**  
`src/prepare_data.py::prepare_session` (`allow_nans=True`);
`src/utils/ibl_data_utils.py::align_data` (`valid.any(axis=1)`);
`src/utils/visual_data.py::resample_features`.

**Finding:**  
Unavailable visual positions become zero placeholders with false masks, and a
trial survives if even one visual bin is valid. This exposes incompletely
covered sequences downstream. The legacy resampler checks only timestamp order,
range, and the validity of its two endpoints. If an expected observation is
absent from the arrays, it interpolates across the enlarged gap; it has no
continuity/coverage evidence to distinguish missing-data repair from resampling.
The current visual interface explicitly retains unselected and unavailable
observation metadata and trial coverage partitions for this distinction.

**Expected:**  
Specification §§5 and 13–15 require valid visual coverage at every retained
neural query, with missing expected observations treated as upstream failures.
Alignment must not publish missing observations as masked or synthetic data.

**Suggested disposition:**  
Refactor.

---

### A04 — Neural recording coverage is discarded

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Interface

**Location:**  
`src/utils/ibl_data_utils.py::load_spiking_data`, `prepare_data`,
`get_spike_data_per_interval`, `align_data`.

**Finding:**  
The legacy path returns spike times/assignments and unit metadata, bins tallies,
and considers every finite neural array acceptable. It does not consume neural
coverage status, observed duration, or per-cell validity. Unknown support and
known invalid support therefore cannot be distinguished from a valid zero-spike
bin. The neural interface explicitly says that raw tallies with unknown,
assumed, partial, or invalid support must not be used as fully observed targets.
The alignment specification's complete-coverage assumption does not currently
have an explicit implemented reconciliation with that upstream contract.

**Expected:**  
Consume the supplied neural timing/coverage evidence and resolve the documented
complete-coverage prerequisite explicitly. Do not silently promote unusable
counts to observed neural targets or infer recording support from spikes.

**Suggested disposition:**  
Refactor.

---

### A05 — Compact spike axes can be associated with the wrong neuron metadata

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Data integrity

**Location:**  
`src/utils/ibl_data_utils.py::get_spike_data_per_interval`, `bin_spiking_data`;
`src/prepare_data.py:79–102`.

**Finding:**  
Binning defines the neuron axis using `np.unique` of spike assignments, omitting
selected units with no events. `bin_spiking_data` returns the corresponding unit
positions, but the caller discards them. It then uses compact-axis firing-rate
positions to index full-population metadata. If an earlier selected unit is
silent, a later unit's counts can be labelled with the silent unit's metadata.
Anatomical subsetting can produce the same mismatch. Scoped source unit IDs,
collection, and revision also do not survive into the legacy metadata boundary;
optional UUIDs alone cannot establish unit identity.

**Expected:**  
Specification §§4 and 21 require neural values and neuron identities to remain
associated. Preserve the prepared population axis, including silent units, and
its scoped source identities through alignment.

**Suggested disposition:**  
Refactor.

---

### A06 — Trial identities and modality identity checks are insufficient

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Data integrity

**Location:**  
`src/utils/ibl_data_utils.py::prepare_data` (`original_trial_id = np.arange(...)`),
`bin_behaviors`, `align_data`.

**Finding:**  
The trial table's original index is replaced as the visual lookup identity by
new positional IDs. This disagrees with the neural interface when original
indices are nonconsecutive. `align_data` receives only modality arrays and an
optional mask index, so matching array lengths are sufficient even if neural
and visual arrays belong to different trials or sessions. The legacy visual
loader checks its EID against a filename-selected session, but this does not
establish a paired neural/visual identity at the alignment entry point.

**Expected:**  
Specification §§3,16–17,21 and 24 require original trial identity preservation
and rejection of session/trial mismatches before temporal mapping.

**Suggested disposition:**  
Refactor.

---

### A07 — Alignment owns acquisition, selection, and dataset responsibilities

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Architecture

**Location:**  
`src/prepare_data.py::prepare_session`;
`src/utils/ibl_data_utils.py::prepare_data`, `load_trials_and_mask`, `align_data`.

**Finding:**  
The sole active alignment launcher creates `SessionAccess`, loads probe/trial
sources, selects anatomy and responsive units, imposes behavioral eligibility,
and then splits and constructs Hugging Face datasets. `align_data` additionally
standardizes an optional LFP modality. Thus alignment has no independent public
operation accepting its two prepared modalities and returning unsplit aligned
trials. Its launcher directly depends on session-data and training-dataset
implementation despite the declared alignment dependencies being only
visual-features and neural-data. Working acquisition and packaging code need
not be rewritten, but their ownership must be separated from alignment.

**Expected:**  
Specification §§2 and 25 and `docs/dependencies.md` assign acquisition/neural
preparation upstream and splitting, padding, batching, and packaging downstream.

**Suggested disposition:**  
Wrap / centralize.

---

### A08 — The output lacks the required aligned temporal contract

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Spec

**Location:**  
`src/utils/ibl_data_utils.py::align_data` return;
`src/prepare_data.py:148–179,187–233`.

**Finding:**  
The alignment result is a tuple of arrays and selection masks. It carries no
session/trial keys, neuron identities, physical bin edges/centers, exact stimulus
offset, aligned endpoint, discarded tail, or resampling policy. The published
dataset/provenance restores some session and positional trial information and
the fixed window configuration, but does not restore the missing stimulus/grid
contract or current prepared generation identities. There is no standalone
aligned-trial output for the next component to consume.

**Expected:**  
Specification §§18–22 requires synchronized `[T,N]` and `[T,D]` observations
with identities, explicit physical time, exact stimulus bounds, complete-bin
metadata, resampling configuration, and relevant source/preprocessing identity.

**Suggested disposition:**  
Refactor.

## Conforming areas

- The legacy visual loader checks session identity, unique nonnegative trial
  IDs, timestamp order, feature width, and numeric validity.
- Visual resampling uses physical bin-center queries and L2-normalized linear
  interpolation. It avoids extrapolation and pairs touching explicitly invalid
  source observations; preserve these mechanics while adding coverage checks.
- Zero spike counts are not themselves used to reject neural bins in
  `align_data`.
- The legacy launcher records source/configuration fingerprints, checks that
  visual input bytes did not change during processing, and stages publication
  before renaming to a fresh destination.
- Caller/launcher inspection confirms the legacy helpers remain reachable from
  `script/prepare_data.sh`; they are not established dead code. No removal-only
  dead-code finding is justified by this audit.
