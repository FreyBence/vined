# Neural Data Specification

## 1. Purpose and status

The `neural-data` component produces interpretable neural activity representations and associated metadata from session electrophysiological data. Its outputs support downstream alignment with visual observations and modeling of neural responses.

This is a proposed specification under the component boundaries defined in `dependencies.md`. It defines required behavior without prescribing package structure, concrete APIs, storage schemas, or processing algorithms.

The earlier NEDS implementation and project report provide background. Their model encoders, fixed dimensions, sampling settings, and processing layout are not mandatory compatibility requirements. This document does not claim that the required behavior is already implemented.

## 2. Scope and ownership

The initial supported input is previously spike-sorted electrophysiological data: spike times, spike-to-unit assignments, and relevant recording and unit metadata. The initial neural activity representation is spike counts per selected unit over explicit time intervals.

`neural-data` owns:

- interpretation and validation of supplied spike and unit data;
- explicit recording, unit-quality, and anatomical selection;
- formation of neural activity representations over requested source-time intervals;
- preservation of recording, unit, session, and original trial identity where applicable;
- neural coverage, validity, processing provenance, and reusable output loading.

| Component | Responsibility |
| --- | --- |
| `session-data` | Source acquisition, session identity, dataset discovery, collections/revisions, local materialization, and source metadata |
| `neural-data` | Neural selection, spike-count formation, neural metadata, and source-time validity |
| `alignment` | Common visual/neural temporal structure, cross-modal compatibility, temporal aggregation/resampling policy, and joint coverage decisions |
| `training-dataset` | Model-ready sample construction, padding, and experiment split application |
| `model` | Learnable neural encoders, projections, tokenization, and session adaptation |
| `training` / `evaluation` | Optimization and assessment of model predictions |

The component depends only on `session-data`. It must be usable without visual replay, visual features, or model internals. It must not independently implement ONE/Alyx acquisition or project-wide source cache policies.

## 3. Input requirements

Input must provide enough information to identify and interpret:

- the session and source recording/probe or insertion;
- the selected spike-sorting output and its source dataset/revision identity;
- each spike's timestamp and assigned source unit/cluster;
- the unit identities to which spike assignments refer;
- timestamp units, clock reference, and any supplied synchronization mapping;
- recording coverage and known invalid intervals, or the limitations of that information;
- quality or anatomical metadata required by the requested selection policy;
- original trial identities and source trial-table identity when trial-based output is requested.

Recording channels, sorted units, and array positions must not be treated as interchangeable identities. The component consumes existing sorting results; it does not perform raw-voltage filtering, spike detection, spike sorting, or manual curation in its initial scope.

Multiple source recordings and sorting versions must remain distinguishable. A request must resolve which sources are used; the component must not silently choose a different sorting result when a requested source is unavailable.

Essential missing or contradictory information must produce an explicit failure or unavailable result. Optional metadata may remain unknown when the requested processing does not depend on it.

## 4. Unit selection and metadata

### 4.1 Explicit selection

The component must support selection by recording, available unit-quality information, and anatomical region. Effective criteria and their interpretation must be recorded. Any default is part of the documented processing configuration rather than a hidden filter.

Quality labels and metrics retain their source meaning. A label such as `good` must be interpreted according to the supplied metadata rather than an assumed universal numeric code. Missing required quality information must not silently pass a quality filter.

Anatomical selection must distinguish exact region matching from inclusion of descendants or subregions. Record the atlas/mapping identity and effective selected regions where relevant. Missing anatomical assignments remain unknown and must not be fabricated or silently treated as a match.

No fixed anatomical region list, quality threshold, unit count, or maximum population size is imposed by this specification.

### 4.2 Stable population identity

Every output unit must map unambiguously to its original source unit within its session, recording, and sorting context. Unit identifiers need not be contiguous or globally unique on their own.

Selection and ordering must be reproducible. All intervals in a generation use the declared unit axis; a unit must not disappear or change position because it has no spikes in a particular trial. Unit metadata must remain aligned with the activity values through filtering, storage, and loading.

If multiple recordings are combined, preserve the recording identity and coverage of each unit. Do not merge units merely because they share a cluster number or anatomical label. Corresponding array positions in different sessions do not establish correspondence between biological units.

Different sessions may contain different numbers of units. Model-specific padding, truncation, neuron matching, and learned population adaptation are outside this component.

## 5. Neural activity representation

### 5.1 Spike-count semantics

For each requested interval and selected unit, the initial representation reports the number of assigned spikes in that interval. The population is represented by separate unit values, not collapsed into one count summed across units.

Counts must be finite, nonnegative, integer-valued quantities. Each included source event contributes once to its unit's count within a given interval. A count of zero is valid only where the corresponding neural observation is valid.

The representation definition must identify the value semantics, unit axis, time intervals, and any applied transformation. Spike counts, binary spike occurrence, firing rates, smoothed signals, and normalized values must not be presented as interchangeable representations.

The initial output preserves unsmoothed, unnormalized counts. A future derived representation requires an explicit definition and distinct processing provenance. A learned neural encoder or a transformer token sequence is not a neural-data output requirement.

### 5.2 Counting intervals

Temporal resolution and requested domain must be configurable. Actual interval boundaries in source-session time must be available to consumers; bin centers or a nominal bin size alone are insufficient.

The default temporal bin size is exactly `1/60` second (approximately 16.67 ms,
60 Hz), matching the confirmed experimental visual projection and replay cadence.
The previously used 20 ms bin size is the reference temporal resolution;
candidate sizes for this experiment must remain within ±5 ms of that reference
(15–25 ms) to avoid substantially changing spike-count sparsity. The selected
16.67 ms bin is approximately 3.33 ms below the reference and satisfies this
constraint. This is the experiment's selection rationale, not a measured claim
that sparsity is unchanged. Explicit alternative resolutions and single-interval
counting remain supported for other requests.

Use half-open counting intervals `[start, end)`: include a spike at the left boundary and exclude one at the right boundary. Adjacent intervals therefore do not count their shared boundary twice. Record any shortened final interval explicitly; it must not appear to have the full nominal duration.

Overlapping requested windows may legitimately include the same spike in different samples. Preserve their physical intervals so downstream processing can detect that overlap. Processing batches must not introduce accidental duplication or omission within a requested interval.

The component must not clip counts to binary values, overflow silently, or change numerical meaning to fit a legacy dtype or model input width.

### 5.3 Boundary with alignment

Neural counting operates on explicitly requested physical-time intervals without consulting visual data. The configured neural binning is not an assertion that a common multimodal grid has already been established.

Alignment owns the choice and compatibility of the shared visual/neural time structure. It may consume an appropriately configured neural generation. Neural-data must expose sufficient interval and representation information to determine whether downstream aggregation is valid.

Binned counts do not retain spike positions within a bin. When a requested downstream grid cannot be constructed from the available representation without inventing that information, the mismatch must be explicit. The public neural-data processing boundary must permit regeneration from source spikes for the required intervals; alignment must not need private neural-data internals or independent source acquisition.

## 6. Time and trial identity

Spike times and output intervals must use the shared source-session time reference, retaining sufficient source precision for the requested binning. Any clock conversion must use an identified mapping and record its provenance. Do not infer time origin or synchronization from array positions or trial order.

For trial-based processing, the interval definition must explicitly identify its reference event and offsets or its absolute boundaries. Stimulus onset, response, feedback, and other events are not interchangeable. Missing required event times must not trigger an undocumented substitute.

Original trial IDs and the consumed source trial-table identity must survive filtering and output generation. Trials with no usable neural output remain traceable with a status and available reason. Filtering must not renumber later trials.

Neural processing must not discard a trial because visual data are unavailable or because the replay excluded it. Joint modality selection belongs to alignment. Trial counts and ordering alone are insufficient evidence that visual and neural records refer to the same trials.

## 7. Coverage, validity, and empty results

The output must distinguish:

| Situation | Required meaning |
| --- | --- |
| Valid observed interval with no spikes for a selected unit | Valid zero count |
| Missing recording or unavailable source interval | Unavailable neural data, not zero activity |
| Known invalid interval | Invalid observation with the available reason |
| No units satisfy the selection | Explicit empty selection, not a synthetic all-zero population |
| Missing required trial timing | Unavailable trial result, not an invented window |
| Processing failure | Failed processing, distinct from expected source unavailability |

Coverage must follow recording evidence. The first and last spike times alone do not establish recording start/end or the validity of silent intervals. Where coverage relies on a documented assumption, preserve that qualification; unknown coverage must not silently become confirmed full coverage.

Coverage may differ by recording and, where supplied, by unit. Combined outputs must preserve these differences rather than extending every unit over the union of recording times.

A count for a partially observed bin must not be presented as a fully observed count. Preserve the actual observed duration/coverage and distinguish partial observations from fully valid bins. If the implementation cannot represent that distinction, it must mark the bin unusable or reject the request rather than silently fill the gap.

Do not interpolate spikes across missing intervals, treat source padding as observation, or remove known gaps during temporal aggregation. Neural validity describes source support and processing; alignment decides whether that support suffices for a multimodal sample.

## 8. Output, provenance, and reuse

The component must provide loadable neural outputs together with:

- source session, recording, sorting, and original unit identities;
- original trial identity and source trial-table identity where applicable;
- activity values, ordered unit metadata, and representation semantics;
- actual interval boundaries, clock reference, and timing provenance;
- coverage, validity, trial/request outcomes, and available reasons;
- effective selection and processing configuration;
- source provenance and an identifiable completed generation.

Variable trial lengths and population sizes must remain representable without model padding or a fixed neural-bin count. Storage layout, sparse or dense organization, concrete dtypes, and public API signatures belong in the implemented interface.

Provenance must identify the content actually consumed, not merely its pathname. Relevant changes in sorting output, unit membership/order, timing, source trial table, selection, or binning must not silently reuse an incompatible result. Processing implementation and relevant configuration must remain traceable.

Loading must preserve the selected generation's identity and permit downstream compatibility checks. Older explicitly selected generations may remain usable for reproducing earlier experiments. Neural-data does not own recursive invalidation of aligned datasets or training caches.

A generation must not appear complete until its requested recordings, units, and intervals/trial outcomes have been accounted for. Completion accounting does not mean that every requested result is valid. Partial or failed processing must be reported explicitly, and an interrupted run must not silently substitute an older artifact as its newly produced result.

## 9. Correctness and practical processing

Validate essential source associations and numerical consistency, including finite interpretable timestamps, resolvable spike-to-unit references, unique unit identities within their declared scope, valid interval boundaries, and correspondence between count columns and unit metadata.

Processing must preserve associations if source events need ordering. It must not silently repair ambiguous unit references or deduplicate distinct spikes merely because timestamps coincide. Invalid source records require an explicit handling outcome.

Selection and count formation must be deterministic for identical sources and configuration, independent of processing batch boundaries. Processing must support practical session sizes without requiring a dense session-duration-by-unit array or retaining all processed sessions in memory.

Functional verification should establish count correctness at interval boundaries, preservation of silent units, separation of multi-recording identities, traceable trials, and correct treatment of missing/partial coverage. Poor historical prediction metrics alone do not demonstrate a neural preprocessing defect or justify changing neural targets.

## 10. Exclusions

The initial component does not own raw-voltage processing, spike sorting, visual reconstruction, visual feature extraction, cross-modal alignment, behavioral prediction, experiment split selection, model padding, trainable encoders, tokenization, model training, or prediction evaluation.

Training-derived normalization statistics and transformations requiring knowledge of training/validation/test membership are outside this initial component scope. Neural preprocessing must not fit such transformations across all sessions or trials as an implicit preparation step.

The historical NEDS bin width, fixed trial window, maximum unit count, archive schema, and model-specific neural encoder are not compatibility requirements. Additional signal modalities, general backend frameworks, and historical artifact migration are not required unless separately specified.
