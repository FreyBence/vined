# Session data implementation plan

## Outcome and scope

Provide the shared source-access boundary defined by [spec.md](spec.md) and [architecture.md](architecture.md), resolving the gaps in [audit.md](audit.md) through incremental reuse of ONE, Brainbox, and SpikeInterface. Follow the component boundaries in [dependencies.md](../dependencies.md).

The plan includes the source-access call-site migration required in visual replay and neural preparation. Consumer algorithms, experiment selection, and specialized inventory tooling remain outside scope. Create and maintain `interface.md` alongside each implemented public capability; document actual requests, results, source identity, access policy, and failures rather than a proposed API.

Validate within each task using code inspection, existing checks, and direct execution on representative available sources. Check identity, source selection, local-only behavior, and meaningful failure conditions where relevant, without adding test code or validation infrastructure.

## Tasks

### [x] SD01 — Establish shared session access and acquisition policy

**Summary:**  
Expose a recognizable session-data boundary with canonical EIDs, shared ONE configuration, the existing configured source cache, and explicit local-only or remote-allowed access. Reuse generic identity and path helpers. Distinguish invalid identities, unresolved sessions, access failures, and unavailable local sources without treating valid empty data as missing. Apply this policy consistently to the source capabilities added below. Addresses the shared foundation of A03–A04.

**Out of scope:**  
A second source-data store, experiment-role semantics, downstream orchestration imports, and a universal data container or elaborate error framework.

### [x] SD02 — Expose metadata, discovery, and coherent named source access

**Summary:**  
Provide session/probe metadata, dataset discovery, and access to trials, wheel, timing/synchronization, and requested raw task evidence through supported IBL mechanisms. Preserve explicit collection/revision requests and expose effective dataset identity, provenance, and meaningful availability or unresolved states. Make incompatible or unresolved joint source selections visible; retain the distinction between reported, locally derived, and unavailable metadata. Reuse ONE and appropriate trial-loading representations. Addresses A05–A06 and the ALF portion of A03.

**Out of scope:**  
Historical behavior interpretation, reconstruction of unavailable revisions, inventory-cache traversal as the runtime API, and a separate revision-management framework.

### [x] SD03 — Expose probe-aware spike-sorting sources

**Summary:**  
Wrap the existing Brainbox source-loading behavior behind the shared boundary, returning spike, cluster, and channel sources with their EID, PID/probe, and relevant collection/revision identity. Respect acquisition policy and requested source selections. Report absent insertions or unavailable spike sources with their source context before downstream merging, preserving valid empty data and the consumer's decision to skip or fail. Addresses the spike-source portions of A01, A03, and A05.

**Out of scope:**  
Spike sorting, neuron filtering, cross-probe merging, binning, and model-oriented neural representations.

### [x] SD04 — Expose correctly identified electrophysiology streams

**Summary:**  
Adapt existing specialized ephys access so each requested insertion resolves its own stream and associated source time conversion, retaining source identity and the shared acquisition policy. Eliminate repeated first-insertion selection and report sessions without the requested source explicitly. Local-only access must not silently enable remote streaming. Addresses A02 and the ephys portions of A01, A03–A05.

**Out of scope:**  
LFP filtering, bad-channel processing, trial-window extraction, feature generation, and replacement of working stream loaders.

### [x] SD05 — Route active consumers through the session-data boundary

**Summary:**  
Redirect source acquisition in `src/visual_stim_gen.py`, `src/prepare_data.py`, and the source-loading portions of `src/utils/ibl_data_utils.py` and `src/utils/preprocess_lfp.py` through the implemented boundary. Centralize client/cache configuration and expose consistent access-policy selection at the relevant entry points. Preserve source/probe associations, original trial identity and timing, replay's fingerprint safeguard, and per-session skip/failure reporting. Demonstrate that replay and neural preparation can consume the returned sources, including the optional LFP path where available. Completes the consumer integration required by A01–A05.

**Out of scope:**  
Rewriting rendering or neural-processing algorithms, changing experiment splits or dataset construction, migrating specialized inspection tools, and unrelated optimization or cleanup.

## Dependencies

- SD02 depends on SD01.
- SD03 depends on SD02.
- SD04 depends on SD02.
- SD05 depends on SD02, SD03, SD04.
