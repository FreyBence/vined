# Condition- and movement-matched response reliability

The Scenario 11 / Attempt 002 research audit is published in
[`response-reliability-001`](../../output/audit/scenario-11/002/response-reliability-001/report.md),
with a [standalone HTML report](../../output/audit/scenario-11/002/response-reliability-001/report.html).
That report contains the complete methods, contributor tables, quantitative
results, limitations, verified findings, figures and provenance. This document
is its repository entry point; it does not replace the component specification
or the earlier representation-response audit.

## Findings

The analysis preserves the exact dataset generation
`01e5d3830786d45e1aa6f074d3e6c08fb0e476165f93b11a94d605edefb7995c`,
218 training trials, 31 validation trials and the original ordered 99 VISpm
neurons. No test payload, new model inference or training was used.

Matching uses recorded side/contrast, a training-median first-second wheel-path
threshold and response timing relative to the first-second observation interval.
Measured wheel positions are available and match replay input hashes. Velocity
and movement summaries are derived; closed-loop/freeze timing uses documented
go-cue/response proxies. Residual trajectory differences, irregular wheel
sampling, synthetic phase and the near-confounding of movement and task phase
prevent an exact visual-repeat interpretation.

Some early-response regional averages are reproducible at coarse temporal
resolution: right/0.125 high-wheel/responded has mean 200 ms whole-trial
split-half correlation 0.912 (12 trials), and right/1 has 0.968 (19 trials).
Their raw-bin values are 0.182 and 0.303. Ongoing-response groups are often
weak or negative. Matching clarifies this heterogeneity rather than improving
every group. All widths use identical temporal endpoints.

There are 16 matched training groups with at least six contributors, but no
matched validation group reaches that minimum. Three groups support descriptive
cross-split profile comparisons with at least three trials per split. These
measurements are condition-average consistency, not a noise ceiling or
individual-trial predictability estimate.

Zero-contrast training trials show movement-associated neural modulation even
though CLIP is constant: around recorded first movement, regional counts rise
by 1.875 counts per original bin (22 complete trials; whole-trial bootstrap
interval [1.170, 2.727]). Validation has only three such complete windows and
an interval spanning zero. Visual and behavioral contributions remain
observationally confounded.

Individual-unit reliability is heterogeneous and usually below the regional
value in the strongest groups. No defensible evidence establishes that regional
summation obscures reproducible stimulus-specific information. Some frozen
early-response averages are also predicted well, despite weak overall temporal
performance. Preserve the current target, representation, alignment and model.

## Recommendation

**Outcome B: heterogeneous reproducibility.** Separately specify a controlled,
training-only memorization experiment, beginning with trial **4**. A small
diverse follow-up group is **[4, 261, 272, 137]**, selected using metadata/features
and support rather than favorable model correlations. Trial 14 is an optional
long-duration stress example. Include elapsed-position and visual-content
controls; do not equate memorization with generalization or physiological
reliability. No overfitting experiment was conducted in this audit.

## Reproduction

The focused scientific analysis is
[`script/audit/audit_response_reliability.py`](../../script/audit/audit_response_reliability.py).
Run it with the project Python to produce a uniquely named staging directory
for the next unused diagnostic generation. After inspecting its figures, use
the module's `write_report(stage)` and `publish(stage)` functions to finalize
the reports and atomically publish hashed payloads. Existing generations are
never overwritten. The full report documents the policy, seeds, support,
resampling, input identities and artifact schemas.
