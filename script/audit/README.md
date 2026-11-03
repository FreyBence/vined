# Audit scripts

Run these commands with the project Python. Python entry points resolve checkout
imports and default output paths independently of the current working directory.
Audit reports and diagnostics default to `output/audit/` (or
`<VINED_OUTPUT_DIR>/audit/` when configured).

| Command | Default result location |
| --- | --- |
| `audit_replay.py` | `output/audit/audit-replay.txt` |
| `find_visual_eids.py` | `output/audit/eid-relevanc.txt` |
| `temporal_diagnostics.py --checkpoint CHECKPOINT` | `output/audit/<scenario>/<attempt>/temporal-NNN/` |
| `region_diagnostics.py --checkpoint CHECKPOINT --source-checkpoint SOURCE` | `output/audit/<scenario>/<attempt>/regional-NNN/` |
| `audit_representation_response.py` | `output/audit/scenario-11/002/representation-response-001/` |
| `audit_response_profiles.py` | Supplements `representation-response-001/` |
| `audit_lag_comparison.py` | Supplements `representation-response-001/` |
| `audit_response_reliability.py` | Stages the next `output/audit/scenario-11/002/response-reliability-NNN/` generation |

For example:

```text
.venv/Scripts/python.exe script/audit/audit_replay.py
.venv/Scripts/python.exe script/audit/temporal_diagnostics.py --checkpoint output/runs/scenario-11/002/model_best.pt --device cuda
```

Replay auditing and both model diagnostic commands accept `--output-dir`.
Temporal and regional commands choose the next unused numbered generation when
that option is omitted; explicit existing diagnostic destinations are rejected.
For checkpoints outside a scenario directory, defaults use
`output/audit/<run-directory>/<kind>-NNN/`.
The Scenario 11 analyses retain their fixed experiment inputs and refusal to
replace published artifacts. Reliability publication remains a manually directed
`write_report(stage)` followed by `publish(stage)` after figure inspection.
These functions are in `audit_response_reliability.py`.

`auto_fix_replay.py` and `auto-fix-replay.sh` read
`output/audit/audit-replay.txt` by default. Their generated replay data still go
to the configured replay root. Use `--dry-run` to inspect the retry selection
without generating replay data. The shell launcher uses the existing
`script/environment.sh` environment helper.

## Relocated historical results

The complete former `output/diagnostics/` tree now lives directly under
`output/audit/`, preserving its scenario/attempt structure, setup evidence, and
incomplete staging directories. The former `output/audit-replay.txt` is now
`output/audit/audit-replay.txt`.

Historical payloads and manifests were moved without changing their bytes or
hashes. Embedded historical source paths, script hashes and reproduction commands
describe the original run; they are not rewritten inside immutable artifacts.
Resolve historical `output/diagnostics/...` paths as `output/audit/...`, former
`script/audit_*.py` paths under `script/audit/`, and former
`src/{temporal,region}_diagnostics.py` paths under `script/audit/`.
Current documentation links and script defaults use the new locations.

The reusable `trainer.temporal_diagnostics` implementation and its HTML template
remain in `src/trainer/`; the standalone command lives in this folder.
