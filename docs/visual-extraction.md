# Streaming visual feature extraction

The F06 implementation reuses one immutable CLIP model and processor across the selected sessions. Image batches contain at most `--batch_size` sampled frames. Features, float64 session timestamps, and validity masks are spooled to temporary numeric files, then compressed into the existing schema-v2 `<eid>_visual_clip.npz` format. Original trial IDs and offsets are preserved; downstream loaders require no format changes. Temporary files are removed on ordinary failure or interruption, and the final archive is replaced only after validation.

Memory use for images and features is bounded by the batch size; sidecar time/trajectory arrays still occupy memory for one trial. Temporary disk space must accommodate uncompressed numeric features (approximately 3 KB per sampled frame) plus the final compressed archive. Archive integrity verification streams compressed members rather than loading all session features back into memory.

## Video input (default)

Existing timestamped replays remain supported:

```powershell
.venv/Scripts/python.exe src/prepare_visual_stim.py --eid EID --video_dir ./ibl_task_replay --output_dir ./datasets/vis_stim --frame-source video --sample_fps 5 --batch_size 32
```

All video frames are decoded to detect incomplete outputs, but only selected frames enter image batches. The sampling rule is unchanged: select the first native frame at or after each requested sample time, anchored at onset. For example, 30 FPS sampled at 7 FPS selects frame indices 0, 5, 9, 13, 18, 22, 26 within the first second. Batch boundaries do not reset the sampling clock.

## Direct rendering without MP4

Newly generated sidecars contain wheel displacement for every native frame. Together with the manifest's effective parameters, phase, side, and contrast, these reproduce the renderer's uncompressed frame inputs. Extraction checks the renderer source hash and renders only selected frames. Older sidecars must be regenerated to use this mode.

Use a fresh replay directory and a separate feature output directory:

```powershell
$env:VINED_REPLAY_DIR = './ibl_task_replay_direct'
.venv/Scripts/python.exe src/visual_stim_gen.py --eid EID --no-video
.venv/Scripts/python.exe src/prepare_visual_stim.py --eid EID --video_dir ./ibl_task_replay_direct --output_dir ./datasets/vis_stim_direct --frame-source render --sample_fps 5 --batch_size 32
```

Omit `--no-video` when generating replays to also save inspection videos; the same new sidecars support both extraction modes. Session selection and calibrated parameter options remain available on the generator. Removing MP4 does not resolve uncertainty in the stimulus calibration.

Direct rendering uses the same RGB conversion, CLIP processor, normalization, IDs, timestamps, and masks. Its pixels bypass lossy MP4 compression, so feature vectors can differ from video-derived vectors. Keep the two feature generations separate and rebuild dependent data/caches when changing modes. Provenance records `frame_source`, renderer hash, pinned CLIP revision, and extraction elapsed seconds; console timing also includes archive publication.

## Verification and measurement

Syntax compilation and both CLI help commands passed locally. The existing `script/check_environment.py --device cpu --clip` also passed, including cached CLIP extraction, MP4/dataset round trips, project imports, and CLI checks. These checks do not exercise the complete new streaming/direct-render pipeline. No tests or CI workflows are added for this change.

Real-session performance and feature differences have not been measured. For a manual comparison, generate one session with inspection videos, extract it through both modes into separate directories with the same immutable `--clip-revision`, sampling rate, batch size, and device, and compare recorded extraction times, peak process memory, IDs/timestamps/masks, and feature cosine differences. Include generation time when assessing the benefit of `--no-video`; exclude first-time model downloads from steady-state throughput comparisons.
