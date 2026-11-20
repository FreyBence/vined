"""Retained-artifact E1 views; never construct stimuli or invoke an encoder."""
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from utils.provenance import file_hash, fingerprint
from visual_features import FeatureArtifactReader
from visual_replay import ReplayArtifactReader

EID = '0cad7ea8-8e6c-4ad1-a5c5-53fbb2df1a63'
DATASET = '01e5d3830786d45e1aa6f074d3e6c08fb0e476165f93b11a94d605edefb7995c'
ALIGNMENT = '7e635484272fdd3e209ab442db97a391b6fc9352f31101fa9e08dc2a3145b229'
FEATURE = '8605a05d5fbf67eb01d4155fb3d552e3cbae666509dfd9148747869718714cb7'
REPLAY = '4f1b31578f8e28ec1b926b1cab7d36e9f0124cd6eb31728274cb7b16c3ca8184'
CONTRASTS = [0., .0625, .125, .25, 1.]
TARGETS = ['position', 'signed_displacement', 'absolute_displacement',
           'direction', 'visible_motion', 'contrast']
ATOL = 1e-10


def verify_integrity(repo, output, workers=4):
    """Exhaust public readers; test payloads are accessed only for integrity.

    Do not return observations or samples to the scientific experiment view.
    """
    from alignment import load_alignment
    from training_dataset import load_dataset

    results = []
    with FeatureArtifactReader(repo / 'output/visual_features' / f'{EID}_visual_clip.npz',
                               eid=EID, replay_generation_id=REPLAY) as reader:
        count = 0
        for _ in reader:
            count += 1
        require(reader.completion['generation_id'] == FEATURE, 'Unexpected feature generation')
        results.append({'reader': 'FeatureArtifactReader', 'generation_id': FEATURE, 'records': count, 'complete': True})
    print('Feature public-reader integrity complete', flush=True)
    generation = load_alignment(repo / 'output/alignment' / EID / ALIGNMENT,
                                expected_generation_id=ALIGNMENT)
    results.append({'reader': 'load_alignment', 'generation_id': generation.generation_id, 'complete': True})
    del generation
    print('Alignment public-reader integrity complete', flush=True)
    generation = load_dataset(repo / 'output/datasets/training-dataset/generations' / DATASET,
                              expected_generation_id=DATASET)
    results.append({'reader': 'load_dataset', 'generation_id': generation.generation_id, 'complete': True})
    del generation
    print('Dataset public-reader integrity complete', flush=True)
    with ReplayArtifactReader(repo / 'output/visual_replays' / EID, workers=workers) as reader:
        count = 0
        for _ in reader:
            count += 1
            if count % 25000 == 0:
                print(f'Replay public-reader integrity: {count} records', flush=True)
        require(reader.completion['generation_id'] == REPLAY, 'Unexpected replay generation')
        results.append({'reader': 'ReplayArtifactReader', 'generation_id': REPLAY, 'records': count, 'complete': True})
    result = {'readers': results, 'test_access': 'integrity only; no test data returned to probe views'}
    (output / 'integrity.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def checked_manifest(path, expected):
    m = read_json(path / 'manifest.json')
    if m['generation_id'] != expected or fingerprint({k: v for k, v in m.items()
                                                    if k != 'generation_id'}) != expected:
        raise ValueError(f'Generation identity mismatch: {path}')
    if not m['complete']:
        raise ValueError('Incomplete generation')
    return m


def require(condition, message):
    if not condition:
        raise ValueError(message)


def image(root, m):
    path = root / 'trials' / str(m['trial_id']) / 'images' / f"{m['schedule_index']}.npz"
    with np.load(path, allow_pickle=False) as arrays:
        rgb = arrays['rgb']
    require(rgb.dtype == np.uint8 and list(rgb.shape) == m['shape'], 'RGB schema mismatch')
    require(hashlib.sha256(rgb.tobytes()).hexdigest() == m['rgb_sha256'], 'RGB hash mismatch')
    return rgb


def pair_reason(a, b):
    if a['status'] != 'valid' or b['status'] != 'valid':
        return 'invalid_observation'
    if (a['trial_id'] != b['trial_id'] or b['schedule_index'] != a['schedule_index'] + 1):
        return 'schedule_gap'
    if (not np.isclose(b['session_time'] - a['session_time'], 1 / 60, atol=1e-8, rtol=0)
            or any(x['timing']['cadence_hz'] != 60 for x in (a, b))):
        return 'irregular_interval'
    x, y = a['stimulus_state'], b['stimulus_state']
    if x['stage'] != y['stage'] or x['visible'] != y['visible']:
        return 'event_transition'
    if x['parameters'] != y['parameters']:
        return 'parameter_transition'
    # Applied azimuth changes are the intended translation; every other applied
    # rendering parameter must remain fixed.
    if ({k: v for k, v in a['render_parameters'].items() if k != 'azimuth_deg'} !=
            {k: v for k, v in b['render_parameters'].items() if k != 'azimuth_deg'}):
        return 'render_transition'
    if x['azimuth_deg'] is None or y['azimuth_deg'] is None:
        return 'unknown_position'
    if abs(y['azimuth_deg'] - x['azimuth_deg']) >= 180:
        return 'angular_wrap'
    return None


def facts(a, b, neutral_hash):
    x, y = a['stimulus_state'], b['stimulus_state']
    contrast = y['parameters']['contrast']
    delta = y['azimuth_deg'] - x['azimuth_deg']
    same = a['rgb_sha256'] == b['rgb_sha256']
    nonblank = all(m['rgb_sha256'] != neutral_hash and not m['known_blank']
                   and m['stimulus_state']['visible']
                   and m['stimulus_state']['parameters']['contrast'] > 0 for m in (a, b))
    if same and abs(delta) > ATOL:
        category = 'identical_nonzero_command'
    elif same and all(m['known_blank'] and m['rgb_sha256'] == neutral_hash for m in (a, b)):
        category = 'true_blank'
    elif same and nonblank and abs(delta) <= ATOL:
        category = 'visible_stationary'
    elif not same and nonblank and abs(delta) > ATOL:
        category = 'visible_motion'
    else:
        category = 'other_visual_change'
    # Conservative spatial domain: the commanded center is within the source
    # ViewWindow at both endpoints. Pixel nonblankness alone is insufficient.
    on_screen = all(abs(m['stimulus_state']['azimuth_deg']) <=
                    m['stimulus_state']['parameters']['horizontal_fov_deg'] / 2 for m in (a, b))
    return {'position': y['azimuth_deg'], 'signed_displacement': delta,
                'absolute_displacement': abs(delta), 'direction': int(delta > 0),
                'visible_motion': int(category == 'visible_motion'),
                'contrast': CONTRASTS.index(contrast), 'contrast_value': contrast,
                'category': category, 'on_screen': on_screen, 'nonblank': nonblank,
                'dt': b['session_time'] - a['session_time'], 'time': b['session_time'],
                'initial_side': int(y['parameters']['initial_azimuth_deg'] > 0),
                'phase': y['parameters']['phase_rad'], 'stage': y['stage'],
                'observation_id': b['observation_id'], 'previous_observation_id': a['observation_id']}


def eligible(row, target):
    if target == 'contrast':
        return True
    if not row['on_screen'] or not row['nonblank']:
        return False
    if target == 'position':
        return True
    if target == 'visible_motion':
        return row['category'] in ('visible_motion', 'visible_stationary')
    # Displacement regressions include genuine unchanged visible stationary
    # controls; commanded movement behind identical pixels is separate.
    if target in ('signed_displacement', 'absolute_displacement'):
        return row['category'] in ('visible_motion', 'visible_stationary')
    return row['category'] == 'visible_motion'


def assemble(repo, output):
    dataset_path = repo / 'output/datasets/training-dataset/generations' / DATASET
    alignment_path = repo / 'output/alignment' / EID / ALIGNMENT
    replay_path = repo / 'output/visual_replays' / EID
    feature_path = repo / 'output/visual_features' / f'{EID}_visual_clip.npz'
    dataset = checked_manifest(dataset_path, DATASET)
    alignment = checked_manifest(alignment_path, ALIGNMENT)
    require(dataset['sessions'] == [EID] and alignment['eid'] == EID, 'Session mismatch')
    memberships = dataset['dataset_metadata']['memberships']
    require([len(memberships[s]) for s in ('train', 'val', 'test')] == [218, 31, 63],
            'Unexpected frozen split sizes')
    selected = {e['trial_id']: e['split'] for e in dataset['samples'] if e['split'] in ('train', 'val')}
    require(len(selected) == 249, 'Duplicate train/validation trials')
    require(not set(selected) & {e['trial_id'] for e in memberships['test']}, 'Split overlap')
    # Constructors validate public generation/definition contracts without
    # iterating over test observations. Completion intentionally stays absent.
    with ReplayArtifactReader(replay_path) as reader:
        replay_definition, replay_manifest = reader.definition, reader.artifact_manifest
    with FeatureArtifactReader(feature_path, eid=EID, replay_generation_id=REPLAY) as reader:
        feature_definition, feature_manifest = reader.definition, reader.artifact_manifest
    require(feature_manifest['generation_id'] == FEATURE, 'Feature generation mismatch')
    require(replay_manifest['completion']['generation_id'] == REPLAY, 'Replay generation mismatch')
    require(feature_definition['replay_definition'] == replay_definition, 'Replay definition mismatch')
    table = replay_definition['inputs']['trial_table_fingerprint']
    trial_def = {e['trial_id']: e for e in replay_definition['inputs']['trials'] if e['trial_id'] in selected}
    entries = {e['trial_id']: (i, e) for i, e in enumerate(dataset['samples']) if e['trial_id'] in selected}
    align_entries = {e['trial_id']: (i, e) for i, e in enumerate(alignment['trials']) if e['trial_id'] in selected}
    replay_entries = {e['trial_id']: e for e in replay_manifest['trials'] if e['trial_id'] in selected}
    source = defaultdict(list)
    # records/features are session ZIP members. Skip unselected record lines
    # using manifest trial counts, without JSON decoding them; seek past their
    # numeric vectors. ZIP decompression may traverse bytes, not observations.
    with zipfile.ZipFile(feature_path) as archive, archive.open('records.jsonl') as records, archive.open('features.npy') as vectors:
        require(np.lib.format.read_magic(vectors) == (2, 0), 'Unexpected feature header')
        shape, order, dtype = np.lib.format.read_array_header_2_0(vectors)
        require(shape == (feature_manifest['feature_count'], 768) and not order
                and dtype == np.dtype('<f4'), 'Invalid feature shape/dtype')
        start = vectors.tell()
        feature_index = 0
        for outcome, feature_trial in zip(feature_manifest['replay_completion']['trials'], feature_manifest['trials']):
            tid = outcome['trial_id']
            if tid not in selected:
                for _ in range(outcome['emitted_count'] + 1):
                    require(bool(records.readline()), 'Truncated feature records')
                feature_index += feature_trial['feature_count']
                continue
            path = replay_path / 'trials' / str(tid)
            for name, key in [('observations.jsonl', 'observations_file_sha256'), ('outcome.json', 'outcome_file_sha256')]:
                require(file_hash(path / name) == replay_entries[tid][key], 'Replay trial hash mismatch')
            require(read_json(path / 'outcome.json') == outcome, 'Replay outcome mismatch')
            vectors.seek(start + feature_index * 768 * 4)
            with (path / 'observations.jsonl').open(encoding='utf-8') as replay_records:
                for index in range(outcome['emitted_count']):
                    record = json.loads(records.readline())
                    m = record['metadata']
                    require(m == json.loads(replay_records.readline()), 'Replay/feature observation mismatch')
                    require(m['trial_id'] == tid and m['eid'] == EID
                            and m['trial_table_fingerprint'] == table
                            and m['schedule_index'] == index
                            and m['observation_id'] == f'trial/{tid}/observation/{index}', 'Source identity mismatch')
                    vector = None
                    if record['status'] == 'encoded':
                        require(record['feature_index'] == feature_index and m['status'] == 'valid', 'Feature index/status mismatch')
                        vector = np.frombuffer(vectors.read(768 * 4), dtype='<f4').copy()
                        require(vector.shape == (768,) and np.isfinite(vector).all()
                                and np.isclose(np.linalg.norm(vector), 1, atol=1e-5), 'Invalid feature')
                        feature_index += 1
                    source[tid].append((m, vector))
                require(not replay_records.readline(), 'Extra replay records')
            require(json.loads(records.readline())['metadata'] == outcome, 'Feature outcome mismatch')
            if len(source) % 25 == 0:
                print(f'Loaded {len(source)}/249 selected source trials', flush=True)
    # Neutral RGB content is established from actual retained blank pixels.
    blank = next(m for tid in sorted(source) if selected[tid] == 'train'
                 for m, _ in source[tid] if m['known_blank'])
    rgb = image(replay_path, blank)
    require(np.all(rgb == 128), 'Known blank is not neutral RGB128')
    neutral_hash = blank['rgb_sha256']
    rows, static, difference = defaultdict(list), defaultdict(list), defaultdict(list)
    exclusions = defaultdict(Counter)
    examples, interpolation_checks = {}, []
    duplicates = defaultdict(lambda: {'train': set(), 'val': set()})
    trial_summary = []
    for tid in sorted(source):
        split = selected[tid]
        observations = source[tid]
        metadata = [m for m, _ in observations]
        for m in metadata:
            if m['rgb_sha256']:
                duplicates[m['rgb_sha256']][split].add(tid)
        trial_summary.append({'trial_id': tid, 'split': split,
            'duration': trial_def[tid]['events']['offset'] - trial_def[tid]['events']['onset'],
            'contrast': trial_def[tid]['parameters']['contrast'],
            'phase': trial_def[tid]['parameters']['phase_rad'],
            'initial_side': trial_def[tid]['side'], 'source_observations': len(metadata)})
        for j in range(1, len(observations)):
            a, ua = observations[j - 1]
            b, ub = observations[j]
            reason = pair_reason(a, b)
            if ua is None or ub is None:
                reason = 'missing_feature'
            if reason:
                exclusions['source'][f'{split}:{reason}'] += 1
                continue
            row = facts(a, b, neutral_hash)
            row.update(trial_id=tid, split=split, anchor=j)
            rows['source'].append(row)
            static['source'].append(ub)
            difference['source'].append(ub - ua)
            if split == 'train' and row['category'] not in examples:
                ra, rb = image(replay_path, a), image(replay_path, b)
                require(bool(np.array_equal(ra, rb)) == (a['rgb_sha256'] == b['rgb_sha256']), 'Pixel equality mismatch')
                examples[row['category']] = dict(row, rgb_mean_absolute_difference=float(np.abs(ra.astype(float)-rb).mean()),
                    rgb_min=int(rb.min()), rgb_max=int(rb.max()), hashes=[a['rgb_sha256'], b['rgb_sha256']])
                # Save lossless-derived previews for manual inspection only.
                from PIL import Image
                Image.fromarray(np.concatenate([ra, rb], axis=1)).resize((800, 300)).save(output / f"control-{row['category']}.png")
        i, e = entries[tid]
        require(e['split'] == split and e['session_id'] == EID, 'Sample identity mismatch')
        require(e['metadata']['source_alignment']['generation_id'] == ALIGNMENT, 'Dataset alignment mismatch')
        require(e['metadata']['alignment']['visual_completion']['generation_id'] == FEATURE, 'Dataset feature mismatch')
        sample_name = f'samples/{i:06d}.npz'
        require(file_hash(dataset_path / sample_name) == dataset['files'][sample_name], 'Sample hash mismatch')
        ai, ae = align_entries[tid]
        require(ae['alignment_metadata'] == e['metadata']['alignment'], 'Alignment metadata mismatch')
        align_name = f'trials/{ai:06d}.npz'
        require(file_hash(alignment_path / align_name) == alignment['files'][align_name], 'Alignment payload hash mismatch')
        with np.load(dataset_path / sample_name, allow_pickle=False) as saved, np.load(alignment_path / align_name, allow_pickle=False) as aligned:
            n = e['sequence_length']
            v = saved['visual'][:n]
            times = saved['physical_timestamps'][:n]
            require(saved['temporal_mask'][:n].all() and not saved['temporal_mask'][n:].any(), 'Invalid temporal mask')
            require(np.array_equal(v, aligned['visual_features']) and np.array_equal(times, aligned['bin_center_times']), 'Saved aligned values mismatch')
        res = e['metadata']['alignment']['resampling']
        lookup = {m['observation_id']: j for j, m in enumerate(metadata)}
        indices = [lookup[oid] for oid in res['source_observation_ids']]
        require([metadata[j]['schedule_index'] for j in indices] == res['source_schedule_indices'], 'Source schedule associations mismatch')
        require([metadata[j]['session_time'] for j in indices] == res['source_times'], 'Source time associations mismatch')
        u = np.stack([observations[j][1] for j in indices]).astype(np.float64)
        left, right = np.asarray(res['left_source_indices']), np.asarray(res['right_source_indices'])
        weights = np.asarray(res['interpolation_weights'])
        require(np.allclose(np.asarray(res['source_times'])[left] * (1 - weights) + np.asarray(res['source_times'])[right] * weights, times, atol=1e-9, rtol=0), 'Interpolation query mismatch')
        reconstructed = (1 - weights[:, None]) * u[left] + weights[:, None] * u[right]
        reconstructed = (reconstructed / np.linalg.norm(reconstructed, axis=1, keepdims=True)).astype(np.float32)
        require(np.array_equal(reconstructed, v), 'Source interpolation does not reproduce saved feature')
        interpolation_checks.append({'trial_id': tid, 'bins': n, 'maximum_absolute_difference': float(np.max(np.abs(reconstructed-v)))})
        centers = []
        for k in range(n):
            a, b = metadata[indices[left[k]]], metadata[indices[right[k]]]
            reason = pair_reason(a, b) if left[k] != right[k] else None
            if reason:
                centers.append(None)
                continue
            # Linear reconstructed state ONLY within qualified source brackets.
            x = dict(b)
            x['stimulus_state'] = dict(b['stimulus_state'])
            x['stimulus_state']['azimuth_deg'] = ((1 - weights[k]) * a['stimulus_state']['azimuth_deg'] + weights[k] * b['stimulus_state']['azimuth_deg'])
            x['session_time'] = float(times[k])
            centers.append(x)
        for k in range(1, n):
            lo, hi = indices[left[k - 1]], indices[right[k]]
            reason = None
            if centers[k - 1] is None or centers[k] is None:
                reason = 'ambiguous_center_bracket'
            elif not np.isclose(times[k]-times[k-1], 1/60, atol=1e-8, rtol=0):
                reason = 'irregular_interval'
            else:
                for j in range(lo + 1, hi + 1):
                    reason = pair_reason(metadata[j - 1], metadata[j])
                    if reason:
                        break
            if reason:
                exclusions['aligned'][f'{split}:{reason}'] += 1
                continue
            a, b = centers[k - 1], centers[k]
            row = facts(a, b, neutral_hash)
            # Source images, not invented center pixels, determine observability.
            span = metadata[lo:hi+1]
            span_hashes = {m['rgb_sha256'] for m in span}
            deltas = np.diff([m['stimulus_state']['azimuth_deg'] for m in span])
            if np.any(deltas > ATOL) and np.any(deltas < -ATOL):
                exclusions['aligned'][f'{split}:direction_reversal_within_brackets'] += 1
                continue
            if len(span_hashes) == 1:
                row['category'] = ('identical_nonzero_command' if row['absolute_displacement'] > ATOL else
                                   'true_blank' if all(m['known_blank'] and m['rgb_sha256'] == neutral_hash for m in span) else
                                   'visible_stationary' if row['nonblank'] else 'other_visual_change')
            elif row['nonblank'] and row['absolute_displacement'] > ATOL and all(
                    m['rgb_sha256'] != neutral_hash and not m['known_blank'] for m in span):
                # Every moving constituent must itself be observable.
                hidden = any(abs(d) > ATOL and span[j]['rgb_sha256'] == span[j+1]['rgb_sha256'] for j, d in enumerate(deltas))
                row['category'] = 'other_visual_change' if hidden else 'visible_motion'
            else:
                row['category'] = 'other_visual_change'
            row['visible_motion'] = int(row['category'] == 'visible_motion')
            row.update(trial_id=tid, split=split, anchor=k, source_bracket=[lo, hi],
                       previous_observation_id=f'aligned/{k-1}', observation_id=f'aligned/{k}')
            rows['aligned'].append(row)
            static['aligned'].append(v[k])
            difference['aligned'].append(v[k] - v[k-1])
    views = {stage: {'rows': rows[stage], 'static': np.stack(static[stage]), 'difference': np.stack(difference[stage])} for stage in rows}
    shared = {h: {s: sorted(v) for s, v in ss.items()} for h, ss in duplicates.items() if ss['train'] and ss['val']}
    provenance = {'dataset_generation': DATASET, 'alignment_generation': ALIGNMENT,
        'feature_generation': FEATURE, 'replay_generation': REPLAY, 'session': EID,
        'trial_table_fingerprint': table, 'memberships': memberships,
        'feature_definition': feature_definition,
        'paths': {k: str(p.relative_to(repo)) for k, p in {'dataset': dataset_path, 'alignment': alignment_path, 'replay': replay_path, 'features': feature_path}.items()},
        'verification': {'public_reader_headers': ['ReplayArtifactReader', 'FeatureArtifactReader'],
            'complete_generation_readers_exhausted': False,
            'boundary': 'Train/validation payload hashes and associations checked directly; no test observation decoded. Session ZIP decompression can traverse skipped compressed bytes. Whole-generation payload verification is not claimed.',
            'selected_trials': len(selected), 'interpolation': interpolation_checks,
            'representative_pixel_checks': examples, 'feature_archive_sha256': file_hash(feature_path)},
        'unavailable': ['pre-normalization projected features and norms', 'original experimental framebuffer captures'],
        'duplicate_images': {'shared_hash_count': len(shared), 'shared_nonblank_hash_count': sum(h != neutral_hash for h in shared),
            'shared': shared, 'neutral_hash': neutral_hash}, 'trial_summary': trial_summary}
    return views, provenance, {k: dict(v) for k, v in exclusions.items()}
