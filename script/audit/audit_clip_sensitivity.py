"""Compare frozen CLIP preprocessing and representations on verified replay pixels."""

import argparse
from contextlib import closing, nullcontext
import hashlib
import json
from pathlib import Path
import sys
import time
from uuid import UUID

import numpy as np

from audit_paths import AUDIT_ROOT
from analyze_clip_similarity import collect, cosine
from prepare_visual_stim import resolve_replay
from utils.config_utils import load_config
from utils.paths import REPO_ROOT, replay_dir, visual_dir
from visual_features import (ClipEncoder, EncodedObservation, FeatureArtifactReader,
                             ObservationSelection, iter_encoded_observations)
from visual_replay import ReplayArtifactReader, ReplayObservation


def distance(a, b):
    value = float(cosine(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), 1e-12))
    if not np.isfinite(value):
        raise ValueError("Undefined cosine for extracted representation")
    return 1 - value


def select(path, eid, args):
    # Use the previous audit's exact primary selection, including gap handling.
    args.reference_samples = args.pair_samples = 1
    args.norm_floor = 1e-12
    args.seed = 42
    _, _, primary, _, _, definition, completion = collect(path, eid, args)
    if primary is None:
        raise ValueError("No consecutive differing pair above tolerance")
    pairs, previous = [], None
    categories = {True: 0, False: 0}
    with FeatureArtifactReader(path, eid=eid) as reader:
        for item in reader:
            if not isinstance(item, EncodedObservation) or item.feature is None:
                previous = None
                continue
            current = (item.metadata, item.feature.copy())
            if (previous is not None and previous[0]['trial_id'] == current[0]['trial_id']
                    and previous[0]['schedule_index'] + 1 == current[0]['schedule_index']):
                same = previous[0].get('rgb_sha256') == current[0].get('rgb_sha256')
                ids = [p[0]['observation_id'] for p in (previous, current)]
                if ids != [p[0]['observation_id'] for p in primary] and categories[same] < args.secondary_pairs:
                    pairs.append((previous, current))
                    categories[same] += 1
            previous = current
    return [primary, *pairs], definition, completion


def recover(source, pairs, definition, completion, workers):
    wanted = {m['observation_id']: m for pair in pairs for m, _ in pair}
    images = {}
    with ReplayArtifactReader(source, workers=workers) as reader:
        if reader.definition_id != definition['replay_definition_id']:
            raise ValueError("Replay definition differs from feature source")
        if reader.artifact_manifest['completion'] != completion['replay_completion']:
            raise ValueError("Published replay completion differs from consumed feature source")
        for item in reader:
            if isinstance(item, ReplayObservation) and item.metadata['observation_id'] in wanted:
                key = item.metadata['observation_id']
                expected = wanted[key]
                if item.metadata != expected or item.rgb is None:
                    raise ValueError(f"Source metadata/pixels mismatch: {key}")
                if hashlib.sha256(item.rgb.tobytes()).hexdigest() != expected.get('rgb_sha256'):
                    raise ValueError(f"Source RGB hash mismatch: {key}")
                images[key] = item.rgb.copy()
                if images.keys() == wanted.keys():
                    break
    if images.keys() != wanted.keys():
        raise ValueError("Original images unavailable; refusing approximate substitutes")
    return images


def transform(rgb, size, config):
    from PIL import Image

    h, w = rgb.shape[:2]
    fx, fy = config.get('crop', (1, 1))
    cw, ch = max(1, round(w * fx)), max(1, round(h * fy))
    x, y = (w - cw) // 2, (h - ch) // 2
    pixels = rgb[y:y + ch, x:x + cw].astype(np.float64)
    # Fixed neutral pivot; never fit a per-image contrast transform.
    changed = 128 + config.get('contrast', 1) * (pixels - 128)
    clipping = float(np.mean((changed < 0) | (changed > 255)))
    image = Image.fromarray(np.rint(np.clip(changed, 0, 255)).astype(np.uint8))
    method = getattr(Image.Resampling, config.get('interpolation', 'BICUBIC'))
    direct = config.get('direct', False)
    scale = size / max(cw, ch)
    dims = (size, size) if direct else (max(1, round(cw * scale)), max(1, round(ch * scale)))
    image = image.resize(dims, method)
    offset = ((size - dims[0]) // 2, (size - dims[1]) // 2)
    if config.get('padding') == 'edge':
        data = np.asarray(image)
        canvas = Image.fromarray(np.pad(data, ((offset[1], size-dims[1]-offset[1]),
                                              (offset[0], size-dims[0]-offset[0]), (0, 0)), mode='edge'))
    else:
        canvas = Image.new('RGB', (size, size), (128, 128, 128))
        canvas.paste(image, offset)
    outside = np.ones((h, w), dtype=bool)
    outside[y:y + ch, x:x + cw] = False
    # Conservative content check against the renderer's neutral background.
    lost = bool(np.any(np.abs(rgb[outside].astype(np.int16) - 128) > 2))
    distorted = direct and cw != ch
    return canvas, dict(crop_xyxy=[x, y, x+cw, y+ch], resized_wh=list(dims),
                        scale_xy=[dims[0]/cw, dims[1]/ch], padding_offset_xy=list(offset),
                        clipping_fraction=clipping, removed_non_neutral_content=lost,
                        aspect_ratio_distorted=distorted,
                        geometry_pass=not lost and not distorted and clipping == 0)


def representations(encoder, images, precision, layers):
    import torch

    # Deliberately inspect the existing encoder instance here: audit-only access,
    # without adding a production API for experimental hidden representations.
    inputs = encoder._processor(images=images, return_tensors='pt')
    context = torch.autocast('cuda', dtype=torch.float16) if precision == 'float16' else nullcontext()
    with torch.inference_mode(), context:
        output = encoder._model.vision_model(
            pixel_values=inputs['pixel_values'].to(encoder._device), output_hidden_states=bool(layers))
        vectors = {'projected_cls': encoder._model.visual_projection(output.pooler_output),
                   'pooled_cls': output.pooler_output}
        for layer in layers:
            hidden = output.hidden_states[layer]
            vectors[f'layer_{layer}_cls'] = hidden[:, 0]
            vectors[f'layer_{layer}_patch_mean'] = hidden[:, 1:].mean(dim=1)
            vectors[f'layer_{layer}_patch_concat'] = hidden[:, 1:].flatten(1)
    result = {}
    for name, vector in vectors.items():
        vector = vector.float()
        if not torch.isfinite(vector).all() or (torch.linalg.vector_norm(vector, dim=-1) <= 1e-12).any():
            raise ValueError(f'Invalid representation: {name}')
        result[name] = torch.nn.functional.normalize(vector, dim=-1).cpu().numpy()
    return result


def run(args, report):
    import torch
    import transformers
    from PIL import Image

    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    eid = str(UUID(load_config(args.run_config)['eid']))
    path = args.visual_dir / f'{eid}_visual_clip.npz'
    print(f'Verifying features: {path}', flush=True)
    pairs, definition, completion = select(path, eid, args)
    report.update(eid=eid, feature_source=str(path), feature_generation_id=completion['generation_id'],
                  selected_pairs=[[m for m, _ in pair] for pair in pairs],
                  stored_representation=definition['representation'])
    report['source_verification'] = ('Feature archive fully verified; replay published identities checked; '
                                     'consumed image metadata and RGB hashes verified through public reader. '
                                     'Replay reading stops after required images; unrelated replay payloads not reverified.')
    source = resolve_replay(args.replay_dir, eid)
    print(f'Verifying original replay images: {source}', flush=True)
    images = recover(source, pairs, definition, completion, args.workers)
    rep = definition['representation']
    inference = rep.get('inference', {})
    precision = inference.get('precision', 'float32')
    encoder = ClipEncoder(rep['model'], revision=rep['revision'], device=args.device or rep['device'],
                          precision=precision, attention_backend=inference.get('attention_backend', 'eager'),
                          feature_cache_size=inference.get('image_reuse', {}).get('capacity', 0))
    actual = encoder.provenance
    for key in ('preparation', 'processor'):
        if actual[key] != rep[key]:
            raise ValueError(f'Baseline {key} mismatch with stored representation')
    for name, digest in rep['artifact_hashes'].items():
        if actual['artifact_hashes'].get(name) != digest:
            raise ValueError(f'Checkpoint artifact mismatch: {name}')
    primary = [images[m['observation_id']] for m, _ in pairs[0]]
    # Reproduce original batching and cache history through the public extraction
    # path, stopping after the selected pair instead of publishing new features.
    wanted = {m['observation_id'] for m, _ in pairs[0]}
    regenerated_by_id = {}
    replay = ReplayArtifactReader(source, workers=args.workers)
    selection = ObservationSelection(replay, sample_fps=definition['selection']['sample_fps'])
    with closing(iter_encoded_observations(selection, encoder, batch_size=definition['batch_size'])) as stream:
        for item in stream:
            if isinstance(item, EncodedObservation) and item.metadata['observation_id'] in wanted:
                regenerated_by_id[item.metadata['observation_id']] = item.feature
                if regenerated_by_id.keys() == wanted:
                    break
    regenerated = np.stack([regenerated_by_id[m['observation_id']] for m, _ in pairs[0]])
    encoder._feature_cache.clear()
    encoder._feature_cache_size = 0
    pair_batch = encoder.encode(primary)
    stored = np.stack([v for _, v in pairs[0]])
    baseline_distance = distance(*pair_batch)
    reproduced = bool(np.allclose(regenerated, stored, atol=args.reproduction_atol, rtol=0))
    report.update(versions=dict(torch=torch.__version__, transformers=transformers.__version__),
                  loaded_representation=actual,
                  historical_baseline=dict(cosine_similarity=1-distance(*regenerated), cosine_distance=distance(*regenerated),
                                stored_cosine_similarity=1-distance(*stored),
                                max_absolute_difference=float(np.max(np.abs(regenerated-stored))),
                                difference_l2_per_frame=np.linalg.norm(regenerated-stored, axis=1).tolist(),
                                tolerance_atol=args.reproduction_atol, tolerance_rtol=0,
                                reproduced=reproduced,
                                original_batch_size=definition['batch_size'],
                                pair_batch_cosine_similarity=1-distance(*pair_batch),
                                pair_batch_max_abs_difference=float(np.max(np.abs(pair_batch-regenerated))),
                                note='Original batch/cache history reproduced via public extraction. Small audit batches are reported separately.'))
    historical = report['historical_baseline']
    historical['status'] = 'reproduced' if reproduced else 'baseline_mismatch'
    historical['pair_batch_reproduces_stored'] = bool(np.allclose(pair_batch, stored, atol=args.reproduction_atol, rtol=0))
    historical['mismatch_assessment'] = (
        'Original extraction context reproduces stored vectors; fixed-pixel batch/precision probes quantify the small-batch discrepancy.'
        if reproduced else 'Historical execution context remains unreproduced; numerical cause is not fully established. Provenance verified independently.')
    report['warnings'] = [] if reproduced else [
        'Historical reproduction exceeds the unchanged tolerance. Provenance verified; experiments continue using the current execution baseline.']
    report['provenance_verified'] = True
    report['baseline'] = dict(cosine_similarity=1-baseline_distance, cosine_distance=baseline_distance,
                            precision=precision, batch_size=2, image_reuse=False,
                            role='Current audit projected CLS baseline; historical vectors are never comparison denominators.')
    prepared_primary = [transform(rgb, encoder._size, {})[0] for rgb in primary]
    stability = []
    for measured_precision in dict.fromkeys([precision, 'float32']):
        for batch_size in sorted(set([1, 2, 8, definition['batch_size']])):
            if batch_size == 1:
                vectors = np.concatenate([representations(encoder, [im], measured_precision, [])['projected_cls']
                                          for im in prepared_primary])
            else:
                # Repeat only the verified pair to measure batch shape; these are
                # numerical probes, never additional stimulus observations.
                batch = [prepared_primary[i % 2] for i in range(batch_size)]
                vectors = representations(encoder, batch, measured_precision, [])['projected_cls'][:2]
            stability.append(dict(precision=measured_precision, batch_size=batch_size,
                                  batch_content='alternating original pair; singleton separately',
                                  cosine_similarity=1-distance(*vectors), cosine_distance=distance(*vectors),
                                  max_abs_vs_stored=float(np.max(np.abs(vectors-stored))),
                                  max_abs_vs_current_pair_batch=float(np.max(np.abs(vectors-pair_batch))),
                                  reproduces_stored=bool(np.allclose(vectors, stored, atol=args.reproduction_atol, rtol=0))))
    report['numerical_stability'] = dict(batch_precision_probes=stability,
        projected_distance_range=float(np.ptp([s['cosine_distance'] for s in stability])),
        interpretation=('Batch/precision probes isolate execution variation with fixed pixels and weights. '
                        'Candidate noise envelopes below are empirical bounds on sampled contexts, not confidence intervals.'))
    print('Historical reproduction: ' + historical['status'], flush=True)
    size = encoder._size
    depth = encoder._model.config.vision_config.num_hidden_layers
    layers = sorted(set([max(1, depth//4), depth//2, 3*depth//4, depth]))
    report['hidden_state_indices'] = layers
    report['hidden_state_index_semantics'] = 'Index 0 is embedding output; index N follows transformer block N.'
    configs = [dict(name='baseline'),
               *[dict(name=f'fit_{method.lower()}', interpolation=method) for method in ('BILINEAR', 'LANCZOS')],
               dict(name='edge_padding', padding='edge'), dict(name='direct_resize', direct=True),
               *[dict(name=f'contrast_{factor}', contrast=factor) for factor in (0.75, 1.25, 1.5)],
               dict(name='central_75', crop=(0.75, 0.75)), dict(name='central_50', crop=(0.5, 0.5)),
               dict(name='horizontal_band', crop=(1, 0.5)), dict(name='vertical_band', crop=(0.5, 1))]
    report['unsupported'] = ['Non-native input sizes excluded: fixed positional grid; no positional interpolation experiment.',
                             'FP16 on CPU is unsupported by the repository encoder.']
    report['notes'] = ['Frozen eval/inference mode; channel normalization unchanged; no temporal selection changes.',
                       'Ranking is descriptive for one primary pair; feature spaces have different similarity distributions.',
                       'Crop rejection conservatively detects any removed non-neutral content; no pair-fitted crop or token selection.',
                       'Secondary samples are first eligible same/different-hash pairs, bounded per category; not population estimates.',
                       'Noise envelope: maximum observed context drift, FP32 discrepancy, identical-image distance, and 1e-12 cosine arithmetic floor. Baseline also includes all batch/precision probes.',
                       'Meaningful candidate: gain exceeds candidate plus baseline noise envelopes at the same precision on primary and every differing secondary pair, with geometry/repeat/identical checks passing.',
                       'Empirical noise envelopes cover only sampled execution contexts; different feature spaces still have different natural similarity scales.',
                       'Memory reports vector payload and peak CUDA allocation, excluding Python/CPU model overhead.']
    keys = list(images)
    positions = {key: i for i, key in enumerate(keys)}
    pair_indices = [[positions[m['observation_id']] for m, _ in pair] for pair in pairs]
    results = []
    report['results'] = results
    for config in configs:
        print(f"Evaluating {config['name']}", flush=True)
        transformed = [transform(images[key], size, config) for key in keys]
        prepared = [item[0] for item in transformed]
        geometry = [item[1] for item in transformed]
        previews = Image.new('RGB', (2*size, size))
        for j, index in enumerate(pair_indices[0]):
            previews.paste(prepared[index], (j*size, 0))
        previews.save(args.output_dir / f"{config['name']}.png")
        for current_precision in dict.fromkeys([precision, 'float32']):
            started = time.perf_counter()
            cuda = torch.device(encoder._device).type == 'cuda'
            if cuda:
                torch.cuda.reset_peak_memory_stats(encoder._device)
            # Fixed batches of two bound hidden-state memory independently of archive length.
            chunks = [representations(encoder, prepared[i:i+2], current_precision, layers)
                      for i in range(0, len(prepared), 2)]
            values = {name: np.concatenate([chunk[name] for chunk in chunks]) for name in chunks[0]}
            duplicate = representations(encoder, [prepared[pair_indices[0][0]]]*2, current_precision, layers)
            # Singleton and reversed pair contexts retain every physical image,
            # measuring execution variation for all representations and pairs.
            single_chunks = [representations(encoder, [im], current_precision, layers) for im in prepared]
            single_values = {name: np.concatenate([c[name] for c in single_chunks]) for name in values}
            reverse_chunks = [representations(encoder, prepared[i:i+2][::-1], current_precision, layers)
                              for i in range(0, len(prepared), 2)]
            reverse_values = {name: np.concatenate([c[name][::-1] for c in reverse_chunks]) for name in values}
            elapsed = time.perf_counter()-started
            if config['name'] == 'baseline' and current_precision == precision:
                audit_vectors = values['projected_cls'][pair_indices[0]]
                report['baseline']['public_encoder_max_abs_difference'] = float(np.max(np.abs(audit_vectors-pair_batch)))
                baseline_distance = distance(*audit_vectors)
                report['baseline'].update(cosine_similarity=1-baseline_distance, cosine_distance=baseline_distance)
            for name, vectors in values.items():
                distances = [distance(vectors[a], vectors[b]) for a, b in pair_indices]
                context_distances = [[distance(context[name][a], context[name][b]) for a, b in pair_indices]
                                     for context in (single_values, reverse_values)]
                noise = [max(abs(context[i]-d) for context in context_distances)
                         for i, d in enumerate(distances)]
                repeat_error = float(np.max(np.abs(duplicate[name][0]-duplicate[name][1])))
                cross_batch_error = float(np.max(np.abs(duplicate[name][0]-vectors[pair_indices[0][0]])))
                repeat_pass = max(repeat_error, cross_batch_error) <= args.reproduction_atol
                geometry_pass = all(g['geometry_pass'] for g in geometry)
                secondary = []
                for i, pair in enumerate(pairs[1:], 1):
                    identical = pair[0][0]['rgb_sha256'] == pair[1][0]['rgb_sha256']
                    secondary.append(dict(pair_index=i, identical_images=identical,
                                          cosine_distance=distances[i],
                                          context_noise=noise[i],
                                          identical_pass=distances[i] <= 1e-10 if identical else None))
                secondary_pass = False  # Resolve against same-precision baseline after all rows exist.
                dim = vectors.shape[1]
                results.append(dict(configuration=config, representation=name, precision=current_precision,
                                    cosine_similarity=1-distances[0], cosine_distance=distances[0],
                                    distance_change=distances[0]-baseline_distance,
                                    distance_ratio=distances[0]/baseline_distance if baseline_distance else None,
                                    context_noise=noise[0], context_pair_distances=context_distances,
                                    output_dimension=dim, remains_768=dim == 768,
                                    downstream_dimension_change=dim != 768, independent_per_frame=True,
                                    patch_token_count=(size//encoder._model.config.vision_config.patch_size)**2
                                    if 'patch' in name else 0,
                                    vector_bytes_float32=dim*4, preprocessing=geometry,
                                    repeat_max_abs=repeat_error, repeat_cross_batch_max_abs=cross_batch_error,
                                    repeat_pass=repeat_pass, geometry_pass=geometry_pass,
                                    secondary=secondary, secondary_pass=secondary_pass,
                                    inference_group_seconds=elapsed,
                                    peak_cuda_bytes=torch.cuda.max_memory_allocated(encoder._device) if cuda else None))
    results.sort(key=lambda row: row['cosine_distance'], reverse=True)
    fp32 = {(r['configuration']['name'], r['representation']): r for r in results if r['precision'] == 'float32'}
    baselines = {r['precision']: r for r in results if r['configuration']['name'] == 'baseline' and r['representation'] == 'projected_cls'}
    for rank, row in enumerate(results, 1):
        row['rank'] = rank
        reference = fp32[(row['configuration']['name'], row['representation'])]
        row['cosine_distance_change_from_fp32'] = row['cosine_distance']-reference['cosine_distance']
        row['precision_effect_fraction_of_fp32_distance'] = (
            abs(row['cosine_distance_change_from_fp32']) / reference['cosine_distance']
            if reference['cosine_distance'] else None)
        row['noise_envelope'] = max(1e-12, row['context_noise'], reference['context_noise'],
                                    abs(row['cosine_distance_change_from_fp32']),
                                    max((s['cosine_distance'] for s in row['secondary'] if s['identical_images']), default=0))
        if row['configuration']['name'] == 'baseline' and row['representation'] == 'projected_cls':
            row['noise_envelope'] = max(row['noise_envelope'],
                max(abs(p['cosine_distance']-row['cosine_distance']) for p in stability))
    for row in results:
        base = baselines[row['precision']]
        row['same_precision_baseline_distance'] = base['cosine_distance']
        row['same_precision_distance_change'] = row['cosine_distance']-base['cosine_distance']
        row['improvement_noise_bound'] = row['noise_envelope']+base['noise_envelope']
        row['improvement_exceeds_noise'] = row['same_precision_distance_change'] > row['improvement_noise_bound']
        row['noise_limited_improvement'] = row['same_precision_distance_change'] > 0 and not row['improvement_exceeds_noise']
        for sample, baseline_sample, fp32_sample in zip(row['secondary'], base['secondary'],
                                                       fp32[(row['configuration']['name'], row['representation'])]['secondary']):
            sample['baseline_distance'] = baseline_sample['cosine_distance']
            sample['improved'] = sample['cosine_distance'] > sample['baseline_distance']
            sample['noise_envelope'] = max(sample['context_noise'], fp32_sample['context_noise'],
                                          abs(sample['cosine_distance']-fp32_sample['cosine_distance']))
            # Include precision and context variation of the matching baseline.
            fp32_base = baselines['float32']['secondary'][sample['pair_index']-1]
            bound = sample['noise_envelope']+max(baseline_sample['context_noise'], fp32_base['context_noise'],
                                                abs(baseline_sample['cosine_distance']-fp32_base['cosine_distance']))
            sample['improvement_noise_bound'] = bound
            sample['improvement_exceeds_noise'] = sample['cosine_distance']-sample['baseline_distance'] > bound
        differing = [s for s in row['secondary'] if not s['identical_images']]
        row['secondary_pass'] = (row['repeat_pass'] and row['geometry_pass'] and bool(differing)
                                 and all(s['identical_pass'] for s in row['secondary'] if s['identical_images'])
                                 and all(s['improvement_exceeds_noise'] for s in differing))
        row['stable_meaningful_candidate'] = row['secondary_pass'] and row['improvement_exceeds_noise']
    def best(predicate):
        row = next((r for r in results if predicate(r)), None)
        return {k: row[k] for k in ('rank', 'configuration', 'representation', 'precision', 'cosine_distance')} if row else None
    report['recommendations'] = dict(lowest_similarity=best(lambda r: True),
                                     strongest_secondary_validated=best(lambda r: r['stable_meaningful_candidate']),
                                     strongest_valid_768=best(lambda r: r['stable_meaningful_candidate'] and r['remains_768']),
                                     strongest_valid_spatial_or_intermediate=best(lambda r: r['stable_meaningful_candidate'] and r['representation'].startswith('layer_')))
    report['status'] = 'ok'
    for i, rgb in enumerate(primary):
        Image.fromarray(rgb).save(args.output_dir / f'source-{i}.png')
    print('| Rank | Configuration | Representation | Precision | Dim | Cosine | Distance | Baseline ratio | Sanity / secondary |')
    print('|---:|---|---|---|---:|---:|---:|---:|---|')
    for r in results:
        ratio = f"{r['distance_ratio']:.4g}" if r['distance_ratio'] is not None else 'undefined'
        print(f"| {r['rank']} | {r['configuration']['name']} | {r['representation']} | {r['precision']} | {r['output_dimension']} | {r['cosine_similarity']:.10f} | {r['cosine_distance']:.8g} | {ratio} | {r['repeat_pass'] and r['geometry_pass']} / {r['secondary_pass']} |")
    print('Historical reproduction:\n' + json.dumps(report['historical_baseline'], indent=2))
    print('Current audit baseline:\n' + json.dumps(report['baseline'], indent=2))
    print('Recommendations:\n' + json.dumps(report['recommendations'], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-config', type=Path, default=REPO_ROOT/'src/configs/run.json')
    parser.add_argument('--visual-dir', type=Path, default=visual_dir())
    parser.add_argument('--replay-dir', type=Path, default=replay_dir())
    parser.add_argument('--output-dir', type=Path, default=AUDIT_ROOT/'clip-sensitivity')
    parser.add_argument('--device', default=None)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--secondary-pairs', type=int, default=2, help='Maximum additional pairs per identical/differing source-hash category')
    parser.add_argument('--tolerance', type=float, default=1e-7)
    parser.add_argument('--reproduction-atol', type=float, default=1e-4, help='Absolute per-coordinate reproduction/repeat tolerance; no relative tolerance')
    args = parser.parse_args()
    if args.workers < 1 or args.secondary_pairs < 1:
        parser.error('workers and secondary-pairs must be positive')
    if any(not np.isfinite(v) or v <= 0 for v in (args.tolerance, args.reproduction_atol)):
        parser.error('tolerances must be finite and positive')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(status='started', arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    started = time.perf_counter()
    try:
        run(args, report)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        report.update(status='blocked', blocker=str(exc))
        print(f'CLIP sensitivity audit blocked: {exc}', file=sys.stderr)
    finally:
        report['runtime_seconds'] = time.perf_counter()-started
        with (args.output_dir/'report.json').open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
        print(f"Report: {args.output_dir/'report.json'}", flush=True)
    return 0 if report['status'] == 'ok' else 1


if __name__ == '__main__':
    sys.exit(main())
