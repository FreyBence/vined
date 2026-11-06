"""Audit four frozen FP32 CLIP representations against multi-trial azimuth motion."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from uuid import UUID

import numpy as np

from audit_paths import AUDIT_ROOT
from audit_clip_sensitivity import distance, recover, representations, transform
from prepare_visual_stim import resolve_replay
from utils.config_utils import load_config
from utils.paths import REPO_ROOT, replay_dir, visual_dir
from visual_features import ClipEncoder, EncodedObservation, FeatureArtifactReader


CONFIGURATIONS = [
    dict(identifier='clip_input_contrast_gain_1_0_layer_24_cls', clip_input_contrast_gain=1.0,
         representation='layer_24_cls', dimension=1024),
    dict(identifier='baseline', clip_input_contrast_gain=1.0,
         representation='projected_cls', dimension=768),
    dict(identifier='clip_input_contrast_gain_1_5_projected_cls', clip_input_contrast_gain=1.5,
         representation='projected_cls', dimension=768),
    dict(identifier='clip_input_contrast_gain_1_5_layer_24_cls', clip_input_contrast_gain=1.5,
         representation='layer_24_cls', dimension=1024),
]


def spread(items, count):
    return [items[i] for i in np.unique(np.linspace(0, len(items)-1, min(count, len(items))).round().astype(int))] if items else []


def select_pairs(path, eid, args):
    trials, previous, skipped = {}, None, 0
    with FeatureArtifactReader(path, eid=eid) as reader:
        definition = reader.definition
        for item in reader:
            if not isinstance(item, EncodedObservation) or item.feature is None:
                previous = None
                continue
            current = item.metadata
            if previous is not None and current['trial_id'] == previous['trial_id'] and current['schedule_index'] == previous['schedule_index']+1:
                pair = [previous, current]
                try:
                    az = [float(m['stimulus_state']['azimuth_deg']) for m in pair]
                    dt = float(current['session_time']-previous['session_time'])
                    if not np.isfinite(az).all() or dt <= 0:
                        raise ValueError('Invalid displacement/time')
                    if any(m['timing']['cadence_hz'] != 60 for m in pair):
                        raise ValueError('Expected unchanged 60 Hz source')
                    trials.setdefault(current['trial_id'], []).append(pair)
                except (KeyError, TypeError, ValueError):
                    skipped += 1
            previous = current
        completion = reader.completion
    selected = []
    for trial in spread(list(trials), args.trial_limit):
        groups = [[], [], []]
        for pair in trials[trial]:
            delta = pair[1]['stimulus_state']['azimuth_deg']-pair[0]['stimulus_state']['azimuth_deg']
            groups[0 if abs(delta) <= args.stationary_atol_deg else (1 if delta < 0 else 2)].append(pair)
        # Equal quotas across stationary/left/right; displacement quantiles within
        # movement strata, chronological coverage within stationary strata.
        chosen = []
        for group in groups:
            group.sort(key=lambda p: (abs(p[1]['stimulus_state']['azimuth_deg']-p[0]['stimulus_state']['azimuth_deg']), p[0]['schedule_index']))
            chosen.extend(spread(group, max(1, args.pairs_per_trial//3)))
        ids = {p[0]['observation_id'] for p in chosen}
        chosen.extend(spread([p for p in trials[trial] if p[0]['observation_id'] not in ids], args.pairs_per_trial-len(chosen)))
        selected.extend(sorted(chosen, key=lambda p: p[0]['schedule_index']))
    if not selected:
        raise ValueError('No eligible adjacent 60 Hz pairs with physical azimuth metadata')
    return selected, definition, completion, skipped


def distribution(values):
    x = np.asarray(values, dtype=np.float64)
    return dict(count=len(x), minimum=float(x.min()), median=float(np.median(x)),
                mean=float(x.mean()), std=float(x.std()), maximum=float(x.max())) if len(x) else dict(count=0)


def correlation(x, y):
    from scipy.stats import pearsonr, spearmanr

    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 3 or np.ptp(x) <= 1e-12 or np.ptp(y) <= 1e-12:
        return dict(count=len(x), pearson=None, spearman=None, reason='Insufficient sample count or variation')
    return dict(count=len(x), pearson=float(pearsonr(x, y).statistic),
                spearman=float(spearmanr(x, y).statistic))


def infer(encoder, prepared, order, batch_size):
    collected = {}
    for start in range(0, len(order), batch_size):
        ids = order[start:start+batch_size]
        outputs = representations(encoder, [prepared[i] for i in ids], 'float32', [24])
        for key in ('projected_cls', 'layer_24_cls'):
            collected.setdefault(key, {}).update(zip(ids, outputs[key]))
    return collected


def pair_facts(pairs, images, stationary_atol):
    facts = []
    for index, pair in enumerate(pairs):
        a, b = [images[m['observation_id']] for m in pair]
        az0, az1 = [float(m['stimulus_state']['azimuth_deg']) for m in pair]
        contrasts = [m['stimulus_state']['parameters']['contrast'] for m in pair]
        delta = az1-az0
        stationary = abs(delta) <= stationary_atol
        identical = bool(np.array_equal(a, b))
        visible = all(m['stimulus_state']['visible'] and c > 0 for m, c in zip(pair, contrasts))
        parameters_same = pair[0]['stimulus_state']['parameters'] == pair[1]['stimulus_state']['parameters']
        if not stationary and identical:
            category = 'visually_unobservable_motion'
        elif not stationary and visible and parameters_same:
            category = 'visible_motion'
        elif stationary and visible:
            category = 'visible_stationary'
        elif stationary and identical:
            category = 'blank_stationary_control'
        else:
            category = 'other_visual_change'
        pixel_delta = np.abs(a.astype(np.int16)-b.astype(np.int16))
        facts.append(dict(pair_index=index, trial_id=pair[0]['trial_id'],
                          observation_ids=[m['observation_id'] for m in pair],
                          schedule_indices=[m['schedule_index'] for m in pair],
                          session_times=[m['session_time'] for m in pair], azimuth_deg=[az0, az1],
                          signed_displacement_deg=delta, absolute_displacement_deg=abs(delta),
                          dt_seconds=pair[1]['session_time']-pair[0]['session_time'], stationary=stationary,
                          direction='stationary' if stationary else ('left' if delta < 0 else 'right'),
                          stimulus_generator_contrast=contrasts,
                          contrast_stratum=str(contrasts[0]) if contrasts[0] == contrasts[1] else 'transition',
                          identical_source_images=identical, source_visible=visible,
                          visibility_category=category, positive_motion_eligible=category == 'visible_motion',
                          source_rgb_mean_absolute_difference=float(pixel_delta.mean()),
                          source_rgb_max_absolute_difference=int(pixel_delta.max()),
                          source_changed_channel_fraction=float(np.mean(pixel_delta != 0))))
    return facts


def assess_frames(images, metadata, gain, size):
    prepared, assessments = {}, {}
    for key, rgb in images.items():
        entry = dict(observation_id=key, trial_id=metadata[key]['trial_id'],
                     stimulus_generator_contrast=metadata[key]['stimulus_state']['parameters']['contrast'],
                     clip_input_contrast_gain=gain, source_value_range=[int(rgb.min()), int(rgb.max())])
        try:
            image, info = transform(rgb, size, {'contrast': gain})
            unbounded = 128 + gain*(rgb.astype(np.float64)-128)
            reasons = []
            if info['clipping_fraction'] > 0:
                reasons.append('clip_input_pixel_clipping')
            if info['aspect_ratio_distorted']:
                reasons.append('geometric_distortion')
            if info['removed_non_neutral_content']:
                reasons.append('lost_source_content')
            entry.update(info, unbounded_enhancement_range=[float(unbounded.min()), float(unbounded.max())],
                         clipped_rgb8_range=[int(np.rint(np.clip(unbounded, 0, 255)).min()),
                                             int(np.rint(np.clip(unbounded, 0, 255)).max())],
                         prepared_value_range=[int(np.asarray(image).min()), int(np.asarray(image).max())],
                         eligibility_reasons=reasons, eligible=not reasons)
            if not reasons:
                prepared[key] = image
        except (ValueError, RuntimeError, OSError) as exc:
            entry.update(eligible=False, eligibility_reasons=[f'preparation_failure: {exc}'])
        assessments[key] = entry
    return prepared, assessments


def safe_infer(encoder, prepared, order, batch_size):
    """Retain usable frames when a batch or an individual image fails inference."""
    collected = {'projected_cls': {}, 'layer_24_cls': {}}
    failures = {}
    for start in range(0, len(order), batch_size):
        ids = order[start:start+batch_size]
        try:
            output = infer(encoder, prepared, ids, batch_size)
            for key in collected:
                collected[key].update(output[key])
        except (ValueError, RuntimeError) as exc:
            if len(ids) > 1:
                output, errors = safe_infer(encoder, prepared, ids, 1)
                for key in collected:
                    collected[key].update(output[key])
                failures.update(errors)
            else:
                failures[ids[0]] = str(exc)
    return collected, failures


def coverage(rows):
    from collections import Counter

    # Count each reason once per excluded pair; endpoint-specific details stay in rows.
    reasons = Counter(reason for r in rows for reason in {v.split(': ', 1)[-1] for v in r['eligibility_reasons']})
    return dict(selected_pairs=len(rows), eligible_pairs=sum(r['eligible'] for r in rows),
                excluded_pairs=sum(not r['eligible'] for r in rows), exclusion_reasons=dict(reasons),
                excluded_pair_indices=[r['pair_index'] for r in rows if not r['eligible']])


def summarize(rows, singleton_drift):
    positives = [r for r in rows if r['positive_motion_eligible']]
    stationary = [r for r in rows if r['visibility_category'] == 'visible_stationary']
    controls = [r for r in rows if r['visibility_category'] in ('visually_unobservable_motion', 'blank_stationary_control')]
    noise = max([1e-12, singleton_drift, *[r['context_distance_drift'] for r in rows],
                 *[r['cosine_distance'] for r in rows if r['identical_source_images']]])
    def corr(items):
        return correlation([r['absolute_displacement_deg'] for r in items], [r['cosine_distance'] for r in items])
    split = float(np.median([r['absolute_displacement_deg'] for r in positives])) if positives else None
    small = [r for r in positives if r['absolute_displacement_deg'] <= split]
    large = [r for r in positives if r['absolute_displacement_deg'] > split]
    floor = max([noise, *[r['cosine_distance'] for r in stationary]])
    edges = np.unique(np.quantile([r['absolute_displacement_deg'] for r in positives], np.linspace(0, 1, 5))).tolist() if positives else []
    bins = []
    for i, (lo, hi) in enumerate(zip(edges, edges[1:])):
        values = [r['cosine_distance'] for r in positives if lo <= r['absolute_displacement_deg'] and
                  (r['absolute_displacement_deg'] <= hi if i == len(edges)-2 else r['absolute_displacement_deg'] < hi)]
        bins.append(dict(lower_deg=lo, upper_deg=hi, upper_inclusive=i == len(edges)-2, **distribution(values)))
    return dict(eligible_pairs=len(rows), visible_motion_pairs=len(positives),
                correlation= corr(positives + stationary), moving_only_correlation=corr(positives),
                correlation_population='Visible stationary plus source-differing visible motion; unobservable motion and blank controls excluded',
                source_difference_correlation=correlation([r['source_rgb_mean_absolute_difference'] for r in positives], [r['cosine_distance'] for r in positives]),
                distributions={label: distribution([r['cosine_distance'] for r in subset]) for label, subset in
                               [('visible_motion', positives), ('visible_stationary', stationary),
                                ('negative_controls', controls), ('small_movement', small), ('larger_movement', large)]},
                numerical_stability=dict(empirical_noise_envelope=noise,
                                         moving_fraction_above_noise=float(np.mean([r['cosine_distance'] > noise for r in positives])) if positives else None,
                                         negative_control_max_distance=max([r['cosine_distance'] for r in controls], default=None),
                                         identical_source_pair_count=sum(r['identical_source_images'] for r in rows)),
                separation=dict(stationary_max_or_noise=floor,
                                small_motion_fraction_above_stationary_max_or_noise=float(np.mean([r['cosine_distance'] > floor for r in small])) if small else None),
                small_motion_upper_deg=split, displacement_bins=bins, displacement_bin_edges_deg=edges)


def run(args, report):
    import torch

    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    eid = str(UUID(load_config(args.run_config)['eid']))
    path = args.visual_dir/f'{eid}_visual_clip.npz'
    print(f'Verifying feature archive: {path}', flush=True)
    pairs, definition, completion, skipped = select_pairs(path, eid, args)
    report.update(eid=eid, feature_source=str(path), feature_generation_id=completion['generation_id'],
                  configurations=CONFIGURATIONS, selected_pairs=pairs,
                  selected_trial_ids=sorted({p[0]['trial_id'] for p in pairs}),
                  selection=dict(method='Uniform trial coverage, stationary/left/right quotas and displacement quantiles; no embedding-based selection', skipped_metadata_pairs=skipped),
                  contrast_operation='Fixed pivot: 128 + clip_input_contrast_gain*(RGB8-128), float64; previous operation clips [0,255], numpy rint, then bicubic fit/pad. Any pre-rounding overflow is now ineligible, never inferred.',
                  rejection_investigation=dict(cause='The original global geometry_pass combined no cropping, no distortion and zero clipping. Gain 1.5 overflows channels below 42.6666667 or above 212.6666667, so any such frame aborted all configurations. The subsequent run allowed clipping; this revision excludes it per frame/configuration.',
                                               clipping_fraction_definition='Fraction of source RGB channel values outside [0,255] after enhancement, before rounding, resizing or padding.'),
                  source_verification='Full feature archive verification; selected source metadata and pixels verified.')
    metadata = {m['observation_id']: m for pair in pairs for m in pair}
    if args.source_cache:
        with np.load(args.source_cache, allow_pickle=False) as cache:
            images = {key: cache[key].copy() for key in metadata}
        for key, rgb in images.items():
            if (rgb.dtype != np.uint8 or list(rgb.shape) != metadata[key]['shape'] or
                    hashlib.sha256(rgb.tobytes()).hexdigest() != metadata[key]['rgb_sha256']):
                raise ValueError(f'Audit cache source identity mismatch: {key}')
        report['source_verification'] += ' Audit-owned cache shape/dtype/RGB SHA256 checked against verified original feature metadata; replay payloads not reopened.'
        report['source_cache_sha256'] = hashlib.sha256(args.source_cache.read_bytes()).hexdigest()
    else:
        source = resolve_replay(args.replay_dir, eid)
        images = recover(source, [[(m, None) for m in pair] for pair in pairs], definition, completion, args.workers)
        np.savez_compressed(args.output_dir/'source-frames.npz', **images)
    facts = pair_facts(pairs, images, args.stationary_atol_deg)
    report['source_pair_facts'] = facts
    report['visibility_counts'] = {kind: sum(r['visibility_category'] == kind for r in facts)
                                   for kind in sorted({r['visibility_category'] for r in facts})}
    rep = definition['representation']
    if rep['model'] != 'openai/clip-vit-large-patch14':
        raise ValueError('Require CLIP ViT-L/14')
    encoder = ClipEncoder(rep['model'], revision=rep['revision'], device=args.device or rep['device'], precision='float32',
                          feature_cache_size=0, attention_backend=rep.get('inference', {}).get('attention_backend', 'eager'))
    actual = encoder.provenance
    for key in ('preparation', 'processor', 'artifact_hashes'):
        if actual[key] != rep[key]:
            raise ValueError(f'Checkpoint/preparation mismatch: {key}')
    report['loaded_representation'] = actual
    report['audit_source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                    (Path(__file__), Path(__file__).with_name('audit_clip_sensitivity.py'))}
    measurements, frame_assessments, execution = [], [], {}
    report.update(pair_measurements=measurements, frame_assessments=frame_assessments, execution=execution)
    # Recompute vectors: old reports did not persist intermediate representations,
    # and removing ineligible frames changes batch contexts. Never reuse stale vectors.
    for gain in (1.0, 1.5):
        prepared, assessments = assess_frames(images, metadata, gain, actual['preparation']['image_size'])
        order = list(prepared)
        print(f'Input gain {gain}: {len(order)}/{len(images)} eligible frames', flush=True)
        started = time.perf_counter()
        if torch.device(encoder._device).type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        first, failure1 = safe_infer(encoder, prepared, order, args.batch_size)
        repeat, failure2 = safe_infer(encoder, prepared, order, args.batch_size)
        alternate, failure3 = safe_infer(encoder, prepared, order[::-1], max(1, args.batch_size//2))
        probes = spread(order, args.stability_frames)
        singleton, failure4 = safe_infer(encoder, prepared, probes, 1)
        seconds = time.perf_counter()-started
        for config in [c for c in CONFIGURATIONS if c['clip_input_contrast_gain'] == gain]:
            name, representation = config['identifier'], config['representation']
            vectors, repeats, alternatives = [v[representation] for v in (first, repeat, alternate)]
            valid = set(vectors) & set(repeats) & set(alternatives)
            failed = {**failure1, **failure2, **failure3, **failure4}
            for key, entry in assessments.items():
                entry = dict(entry, configuration=name)
                if key in failed:
                    entry.update(eligible=False, eligibility_reasons=entry['eligibility_reasons'] + [f'inference_failure: {failed[key]}'])
                    valid.discard(key)
                frame_assessments.append(entry)
            probes_valid = [key for key in probes if key in valid and key in singleton[representation]]
            drift = max([distance(vectors[key], singleton[representation][key]) for key in probes_valid], default=0.0)
            execution[name] = dict(runtime_group_seconds=seconds, shared_gain_group=gain, singleton_image_drift=drift,
                                   singleton_probe_observation_ids=probes_valid,
                                   repeat_max_abs=max([float(np.max(np.abs(vectors[key]-repeats[key]))) for key in valid], default=None),
                                   context_max_abs=max([float(np.max(np.abs(vectors[key]-alternatives[key]))) for key in valid], default=None),
                                   peak_cuda_bytes=torch.cuda.max_memory_allocated() if torch.device(encoder._device).type == 'cuda' else None,
                                   inference_failures=failed, vector_bytes_float32=config['dimension']*4)
            for fact in facts:
                a, b = fact['observation_ids']
                reasons = [f'{key}: {reason}' for key in (a, b) for reason in assessments[key]['eligibility_reasons']]
                reasons += [f'{key}: inference_failure: {failed[key]}' for key in (a, b) if key in failed]
                eligible = a in valid and b in valid
                if not eligible and not reasons:
                    reasons = ['Representation unavailable in required inference contexts']
                row = dict(fact, configuration=name, dimension=config['dimension'], eligible=eligible,
                           eligibility_reasons=reasons, cosine_distance=None, cosine_similarity=None, context_distance_drift=None,
                           identical_prepared_images=bool(np.array_equal(np.asarray(prepared[a]), np.asarray(prepared[b]))) if a in prepared and b in prepared else None)
                if eligible:
                    d = distance(vectors[a], vectors[b])
                    row.update(cosine_distance=d, cosine_similarity=1-d,
                               context_distance_drift=max(abs(d-distance(context[a], context[b])) for context in (repeats, alternatives)))
                measurements.append(row)
            # Persist audit intermediates with explicit order and context provenance.
            keys = [key for key in order if key in valid]
            if keys:
                np.savez_compressed(args.output_dir/f'{name}-vectors.npz', observation_ids=np.asarray(keys),
                                    first=np.stack([vectors[k] for k in keys]), repeat=np.stack([repeats[k] for k in keys]),
                                    alternate=np.stack([alternatives[k] for k in keys]))
    eligible_sets = [{r['pair_index'] for r in measurements if r['configuration'] == c['identifier'] and r['eligible']} for c in CONFIGURATIONS]
    common = set.intersection(*eligible_sets)
    report['common_eligible_pair_indices'] = sorted(common)
    report['counts'] = dict(selected_pairs=len(pairs), selected_frames=len(images), common_eligible_pairs=len(common),
                          common_visible_motion_pairs=sum(r['pair_index'] in common and r['positive_motion_eligible'] for r in facts))
    report['coverage'] = {}
    report['summaries'] = []
    for config in CONFIGURATIONS:
        name = config['identifier']
        rows = [r for r in measurements if r['configuration'] == name]
        eligible = [r for r in rows if r['eligible']]
        matched = [r for r in eligible if r['pair_index'] in common]
        report['coverage'][name] = dict(overall=coverage(rows),
            by_trial={str(trial): coverage([r for r in rows if r['trial_id'] == trial]) for trial in report['selected_trial_ids']},
            by_stimulus_generator_contrast={contrast: coverage([r for r in rows if r['contrast_stratum'] == contrast]) for contrast in sorted({r['contrast_stratum'] for r in rows})})
        singleton_drift = execution[name]['singleton_image_drift']
        def stratify(subset):
            return {contrast: summarize([r for r in subset if r['contrast_stratum'] == contrast], singleton_drift)
                    for contrast in sorted({r['contrast_stratum'] for r in rows})}
        summary = dict(configuration=name, dimension=config['dimension'], precision='float32',
                       per_configuration_eligible=summarize(eligible, singleton_drift),
                       per_configuration_by_contrast=stratify(eligible),
                       per_configuration_per_trial={str(trial): summarize([r for r in eligible if r['trial_id'] == trial], singleton_drift) for trial in report['selected_trial_ids']},
                       matched=summarize(matched, singleton_drift), matched_by_contrast=stratify(matched),
                       matched_per_trial={str(trial): summarize([r for r in matched if r['trial_id'] == trial], singleton_drift) for trial in report['selected_trial_ids']})
        report['summaries'].append(summary)
        print(f"{name}: eligible={len(eligible)}, matched={len(matched)}, visible motion={summary['matched']['visible_motion_pairs']}", flush=True)
    report['limitations'] = ['Reconstructed source azimuth and times; no measured display-refresh claim.',
                            'Stratified deterministic sample, dependent neighboring pairs; no population significance inference.',
                            'Primary comparisons use the same four-way intersection; per-configuration-only summaries are coverage diagnostics.',
                            'Positive motion requires source RGB differences, positive source contrast, visible state and unchanged stimulus parameters.',
                            'Identical source images with changing azimuth are negative controls, never positive motion targets.',
                            'RGB differences establish observable change, not uniquely recoverable position or displacement.',
                            'Gabor periodicity and starting position can break global monotonicity; distance is unsigned.',
                            'Empirical numerical envelopes cover sampled execution contexts only; no feature-space scale equivalence.',
                            'No decoding model was fitted; recoverable information and neural prediction improvement are unproven.']
    meaningful = any(s['per_configuration_eligible']['visible_motion_pairs'] >= 3 for s in report['summaries'])
    if not meaningful:
        report.update(status='blocked', blocker='No configuration has at least three eligible observable-motion pairs')
    else:
        report['status'] = 'ok' if common and report['counts']['common_visible_motion_pairs'] >= 3 else 'partial'
    report['recommendation'] = ('Use unenhanced projected CLS (768) as the reference in held-out position and signed-displacement decoding, with unenhanced layer 24 CLS as the extraction-only ablation. '
        'Compare gain 1.5 variants on the same unclipped, visible, contrast-stratified pairs; retain blank/identical-image motion as negative controls. '
        'Split by trial, fit any decoder on training trials only, and assess held-out position error and signed displacement error within source-contrast strata. '
        'Choose a representation only if decoding improves consistently across trials and contrasts above numerical noise. Larger cosine distances or pooled correlation alone do not establish recoverability. '
        'Most full-contrast pairs are clipped by gain 1.5; the matched sample is insufficient for a full-contrast correlation. Use the separately labeled unenhanced two-way summaries for that condition. No production integration is recommended from this audit alone.')
    write_analysis(args.output_dir, report)


def write_analysis(output, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    common = set(report['common_eligible_pair_indices'])
    contrasts = sorted({r['contrast_stratum'] for r in report['source_pair_facts']})
    fig, axes = plt.subplots(len(contrasts), 4, figsize=(17, 3.5*len(contrasts)), squeeze=False, constrained_layout=True)
    labels = {c['identifier']: f"gain {c['clip_input_contrast_gain']} {c['representation']} ({c['dimension']})" for c in CONFIGURATIONS}
    for j, contrast in enumerate(contrasts):
        for summary in report['summaries']:
            name = summary['configuration']
            rows = [r for r in report['pair_measurements'] if r['configuration'] == name and r['pair_index'] in common and r['contrast_stratum'] == contrast]
            positive = [r for r in rows if r['positive_motion_eligible']]
            stats = summary['matched_by_contrast'][contrast]
            label = labels[name]
            axes[j, 0].scatter([r['absolute_displacement_deg'] for r in positive], [r['cosine_distance'] for r in positive], s=16, alpha=.6, label=label)
            bins = [b for b in stats['displacement_bins'] if b['count']]
            axes[j, 1].plot([(b['lower_deg']+b['upper_deg'])/2 for b in bins], [b['median'] for b in bins], 'o-', label=label)
            for category, style in [('visible_motion', '-'), ('visible_stationary', '--'), ('visually_unobservable_motion', ':')]:
                values = sorted(r['cosine_distance'] for r in rows if r['visibility_category'] == category)
                if values:
                    axes[j, 2].plot(values, np.arange(1, len(values)+1)/len(values), style, label=f'{label}: {category}')
            noise = stats['numerical_stability']['empirical_noise_envelope']
            axes[j, 3].scatter([r['absolute_displacement_deg'] for r in positive], [max(r['cosine_distance']/noise, 1e-12) for r in positive], s=16, alpha=.6, label=label)
        for k, ax in enumerate(axes[j]):
            ax.grid(alpha=.2)
            ax.set_title(f'Source contrast {contrast}; matched pairs')
            if k != 2:
                ax.set_xlabel('Absolute angular displacement (degrees)')
            if ax.has_data():
                ax.legend(fontsize=5)
            else:
                ax.text(.5, .5, 'No eligible matched data', ha='center', transform=ax.transAxes)
        axes[j, 0].set_ylabel('Cosine distance; observable motion')
        axes[j, 1].set_ylabel('Median cosine distance')
        axes[j, 2].set(xlabel='Cosine distance', ylabel='Empirical cumulative fraction', xscale='symlog')
        axes[j, 2].set_xscale('symlog', linthresh=1e-10)
        distances = [r['cosine_distance'] for r in report['pair_measurements'] if r['pair_index'] in common and r['contrast_stratum'] == contrast]
        axes[j, 2].set_xlim(0, max([1e-10, *distances])*1.1)
        axes[j, 2].set_xticks([v for v in (0, 1e-10, 1e-6, 1e-2, 1e-1) if v <= axes[j, 2].get_xlim()[1]])
        positive_count = sum(r['pair_index'] in common and r['contrast_stratum'] == contrast and r['positive_motion_eligible'] for r in report['source_pair_facts'])
        if not positive_count:
            for k in (0, 1, 3):
                axes[j, k].text(.5, .5, 'No observable motion; negative controls only', ha='center', fontsize=8, transform=axes[j, k].transAxes)
        elif positive_count < 3:
            axes[j, 1].text(.5, .5, 'Insufficient moving pairs for bins/correlation', ha='center', fontsize=8, transform=axes[j, 1].transAxes)
        axes[j, 3].set(ylabel='Distance / sampled noise', yscale='log')
        axes[j, 3].axhline(1, color='black', linewidth=.8)
    fig.savefig(output/'comparison.png', dpi=150)
    plt.close(fig)
    lines = ['# Visibility-stratified CLIP motion audit', '', f"Status: {report['status']}; EID: {report['eid']}", '',
             f"Counts: {report['counts']}", '', f"Source visibility categories: {report['visibility_counts']}", '',
             report['rejection_investigation']['cause'], '', report['contrast_operation'], '',
             '## Preprocessing exclusions', '',
             '| Configuration | Trial | Source contrast | Ineligible frames | Source range | Unbounded gain range | Clipped channel fraction range |',
             '|---|---:|---:|---:|---|---|---|']
    for name in labels:
        for trial in report['selected_trial_ids']:
            frames = [f for f in report['frame_assessments'] if f['configuration'] == name and f['trial_id'] == trial and not f['eligible']]
            if frames:
                lines.append(f"| {name} | {trial} | {frames[0]['stimulus_generator_contrast']} | {len(frames)} | {[min(f['source_value_range'][0] for f in frames), max(f['source_value_range'][1] for f in frames)]} | {[min(f.get('unbounded_enhancement_range', [0,0])[0] for f in frames), max(f.get('unbounded_enhancement_range', [0,0])[1] for f in frames)]} | {[min(f.get('clipping_fraction',0) for f in frames), max(f.get('clipping_fraction',0) for f in frames)]} |")
    lines += ['', '## Per-configuration pair coverage', '', '| Configuration | Eligible | Excluded |', '|---|---:|---:|']
    for name, cov in report['coverage'].items():
        lines.append(f"| {name} | {cov['overall']['eligible_pairs']} | {cov['overall']['excluded_pairs']} |")
    lines += ['', '## Fair four-way comparison by original contrast', '',
              'Correlations below use source-differing visible moving pairs only. Stationary and blank controls are reported separately.', '',
              '| Source contrast | Configuration | Matched pairs | Observable moving | Pearson | Spearman | Small motion above stationary/noise | Noise envelope |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for contrast in contrasts:
        for summary in report['summaries']:
            stats = summary['matched_by_contrast'][contrast]
            lines.append(f"| {contrast} | {summary['configuration']} | {stats['eligible_pairs']} | {stats['visible_motion_pairs']} | {stats['moving_only_correlation']['pearson']} | {stats['moving_only_correlation']['spearman']} | {stats['separation']['small_motion_fraction_above_stationary_max_or_noise']} | {stats['numerical_stability']['empirical_noise_envelope']} |")
    lines += ['', '## Recommendation', '', report['recommendation'], '', '## Interpretation limits', '', *report['limitations'], '',
              'Generator, original stimulus parameters, stored source pixels, production features, CLIP weights and neural pipeline are unchanged.', '',
              'All per-frame exclusion reasons/ranges, per-trial and per-contrast coverage, per-pair source differences, distributions, correlations, bins and FP32 context probes are in `report.json`.', '',
              '![Matched comparison by source contrast](comparison.png)', '']
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-config', type=Path, default=REPO_ROOT/'src/configs/run.json')
    parser.add_argument('--visual-dir', type=Path, default=visual_dir())
    parser.add_argument('--replay-dir', type=Path, default=replay_dir())
    parser.add_argument('--output-dir', type=Path, default=AUDIT_ROOT/'clip-motion')
    parser.add_argument('--device', default=None)
    parser.add_argument('--source-cache', type=Path, default=None,
                        help='Optional prior audit source-frames.npz; verified against source RGB hashes')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--trial-limit', type=int, default=8)
    parser.add_argument('--pairs-per-trial', type=int, default=18)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--stability-frames', type=int, default=12)
    parser.add_argument('--stationary-atol-deg', type=float, default=1e-10)
    args = parser.parse_args()
    if min(args.workers, args.trial_limit, args.batch_size, args.stability_frames) < 1 or args.pairs_per_trial < 3:
        parser.error('Limits must be positive; pairs-per-trial must be at least 3')
    if not np.isfinite(args.stationary_atol_deg) or args.stationary_atol_deg < 0:
        parser.error('stationary-atol-deg must be finite and nonnegative')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(status='started', arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    started = time.perf_counter()
    try:
        run(args, report)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        report.update(status='blocked', blocker=str(exc))
        print(f'Audit blocked: {exc}', file=sys.stderr)
    finally:
        report['runtime_seconds'] = time.perf_counter()-started
        (args.output_dir/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        print(f"Report: {args.output_dir/'report.json'}", flush=True)
    return 0 if report['status'] in ('ok', 'partial') else 1


if __name__ == '__main__':
    sys.exit(main())
