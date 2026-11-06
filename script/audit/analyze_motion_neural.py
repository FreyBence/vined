"""Standalone descriptive physical-motion/spike audit; never fits the ViNED model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits

from audit_paths import ROOT, AUDIT_ROOT
from training_dataset import load_dataset
from utils.paths import visual_dir
from utils.provenance import fingerprint


DEFAULTS = dict(windows=[1, 3, 6, 12], max_lag=12, min_pairs=12,
                min_trials=8, bootstrap=300, seed=42, stationary_atol_deg=1e-10,
                event_radius=12, representative_trials=6, population_components=3)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding='utf-8')


def distribution(x):
    x = np.asarray(x); x = x[np.isfinite(x)]
    return dict(n=len(x), mean=x.mean() if len(x) else None,
                std=x.std() if len(x) else None,
                quantiles=np.quantile(x, [0, .25, .5, .75, .95, 1]) if len(x) else [])


def summary(x, cfg, rng):
    x = np.asarray(x); x = x[np.isfinite(x)]
    supported = len(x) >= cfg['min_trials']
    mean = x.mean() if len(x) else np.nan
    ci = (np.quantile(rng.choice(x, (cfg['bootstrap'], len(x))).mean(1), [.025, .975])
          if supported and cfg['bootstrap'] else None)
    return dict(trials=len(x), mean=mean, median=np.median(x) if len(x) else None,
                std=x.std() if len(x) else None, ci95=ci,
                sign_consistency=np.mean(np.sign(x) == np.sign(mean)) if len(x) else None,
                supported=supported,
                reason=None if supported else 'Insufficient independent trials')


def correlations(x, y):
    """Columnwise Pearson; undefined constant series stay unavailable."""
    x = x - x.mean(); y = y - y.mean(0)
    den = np.sqrt((x @ x) * (y*y).sum(0))
    return np.divide(x @ y, den, out=np.full(y.shape[1], np.nan), where=den > 1e-12)


def forward_mean(y, width):
    cs = np.vstack([np.zeros((1, y.shape[1])), np.cumsum(y, axis=0)])
    return (cs[width:] - cs[:-width]) / width


def metadata_records(path, eid, samples):
    """Verify metadata bytes without decoding unused CLIP vectors or replay pixels."""
    wanted = {s.trial_id for s in samples}; records = {t: [] for t in wanted}
    digest = hashlib.sha256()
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        definition = json.loads(archive.read('definition.json'))
        if fingerprint({k: v for k, v in manifest.items() if k != 'generation_id'}) != manifest['generation_id']:
            raise ValueError('Feature manifest identity mismatch')
        if fingerprint(definition) != manifest['definition_id']:
            raise ValueError('Feature definition identity mismatch')
        with archive.open('records.jsonl') as stream:
            for line in stream:
                digest.update(line); row = json.loads(line)
                if row['kind'] != 'observation':
                    continue
                meta = row['metadata']
                if meta['eid'] != eid:
                    raise ValueError('Feature metadata EID mismatch')
                if meta['trial_id'] in wanted and row.get('feature_index') is not None:
                    records[meta['trial_id']].append(meta)
    if digest.hexdigest() != manifest['records_sha256']:
        raise ValueError('Feature metadata digest mismatch')
    for sample in samples:
        al = sample.metadata['alignment']
        if al['visual_completion']['generation_id'] != manifest['generation_id']:
            raise ValueError('Dataset and feature generation mismatch')
    return records, manifest


def prepare(sample, records, cfg):
    n, nc = sample.sequence_length, sample.neuron_count
    if not np.isclose(sample.bin_size, 1/60, rtol=0, atol=1e-12):
        raise ValueError('Analysis requires existing 60 Hz bins; resampling counts is forbidden')
    if not sample.temporal_mask[:n].all() or not sample.neuron_mask[:nc].all():
        raise ValueError('Unexpected unsupported real sample cells')
    al = sample.metadata['alignment']; rs = al['resampling']
    times = np.array([r['session_time'] for r in records])
    if ([r['observation_id'] for r in records] != rs['source_observation_ids']
            or [r['schedule_index'] for r in records] != rs['source_schedule_indices']
            or not np.array_equal(times, rs['source_times'])
            or not np.array_equal(sample.physical_timestamps[:n], rs['query_times'])):
        raise ValueError('Source observation/time associations differ')
    if (len(times) < 2 or not np.allclose(np.diff(times), 1/60, rtol=0, atol=1e-8)
            or any(r['status'] != 'valid' or r['timing']['cadence_hz'] != 60 for r in records)):
        raise ValueError('Incomplete or non-60 Hz source schedule')
    left = np.array(rs['left_source_indices']); right = np.array(rs['right_source_indices'])
    weight = np.array(rs['interpolation_weights'])
    state = [r['stimulus_state'] for r in records]
    az = np.array([s['azimuth_deg'] for s in state], dtype=float)
    az = (1-weight)*az[left] + weight*az[right]
    contrast = sample.metadata['alignment']['visual_definition']['trial']['parameters']['contrast']
    if any(s['parameters']['contrast'] != contrast for s in state):
        raise ValueError('Original contrast changes inside trial')
    visible = np.array([s['visible'] for s in state], dtype=bool)
    visible = visible[left] & visible[right]
    d = np.r_[np.nan, np.diff(az)]
    valid_motion = visible & np.r_[False, visible[:-1]]
    d[~valid_motion] = np.nan
    features = {'azimuth': np.where(visible, az, np.nan), 'signed_1': d, 'absolute_1': abs(d),
                'direction': np.where(np.isfinite(d), np.where(abs(d) <= cfg['stationary_atol_deg'], 0, np.sign(d)), np.nan)}
    for width in cfg['windows']:
        net = np.full(n, np.nan); travel = net.copy()
        for t in range(width, n):
            if np.isfinite(d[t-width+1:t+1]).all():
                net[t] = az[t] - az[t-width]
                travel[t] = abs(d[t-width+1:t+1]).sum()
        if width != 1:
            features[f'net_{width}'] = net; features[f'travel_{width}'] = travel
    source_equal = np.array([records[i]['rgb_sha256'] == records[j]['rgb_sha256'] for i, j in zip(left, right)])
    return dict(sample=sample, az=az, d=d, visible=visible, contrast=float(contrast),
                stage=np.array([state[i]['stage'] for i in left]), features=features,
                y=sample.neural[:n, :nc].astype(float), source_equal=source_equal,
                time=sample.physical_timestamps[:n]-sample.stim_on)


def associations(trials, targets, cfg, out, rng, neuron=False):
    names = ['absolute_1'] if neuron else list(trials[0]['features'])
    configs = [(f, w, lag) for f in names for w in cfg['windows'] for lag in range(cfg['max_lag']+1)]
    shape = (len(trials), len(configs), len(targets))
    pearson = np.full(shape, np.nan, dtype=np.float32); spearman = pearson.copy(); null = pearson.copy()
    counts = np.zeros(shape[:2], dtype=np.int32)
    for ti, tr in enumerate(trials):
        y = tr['y']*60 if neuron else tr['targets']
        means = {w: forward_mean(y, w) for w in cfg['windows']}
        # Common support for every lag and integration width, within this trial.
        t = np.arange(max(cfg['windows']), len(y)-cfg['max_lag']-max(cfg['windows'])+1)
        for ci, (feature, width, lag) in enumerate(configs):
            x = tr['features'][feature][t]; response = means[width][t+lag]
            good = np.isfinite(x) & np.isfinite(response).all(1)
            x, response = x[good], response[good]; counts[ti, ci] = len(x)
            if len(x) < cfg['min_pairs']:
                continue
            pearson[ti, ci] = correlations(x, response)
            spearman[ti, ci] = correlations(rankdata(x), rankdata(response, axis=0))
            null[ti, ci] = correlations(np.roll(x, len(x)//2), response)
        if (ti+1) % 50 == 0:
            print(f"{'Neuron' if neuron else 'Region/population'} associations: {ti+1}/{len(trials)}", flush=True)
    np.savez_compressed(out/('neuron-associations.npz' if neuron else 'associations.npz'),
                        pearson=pearson, spearman=spearman, half_trial_shift=null, pair_counts=counts,
                        trial_ids=[t['sample'].trial_id for t in trials],
                        targets=np.array(targets), features=np.array([c[0] for c in configs]),
                        response_windows=[c[1] for c in configs], lags=[c[2] for c in configs])
    rows = []
    for contrast in sorted({t['contrast'] for t in trials}):
        use = np.array([t['contrast'] == contrast for t in trials])
        for ci, (feature, width, lag) in enumerate(configs):
            for j, target in enumerate(targets):
                r = summary(pearson[use, ci, j], cfg, rng)
                row = dict(contrast=contrast, target=target, feature=feature, response_window=width,
                           lag_bins=lag, lag_ms=lag*1000/60, pearson=r,
                           spearman=summary(spearman[use, ci, j], {**cfg, 'bootstrap': 0}, rng),
                           shift_mean=distribution(null[use, ci, j])['mean'],
                           contributing_pairs=int(counts[use, ci][np.isfinite(pearson[use, ci, j])].sum()))
                rows.append(row)
    return rows


def events_and_bins(trials, regions, cfg, out, rng):
    event_rows = []; binned = []; thresholds = {}
    for contrast in sorted({t['contrast'] for t in trials}):
        group = [t for t in trials if t['contrast'] == contrast]
        d = np.concatenate([abs(t['d']) for t in group]); positive = d[np.isfinite(d) & (d > cfg['stationary_atol_deg'])]
        edges = np.unique(np.quantile(positive, [0, .25, .5, .75, 1])) if len(positive) else np.array([])
        thresholds[str(contrast)] = dict(positive_displacement=distribution(positive), bin_edges=edges,
                                       criterion='Within-contrast displacement quartiles, independent of neural activity')
        for ri, region in enumerate(regions):
            for bi in range(-1, max(0, len(edges)-1)):
                vals = []; observations = 0; baseline_effect = []
                for tr in group:
                    magnitude = abs(tr['d']); stationary = np.isfinite(magnitude) & (magnitude <= cfg['stationary_atol_deg'])
                    mask = stationary if bi == -1 else ((magnitude >= edges[bi]) & (magnitude <= edges[bi+1]) if bi == len(edges)-2 else (magnitude >= edges[bi]) & (magnitude < edges[bi+1]))
                    # Exclude floating point stationary values from positive bins.
                    if bi >= 0: mask &= magnitude > cfg['stationary_atol_deg']
                    if mask.any():
                        vals.append(tr['targets'][mask, ri].mean()); observations += int(mask.sum())
                        if stationary.any(): baseline_effect.append(tr['targets'][mask, ri].mean()-tr['targets'][stationary, ri].mean())
                binned.append(dict(contrast=contrast, region=region, bin='stationary' if bi == -1 else bi,
                                   bounds_deg=None if bi == -1 else edges[bi:bi+2], observations=observations,
                                   rate_hz_per_neuron=summary(vals, cfg, rng),
                                   paired_stationary_difference_hz=summary(baseline_effect, cfg, rng)))
            radius = cfg['event_radius']
            for kind in ['onset', 'cessation', 'reversal', 'small', 'large']:
                curves = []; event_count = 0; trial_ids = []
                for tr in group:
                    delta = tr['d']; finite = np.isfinite(delta)
                    moving = finite & (abs(delta) > cfg['stationary_atol_deg']); direction = np.sign(delta)
                    previous = np.r_[False, moving[:-1]]; adjacent = finite & np.r_[False, finite[:-1]]
                    select = {'onset': moving & ~previous, 'cessation': ~moving & previous,
                              'reversal': moving & previous & (direction != np.r_[np.nan, direction[:-1]]),
                              'small': moving & (abs(delta) <= (np.median(positive) if len(positive) else 0)),
                              'large': moving & (abs(delta) > (np.median(positive) if len(positive) else np.inf))}[kind] & adjacent
                    indices = np.flatnonzero(select); kept = []; last = -2*radius-1
                    for idx in indices:
                        if idx-radius >= 0 and idx+radius < len(delta) and idx-last > 2*radius and tr['visible'][idx-radius:idx+radius+1].all():
                            kept.append(idx); last = idx
                    if kept:
                        curves.append(np.mean([tr['targets'][i-radius:i+radius+1, ri] for i in kept], axis=0))
                        trial_ids.append(tr['sample'].trial_id); event_count += len(kept)
                curves = np.array(curves)
                points = [summary(curves[:, k], cfg, rng) for k in range(2*radius+1)] if len(curves) else []
                event_rows.append(dict(contrast=contrast, region=region, event=kind, events=event_count,
                                       trials=len(curves), trial_ids=trial_ids, relative_ms=np.arange(-radius, radius+1)*1000/60,
                                       curve=points, trial_curves=curves))
    write(out/'movement-bins.json', binned); write(out/'events.json', event_rows)
    return thresholds, binned, event_rows


def figures(trials, regions, rows, binned, events, cfg, out):
    (out/'trials').mkdir(exist_ok=True); (out/'regions').mkdir(exist_ok=True)
    # Uniform deterministic trial coverage; no neural-effect based selection.
    selected = np.unique(np.linspace(0, len(trials)-1, min(len(trials), cfg['representative_trials']), dtype=int))
    for index in selected:
        tr = trials[index]; fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        for ax, values, label in zip(axes[:3], [tr['az'], tr['d']*60, abs(tr['d'])], ['Azimuth (deg)', 'Velocity (deg/s)', '|Displacement| (deg)']):
            ax.plot(tr['time'], values); ax.set_ylabel(label)
        for j, name in enumerate(regions):
            # Causal trailing three-bin mean for display only, timestamps at final bin.
            axes[3].plot(tr['time'][2:], forward_mean(tr['targets'][:, j:j+1], 3)[:, 0], label=name)
        axes[3].set_ylabel('Hz / neuron (3-bin trailing)'); axes[3].legend(ncol=4, fontsize=7)
        axes[3].set_xlabel('Seconds after stimulus onset'); fig.suptitle(f"Trial {tr['sample'].trial_id}; original contrast {tr['contrast']:g}")
        fig.tight_layout(); fig.savefig(out/'trials'/f"trial-{tr['sample'].trial_id}.png", dpi=120); plt.close(fig)
    contrasts = sorted({t['contrast'] for t in trials})
    for region in regions:
        fig, axes = plt.subplots(len(contrasts), 3, figsize=(16, 3.3*len(contrasts)), squeeze=False)
        for i, contrast in enumerate(contrasts):
            for window in cfg['windows']:
                subset = [r for r in rows if r['target'] == region and r['contrast'] == contrast and r['feature'] == 'absolute_1' and r['response_window'] == window]
                line, = axes[i, 0].plot([r['lag_ms'] for r in subset], [r['pearson']['mean'] for r in subset], label=f'{window} bins')
                ci = np.array([r['pearson']['ci95'] if r['pearson']['ci95'] is not None else [np.nan, np.nan] for r in subset])
                axes[i, 0].fill_between([r['lag_ms'] for r in subset], ci[:, 0], ci[:, 1], color=line.get_color(), alpha=.1)
            axes[i, 0].axhline(0, color='gray', lw=.7); axes[i, 0].legend(fontsize=7); axes[i, 0].set_xlabel('Stimulus precedes response (ms)'); axes[i, 0].set_ylabel(f'Contrast {contrast:g}\nTrial-mean Pearson r')
            subset = [r for r in binned if r['region'] == region and r['contrast'] == contrast]
            axes[i, 1].bar(range(len(subset)), [r['rate_hz_per_neuron']['mean'] for r in subset])
            for j, row in enumerate(subset):
                ci = row['rate_hz_per_neuron']['ci95']
                if ci is not None: axes[i, 1].vlines(j, ci[0], ci[1], color='black', lw=1)
            axes[i, 1].set_xticks(range(len(subset)), [f"{r['bin']}\n{r['rate_hz_per_neuron']['trials']} trials" for r in subset], fontsize=7)
            axes[i, 1].set_ylabel('Hz / neuron'); axes[i, 1].set_xlabel('Displacement quartile; stationary separate')
            for event in events:
                if event['region'] == region and event['contrast'] == contrast and event['curve'] and event['trials'] >= cfg['min_trials']:
                    mean = np.array([p['mean'] for p in event['curve']])
                    line, = axes[i, 2].plot(event['relative_ms'], mean, label=f"{event['event']} ({event['trials']} trials)")
                    ci = np.array([p['ci95'] if p['ci95'] is not None else [np.nan, np.nan] for p in event['curve']])
                    axes[i, 2].fill_between(event['relative_ms'], ci[:, 0], ci[:, 1], color=line.get_color(), alpha=.08)
            axes[i, 2].axvline(0, color='gray', lw=.7); axes[i, 2].legend(fontsize=6); axes[i, 2].set_xlabel('Time from movement event (ms)'); axes[i, 2].set_ylabel('Hz/neuron; supported events only')
        fig.suptitle(f'{region}: descriptive motion relationships, trial-weighted'); fig.tight_layout()
        fig.savefig(out/'regions'/f'{region.replace("/", "_")}.png', dpi=120); plt.close(fig)


def report_text(result, out):
    rows = result['associations']; regions = result['region_coverage']
    winners = []
    for region in regions:
        eligible = [r for r in rows if r['target'] == region and r['contrast'] > 0 and r['pearson']['supported'] and r['pearson']['mean'] is not None]
        if eligible: winners.append(max(eligible, key=lambda r: abs(r['pearson']['mean'])))
    winners.sort(key=lambda r: abs(r['pearson']['mean']), reverse=True)
    result['regional_descriptive_maxima'] = winners
    lines = ['# Physical stimulus motion and neural activity', '',
             f"EID: `{result['eid']}`. {result['trials']} trials; {result['neurons']} units. This is an exploratory audit of all available dataset splits, not held-out model evaluation.", '',
             '## Motion, neural changes, direction and history (questions 1–4, 6, 9)', '',
             'The JSON compares absolute position, signed displacement, direction, absolute displacement, net movement and traveled distance. Associations describe subsequent forward-mean firing rates; negative signs indicate decreases. Movement-bin effects compare each trial to its own stationary bins. Events include peaks/troughs and sustained shapes without assuming motion increases firing.', '',
             '## Lag, contrast and regions (questions 5, 7, 8)', '',
             '| Region label | Units | Contrast | Variable | Response bins | Lag ms | Trial mean r | Trials | Sign agreement |',
             '|---|---:|---:|---|---:|---:|---:|---:|---:|']
    for r in winners:
        s = r['pearson']; lines.append(f"| {r['target']} | {regions[r['target']]} | {r['contrast']:g} | {r['feature']} | {r['response_window']} | {r['lag_ms']:.1f} | {s['mean']:.3f} | {s['trials']} | {s['sign_consistency']:.2f} |")
    lines += ['', '### Variable and integration comparison', '',
              '| Variable | Region | Contrast | Window bins | Lag ms | Mean r | Shift reference r | 95% trial bootstrap interval |',
              '|---|---|---:|---:|---:|---:|---:|---|']
    comparisons = []
    for feature in dict.fromkeys(r['feature'] for r in rows):
        eligible = [r for r in rows if r['target'] in regions and r['contrast'] > 0 and r['feature'] == feature and r['pearson']['supported']]
        if not eligible: continue
        r = max(eligible, key=lambda x: abs(x['pearson']['mean'])); comparisons.append(r)
        ci = r['pearson']['ci95']; ci_text = f'[{ci[0]:.3f}, {ci[1]:.3f}]' if ci is not None else 'unavailable'
        shifted = f"{r['shift_mean']:.3f}" if r['shift_mean'] is not None else 'unavailable'
        lines.append(f"| {feature} | {r['target']} | {r['contrast']:g} | {r['response_window']} | {r['lag_ms']:.1f} | {r['pearson']['mean']:.3f} | {shifted} | {ci_text} |")
    result['variable_descriptive_maxima'] = comparisons
    if winners:
        r = winners[0]; s = r['pearson']
        lines += ['', f"The largest supported region-level association in this search is {r['target']} with {r['feature']} at contrast {r['contrast']:g}: mean r={s['mean']:.3f}, {s['trials']} contributing trials, sign agreement {s['sign_consistency']:.0%}. The response window is {r['response_window']} bins beginning {r['lag_ms']:.1f} ms after the stimulus. This is evidence of an observable temporal relationship within the selected search, with the confounds below still unresolved."]
    lines += ['', '### Response integration windows', '',
              '| Response bins | Region | Contrast | Absolute displacement mean r | Lag ms | Trial sign agreement |',
              '|---:|---|---:|---:|---:|---:|']
    for width in result['config']['windows']:
        eligible = [r for r in rows if r['target'] in regions and r['contrast'] > 0 and r['feature'] == 'absolute_1' and r['response_window'] == width and r['pearson']['supported']]
        if eligible:
            r = max(eligible, key=lambda x: abs(x['pearson']['mean']))
            lines.append(f"| {width} | {r['target']} | {r['contrast']:g} | {r['pearson']['mean']:.3f} | {r['lag_ms']:.1f} | {r['pearson']['sign_consistency']:.2f} |")
    lines += ['', 'Each row above is a separate selected maximum; different winning regions/contrasts are not matched evidence that one window is superior. Use region plots for matched lag/window comparisons.']
    bins = read(out/'movement-bins.json')
    effects = [b for b in bins if b['bin'] != 'stationary' and b['contrast'] > 0 and b['paired_stationary_difference_hz']['supported']]
    effects.sort(key=lambda b: abs(b['paired_stationary_difference_hz']['mean']), reverse=True)
    lines += ['', '### Motion magnitude and activity changes', '',
              '| Region label | Units | Contrast | Displacement quartile | Paired difference from stationary (Hz/neuron) | Independent trials |',
              '|---|---:|---:|---:|---:|---:|']
    for b in effects[:8]:
        s = b['paired_stationary_difference_hz']
        lines.append(f"| {b['region']} | {regions[b['region']]} | {b['contrast']:g} | {b['bin']} | {s['mean']:.3f} | {s['trials']} |")
    lines += ['', 'Large absolute rate effects can be driven by a single recorded unit (for example labels with one unit). Anatomical labels are retained verbatim and are not automatically suitable regional targets. Compare unit coverage, trial consistency and uncertainty before prioritizing a region.']
    event_effects = []
    for e in read(out/'events.json'):
        if e['trials'] < result['config']['min_trials']: continue
        curves = np.asarray(e['trial_curves']); radius = result['config']['event_radius']; w = min(6, radius)
        changes = curves[:, radius:radius+w].mean(1)-curves[:, radius-w:radius].mean(1)
        event_effects.append(dict(region=e['region'], contrast=e['contrast'], event=e['event'],
                                 mean_change_hz=changes.mean(), std_change_hz=changes.std(),
                                 trials=len(changes), positive_fraction=np.mean(changes > 0)))
    event_effects.sort(key=lambda e: abs(e['mean_change_hz']), reverse=True)
    result['event_pre_post_changes'] = event_effects
    lines += ['', '### Event-related temporal shape', '',
              'The following selected effects compare the fixed first 100 ms after an event to the preceding 100 ms (or a shorter configured radius). Each trial contributes its own event-averaged curve. Full plots retain the response shape and zero-contrast controls.', '',
              '| Region | Contrast | Event | Post minus pre (Hz/neuron) | Trials | Positive trial fraction |',
              '|---|---:|---|---:|---:|---:|']
    for e in event_effects[:8]:
        lines.append(f"| {e['region']} | {e['contrast']:g} | {e['event']} | {e['mean_change_hz']:.3f} | {e['trials']} | {e['positive_fraction']:.2f} |")
    lines += ['', '### Population variation and zero-contrast context', '',
              f"The first {len(result['population']['explained_variance_fraction'])} population PCs explain " + ', '.join(f'{v:.1%}' for v in result['population']['explained_variance_fraction']) + ' of trial-centered raw-count variance. These components can reveal opposing neuron responses, but are descriptive and may be dominated by high-rate units. Their complete lag curves are in JSON and NPZ.', '']
    zero = [r for r in rows if r['contrast'] == 0 and r['target'] in regions and r['feature'] == 'absolute_1' and r['pearson']['supported']]
    if zero:
        z = max(zero, key=lambda r: abs(r['pearson']['mean']))
        lines.append(f"Zero-contrast motion has a largest selected regional mean correlation of {z['pearson']['mean']:.3f} ({z['target']}, {z['pearson']['trials']} trials). Association without visible contrast reinforces the need to consider closed-loop motor/state/task timing confounds; it does not isolate their contributions.")
    else:
        lines.append('Zero-contrast trials do not provide enough varying-motion trial support for a regional correlation summary; see contrast coverage and unavailable results.')
    lines += ['', 'These are selected descriptive maxima over many dependent comparisons, with changing contrast-specific trial support. Their bootstrap intervals are pointwise, not corrected for selection. They are not physiological latency estimates or rankings of causal visual sensitivity. Inspect the full lag curves, the half-trial shifted reference, neuron summaries and trial variability before interpreting any maximum.', '',
              '## Practical model considerations (question 10)', '',
              '1. Compare signed movement, position and accumulated physical movement as candidate auxiliary inputs in a future held-out-trial experiment. The audit establishes association only; it does not establish predictive gain. Complexity/cost: low to moderate.',
              '2. Compare fixed stimulus-response shifts and 1/3/6/12-bin response aggregation on identical held-out trials. Use the observed curve breadth and trial consistency, rather than the single highest correlation. The present 12-bin context covers about 200 ms; this audit cannot assess longer dependencies. Complexity/cost: low.',
              '3. Evaluate temporal curves by region alongside individual neurons. Prioritize regions with stable signed effects across trials and contrasts in the table and full results; opposing neuron responses may disappear in region means. Population PCs provide an additional descriptive check, not a learned encoding target. Complexity/cost: moderate.', '',
              'A useful first encoding model should recover repeatable rises, falls and delayed response shapes beyond a constant within-trial mean. The reported within-trial variance is the descriptive constant-mean residual; no predictive improvement is claimed. Isolated spikes and inconsistent effects should remain unpredictable expectations.', '',
              '## Limitations', '']
    lines += [f'- {x}' for x in result['limitations']]
    lines += ['', '## Artifacts and inspection', '',
              '`report.json` separates observed coverage/distributions, derived associations and descriptive maxima. `associations.npz` and `neuron-associations.npz` retain trial-level Pearson/Spearman, shifted reference and counts; undefined constant/short series are NaN. `events.json` stores per-trial event curves and uncertainty. `movement-bins.json` stores counts and paired stationary effects. `observations/` stores every analyzed trial with original counts, identities via `units.csv`, motion, stage, contrasts and physical time. Inspect any trial/region from these files; representative plots use uniform trial selection.', '',
              'Numeric response windows are forward means beginning at t+lag; movement history ends at t. All lag/window comparisons use the same within-trial support. Rates are counts ×60 Hz per neuron. Only trial plots smooth (three-bin trailing mean). No overlapping windows are treated as independent replicates. Event curves average episodes within trial before averaging trials; episodes of the same type are separated by more than the full event-window width. Small/large thresholds use stimulus-only contrast-specific medians.']
    (out/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-config', type=Path, default=ROOT/'src/configs/run.json')
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('motion-neural.json'))
    parser.add_argument('--visual-dir', type=Path, default=visual_dir())
    parser.add_argument('--output-dir', type=Path, default=AUDIT_ROOT/'motion-neural')
    args = parser.parse_args(); cfg = {**DEFAULTS, **read(args.config)}
    if (set(cfg) != set(DEFAULTS) or any(type(cfg[k]) is not int or cfg[k] < 1 for k in ['min_pairs', 'min_trials', 'bootstrap', 'event_radius', 'representative_trials', 'population_components'])
            or type(cfg['max_lag']) is not int or cfg['max_lag'] < 0 or cfg['max_lag'] > 12
            or cfg['windows'] != [1, 3, 6, 12] or not np.isfinite(cfg['stationary_atol_deg']) or cfg['stationary_atol_deg'] < 0):
        parser.error('Invalid audit configuration (60 Hz windows 1/3/6/12 and maximum 12-bin lag required)')
    run = read(args.run_config); eid = run.get('eid')
    if not isinstance(eid, str) or not eid:
        parser.error('Run configuration must supply one explicit EID')
    out = args.output_dir; out.mkdir(parents=True, exist_ok=False)
    write(out/'config.json', dict(parameters=cfg, run_config=str(args.run_config.resolve()), run=run,
                                visual_dir=str(args.visual_dir.resolve()), command=sys.argv))
    print('Loading and verifying prepared dataset', flush=True)
    path = Path(run['dataset_generation']); path = path if path.is_absolute() else ROOT/path
    generation = load_dataset(path, expected_generation_id=run.get('expected_dataset_generation_id'))
    samples = sorted([s for split in ['train', 'val', 'test'] for s in getattr(generation.dataset, split) if s.session_id == eid], key=lambda s: s.trial_id)
    if not samples: raise ValueError('Active EID has no prepared samples')
    print('Reading hash-verified physical stimulus metadata', flush=True)
    source = args.visual_dir/f'{eid}_visual_clip.npz'; records, manifest = metadata_records(source, eid, samples)
    trials = [prepare(s, records[s.trial_id], cfg) for s in samples]; units = samples[0].neuron_identity
    region_labels = units['acronym'].fillna('unknown').astype(str).to_numpy()
    regions = sorted(set(region_labels)); rng = np.random.default_rng(cfg['seed'])
    units.to_csv(out/'units.csv', index=False); (out/'observations').mkdir()
    # Descriptive population PCA: trial-centered rates, raw-count covariance, no model fitting.
    covariance = sum((tr['y']-tr['y'].mean(0)).T @ (tr['y']-tr['y'].mean(0)) for tr in trials)
    eigenvalues, vectors = np.linalg.eigh(covariance)
    k = min(cfg['population_components'], len(units)); vectors = vectors[:, -k:][:, ::-1]
    # Deterministic signs: largest absolute loading positive.
    vectors *= np.sign(vectors[np.argmax(abs(vectors), axis=0), np.arange(k)])
    targets = regions + [f'population_PC{i+1}' for i in range(k)]
    for tr in trials:
        tr['targets'] = np.column_stack([tr['y'][:, region_labels == r].mean(1)*60 for r in regions] + [((tr['y']-tr['y'].mean(0)) @ vectors[:, i])*60 for i in range(k)])
        s = tr['sample']
        np.savez_compressed(out/'observations'/f'{s.trial_id}.npz', neural_counts=s.neural[:s.sequence_length, :s.neuron_count],
                            physical_time=s.physical_timestamps[:s.sequence_length], trial_relative_time=tr['time'],
                            stimulus_generator_contrast=tr['contrast'], azimuth_deg=tr['az'], velocity_deg_s=tr['d']*60,
                            stage=tr['stage'], visible=tr['visible'], **tr['features'])
    coverage = {r: int((region_labels == r).sum()) for r in regions}
    input_audit = dict(eid=eid, trials=len(samples), neurons=len(units), region_coverage=coverage,
                       trial_ids=[s.trial_id for s in samples], splits={s: sum(t['sample'].split == s for t in trials) for s in ['train', 'val', 'test']},
                       dataset_generation_id=generation.generation_id, dataset_path=str(path.resolve()),
                       feature_generation_id=manifest['generation_id'], feature_path=str(source.resolve()),
                       metadata_sha256=manifest['records_sha256'], cadence_hz=60,
                       timing='Reconstructed source azimuth linearly interpolated at existing neural bin centers; not measured display refresh timing',
                       verification='Public dataset loader verifies payloads; feature manifest/definition and complete records.jsonl verified. CLIP vectors and replay RGB payloads not read.',
                       limitations='Coverage limited to trials and selected recorded units in the configured prepared dataset; no inference about omitted trials or unrecorded regions')
    write(out/'input-audit.json', input_audit)
    rows = associations(trials, targets, cfg, out, rng)
    neuron_rows = associations(trials, [f'unit_{i}' for i in range(len(units))], cfg, out, rng, neuron=True)
    write(out/'neurons.json', dict(ordered_identity_file='units.csv', rate_hz=distribution(np.concatenate([t['y'] for t in trials]).mean(0)*60), associations=neuron_rows))
    thresholds, binned, events = events_and_bins(trials, regions, cfg, out, rng)
    contrasts = {}
    for contrast in sorted({t['contrast'] for t in trials}):
        group = [t for t in trials if t['contrast'] == contrast]
        contrasts[str(contrast)] = dict(trials=len(group), bins=sum(len(t['y']) for t in group),
                                      displacement_deg=distribution(np.concatenate([abs(t['d']) for t in group])),
                                      neural_rate_hz_per_neuron=distribution(np.concatenate([t['y'].mean(1)*60 for t in group])),
                                      stationary_bins=sum(int(np.sum(abs(t['d']) <= cfg['stationary_atol_deg'])) for t in group),
                                      moving_bins=sum(int(np.sum(abs(t['d']) > cfg['stationary_atol_deg'])) for t in group))
    result = {**input_audit, 'status': 'complete', 'config': cfg, 'contrast_observations': contrasts,
              'movement_thresholds': thresholds, 'associations': rows,
              'population': dict(method='Trial-centered raw-count covariance PCA; descriptive all-trial basis, not predictive validation',
                                 explained_variance_fraction=eigenvalues[-k:][::-1]/eigenvalues.sum(), loadings=vectors),
              'neural_variability': {r: distribution([np.var(t['targets'][:, i]) for t in trials]) for i, r in enumerate(regions)},
              'limitations': [input_audit['limitations'],
                  'Closed-loop wheel motion drives Gabor displacement. Visual, motor and state contributions are confounded; causal separation is outside scope. Zero contrast is descriptive context, not a complete causal control.',
                  'Physical trajectories and display times are reconstructed. Original contrast is preserved; no CLIP gain/preprocessing, rendering, neural regeneration, model changes or retraining.',
                  'Saved RGB hashes are retained source metadata, not reverified pixel differences. Positive contrast and visible state do not prove that every reconstructed displacement changes rendered pixels.',
                  'Within-trial autocorrelation, task stage and elapsed time can explain associations. Half-trial circular shifts are a descriptive phase reference, not calibrated null p-values.',
                  'Trial bootstrap uncertainty uses independent trials, with equal trial weights; overlapping bins/events are not independent observations. Short/constant series are unavailable, not zero effects.',
                  'Descriptive maxima are selected across many dependent variables/lags/windows; intervals do not adjust for selection. No definitive biological latency, detectability threshold or predictive benefit is claimed.',
                  'All prepared splits are explored. Future modeling recommendations require new held-out evaluation; these observations are not unbiased test evidence.',
                  'Region means can cancel opposing neurons; raw-count population PCA can emphasize high-variance neurons. Inspect individual-unit results as well.',
                  'Same-trial support excludes initial history bins and final lag/response bins. Longer windows average counts into rates; they do not add independent observations.']}
    figures(trials, regions, rows, binned, events, cfg, out)
    report_text(result, out); write(out/'report.json', result)
    print(f'Completed motion/neural audit: {out.resolve()}', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
