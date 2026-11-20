"""E1b: fixed linear diagnostics using frozen E1 labels; no test payload reads."""
import argparse
import hashlib
import json
import sys
import warnings
import zipfile
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
E1_CODE = ROOT / 'script/experiment/visual-information'
sys.path[:0] = [str(ROOT / 'src'), str(E1_CODE)]

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from data import checked_manifest, eligible, require
from probes import evaluate, trial_weights, weighted_median
from utils.provenance import file_hash

TARGETS = ['visible_motion', 'direction', 'signed_displacement', 'absolute_displacement']
REGRESSION = TARGETS[2:]
MIN_TRIALS = 3


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def plain(x):
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [plain(v) for v in x]
    if isinstance(x, np.generic):
        return plain(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def write(path, x):
    path.write_text(json.dumps(plain(x), indent=2, allow_nan=False), encoding='utf-8')


def load(e1, out):
    """Select only retained train/validation numeric payloads and frozen labels."""
    m = read(e1 / 'manifest.json')
    dataset = checked_manifest(ROOT / m['paths']['dataset'], m['dataset_generation'])
    alignment = checked_manifest(ROOT / m['paths']['alignment'], m['alignment_generation'])
    selected = {e['trial_id']: (i, e) for i, e in enumerate(dataset['samples']) if e['split'] in ('train', 'val')}
    require(Counter(e['split'] for _, e in selected.values()) == {'train': 218, 'val': 31}, 'Frozen split mismatch')
    for split in ('train', 'val'):
        require({t for t, (_, e) in selected.items() if e['split'] == split} ==
                {e['trial_id'] for e in m['memberships'][split]}, 'E1 membership mismatch')
    require(int(np.floor(np.median([e['sequence_length'] for _, e in selected.values() if e['split'] == 'train']))) == 146,
            'Training median changed')
    provenance = {k: m[k] for k in ('dataset_generation', 'alignment_generation', 'feature_generation',
                 'replay_generation', 'session', 'trial_table_fingerprint', 'paths', 'feature_definition')}
    provenance['memberships'] = {s: m['memberships'][s] for s in ('train', 'val')}
    provenance['e1_files'] = {p.name: file_hash(p) for p in [e1 / n for n in
        ('manifest.json', 'integrity.json', 'support.json', 'diagnostics.json', 'metrics.json', 'anchors-source.jsonl', 'anchors-aligned.jsonl')]}
    archive_path = ROOT / m['paths']['features']
    archive_hash = file_hash(archive_path)
    require(archive_hash == m['verification']['feature_archive_sha256'], 'Frozen feature archive changed')
    provenance['feature_archive_sha256'] = archive_hash
    source, source_times, selected_hashes = {}, {}, {}
    # Skip unselected JSON lines using manifest counts and seek over their
    # numeric byte ranges. Never decode a test record or load a test vector.
    with zipfile.ZipFile(archive_path) as z:
        fm = json.loads(z.read('manifest.json'))
        require(fm['generation_id'] == m['feature_generation'], 'Feature generation mismatch')
        with z.open('records.jsonl') as records, z.open('features.npy') as vectors:
            magic = np.lib.format.read_magic(vectors)
            shape, order, dtype = np.lib.format._read_array_header(vectors, magic)
            require(shape == (fm['feature_count'], 768) and not order and dtype == np.dtype('<f4'), 'Feature schema')
            start, offset = vectors.tell(), 0
            for outcome, ft in zip(fm['replay_completion']['trials'], fm['trials']):
                tid = outcome['trial_id']
                if tid not in selected:
                    for _ in range(outcome['emitted_count'] + 1):
                        require(bool(records.readline()), 'Truncated records')
                    offset += ft['feature_count']
                    continue
                vectors.seek(start + offset * 768 * 4)
                array = np.full((outcome['emitted_count'], 768), np.nan, dtype=np.float32)
                times = np.empty(outcome['emitted_count'])
                for j in range(outcome['emitted_count']):
                    r = json.loads(records.readline())
                    require(r['metadata']['trial_id'] == tid and r['metadata']['schedule_index'] == j, 'Source identity')
                    times[j] = r['metadata']['session_time']
                    if r['status'] == 'encoded':
                        require(r['feature_index'] == offset, 'Source index')
                        array[j] = np.frombuffer(vectors.read(768 * 4), dtype='<f4')
                        offset += 1
                require(json.loads(records.readline())['metadata'] == outcome, 'Source outcome')
                source[tid] = array
                source_times[tid] = times
    aligned = {}
    align_entries = {e['trial_id']: i for i, e in enumerate(alignment['trials'])}
    for tid, (i, e) in selected.items():
        name = f'samples/{i:06d}.npz'
        path = ROOT / m['paths']['dataset'] / name
        require(file_hash(path) == dataset['files'][name], 'Dataset payload hash')
        aname = f'trials/{align_entries[tid]:06d}.npz'
        apath = ROOT / m['paths']['alignment'] / aname
        require(file_hash(apath) == alignment['files'][aname], 'Alignment payload hash')
        selected_hashes[str(tid)] = {'dataset': dataset['files'][name], 'alignment': alignment['files'][aname]}
        with np.load(path, allow_pickle=False) as a, np.load(apath, allow_pickle=False) as b:
            n = e['sequence_length']
            require(a['temporal_mask'][:n].all() and not a['temporal_mask'][n:].any(), 'Validity changed')
            require(np.array_equal(a['visual'][:n], b['visual_features']), 'Alignment values changed')
            aligned[tid] = (a['visual'][:n].copy(), a['physical_timestamps'][:n].copy())
    provenance['selected_payload_sha256'] = selected_hashes
    views = {}
    for stage, arrays in [('source', source), ('aligned', aligned)]:
        rows, static, delta = [], [], []
        with (e1 / f'anchors-{stage}.jsonl').open(encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                tid, k = r['trial_id'], r['anchor']
                require(tid in selected and r['split'] == selected[tid][1]['split'], 'Frozen label membership')
                e = selected[tid][1]
                if stage == 'aligned':
                    v, times = arrays[tid]
                    require(abs(times[k] - r['time']) < 1e-9, 'Aligned label time')
                    primary = 11 <= k < min(e['sequence_length'], 146)
                else:
                    v = arrays[tid]
                    require(abs(source_times[tid][k] - r['time']) < 1e-9, 'Source label time')
                    # E0 counts complete bins on the grid starting at aligned_start.
                    # Source anchors use actual physical times in those bin edges,
                    # not an invented equality of source index and center index.
                    lower = e['aligned_start'] + 11 * e['bin_size']
                    upper = e['aligned_start'] + min(e['sequence_length'], 146) * e['bin_size']
                    primary = lower - 1e-9 <= r['time'] < upper - 1e-9
                require(k > 0 and np.isfinite(v[k-1:k+1]).all(), 'Missing pair feature')
                r['primary'] = bool(primary)
                r['relative_time'] = r['time'] - e['aligned_start']
                rows.append(r)
                static.append(v[k])
                delta.append(v[k] - v[k-1])
        views[stage] = {'rows': rows, 'static': np.stack(static), 'difference': np.stack(delta)}
        print(f'Loaded frozen {stage}: {len(rows)} pairs', flush=True)
    provenance['e1_continuity_exclusions'] = read(e1 / 'support.json').get('continuity_exclusions', {})
    provenance['e1_duplicate_images'] = m['duplicate_images']
    provenance['test_access'] = 'No test observations decoded or numeric test payloads loaded; generation metadata and compressed-byte traversal only.'
    provenance['inherited_integrity'] = read(e1 / 'integrity.json')
    provenance['protocol'] = {
        'primary': 'aligned 11 <= t < min(T,146); source physical anchor time in [aligned_start+11/60, aligned_start+min(T,146)/60). Aligned centers are start+(t+0.5)/60.',
        'full': 'All frozen qualified E1 pairs; supplementary. Independently fitted train-only probes for each interval.',
        'matching': 'floor(position/10) crossed with E1 contrast class; >=3 distinct trials/class/split AND >=6 union trials; common supported strata across train and val. No widening.',
        'weights': 'Equal total per training trial; within trial equal stratum/class cell totals for matched fits. Primary evaluation equal trial totals. Additional matched-balanced evaluation gives each stratum/class cell equal total and each trial within cell equal total. Global trial and cell margins need not both be equal.',
        'numerics': 'Weighted train-only StandardScaler; Ridge alpha=1 cholesky; logistic C=1 lbfgs tol=1e-4. 20000 iterations, then warm-start 80000 only on training ConvergenceWarning. Unconverged fits not scored. No validation tuning.',
        'variants': {'A': 'V[t]', 'B': 'V[t]-V[t-1]', 'C': '[A,B]', 'D': 'abs(B)', 'E': 'L2(B) scalar'},
        'magnitude_only': 'D/E fit motion presence and absolute displacement only; excluded from signed targets.',
        'refined_support': 'Direction: add initial side and training-derived speed tertile; presence: add initial side. Phase: add fixed quadrants. Supplemental evaluations of the same matched models. Signed matched regression uses visible opposite directions; absolute regression uses moving/stationary.',
        'min_trials_per_class': MIN_TRIALS, 'median_bins': 146,
        'packages': {p: version(p) for p in ('numpy', 'scipy', 'scikit-learn')},
    }
    return views, provenance


def confounds(views, e1):
    inherited = read(e1/'diagnostics.json')
    result = {'inherited_e1': inherited, 'intervals': {}}
    for stage,view in views.items():
        rows = view['rows']
        for interval in ('primary','full'):
            for split in ('train','val'):
                selected = [r for r in rows if r['split']==split and (interval=='full' or r['primary'])]
                motion = [r for r in selected if eligible(r,'direction')]
                trial = np.array([r['trial_id'] for r in motion])
                w = trial_weights(trial) if motion else np.array([])
                correlations = {}
                for field in ('position','initial_side','phase','absolute_displacement'):
                    if not motion:
                        correlations[field] = None
                        continue
                    x=np.array([r[field] for r in motion],dtype=float)
                    y=np.array([r['direction'] for r in motion],dtype=float)
                    x-=np.average(x,weights=w)
                    y-=np.average(y,weights=w)
                    den=np.sqrt(np.average(x*x,weights=w)*np.average(y*y,weights=w))
                    correlations[field]=float(np.average(x*y,weights=w)/den) if den>1e-15 else None
                group={}
                for field,fn in [('initial_side',lambda r:r['initial_side']),
                    ('phase_quadrant',lambda r:int((r['phase']%(2*np.pi))//(np.pi/2))),
                    ('position_bin10',lambda r:int(np.floor(r['position']/10))),
                    ('stage',lambda r:r['stage'])]:
                    cells=defaultdict(list)
                    for r in motion:
                        cells[str(fn(r))].append(r)
                    group[field]={k:{str(c):{'observations':sum(r['direction']==c for r in rr),
                        'trials':len({r['trial_id'] for r in rr if r['direction']==c})} for c in (0,1)} for k,rr in cells.items()}
                category={}
                for c in sorted({r['category'] for r in selected}):
                    rr=[r for r in selected if r['category']==c]
                    speeds=[r['absolute_displacement']/r['dt'] for r in rr]
                    category[c]={'observations':len(rr),'trials':len({r['trial_id'] for r in rr}),
                        'speed_deg_per_second_quantiles':np.quantile(speeds,[0,.25,.5,.75,1]).tolist(),
                        'unique_trial_phases':len({(r['trial_id'],r['phase']) for r in rr})}
                longest=0
                previous=None
                run=0
                for r in selected:
                    if r['category']=='visible_stationary':
                        run=run+1 if previous==(r['trial_id'],r['anchor']-1) else 1
                        longest=max(longest,run)
                    else:
                        run=0
                    previous=(r['trial_id'],r['anchor'])
                result['intervals'][f'{stage}-{interval}-{split}']={'motion_direction_correlations':correlations,
                    'motion_group_counts':group,'categories':category,'longest_stationary_pair_run':longest}
    return result


def key(r, target, refinement='', speed_edges=()):
    k = [int(np.floor(r['position'] / 10)), r['contrast']]
    if refinement:
        k.append(r['initial_side'])
    if 'speed' in refinement and target == 'direction':
        k.append(int(np.digitize(r['absolute_displacement'] / r['dt'], speed_edges)))
    if 'phase' in refinement:
        k.append(int(np.floor((r['phase'] % (2*np.pi)) / (np.pi/2))))
    return tuple(k)


def class_label(r, target):
    return r['direction'] if target in ('direction','signed_displacement') else r['visible_motion']


def match(rows, mask, target, refinement='', speed_edges=()):
    cells = defaultdict(lambda: {'train': {0: set(), 1: set()}, 'val': {0: set(), 1: set()}})
    counts = defaultdict(Counter)
    for r, yes in zip(rows, mask):
        if yes:
            k, c = key(r, target, refinement, speed_edges), class_label(r, target)
            cells[k][r['split']][c].add(r['trial_id'])
            counts[k][(r['split'], c)] += 1
    kept, table, allocated = set(), {}, {}
    for k, s in sorted(cells.items()):
        passes = {}
        detail = {}
        for split, cc in s.items():
            # Six union contributors permit disjoint groups of >=3 even if
            # individual trials contain both labels. Actual disjoint feasibility
            # follows from each marginal >=3 and union >=6 for two classes.
            passes[split] = min(map(len, cc.values())) >= MIN_TRIALS and len(cc[0] | cc[1]) >= 2*MIN_TRIALS
            detail[split] = {str(c): {'trials': sorted(cc[c]), 'trial_count': len(cc[c]),
                'observations': counts[k][(split, c)]} for c in (0, 1)}
            detail[split]['union_trials'] = len(cc[0] | cc[1])
            detail[split]['passes'] = passes[split]
            groups = {c: set(cc[c] - cc[1-c]) for c in (0,1)}
            for tid in sorted(cc[0] & cc[1]):
                c = min((0,1), key=lambda c: (len(groups[c]), c))
                groups[c].add(tid)
            allocated[(k,split)] = groups
            detail[split]['disjoint_contributors'] = {str(c): sorted(groups[c]) for c in (0,1)}
        if all(passes.values()):
            kept.add(k)
        table[str(k)] = detail
    matched = mask & np.array([key(r, target, refinement, speed_edges) in kept and
        r['trial_id'] in allocated[(key(r,target,refinement,speed_edges),r['split'])][class_label(r,target)] for r in rows])
    return matched, {'strata': table, 'retained_strata': [str(k) for k in sorted(kept)],
        'reason': None if kept else 'Inconclusive: no common stratum has independent class support in both splits.'}


def balanced_weights(rows, indices, target, matched):
    trials = np.array([rows[i]['trial_id'] for i in indices])
    if not matched:
        labels = np.array([rows[i][target] for i in indices]) if target not in REGRESSION else None
        return trial_weights(trials, labels), {'converged': True, 'method': 'E1 equal trial weights'}
    # Equal trial totals are retained; balance cells within each trial, just as
    # E1 balances classes within trial. Exact global trial+cell margins can be
    # incompatible and are not imposed as an artificial eligibility criterion.
    cells = [(*key(rows[i], target), class_label(rows[i], target)) for i in indices]
    _, ti = np.unique(trials, return_inverse=True)
    _, ci = np.unique(np.asarray(cells), axis=0, return_inverse=True)
    w = np.empty(len(indices), dtype=float)
    for t in np.unique(ti):
        present = np.unique(ci[ti==t])
        for c in present:
            keep = (ti==t) & (ci==c)
            w[keep] = 1/(len(present)*keep.sum())
    return w, {'converged': True, 'method': 'Equal trial totals; equal stratum/class cells within trial'}


def standardized_weights(rows, indices, target):
    cells = [(*key(rows[i], target), class_label(rows[i], target)) for i in indices]
    _, ci = np.unique(np.asarray(cells), axis=0, return_inverse=True)
    w = np.empty(len(indices))
    for c in np.unique(ci):
        keep = ci==c
        w[keep] = trial_weights(np.array([rows[i]['trial_id'] for i in indices[keep]]))
        w[keep] /= w[keep].sum()
    return w


def summary(rows, mask, target):
    subset = [r for r, yes in zip(rows, mask) if yes]
    return {'observations': len(subset), 'trials': len({r['trial_id'] for r in subset}),
        'classes': {str(c): {'observations': sum(class_label(r, target) == c for r in subset),
            'trials': len({r['trial_id'] for r in subset if class_label(r, target) == c})} for c in (0, 1)},
        'categories': dict(Counter(r['category'] for r in subset)),
        'per_trial': dict(Counter(str(r['trial_id']) for r in subset))}


def matrix(view, indices, variant):
    if variant == 'A':
        x = view['static'][indices]
    elif variant == 'B':
        x = view['difference'][indices]
    elif variant == 'C':
        x = np.concatenate([view['static'][indices], view['difference'][indices]], axis=1)
    elif variant == 'D':
        x = np.abs(view['difference'][indices])
    else:
        x = np.linalg.norm(view['difference'][indices].astype(float), axis=1)[:, None]
    return x.astype(float)


def score(y, p, trials, target, weights=None):
    s = evaluate(y, p, trials, target, [0, 1])
    if weights is not None:
        from probes import regression, classification
        s['matched_balanced'] = regression(y, p, weights) if target in REGRESSION else classification(y, p, weights, [0, 1])
    return s


def run_probes(stage, view, out):
    rows = view['rows']
    splits = np.array([r['split'] for r in rows])
    trials = np.array([r['trial_id'] for r in rows])
    primary = np.array([r['primary'] for r in rows])
    support, results, pairs = {}, {}, {}
    for interval in ('primary', 'full'):
        interval_mask = primary if interval == 'primary' else np.ones(len(rows), dtype=bool)
        for target in TARGETS:
            name = f'{stage}-{interval}-{target}'
            available = np.array([eligible(r, target) for r in rows])
            mask = available & interval_mask
            matching_mask = mask & np.array([r['category']=='visible_motion' for r in rows]) if target=='signed_displacement' else mask
            matched, strata = match(rows, matching_mask, target)
            motion_train = [r['absolute_displacement']/r['dt'] for r, yes in zip(rows, mask) if yes and r['split']=='train' and r['visible_motion']]
            edges = np.unique(np.quantile(motion_train, [1/3, 2/3])).tolist() if motion_train else []
            refined = {}
            for refinement in ('side_speed', 'side_speed_phase'):
                rm, rs = match(rows, matched, target, refinement, edges)
                refined[refinement] = (rm, rs)
            support[name] = {'strata': strata, 'speed_edges_deg_per_second_train': edges,
                'refined': {k: v[1] for k, v in refined.items()},
                'interval_categories': {s: summary(rows, interval_mask & (splits == s), target) for s in ('train','val')},
                'exclusions': {s: dict(Counter('outside_interval' if not r['primary'] and interval=='primary' else
                    'off_screen_center' if not r['on_screen'] else 'blank_or_unrepresented' if not r['nonblank'] else r['category']
                    for r, yes in zip(rows, mask) if not yes and r['split']==s)) for s in ('train','val')}}
            for mode, use in [('unrestricted', mask), ('matched', matched)]:
                runname = f'{name}-{mode}'
                ti, vi = np.flatnonzero(use & (splits=='train')), np.flatnonzero(use & (splits=='val'))
                coverage = {s: summary(rows, use & (splits==s), target) for s in ('train','val')}
                coverage['shared_anchor_sha256'] = hashlib.sha256(json.dumps([(rows[i]['trial_id'], rows[i]['anchor']) for i in np.flatnonzero(use)]).encode()).hexdigest()
                support[name][mode] = coverage
                # Publish support before any optimization for this comparison.
                write(out / f'support-{stage}.json', support)
                print(f'{runname}: train {len(ti)} val {len(vi)}', flush=True)
                if not len(ti) or not len(vi):
                    results[runname] = {'inconclusive': 'No eligible independent trial support'}
                    continue
                y = np.array([r[target] for r in rows])
                if min(len(np.unique(trials[ti])), len(np.unique(trials[vi]))) < 3 or (target not in REGRESSION and
                        (len(np.unique(y[ti])) < 2 or len(np.unique(y[vi])) < 2)):
                    results[runname] = {'inconclusive': 'Fewer than three trials or missing classification class'}
                    continue
                w, wd = balanced_weights(rows, ti, target, mode=='matched')
                vw, vwd = balanced_weights(rows, vi, target, False)
                if mode=='matched':
                    vw = standardized_weights(rows, vi, target)
                coverage['weighting'] = {'train': wd, 'val': vwd}
                if not wd['converged'] or not vwd['converged']:
                    results[runname] = {'inconclusive': 'Equal-trial and equal-cell margins incompatible or insufficient IPF convergence', 'weighting': coverage['weighting']}
                    continue
                w *= len(w) / w.sum()
                basew = trial_weights(trials[ti])
                if target in REGRESSION:
                    constants = {'training_mean': float(np.average(y[ti], weights=basew)),
                        'training_median': weighted_median(y[ti], basew), 'zero': 0.}
                else:
                    constants = {'training_majority': int(np.argmax([basew[y[ti]==c].sum() for c in (0,1)]))}
                result = {'baselines': {k: score(y[vi], np.full(len(vi), c), trials[vi], target, vw if mode=='matched' else None) for k,c in constants.items()},
                    'constant_values': constants, 'variants': {}, 'anchor_sha256': coverage['shared_anchor_sha256']}
                predictions = {}
                for variant in ('A','B','C','D','E') if target in ('visible_motion','absolute_displacement') else ('A','B','C'):
                    print(f'Fit {runname}/{variant}', flush=True)
                    scaler = StandardScaler()
                    xt = matrix(view, ti, variant)
                    scaler.fit(xt, sample_weight=w)
                    xt = scaler.transform(xt, copy=False)
                    if target in REGRESSION:
                        model = Ridge(alpha=1., solver='cholesky')
                        model.fit(xt, y[ti], sample_weight=w)
                        optimization = {'converged': True, 'solver': 'cholesky'}
                    else:
                        model = LogisticRegression(C=1., solver='lbfgs', max_iter=20000, tol=1e-4, random_state=0, warm_start=True)
                        attempts = []
                        for limit in (20000, 80000):
                            model.set_params(max_iter=limit)
                            with warnings.catch_warnings(record=True) as caught:
                                warnings.simplefilter('always', ConvergenceWarning)
                                model.fit(xt, y[ti], sample_weight=w)
                            bad = [str(c.message) for c in caught if issubclass(c.category, ConvergenceWarning)]
                            attempts.append({'limit': limit, 'iterations': model.n_iter_.tolist(), 'warnings': bad})
                            if not bad:
                                break
                        optimization = {'converged': not bad, 'attempts': attempts}
                    if not optimization['converged']:
                        result['variants'][variant] = {'inconclusive': 'Training optimizer did not converge', 'optimization': optimization}
                        continue
                    p = model.predict(scaler.transform(matrix(view, vi, variant)))
                    predictions[variant] = p
                    scores = score(y[vi], p, trials[vi], target, vw if mode=='matched' else None)
                    scores['optimization'] = optimization
                    scores['refined'] = {}
                    if mode=='matched':
                        for refinement, (rm, rs) in refined.items():
                            vv = rm[vi]
                            scores['refined'][refinement] = score(y[vi][vv],p[vv],trials[vi][vv],target) if vv.any() else {'inconclusive': rs['reason']}
                    if target=='visible_motion':
                        cm = interval_mask & (splits=='val') & np.array([r['category'] in ('identical_nonzero_command','true_blank','visible_stationary') for r in rows])
                        ci = np.flatnonzero(cm)
                        cp = model.predict(scaler.transform(matrix(view, ci, variant))) if len(ci) else np.array([])
                        scores['controls'] = {}
                        for category in ('identical_nonzero_command','true_blank','visible_stationary'):
                            for positive in (False, True) if category=='identical_nonzero_command' else (False,):
                                keep = np.array([rows[i]['category']==category and (not positive or rows[i]['contrast_value']>0) for i in ci])
                                label = category + ('_positive_contrast' if positive else '')
                                scores['controls'][label] = {'observations': int(keep.sum()), 'trials': len(np.unique(trials[ci][keep])),
                                    'false_positives': int(np.sum(cp[keep]==1)),
                                    'trial_balanced_fpr': float(np.average(cp[keep]==1, weights=trial_weights(trials[ci][keep]))) if keep.any() else None,
                                    'pooled_fpr': float(np.mean(cp[keep]==1)) if keep.any() else None}
                    result['variants'][variant] = scores
                    np.savez_compressed(out / 'models' / f'{runname}-{variant}.npz', mean=scaler.mean_, scale=scaler.scale_,
                        coefficients=model.coef_, intercept=model.intercept_, classes=getattr(model,'classes_',np.array([])))
                results[runname] = result
                np.savez_compressed(out / f'predictions-{runname}.npz', trial=trials[vi], anchor=np.array([rows[i]['anchor'] for i in vi]),
                    time=np.array([rows[i]['time'] for i in vi]), target=y[vi], evaluation_weight=vw, **predictions)
                comparisons = {}
                metric_names = ('mae','rmse') if target in REGRESSION else ('balanced_accuracy','macro_f1')
                for a,b in [('A','B'),('A','C'),('A','D'),('A','E'),('B','D'),('B','E')]:
                    if a not in predictions or b not in predictions:
                        continue
                    sa,sb = result['variants'][a],result['variants'][b]
                    common = sa['per_trial'].keys() & sb['per_trial'].keys()
                    comparisons[f'{b}_minus_{a}'] = {'trial_balanced': {k: sb['trial_balanced'][k]-sa['trial_balanced'][k] for k in metric_names},
                        'per_trial': {tid: {k: sb['per_trial'][tid][k]-sa['per_trial'][tid][k] for k in metric_names} for tid in sorted(common)}}
                    comparisons[f'{b}_minus_{a}']['mean_paired_trial_difference'] = {k: float(np.mean([comparisons[f'{b}_minus_{a}']['per_trial'][t][k] for t in common])) for k in metric_names}
                pairs[runname] = {'anchor_sha256': coverage['shared_anchor_sha256'], 'differences': comparisons,
                    'note': 'Descriptive reused-validation comparisons, not independent confirmation. Per-trial BA averages omit absent classes within that trial; aggregate BA is reported separately.'}
                write(out / f'results-{stage}.json', results)
                write(out / f'paired-{stage}.json', pairs)
    write(out / f'support-{stage}.json', support)
    write(out / f'results-{stage}.json', results)
    write(out / f'paired-{stage}.json', pairs)
    return support, results


def report(out, all_support, all_results):
    lines = ['# E1b — Confound-controlled visual motion probe', '',
        'Frozen E1 training/validation follow-up; reused validation is exploratory and supplies no independent confirmation. Test observations excluded.', '',
        '## Coverage before scoring', '',
        '| Stage | Interval | Target | Mode | Train pairs/trials | Validation pairs/trials | Status |',
        '|---|---|---|---|---|---|---|']
    for stage, support in all_support.items():
        for name,s in support.items():
            for mode in ('unrestricted','matched'):
                c=s[mode]
                result=all_results[stage][f'{name}-{mode}']
                lines.append(f"| {stage} | {name.split('-')[1]} | {name.split('-',2)[2]} | {mode} | {c['train']['observations']}/{c['train']['trials']} | {c['val']['observations']}/{c['val']['trials']} | {result.get('inconclusive','scored')} |")
    lines += ['', '## Comparisons', '', 'BA and F1 use equal validation-trial weights; errors are degrees per adjacent physical interval. Matched-balanced scores, pooled scores, confusion matrices and individual trial differences are in JSON.', '',
        '| Comparison | Variant | BA | Macro-F1 | MAE | RMSE |', '|---|---|---|---|---|---|']
    for results in all_results.values():
        for name,r in results.items():
            if 'inconclusive' in r:
                continue
            for variant,s in list(r['baselines'].items())+list(r['variants'].items()):
                if 'trial_balanced' not in s:
                    continue
                m=s['trial_balanced']
                values=[f'{m[k]:.4f}' if k in m else '—' for k in ('balanced_accuracy','macro_f1','mae','rmse')]
                lines.append(f"| {name} | {variant} | {' | '.join(values)} |")
    lines += ['', '## Protocol and limitations', '',
        'A=current feature; B=signed difference; C=concatenation; D=absolute elementwise difference; E=scalar difference norm. D/E are restricted to presence and absolute displacement.', '',
        'The primary aligned interval is 11 <= t < min(T,146). Source anchor times occupy the corresponding physical bin-edge interval [start+11/60,start+min(T,146)/60); source frames and aligned centers differ by half a cadence. A difference may use its preceding observation outside the scored anchor interval, as E0 uses preceding context. Full-trial models are separately trained supplementary diagnostics.', '',
        'Strata are the fixed floor(position/10) and E1 contrast classes. At least three disjoint trial contributors per class are required per split, allocated deterministically before fitting. Training retains equal trial totals with cell balance within trial; matched-balanced evaluation additionally standardizes stratum/class cells and trials within each cell. No stratum is broadened. Refined initial-side/speed/phase support is disclosed separately.', '',
        'Qualified frozen E1 labels exclude unsupported transitions, angular wraps, freeze jumps, gaps, reversals inside aligned brackets and ambiguous pixel supports. Pixel identity with commanded motion is a negative control, never relabeled visible motion. Aligned labels are interpolated source-state estimates with saved right-bracket look-ahead, not center-time rendered ground truth.', '',
        'Synthetic phase can identify appearance/trial, initial side and position can predict direction, and overlapping pairs/repeated stationary states reduce effective independence. Trial splits and weights address some shortcuts; matching does not randomize phase or certify historical retinal motion. Magnitude and speed coincide up to the fixed regular interval; train-only speed tertiles further qualify direction support. See inherited E1 repeated-image exposure and confound summaries in manifest/confounds JSON.', '',
        'Logistic optimization uses the same C=1 and tolerance=1e-4 uniformly, with a training-only 20000/80000 iteration policy. Train-only scaling, fixed ridge alpha=1 and no validation-driven threshold or parameter search apply throughout.', '',
        '## Scientific interpretation', '',
        'Interpret matched support before aggregate accuracy. Where independent matched support is absent, the direction or presence question remains inconclusive regardless of unrestricted performance. Absence of linear recovery does not establish information-theoretic absence or predict neural-model gains. E3 is justified by a supported motion target that remains weak after temporal controls; pixel-identical commanded motion instead indicates stimulus observability limits that feature engineering cannot repair.', '']
    (out/'report.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--e1', type=Path, default=ROOT/'output/experiment/visual-information/e1')
    parser.add_argument('--output', type=Path, default=ROOT/'output/experiment/visual-information/e1b')
    parser.add_argument('--threads', type=int, default=4)
    args=parser.parse_args()
    require(args.threads>0,'Positive thread count required')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'models').mkdir()
    (args.output/'implementation').mkdir()
    with threadpool_limits(limits=args.threads):
        views, manifest=load(args.e1,args.output)
        manifest['protocol']['threads']=args.threads
        manifest['sources']={str(p.relative_to(ROOT)):file_hash(p) for p in [Path(__file__), E1_CODE/'data.py', E1_CODE/'probes.py']}
        for path in [Path(__file__), E1_CODE/'data.py', E1_CODE/'probes.py']:
            (args.output/'implementation'/path.name).write_bytes(path.read_bytes())
        write(args.output/'manifest.json',manifest)
        write(args.output/'confounds.json',confounds(views,args.e1))
        all_support,all_results={},{}
        for stage,view in views.items():
            all_support[stage],all_results[stage]=run_probes(stage,view,args.output)
        report(args.output,all_support,all_results)
        write(args.output/'completion.json',{'complete':True,'test_observations_accessed':False,
            'artifacts':{p.name:file_hash(p) for p in args.output.iterdir() if p.is_file()}})
    print(f'Completed {args.output}',flush=True)


if __name__=='__main__':
    main()
