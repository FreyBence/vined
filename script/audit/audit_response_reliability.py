"""Scenario 11 condition/movement reliability audit (no fitting or test access).

Run with the project Python from any directory. Publishes the next unused
response-reliability-NNN generation atomically; existing generations are immutable.
The metadata-only matching policy is frozen before neural arrays are opened.
"""
import hashlib
import html
import itertools
import json
import math
from pathlib import Path
import platform
import re
import uuid

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from audit_representation_response import ROOT, BASE, EID, read, digest, clean, corr, dist

SEED = 20261103
WIDTHS = (1, 3, 6, 12)
EXPECTED = "01e5d3830786d45e1aa6f074d3e6c08fb0e476165f93b11a94d605edefb7995c"


def column_corr(a, b):
    a = a - a.mean(0)
    b = b - b.mean(0)
    den = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0)
    return np.divide((a*b).sum(0), den, out=np.full(a.shape[1], np.nan), where=den > 1e-12)


def trailing(y, width, endpoints):
    """Complete windows, fixed physical endpoints for every diagnostic width."""
    cs = np.concatenate([np.zeros((1, y.shape[1])), np.cumsum(y, axis=0)])
    return (cs[endpoints+1] - cs[endpoints+1-width]) / width


def partitions(n, key):
    half = n//2
    possible = math.comb(n, half) // (2 if n % 2 == 0 else 1)
    if possible <= 200:
        selected = [c for c in itertools.combinations(range(n), half)
                    if n % 2 or 0 in c]
    else:
        seed = int.from_bytes(hashlib.sha256(f"{SEED}:{key}".encode()).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        seen = set()
        while len(seen) < 200:
            c = tuple(sorted(rng.choice(n, half, replace=False)))
            if n % 2 == 0 and 0 not in c:
                c = tuple(i for i in range(n) if i not in c)
            seen.add(c)
        selected = sorted(seen)
    return [(list(c), [i for i in range(n) if i not in c]) for c in selected], possible


def bootstrap(values, key):
    values = np.asarray(values)
    if not len(values):
        return dict(n=0, mean=None, ci95=None)
    if len(values)<3:
        return dict(n=len(values),mean=values.mean(),ci95=None,uncertainty_status='fewer_than_3_trials')
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"{SEED}:{key}".encode()).digest()[:8], "big"))
    means = values[rng.integers(len(values), size=(1000, len(values)))].mean(1)
    return dict(n=len(values), mean=values.mean(), ci95=np.quantile(means, [.025, .975]))


def analyze(group, key, endpoints, reference=None, event_indices=None):
    n = len(group)
    out = dict(n=n, trial_ids=[x['trial_id'] for x in group], widths={}, status="split_half" if n >= 6 else "descriptive_only" if n else "unavailable")
    if not n:
        return out
    stacks = np.stack([x['y'] if event_indices is None else
                       x['y'][event_indices[x['trial_id']]-12:event_indices[x['trial_id']]+24]
                       for x in group]) if event_indices is not None else None
    pairs, possible = partitions(n, key) if n >= 6 else ([], 0)
    out.update(possible_partitions=possible, unique_partitions=len(pairs), half_sizes=[n//2, n-n//2])
    for width in WIDTHS:
        st = np.stack([trailing(x['y'] if stacks is None else stacks[i], width, endpoints)
                       for i, x in enumerate(group)])
        regional = st.sum(2)
        rs, nr, residual, shifted = [], [], [], []
        ref = trailing(reference, width, endpoints).sum(1) if reference is not None else None
        for left, right in pairs:
            a, b = st[left].mean(0), st[right].mean(0)
            rs.append(corr(a.sum(1), b.sum(1)))
            nr.append(column_corr(a, b))
            shifted.append(corr(a.sum(1), np.roll(b.sum(1), len(b)//2)))
            if ref is not None:
                residual.append(corr(a.sum(1)-ref, b.sum(1)-ref))
        nr = np.asarray(nr) if nr else np.empty((0, 99))
        out['widths'][width] = dict(
            mean=regional.mean(0), trial_sd=regional.std(0, ddof=1) if n>1 else None,
            neuron_mean=st.mean(0), regional_split_half=dist(rs), correlations=rs,
            shifted_half_reference=dist(shifted), elapsed_reference_removed=dist(residual),
            neuron_split_half=[dist(nr[:, j]) for j in range(99)],
            individual_trial_to_other_mean=dist([corr(regional[i], (regional.sum(0)-regional[i])/(n-1)) for i in range(n)]) if n>1 else None)
    return out


def table_md(frame):
    # Avoid an optional tabulate dependency in the project runtime.
    cells = frame.fillna('NA').astype(str).map(lambda s:s.replace('|','\\|'))
    return '| ' + ' | '.join(cells.columns) + ' |\n|' + '|'.join(['---']*len(cells.columns)) + '|\n' + '\n'.join('| '+' | '.join(row)+' |' for row in cells.itertuples(index=False, name=None))


def main():
    prev = BASE/'representation-response-001'
    frozen = read(BASE/'temporal-001/results.json')
    previous = read(prev/'results.json')
    if digest(Path(frozen['checkpoint']['path'])) != frozen['checkpoint']['sha256']:
        raise ValueError('Current checkpoint differs from frozen diagnostic')
    dp = Path(frozen['dataset_path'])
    dm = read(dp/'manifest.json')
    if dm['generation_id'] != EXPECTED or frozen['dataset_generation_id'] != EXPECTED or previous['dataset_generation_id'] != EXPECTED:
        raise ValueError('Unexpected dataset generation')
    rp = ROOT/f'output/visual_replays/{EID}'
    rd = read(rp/'definition.json')['inputs']
    rm = read(rp/'manifest.json')
    source_paths = {s['name']: Path(s['path']) for s in rd['source_datasets']}
    wt = np.load(source_paths['_ibl_wheel.timestamps.npy'], allow_pickle=False)
    wp = np.load(source_paths['_ibl_wheel.position.npy'], allow_pickle=False)
    for array, field in [(wt, 'timestamps_sha256'), (wp, 'positions_sha256')]:
        if hashlib.sha256(array.astype('<f8').tobytes()).hexdigest() != rd['wheel'][field]:
            raise ValueError('Wheel differs from replay input')
    keep = np.r_[np.diff(wt) != 0, True]
    duplicates = len(wt)-int(keep.sum())
    wt, wp = wt[keep], wp[keep]
    if not np.isfinite(wp).all() or not np.isfinite(wt).all() or not (np.diff(wt)>0).all():
        raise ValueError('Invalid wheel')
    trial_table = pd.read_parquet(source_paths['_ibl_trials.table.pqt'])
    off = np.load(source_paths['_ibl_trials.stimOff_times.npy'], allow_pickle=False)
    units = frozen['regional_targets'][EID]['ordered_units']
    if units != previous['ordered_units'] or len(units) != 99:
        raise ValueError('Ordered regional identities differ')
    unit_table = pd.read_parquet(dp/f'units/{EID}.parquet')
    cols = []
    for u in units:
        hit = np.flatnonzero(np.logical_and.reduce([unit_table[k].eq(u[k]).to_numpy() for k in ['source_unit_id','pid','probe','collection','revision']]))
        if len(hit)!=1:
            raise ValueError('Ambiguous source identity')
        cols.append(int(hit[0]))
    if cols != previous['source_columns_in_persisted_280']:
        raise ValueError('Source columns differ')
    inputs = {str(dp/'manifest.json'): digest(dp/'manifest.json'), str(rp/'definition.json'):digest(rp/'definition.json'),
              str(rp/'manifest.json'):digest(rp/'manifest.json'), str(dp/f'units/{EID}.parquet'):digest(dp/f'units/{EID}.parquet')}
    for p in source_paths.values():
        inputs[str(p)] = digest(p)
    for p in [prev/'results.json', prev/'response-profiles.json', prev/'lag-comparison.json', prev/'trials.csv', prev/'neurons.csv', BASE/'temporal-001/results.json']:
        inputs[str(p)] = digest(p)
    # Only permitted memberships enter metadata calculations; test payloads are never opened.
    data = []
    records = {r['trial_id']: r for r in rd['trials']}
    for index, sample in enumerate(dm['samples']):
        if sample['split'] not in ('train', 'val'):
            continue
        tid = sample['trial_id']
        r = records[tid]
        tab = trial_table.iloc[tid]
        if not np.isclose(tab.stimOn_times, sample['stim_on'], atol=1e-9, rtol=0) or not np.isclose(off[tid], sample['stim_off'], atol=1e-9, rtol=0):
            raise ValueError('Recorded stimulus interval mismatch')
        if r != sample['metadata']['alignment']['visual_definition']['trial']:
            raise ValueError('Replay trial definition differs from dataset')
        onset = sample['stim_on']
        obs_path = rp/f'trials/{tid}/observations.jsonl'
        obs = [json.loads(line) for line in obs_path.read_text().splitlines()]
        inputs[str(obs_path)] = digest(obs_path)
        times = np.array([o['session_time'] for o in obs])
        az = np.array([o['stimulus_state']['azimuth_deg'] for o in obs])
        az_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(az)))
        stages = np.array([o['stimulus_state']['stage'] for o in obs])
        n = sample['sequence_length']
        support = onset + (np.arange(n)+.5)/60
        edges = onset+np.arange(n+1)/60
        if edges[0] < wt[0] or edges[-1]>wt[-1]:
            raise ValueError('Wheel does not cover aligned support')
        wheel = np.interp(edges, wt, wp)
        path_length = np.abs(np.diff(wheel))
        if n>=60:
            window = (wt>onset)&(wt<onset+1)
            raw = np.r_[np.interp(onset,wt,wp), wp[window], np.interp(onset+1,wt,wp)]
            movement = np.abs(np.diff(raw)).sum()
            gap_indices = np.arange(max(0,np.searchsorted(wt,onset,side='right')-1), min(len(wt)-1,np.searchsorted(wt,onset+1,side='left')))
            max_gap = np.diff(wt)[gap_indices].max()
        else:
            movement, max_gap = np.nan, np.nan
        moves = np.flatnonzero(np.abs(np.diff(az_unwrapped))>1e-6)+1
        row = dict(trial_id=tid, split=sample['split'], condition=f"{r['side']}:{r['parameters']['contrast']:g}",
                   side=r['side'], contrast=r['parameters']['contrast'], bins=n, duration=n/60,
                   recorded_trial_duration=tab.intervals_1-tab.intervals_0,
                   stim_on=onset, stim_off=sample['stim_off'], response=tab.response_times-onset,
                   go_cue=tab.goCue_times-onset, feedback=tab.feedback_times-onset,
                   first_movement=tab.firstMovement_times-onset, choice=tab.choice, feedback_type=tab.feedbackType,
                   closed_loop_proxy=r['events']['closed_loop']-onset,
                   freeze_proxy=(r['events'].get('freeze')-onset) if r['events'].get('freeze') is not None else np.nan,
                   wheel_path_first_second_rad=movement, wheel_max_sample_gap_first_second_s=max_gap,
                   wheel_path_full_aligned_rad=path_length.sum(), wheel_mean_abs_binned_velocity_rad_s=path_length.mean()*60,
                   visual_path_first_second_deg=np.abs(np.diff(az_unwrapped[times<onset+1])).sum(),
                   visual_movement_onset=(times[moves[0]]-onset) if len(moves) else np.nan,
                   frozen_fraction_full=float(np.mean(np.char.startswith(stages,'freeze'))),
                   frozen_fraction_first_second=float(np.mean(np.char.startswith(stages[times<onset+1],'freeze'))),
                   source_stages=';'.join(sorted(set(stages))),
                   eligible_first_second=n>=60)
        data.append(dict(**row, sample=sample, sample_index=index, times=support, wheel_path=path_length,
                         obs_times=times, az=az_unwrapped, stages=stages, row=row))
    train = [x for x in data if x['split']=='train']
    cutoff = float(np.median([x['wheel_path_first_second_rad'] for x in train if x['bins']>=60]))
    duration_cut = float(np.median([x['duration'] for x in train]))
    policy = dict(seed=SEED, minimum_split_half_trials=6, minimum_cross_split_trials=3,
                  support_rule='>=6 gives >=3 distinct whole trials per half; 1..5 descriptive only; cross-split >=3 each; no precision guarantee',
                  wheel_cut_rad=cutoff, duration_cut_s=duration_cut,
                  grouping='side:contrast | first-second raw-wheel total variation <= training median versus > median | response <=1s versus later/missing',
                  phase='response<=1s is a coarse phase stratum, not exact response-time or freeze matching',
                  support='>=60 observed bins; scoring endpoints 22..59, widths 1/3/6/12 use only bins >=11; common endpoints across widths',
                  onset='separate source-only endpoints 11..23, trailing windows may include bins 0..10; raw profiles 0..59 also saved',
                  events='nearest recorded neural center; -12..23 bins; no extension beyond stimulus data; smoothing endpoints -1..23',
                  partitions='all unique unlabeled equal halves if <=200, otherwise 200 unique fixed-seed partitions; odd n uses labeled smaller half',
                  uncertainty='1000 trial-bootstrap means for scalar changes/contrasts; partition quantiles are not confidence intervals',
                  weighting='equal-trial group profiles; region sums original 99 units; neuron summaries equal-neuron; correlations over common endpoints, no binwise inference',
                  null='half-profile circular shift by half its endpoint length is a phase reference, not a significance test',
                  elapsed_reference='training-only leave-condition-out mean of other side/contrast groups; fixed for both halves and validation; descriptive sensitivity, not fitted model',
                  matching_not_optimized='policy and both training-metadata medians fixed before reading any neural payload')
    for x in data:
        wheel_class = 'low' if x['wheel_path_first_second_rad']<=cutoff else 'high'
        phase = 'responded' if np.isfinite(x['response']) and x['response']<=1 else 'ongoing'
        x['movement_class'], x['phase'] = wheel_class, phase
        x['group'] = f"{x['condition']}|{wheel_class}|{phase}"
        x['duration_class'] = 'short' if x['duration']<=duration_cut else 'long'
        x['row'].update(group=x['group'], movement_class=wheel_class, phase=phase, duration_class=x['duration_class'])
    # Definition is complete above. Neural outcomes enter only below this line.
    for x in data:
        s, index = x['sample'], x['sample_index']
        path = dp/f'samples/{index:06d}.npz'
        if digest(path)!=dm['files'][f'samples/{index:06d}.npz']:
            raise ValueError('Dataset payload hash mismatch')
        inputs[str(path)] = digest(path)
        with np.load(path, allow_pickle=False) as z:
            n=s['sequence_length']
            if not z['temporal_mask'][:n].all() or z['temporal_mask'][n:].any() or not z['neuron_mask'][cols].all():
                raise ValueError('Invalid selected support')
            if not np.allclose(z['physical_timestamps'][:n],x['times'],atol=1e-9,rtol=0) or not np.array_equal(z['temporal_positions'][:n],np.arange(n)):
                raise ValueError('Physical/model timestamps differ')
            x['y']=z['neural'][:n,cols].astype(float)
            x['feature_delta']=np.r_[0,np.linalg.norm(np.diff(z['visual'][:n].astype(float),axis=0),axis=1)]
        fr=next(r for r in frozen['records'] if r['trial_id']==x['trial_id'] and r['split']==x['split'])
        predpath=BASE/'temporal-001/predictions'/f"{fr['array_key']}.npz"
        pkey=str(predpath.relative_to(BASE/'temporal-001')).replace('\\','/')
        hashes={k.replace('\\','/'):v for k,v in frozen['payload_hashes'].items()}
        if digest(predpath)!=hashes[pkey]:
            raise ValueError('Frozen payload hash mismatch')
        inputs[str(predpath)]=digest(predpath)
        with np.load(predpath,allow_pickle=False) as z:
            if not np.array_equal(z['observed'][:,0],x['y'].sum(1)) or not np.array_equal(z['scoring_mask'][:,0],np.arange(n)>=11):
                raise ValueError('Regional sum or H=12 mask mismatch')
            x['p']=z['expected'][:,0]
        x['row'].update(feature_delta_mean=float(x['feature_delta'][1:].mean()), neural_variance=float(x['y'][11:].sum(1).var()))
    for split in ('train','val'):
        actual=[x['trial_id'] for x in data if x['split']==split]
        if actual!=previous['memberships'][split] or len(actual)!={'train':218,'val':31}[split]:
            raise ValueError('Membership mismatch')
    number=1
    while (BASE/f'response-reliability-{number:03d}').exists():
        number+=1
    name=f'response-reliability-{number:03d}'
    stage=BASE/f'.{name}-{uuid.uuid4().hex}.staging'
    stage.mkdir()
    results=dict(schema_version=1, diagnostic_generation=name, dataset_generation_id=EXPECTED, session=EID,
                 checkpoint=frozen['checkpoint'], policy=policy, ordered_units=units, source_columns=cols,
                 memberships={s:[x['trial_id'] for x in data if x['split']==s] for s in ('train','val')},
                 generations={k:sorted(set(x['sample']['metadata']['alignment'][k] for x in data)) for k in ['neural_generation_id']},
                 alignment_generations=sorted(set(x['sample']['metadata']['source_alignment']['generation_id'] for x in data)),
                 feature_generations=sorted(set(x['sample']['metadata']['alignment']['visual_completion']['generation_id'] for x in data)),
                 replay_artifact_id=rm['artifact_id'], wheel_duplicates_removed=duplicates,
                 source_datasets=rd['source_datasets'], missing_events=rd['missing_events'], groups={}, onset={}, events={}, contrasts=[], duration={},
                 software=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__))
    endpoints=np.arange(22,60)
    eligible=[x for x in data if x['bins']>=60]
    conditions=sorted(set(x['condition'] for x in data))
    # Include absent combinations explicitly rather than silently dropping sparse groups.
    keys=[(kind,key) for kind in ('coarse','matched') for key in
          (conditions if kind=='coarse' else [f'{c}|{m}|{p}' for c in conditions for m in ('low','high') for p in ('ongoing','responded')])]
    summary=[]; neurons=[]; coverage=[]; profile_arrays={}
    for kind,key in keys:
        c=key.split('|')[0]
        reference=np.mean([x['y'][:60] for x in eligible if x['split']=='train' and x['condition']!=c],axis=0)
        entry={}
        for split in ('train','val'):
            group=[x for x in eligible if x['split']==split and x['condition' if kind=='coarse' else 'group']==key]
            a=analyze(group,f'{kind}:{key}:{split}',endpoints,reference)
            if group:
                azs=np.stack([np.interp(x['times'][:60],x['obs_times'],x['az']) for x in group])
                distances=[np.sqrt(np.mean(((azs[i]-azs[j]+180)%360-180)**2)) for i in range(len(group)) for j in range(i)]
                a['trajectory_pairwise_rms_deg']=dist(distances)
                a['raw_regional_mean_first_second']=np.mean([x['y'][:60].sum(1) for x in group],axis=0)
                a['raw_neuron_mean_first_second']=np.mean([x['y'][:60] for x in group],axis=0)
                profile_arrays[f'{kind}_{key}_{split}_raw']=a['raw_neuron_mean_first_second']
                a['response_time_s']=dist([x['response'] for x in group])
                a['wheel_path_rad']=dist([x['wheel_path_first_second_rad'] for x in group])
            entry[split]=a
            coverage.append(dict(kind=kind,group=key,split=split,n=a['n'],status=a['status'],
                                 all_trials=sum(x['split']==split and x['condition' if kind=='coarse' else 'group']==key for x in data),
                                 trial_ids=';'.join(map(str,a['trial_ids'])), trajectory_rms_mean=a.get('trajectory_pairwise_rms_deg',{}).get('mean')))
            for w,z in a['widths'].items():
                summary.append(dict(kind=kind,group=key,split=split,width=w,n=a['n'],status=a['status'],partitions=a['unique_partitions'],possible_partitions=a['possible_partitions'],
                                    mean_r=z['regional_split_half']['mean'],median_r=(z['regional_split_half']['quantiles'][3] if z['regional_split_half']['n'] else None),
                                    p05=(z['regional_split_half']['quantiles'][1] if z['regional_split_half']['n'] else None),p95=(z['regional_split_half']['quantiles'][5] if z['regional_split_half']['n'] else None),
                                    elapsed_removed_mean_r=z['elapsed_reference_removed']['mean'],shifted_mean_r=z['shifted_half_reference']['mean'],
                                    individual_loo_mean_r=z['individual_trial_to_other_mean']['mean'] if a['n']>1 else None))
                if a['n']>=6:
                    for j,u in enumerate(units):
                        nz=z['neuron_split_half'][j]
                        neurons.append(dict(kind=kind,group=key,split=split,width=w,n=a['n'],source_unit_id=u['source_unit_id'],column=j,
                                            mean_r=nz['mean'],defined_partitions=nz['n'],regional_mean_r=z['regional_split_half']['mean'],
                                            p05=nz['quantiles'][1] if nz['n'] else None,median=nz['quantiles'][3] if nz['n'] else None,p95=nz['quantiles'][5] if nz['n'] else None))
        entry['cross_split']={}
        for w in WIDTHS:
            available=min(entry['train']['n'],entry['val']['n'])>=3
            entry['cross_split'][w]=dict(status='available_descriptive' if available else 'unavailable_fewer_than_3_per_split',
                regional_r=corr(entry['train']['widths'][w]['mean'],entry['val']['widths'][w]['mean']) if available else None,
                neuron_r=column_corr(entry['train']['widths'][w]['neuron_mean'],entry['val']['widths'][w]['neuron_mean']) if available else None)
        results['groups'][f'{kind}:{key}']=entry
    # Equal-half-size unmatched reference isolates changes in contributor number.
    for key,entry in results['groups'].items():
        if not key.startswith('matched:') or entry['train']['n']<6:
            continue
        coarse=results['groups']['coarse:'+key[len('matched:'):].split('|')[0]]['train']
        pool=[x for x in eligible if x['split']=='train' and x['trial_id'] in coarse['trial_ids']]
        n=entry['train']['n']; half=n//2
        rng=np.random.default_rng(int.from_bytes(hashlib.sha256(f'{SEED}:size:{key}'.encode()).digest()[:8],'big'))
        chosen=[rng.choice(len(pool),n,replace=False) for _ in range(200)]
        # Each draw splits disjoint trials; across draws samples/partitions overlap.
        entry['size_matched_coarse_reference']={}
        for w in WIDTHS:
            st=np.stack([trailing(x['y'],w,endpoints).sum(1) for x in pool])
            rs=[corr(st[ix[:half]].mean(0),st[ix[half:]].mean(0)) for ix in chosen]
            entry['size_matched_coarse_reference'][w]=dist(rs)
    # Source-only onset changes: local early-stimulus reference, never a prestim baseline.
    for split in ('train','val'):
        for c in conditions:
            g=[x for x in eligible if x['split']==split and x['condition']==c]
            a=analyze(g,f'onset:{split}:{c}',np.arange(11,24))
            changes=np.array([x['y'][11:24].mean(0)-x['y'][:11].mean(0) for x in g])
            a['local_reference']='bins 0..10 versus bins 11..23; both within stimulus'
            a['regional_change']=bootstrap(changes.sum(1),f'onset:{split}:{c}') if g else None
            a['unit_changes']=changes.mean(0) if g else None
            results['onset'][f'{split}:{c}']=a
    # Duration sensitivity keeps exactly the same first-second endpoint support.
    for c in conditions:
        for split in ('train','val'):
            for duration in ('short','long'):
                g=[x for x in eligible if x['split']==split and x['condition']==c and x['duration_class']==duration]
                results['duration'][f'{c}:{split}:{duration}']=analyze(g,f'duration:{c}:{split}:{duration}',endpoints)
    # Events: only full observed windows, separately counted scoring-compatible windows.
    for event in ('go_cue','first_movement','response','feedback','visual_movement_onset','freeze_proxy'):
        for split in ('train','val'):
            xs=[x for x in data if x['split']==split]
            indices={x['trial_id']:int(np.argmin(abs((x['times']-x['stim_on'])-x[event]))) for x in xs if np.isfinite(x[event])}
            g=[x for x in xs if x['trial_id'] in indices and indices[x['trial_id']]>=12 and indices[x['trial_id']]+24<=x['bins']
               and x['times'][0]-.5/60<=x['stim_on']+x[event]<=x['times'][-1]+.5/60]
            a=analyze(g,f'event:{event}:{split}',np.arange(11,36),event_indices=indices)
            a.update(event=event, finite_events=len(indices), complete_source_windows=len(g),
                     complete_scoring_windows=sum(indices[x['trial_id']]-12>=11 for x in g),
                     timing_kind='reconstructed' if event=='visual_movement_onset' else 'recorded timestamp used as freeze proxy' if event=='freeze_proxy' else 'recorded',
                     max_rounding_error_ms=max([abs(x['times'][indices[x['trial_id']]]-x['stim_on']-x[event])*1000 for x in g],default=None))
            changes=np.array([x['y'][indices[x['trial_id']]:indices[x['trial_id']]+12].mean(0)-x['y'][indices[x['trial_id']]-12:indices[x['trial_id']]].mean(0) for x in g])
            a['regional_post_minus_pre']=bootstrap(changes.sum(1),f'eventdelta:{event}:{split}') if g else None
            a['unit_post_minus_pre']=changes.mean(0) if g else None
            if len(g)>=6:
                pp,_=partitions(len(g),f'event:{event}:{split}')
                signs=np.stack([(np.sign(changes[l].mean(0))==np.sign(changes[r].mean(0))) & (changes[l].mean(0)!=0) & (changes[r].mean(0)!=0) for l,r in pp])
                a['unit_sign_agreement_fraction']=signs.mean(0)
            results['events'][f'{event}:{split}']=a
    # All zero-contrast trials are retained, including those lacking first-second support.
    results['zero_trial_ids']={s:[x['trial_id'] for x in data if x['split']==s and x['contrast']==0] for s in ('train','val')}
    # Observational visible-vs-zero comparison: side, wheel category, phase, duration support.
    for split in ('train','val'):
        for side in ('left','right'):
            for movement in ('low','high'):
                for phase in ('ongoing','responded'):
                    base=[x for x in eligible if x['split']==split and x['side']==side and x['movement_class']==movement and x['phase']==phase]
                    for contrast in sorted(set(x['contrast'] for x in base if x['contrast']>0)):
                        groups=[[x for x in base if x['contrast']==c] for c in (0,contrast)]
                        if not groups[0]:
                            continue
                        a=dict(split=split,side=side,movement=movement,phase=phase,contrast=contrast,n_zero=len(groups[0]),n_visible=len(groups[1]),
                               zero_ids=[x['trial_id'] for x in groups[0]],visible_ids=[x['trial_id'] for x in groups[1]],
                               status='descriptive_supported' if min(map(len,groups))>=3 else 'insufficient_contributors')
                        stats=[bootstrap([x['y'][22:60].sum(1).mean() for x in g],f'contrast:{split}:{side}:{movement}:{phase}:{c}') for c,g in zip((0,contrast),groups)]
                        a['mean_counts']=stats
                        a['visible_minus_zero']=stats[1]['mean']-stats[0]['mean']
                        a['wheel_path_means']=[np.mean([x['wheel_path_first_second_rad'] for x in g]) for g in groups]
                        a['response_time_means']=[np.nanmean([x['response'] for x in g]) for g in groups]
                        # Independent whole-trial bootstrap difference, not temporal-bin resampling.
                        if min(map(len,groups))>=3:
                            rng=np.random.default_rng(SEED)
                            means=[]
                            for g in groups:
                                values=np.array([x['y'][22:60].sum(1).mean() for x in g])
                                means.append(values[rng.integers(len(values),size=(1000,len(values)))].mean(1))
                            a['difference_bootstrap_ci95']=np.quantile(means[1]-means[0],[.025,.975])
                        results['contrasts'].append(a)
    # Frozen model comparison at identical endpoints; never used for matching.
    results['frozen_model']={}
    for key,entry in results['groups'].items():
        for split in ('train','val'):
            g=[x for x in eligible if x['split']==split and x['trial_id'] in entry[split]['trial_ids']]
            if not g:
                continue
            results['frozen_model'][f'{key}:{split}']={w:dict(n=len(g),equal_trial_r=dist([corr(trailing(x['y'],w,endpoints).sum(1),trailing(x['p'][:,None],w,endpoints)[:,0]) for x in g]),
                condition_mean_r=corr(np.mean([trailing(x['y'],w,endpoints).sum(1) for x in g],axis=0),np.mean([trailing(x['p'][:,None],w,endpoints)[:,0] for x in g],axis=0))) for w in WIDTHS}
    # Candidate examples chosen by metadata/features, never by model/neural correlation.
    pool=[x for x in eligible if x['split']=='train' and x['contrast']>0]
    dynamic=max(pool,key=lambda x:(x['feature_delta'][1:].mean(),-x['trial_id']))
    sustained=min([x for x in pool if x['duration']>=duration_cut and x['feature_delta'][1:].mean()>1e-6],key=lambda x:(x['feature_delta'][1:].mean(),x['trial_id']))
    zero=sorted([x for x in eligible if x['split']=='train' and x['contrast']==0],key=lambda x:(abs(x['wheel_path_first_second_rad']-cutoff),x['trial_id']))[0]
    # Best-supported condition is determined solely by contributor number, not reliability.
    c=max(conditions,key=lambda c:(sum(x['condition']==c and x['split']=='train' for x in eligible),c))
    cg=sorted([x for x in pool if x['condition']==c],key=lambda x:(abs(x['duration']-duration_cut),x['trial_id']))
    candidate=[('visually_dynamic',dynamic),('sustained_low_feature_change',sustained),('zero_control',zero),('best_supported_condition',cg[0])]
    early=[x for x in pool if x['phase']=='responded' and x['movement_class']=='high']
    ec=max(conditions,key=lambda c:(sum(x['condition']==c for x in early),c))
    eg=[x for x in early if x['condition']==ec]
    em=np.median([x['duration'] for x in eg])
    candidate.append(('best_supported_early_response',min(eg,key=lambda x:(abs(x['duration']-em),x['trial_id']))))
    results['candidates']=[dict(category=label,trial_id=x['trial_id'],condition=x['condition'],group=x['group'],duration=x['duration'],
                                mean_feature_delta=x['feature_delta'][1:].mean(),neural_variance=x['y'][11:].sum(1).var(),
                                selection='feature/metadata criterion; neural variance reported after selection') for label,x in candidate]
    results['coverage_totals']={s:dict(n=sum(x['split']==s for x in data),eligible=sum(x['split']==s for x in eligible),
        excluded_ids=[x['trial_id'] for x in data if x['split']==s and x['bins']<60],
        full_observed_bins=sum(x['bins'] for x in data if x['split']==s),
        first_second_retained_bins=60*sum(x['split']==s for x in eligible),
        scoring_comparison_endpoints=38*sum(x['split']==s for x in eligible),
        scoring_valid_bins=sum(x['bins']-11 for x in data if x['split']==s)) for s in ('train','val')}
    frame=pd.DataFrame([x['row'] for x in data])
    frame.to_csv(stage/'trials.csv',index=False)
    pd.DataFrame(coverage).to_csv(stage/'condition-coverage.csv',index=False)
    pd.DataFrame([r for r in coverage if r['kind']=='matched']).to_csv(stage/'matched-groups.csv',index=False)
    pd.DataFrame(summary).to_csv(stage/'reliability-summary.csv',index=False)
    pd.DataFrame(neurons).to_csv(stage/'source-neuron-summary.csv',index=False)
    pd.DataFrame(results['contrasts']).to_csv(stage/'zero-visible-contrasts.csv',index=False)
    np.savez_compressed(stage/'mean-profiles.npz',**profile_arrays)
    enrich(results,data,stage)
    make_figures(stage,results,data,pd.DataFrame(summary),pd.DataFrame(coverage),pd.DataFrame(neurons))
    results['input_sha256']=inputs
    results['analysis_sha256']=digest(__file__)
    results['utility_sha256']=digest(Path(__file__).with_name('audit_representation_response.py'))
    results['implementation_sha256']={p:digest(ROOT/p) for p in (
        'src/visual_replay/preparation.py','src/visual_replay/timeline.py','src/visual_features/encoder.py',
        'src/alignment/temporal.py','src/neural_data/sources.py','src/neural_data/counting.py',
        'src/training_dataset/samples.py','src/training_dataset/artifacts.py','src/trainer/selection.py',
        'src/trainer/targets.py','src/trainer/temporal_diagnostics.py')}
    # Per-bin neuron means are stored once in compressed NPZ, not repeated in JSON.
    extra_profiles={}
    def compact(obj,prefix=''):
        if isinstance(obj,dict):
            for key in list(obj):
                if key in ('neuron_mean','raw_neuron_mean_first_second'):
                    if key=='neuron_mean' and prefix.endswith('/widths/1'):
                        extra_profiles[prefix]=obj[key]
                    del obj[key]
                else:
                    compact(obj[key],prefix+'/'+str(key))
        elif isinstance(obj,list):
            for i,value in enumerate(obj):
                compact(value,prefix+'/'+str(i))
    compact(results)
    np.savez_compressed(stage/'source-profiles.npz',**extra_profiles)
    (stage/'results.json').write_text(json.dumps(clean(results),separators=(',',':'),allow_nan=False),encoding='utf8')
    # Report and publication can also be finalized from this staging directory
    # after visual review, without reopening scientific inputs.
    print('Analysis staging directory:',stage,flush=True)
    print('Training metadata thresholds:',cutoff,duration_cut,flush=True)
    print(pd.DataFrame(summary).query("kind=='matched' and split=='train' and width==12")[['group','n','mean_r','elapsed_removed_mean_r']].to_string(index=False),flush=True)


def write_report(stage):
    r=read(stage/'results.json')
    coverage=pd.read_csv(stage/'condition-coverage.csv')
    summary=pd.read_csv(stage/'reliability-summary.csv')
    neurons=pd.read_csv(stage/'source-neuron-summary.csv')
    trials=pd.read_csv(stage/'trials.csv')
    def fmt(v):
        return 'NA' if v is None or not np.isfinite(v) else f'{v:.3f}'
    counts=coverage[coverage.kind=='matched'].pivot(index='group',columns='split',values='n').reset_index()
    counts['interpretation']=np.where(counts.train>=6,'training split-half','descriptive / unavailable')
    train_summary=summary[(summary.kind=='matched')&(summary.split=='train')&(summary.width==12)&(summary.n>=6)].copy()
    reliable=[]
    for _,row in train_summary.iterrows():
        e=r['groups']['matched:'+row.group]
        nm=neurons[(neurons.kind=='matched')&(neurons.split=='train')&(neurons.width==12)&(neurons.group==row.group)]
        reliable.append(dict(group=row.group,n=int(row.n),mean_r=fmt(row.mean_r),p05=fmt(row.p05),p95=fmt(row.p95),
                             size_matched_coarse=fmt(e['size_matched_coarse_reference']['12']['mean']),
                             median_unit_r=fmt(nm.mean_r.median()),trajectory_RMS_deg=fmt(e['train']['trajectory_pairwise_rms_deg']['mean'])))
    widths=[]
    for key,e in r['groups'].items():
        if key.startswith('matched:') and e['train']['n']>=6:
            widths.append(dict(group=key[8:],n=e['train']['n'],**{f'{w}_bins':fmt(e['train']['widths'][str(w)]['regional_split_half']['mean']) for w in WIDTHS}))
    cross=[]
    for key,e in r['groups'].items():
        if key.startswith('matched:') and e['cross_split']['12']['regional_r'] is not None:
            cross.append(dict(group=key[8:],train=e['train']['n'],val=e['val']['n'],r_200ms=fmt(e['cross_split']['12']['regional_r'])))
    eventrows=[]
    for key,a in r['events'].items():
        b=a['regional_post_minus_pre']
        eventrows.append(dict(event_split=key,finite=a['finite_events'],source_windows=a['n'],scoring_windows=a['complete_scoring_windows'],
                              post_minus_pre=fmt(b['mean']) if b else 'NA',ci95=str([round(v,3) for v in b['ci95']]) if b and b['ci95'] is not None else 'NA'))
    zerorows=[]
    for key,a in r['zero_events'].items():
        b=a['regional_post_minus_pre']
        zerorows.append(dict(event_split=key,n=a['n'],change=fmt(b['mean']) if b else 'NA',ci95=str([round(v,3) for v in b['ci95']]) if b and b['ci95'] is not None else 'NA'))
    dur=[]
    for key,a in r['duration'].items():
        if ':train:' in key:
            dur.append(dict(condition_duration=key.replace(':train:',':'),n=a['n'],r_200ms=fmt(a['widths']['12']['regional_split_half']['mean']) if a['n'] else 'NA'))
    candidates=pd.DataFrame(r['candidates'])[['category','trial_id','condition','duration','mean_feature_delta','neural_variance']]
    for c in ('duration','mean_feature_delta','neural_variance'):
        candidates[c]=candidates[c].map(lambda x:f'{x:.5g}')
    totalrows=[]
    for split,a in r['coverage_totals'].items():
        totalrows.append(dict(split=split,trials=a['n'],excluded_trials=len(a['excluded_ids']),observed_bins=a['full_observed_bins'],
                              first_second_bins=a['first_second_retained_bins'],comparison_endpoints=a['scoring_comparison_endpoints'],
                              unexamined_by_fixed_first_second=f"{100*(1-a['first_second_retained_bins']/a['full_observed_bins']):.2f}%"))
    wheelrows=[]
    for k,a in r['wheel_association'].items():
        b=a['trial_bootstrap']
        wheelrows.append(dict(split_zero=k,n_defined=b['n'],mean_r=fmt(b['mean']),ci95=str([round(v,3) for v in b['ci95']]) if b['ci95'] is not None else 'NA'))
    models=[]
    for key in ('matched:left:0.25|high|responded','matched:right:0.125|high|responded','matched:right:1|high|responded'):
        a=r['frozen_model'][key+':train']['12']
        models.append(dict(group=key[8:],n=a['n'],trial_mean_r=fmt(a['equal_trial_r']['mean']),mean_profile_r=fmt(a['condition_mean_r'])))
    report=f"""# Condition- and movement-matched VISpm response reliability

Scenario 11 / Attempt 002; diagnostic `{r['diagnostic_generation']}`. Research audit using training and validation only. This is not final test evaluation.

## Executive answers

1. **Comparable-condition averages can repeat, but repeatability is heterogeneous.** At 200 ms, early-response/high-wheel right/0.125 has mean whole-trial split-half r=0.912 (n=12) and right/1 has r=0.968 (n=19). Their raw-bin values are only 0.182 and 0.303. Ongoing-response groups are often near zero or negative. This supports repeatable slow regional dynamics in some approximate conditions, not reliable prediction of every spike-count fluctuation.
2. **Matching improves interpretation more consistently than reliability.** It separates early-response and ongoing task phases. Left/0.25 divides into r=0.770 (n=15) versus -0.285 (n=16), against coarse r=0.546. Equal-size coarse references are 0.318 and 0.304. Other groups improve little or become weaker. High-wheel and early response almost coincide, preventing independent attribution to movement and task phase.
3. **Zero-contrast trials expose task/movement associations without visible Gabor variation.** All 27 training and six validation zero-contrast trials are inspected. Recorded-event changes and full-support wheel associations are reported below. They support a behavioral/task confound check; sparse fast-response controls and residual timing differences prevent separation of visual and behavioral contributions.
4. **Source-neuron reliability is heterogeneous and usually lower than regional reliability in the strongest groups.** At right/1 high/responded the median defined unit r is 0.547 versus regional 0.968; none of the 99 units has a larger mean split-half correlation. Low-count undefined metrics remain unavailable, and no neurons are removed.
5. **Harmful regional summation is not demonstrated.** Opposing within-stimulus deviations exist, but they do not establish reproducible stimulus-specific information lost by summation. High unit correlations selected from many exploratory comparisons are not sufficient evidence. Preserve the 99-unit regional target.
6. **There is useful training support but insufficient matched validation repetition.** Sixteen of 40 predefined matched combinations have >=6 training trials; none has >=6 validation trials. Only three combinations permit descriptive cross-split profile comparisons with >=3 contributors on each side. Exact repeated visual trajectories and a physiological noise ceiling are unidentified.
7. **A controlled memorization diagnostic remains informative (Outcome B).** Use a small, diverse training-only sample to distinguish optimization/memorization capacity from content-dependent learning. Do not infer intrinsic model incapacity or change architecture, targets, likelihood, or context from these results.

## Scope, provenance, and inspected contracts

Dataset `{r['dataset_generation_id']}`; checkpoint SHA-256 `{r['checkpoint']['sha256']}`, selected epoch {r['checkpoint']['epoch']}. Session `{EID}`. Neural generation `{r['generations']['neural_generation_id'][0]}`; alignment `{r['alignment_generations'][0]}`; visual-feature generation `{r['feature_generations'][0]}`; replay artifact `{r['replay_artifact_id']}`.

Ordered identities match the previous audit and frozen diagnostic, including source row, PID, probe, collection and revision. Every selected sample payload and frozen prediction hash was checked. All 249 regional sums equal frozen observations exactly, physical centers equal onset+(k+0.5)/60, and scoring masks equal k>=11. Wheel decoded-array hashes equal replay input hashes. The current checkpoint hash was checked without loading a model. No new inference, predictive fitting, training, source acquisition or test payload access occurs.

Repository rules, evaluation specification/interface, documentation ownership/dependency graph, relevant upstream interfaces and existing audit definitions were inspected. Relevant implementation evidence includes replay preparation/timeline, CLIP encoder, neural source selection/counting, interpolation, dataset packaging, runtime selection/regional targets, and frozen temporal diagnostics. This is a cross-pipeline scientific inspection, not new production dependencies or a component redesign. `results.json` records implementation, input and script hashes. Full split memberships and all 99 ordered unit records are preserved there. Existing model/data/configuration and historical diagnostic artifacts remain unchanged.

## Matching feasibility and variable evidence

| Variable | Evidence class | Availability / limitation |
|---|---|---|
| Side, contrast | Recorded | Trial-table rows, cross-checked against dataset replay definition; zero-contrast side is bookkeeping, not visible lateralized input |
| Stimulus onset / offset | Recorded | Trial table plus explicit unrevisioned stimOff array, checked against each aligned sample |
| Wheel position | Recorded | Original ALF radians and session-clock seconds; decoded hashes match replay inputs |
| Wheel path / velocity summaries | Derived from recorded inputs | Raw-position total variation on the first second, with interpolated boundary positions; per-bin edge interpolation for full-support velocity associations; not measured continuous velocity |
| First movement | Recorded metadata | Trial-table firstMovement_times; an upstream detected event, not a new audit detection of physical onset |
| Go cue, response, choice, feedback | Recorded | Trial table; response timestamp does not guarantee a movement was made; feedback is a separate event |
| Closed-loop and freeze times | Reconstructed task-state proxies | goCue and response respectively; dedicated closedLoop_times and stimFreeze_times unavailable at selected revision; some activation falls back to onset under documented assumptions |
| Position, movement, stage / frozen fractions | Reconstructed / derived | Nominal 60 Hz replay, angular trajectory from wheel plus assumed gain/sign/geometry, not measured display motion |
| Stimulus duration / recorded trial duration | Derived | Separate onset-offset/aligned-bin interval versus trial intervals_0..intervals_1 |
| Physical frame timing, eye/head/other movement | Unavailable | No measured frame presentation or complete behavioral control; no invented covariates |

Wheel observations span each aligned trial. Duplicate timestamps retain the last acquired value ({r['wheel_duplicates_removed']} removed), matching replay behavior. Sampling is irregular: median largest bracketing gap is {trials.wheel_max_sample_gap_first_second_s.median():.3f} s and maximum {trials.wheel_max_sample_gap_first_second_s.max():.3f} s; `trials.csv` reports each first-second window. Gaps are not automatically evidence of movement or a recording dropout, but interpolation across them cannot certify continuous behavior or physical stationarity. No new gap exclusion was optimized using neural results. Synthetic trial-specific phase and nominal replay parameters further prevent an exact repeated-input interpretation.

Primary groups are recorded side/contrast, first-second wheel-path <= **{r['policy']['wheel_cut_rad']:.9f} rad** versus larger, and recorded response timestamp <=1 s versus later/missing. Wheel cutoff is the training median; 1 s is the fixed observation interval. `low/high` are relative wheel-path categories, not physical stationary/moving labels. `ongoing/responded` are broad response phases, not exact event-time matching. The training duration median is **{r['policy']['duration_cut_s']:.6f} s**. Both medians and all rules were fixed from training metadata before opening neural arrays; validation used the same values. All 40 combinations are retained, including empty ones and a validation-only combination.

Across both splits, the wheel/response cells are high/ongoing=2, high/responded=121, low/ongoing=124 and low/responded=2. Thus these covariates cannot be independently controlled with this sample. First-second pairwise circular-azimuth RMS is exported for every group; high/responded training groups range from about 5 to 33 degrees despite this matching. Low/ongoing groups generally have much closer position trajectories. Neither category nor small RMS matches synthetic phase, exact pixels, continuous wheel dynamics, all events or other behavior. Further splitting would reduce already sparse validation support and is not warranted for this audit.

### All matched condition counts

{table_md(counts)}

Ten coarse side/contrast conditions and their exact trial lists are also in `condition-coverage.csv`; each row identifies the split and support status. The explicit minimum is six distinct whole trials (>=3 per half). One to five trials support descriptive profiles only; zero means unavailable. Cross-split comparison requires >=3 per split. These thresholds enable the procedures but do not guarantee stable population estimates. In particular, one validation trial never receives an internal split-half estimate.

## Support, weighting, and uncertainty

{table_md(pd.DataFrame(totalrows))}

All trials contain a complete first second, so no trial was discarded for duration. The fixed first-second analysis deliberately leaves most later bins unexamined by condition profiles: this is an early-window claim, not whole-stimulus reliability. Later observed data enter full-support wheel associations and complete event windows. No padding or difficult-trial exclusion fills the gap. Training durations range 1.15..62.9 s; validation 1.283..60.633 s.

Regional scoring comparisons use endpoints 22..59 (centers 375..991.7 ms) for every width. At width 12, each mean contains only complete bins >=11; at width 1 the same endpoints retain raw counts. Thus improved r across widths cannot be attributed to different endpoints. Each comparison averages whole trials equally, sums the exact 99 source units, and correlates the two mean temporal profiles. Individual-neuron summaries give each neuron equal weight. Trailing averages are counts per original 16.67 ms bin, not counts in a new 200 ms target. Overlapping endpoints are not independent observations and receive no binwise significance or confidence claim.

Whole-trial partitions are disjoint within a split, deterministic seed {SEED}, all unique unlabeled halves when <=200 possible, otherwise 200 unique partitions. For odd group size the smaller half defines the partition. Full regional correlation lists, possible/used partition counts, and per-neuron defined counts/quantiles are recorded. Negative r is preserved; constant profiles yield null. Finite-partition quantiles describe re-partitioning sensitivity, not independent repetitions, confidence intervals, individual-trial predictability or a noise ceiling. Trial-level bootstrap intervals use 1000 whole-trial resamples for scalar changes/contrasts; sparse or imbalanced intervals remain exploratory.

## Quantitative regional and source-neuron reliability

All supported training matched groups at 200 ms:

{table_md(pd.DataFrame(reliable))}

`size_matched_coarse` draws the same number of trials from the parent side/contrast condition and uses identical half sizes/endpoints. Its 200 resamples overlap across draws; they are a descriptive contributor-size reference. Matching improves some groups and worsens others. `median_unit_r` uses available per-unit partition means, retaining all 99 units in source exports; undefined values are not replaced by zero. `source-neuron-summary.csv` includes n, defined partitions and 5th/50th/95th partition quantiles for every unit and width.

Widths are predefined, not selected by the observed maximum:

{table_md(pd.DataFrame(widths))}

The difference between modest raw-bin repeatability and strong slow-profile repeatability is scientifically material. Group means average spike variability; the `individual_trial_to_other_mean` distribution in results separately describes each trial against the average of the others. It is not a fitted prediction score or a reliable-variance estimate.

The half-profile circular-shift reference preserves smoothness but changes temporal phase; it is not a calibrated null test. The training-only leave-condition-out elapsed reference is fixed for both halves and validation. Removing it is only a sensitivity analysis: subtracting a shared nonconstant curve can itself induce correlation, so a large residual r is not proof of condition-specific information. Cross-condition profiles, coarse/matched differences, event covariation and sparse cross-split contrast support are considered together. `profile_contrasts` records all training-supported within-side/wheel/phase contrasts and whether the same dynamic contrast is estimable in validation.

### Training versus validation

{table_md(pd.DataFrame(cross))}

These are descriptive correlations of different trial averages, not within-validation reliability. All other matched cross-split comparisons are explicitly unavailable. The apparently encouraging right/0.0625 comparison has only 6/3 contributors and much larger validation trajectory RMS (~40 degrees) than training (~5 degrees); it is not an exact stimulus replication. The right/1 condition has one validation trial. Validation already selected the checkpoint and is not an untouched final test set.

## Zero-contrast and movement controls

All zero-contrast trials appear in `trials.csv`, with duration, measured wheel summaries, task events and reconstructed state. CLIP change is zero for these trials. Reconstructed azimuth in them is latent task geometry, not meaningful visible Gabor motion. The zero-control figure pools sides and response phases for display and must not be read as a matched movement effect.

Within-trial full-support 200 ms wheel-derived speed/count associations (equal-trial summaries; `split_zero=True` means zero contrast):

{table_md(pd.DataFrame(wheelrows))}

Correlations remain observational, are not detrended or event-adjusted, and can reflect elapsed time/task state. Source positions are measured; interpolated velocity is derived. Trial-level uncertainty is not obtained by treating overlapping windows as independent samples.

Zero-contrast complete recorded-event windows, post 0..11 minus pre -12..-1 bins (counts/original bin):

{table_md(pd.DataFrame(zerorows))}

Training zero-contrast activity rises by +1.875 counts/bin around recorded first movement (22 complete trials, bootstrap interval [1.170,2.727]); the three-trial validation interval spans zero. The full-support zero-contrast wheel-derived speed correlation averages 0.196 in training (27 trials, bootstrap interval [0.104,0.289]); the six-trial validation interval spans zero. These controls assess modulation in the absence of visible Gabor variation. A nonzero contrast is neither required for all neural changes nor sufficient to assign them to wheel movement. Exact wheel trajectories, choices and response times remain unmatched. Zero high-wheel/responded groups have only 2 left and 3 right training trials (one each in validation); internal reliability is unsupported. Continuous-movement adjustment and causal separation are unavailable.

`zero-visible-contrasts.csv` and `results.json:contrasts` compare each available nonzero contrast separately against zero within side/wheel/phase and identical endpoints. They include contributor lists, count differences, wheel/response imbalances and trial-bootstrap difference intervals where >=3 per group. Most supported training intervals include zero; the right/high-contrast low/ongoing contrast is based on only three visible trials and many exploratory contrasts. No validation contrast has adequate zero contributors. No isolated visible-stimulus effect is established.

## Aggregation and baseline-relative heterogeneity

`results.json:aggregation` is an exploratory supplement using a common early reference (bins 12..23, 200..400 ms) and late window (36..59, 600..1000 ms), preserving all units. These windows were not optimized to maximize neural correlation and do not redefine primary matching or the four requested widths. It records regional change and whole-trial bootstrap interval, per-unit directions, partition sign agreement, signed/absolute deviation ratio, top-10 absolute-deviation share, and cross-split directions where supported. These windows are not pre-stimulus baselines. Counts are nonnegative and cannot cancel; signed deviations can.

Opposing signs and concentration of mean changes are descriptive and susceptible to noise. Stable signs alone do not establish selective visual information. All-99-unit profiles, source means, and unavailable correlations are exported. In strong early-response conditions the regional average is more consistent than typical individual units. In weak regional conditions some unit means are larger, but neuron/condition/window multiplicity and few independent validation trials preclude a defensible harmful-summation claim. Timing heterogeneity and a small contributing subset remain plausible explanations, not identified mechanisms. No neurons or targets were reselected and no neuron model was fitted.

## Source-only onset, recorded events, and trial duration

Raw first-second source-neural means include bins 0..10, which have no model-scored prediction. Separate onset reliability uses endpoints 11..23 for all widths, allowing width-12 source windows beginning at bin 0. Local source change compares bins 0..10 against 11..23, both already under stimulus exposure. All ten coarse training regional bootstrap intervals for that simple change include zero; this does not rule out heterogeneous or differently timed onset responses. There is no pre-stimulus baseline and the figure does not imply complete physiological latency coverage.

Complete event windows use nearest physical center and observed offsets -12..23; all widths share endpoints -1..23. Event rounding is recorded and is <=8.334 ms where supported. Windows are never extended before stimulus onset or after stimulus offset. `scoring_windows` additionally requires all source bins >=11; source-only windows are not model predictions.

{table_md(pd.DataFrame(eventrows))}

The training first-movement change is +1.366 counts/bin (n=116; trial-bootstrap 95% interval [1.015,1.726]); validation n=17 includes zero. Response and feedback show decreases in both splits, but their timestamps nearly coincide and cannot identify separate effects. Reconstructed first azimuth movement often occurs too close to onset for the complete prewindow, leaving only 8/3 contributors; these events are not necessarily meaningful visible motion in zero contrast. Freeze is a response-based proxy. No event comparison establishes visual causality.

Training short/long sensitivity at the same first-second endpoints (median cut {r['policy']['duration_cut_s']:.3f} s):

{table_md(pd.DataFrame(dur))}

These duration strata retain side/contrast but do not fully control movement/phase. Left/0.25 and right/0.125 short groups are more consistent than long groups; some long high-contrast groups are too sparse. This supports task/duration heterogeneity as a descriptive limitation, not a causal duration effect or an inference that the model receives future trial length. Matched duration subdivision would often destroy repetition support.

## Reliability versus frozen-model learnability

After defining and measuring groups independently, frozen predictions were inspected at the same endpoints. Training examples at 200 ms:

{table_md(pd.DataFrame(models))}

Frozen model correlations can be strong for these early-response averages and substantially smaller on raw bins or full long trials. First-second onset-related task trends can therefore be learned despite weak overall temporal prediction. High condition-mean r is insensitive to amplitude calibration and does not establish visual-content dependence, correct individual stochastic spikes, or adequate fit over the full stimulus interval. Raw/smoothed trial correlations for every analyzed group are in `frozen_model`; there was no new inference or checkpoint selection. Reliability distributions and model scores measure different quantities and do not define a performance ceiling.

## Verified findings

### R01 — Matching support cannot identify fully controlled visual reliability

Priority: High. Confidence: Confirmed. Component/data: replay, recorded wheel/events, train/validation coverage. Evidence: all-count table, `condition-coverage.csv`, `trials.csv`, `groups:*:trajectory_pairwise_rms_deg`; 16 training-supported groups, zero validation split-half groups, 245/249 trials in the two dominant wheel/phase cells. Interpretation: approximately comparable early conditions can be described, but independent behavioral adjustment and exact repeats are unavailable. Limitation: coarse wheel magnitude/response phase, irregular source sampling, synthetic phase and reconstruction assumptions. Suggested disposition: Keep data/contracts; report unresolved exact visual-specific reliability, no noise ceiling.

### R02 — Regional repeatability depends strongly on phase and temporal width

Priority: Medium. Confidence: Confirmed. Component/data: source neural counts. Evidence: `reliability-summary.csv`, full regional partition lists; right/1 high/responded r=0.303 raw versus 0.968 at 12 bins, left/0.25 high/responded r=0.770 versus low/ongoing -0.285. Interpretation: slow condition-average structure exists alongside weak/negative groups. Limitation: correlation of means, overlapping windows and small finite trial sets; outcome-driven group selection was not used. Suggested disposition: Keep original bins and target; use diverse controls in the next diagnostic.

### R03 — Blank controls and recorded events constrain visual interpretation

Priority: Medium. Confidence: Confirmed for observed associations; visual/behavioral attribution unresolved. Component/data: zero-contrast source counts, recorded wheel/firstMovement/response/feedback. Evidence: `wheel-associations.csv`, `zero_events`, `zero-visible-contrasts.csv`, complete-event table. Interpretation: neural modulation can accompany task/movement variables despite constant visual features; this is an observational confound check. Limitation: high-movement blanks and validation controls are sparse, event/phase covary, derived velocity can span gaps. Suggested disposition: Keep blank controls; do not claim isolated visual or causal movement effects.

### R04 — Unit heterogeneity does not establish harmful summation

Priority: Medium. Confidence: Confirmed for heterogeneity; information-loss hypothesis unsupported. Component/data: all original 99 VISpm units and exact regional sum. Evidence: `source-neuron-summary.csv`, `source-profiles.npz`, `aggregation`; median right/1 high/responded unit r=0.547 against region 0.968, zero units higher by mean partition r. Interpretation: summation can improve profile consistency; opposing deviations need independent repeatability before implying lost information. Limitation: undefined sparse unit profiles, multiplicity and poor cross-split support. Suggested disposition: Keep aggregation and source identity; no neuron-level retraining.

### R05 — Weak overall predictions coexist with learned early temporal profiles

Priority: Medium. Confidence: Confirmed. Component/data: frozen training/validation predictions. Evidence: frozen-model table and `frozen_model` records; right/1 high/responded mean-profile r=0.991 at 200 ms. Interpretation: model learnability cannot be diagnosed from a pooled weak score alone. Limitation: average-profile correlation does not assess calibration, causal visual content or full-trial fit; validation selected this checkpoint. Suggested disposition: Keep checkpoint frozen; next use a controlled training-only memorization experiment.

Confirmed observations are the exported counts, correlations, support and provenance checks. Task-state and representation limitations are plausible explanations. Exact predictable variance, causal visual/movement effects, harmful summation and intrinsic model incapacity are unsupported hypotheses. Unavailable comparisons retain null/status rather than zero.

## Intentional-overfitting preparation and one next step

Candidates use training metadata/features only; neural variance is reported after selection, never used to optimize group reliability or frozen-model correlation:

{table_md(candidates)}

Dynamic = maximum equal-trial mean adjacent CLIP distance among visible trials; sustained = lowest nonzero feature change among visible trials at least the training median duration; blank = wheel path nearest the training cutoff; best-supported coarse condition = largest training contributor count, trial nearest the training median duration; best-supported early-response condition = largest high-wheel/early-response training group, trial nearest its own median duration. Ties use original trial ID. All have complete alignment and H=12 support. The sustained trial is about 60 s and is an optional long-context stress example, not the first inexpensive capacity check. The coarse candidate does not inherit the reliability of the condition's early-response subgroup. A reproducible condition mean does not guarantee a predictable individual trial.

**One concrete next step:** separately specify a training-only intentional-memorization experiment beginning with trial **4** (`best_supported_early_response`), then a small diverse group **[4, 261, 272, 137]**. Retain optional sustained trial 14 for a later long-support check. Use the existing contracts and explicitly compare visual-content dependence with elapsed-position memorization. Conceptual controls include shuffled/constant visual inputs with unchanged positions, matched-length visible/blank comparisons, and shifted input/position controls that preserve H=12 validity. They are recommendations only, not implemented here.

For that later experiment inspect training Poisson NLL, raw and predefined-width trial r/R², calibration and mean bias, trial-position structure, and valid-bin counts. Use a training-derived constant-rate comparison if specified. Single-trial memorization tests capacity, not generalization or exact response reliability; identical/similar CLIP contexts can have different stochastic counts. Do not use validation/test trials for candidate selection or force observed-count variance into expected rates.

**Recommendation: Outcome B — heterogeneous reproducibility.** A controlled memorization diagnostic has value because some slow target structure exists and some of it is already learned. Fully behavior-matched visual-specific repeatability remains unresolved (the limitation described by Outcome C); broader session acquisition is not required before the limited capacity question is asked. Preserve all scientific contracts and conduct no training as part of this audit.

## Artifacts and reproduction

Run `.venv/Scripts/python.exe script/audit/audit_response_reliability.py` from the checkout. The script creates a uniquely named staging directory under `output/audit/scenario-11/002` for the next unused generation. After reviewing generated figures, call the same module's `write_report(stage)` and `publish(stage)`; publication hashes every payload and atomically renames to `response-reliability-NNN`, refusing an existing destination. Incomplete staging outputs are not published generations. No test sample array is opened; shared source wheel/trial files are decoded for authorized trial windows only.

`results.json` contains policy, complete memberships/units/provenance, regional partition distributions, per-neuron summaries, onset/events/duration/contrast/model comparisons and candidate definitions. Per-bin neuron means are compressed once in `mean-profiles.npz` (raw first-second group profiles) and `source-profiles.npz` (width-1 endpoint means for groups/onset/events/duration); widths can be reconstructed by the script from immutable inputs. JSON does not duplicate full trial/bin arrays. CSVs provide all condition counts, matched memberships, reliability and neuron summaries, wheel associations and zero-visible contrasts. `manifest.json` identifies the complete generation and payload hashes. No component interface or authoritative design document was changed for this one-off research audit.

## Figures

Selection is fixed by metadata: matched-profile panels use the first four alphabetical conditions, width/coverage/neuron figures show all eligible groups, and representative traces use the candidate rules above. Counts and temporal references are on each figure. Early source bins are explicitly distinguished from scoring support. Zero controls pooled across phase are descriptive. These figures are not a selection of only favorable outcomes.

![Condition and movement coverage](coverage.png)

![Matched versus coarse training profiles](matched-profiles.png)

![Reliability by predefined width](reliability-widths.png)

![Blank movement controls](zero-controls.png)

![All-unit versus regional consistency](neuron-region.png)

![Metadata-selected training traces](representative-traces.png)

![Source-only onset](onset.png)

![Complete event profiles](events.png)
"""
    (stage/'report.md').write_text(report,encoding='utf8')
    # Focused static HTML rendition of this report; no optional package needed.
    def inline(s):
        s=html.escape(s.replace('\\|','|'))
        s=re.sub(r'`([^`]+)`',r'<code>\1</code>',s)
        return re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',s)
    body=[];in_table=False
    for line in report.splitlines():
        if line.startswith('|'):
            if not in_table:
                body.append('<div class="table"><table>');in_table=True
            if re.fullmatch(r'[| :\-]+',line):
                continue
            body.append('<tr>'+''.join('<td>'+inline(c.strip())+'</td>' for c in re.split(r'(?<!\\)\|',line.strip('|')))+'</tr>')
            continue
        if in_table:
            body.append('</table></div>');in_table=False
        match=re.fullmatch(r'!\[(.*)\]\(([^)]+)\)',line)
        if match:
            body.append(f'<figure><img src="{html.escape(match[2])}" alt="{html.escape(match[1])}"><figcaption>{html.escape(match[1])}</figcaption></figure>')
        elif line.startswith('#'):
            level=len(line)-len(line.lstrip('#'));body.append(f'<h{level}>'+inline(line.lstrip('# '))+f'</h{level}>')
        elif line.strip():
            body.append('<p>'+inline(line)+'</p>')
    page='<!doctype html><html lang="en"><meta charset="utf-8"><title>VISpm matched response reliability</title><style>body{font:16px/1.55 system-ui,sans-serif;max-width:1180px;margin:40px auto;padding:0 24px;color:#18252e}h1,h2,h3{line-height:1.2}h2{margin-top:2em}code{overflow-wrap:anywhere;background:#eef2f4}table{border-collapse:collapse;font-size:13px}td{padding:7px 10px;border:1px solid #ccd4da}tr:first-child{font-weight:bold;background:#eaf0f3}.table{overflow:auto}img{width:100%;height:auto}figure{margin:32px 0}figcaption{color:#51616b}</style><body>'+''.join(body)+'</body></html>'
    (stage/'report.html').write_text(page,encoding='utf8')


def publish(stage):
    r=read(stage/'results.json')
    destination=BASE/r['diagnostic_generation']
    if destination.exists():
        raise FileExistsError(destination)
    if not (stage/'report.html').is_file() or not (stage/'report.md').is_file():
        raise ValueError('Final report missing')
    payloads={p.name:digest(p) for p in stage.iterdir() if p.is_file() and p.name!='manifest.json'}
    manifest=dict(schema_version=1,kind='response_reliability_audit',complete=True,
                  diagnostic_generation=r['diagnostic_generation'],dataset_generation_id=EXPECTED,
                  analysis_sha256=r['analysis_sha256'],report_builder_sha256=digest(__file__),
                  payload_sha256=payloads)
    (stage/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    # Rename fails on Windows if a competing publication created the destination.
    stage.rename(destination)
    print('Published',destination)


def enrich(r,data,stage):
    r['zero_events']={}
    for split in ('train','val'):
        zero=[x for x in data if x['split']==split and x['contrast']==0]
        for event in ('first_movement','response','feedback'):
            g=[];changes=[]
            for x in zero:
                if not np.isfinite(x[event]):
                    continue
                idx=int(np.argmin(abs(x['times']-x['stim_on']-x[event])))
                if idx<12 or idx+24>x['bins'] or not x['times'][0]-.5/60<=x['stim_on']+x[event]<=x['times'][-1]+.5/60:
                    continue
                g.append(x);changes.append(x['y'][idx:idx+12].mean(0)-x['y'][idx-12:idx].mean(0))
            changes=np.array(changes)
            r['zero_events'][f'{split}:{event}']=dict(n=len(g),trial_ids=[x['trial_id'] for x in g],
                 regional_post_minus_pre=bootstrap(changes.sum(1),f'zero:{split}:{event}') if g else None,
                 unit_post_minus_pre=changes.mean(0) if g else None)
    # Within-trial wheel association over each trial's full H=12 support.
    # Wheel increments derive from recorded-position interpolation, not visual motion.
    movement=[]
    for x in data:
        ep=np.arange(22,x['bins'])
        w=trailing(x['wheel_path'][:,None]*60,12,ep)[:,0]
        y=trailing(x['y'],12,ep).sum(1)
        movement.append(dict(split=x['split'],trial_id=x['trial_id'],zero=x['contrast']==0,n_endpoints=len(ep),
                             r=corr(w,y),wheel_signal='12-bin mean absolute edge-interpolated wheel velocity rad/s'))
    pd.DataFrame(movement).to_csv(stage/'wheel-associations.csv',index=False)
    r['wheel_association']={f'{split}:{zero}':dict(distribution=dist([a['r'] for a in movement if a['split']==split and a['zero']==zero]),
           trial_bootstrap=bootstrap([a['r'] for a in movement if a['split']==split and a['zero']==zero and np.isfinite(a['r'])],f'wheel:{split}:{zero}')) for split in ('train','val') for zero in (False,True)}
    r['aggregation']={}
    for key,entry in r['groups'].items():
        if not key.startswith('matched:') or entry['train']['n']<6:
            continue
        record={}
        for split in ('train','val'):
            g=[x for x in data if x['split']==split and x['trial_id'] in entry[split]['trial_ids']]
            if not g:
                continue
            # Fixed model-eligible within-stimulus reference, 200..400 vs 600..1000 ms.
            delta=np.array([x['y'][36:60].mean(0)-x['y'][12:24].mean(0) for x in g])
            mean=delta.mean(0)
            record[split]=dict(n=len(g),mean_unit_delta=mean,regional_delta=bootstrap(delta.sum(1),f'aggregate:{key}:{split}'),
                positive_units=int((mean>0).sum()),negative_units=int((mean<0).sum()),
                signed_to_absolute_delta_ratio=abs(mean.sum())/abs(mean).sum() if abs(mean).sum()>0 else None,
                top10_absolute_delta_share=np.sort(abs(mean))[-10:].sum()/abs(mean).sum() if abs(mean).sum()>0 else None)
            if len(g)>=6:
                pairs,_=partitions(len(g),f'aggregate:{key}:{split}')
                record[split]['unit_direction_agreement']=np.mean([(delta[l].mean(0)*delta[q].mean(0))>0 for l,q in pairs],axis=0)
        if 'val' in record and record['val']['n']>=3:
            record['cross_split_direction_agreement']=np.sign(record['train']['mean_unit_delta'])==np.sign(record['val']['mean_unit_delta'])
        r['aggregation'][key]=record
    # Cross-group dynamic contrasts within phase/movement, independent split check.
    r['profile_contrasts']=[]
    for side in ('left','right'):
        for movement in ('low','high'):
            for phase in ('ongoing','responded'):
                labels=[k for k,e in r['groups'].items() if k.startswith(f'matched:{side}:') and k.endswith(f'|{movement}|{phase}') and e['train']['n']>=6]
                for a,b in itertools.combinations(labels,2):
                    ea,eb=r['groups'][a],r['groups'][b]
                    va,vb=ea['val'],eb['val']
                    item=dict(a=a,b=b,train_n=[ea['train']['n'],eb['train']['n']],val_n=[va['n'],vb['n']],
                              train_rms_counts=float(np.sqrt(np.mean((ea['train']['widths'][12]['mean']-eb['train']['widths'][12]['mean'])**2))),
                              cross_split_status='available' if min(va['n'],vb['n'])>=3 else 'unavailable_fewer_than_3_per_validation_group')
                    item['cross_split_dynamic_contrast_r']=corr(ea['train']['widths'][12]['mean']-eb['train']['widths'][12]['mean'],va['widths'][12]['mean']-vb['widths'][12]['mean']) if min(va['n'],vb['n'])>=3 else None
                    r['profile_contrasts'].append(item)


def make_figures(stage,r,data,summary,coverage,neurons):
    def save(fig,name):
        fig.tight_layout(); fig.savefig(stage/name,dpi=140);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(15,7))
    for ax,split in zip(axs,('train','val')):
        f=coverage[(coverage.kind=='matched')&(coverage.split==split)]
        ax.barh(f.group,f.n);ax.set_title(f'{split}: complete first-second trial contributors');ax.set_xlabel('Whole trials, stimulus onset reference');ax.tick_params(axis='y',labelsize=6)
    save(fig,'coverage.png')
    # First four conditions alphabetically, independent of measured response.
    conditions=sorted(set(x['condition'] for x in data))[:4]
    fig,axs=plt.subplots(2,2,figsize=(13,8))
    t=(np.arange(22,60)+.5)/60
    for ax,c in zip(axs.flat,conditions):
        for key,entry in r['groups'].items():
            if key=='coarse:'+c or key.startswith('matched:'+c+'|'):
                a=entry['train']
                if a['n']:
                    ax.plot(t,a['widths'][12]['mean'],label=f"{key.split(':',1)[0]} {key.split('|',1)[-1]} n={a['n']}",lw=2 if key.startswith('coarse:') else 1)
        ax.set_title(c);ax.set_xlabel('Seconds after stimulus onset');ax.set_ylabel('Trailing 12-bin mean, counts/original bin');ax.legend(fontsize=6)
    save(fig,'matched-profiles.png')
    fig,axs=plt.subplots(1,2,figsize=(14,6))
    for ax,kind in zip(axs,('coarse','matched')):
        f=summary[(summary.kind==kind)&(summary.split=='train')&(summary.n>=6)]
        for key,g in f.groupby('group',sort=True):
            ax.plot(g.width*1000/60,g.mean_r,'o-',label=f'{key} n={g.n.iloc[0]}')
        ax.axhline(0,color='grey');ax.set_title(f'{kind}: train whole-trial split halves');ax.set_xlabel('Trailing width (ms), same endpoints 375..992 ms');ax.set_ylabel('Mean profile correlation');ax.legend(fontsize=5,ncol=2)
    save(fig,'reliability-widths.png')
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    for ax,split in zip(axs,('train','val')):
        for movement in ('low','high'):
            g=[x for x in data if x['split']==split and x['contrast']==0 and x['movement_class']==movement and x['bins']>=60]
            if g:
                a=np.mean([trailing(x['y'],12,np.arange(22,60)).sum(1) for x in g],axis=0)
                ax.plot(t,a,label=f"{movement} measured wheel n={len(g)}")
        ax.set_title(f'{split}: zero contrast, pooled sides/phases');ax.set_xlabel('Seconds after stimulus onset');ax.set_ylabel('12-bin mean counts/original bin');ax.legend()
    save(fig,'zero-controls.png')
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    for ax,kind in zip(axs,('coarse','matched')):
        f=neurons[(neurons.kind==kind)&(neurons.split=='train')&(neurons.width==12)]
        for k,(key,g) in enumerate(f.groupby('group',sort=True)):
            ax.scatter(np.full(len(g),k),g.mean_r,s=5,alpha=.25)
            ax.scatter(k,g.regional_mean_r.iloc[0],marker='D',color='black',s=25)
        groups=list(f.groupby('group',sort=True))
        ax.set_xticks(range(len(groups)),[f'{k} n={g.n.iloc[0]}' for k,g in groups],rotation=90,fontsize=6)
        ax.set_title(f'{kind}: all 99 units, 200 ms; black=region');ax.set_ylabel('Mean whole-trial split-half r');ax.axhline(0,color='grey')
    save(fig,'neuron-region.png')
    fig,axs=plt.subplots(len(r['candidates']),1,figsize=(13,10))
    for ax,c in zip(axs,r['candidates']):
        x=next(x for x in data if x['trial_id']==c['trial_id'])
        t=(np.arange(x['bins'])+.5)/60
        ax.plot(t,x['y'].sum(1),alpha=.35,label='raw regional counts/bin')
        ep=np.arange(11,x['bins']);ax.plot(t[ep],trailing(x['y'],12,ep).sum(1),label='12-bin mean counts/bin')
        twin=ax.twinx();twin.plot(x['obs_times']-x['stim_on'],x['az'],color='green',alpha=.6,label='reconstructed azimuth')
        twin.set_ylabel('Reconstructed degrees',color='green')
        ax.axvline(x['response'],color='red',ls=':',label='recorded response')
        ax.set_title(f"{c['category']}: train trial {x['trial_id']}, {x['condition']}; n=1");ax.set_xlabel('Seconds after stimulus onset');ax.legend(fontsize=7,loc='upper left')
    save(fig,'representative-traces.png')
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    for ax,split in zip(axs,('train','val')):
        for key,a in r['onset'].items():
            if key.startswith(split+':') and a['n']:
                c=key.split(':',1)[1]
                source=r['groups']['coarse:'+c][split]
                ax.plot((np.arange(60)+.5)/60,source['raw_regional_mean_first_second'],label=f"{c} n={a['n']}")
        ax.axvspan(0,11/60,color='grey',alpha=.15,label='first 11 source-only bins');ax.set_title(f'{split}: raw source onset, counts/bin');ax.set_xlabel('Seconds after stimulus onset');ax.legend(fontsize=6)
    save(fig,'onset.png')
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    for ax,split in zip(axs,('train','val')):
        for event in ('first_movement','response','feedback','visual_movement_onset'):
            a=r['events'][f'{event}:{split}']
            if a['n']:
                ax.plot(np.arange(-1,24)/60,a['widths'][12]['mean'],label=f"{event} n={a['n']}")
        ax.axvline(0,color='grey');ax.set_title(f'{split}: complete source event windows');ax.set_xlabel('Seconds relative to event (nearest neural center)');ax.set_ylabel('12-bin mean counts/original bin');ax.legend(fontsize=7)
    save(fig,'events.png')


if __name__=='__main__':
    main()
