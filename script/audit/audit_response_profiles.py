"""Supplement Scenario 11 audit with trial-local descriptive response profiles."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

spec=importlib.util.spec_from_file_location('audit',Path(__file__).with_name('audit_representation_response.py'))
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
OUT=a.OUT


def smooth(x,w):
    return np.stack([np.convolve(x[:,j],np.ones(w)/w,'valid') for j in range(x.shape[1])],axis=1)


def main():
    if (OUT/'response-profiles.json').exists(): raise FileExistsError('Supplement exists')
    r=a.read(OUT/'results.json'); dm=a.read(Path(a.read(a.BASE/'temporal-001/results.json')['dataset_path'])/'manifest.json')
    dp=Path(a.read(a.BASE/'temporal-001/results.json')['dataset_path']); units=r['ordered_units'];cols=r['source_columns_in_persisted_280']
    data=[]; source_hashes={};frame_examples=[]; image_changes=[]
    for i,s in enumerate(dm['samples']):
        if s['split'] not in ('train','val'): continue
        with np.load(dp/f'samples/{i:06d}.npz') as z: y=z['neural'][:,cols].astype(float); v=z['visual'].astype(float)
        meta=s['metadata']['alignment']['visual_definition']['trial'];c=f"{meta['side']}:{meta['parameters']['contrast']:g}"
        rp=a.ROOT/f'output/visual_replays/{a.EID}/trials/{s["trial_id"]}'
        # Select only authorized train/validation replay metadata and images.
        obs=[json.loads(line) for line in (rp/'observations.jsonl').read_text().splitlines()]
        az=np.array([z['stimulus_state']['azimuth_deg'] for z in obs]); mov=np.flatnonzero(np.abs(np.diff(az))>1e-6)+1
        event=int(mov[0]) if len(mov) else None
        x=dict(s=s,y=y,v=v,condition=c,event=event,meta=meta);data.append(x)
        source_hashes[str((rp/'observations.jsonl').relative_to(a.ROOT)).replace('\\','/')]=a.digest(rp/'observations.jsonl')
    results=dict(methods=dict(seed=20261027,condition='recorded side plus contrast; not exact visual repeats',
        profiles='positions 11..59 for eligible profiles; positions 0..59 separately labeled source-only onset analysis',
        uncertainty='1000 whole-trial bootstrap resamples for equal-trial lag summaries; repeated random trial-half partitions are descriptive distributions, not confidence intervals',
        smoothing='trailing complete windows within trial, counts per original bin',
        transition='first reconstructed azimuth change >1e-6 degrees, source schedule mapped to same-index neural bin; diagnostic only; prior 6 bins baseline, offsets -6..11',
        lag='t=12..T-1 for all lags; positive k associates ||V[t-k]-V[t-k-1]|| with Y[t]; sample correlations center each pair'),splits={},condition_confirmation={},source_hashes=source_hashes)
    profiles={}; perunit=[];transition_profiles={}
    fig,axs=plt.subplots(2,2,figsize=(13,8))
    for si,split in enumerate(('train','val')):
        xs=[x for x in data if x['s']['split']==split]; conditions=sorted(set(x['condition'] for x in xs));profiles[split]={}
        neuronlags=[]; reglags=[]; durationgroups={}
        trainlengths=[len(x['y']) for x in data if x['s']['split']=='train'];cut=np.quantile(trainlengths,[1/3,2/3])
        for x in xs:
            y=x['y'];v=x['v'];d=np.r_[0,np.linalg.norm(np.diff(v,axis=0),axis=1)];t=np.arange(12,len(y))
            curve=np.empty((12,99))
            for k in range(12):
                sig=d[t-k]; yc=y[t]-y[t].mean(0);dc=sig-sig.mean();den=np.linalg.norm(yc,axis=0)*np.linalg.norm(dc)
                curve[k]=np.divide(dc@yc,den,out=np.full(99,np.nan),where=den>1e-12)
            neuronlags.append(curve);reglags.append([a.corr(d[t-k],y[t].sum(1)) for k in range(12)])
            durationgroups.setdefault(int(np.searchsorted(cut,len(y))),[]).append(reglags[-1])
        nl=np.array(neuronlags);rl=np.array(reglags)
        for j,u in enumerate(units):
            values=nl[:,:,j];means=np.nanmean(values,axis=0) if np.isfinite(values).any() else np.full(12,np.nan)
            perunit.append(dict(split=split,source_unit_id=u['source_unit_id'],lag_curve=means,contributing_trials=np.isfinite(values).sum(0),
                                exploratory_max_lag=int(np.nanargmax(means)) if np.isfinite(means).any() else None))
        sr=dict(duration_cut_bins=cut,duration_stratified_lag={k:[a.boot(np.array(v)[:,j]) for j in range(12)] for k,v in durationgroups.items()},conditions={})
        for c in conditions:
            group=[x for x in xs if x['condition']==c and len(x['y'])>=60]
            if len(group)<2:
                sr['conditions'][c]=dict(trials=len(group),unavailable='fewer than two full-first-second trials');continue
            stack=np.stack([x['y'][11:60] for x in group]);full=np.stack([x['y'][:60] for x in group]);mean=stack.mean(0)
            profiles[split][c]=mean
            cr=dict(trials=len(group),trial_ids=[x['s']['trial_id'] for x in group],scales={},source_onset_mean=full.mean(0),eligible_mean=mean)
            for w in (1,3,6,12):
                smooth_stack=np.stack([smooth(y,w) for y in stack]);rs=[];nr=[]
                for _ in range(200):
                    order=a.RNG.permutation(len(stack));half=len(order)//2
                    am=smooth_stack[order[:half]].mean(0);bm=smooth_stack[order[half:]].mean(0)
                    rs.append(a.corr(am.sum(1),bm.sum(1)))
                    nr.append([a.corr(am[:,j],bm[:,j]) for j in range(99)])
                cr['scales'][w]=dict(regional_split_half=a.dist(rs),per_neuron_split_half_mean=np.nanmean(nr,axis=0),positions=len(smooth_stack[0]))
            sr['conditions'][c]=cr
        # First reconstructed movement, identical pre/post support and trial contribution.
        events=[x for x in xs if x['event'] is not None and x['event']>=17 and x['event']+11<len(x['y'])]
        if events:
            trajectories=np.stack([x['y'][x['event']-6:x['event']+12] for x in events]);baseline=trajectories[:,:6].mean(1)
            deviations=trajectories-baseline[:,None,:];mean=deviations.mean(0)
            sr['transition']=dict(trials=len(events),trial_ids=[x['s']['trial_id'] for x in events],offset_bins=list(range(-6,12)),mean_neuron_deviation=mean,
                mean_regional_deviation=mean.sum(1),per_neuron_post_0_11=mean[6:].mean(0),
                positive_units=int((mean[6:].mean(0)>0).sum()),negative_units=int((mean[6:].mean(0)<0).sum()),
                signed_to_absolute_post_ratio=abs(mean[6:].mean(0).sum())/abs(mean[6:].mean(0)).sum(),
                regional_post_change=a.boot(deviations[:,6:].sum(2).mean(1)))
            transition_profiles[split]=mean
            axs[0,si].plot(np.arange(-6,12)*1000/60,mean.sum(1));axs[0,si].axvline(0,color='k',ls=':');axs[0,si].set_title(f'{split}: first movement, n={len(events)}')
            axs[1,si].imshow(mean.T,aspect='auto',cmap='coolwarm',vmin=-.05,vmax=.05);axs[1,si].set_title('Mean unit deviations, counts/bin')
        results['splits'][split]=sr
    for c in sorted(set(profiles['train'])&set(profiles['val'])):
        am,bm=profiles['train'][c],profiles['val'][c]
        results['condition_confirmation'][c]={w:dict(regional_profile_r=a.corr(smooth(am,w).sum(1),smooth(bm,w).sum(1)),
            per_neuron_profile_r=[a.corr(smooth(am,w)[:,j],smooth(bm,w)[:,j]) for j in range(99)]) for w in (1,3,6,12)}
    if len(transition_profiles)==2:
        results['transition_confirmation']=dict(regional_r=a.corr(transition_profiles['train'].sum(1),transition_profiles['val'].sum(1)),
            neuron_r=[a.corr(transition_profiles['train'][:,j],transition_profiles['val'][:,j]) for j in range(99)])
    fig.tight_layout();fig.savefig(OUT/'movement-responses.png',dpi=150);plt.close(fig)
    # Frame examples are duration quartiles, fixed independently of response strength.
    train=sorted([x for x in data if x['s']['split']=='train'],key=lambda x:len(x['y']))
    examples=[train[int(q*(len(train)-1))] for q in (.25,.5,.75)]
    fig,axs=plt.subplots(3,3,figsize=(12,9))
    for row,x in enumerate(examples):
        rp=a.ROOT/f'output/visual_replays/{a.EID}/trials/{x["s"]["trial_id"]}'
        obs=[json.loads(l) for l in (rp/'observations.jsonl').read_text().splitlines()]
        for col,idx in enumerate([0,int((len(obs)-1)/2),len(obs)-2]):
            path=rp/'images'/f'{idx}.npz'
            with np.load(path) as z: rgb=z[z.files[0]]
            if hashlib_rgb(rgb)!=obs[idx]['rgb_sha256']: raise ValueError('Replay pixel hash mismatch')
            axs[row,col].imshow(rgb);axs[row,col].axis('off');axs[row,col].set_title(f"Trial {x['s']['trial_id']} frame {idx}; {x['condition']}",fontsize=9)
            frame_examples.append(dict(trial_id=x['s']['trial_id'],frame_index=idx,path=str(path.relative_to(a.ROOT)),sha256=a.digest(path),pixel_sha256=hashlib_rgb(rgb)))
    fig.tight_layout();fig.savefig(OUT/'source-frames.png',dpi=130);plt.close(fig)
    results['frame_examples']=frame_examples
    results['per_neuron_lag']=perunit
    results['script_sha256']=a.digest(__file__)
    (OUT/'response-profiles.json').write_text(json.dumps(a.clean(results),indent=2,allow_nan=False),encoding='utf8')
    print('Supplement written',OUT)


def hashlib_rgb(rgb):
    import hashlib
    return hashlib.sha256(rgb.tobytes()).hexdigest()


if __name__=='__main__': main()
