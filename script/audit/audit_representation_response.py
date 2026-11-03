"""Scenario 11 descriptive scientific audit; no fitting, inference, or test analysis.

Run from the checkout with the project Python. Outputs refuse replacement.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile

os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits

from audit_paths import ROOT, AUDIT_ROOT

BASE = AUDIT_ROOT / 'scenario-11/002'
OUT = BASE / 'representation-response-001'
EID = '0cad7ea8-8e6c-4ad1-a5c5-53fbb2df1a63'
RNG = np.random.default_rng(20261027)


def read(p):
    return json.loads(Path(p).read_text(encoding='utf8'))


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def corr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.corrcoef(a, b)[0, 1]) if len(a)>2 and np.std(a)>1e-12 and np.std(b)>1e-12 else np.nan


def dist(a):
    a = np.asarray(a); a = a[np.isfinite(a)]
    return dict(n=len(a), mean=float(a.mean()) if len(a) else None,
                quantiles=np.quantile(a,[0,.05,.25,.5,.75,.95,1]).tolist() if len(a) else [])


def boot(a):
    a=np.asarray(a); a=a[np.isfinite(a)]
    return dict(n=len(a), mean=float(a.mean()) if len(a) else None,
                ci95=np.quantile([RNG.choice(a,len(a)).mean() for _ in range(1000)],[.025,.975]).tolist() if len(a) else [])


def clean(x):
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)): return [clean(v) for v in x]
    if isinstance(x,np.generic): return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x): return None
    return x


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    frozen=read(BASE/'temporal-001/results.json'); dp=Path(frozen['dataset_path']); dm=read(dp/'manifest.json')
    target=frozen['regional_targets'][EID]; table=pd.read_parquet(dp/f'units/{EID}.parquet')
    identities=pd.DataFrame(target['ordered_units'])
    keys=['source_unit_id','pid','probe','collection','revision']
    cols=[]
    for row in identities.to_dict('records'):
        hit=np.flatnonzero(np.logical_and.reduce([table[k].eq(row[k]).to_numpy() for k in keys]))
        if len(hit)!=1: raise ValueError('Ambiguous source identity')
        cols.append(int(hit[0]))
    selected=table.iloc[cols].reset_index(drop=True)
    if selected.source_unit_id.tolist()!=identities.source_unit_id.tolist(): raise ValueError('Unit order differs')
    fp=ROOT/f'output/visual_features/{EID}_visual_clip.npz'
    allowed={s['trial_id'] for s in dm['samples'] if s['split'] in ('train','val')}
    source={t:[] for t in allowed}
    with zipfile.ZipFile(fp) as z:
        fm=json.loads(z.read('manifest.json')); definition=json.loads(z.read('definition.json'))
        if fm['generation_id']!=dm['samples'][0]['metadata']['alignment']['visual_completion']['generation_id']: raise ValueError('Feature generation mismatch')
        # Compressed array decoded for indexed access; test rows never used in statistics.
        features=np.load(io.BytesIO(z.read('features.npy')),allow_pickle=False)
        for line in z.open('records.jsonl'):
            r=json.loads(line)
            if r['kind']=='observation' and r['metadata']['trial_id'] in allowed and r['feature_index'] is not None:
                source[r['metadata']['trial_id']].append(r)
    oldconfig=read(ROOT/'output/runs/scenario-15/006/config.json'); op=Path(oldconfig['dataset']['dataset_path']); om=read(op/'manifest.json')
    oldindex={s['trial_id']:(i,s) for i,s in enumerate(om['samples']) if s['split'] in ('train','val')}
    records=[]; historical=[]; inputs={}; durations=[]
    for i,s in enumerate(dm['samples']):
        if s['split'] not in ('train','val'): continue
        path=dp/f'samples/{i:06d}.npz'
        if digest(path)!=dm['files'][f'samples/{i:06d}.npz']: raise ValueError('Dataset digest mismatch')
        with np.load(path,allow_pickle=False) as a: a={k:a[k] for k in a.files}
        n=s['sequence_length']; v=a['visual'][:n].astype(float); y=a['neural'][:n,cols].astype(float)
        if not a['temporal_mask'][:n].all() or not a['neuron_mask'][cols].all(): raise ValueError('Unexpected invalid support')
        fr=next(r for r in frozen['records'] if r['trial_id']==s['trial_id'] and r['split']==s['split'])
        predpath=BASE/'temporal-001/predictions'/f"{fr['array_key']}.npz"
        payloads={k.replace('\\','/'):v for k,v in frozen['payload_hashes'].items()}
        if digest(predpath)!=payloads[str(predpath.relative_to(BASE/'temporal-001')).replace('\\','/')]: raise ValueError('Prediction digest mismatch')
        with np.load(predpath) as a: p=a['expected'][:,0]; observed=a['observed'][:,0]; valid=a['scoring_mask'][:,0]
        if not np.array_equal(y.sum(1),observed) or not np.array_equal(valid,np.arange(n)>=11): raise ValueError('Regional sum/support mismatch')
        sr=source[s['trial_id']]; sv=features[[r['feature_index'] for r in sr]].astype(float)
        st=np.array([r['metadata']['session_time'] for r in sr]); al=s['metadata']['alignment']; rs=al['resampling']
        l=np.array(rs['left_source_indices']); rr=np.array(rs['right_source_indices']); w=np.array(rs['interpolation_weights'])
        raw=(1-w[:,None])*sv[l]+w[:,None]*sv[rr]; reconstructed=raw/np.linalg.norm(raw,axis=1)[:,None]
        if not np.allclose(reconstructed,v,atol=1e-7,rtol=0): raise ValueError('Interpolation mismatch')
        meta=al['visual_definition']['trial']; condition=f"{meta['side']}:{meta['parameters']['contrast']:g}"
        d=np.r_[np.nan,np.linalg.norm(np.diff(v,axis=0),axis=1)]
        row=dict(trial_id=s['trial_id'],split=s['split'],sample_id=s['sample_id'],bins=n,condition=condition,
                 delta_mean=np.mean(d[1:]),within_trace=np.var(v,axis=0).sum(),duration=n/60,
                 source_exact_repeat_fraction=np.mean([x['metadata']['rgb_sha256']==z['metadata']['rgb_sha256'] for x,z in zip(sr[:-1],sr[1:])]),
                 source_delta_mean=np.linalg.norm(np.diff(sv,axis=0),axis=1).mean(),
                 future_source_fraction=np.mean(st[rr]>np.array(rs['query_times'])),future_lead_ms_mean=np.mean((st[rr]-rs['query_times']))*1000,
                 renormalization_norm_deficit_max=np.max(1-np.linalg.norm(raw,axis=1)),
                 raw_delta_mean=np.linalg.norm(np.diff(raw,axis=0),axis=1).mean(),
                 mean_y=y[11:].sum(1).mean(),mean_p=p[11:].mean(),
                 p_elapsed_r=corr(np.arange(n)[11:],p[11:]),y_elapsed_r=corr(np.arange(n)[11:],y[11:].sum(1)),
                 delta_prediction_r=corr(d[11:],p[11:]))
        if s['trial_id'] in oldindex:
            oi,old=oldindex[s['trial_id']]
            if old['split']==s['split']:
                with np.load(op/f'samples/{oi:06d}.npz') as a:
                    same=np.array_equal(a['neural'],np.load(path)['neural']) and np.array_equal(a['physical_timestamps'],np.load(path)['physical_timestamps'])
                    ov=a['visual'].astype(float)
                historical.append(dict(trial_id=s['trial_id'],split=s['split'],same_counts_times=same,
                    old_delta_mean=np.linalg.norm(np.diff(ov,axis=0),axis=1).mean(),new_delta_mean=row['delta_mean'],
                    old_unchanged=np.mean(np.linalg.norm(np.diff(ov,axis=0),axis=1)<=1e-6)))
        stages=np.array([r['metadata']['stimulus_state']['stage'] for r in sr]); az=np.array([r['metadata']['stimulus_state']['azimuth_deg'] for r in sr])
        sd=np.linalg.norm(np.diff(sv,axis=0),axis=1); da=np.abs(np.diff(az))
        row.update(source_motion_delta_r=corr(da,sd),source_motion_pairs=int((da>1e-6).sum()),source_stages=sorted(set(stages)),
                   source_unchanged_pixels_nonzero_feature=int(sum(sr[j]['metadata']['rgb_sha256']==sr[j+1]['metadata']['rgb_sha256'] and sd[j]>1e-6 for j in range(len(sd)))))
        inputs[s['trial_id']]=dict(s=s,v=v,y=y,p=p,d=d,source=sr,sv=sv,condition=condition)
        records.append(row)
    del features
    pd.DataFrame(records).to_csv(OUT/'trials.csv',index=False)
    pd.DataFrame(historical).to_csv(OUT/'historical-alignment.csv',index=False)
    results=dict(dataset_generation_id=dm['generation_id'],checkpoint=frozen['checkpoint'],session=EID,region='VISpm',
        source_columns_in_persisted_280=cols,ordered_units=target['ordered_units'],feature_generation_id=fm['generation_id'],
        representation={k:v for k,v in definition['representation'].items() if k!='packages'},
        memberships={split:[r['trial_id'] for r in records if r['split']==split] for split in ('train','val')},splits={})
    fig,axs=plt.subplots(2,3,figsize=(15,8))
    neuronstats=[]; lagrows=[]; profiles={}
    for split,color in [('train','tab:blue'),('val','tab:orange')]:
        items=[x for x in inputs.values() if x['s']['split']==split]; rows=[r for r in records if r['split']==split]
        vs=[x['v'] for x in items]; ys=[x['y'][11:] for x in items]; ds=np.concatenate([x['d'][1:] for x in items]); yc=np.concatenate([y-y.mean(0) for y in ys]); yy=np.concatenate(ys)
        means=np.array([v.mean(0) for v in vs]); weights=np.array([len(v) for v in vs]); grand=np.average(means,axis=0,weights=weights)
        within=sum(((v-v.mean(0))**2).sum(0) for v in vs)/weights.sum(); between=np.average((means-grand)**2,axis=0,weights=weights)
        sampled=np.concatenate([(v-v.mean(0))[::8] for v in vs]); ev=np.linalg.eigvalsh(sampled.T@sampled/len(sampled)); ev=np.maximum(ev,0)
        per_neuron=dict(rate_hz=yy.mean(0)*60,variance=yy.var(0),zero_fraction=(yy==0).mean(0),spike_share=yy.sum(0)/yy.sum(),centered_covariance_contribution=(yc*(yc.sum(1)[:,None])).mean(0))
        for j,u in enumerate(target['ordered_units']): neuronstats.append(dict(split=split,column=j,source_unit_id=u['source_unit_id'],**{k:float(v[j]) for k,v in per_neuron.items()}))
        active=yy.std(0)>0; cc=np.corrcoef(yc[:,active],rowvar=False); pairs=cc[np.triu_indices(len(cc),1)]
        # Fixed response positions t=12..T-1, identical for every lag 0..11.
        lag={}
        for signal in ('count','count_difference','detrended_count'):
            curves=[]; curves_s=[]
            for x in items:
                t=np.arange(12,len(x['v'])); y=x['y'].sum(1); sig=y if signal!='count_difference' else np.r_[0,np.diff(y)]
                visual=np.array([x['d'][t-k] for k in range(12)])
                if signal=='detrended_count':
                    # Descriptive linear trend removal, not a predictive model.
                    z=t-t.mean(); sig=sig[t]-sig[t].mean(); sig-=z*np.dot(z,sig)/np.dot(z,z)
                    visual-=visual.mean(1)[:,None];visual-=np.outer(visual@z/np.dot(z,z),z)
                else: sig=sig[t]
                curves.append([corr(v,sig) for v in visual])
                # Half-trial circular reference preserves each sequence's autocorrelation;
                # null: no privileged trial-local temporal phase at these lags.
                curves_s.append([corr(np.roll(v,len(v)//2),sig) for v in visual])
            curves=np.array(curves); curves_s=np.array(curves_s)
            lag[signal]=[dict(lag_bins=k,delay_ms=k*1000/60,**boot(curves[:,k]),shift_reference=boot(curves_s[:,k])) for k in range(12)]
            for j,x in enumerate(items):
                for k in range(12): lagrows.append(dict(split=split,trial_id=x['s']['trial_id'],signal=signal,lag_bins=k,r=curves[j,k]))
        autocorr=[]
        for k in [1,3,6,11,30,60]:
            vals=[]; nn=[]
            for x in items:
                v=x['v']-x['v'].mean(0)
                if len(v)>k:
                    den=np.sqrt((v[:-k]**2).sum()*(v[k:]**2).sum()); vals.append(float((v[:-k]*v[k:]).sum()/den) if den>1e-15 else np.nan)
                    nn.append(corr(x['y'][11:-k].sum(1),x['y'][11+k:].sum(1)))
            autocorr.append(dict(lag=k,feature=boot(vals),neural=boot(nn)))
        sr=dict(trials=len(items),all_bins=int(weights.sum()),eligible_bins=len(yy),delta=dist(ds),cosine=dist(1-ds**2/2),
                exact_unchanged_fraction=float(np.mean(ds==0)),approx_unchanged_fraction=float(np.mean(ds<=1e-6)),
                within_trace=within.sum(),between_trace=between.sum(),within_fraction=within.sum()/(within.sum()+between.sum()),
                within_dimension_variance=within,participation_ratio=ev.sum()**2/(ev@ev),pca95_dimensions=int(np.searchsorted(np.cumsum(ev[::-1])/ev.sum(),.95)+1),
                effective_dimension_support='every eighth bin within each trial, trial-centered; descriptive subsampling',
                autocorrelation=autocorr,lag=lag,source_repeat=boot([r['source_exact_repeat_fraction'] for r in rows]),
                motion_delta_correlation=boot([r['source_motion_delta_r'] for r in rows]),
                duration_delta_r=corr([r['duration'] for r in rows],[r['delta_mean'] for r in rows]),
                duration_mean_p_r=corr([r['duration'] for r in rows],[r['mean_p'] for r in rows]),
                duration_mean_y_r=corr([r['duration'] for r in rows],[r['mean_y'] for r in rows]),
                elapsed_prediction_r=boot([r['p_elapsed_r'] for r in rows]),elapsed_observed_r=boot([r['y_elapsed_r'] for r in rows]),
                unit_rate_hz=dist(per_neuron['rate_hz']),units_below_1hz=int((per_neuron['rate_hz']<1).sum()),
                unit_pair_centered_correlation=dist(pairs),top10_spike_share=float(np.sort(per_neuron['spike_share'])[-10:].sum()),
                top1_spike_share=float(max(per_neuron['spike_share'])),aggregate_centered_variance=float(yc.sum(1).var()),
                sum_individual_centered_variance=float(yc.var(0).sum()),mean_count=float(yy.sum(1).mean()),
                count_distribution=dist(yy.ravel()),regional_count_distribution=dist(yy.sum(1)))
        pd.DataFrame({'dimension':np.arange(768),'within_variance':within,'between_variance':between}).to_csv(OUT/f'{split}-feature-variance.csv',index=False)
        # Matching is coarse side/contrast, never asserted to be exact repeats.
        groups={}
        for x in items:
            if len(x['y'])>=60: groups.setdefault(x['condition'],[]).append(x)
        prof={}; reliability=[]
        for condition,xs in sorted(groups.items()):
            if len(xs)<6: continue
            stack=np.stack([x['y'][11:60] for x in xs]); mean=stack.mean(0); centered=mean-mean.mean(0)
            splitrs=[]
            for _ in range(200):
                order=RNG.permutation(len(stack));half=len(order)//2
                a=stack[order[:half]].mean(0).sum(1); b=stack[order[half:]].mean(0).sum(1)
                splitrs.append(corr(a,b))
            prof[condition]=dict(trials=len(xs),trial_ids=[x['s']['trial_id'] for x in xs],mean=mean,
                regional_mean=mean.sum(1),cancellation_ratio=float(np.linalg.norm(centered.sum(1))/np.linalg.norm(centered,axis=0).sum()) if np.linalg.norm(centered,axis=0).sum()>0 else None,
                regional_split_half=dist(splitrs))
        profiles[split]=prof;sr['coarse_conditions']=prof
        results['splits'][split]=sr
        axs[0,0].hist(ds,bins=80,density=True,histtype='step',label=split,color=color)
        axs[0,1].plot(range(12),[z['mean'] for z in lag['count']],label=split,color=color)
        axs[0,1].fill_between(range(12),[z['ci95'][0] for z in lag['count']],[z['ci95'][1] for z in lag['count']],alpha=.15,color=color)
        axs[0,2].plot(np.arange(1,100),np.cumsum(np.sort(per_neuron['spike_share'])[::-1]),label=split,color=color)
        axs[1,0].scatter([r['duration'] for r in rows],[r['delta_mean'] for r in rows],s=8,label=split,color=color)
        axs[1,1].plot([z['lag'] for z in autocorr],[z['feature']['mean'] for z in autocorr],label=split,color=color)
        axs[1,2].scatter([r['duration'] for r in rows],[r['mean_p'] for r in rows],s=8,label=split,color=color)
    for ax,title in zip(axs.ravel(),['Adjacent CLIP distance (all valid bins)','Equal-trial r: earlier CLIP change / later counts','Cumulative fraction of spikes by rate rank','Trial duration (s) / mean CLIP distance','Trial-centered feature autocorrelation / lag bins','Trial duration (s) / predicted trial mean']):
        ax.set_title(title,fontsize=9);ax.legend()
    fig.tight_layout();fig.savefig(OUT/'overview.png',dpi=150);plt.close(fig)
    pd.DataFrame(neuronstats).to_csv(OUT/'neurons.csv',index=False);pd.DataFrame(lagrows).to_csv(OUT/'lags-by-trial.csv',index=False)
    # Representative selection by duration quantiles, independent of correlation.
    trainitems=sorted([x for x in inputs.values() if x['s']['split']=='train'],key=lambda x:len(x['v']))
    examples=[trainitems[int(q*(len(trainitems)-1))] for q in [.25,.5,.75]]
    fig,axs=plt.subplots(3,2,figsize=(14,9))
    for row,x in enumerate(examples):
        t=np.arange(len(x['v']))/60
        axs[row,0].plot(t,x['d']);axs[row,0].set_title(f"Train trial {x['s']['trial_id']}: CLIP change norm")
        axs[row,1].plot(t,x['y'].sum(1),alpha=.45,label='regional counts');axs[row,1].plot(t[11:],x['p'][11:],label='frozen expected')
        axs[row,1].set_title(f"{x['condition']}; counts per 16.67 ms");axs[row,1].legend()
    fig.tight_layout();fig.savefig(OUT/'representative-trials.png',dpi=150);plt.close(fig)
    # Condition profiles and per-neuron deviations use identical t=11..59 support.
    common=sorted(set(profiles['train'])&set(profiles['val']))
    fig,axs=plt.subplots(max(len(common),1),2,figsize=(13,max(3,len(common)*2.5)),squeeze=False)
    confirmations={}
    for row,c in enumerate(common):
        a=profiles['train'][c];b=profiles['val'][c];t=(np.arange(11,60)+.5)/60
        axs[row,0].plot(t,a['regional_mean'],label=f"train n={a['trials']}");axs[row,0].plot(t,b['regional_mean'],label=f"val n={b['trials']}");axs[row,0].set_title(c);axs[row,0].legend()
        im=axs[row,1].imshow((a['mean']-a['mean'].mean(0)).T,aspect='auto',cmap='coolwarm',vmin=-.15,vmax=.15);axs[row,1].set_title('Train per-unit deviations from profile mean');fig.colorbar(im,ax=axs[row,1])
        confirmations[c]=dict(train_trials=a['trials'],val_trials=b['trials'],regional_profile_r=corr(a['regional_mean'],b['regional_mean']),
            neuron_profile_r=[corr(a['mean'][:,j],b['mean'][:,j]) for j in range(99)])
    fig.tight_layout();fig.savefig(OUT/'condition-profiles.png',dpi=150);plt.close(fig)
    results['condition_confirmation']=confirmations
    results['source_hashes']={str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in [Path(__file__),dp/'manifest.json',BASE/'temporal-001/results.json',ROOT/'output/runs/scenario-11/002/model_best.pt',fp]}
    (OUT/'results.json').write_text(json.dumps(clean(results),indent=2,allow_nan=False),encoding='utf8')
    print('Audit written:',OUT)


if __name__=='__main__':
    with threadpool_limits(limits=4): main()
