"""Compare all candidate lags on the same trials, including paired uncertainty."""
import importlib.util
from pathlib import Path
import json
import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location('audit',Path(__file__).with_name('audit_representation_response.py'))
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


def main():
    out=a.OUT/'lag-comparison.json'
    if out.exists(): raise FileExistsError(out)
    rows=pd.read_csv(a.OUT/'lags-by-trial.csv');result={}
    for split in ('train','val'):
        result[split]={}
        for signal in rows.signal.unique():
            curve=rows[(rows.split==split)&(rows.signal==signal)].pivot(index='trial_id',columns='lag_bins',values='r').dropna()
            samples=curve.to_numpy();summary=[]
            draws=a.RNG.integers(0,len(samples),size=(2000,len(samples)))
            mean_draws=samples[draws].mean(1)
            for k in range(12):
                summary.append(dict(lag=k,trials=len(curve),mean=samples[:,k].mean(),
                    ci95=np.quantile(mean_draws[:,k],[.025,.975]),
                    paired_difference_from_lag0=samples[:,k].mean()-samples[:,0].mean(),
                    paired_difference_ci95=np.quantile(mean_draws[:,k]-mean_draws[:,0],[.025,.975])))
            result[split][signal]=dict(trial_ids=curve.index.tolist(),lags=summary,
                bootstrap_argmax_counts=np.bincount(mean_draws.argmax(1),minlength=12),
                interpretation='Same response bins t=12..T-1 at every lag; same finite-correlation trials at every lag; bootstrap whole trials; pointwise intervals, not multiplicity-corrected; descriptive argmax stability, not physiological latency')
    result['script_sha256']=a.digest(__file__)
    out.write_text(json.dumps(a.clean(result),indent=2,allow_nan=False),encoding='utf8')
    print(out)


if __name__=='__main__': main()
