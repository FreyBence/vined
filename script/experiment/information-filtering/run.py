"""Run independent Experiment 3 stages; default scores and preflights without training."""
import subprocess
import sys
from common import parser, settings, path, HERE, ROOT


def main():
    p=parser(__doc__)
    p.add_argument("--stages",nargs="+",choices=["score","preflight","train","evaluate","compare"],default=["score","preflight"])
    p.add_argument("--runs",nargs="+",choices=[f"{s}_{d:02}" for s in ("neuron_filtered","subregion_filtered") for d in (0,6)],
        default=[f"{s}_{d:02}" for s in ("neuron_filtered","subregion_filtered") for d in (0,6)])
    p.add_argument("--repeat",action="store_true");p.add_argument("--device")
    args=p.parse_args();settings(args)
    for action in dict.fromkeys(args.stages):
        runs=[None] if action in ("score","compare") else (["unfiltered_00","unfiltered_06"]+args.runs if action=="evaluate" else args.runs)
        for run in runs:
            command=[sys.executable,str(HERE / ("train.py" if action=="preflight" else action+".py")),"--experiment-config",str(path(args.experiment_config))]
            if action=="evaluate":
                command += ["--run",run]
                if args.device:command += ["--device",args.device]
            elif run:command += ["--run-config",str(HERE / "configs/training" / (run+".json"))]
            if action=="preflight":command += ["--preflight"]
            elif not args.repeat:command += ["--skip-completed"]
            result=subprocess.run(command,cwd=ROOT)
            if result.returncode:return result.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
