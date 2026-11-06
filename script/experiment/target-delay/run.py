"""Lightweight subprocess orchestration of independent experiment stages."""
import subprocess
import sys
from common import parser, settings, path, HERE, ROOT, read, atomic_json


def main():
    p=parser(__doc__)
    p.add_argument("--stages",nargs="+",choices=["preflight","train","evaluate","compare"],
                   default=["train","evaluate","compare"])
    p.add_argument("--runs",nargs="+",help="Configuration IDs, e.g. delay_00 delay_04")
    p.add_argument("--repeat",action="store_true",help="Repeat stages into fresh attempts instead of skipping completed stages")
    args=p.parse_args()
    exp,evaluation,out=settings(args)
    ids=args.runs or [f"delay_{d:02d}" for d in exp["delays"]]
    from common import name
    for run_id in ids:
        name(run_id)
        if not (HERE/"configs"/(run_id+".json")).is_file():
            raise ValueError(f"Unknown run configuration {run_id}")
    # Always resolve every selected run before optimization starts.
    stages=list(dict.fromkeys((["preflight"] if "train" in args.stages else [])+args.stages))
    for stage in stages:
        selected=[None] if stage=="compare" else ids
        for run_id in selected:
            script="train.py" if stage=="preflight" else stage+".py"
            command=[sys.executable,str(HERE/script),"--experiment-config",str(path(args.experiment_config))]
            if run_id:
                command += ["--run-config",str(HERE/"configs"/(run_id+".json"))]
            else:
                command += ["--run-ids",*ids]
            if stage=="preflight":
                command.append("--preflight")
            elif not args.repeat:
                command.append("--skip-completed")
            result=subprocess.run(command,cwd=ROOT)
            if result.returncode:
                print(f"FAILED: {stage} {run_id or 'comparison'} (exit {result.returncode})",file=sys.stderr)
                return result.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
