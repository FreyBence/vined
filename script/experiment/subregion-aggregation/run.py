"""Run independent Experiment 2 stages; the default performs preflight only."""
import subprocess
import sys
from common import parser, settings, path, HERE, ROOT, read, atomic_json


def main():
    p = parser(__doc__)
    p.add_argument("--stages", nargs="+", choices=["preflight", "train", "evaluate", "compare"], default=["preflight"])
    p.add_argument("--runs", nargs="+", choices=["subregion_00", "subregion_06"], default=["subregion_00", "subregion_06"])
    p.add_argument("--repeat", action="store_true")
    args = p.parse_args()
    exp, evaluation, out = settings(args)
    stages = list(dict.fromkeys((["preflight"] if "train" in args.stages else [])+args.stages))
    for stage in stages:
        for run in ([None] if stage == "compare" else args.runs):
            script = "train.py" if stage == "preflight" else stage+".py"
            command = [sys.executable, str(HERE / script), "--experiment-config", str(path(args.experiment_config))]
            if run:
                command += ["--run-config", str(HERE / "configs/training" / (run+".json"))]
            if stage == "preflight":
                command += ["--preflight"]
            elif not args.repeat:
                command += ["--skip-completed"]
            result = subprocess.run(command, cwd=ROOT)
            if result.returncode:
                return result.returncode
    if "preflight" in stages:
        from common import fingerprint
        snapshots = {r: read(out / "runs" / r / "resolved.json") for r in args.runs}
        atomic_json(out / "experiment-manifest.json", dict(schema_version=1, experiment=exp,
            runs={r: dict(resolved=str(out / "runs" / r / "resolved.json"), identity=fingerprint(s)) for r, s in snapshots.items()},
            regional_sources=next(iter(snapshots.values()))["support"]["regional_sources"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
