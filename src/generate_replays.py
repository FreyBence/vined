"""Batch compressed replay generation with session and current-trial progress."""

import argparse
from pathlib import Path
import sys
import traceback

from tqdm import tqdm

from prepare_replay import main as prepare_session
from utils.sessions import select_sessions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eid")
    parser.add_argument("--projection", choices=("on", "off"), default="on")
    parser.add_argument("--config", type=Path, default=Path("data/replay-config.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force-reload", action="store_true")
    args = parser.parse_args()
    eids = select_sessions(eid=args.eid)
    args.output.mkdir(parents=True, exist_ok=False)
    status = 0
    with tqdm(total=len(eids), desc="EIDs", unit="eid", position=0, dynamic_ncols=True) as progress:
        for eid in eids:
            progress.set_postfix_str(f"{eid} (preparing)")
            command = ["--config", str(args.config), "--output", str(args.output / eid),
                       "--generate", "--eid", eid, "--access-policy", "remote-allowed",
                       "--image-space", "mouse_view" if args.projection == "on" else "display"]
            if args.force_reload:
                command.append("--force-reload")
            try:
                code = prepare_session(command, progress_position=1)
            except SystemExit as exc:
                code = exc.code if isinstance(exc.code, int) else 1
                if code in (130, 143):
                    raise
            except Exception:
                tqdm.write(traceback.format_exc(), file=sys.stderr)
                code = 1
            status = max(status, code)
            progress.set_postfix_str(f"{eid} (status {code})", refresh=False)
            progress.update(1)
            tqdm.write(f"Replay returned status {code}: {eid}")
    return status


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
