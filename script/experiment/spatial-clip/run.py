"""Run the isolated accepted E2 A/B/G experiment without changing E0 guards."""

import argparse
import logging

import evaluate
import prepare
import training
from common import *


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "qualify",
            "extract",
            "qualify-peripheral",
            "derive",
            "preflight",
            "train",
            "validate",
            "freeze",
            "test",
            "report",
            "all",
        ],
    )
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.arm is not None and args.command not in (
        "derive",
        "train",
        "validate",
        "test",
    ):
        parser.error("--arm applies only to derive/train/validate/test")
    arms = [args.arm] if args.arm else ARMS
    if args.command in ("qualify", "all"):
        prepare.qualify()
    if args.command in ("extract", "all"):
        prepare.extract()
    if args.command in ("qualify-peripheral", "all"):
        prepare.qualify_peripheral()
    if args.command in ("derive", "all"):
        for arm in arms:
            prepare.derive(arm)
    if args.command in ("preflight", "all"):
        training.preflight()
    if args.command in ("train", "all"):
        for arm in arms:
            training.train(arm)
    if args.command in ("validate", "all"):
        for arm in arms:
            evaluate.infer(arm, "val")
        if args.arm is None:
            evaluate.compare("val")
    if args.command in ("freeze", "all"):
        evaluate.freeze()
    if args.command in ("test", "all"):
        for arm in arms:
            evaluate.infer(arm, "test")
        if args.arm is None:
            evaluate.compare("test")
    if args.command in ("report", "all"):
        evaluate.resources()
        evaluate.curves()
        evaluate.compare("val")
        evaluate.compare("test")
        save(
            OUT / "completion.json",
            dict(
                complete=True,
                training={a: read(OUT / "runs" / a / "result.json") for a in ARMS},
                final_comparison_sha256=file_hash(OUT / "test-comparison.json"),
            ),
        )


if __name__ == "__main__":
    main()
