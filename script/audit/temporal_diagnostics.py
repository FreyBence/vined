"""Collect frozen regional encoding train/validation temporal diagnostics."""

import argparse

from audit_paths import next_output
from trainer.temporal_diagnostics import run_diagnostic


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", help="New diagnostic directory (default: output/audit/<scenario>/<attempt>/temporal-NNN)")
    parser.add_argument("--dataset-path", help="Relocated exact generation; its recorded hash identity must match")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    destination = args.output_dir or next_output(args.checkpoint, "temporal")
    result = run_diagnostic(args.checkpoint, destination, device=args.device, dataset_path=args.dataset_path)
    print(f"Saved {result['historical_semantics']} diagnostics to {destination}")
