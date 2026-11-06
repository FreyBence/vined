"""Train one selected delay using the production trainer and checkpoint lifecycle."""
import contextlib
import logging
from pathlib import Path
import sys

from common import (parser, run_settings, setup_arguments, support_record, source_identity,
                    bind_experiment, stage, read, completed, path, atomic_json, plain,
                    fingerprint, file_hash)


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def __getattr__(self, name):
        # Preserve terminal encoding, isatty and fileno for tqdm detection.
        return getattr(self.console, name)

    def write(self, text):
        self.console.write(text); self.log.write(text); self.log.flush()
        return len(text)

    def flush(self):
        self.console.flush(); self.log.flush()


def main():
    p = parser(__doc__, run=True)
    p.add_argument("--preflight", action="store_true", help="Resolve all parameters and inspect real support without training")
    p.add_argument("--skip-completed", action="store_true")
    p.add_argument("--resume-checkpoint", help="Epoch-boundary checkpoint from this run only")
    args = p.parse_args()
    exp, evaluation, run, directory = run_settings(args)
    from trainer.setup import resolve_setup
    setup_args = setup_arguments(exp, run)
    config, train_loader, val_loader, metadata = resolve_setup(setup_args)
    if (config.training.target_granularity not in ("neuron", "region") or config.training.objective != "encoding"
            or metadata.get("neural_region_selection", "all_recorded") != "all_recorded"):
        raise ValueError("Target-delay experiment requires neuron or region encoding of all recorded units")
    from math import ceil
    config["optimizer"]["planned_updates"] = config.training.num_epochs * ceil(
        len(train_loader) / config.optimizer.gradient_accumulation_steps)
    record = support_record(exp, evaluation, target_granularity=config.training.target_granularity)
    if config.training.target_granularity == "region":
        # Bias initialization, validation BPS and offline baseline use the same
        # training-only common support, independent of D.
        metadata["regional_training_baseline"] = record["training_baseline"]
        config["dataset"]["regional_training_baseline"] = record["training_baseline"]
    sources = source_identity()
    experiment_record = dict(schema_version=1, experiment=exp, evaluation=evaluation,
                             support=record, sources=sources)
    bind_experiment(directory.parent.parent, experiment_record)
    resolved = plain(dict(run=run, configuration=config, support=record,
                         experiment=exp, evaluation=evaluation, sources=sources))
    identity = fingerprint(resolved)
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = directory / "resolved.json"
    if snapshot.exists() and read(snapshot) != resolved:
        raise ValueError("Existing run has incompatible resolved parameters; use a new --run-id")
    atomic_json(snapshot, resolved)
    atomic_json(directory / "preflight.json", dict(identity=identity, counts=record["counts"],
        mapping_examples=record["mapping_examples"], context=config.training.temporal_context,
        seed=config.seed, optimizer=config.optimizer, architecture=config.model,
        compatibility_id=record["compatibility_id"]))
    if args.preflight:
        return
    resume = path(args.resume_checkpoint) if args.resume_checkpoint else None
    if resume is not None and directory.resolve() not in resume.parents:
        raise ValueError("Resume must use a checkpoint under this run; cross-delay fine-tuning is forbidden")
    with stage(directory, "train", identity, skip=args.skip_completed) as attempt:
        if attempt is None:
            return
        import torch
        from trainer.pretrained import build_model, initialize_regional_bias
        from trainer.runtime import make_accelerator, prepare_optimization
        from trainer.make import make_multimodal_trainer
        from trainer.artifacts import initialize_run, load_training_checkpoint
        accelerator = make_accelerator()
        if accelerator.num_processes != 1:
            raise ValueError("Use one process per experiment invocation; production trainer handles device placement")
        logdir = attempt / "training"
        logdir.mkdir()
        with (attempt / "train.log").open("w", encoding="utf-8") as log:
            with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
                handler = logging.StreamHandler(log)
                logging.getLogger().addHandler(handler)
                try:
                    model = build_model(config=config, metadata=metadata, modal_filter=config.training.modal_filter)
                    initialize_regional_bias(model, metadata)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimizer.lr,
                        weight_decay=config.optimizer.wd, eps=config.optimizer.eps)
                    model, optimizer, train_loader, scheduler = prepare_optimization(
                        accelerator, model, optimizer, train_loader, config)
                    trainer = make_multimodal_trainer(model=model, train_dataloader=train_loader,
                        eval_dataloader=val_loader, optimizer=optimizer, log_dir=str(logdir),
                        accelerator=accelerator, lr_scheduler=scheduler,
                        avail_mod=["spike", "vision-clip"], avail_beh=["vision-clip"],
                        modal_filter=config.training.modal_filter, mixed_training=False,
                        enc_task_var=config.training.enc_task_var, config=config, multi_gpu=False,
                        start_epoch=0, **metadata)
                    initialize_run(trainer, kind="train", resume_checkpoint=str(resume) if resume else None)
                    trainer.train()
                    last = load_training_checkpoint(logdir / "model_last.pt")
                    selected = logdir / ("model_last.pt" if config.training.checkpoint_selection == "final" else "model_best.pt")
                    chosen = load_training_checkpoint(selected)
                    if (last["next_epoch"] != config.training.num_epochs or not last["history"]
                            or chosen["config"]["training"]["temporal_context"] != plain(config.training.temporal_context)):
                        raise ValueError("Training returned without valid final and selected artifacts")
                    for filename in ("config.json", "run.json", "provenance.json", "history.json", "selection.json"):
                        read(logdir / filename)
                    atomic_json(attempt / "result.json", dict(checkpoint=str(selected),
                        checkpoint_sha256=file_hash(selected), final_checkpoint=str(logdir / "model_last.pt"),
                        compatibility_id=record["compatibility_id"], resolved_identity=identity))
                finally:
                    logging.getLogger().removeHandler(handler)


if __name__ == "__main__":
    main()
