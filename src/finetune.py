import argparse
import logging
import os
import json
import threading

import ray
import torch
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler

import wandb
from utils.paths import dataset_dir, output_dir
from utils.progress import configure_progress, logger
from trainer.make import make_multimodal_trainer
from trainer.artifacts import new_run_directory, initialize_run
from trainer.runtime import make_accelerator, prepare_optimization
from trainer.setup import CONFIG_ROOT, add_setup_arguments, resolve_setup, setup_summary
from trainer.pretrained import build_model, load_pretrained_model
from utils.utils import dummy_load


def main(tune_config=None):
    configure_progress()
    logger.info("fine-tuning: resolving configuration and verifying dataset")

    config, train_dataloader, val_dataloader, meta_data = resolve_setup(args, tune_config)
    if args.eid in (None, "None") or meta_data["num_sessions"] != 1:
        raise ValueError("Fine-tuning requires --eid selecting exactly one session")
    if args.setup_only:
        print(json.dumps(setup_summary(config, train_dataloader, val_dataloader), indent=2))
        return
    args.model_mode = config.training.objective
    args.mask_mode = config.model.masker.mode
    args.mask_ratio = config.model.masker.ratio
    args.enc_task_var = config.training.enc_task_var
    args.mixed_training = config.training.mixed_training

    if not args.resume_checkpoint and not args.pretrained_checkpoint:
        raise ValueError("Fine-tuning requires --pretrained-checkpoint PATH or --resume-checkpoint PATH")
    eid = args.eid
    base_path = args.base_path
    lr, wd = config.optimizer.lr, config.optimizer.wd
    neural_mods, static_mods, dynamic_mods = ["spike"], [], ["vision-clip"]
    modal_filter = dict(config.training.modal_filter)

    accelerator = make_accelerator()
    log_dir = new_run_directory(base_path, accelerator)
    log_name = os.path.basename(log_dir)
    logging.info("Run artifacts: %s", log_dir)


    if config.wandb.use and accelerator.is_main_process:
        wandb.init(
            dir=base_path,
            project=config.wandb.project,
            entity=config.wandb.entity,
            config=config,
            name=log_name
        )


    # ----------
    # LOAD MODEL
    # ----------
    logging.info(f"Start model finetuning:")

    if args.resume_checkpoint:
        model = build_model(config=config, metadata=meta_data, modal_filter=modal_filter)
    else:
        model = load_pretrained_model(args.pretrained_checkpoint, config=config,
                                      metadata=meta_data, modal_filter=modal_filter)
    logging.info("Total parameters: %s", sum(parameter.numel() for parameter in model.parameters()))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=wd,
        eps=config.optimizer.eps
    )

    model, optimizer, train_dataloader, lr_scheduler = prepare_optimization(
        accelerator, model, optimizer, train_dataloader, config
    )

    print("modal_filter: ")
    print(modal_filter)

    trainer_kwargs = {
        "log_dir": log_dir,
        "accelerator": accelerator,
        "lr_scheduler": lr_scheduler,
        "avail_mod": neural_mods + static_mods + dynamic_mods,
        "avail_beh": static_mods + dynamic_mods,
        "modal_filter": modal_filter,
        "mixed_training": args.mixed_training,
        "enc_task_var": args.enc_task_var,
        "config": config,
    }

    stop_dummy_load = threading.Event()

    trainer_ = make_multimodal_trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        optimizer=optimizer,
        **trainer_kwargs,
        **meta_data
    )

    initialize_run(trainer_, kind="finetune", resume_checkpoint=args.resume_checkpoint, pretrained_checkpoint=args.pretrained_checkpoint)

    if args.dummy_load:
        logging.info(f"Starting dummy load with {args.dummy_size} samples")
        dummy_thread = threading.Thread(target=dummy_load, args=(stop_dummy_load, args.dummy_size))
        dummy_thread.start()
        try:
            validation_metrics = trainer_.train()
        finally:
            stop_dummy_load.set()
            dummy_thread.join()
    else:
        validation_metrics = trainer_.train()
    if args.search:
        train.report(validation_metrics)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    ap = argparse.ArgumentParser()
    ap.add_argument("--eid", type=str, default=None)
    ap.add_argument("--base_path", type=str, default=str(output_dir()))
    ap.add_argument("--data_path", type=str, default=str(dataset_dir()))
    ap.add_argument("--num_sessions", type=int, default=None)
    ap.add_argument("--model_mode", type=str, default=None)
    ap.add_argument("--mask_mode", type=str, default=None)
    ap.add_argument("--mask_ratio", type=float, default=None)
    ap.add_argument("--mixed_training", action="store_true")
    ap.add_argument("--enc_task_var", type=str, default=None)
    ap.add_argument(
        "--modality", nargs="+",
        default=["ap", "vision-clip"]
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dummy_load", action="store_true")
    ap.add_argument("--dummy_size", type=int, default=50000)
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--num_tune_sample", type=int, default=10)
    ap.add_argument("--config_dir", type=str, default=str(CONFIG_ROOT))
    ap.add_argument("--pretrained-checkpoint", help="Explicit pretrained checkpoint")
    add_setup_arguments(ap)
    args = ap.parse_args()

    if args.search:
        ray.init(address="auto")
        search_space = {
            "learning_rate": tune.loguniform(1e-4, 1e-3),
            "weight_decay": tune.loguniform(0.001, 0.1),
            "mask_ratio": tune.uniform(0.1, 0.4),
        }
        ray_path = os.path.join(args.base_path, "ray_results")
        scheduler = ASHAScheduler(
            metric="eval_avg_metric",
            mode="max",
            grace_period=1,
            reduction_factor=2
        )
        print("Starting hyperparameter search")
        print(f"saving to {ray_path}")

        eid_ = args.eid[:5] if args.eid else "session"

        analysis = tune.run(
            main,
            resources_per_trial={
                "cpu": 1,
                "gpu": 1
            },
            config=search_space,
            num_samples=args.num_tune_sample,
            scheduler=scheduler,
            storage_path=ray_path,
            name=f"{eid_}_{args.model_mode}",
            log_to_file=True,
            verbose=2
        )
        # Get the best hyperparameters
        best_hyperparameters = analysis.get_best_config(
            metric="eval_avg_metric",
            mode="max"
        )
        logging.info(f"Best hyperparameters: {best_hyperparameters}")
    else:
        current_path = os.path.dirname(os.path.realpath(__file__))
        logging.info(f"No hyperparameter search, Starting training")
        main()


