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
from multi_modal.encoder_embeddings import EncoderEmbedding
from multi_modal.mm import MultiModal
from trainer.make import make_multimodal_trainer
from trainer.artifacts import new_run_directory, initialize_run
from trainer.runtime import make_accelerator, prepare_optimization, register_runtime_modules
from trainer.setup import CONFIG_ROOT, add_setup_arguments, resolve_setup, setup_summary
from utils.utils import dummy_load


def main(tune_config=None):

    neural_acronyms = {
        "ap": "spike"
    }
    static_acronyms = {}
    dynamic_acronyms = {
        "vision-clip": "vision-clip",
    }

    config, train_dataloader, val_dataloader, meta_data = resolve_setup(args, tune_config)
    if args.setup_only:
        print(json.dumps(setup_summary(config, train_dataloader, val_dataloader), indent=2))
        return
    args.model_mode = config.training.objective
    args.mask_mode = config.model.masker.mode
    args.mask_ratio = config.model.masker.ratio
    args.enc_task_var = config.training.enc_task_var
    args.mixed_training = config.training.mixed_training

    # ------
    # SET UP
    # ------
    eid = args.eid
    base_path = args.base_path
    model_mode = args.model_mode
    modality = ["ap", "vision-clip"]

    mask_ratio = config.model.masker.ratio
    lr = config.optimizer.lr
    wd = config.optimizer.wd
    logging.info(f"EID: {eid} model mode: {args.model_mode} mask ratio: {mask_ratio}")
    logging.info(f"Available modality: {modality}")

    neural_mods, static_mods, dynamic_mods = [], [], []
    for mod in modality:
        if mod in neural_acronyms:
            neural_mods.append(neural_acronyms[mod])
        elif mod in static_acronyms:
            static_mods.append(static_acronyms[mod])
        elif mod in dynamic_acronyms:
            dynamic_mods.append(dynamic_acronyms[mod])

    if model_mode == "mm":
        input_mods = output_mods = neural_mods + static_mods + dynamic_mods
    elif model_mode == "decoding":
        input_mods = neural_mods
        output_mods = static_mods + dynamic_mods
    elif model_mode == "encoding":
        input_mods = static_mods + dynamic_mods
        output_mods = neural_mods
    else:
        raise ValueError(f"Model mode {model_mode} not supported.")

    modal_filter = {"input": input_mods, "output": output_mods}

    accelerator = make_accelerator()

    batch_size = config.training.train_batch_size
    num_epochs = config.training.num_epochs
    max_lr = config.optimizer.lr

    # --------
    # SET PATH
    # --------
    log_dir = new_run_directory(base_path, accelerator)
    log_name = os.path.basename(log_dir)
    logging.info("Run artifacts: %s", log_dir)


    # ------------
    # SET UP MODEL
    # ------------

    logging.info(f"Start model training:")

    if config.wandb.use:
        if accelerator.is_main_process:
            wandb.init(
                dir=base_path,
                project=config.wandb.project,
                entity=config.wandb.entity,
                config=config,
                name=log_name
            )

    encoder_embeddings = {}

    hidden_size = config.model.encoder.transformer.hidden_size
    for mod in modal_filter["input"]:
        encoder_embeddings[mod] = EncoderEmbedding(
            hidden_size = hidden_size,
            n_channel = hidden_size,
            output_channel = hidden_size,
            stitching = True,
            eid_list = meta_data["eid_list"],
            mod = mod,
            config = config.model.encoder,
            max_F = config.data.max_time_length,
        )

    NAME2MODEL = {"MultiModal": MultiModal}
    model_class = NAME2MODEL[config.model.model_class]
    model = model_class(
        encoder_embeddings,
        avail_mod = neural_mods + static_mods + dynamic_mods,
        avail_beh = dynamic_mods,
        model_mode = model_mode,
        config = config.model,
        **({"context_mode": config.training.temporal_context.mode,
            "context_bins": config.training.temporal_context.bins}
           if "temporal_context" in config.training else {}),
        **meta_data
    )

    model = register_runtime_modules(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max_lr,
        weight_decay=wd,
        eps=config.optimizer.eps
    )

    start_epoch = 0
    model, optimizer, train_dataloader, lr_scheduler = prepare_optimization(
        accelerator, model, optimizer, train_dataloader, config
    )

    # -----------------------
    # TRACK MODEL & DATA SIZE
    # -----------------------
    n_mods = len(modal_filter["input"])
    n_tokens_per_mod = config.model.encoder.embedder.max_F
    num_train = len(train_dataloader.dataset)
    logging.info(f"Total modality: {n_mods} Total tokens per modality: {n_tokens_per_mod}")
    logging.info(f"Total trials: {num_train}")

    total_tokens = n_mods*n_tokens_per_mod*num_train
    logging.info(f"Total tokens: {total_tokens}")

    trial_length = 2 # Seconds
    total_neurons = sum(list(meta_data["eid_list"].values()))
    total_hours = num_train * trial_length / 3_600
    neuron_hours = total_neurons * total_hours
    logging.info(f"Total neurons: {total_neurons}")
    logging.info(f"Total hours: {total_hours}")
    logging.info(f"Neuron hours: {neuron_hours}")

    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total parameters: {total_params}")

    total_capacity = sum(
        p.numel() for name, p in model.named_parameters()
        if "stitch" not in name and "static_weight" not in name
    )
    logging.info(f"Total parameters (excluding stitcher): {total_capacity}")


    # -----
    # TRAIN
    # -----
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
        "multi_gpu": args.multi_gpu,
        "start_epoch": start_epoch,
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

    initialize_run(trainer_, kind="train", resume_checkpoint=args.resume_checkpoint)

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
    ap.add_argument("--continue_pretrain", action="store_true")
    ap.add_argument("--multi_gpu", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dummy_load", action="store_true")
    ap.add_argument("--dummy_size", type=int, default=50000)
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--num_tune_sample", type=int, default=50)
    ap.add_argument("--config_dir", type=str, default=str(CONFIG_ROOT))
    add_setup_arguments(ap)
    args = ap.parse_args()

    if args.debug:
        # Debug using deterministic mode
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        logging.info("Deterministic mode is activated. This will negatively impact performance.")

    if args.search:
        ray.init(address="auto")
        search_space = {
            "learning_rate": tune.loguniform(1e-4, 1e-3),
            "weight_decay": tune.loguniform(0.001, 0.1),
            "mask_ratio": tune.uniform(0.1, 0.4),
            "hidden_size": tune.choice([128, 256, 512]),
            "inter_size": tune.choice([256, 512, 1024]),
            "n_layers": tune.choice([5, 6]),
        }
        ray_path = os.path.join(args.base_path, "ray_results")
        scheduler = ASHAScheduler(
            metric="eval_avg_metric",
            mode="max",
            grace_period=1,
            reduction_factor=2
        )
        print("Starting hyperparameter search")
        print(f"Saving to {ray_path}")

        eid_ = args.eid[:5] if args.eid and args.eid != "None" else "multi"

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

