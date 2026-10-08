import os
from contextlib import nullcontext
from itertools import islice
import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import wandb
from trainer.artifacts import capture_rank_states, save_training_checkpoint
from trainer.objective import validate_batch, prepare_inputs, forward_objective
from utils.utils import (
    move_batch_to_device,
    plot_gt_pred,
    plot_neurons_r2,
)

OUTPUT_DIM = {
    "vision-clip": 768,
}

def set_seed(epoch, base_seed=42):
    seed = (base_seed + epoch) % 2**32
    print("Train seed set to {}.".format(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class MultiModalTrainer():
    def __init__(
        self,
        model,
        train_dataloader,
        eval_dataloader,
        optimizer,
        **kwargs
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.optimizer = optimizer

        self.log_dir = kwargs.get("log_dir", None)
        self.accelerator = kwargs.get("accelerator", None)
        self.lr_scheduler = kwargs.get("lr_scheduler", None)
        self.config = kwargs.get("config", None)
        self.num_neurons = kwargs.get("num_neurons", None)
        self.eid_list = list(kwargs.get("eid_list", []))
        self.multi_gpu = kwargs.get("multi_gpu", None)

        self.model_class = self.config.model.model_class
        self.session_active_neurons = {}
        self.mod_to_indx = self.accelerator.unwrap_model(self.model).mod_to_indx
        self.optimizer_steps = 0
        self.scheduler_steps = 0
        self.populations = {}
        self.memberships = {}
        for loader in (self.train_dataloader, self.eval_dataloader):
            if loader is None:
                continue
            view = loader.dataset.scientific_split
            for sample in view.samples:
                self.populations[sample.session_id] = sample.neuron_identity
                self.memberships[sample.sample_id] = (sample.session_id, sample.trial_id, sample.split)
        self.avail_mod = kwargs.get("avail_mod", None)
        self.avail_beh = kwargs.get("avail_beh", None)
        self.modal_filter = kwargs.get("modal_filter", None)

        self.n_output_mods = len(self.modal_filter["output"])

        self.mixed_training = kwargs.get("mixed_training", False)

        if self.mixed_training:
            self.training_mode = "mixed"
        else:
            self.training_schemes = self.config.training.training_schemes

        self.enc_task_var = kwargs.get("enc_task_var", False)

        self.start_epoch = kwargs.get("start_epoch", 0)

        self.STATIC_VARS = []
        self.DYNAMIC_VARS = ["vision-clip"]

    def cosine_similarity_metric(self, gt, pred):
        """
        gt/pred:
            [B, T, D]
        """

        gt = F.normalize(gt, dim=-1)
        pred = F.normalize(pred, dim=-1)

        sim = (gt * pred).sum(dim=-1)

        return torch.nanmean(sim).item()

    def _prepare_inputs(self, batch, training_mode, enc_task_var=None):
        batch = move_batch_to_device(batch, self.accelerator.device)
        validate_batch(batch, config=self.config, populations=self.populations,
                       split="train" if self.model.training else "val")
        for index, sample_id in enumerate(batch["sample_id"]):
            expected = (batch["session_id"][index], int(batch["trial_id"][index]), batch["split"][index])
            if self.memberships.get(sample_id) != expected:
                raise ValueError("Batch sample identity does not match persisted split membership")
        model = self.accelerator.unwrap_model(self.model)
        for embedding in model.encoder_embeddings.values():
            if any(session not in embedding.embedder.eid_to_indx for session in batch["session_id"]):
                raise ValueError("Session is absent from the retained model session-embedding mapping")
        return prepare_inputs(batch, model=model, training_mode=training_mode, enc_task_var=enc_task_var)

    def _forward_model_inputs(self, batch, training_mode, enc_task_var=None):
        data, selectors = self._prepare_inputs(batch, training_mode, enc_task_var)
        outputs, _ = forward_objective(self.model, data, selectors,
                                       components=self.config.training.active_loss_components)
        return outputs

    def _check_all_ranks(self, error):
        failed = torch.tensor(int(error is not None), device=self.accelerator.device)
        if self.accelerator.reduce(failed, reduction="sum").item():
            raise ValueError(str(error) if error is not None else "Training failed on another distributed rank")

    def _plot_log_epoch(self, epoch, eval_epoch_results, n_viz=5):

        import matplotlib.pyplot as plt
        for idx, modalities in eval_epoch_results["eval_gt"].items():
            session = self.eid_list[idx]
            for mod, target in modalities.items():
                figures = self.plot_epoch(target, eval_epoch_results["eval_preds"][idx][mod], epoch,
                                          self.session_active_neurons[session][mod][:n_viz], mod)
                if self.config.wandb.use:
                    wandb.log({f"{session}/{mod}/{name}": wandb.Image(fig) for name, fig in figures.items()})
                for figure in figures.values():
                    plt.close(figure)

    def train(self):
        for epoch in range(self.start_epoch, self.config.training.num_epochs):
            train_results = self.train_epoch(epoch)
            self.accelerator.wait_for_everyone()
            error = None
            improved = False
            public = {}
            if (self.eval_dataloader is not None and self.config.training.eval_every
                    and epoch % self.config.training.eval_every == 0
                    and self.accelerator.is_main_process):
                wrapped = self.model
                try:
                    self.model = self.accelerator.unwrap_model(wrapped)
                    results = self.eval_epoch()
                    public = {key: value for key, value in results.items()
                              if key not in ("eval_gt", "eval_preds")}
                    if self.config.training.checkpoint_selection != "final":
                        metric = self.config.training.selection.metric
                        value = results[metric]
                        if value is None or not np.isfinite(value):
                            raise ValueError(f"Unusable checkpoint selection metric {metric}: "
                                             f"{results['eval_metric_unavailable']}")
                        previous = self.selected_checkpoint
                        improved = previous is None or (
                            value < previous["value"] if self.config.training.selection.direction == "min"
                            else value > previous["value"])
                        if improved:
                            self.selected_checkpoint = dict(epoch=epoch, rule=self.config.training.checkpoint_selection,
                                                            metric=metric, value=value, validation=public,
                                                            checkpoint=f"model_selected_epoch_{epoch}.pt")
                    if (self.config.training.save_plot_every_n_epochs
                            and epoch % self.config.training.save_plot_every_n_epochs == 0):
                        self._plot_log_epoch(epoch, results)
                    logs = dict(epoch=epoch, **train_results, **public)
                    wandb.log(logs) if self.config.wandb.use else print(logs)
                except (ValueError, KeyError, TypeError, RuntimeError, IndexError) as caught:
                    error = caught
                finally:
                    self.model = wrapped
            self._check_all_ranks(error)
            if self.accelerator.is_main_process:
                self.history.append(dict(epoch=epoch, training=train_results, validation=public))
            capture_rank_states(self)
            error = None
            try:
                if improved:
                    self.save_model(name=f"selected_epoch_{epoch}", epoch=epoch)
                    self.save_model(name="best", epoch=epoch)
                    self.save_model(name="best_avg", epoch=epoch)
                if epoch % self.config.training.save_every == 0:
                    self.save_model(name="epoch", epoch=epoch)
            except (ValueError, TypeError, RuntimeError, OSError) as caught:
                error = caught
            self._check_all_ranks(error)
            self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            if self.config.training.checkpoint_selection == "final":
                self.selected_checkpoint = dict(epoch=epoch, rule="final", metric=None, value=None,
                                                validation={})
            elif self.selected_checkpoint is None:
                raise ValueError("No usable validation checkpoint was selected")
        error = None
        try:
            self.save_model(name="last", epoch=epoch)
        except (ValueError, TypeError, RuntimeError, OSError) as caught:
            error = caught
        self._check_all_ranks(error)
        selected = self.selected_checkpoint or {}
        best = selected.get("validation", {})
        report = {key: value for key, value in best.items()
                  if key == "eval_loss" or key.endswith("_metric")}
        if self.accelerator.is_main_process:
            print({"selected_checkpoint": selected})
            if self.config.wandb.use:
                wandb.log(dict(best_epoch=selected["epoch"], **{f"best_{k}": v for k, v in report.items()}))
        return report


    def train_epoch(self, epoch):
        components = self.config.training.active_loss_components
        mods = list(components)
        totals = torch.zeros((2, len(mods)), dtype=torch.float64, device=self.accelerator.device)
        accumulation = self.config.optimizer.gradient_accumulation_steps
        set_seed(epoch * self.accelerator.num_processes,
                 base_seed=self.config.seed + self.accelerator.process_index)
        self.model.train()
        iterator = iter(self.train_dataloader)
        updates_before = self.optimizer_steps
        with tqdm(total=len(self.train_dataloader), disable=not self.accelerator.is_local_main_process) as progress:
            while True:
                error = None
                try:
                    window = list(islice(iterator, accumulation))
                except (ValueError, KeyError, TypeError, RuntimeError) as caught:
                    error, window = caught, []
                self._check_all_ranks(error)
                if not window:
                    break
                prepared, error = [], None
                try:
                    for batch in window:
                        mode = "mixed" if self.mixed_training else random.choice(self.training_schemes)
                        task = (random.choice(self.DYNAMIC_VARS + ["all"]) if self.enc_task_var == "random"
                                else self.enc_task_var) if mode == "encoding" else None
                        prepared.append(self._prepare_inputs(batch, mode, task))
                except (ValueError, KeyError, TypeError, RuntimeError) as caught:
                    error = caught
                self._check_all_ranks(error)
                local_counts = torch.stack([sum(selectors[mod].sum() for _, selectors in prepared) for mod in mods])
                counts = self.accelerator.reduce(local_counts, reduction="sum")
                self.optimizer.zero_grad(set_to_none=True)
                if not counts.any():
                    if self.accelerator.is_main_process:
                        print("Skipping accumulation window: no eligible objective targets")
                    progress.update(len(window))
                    continue
                for index, (data, selectors) in enumerate(prepared):
                    context = self.accelerator.no_sync(self.model) if index < len(prepared) - 1 else nullcontext()
                    with context:
                        error = None
                        try:
                            outputs, numerators = forward_objective(self.model, data, selectors, components=components)
                            loss = sum(components[mod]["weight"] * numerators[mod] * self.accelerator.num_processes
                                       / counts[i].clamp_min(1) for i, mod in enumerate(mods))
                            if not torch.isfinite(loss):
                                raise ValueError("Non-finite accumulation objective")
                        except (ValueError, KeyError, TypeError, RuntimeError) as caught:
                            error = caught
                        self._check_all_ranks(error)
                        self.accelerator.backward(loss)
                    for i, mod in enumerate(mods):
                        totals[0,i] += numerators[mod].detach().double()
                        totals[1,i] += selectors[mod].sum()
                    progress.update(1)
                self.accelerator.unscale_gradients(self.optimizer)
                finite = all(torch.isfinite(parameter.grad).all().item() for parameter in self.model.parameters()
                             if parameter.grad is not None)
                self._check_all_ranks(None if finite else ValueError("Non-finite gradients; optimizer update aborted"))
                self.optimizer.step()
                if self.accelerator.optimizer_step_was_skipped:
                    raise ValueError("Optimizer update was skipped; numerical recovery is unsupported")
                self.optimizer_steps += 1
                self.lr_scheduler.step()
                self.scheduler_steps += 1
                self.optimizer.zero_grad(set_to_none=True)
        totals = self.accelerator.reduce(totals, reduction="sum")
        if any(totals[1, i] == 0 for i in range(len(mods))):
            raise ValueError("Training epoch has no eligible targets for an active objective")
        losses = {f"train_{mod}_loss": (totals[0,i] / totals[1,i].clamp_min(1)).item() for i, mod in enumerate(mods)}
        results = dict(train_loss=sum(components[mod]["weight"] * losses[f"train_{mod}_loss"] for mod in mods),
                       **losses, optimizer_updates=self.optimizer_steps - updates_before,
                       optimizer_step=self.optimizer_steps, scheduler_step=self.scheduler_steps,
                       valid_target_counts={mod: int(totals[1,i]) for i, mod in enumerate(mods)},
                       learning_rate=self.optimizer.param_groups[0]["lr"])
        if self.accelerator.is_main_process:
            print({"epoch": epoch, **results})
        return results

    @torch.no_grad()
    def eval_epoch(self):
        if self.eval_dataloader is None:
            raise ValueError("Validation requires a validation loader")
        self.model.eval()
        components = self.config.training.active_loss_components
        totals = {mod: [0., 0] for mod in components}
        records = {}
        seen = set()
        for batch in self.eval_dataloader:
            for sample_id in batch["sample_id"]:
                if sample_id in seen:
                    raise ValueError("Validation sample was visited more than once")
                seen.add(sample_id)
            for mod in components:
                mode = "encoding" if mod == "spike" else "decoding"
                data, selectors = self._prepare_inputs(batch, mode, "all" if mode == "encoding" else None)
                outputs, numerators = forward_objective(self.model, data, selectors, components={mod: components[mod]})
                totals[mod][0] += numerators[mod].double().item()
                totals[mod][1] += selectors[mod].sum().item()
                for index, session in enumerate(batch["session_id"]):
                    entry = records.setdefault(session, {}).setdefault(mod, {"gt": [], "preds": []})
                    valid = selectors[mod][index]
                    gt = outputs.mod_targets[mod][index].detach().double()
                    pred = outputs.mod_preds[mod][index].detach().double()
                    if mod == "spike":
                        width = len(self.populations[session])
                        gt, pred, valid = gt[:, :width], pred[:, :width], valid[:, :width]
                        gt = gt.masked_fill(~valid, float("nan"))
                        entry.setdefault("log_preds", []).append(pred.cpu())
                        pred = pred.exp().masked_fill(~valid, float("nan"))
                        if not torch.isfinite(pred[valid]).all():
                            raise ValueError("Non-finite neural validation rates")
                    else:
                        gt = gt.masked_fill(~valid.unsqueeze(-1), float("nan"))
                        pred = pred.masked_fill(~valid.unsqueeze(-1), float("nan"))
                    entry["gt"].append(gt.cpu())
                    entry["preds"].append(pred.cpu())
        expected = {sample.sample_id for sample in self.eval_dataloader.dataset.scientific_split.samples}
        if seen != expected:
            raise ValueError("Validation did not visit every persisted validation sample exactly once")
        losses = {}
        for mod, (numerator, count) in totals.items():
            if count == 0:
                raise ValueError(f"No valid validation targets for {mod}")
            losses[f"eval_{mod}_loss"] = numerator / count
        gt, preds, metrics, unavailable = {}, {}, {}, {}
        session_metrics = {}
        for session, modalities in records.items():
            idx = self.eid_list.index(session)
            gt[idx], preds[idx], session_metrics[session] = {}, {}, {}
            self.session_active_neurons[session] = {}
            for mod, entry in modalities.items():
                target = torch.stack(entry["gt"])
                prediction = torch.stack(entry["preds"])
                gt[idx][mod], preds[idx][mod] = target, prediction
                self.session_active_neurons[session][mod] = list(range(target.shape[-1]))
                if mod == "spike":
                    valid = torch.isfinite(target)
                    count = valid.sum((0, 1))
                    spikes = target.nan_to_num().sum((0, 1))
                    mean = spikes / count.clamp_min(1)
                    model_nll = (prediction - target * torch.stack(entry["log_preds"])).masked_fill(~valid, 0).sum((0, 1))
                    null_nll = (mean - target * mean.clamp_min(torch.finfo(mean.dtype).tiny).log()).masked_fill(~valid, 0).sum((0, 1))
                    eligible = spikes > 0
                    values = (null_nll[eligible] - model_nll[eligible]) / spikes[eligible] / np.log(2)
                    value = values.mean().item() if values.numel() else None
                    if (~eligible).any():
                        unavailable[f"{session}/bps_neurons"] = {"indices": (~eligible).nonzero().flatten().tolist(),
                                                                "reason": "zero total observed spikes"}
                else:
                    valid = torch.isfinite(target).all(-1)
                    value = F.cosine_similarity(target[valid], prediction[valid], dim=-1).mean().item()
                if value is not None and not np.isfinite(value):
                    raise ValueError(f"Non-finite validation metric for {session}/{mod}")
                session_metrics[session][mod] = value
        for mod in components:
            values = [entry[mod] for entry in session_metrics.values()]
            # An undefined session metric makes selection undefined; never silently nanmean it away.
            metrics[f"eval_{mod}_metric"] = (sum(values) / len(values)
                                               if values and all(value is not None for value in values) else None)
        values = list(metrics.values())
        metrics["eval_avg_metric"] = sum(values) / len(values) if all(v is not None for v in values) else None
        return dict(eval_loss=sum(components[mod]["weight"] * losses[f"eval_{mod}_loss"] for mod in components),
                    **losses, **metrics, eval_valid_target_counts={mod: count for mod, (_, count) in totals.items()},
                    eval_sample_count=len(seen), eval_session_metrics=session_metrics,
                    eval_sessions_absent=[session for session in self.eid_list if session not in records],
                    eval_metric_unavailable=unavailable, eval_gt=gt, eval_preds=preds)

    def plot_epoch(self, gt, preds, epoch, active_neurons, modality):
        target, prediction = gt.nanmean(0), preds.nanmean(0)
        valid = torch.isfinite(target).all(-1) & torch.isfinite(prediction).all(-1)
        target, prediction = target[valid], prediction[valid]
        figures = {"plot_gt_pred": plot_gt_pred(gt=target.T.numpy(), pred=prediction.T.numpy(),
                                               epoch=epoch, modality=modality)}
        if len(target) >= 2 and active_neurons:
            figures["plot_r2"] = plot_neurons_r2(gt=target, pred=prediction,
                                                neuron_idx=active_neurons, epoch=epoch)
        return figures

    def save_model(self, name="last", epoch=0):
        save_training_checkpoint(self, name, epoch)
