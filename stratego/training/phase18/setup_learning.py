"""Phase 18: the setup update -- loss, AdamW at zero decay, clipping, EMA and
checkpoints (S12, S16, S17, S18, S22, S25-S29).

The loss, once (published `arr_train`, transcribed)
---------------------------------------------------
```text
log_pi   = log_softmax(piece_logits masked with a finite sentinel)   [B, 40, 12]
r_k      = exp(log_pi[t_k] - log_pi_b[t_k])                          chosen token only
L_PPO    = -mean min(r delta, clip(r, 0.8, 1.2) delta)               all 40 prefixes, no filter
L_v      = -mean sum_c y_bar_c log softmax(wdl_logits)_c              soft (loss, draw, win) target
L_h      = mean (h - I/10)^2
L_KL     = mean sum_a mask_a pi(a) (log pi(a) - log pi_b(a))          KL(current || behavior)
L_setup  = 1.0 L_PPO + 0.5 L_v + 1.0 L_h + 0.1 L_KL
```

Everything subscripted `b` is read from the buffer's recorded behavior
fields and never recomputed (S22).

The update, once
----------------
```text
for epoch in 1..5:
    for minibatch of up to 1,024 ready setups (shuffled per epoch):
        loss -> backward -> clip_grad_norm_(setup params, 0.5) -> AdamW step
EMA.update()   exactly once, after all epochs and minibatches (S28)
```

Raw weights generate pools and are trained; the EMA never enters the training
population and is the only evaluation model (S28).
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .setup_buffer import SetupBatch, SetupBuffer
from .setup_contract import (
    MASKED_LOGIT,
    SETUP_CHECKPOINT_VERSION,
    SETUP_EQUATION_VERSION,
    SETUP_PREFIXES,
    SETUP_RECIPE_VERSION,
    Phase18SetupError,
    SetupTrainingConfig,
    file_sha256,
    json_document_digest,
    shuffle_seed,
)
from .setup_model import Phase18SetupModel, build_setup_model, state_dict_digest


# ---------------------------------------------------------------------------
# The loss (S12, S16, S17, S18)
# ---------------------------------------------------------------------------


def setup_batch_loss(model: Phase18SetupModel, batch: SetupBatch, *, config: SetupTrainingConfig) -> tuple:
    """Every term of `L_setup`, each returned under its own name."""
    outputs = model(batch.sequence)
    # The loss runs in the model's own dtype: float32 in production, float64
    # when the parity oracle differentiates a double-precision copy.
    dtype = outputs["piece_logits"].dtype
    logits = outputs["piece_logits"].masked_fill(~batch.masks, MASKED_LOGIT)
    log_probabilities = torch.log_softmax(logits, dim=-1)
    probabilities = log_probabilities.exp()

    selected = log_probabilities.gather(2, batch.tokens.unsqueeze(-1)).squeeze(-1)
    ratio = torch.exp(selected - batch.behavior_selected_log_prob.to(dtype))
    low, high = 1.0 - config.ppo_clip_epsilon, 1.0 + config.ppo_clip_epsilon
    clipped = torch.clamp(ratio, low, high)
    advantage = batch.advantage.to(dtype)
    policy_loss = -torch.min(ratio * advantage, clipped * advantage).mean()
    clip_fraction = ((ratio - 1.0).abs() > config.ppo_clip_epsilon).to(torch.float32).mean()

    log_wdl = torch.log_softmax(outputs["wdl_logits"], dim=-1)      # [B, 40, 3]
    target = batch.value_target.to(dtype).unsqueeze(1).expand(-1, SETUP_PREFIXES, -1)
    value_loss = -(target * log_wdl).sum(-1).mean()

    entropy_loss = ((outputs["entropy_prediction"] - batch.entropy_target.to(dtype)) ** 2).mean()

    # KL(pi_current || pi_behavior) over the legal types; illegal types
    # contribute exactly zero rather than a small finite amount (S17).
    surprise = (log_probabilities - batch.behavior_log_probs.to(dtype)).masked_fill(~batch.masks, 0.0)
    per_prefix_kl = (probabilities.masked_fill(~batch.masks, 0.0) * surprise).sum(dim=-1)
    kl = per_prefix_kl.mean()

    total = (
        config.policy_loss_weight * policy_loss
        + config.value_loss_weight * value_loss
        + config.entropy_prediction_loss_weight * entropy_loss
        + config.behavior_kl_coefficient * kl
    )

    with torch.no_grad():
        policy_entropy = -(probabilities * log_probabilities.masked_fill(~batch.masks, 0.0)).sum(dim=-1)

    terms = {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy_prediction_loss": entropy_loss,
        "behavior_kl": kl,
        "total_loss": total,
        "clip_fraction": clip_fraction,
        "ratio_mean": ratio.mean(),
        "ratio_min": ratio.min(),
        "ratio_max": ratio.max(),
        "mean_prefix_entropy_nats": policy_entropy.mean(),
        "advantage_mean": batch.advantage.mean(),
        "advantage_std": batch.advantage.std(unbiased=False),
    }
    return total, terms


# ---------------------------------------------------------------------------
# EMA (S28)
# ---------------------------------------------------------------------------


class SetupEMA:
    """Parameter EMA at decay 0.999, updated once after the complete update.

    `ema = decay * ema + (1 - decay) * raw`, over the model's parameters. The
    setup model has no buffers, so `state_dict` and `parameters` cover the
    same tensors.
    """

    def __init__(self, model: Phase18SetupModel, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise Phase18SetupError(f"EMA decay must be in (0, 1), got {decay}")
        self.decay = float(decay)
        self.shadow = {
            name: tensor.detach().clone().to(torch.float32) for name, tensor in model.state_dict().items()
        }
        self.device = next((tensor.device for tensor in self.shadow.values()), torch.device("cpu"))
        self.updates = 0

    @torch.no_grad()
    def update(self, model: Phase18SetupModel) -> None:
        for name, tensor in model.state_dict().items():
            current = tensor.detach().to(torch.float32)
            self.shadow[name].mul_(self.decay).add_(current, alpha=1.0 - self.decay)
        self.updates += 1

    def state_dict(self) -> dict:
        return {name: tensor.clone() for name, tensor in self.shadow.items()}

    def load_state_dict(self, state: dict) -> None:
        """Restore the shadow ONTO THE MODEL'S DEVICE, whatever the payload's."""
        missing = set(self.shadow) - set(state)
        if missing:
            raise Phase18SetupError(f"EMA state is missing {sorted(missing)}")
        self.shadow = {
            name: tensor.detach().to(device=self.device, dtype=torch.float32).clone()
            for name, tensor in state.items()
        }

    def as_model(self, device: str = "cpu") -> Phase18SetupModel:
        """A materialised EMA model. Evaluation only, by contract."""
        model = build_setup_model(device=device)
        model.load_state_dict({name: tensor.to(device) for name, tensor in self.shadow.items()})
        model.eval()
        return model


# ---------------------------------------------------------------------------
# The trainer (S25, S26, S27, S28)
# ---------------------------------------------------------------------------


@dataclass
class SetupUpdateResult:
    update: int
    alpha: float
    ready_rows: int
    minibatches_per_epoch: int
    epochs: int
    optimizer_steps: int
    digest_before: str
    digest_after: str
    ema_digest_after: str
    ema_updates: int
    pre_clip_grad_norms: list = field(default_factory=list)
    post_clip_grad_norms: list = field(default_factory=list)
    clip_activations: int = 0
    epoch_terms: list = field(default_factory=list)
    process_telemetry: dict | None = None
    non_finite_events: int = 0

    def document(self) -> dict:
        norms = np.asarray(self.pre_clip_grad_norms, dtype=np.float64)
        post = np.asarray(self.post_clip_grad_norms, dtype=np.float64)
        return {
            "recipe": SETUP_RECIPE_VERSION,
            "setup_equation_version": SETUP_EQUATION_VERSION,
            "update": self.update,
            "alpha": self.alpha,
            "ready_rows": self.ready_rows,
            "minibatches_per_epoch": self.minibatches_per_epoch,
            "epochs": self.epochs,
            "optimizer_steps": self.optimizer_steps,
            "raw_digest_before": self.digest_before,
            "raw_digest_after": self.digest_after,
            "ema_digest_after": self.ema_digest_after,
            "ema_updates": self.ema_updates,
            "pre_clip_grad_norm": {
                "mean": float(norms.mean()) if norms.size else None,
                "max": float(norms.max()) if norms.size else None,
                "min": float(norms.min()) if norms.size else None,
            },
            "post_clip_grad_norm_max": float(post.max()) if post.size else None,
            "clip_activation_rate": (self.clip_activations / len(self.pre_clip_grad_norms)) if self.pre_clip_grad_norms else None,
            "epochs_detail": self.epoch_terms,
            "process": self.process_telemetry,
            "non_finite_events": self.non_finite_events,
        }


class SetupTrainer:
    """Raw weights, AdamW(wd=0), EMA, updated together in the published order."""

    def __init__(self, model: Phase18SetupModel, config: SetupTrainingConfig, *, namespace: str, seed_index: int) -> None:
        self.model = model
        self.config = config
        self.namespace = namespace
        self.seed_index = int(seed_index)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=tuple(config.adam_betas),
            eps=config.adam_epsilon,
            weight_decay=config.weight_decay,
        )
        self.ema = SetupEMA(model, config.ema_decay)
        self.updates = 0
        self.optimizer_step_count = 0
        self.non_finite_events = 0

    @property
    def generation_actor(self) -> Phase18SetupModel:
        """S28: the RAW model generates pools. Never the EMA."""
        return self.model

    def evaluation_model(self, device: str | None = None) -> Phase18SetupModel:
        """S28: the EMA is the only evaluation model."""
        return self.ema.as_model(device=device or self.config.device)

    def update(self, buffer: SetupBuffer, *, global_iteration: int) -> SetupUpdateResult:
        if int(global_iteration) < 1:
            raise Phase18SetupError("the global iteration is one-based")
        alpha = self.config.alpha(int(global_iteration))
        processed = buffer.process(alpha=alpha, td_lambda=self.config.td_lambda, gae_lambda=self.config.gae_lambda)
        ready = int(processed.indices.size)
        per_epoch = math.ceil(ready / self.config.batch_size)
        digest_before = state_dict_digest(self.model)

        self.model.train()
        pre_norms: list = []
        post_norms: list = []
        clip_activations = 0
        epoch_records: list = []
        steps = 0
        for epoch in range(self.config.epochs_per_update):
            seed = shuffle_seed(self.namespace, self.seed_index, int(global_iteration), epoch)
            epoch_terms: list = []
            for batch in buffer.minibatches(self.config.batch_size, seed=seed):
                total, terms = setup_batch_loss(self.model, batch, config=self.config)
                if not torch.isfinite(total):
                    self.non_finite_events += 1
                    raise Phase18SetupError(f"non-finite setup loss at update {global_iteration} epoch {epoch}")
                self.optimizer.zero_grad(set_to_none=True)
                total.backward()
                pre = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                if not torch.isfinite(pre):
                    self.non_finite_events += 1
                    raise Phase18SetupError(f"non-finite setup gradient norm at update {global_iteration}")
                post = gradient_norm(self.model)
                self.optimizer.step()
                self.optimizer_step_count += 1
                steps += 1
                pre_norms.append(float(pre))
                post_norms.append(float(post))
                if float(pre) > self.config.gradient_clip_norm:
                    clip_activations += 1
                epoch_terms.append({name: float(value.detach()) for name, value in terms.items()} | {"batch_rows": batch.count})
            epoch_records.append({"epoch": epoch, "minibatches": len(epoch_terms), **_mean_documents(epoch_terms)})

        # Raw first, EMA only after the complete update.
        self.ema.update(self.model)
        self.updates += 1
        return SetupUpdateResult(
            update=int(global_iteration),
            alpha=alpha,
            ready_rows=ready,
            minibatches_per_epoch=per_epoch,
            epochs=self.config.epochs_per_update,
            optimizer_steps=steps,
            digest_before=digest_before,
            digest_after=state_dict_digest(self.model),
            ema_digest_after=state_dict_digest(self.ema.as_model(device="cpu")),
            ema_updates=self.ema.updates,
            pre_clip_grad_norms=pre_norms,
            post_clip_grad_norms=post_norms,
            clip_activations=clip_activations,
            epoch_terms=epoch_records,
            process_telemetry=processed.telemetry,
            non_finite_events=self.non_finite_events,
        )

    # -- checkpoints (S29): three separately identified objects ------------------

    def save_checkpoint(self, directory) -> dict:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        raw_path, optimizer_path, ema_path = (directory / "raw.pt", directory / "optimizer.pt", directory / "ema.pt")
        raw_state = {name: tensor.detach().cpu().clone() for name, tensor in self.model.state_dict().items()}
        ema_state = {name: tensor.detach().cpu().clone() for name, tensor in self.ema.state_dict().items()}
        optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        torch.save(raw_state, raw_path)
        torch.save(ema_state, ema_path)
        torch.save(optimizer_state, optimizer_path)
        manifest = {
            "checkpoint_version": SETUP_CHECKPOINT_VERSION,
            "recipe": SETUP_RECIPE_VERSION,
            "setup_equation_version": SETUP_EQUATION_VERSION,
            "run_id": self.config.run_id,
            "namespace": self.namespace,
            "seed_index": self.seed_index,
            "updates": self.updates,
            "optimizer_step_count": self.optimizer_step_count,
            "ema_updates": self.ema.updates,
            "config_digest": self.config.config_digest(),
            "raw": {"file": raw_path.name, "sha256": file_sha256(raw_path), "state_digest": state_dict_digest(self.model)},
            "ema": {"file": ema_path.name, "sha256": file_sha256(ema_path), "state_digest": state_dict_digest(self.ema.as_model(device="cpu"))},
            "optimizer": {"file": optimizer_path.name, "sha256": file_sha256(optimizer_path), "class": type(self.optimizer).__name__},
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
        return manifest

    @classmethod
    def load_checkpoint(cls, directory, config: SetupTrainingConfig, *, namespace: str, seed_index: int, device: str | None = None) -> tuple:
        """Rebuild a trainer from the three objects, refusing any identity drift."""
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["checkpoint_version"] != SETUP_CHECKPOINT_VERSION:
            raise Phase18SetupError(f"checkpoint version {manifest['checkpoint_version']!r} is not {SETUP_CHECKPOINT_VERSION!r}")
        if manifest["recipe"] != SETUP_RECIPE_VERSION or manifest["setup_equation_version"] != SETUP_EQUATION_VERSION:
            raise Phase18SetupError("checkpoint was written under a different recipe or equation")
        if manifest["config_digest"] != config.config_digest():
            raise Phase18SetupError("checkpoint was written under a different setup config digest")
        if manifest["run_id"] != config.run_id:
            raise Phase18SetupError(f"checkpoint belongs to run {manifest['run_id']!r}, not {config.run_id!r}")
        for key in ("raw", "ema", "optimizer"):
            path = directory / manifest[key]["file"]
            observed = file_sha256(path)
            if observed != manifest[key]["sha256"]:
                raise Phase18SetupError(f"{key} checkpoint file digest moved: {observed} != {manifest[key]['sha256']}")
        target = device or config.device
        model = build_setup_model(device=target)
        raw_state = torch.load(directory / manifest["raw"]["file"], map_location="cpu")
        model.load_state_dict({name: tensor.to(target) for name, tensor in raw_state.items()})
        if state_dict_digest(model) != manifest["raw"]["state_digest"]:
            raise Phase18SetupError("restored raw weights do not reproduce the recorded digest")
        trainer = cls(model, config.replace(device=target), namespace=namespace, seed_index=seed_index)
        trainer.optimizer.load_state_dict(torch.load(directory / manifest["optimizer"]["file"], map_location="cpu"))
        trainer.ema.load_state_dict(torch.load(directory / manifest["ema"]["file"], map_location="cpu"))
        if state_dict_digest(trainer.ema.as_model(device="cpu")) != manifest["ema"]["state_digest"]:
            raise Phase18SetupError("restored EMA weights do not reproduce the recorded digest")
        trainer.updates = int(manifest["updates"])
        trainer.optimizer_step_count = int(manifest["optimizer_step_count"])
        trainer.ema.updates = int(manifest["ema_updates"])
        return trainer, manifest


def gradient_norm(model: torch.nn.Module) -> float:
    """The global L2 norm of the current gradients, accumulated in float64.

    Moved to the CPU first and widened second: MPS has no float64, and an
    in-place `.to(float64)` on an MPS tensor raises.
    """
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().cpu().to(torch.float64).pow(2).sum())
    return math.sqrt(total)


def _mean_documents(documents: list) -> dict:
    if not documents:
        return {}
    keys = [key for key in documents[0] if isinstance(documents[0][key], (int, float))]
    return {key: float(np.mean([document[key] for document in documents])) for key in keys}


__all__ = [
    "SetupEMA",
    "SetupTrainer",
    "SetupUpdateResult",
    "gradient_norm",
    "json_document_digest",
    "setup_batch_loss",
]
