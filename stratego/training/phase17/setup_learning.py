"""Phase 17: the setup update -- advantage, losses, fixed behavior KL, EMA.

Specification sources:

- `09_OPERATOR_DECISION_D10_SIMPLIFIED_PAPER_TANDEM.md` sections 4 and 7
- `00_PHASE_17_SEQUENCE_AND_COMMON_CONTRACT.md` sections 8 and 10
- `reports/phase17/ataraxos_method_map_v1.md` rows S06-S17

The equation, once
------------------
```text
delta_k = (o - E[v_theta_t(sigma_k)]) + alpha(n) * (I_k - h_theta_t(sigma_k))
r_k     = pi_theta(t_k | sigma_k) / pi_theta_t(t_k | sigma_k)
L_pi    = -mean_k min(r_k * delta_k, clip(r_k, 0.8, 1.2) * delta_k)
L_v     = mean_k CE(wdl_theta(sigma_k), outcome_class)
L_h     = mean_k (I_k/10 - h_theta(sigma_k))^2
L_setup = L_pi + 0.5 * L_v + 1.0 * L_h + 0.1 * KL(pi_theta || pi_theta_t)
alpha(n)= 0.1 * n**-0.3, n the shared one-based global tandem iteration
```

This is the paper's printed advantage, used directly (operator decision D10
section 4). Two locally invented pieces are gone:

- D7-B's uncentered `0.9 * alpha * (I/10)` bonus. It was introduced because the
  centered residual `alpha*(I/10 - h)` converges to zero by construction --
  `L_h` is actively training `h` toward `I/10` -- and it was rescaled into
  normalized units so it stayed commensurate with an outcome term bounded by 2.
- D5's adaptive reverse-KL controller, replaced by the fixed coefficient 0.1.

The advantage's entropy term therefore mixes units on purpose: `I` is in raw
nats and `h` is the recorded prediction of `I/10`. D10 saw that and kept it --
"do not add a compensating scale, floor, centering rule, horizon map, or
controller" -- so once `h` converges the term is about `0.9 * alpha * I` in
nats, several times an outcome term bounded by 2. That ratio is a property of
the recipe being tested, not a defect to patch here, and `advantage_terms`
measures both magnitudes every iteration so the 12-hour curve can be read
against the balance that actually held rather than against an estimate.

Everything subscripted `theta_t` is read from the episode's recorded behavior
fields and never recomputed. That is what makes a completed game valid: its
advantage and its ratio denominator both belong to the snapshot that drew it.
Recomputing either from the latest network would turn PPO's importance ratio
into 1.0 and quietly delete the correction it exists to apply.

Off-policy by design
--------------------
Row S05: a setup is drawn once at game creation and stays bound to that game,
unlike the move side's per-decision rebind. So `policy_age > 0` is normal, is
telemetry, and is never a reason to refuse an episode.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch

from ...engine.constants import NUM_PIECE_TYPES
from .setup_contract import (
    MASKED_LOGIT,
    SETUP_ADAM_BETAS,
    SETUP_ADAM_EPSILON,
    SETUP_CONDITIONAL_ENTROPY_NORMALIZER,
    SETUP_EQUATION_VERSION,
    SETUP_PREFIXES,
    SETUP_RECIPE_VERSION,
    SETUP_SEQUENCE_LENGTH,
    START_TOKEN,
    Phase17SetupError,
    SetupTrainingConfig,
    derive_shuffle_seed,
)
from .setup_episode import SetupEpisode, SetupEpisodeQueue, wdl_class
from .setup_model import Phase17SetupModel, build_setup_model
from .setup_sampling import batched_remaining

_LOG_EPSILON = 1e-12


# ---------------------------------------------------------------------------
# The EMA
# ---------------------------------------------------------------------------


class SetupEMA:
    """Setup-parameter EMA at decay 0.999, updated after the complete update.

    Row S16 and common contract section 10: RAW generates every setup, EMA is
    exported for evaluation only. Updating inside the epoch loop would make
    the EMA a running average of five intermediate states rather than of the
    iteration's outputs, which is why `update` is called once by the trainer
    after all epochs finish.
    """

    def __init__(self, model: Phase17SetupModel, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise Phase17SetupError(f"EMA decay must be in (0, 1), got {decay}")
        self.decay = float(decay)
        self.shadow = {
            name: tensor.detach().clone().to(torch.float32)
            for name, tensor in model.state_dict().items()
        }
        #: The device the shadow lives on -- the model's, because `update`
        #: accumulates against live parameters in place. Captured here so a
        #: restore can put a CPU-serialized state back where the arithmetic
        #: expects it. See `load_state_dict`.
        self.device = next(
            (tensor.device for tensor in self.shadow.values()),
            torch.device("cpu"),
        )
        self.updates = 0

    @torch.no_grad()
    def update(self, model: Phase17SetupModel) -> None:
        for name, tensor in model.state_dict().items():
            current = tensor.detach().to(torch.float32)
            self.shadow[name].mul_(self.decay).add_(current, alpha=1.0 - self.decay)
        self.updates += 1

    def state_dict(self) -> dict:
        return {name: tensor.clone() for name, tensor in self.shadow.items()}

    def load_state_dict(self, state: dict) -> None:
        """Restore the shadow onto the device the model is on.

        The device is not incidental. A paired checkpoint serializes the EMA to
        CPU and `read_joint_checkpoint` loads it with `map_location="cpu"`, so a
        restore that preserved the payload's device would leave a CPU shadow
        accumulating against MPS parameters -- and `update` raises on the first
        setup update after the resume, which is several hours into a run. It
        does not surface on a CPU-device rehearsal, which is why it survived
        until the D10 smoke resumed on the production device.
        """
        missing = set(self.shadow) - set(state)
        if missing:
            raise Phase17SetupError(f"EMA state is missing {sorted(missing)}")
        self.shadow = {
            name: tensor.detach().to(device=self.device, dtype=torch.float32).clone()
            for name, tensor in state.items()
        }

    def as_model(self, device: str = "cpu") -> Phase17SetupModel:
        """A materialised EMA model. Evaluation-only by contract."""
        model = build_setup_model(device=device)
        model.load_state_dict({name: tensor.to(device) for name, tensor in self.shadow.items()})
        return model


# ---------------------------------------------------------------------------
# Batch assembly
# ---------------------------------------------------------------------------


@dataclass
class SetupBatch:
    """One consumable batch of completed episodes, as device tensors."""

    sequence: torch.Tensor          # [B, 41] start token + 40 placements
    tokens: torch.Tensor            # [B, 40] selected type per prefix
    masks: torch.Tensor             # [B, 40, 12] legal-by-inventory
    behavior_log_probabilities: torch.Tensor   # [B, 40]
    behavior_probabilities: torch.Tensor       # [B, 40, 12]
    advantage: torch.Tensor         # [B, 40]
    normalized_information: torch.Tensor       # [B, 40] I/10
    outcome_class: torch.Tensor     # [B]
    outcome: torch.Tensor           # [B]
    expected_value: torch.Tensor    # [B, 40]
    behavior_conditional_entropy: torch.Tensor  # [B, 40]
    episode_count: int
    alpha: float

    def to(self, device: str) -> "SetupBatch":
        moved = {
            name: value.to(device) if isinstance(value, torch.Tensor) else value
            for name, value in self.__dict__.items()
        }
        return SetupBatch(**moved)


def expected_value_from_wdl(wdl: np.ndarray) -> np.ndarray:
    """`E[v] = p_win - p_loss` under the behavior W/D/L prediction.

    The classes are ordered win/draw/loss *from the episode owner's own
    perspective*, so one head reads correctly for both colours and no sign
    flip is applied anywhere downstream.
    """
    probabilities = np.asarray(wdl, dtype=np.float64)
    return (probabilities[..., 0] - probabilities[..., 2]).astype(np.float32)


def setup_advantage(episode: SetupEpisode, alpha: float) -> np.ndarray:
    """`delta_k` at all 40 prefixes, from behavior-snapshot quantities only.

    The paper's printed form, `(o - E[v]) + alpha * (I - h)`, with `I` the
    realized suffix information in nats and `h` the conditional-entropy
    prediction the raw setup snapshot recorded when it drew the episode.

    Constant across the five epochs by construction, so it is computed once
    per episode here rather than inside the epoch loop.
    """
    if episode.outcome is None:
        raise Phase17SetupError(
            f"episode {episode.game_id}/{episode.color} has no outcome; "
            "an open episode can never enter the update"
        )
    outcome = float(episode.outcome)
    expected = expected_value_from_wdl(episode.prefix_wdl_predictions).astype(np.float64)
    information = np.asarray(episode.suffix_information_content, dtype=np.float64)
    predicted = np.asarray(
        episode.prefix_conditional_entropy_predictions, dtype=np.float64
    )
    outcome_term = outcome - expected
    entropy_term = float(alpha) * (information - predicted)
    return (outcome_term + entropy_term).astype(np.float32)


def advantage_terms(episode: SetupEpisode, alpha: float) -> dict:
    """The two advantage terms separately -- row S07's required telemetry.

    D10 section 4: "Record the component magnitudes; do not add a compensating
    scale, floor, centering rule, horizon map, or controller." That is what
    this is for. The entropy term is `alpha * (I - h)` with `I` in nats and `h`
    predicting `I/10`, so it settles near `0.9 * alpha * I` and is reported
    alongside an outcome term bounded by 2. The ratio between them is what the
    12-hour curve has to be read against, so it is measured every iteration
    rather than argued about once.
    """
    outcome = float(episode.outcome)
    expected = expected_value_from_wdl(episode.prefix_wdl_predictions).astype(np.float64)
    information = np.asarray(episode.suffix_information_content, dtype=np.float64)
    predicted = np.asarray(episode.prefix_conditional_entropy_predictions, dtype=np.float64)
    normalized = information * SETUP_CONDITIONAL_ENTROPY_NORMALIZER
    outcome_term = outcome - expected
    entropy_term = alpha * (information - predicted)
    return {
        "outcome_term_mean": float(np.mean(outcome_term)),
        "entropy_term_mean": float(np.mean(entropy_term)),
        "outcome_term_abs_mean": float(np.mean(np.abs(outcome_term))),
        "entropy_term_abs_mean": float(np.mean(np.abs(entropy_term))),
        "outcome_term_max_abs": float(np.max(np.abs(outcome_term))),
        "entropy_term_max_abs": float(np.max(np.abs(entropy_term))),
        "entropy_to_outcome_abs_ratio": float(
            np.mean(np.abs(entropy_term)) / max(float(np.mean(np.abs(outcome_term))), 1e-12)
        ),
        "information_nats_mean": float(np.mean(information)),
        "normalized_information_mean": float(np.mean(normalized)),
        "predicted_conditional_entropy_mean": float(np.mean(predicted)),
        # `L_h` trains `h` toward `I/10`; this is how far it has got.
        # Descriptive only -- unlike the retired D7-B form, the advantage does
        # not go inert when this reaches zero, because it subtracts `h` from
        # `I` in nats rather than from `I/10`.
        "conditional_entropy_residual_abs_mean": float(
            np.mean(np.abs(normalized - predicted))
        ),
        "expected_value_mean": float(np.mean(expected)),
        "alpha": float(alpha),
    }


def build_batch(
    episodes: "list[SetupEpisode]", *, alpha: float, device: str = "cpu"
) -> SetupBatch:
    """Assemble a batch, re-deriving each mask from the tokens as a check.

    The recorded mask and the mask the prefix implies must agree. They are
    computed by different code paths at different times, so a disagreement
    means the episode's tokens and its behavior record came apart -- which
    would silently corrupt every ratio in the batch.
    """
    if not episodes:
        raise Phase17SetupError("build_batch needs at least one episode")

    count = len(episodes)
    sequence = np.full((count, SETUP_SEQUENCE_LENGTH), START_TOKEN, dtype=np.int64)
    tokens = np.zeros((count, SETUP_PREFIXES), dtype=np.int64)
    masks = np.zeros((count, SETUP_PREFIXES, NUM_PIECE_TYPES), dtype=bool)
    behavior_log = np.zeros((count, SETUP_PREFIXES), dtype=np.float32)
    behavior_probabilities = np.zeros((count, SETUP_PREFIXES, NUM_PIECE_TYPES), dtype=np.float32)
    advantage = np.zeros((count, SETUP_PREFIXES), dtype=np.float32)
    normalized_information = np.zeros((count, SETUP_PREFIXES), dtype=np.float32)
    expected_value = np.zeros((count, SETUP_PREFIXES), dtype=np.float32)
    behavior_entropy = np.zeros((count, SETUP_PREFIXES), dtype=np.float32)
    outcome_class = np.zeros(count, dtype=np.int64)
    outcome = np.zeros(count, dtype=np.float32)

    for row, episode in enumerate(episodes):
        drawn = np.asarray(episode.tokens, dtype=np.int64)
        tokens[row] = drawn
        sequence[row, 1:] = drawn
        masks[row] = np.asarray(episode.inventory_masks, dtype=bool)
        behavior_log[row] = episode.behavior_log_probabilities
        behavior_probabilities[row] = episode.behavior_probabilities
        advantage[row] = setup_advantage(episode, alpha)
        normalized_information[row] = (
            np.asarray(episode.suffix_information_content, dtype=np.float32)
            * SETUP_CONDITIONAL_ENTROPY_NORMALIZER
        )
        expected_value[row] = expected_value_from_wdl(episode.prefix_wdl_predictions)
        behavior_entropy[row] = episode.prefix_conditional_entropy_predictions
        outcome_class[row] = wdl_class(int(episode.outcome))
        outcome[row] = float(episode.outcome)

    token_tensor = torch.as_tensor(tokens)
    for prefix in range(SETUP_PREFIXES):
        derived = (batched_remaining(token_tensor, prefix) > 0).numpy()
        if not np.array_equal(derived, masks[:, prefix]):
            raise Phase17SetupError(
                f"recorded inventory mask at prefix {prefix} disagrees with the mask "
                "its own tokens imply; the episode's behavior record is corrupt"
            )

    return SetupBatch(
        sequence=torch.as_tensor(sequence, device=device),
        tokens=torch.as_tensor(tokens, device=device),
        masks=torch.as_tensor(masks, device=device),
        behavior_log_probabilities=torch.as_tensor(behavior_log, device=device),
        behavior_probabilities=torch.as_tensor(behavior_probabilities, device=device),
        advantage=torch.as_tensor(advantage, device=device),
        normalized_information=torch.as_tensor(normalized_information, device=device),
        outcome_class=torch.as_tensor(outcome_class, device=device),
        outcome=torch.as_tensor(outcome, device=device),
        expected_value=torch.as_tensor(expected_value, device=device),
        behavior_conditional_entropy=torch.as_tensor(behavior_entropy, device=device),
        episode_count=count,
        alpha=float(alpha),
    )


# ---------------------------------------------------------------------------
# The loss
# ---------------------------------------------------------------------------


def setup_batch_loss(
    model: Phase17SetupModel, batch: SetupBatch, *, config: SetupTrainingConfig
) -> tuple:
    """Every term of `L_setup`, each returned under its own name.

    Section 5 requires distinct logged terms; they are returned as separate
    tensors rather than summed inside so that no caller can report the KL as
    the policy loss or the entropy contribution as the entropy coefficient.
    """
    outputs = model(batch.sequence)
    logits = outputs["piece_logits"]
    excluded = logits.to(torch.float32).masked_fill(~batch.masks, MASKED_LOGIT)
    log_probabilities = torch.log_softmax(excluded, dim=-1)
    probabilities = log_probabilities.exp()

    selected = log_probabilities.gather(2, batch.tokens.unsqueeze(-1)).squeeze(-1)
    ratio = torch.exp(selected - batch.behavior_log_probabilities)

    clipped = torch.clamp(
        ratio, 1.0 - config.ppo_clip_epsilon, 1.0 + config.ppo_clip_epsilon
    )
    policy_loss = -torch.min(ratio * batch.advantage, clipped * batch.advantage).mean()
    clip_fraction = ((ratio < 1.0 - config.ppo_clip_epsilon) | (ratio > 1.0 + config.ppo_clip_epsilon)).to(
        torch.float32
    ).mean()

    value_loss = torch.nn.functional.cross_entropy(
        outputs["wdl_logits"].reshape(-1, 3),
        batch.outcome_class.unsqueeze(1).expand(-1, SETUP_PREFIXES).reshape(-1),
    )

    entropy_loss = ((batch.normalized_information - outputs["conditional_entropy"]) ** 2).mean()

    # Reverse KL, D(pi_current || pi_behavior), over the masked 12-way
    # distribution -- the paper's direction (row S11), NOT the move
    # controller's forward direction. Its coefficient is the FIXED 0.1 of D10
    # section 4, read straight off the config: there is no controller to ask
    # and no beta to carry between iterations.
    behavior_log = torch.log(batch.behavior_probabilities.clamp_min(_LOG_EPSILON))
    surprise = (log_probabilities - behavior_log).masked_fill(~batch.masks, 0.0)
    per_prefix_kl = (probabilities * surprise).sum(dim=-1)
    kl = per_prefix_kl.mean()

    total = (
        policy_loss
        + config.value_loss_weight * value_loss
        + config.conditional_entropy_loss_weight * entropy_loss
        + float(config.behavior_kl_coefficient) * kl
    )

    with torch.no_grad():
        policy_entropy = -(
            probabilities * log_probabilities.masked_fill(~batch.masks, 0.0)
        ).sum(dim=-1)

    terms = {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "conditional_entropy_loss": entropy_loss,
        "behavior_kl": kl,
        "total_loss": total,
        "clip_fraction": clip_fraction,
        "ratio_mean": ratio.mean(),
        "mean_prefix_entropy_nats": policy_entropy.mean(),
        "advantage_mean": batch.advantage.mean(),
        "advantage_std": batch.advantage.std(unbiased=False),
    }
    return total, terms


# ---------------------------------------------------------------------------
# The trainer
# ---------------------------------------------------------------------------


@dataclass
class SetupUpdateResult:
    """What one complete setup iteration did, with nothing summarised away."""

    #: The shared global tandem iteration this update belongs to (D10).
    setup_iteration: int
    skipped: bool
    skip_reason: str | None
    episodes_consumed: int
    alpha: float
    #: Fixed by contract; carried on the result so every row states it.
    behavior_kl_coefficient: float
    optimizer_steps: int
    epochs: "list[dict]" = field(default_factory=list)
    digest_before: str | None = None
    digest_after: str | None = None
    gradient_norm_mean: float | None = None
    buffer: dict | None = None
    advantage_telemetry: dict | None = None
    mean_iteration_kl: float | None = None
    final_epoch_kl: float | None = None
    per_epoch_kl: "list[float]" = field(default_factory=list)
    shuffle_orders: "list[list[int]]" = field(default_factory=list)

    def document(self) -> dict:
        return {
            "recipe": SETUP_RECIPE_VERSION,
            "setup_equation_version": SETUP_EQUATION_VERSION,
            "setup_iteration": self.setup_iteration,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "episodes_consumed": self.episodes_consumed,
            "alpha": self.alpha,
            "behavior_kl_coefficient": self.behavior_kl_coefficient,
            "behavior_kl_is_adaptive": False,
            "optimizer_steps": self.optimizer_steps,
            "epochs": self.epochs,
            "setup_raw_model_state_digest_before": self.digest_before,
            "setup_raw_model_state_digest_after": self.digest_after,
            "gradient_norm_mean": self.gradient_norm_mean,
            "mean_iteration_kl": self.mean_iteration_kl,
            "final_epoch_kl": self.final_epoch_kl,
            "per_epoch_kl": self.per_epoch_kl,
            "completed_episode_buffer": self.buffer,
            "advantage_telemetry": self.advantage_telemetry,
            "shuffle_orders": self.shuffle_orders,
        }


class SetupTrainer:
    """Raw weights, optimizer, EMA and pending buffer, updated together.

    Ordering is contractual, not incidental: raw weights update first across
    all five epochs, and the setup EMA updates only after the complete setup
    update.

    One index, shared with the move half
    ------------------------------------
    `setup_iteration` is the GLOBAL tandem iteration the runner is on, not a
    count of setup updates. Operator decision D10 section 4 makes alpha a
    function of that shared index, which settles Agent 4's carry-forward
    `A4-CF6`: the two diverge exactly when a setup update is skipped, and a
    skipped update must still let the run's regularization advance.
    """

    def __init__(
        self,
        model: Phase17SetupModel,
        config: SetupTrainingConfig,
        *,
        queue: SetupEpisodeQueue | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.queue = queue or SetupEpisodeQueue()
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            betas=SETUP_ADAM_BETAS,
            eps=SETUP_ADAM_EPSILON,
        )
        self.ema = SetupEMA(model, config.ema_decay)
        self.setup_iteration = 0
        self.updates = 0
        self.skips = 0
        self.optimizer_step_count = 0

    # -- helpers ---------------------------------------------------------

    def _digest(self) -> str:
        from ..phase9_behavior import state_dict_digest

        return state_dict_digest(self.model)

    def _shuffle(self, count: int, epoch: int) -> np.ndarray:
        """Deterministic per-(iteration, epoch) permutation, recorded verbatim.

        Seeded through the same domain-separated blake2b derivation the rest
        of the phase uses. Python's built-in `hash` is randomized per process,
        so a resume in a new process would silently reshuffle -- which is
        exactly the identity the checkpoint round trip is meant to prove.
        """
        stream = derive_shuffle_seed(
            self.config.run_id, self.setup_iteration, epoch, self.config.seed_offset
        )
        return np.random.RandomState(stream % (2**32)).permutation(count)

    # -- the update ------------------------------------------------------

    def update(self, *, global_iteration: int) -> SetupUpdateResult:
        """Train five epochs on every episode that completed this iteration.

        D10 section 4: every setup episode whose game completed during the
        fixed-transition iteration, both sides, exactly once. There is no
        quota to meet, so there is nothing to be short of -- an empty buffer is
        the only skip, and it is recorded rather than waited out.
        """
        if global_iteration < 1:
            raise Phase17SetupError(
                f"the global tandem iteration is one-based, got {global_iteration}"
            )
        self.setup_iteration = int(global_iteration)
        alpha = self.config.alpha(self.setup_iteration)
        coefficient = float(self.config.behavior_kl_coefficient)

        episodes = self.queue.consume_all(setup_iteration=self.setup_iteration)
        if not episodes:
            self.skips += 1
            return SetupUpdateResult(
                setup_iteration=self.setup_iteration,
                skipped=True,
                skip_reason="no game completed during this fixed-transition iteration",
                episodes_consumed=0,
                alpha=alpha,
                behavior_kl_coefficient=coefficient,
                optimizer_steps=0,
                buffer=self.queue.telemetry(self.setup_iteration).__dict__,
            )

        digest_before = self._digest()
        batch = build_batch(episodes, alpha=alpha, device=self.config.device)
        telemetry = _mean_documents([advantage_terms(episode, alpha) for episode in episodes])

        gradient_norms: "list[float]" = []
        epoch_records: "list[dict]" = []
        iteration_kl: "list[float]" = []
        shuffle_orders: "list[list[int]]" = []
        self.model.train()

        for epoch in range(self.config.epochs_per_iteration):
            order = self._shuffle(len(episodes), epoch)
            shuffle_orders.append([int(value) for value in order])
            epoch_kl: "list[float]" = []
            epoch_terms: "list[dict]" = []
            for start in range(0, len(order), self.config.minibatch_episodes):
                indices = order[start : start + self.config.minibatch_episodes]
                minibatch = _select(batch, indices, self.config.device)
                total, terms = setup_batch_loss(
                    self.model, minibatch, config=self.config
                )
                if not torch.isfinite(total):
                    raise Phase17SetupError(
                        f"non-finite setup loss at iteration {self.setup_iteration} "
                        f"epoch {epoch}: {float(total)}"
                    )
                self.optimizer.zero_grad(set_to_none=True)
                total.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_norm
                )
                if not torch.isfinite(norm):
                    raise Phase17SetupError(
                        f"non-finite setup gradient norm at iteration {self.setup_iteration}"
                    )
                self.optimizer.step()
                self.optimizer_step_count += 1
                gradient_norms.append(float(norm))
                epoch_kl.append(float(terms["behavior_kl"].detach()))
                epoch_terms.append(
                    {name: float(value.detach()) for name, value in terms.items()}
                )

            mean_kl = float(np.mean(epoch_kl))
            iteration_kl.append(mean_kl)
            epoch_records.append(
                {
                    "epoch": epoch,
                    "minibatches": len(epoch_terms),
                    "mean_behavior_kl": mean_kl,
                    "kl_direction": self.config.kl_direction,
                    "behavior_kl_coefficient": coefficient,
                    **_mean_documents(epoch_terms),
                }
            )

        # Both KL readings are TELEMETRY. Nothing consumes them: the
        # coefficient is fixed and no controller steps here.
        #
        # They are still both reported, and the split matters for reading the
        # run. Epoch 0 begins with the current policy sitting exactly on the
        # behavior snapshot, so its KL is near zero by construction rather than
        # by evidence -- Agent 3 measured 0.000187, 0.000151, 0.000081 on its
        # first three iterations against a final-epoch reading around 0.0108.
        # A mean across epochs therefore understates the drift by roughly 2.5x.
        # What a behavior-KL term actually limits is where the policy ENDS UP
        # relative to the snapshot, which is the final epoch's reading.
        mean_iteration_kl = float(np.mean(iteration_kl))
        final_epoch_kl = float(iteration_kl[-1])

        # Raw first, EMA only after the complete update.
        self.ema.update(self.model)
        digest_after = self._digest()
        self.updates += 1

        return SetupUpdateResult(
            setup_iteration=self.setup_iteration,
            skipped=False,
            skip_reason=None,
            episodes_consumed=len(episodes),
            alpha=alpha,
            behavior_kl_coefficient=coefficient,
            optimizer_steps=len(gradient_norms),
            epochs=epoch_records,
            digest_before=digest_before,
            digest_after=digest_after,
            gradient_norm_mean=float(np.mean(gradient_norms)),
            mean_iteration_kl=mean_iteration_kl,
            final_epoch_kl=final_epoch_kl,
            per_epoch_kl=[float(value) for value in iteration_kl],
            buffer=self.queue.telemetry(self.setup_iteration).__dict__,
            advantage_telemetry=telemetry,
            shuffle_orders=shuffle_orders,
        )

    # -- state -----------------------------------------------------------

    def state_document(self) -> dict:
        """Every setup state fragment the paired checkpoint binds."""
        from ..phase9_behavior import state_dict_digest

        ema_model = self.ema.as_model(device="cpu")
        return {
            "recipe": SETUP_RECIPE_VERSION,
            "setup_contract_version": self.config.document()["setup_contract_version"],
            "setup_equation_version": SETUP_EQUATION_VERSION,
            "run_id": self.config.run_id,
            "setup_iteration": self.setup_iteration,
            "setup_updates": self.updates,
            "setup_skips": self.skips,
            "setup_optimizer_step_count": self.optimizer_step_count,
            "setup_raw_state": {
                name: tensor.detach().cpu().clone()
                for name, tensor in self.model.state_dict().items()
            },
            "setup_raw_model_state_digest": state_dict_digest(self.model),
            "setup_ema_state": {
                name: tensor.detach().cpu().clone() for name, tensor in self.ema.state_dict().items()
            },
            "setup_ema_model_state_digest": state_dict_digest(ema_model),
            "setup_ema_updates": self.ema.updates,
            # Deep-copied on purpose. `Optimizer.state_dict` hands back live
            # tensors, so a captured checkpoint would be silently rewritten by
            # the next optimizer step before it ever reached disk.
            "setup_optimizer_state": copy.deepcopy(self.optimizer.state_dict()),
            # A FIXED COEFFICIENT, not controller state. Serialized as a scalar
            # with its direction so no reader can restore it as a beta.
            "setup_behavior_kl": {
                "direction": self.config.kl_direction,
                "coefficient": float(self.config.behavior_kl_coefficient),
                "adaptive": False,
            },
            "setup_scheduler_position": {
                "iteration": self.setup_iteration,
                "alpha": self.config.alpha(max(1, self.setup_iteration)),
                "alpha_formula": "0.1 * n**-0.3",
                "learning_rate": self.config.learning_rate,
            },
            "completed_setup_buffer": self.queue.state_document(),
            "config_digest": self.config.config_digest(),
        }

    def load_state_document(self, document: dict) -> None:
        """Restore every fragment, refusing a partial or mismatched load."""
        required = (
            "recipe",
            "setup_equation_version",
            "run_id",
            "setup_iteration",
            "setup_optimizer_step_count",
            "setup_raw_state",
            "setup_ema_state",
            "setup_optimizer_state",
            "setup_behavior_kl",
            "completed_setup_buffer",
            "config_digest",
        )
        missing = [name for name in required if name not in document]
        if missing:
            raise Phase17SetupError(f"setup state document is missing {missing}")
        if document["recipe"] != SETUP_RECIPE_VERSION:
            raise Phase17SetupError(
                f"setup state was written under recipe {document['recipe']!r}, "
                f"this process is {SETUP_RECIPE_VERSION!r}"
            )
        if document["setup_equation_version"] != SETUP_EQUATION_VERSION:
            raise Phase17SetupError(
                f"setup equation {document['setup_equation_version']!r}, "
                f"expected {SETUP_EQUATION_VERSION!r}"
            )
        if document["run_id"] != self.config.run_id:
            raise Phase17SetupError(
                f"setup state belongs to run {document['run_id']!r}, "
                f"this trainer is {self.config.run_id!r}"
            )
        if document["config_digest"] != self.config.config_digest():
            raise Phase17SetupError(
                "setup state was written under a different setup config digest"
            )
        recorded_kl = document["setup_behavior_kl"]
        if not isinstance(recorded_kl, dict) or "adaptive" not in recorded_kl:
            raise Phase17SetupError(
                "setup_behavior_kl does not declare `adaptive`; refusing a state "
                f"document that cannot answer it: {recorded_kl!r}"
            )
        if recorded_kl["adaptive"]:
            raise Phase17SetupError(
                "setup state carries an ADAPTIVE behavior-KL state; operator "
                "decision D10 retired the controller and this recipe has only a "
                "fixed coefficient"
            )
        if float(recorded_kl["coefficient"]) != float(self.config.behavior_kl_coefficient):
            raise Phase17SetupError(
                f"setup state was written under behavior-KL coefficient "
                f"{recorded_kl['coefficient']}, this process uses "
                f"{self.config.behavior_kl_coefficient}"
            )
        if recorded_kl["direction"] != self.config.kl_direction:
            raise Phase17SetupError(
                f"setup state was written under KL direction "
                f"{recorded_kl['direction']!r}, this process uses "
                f"{self.config.kl_direction!r}"
            )
        device = self.config.device
        self.model.load_state_dict(
            {name: tensor.to(device) for name, tensor in document["setup_raw_state"].items()}
        )
        self.ema.load_state_dict(document["setup_ema_state"])
        self.optimizer.load_state_dict(document["setup_optimizer_state"])
        self.queue = SetupEpisodeQueue.from_state_document(document["completed_setup_buffer"])
        self.setup_iteration = int(document["setup_iteration"])
        self.optimizer_step_count = int(document["setup_optimizer_step_count"])
        self.updates = int(document.get("setup_updates", 0))
        self.skips = int(document.get("setup_skips", 0))
        self.ema.updates = int(document.get("setup_ema_updates", 0))


def _select(batch: SetupBatch, indices: np.ndarray, device: str) -> SetupBatch:
    index = torch.as_tensor(np.asarray(indices, dtype=np.int64), device=device)
    return SetupBatch(
        sequence=batch.sequence.index_select(0, index),
        tokens=batch.tokens.index_select(0, index),
        masks=batch.masks.index_select(0, index),
        behavior_log_probabilities=batch.behavior_log_probabilities.index_select(0, index),
        behavior_probabilities=batch.behavior_probabilities.index_select(0, index),
        advantage=batch.advantage.index_select(0, index),
        normalized_information=batch.normalized_information.index_select(0, index),
        outcome_class=batch.outcome_class.index_select(0, index),
        outcome=batch.outcome.index_select(0, index),
        expected_value=batch.expected_value.index_select(0, index),
        behavior_conditional_entropy=batch.behavior_conditional_entropy.index_select(0, index),
        episode_count=int(index.numel()),
        alpha=batch.alpha,
    )


def _mean_documents(documents: "list[dict]") -> dict:
    if not documents:
        return {}
    keys = documents[0].keys()
    return {key: float(np.mean([document[key] for document in documents])) for key in keys}


__all__ = [
    "SetupBatch",
    "SetupEMA",
    "SetupTrainer",
    "SetupUpdateResult",
    "advantage_terms",
    "build_batch",
    "expected_value_from_wdl",
    "setup_advantage",
    "setup_batch_loss",
]
