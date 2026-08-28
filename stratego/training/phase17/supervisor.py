"""Phase 17 Agent 4: the collapse supervisor.

Specification sources: common contract section 13, Agent 4 instruction section
9, operator decision D9-B section 6.

What a supervisor may and may not do
------------------------------------
It may checkpoint and stop. It may **not** change a learning rate, a KL target,
an entropy coefficient, the population size, an epoch count, the setup batch,
or a benchmark case. That is not a style preference: a guard that adjusts the
thing it is measuring stops being evidence about the run and becomes part of
the recipe, and the run's telemetry would then describe a schedule nobody
froze. Every method here returns a verdict; nothing here writes a
hyperparameter.

Immediate versus persistent
---------------------------
An immediate predicate (`I1`..`I7`) fires on one observation, because each one
means the data already recorded is wrong. A persistent predicate (`P1`..`P8`)
needs its frozen consecutive count, because one noisy EWR, KL or entropy
reading is a warning -- contract section 13's `single_reading_rule`.

Consecutive counting has a reset rule, and it is per predicate: a reading that
does *not* trip resets that predicate's run to zero. Without the reset, three
trips spread across a twelve-hour run would eventually stop a healthy job.

The D9-B exception, stated precisely
------------------------------------
`P4` -- setup mean prefix entropy below 60% of its initial baseline for three
checks -- remains the default **production** stop predicate. During Agent 4's
bounded integration rehearsal it is recorded as a *diagnostic*: the verdict
still carries the full evidence payload and the consecutive count, and
`severity` becomes `diagnostic` instead of `stop`. Nothing else moves.
`P5` (flag effective support below four) and every absolute floor stay hard in
both modes, and no threshold is ever rewritten to manufacture a pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .move_contract import Phase17MoveError

SUPERVISOR_VERSION = "phase17_collapse_supervisor_v1"

#: Contract section 13 thresholds. Frozen here; the supervisor reads them and
#: never writes them.
MOVE_KL_HARD_LIMIT = 0.08
SETUP_KL_HARD_LIMIT = 0.08
FLAG_EFFECTIVE_SUPPORT_FLOOR = 4.0
EWR_COLLAPSE_DROP = 0.15
MOVE_ENTROPY_COLLAPSE_FRACTION = 0.25
SETUP_ENTROPY_RELATIVE_FRACTION = 0.60

SEVERITY_STOP = "stop"
SEVERITY_WARNING = "warning"
SEVERITY_DIAGNOSTIC = "diagnostic"

MODE_PRODUCTION = "production"
MODE_INTEGRATION = "integration"


class Phase17SupervisorError(Phase17MoveError):
    """The supervisor was configured or driven outside its contract."""


@dataclass
class Predicate:
    """One stop rule: its code, its consecutive state, and its reset rule."""

    code: str
    description: str
    consecutive_required: int
    severity: str = SEVERITY_STOP
    consecutive: int = 0
    trips: int = 0
    last_evidence: dict = field(default_factory=dict)

    def observe(self, tripped: bool, evidence: dict) -> dict:
        """Fold one reading in. Returns the verdict for this observation."""
        if tripped:
            self.consecutive += 1
            self.trips += 1
            self.last_evidence = dict(evidence)
        else:
            self.consecutive = 0
        fired = tripped and self.consecutive >= self.consecutive_required
        return {
            "code": self.code,
            "description": self.description,
            "tripped": bool(tripped),
            "fired": bool(fired),
            "consecutive": int(self.consecutive),
            "consecutive_required": int(self.consecutive_required),
            "severity": self.severity if fired else (SEVERITY_WARNING if tripped else "ok"),
            "evidence": dict(evidence),
        }

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "description": self.description,
            "consecutive": int(self.consecutive),
            "consecutive_required": int(self.consecutive_required),
            "trips": int(self.trips),
            "severity": self.severity,
            "last_evidence": dict(self.last_evidence),
        }


def _immediate(code: str, description: str) -> Predicate:
    return Predicate(code=code, description=description, consecutive_required=1)


class CollapseSupervisor:
    """Every contract section 13 predicate, with frozen thresholds.

    `mode` selects only how `P4` is *reported*. It changes no threshold, no
    consecutive count and no other predicate.
    """

    def __init__(
        self,
        *,
        run_id: str,
        mode: str = MODE_PRODUCTION,
        setup_entropy_baseline: float,
        flag_support_floor: float = FLAG_EFFECTIVE_SUPPORT_FLOOR,
        setup_kl_hard_limit: float = SETUP_KL_HARD_LIMIT,
        move_kl_hard_limit: float = MOVE_KL_HARD_LIMIT,
    ) -> None:
        if mode not in (MODE_PRODUCTION, MODE_INTEGRATION):
            raise Phase17SupervisorError(f"unknown supervisor mode {mode!r}")
        if not setup_entropy_baseline > 0.0:
            raise Phase17SupervisorError(
                "the setup entropy baseline must be positive; it is the initial "
                "masked model's measured mean prefix entropy, never a default"
            )
        self.run_id = run_id
        self.mode = mode
        self.setup_entropy_baseline = float(setup_entropy_baseline)
        self.setup_entropy_floor = (
            self.setup_entropy_baseline * SETUP_ENTROPY_RELATIVE_FRACTION
        )
        self.flag_support_floor = float(flag_support_floor)
        self.setup_kl_hard_limit = float(setup_kl_hard_limit)
        self.move_kl_hard_limit = float(move_kl_hard_limit)
        self.move_entropy_first_hour_median: "float | None" = None
        self.hour0_ewr: "float | None" = None
        self.stopped: "dict | None" = None
        self.warnings: list = []
        self.verdicts: list = []

        self.predicates = {
            "I1": _immediate("I1", "rules, orientation, legality, candidate or digest mismatch"),
            "I2": _immediate("I2", "a decision recorded under the wrong current move-policy digest"),
            "I3": _immediate("I3", "nonfinite loss, gradient, parameter or schedule value"),
            "I4": _immediate("I4", "setup generation/masking failure or silent fallback attempt"),
            "I5": _immediate("I5", "search or a non-current training opponent entered collection"),
            "I6": _immediate("I6", "evaluation result bound to the wrong candidate or benchmark"),
            "I7": _immediate("I7", "unrecoverable checkpoint/resume identity failure"),
            "P1": Predicate("P1", f"fixed-pack EWR at least {EWR_COLLAPSE_DROP} below hour 0", 3),
            "P2": Predicate("P2", f"move mean KL above {move_kl_hard_limit}", 3),
            "P3": Predicate("P3", f"setup KL above {setup_kl_hard_limit}", 3),
            "P4": Predicate(
                "P4",
                f"setup mean prefix entropy below {SETUP_ENTROPY_RELATIVE_FRACTION:.0%} "
                f"of its initial baseline ({self.setup_entropy_floor:.10f} nats)",
                3,
                severity=(
                    SEVERITY_DIAGNOSTIC if mode == MODE_INTEGRATION else SEVERITY_STOP
                ),
            ),
            "P5": Predicate("P5", f"flag effective support below {flag_support_floor}", 1),
            "P6": Predicate("P6", "move entropy below 25% of its first-hour median", 5),
            "P7": Predicate(
                "P7",
                "no setup optimizer update for one complete 30-minute interval "
                "after warm-up while games and setup episodes complete",
                1,
            ),
            "P8": Predicate("P8", "setup queue age or backlog above the frozen ceiling", 3),
        }

    # -- recording ---------------------------------------------------------

    def _record(self, verdict: dict) -> dict:
        self.verdicts.append(verdict)
        if verdict["fired"] and verdict["severity"] == SEVERITY_STOP:
            if self.stopped is None:
                self.stopped = dict(verdict)
        elif verdict["tripped"]:
            self.warnings.append(
                {
                    "code": verdict["code"],
                    "severity": verdict["severity"],
                    "consecutive": verdict["consecutive"],
                    "evidence": verdict["evidence"],
                }
            )
        return verdict

    def observe(self, code: str, tripped: bool, evidence: "dict | None" = None) -> dict:
        """Fold one reading into one named predicate."""
        predicate = self.predicates.get(code)
        if predicate is None:
            raise Phase17SupervisorError(f"unknown stop predicate {code!r}")
        return self._record(predicate.observe(bool(tripped), evidence or {}))

    # -- the immediate family ---------------------------------------------

    def check_identity(self, ok: bool, evidence: dict) -> dict:
        return self.observe("I1", not ok, evidence)

    #: Ledger fields `I2` and `I5` are decided from. Read by name and required:
    #: Agent 1's encoding rules say an absent required field fails closed, and a
    #: `.get(name, 0)` here would turn a renamed ledger field into a guard that
    #: passes forever.
    REQUIRED_LEDGER_FIELDS = (
        "unknown_model_states",
        "rule_or_stress_decisions",
        "historical_participants",
        "search_participants",
    )

    def check_participant_ledger(self, ledger: dict) -> list:
        """`I2` and `I5` from the collector's own runtime ledger."""
        missing = [name for name in self.REQUIRED_LEDGER_FIELDS if name not in ledger]
        if missing:
            raise Phase17SupervisorError(
                f"the participant ledger is missing {missing}; refusing to "
                "evaluate I2/I5 against a ledger that cannot answer them"
            )
        unknown = ledger["unknown_model_states"] or {}
        stale = self.observe(
            "I2",
            bool(unknown),
            {"unknown_model_states": dict(unknown)},
        )
        counts = {
            name: int(ledger[name])
            for name in (
                "rule_or_stress_decisions",
                "historical_participants",
                "search_participants",
            )
        }
        foreign = self.observe("I5", any(counts.values()), counts)
        return [stale, foreign]

    def check_finite(self, values: dict) -> dict:
        """`I3` over every named scalar an iteration produced."""
        offenders = {
            name: value
            for name, value in values.items()
            if value is not None and not _is_finite(value)
        }
        return self.observe("I3", bool(offenders), {"nonfinite": offenders})

    def check_setup_generation(self, *, legality_failures: int, orientation_failures: int, fallback_attempts: int) -> dict:
        return self.observe(
            "I4",
            bool(legality_failures or orientation_failures or fallback_attempts),
            {
                "legality_failures": int(legality_failures),
                "orientation_failures": int(orientation_failures),
                "fallback_attempts": int(fallback_attempts),
            },
        )

    def check_evaluation_binding(self, ok: bool, evidence: dict) -> dict:
        return self.observe("I6", not ok, evidence)

    def check_checkpoint_identity(self, ok: bool, evidence: dict) -> dict:
        return self.observe("I7", not ok, evidence)

    # -- the persistent family --------------------------------------------

    def observe_ewr(self, ewr: float, *, hour0: "float | None" = None) -> dict:
        if hour0 is not None:
            self.hour0_ewr = float(hour0)
        if self.hour0_ewr is None:
            self.hour0_ewr = float(ewr)
        drop = self.hour0_ewr - float(ewr)
        return self.observe(
            "P1",
            drop >= EWR_COLLAPSE_DROP,
            {"ewr": float(ewr), "hour0_ewr": self.hour0_ewr, "drop": drop},
        )

    def observe_move_kl(self, mean_kl: float) -> dict:
        return self.observe(
            "P2",
            float(mean_kl) > self.move_kl_hard_limit,
            {"mean_kl": float(mean_kl), "limit": self.move_kl_hard_limit},
        )

    def observe_setup_kl(self, control_kl: float) -> dict:
        return self.observe(
            "P3",
            float(control_kl) > self.setup_kl_hard_limit,
            {"control_kl": float(control_kl), "limit": self.setup_kl_hard_limit},
        )

    def observe_setup_entropy(self, mean_prefix_entropy: float) -> dict:
        """`P4`. Diagnostic in integration mode, a stop in production mode."""
        value = float(mean_prefix_entropy)
        return self.observe(
            "P4",
            value < self.setup_entropy_floor,
            {
                "mean_prefix_entropy_nats": value,
                "baseline_nats": self.setup_entropy_baseline,
                "floor_nats": self.setup_entropy_floor,
                "percent_of_baseline": 100.0 * value / self.setup_entropy_baseline,
                "mode": self.mode,
                "production_rule": (
                    "the 60%/three-check rule remains the production stop "
                    "predicate; decision D9-B changes only Agent 4's bounded "
                    "integration interpretation"
                ),
            },
        )

    def observe_flag_support(self, flag_effective_support: float) -> dict:
        value = float(flag_effective_support)
        return self.observe(
            "P5",
            value < self.flag_support_floor,
            {"flag_effective_support": value, "floor": self.flag_support_floor},
        )

    def observe_move_entropy(self, entropy: float, *, first_hour_median: "float | None" = None) -> dict:
        if first_hour_median is not None:
            self.move_entropy_first_hour_median = float(first_hour_median)
        median = self.move_entropy_first_hour_median
        if median is None:
            return self.observe(
                "P6", False, {"entropy": float(entropy), "first_hour_median": None}
            )
        floor = median * MOVE_ENTROPY_COLLAPSE_FRACTION
        return self.observe(
            "P6",
            float(entropy) < floor,
            {"entropy": float(entropy), "first_hour_median": median, "floor": floor},
        )

    def observe_setup_update_activity(self, *, updated: bool, interval_complete: bool, warmed_up: bool, episodes_available: bool) -> dict:
        """`P7`: silence for a whole cadence interval while work was available."""
        tripped = bool(
            interval_complete and warmed_up and episodes_available and not updated
        )
        return self.observe(
            "P7",
            tripped,
            {
                "updated": bool(updated),
                "interval_complete": bool(interval_complete),
                "warmed_up": bool(warmed_up),
                "episodes_available": bool(episodes_available),
            },
        )

    def observe_queue(self, alarms: dict) -> dict:
        """`P8`. Required fields, not defaults: a malformed alarm document must
        not read as "not over"."""
        for name in ("backlog", "age"):
            if name not in alarms or "over" not in alarms[name]:
                raise Phase17SupervisorError(
                    f"the queue alarm document has no {name!r}.over; refusing to "
                    "evaluate P8 against a document that cannot answer it"
                )
        backlog, age = alarms["backlog"], alarms["age"]
        return self.observe(
            "P8",
            bool(backlog["over"] or age["over"]),
            {"backlog": dict(backlog), "age": dict(age)},
        )

    # -- verdicts ----------------------------------------------------------

    @property
    def should_stop(self) -> bool:
        return self.stopped is not None

    def stop_record(self) -> "dict | None":
        """The durable record of why the run stopped, or `None`."""
        if self.stopped is None:
            return None
        return {
            "supervisor_version": SUPERVISOR_VERSION,
            "run_id": self.run_id,
            "mode": self.mode,
            "code": self.stopped["code"],
            "description": self.stopped["description"],
            "consecutive": self.stopped["consecutive"],
            "consecutive_required": self.stopped["consecutive_required"],
            "evidence": self.stopped["evidence"],
            "action": "safe checkpoint then exit; no hyperparameter was changed",
        }

    def document(self) -> dict:
        return {
            "supervisor_version": SUPERVISOR_VERSION,
            "run_id": self.run_id,
            "mode": self.mode,
            "may_not_change": [
                "learning rate",
                "KL targets",
                "entropy coefficients",
                "population size",
                "epoch counts",
                "setup batch",
                "benchmark cases",
            ],
            "thresholds": {
                "move_kl_hard_limit": self.move_kl_hard_limit,
                "setup_kl_hard_limit": self.setup_kl_hard_limit,
                "flag_effective_support_floor": self.flag_support_floor,
                "setup_entropy_baseline_nats": self.setup_entropy_baseline,
                "setup_entropy_floor_nats": self.setup_entropy_floor,
                "setup_entropy_relative_fraction": SETUP_ENTROPY_RELATIVE_FRACTION,
                "ewr_collapse_drop": EWR_COLLAPSE_DROP,
                "move_entropy_collapse_fraction": MOVE_ENTROPY_COLLAPSE_FRACTION,
            },
            "d9b_exception": (
                "P4 is reported as a DIAGNOSTIC in integration mode only. It "
                "remains the production stop predicate and its threshold is "
                "unchanged. Agent 6 owns the production decision."
            ),
            "predicates": {code: p.to_dict() for code, p in self.predicates.items()},
            "warnings": list(self.warnings),
            "stopped": self.stop_record(),
        }

    def state_document(self) -> dict:
        """Consecutive-count state, so a resume damps from where it left off."""
        return {
            "supervisor_version": SUPERVISOR_VERSION,
            "mode": self.mode,
            "hour0_ewr": self.hour0_ewr,
            "move_entropy_first_hour_median": self.move_entropy_first_hour_median,
            "predicates": {
                code: {"consecutive": p.consecutive, "trips": p.trips}
                for code, p in self.predicates.items()
            },
        }

    def load_state_document(self, payload: dict) -> None:
        if payload.get("supervisor_version") != SUPERVISOR_VERSION:
            raise Phase17SupervisorError(
                f"supervisor state is {payload.get('supervisor_version')!r}, not "
                f"{SUPERVISOR_VERSION}"
            )
        self.hour0_ewr = payload.get("hour0_ewr")
        self.move_entropy_first_hour_median = payload.get("move_entropy_first_hour_median")
        for code, state in (payload.get("predicates") or {}).items():
            predicate = self.predicates.get(code)
            if predicate is None:
                raise Phase17SupervisorError(
                    f"supervisor state names unknown predicate {code!r}"
                )
            predicate.consecutive = int(state["consecutive"])
            predicate.trips = int(state["trips"])


def _is_finite(value) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and number not in (float("inf"), float("-inf"))


__all__ = [
    "CollapseSupervisor",
    "EWR_COLLAPSE_DROP",
    "FLAG_EFFECTIVE_SUPPORT_FLOOR",
    "MODE_INTEGRATION",
    "MODE_PRODUCTION",
    "MOVE_ENTROPY_COLLAPSE_FRACTION",
    "MOVE_KL_HARD_LIMIT",
    "Phase17SupervisorError",
    "Predicate",
    "SETUP_ENTROPY_RELATIVE_FRACTION",
    "SETUP_KL_HARD_LIMIT",
    "SEVERITY_DIAGNOSTIC",
    "SEVERITY_STOP",
    "SEVERITY_WARNING",
    "SUPERVISOR_VERSION",
]
