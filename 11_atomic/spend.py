"""API spend gate (workorder A5 / G40) — the round's hard money ceiling.

The $30 cap is a WHOLE-ROUND cumulative limit, so the counter persists across
process restarts (every bench arm boots a fresh service) in a small JSON
state file. Crossing the cap is a loud failure — `BudgetExceededError` —
never a silent continue or a degraded mode.

The state file lives outside git (`adapter.spend_state_path`, default
`06_results/spend/spend_state.json`); it is infrastructure state, not
evidence — the ledger's measured COST/MODEL_CALLS/TOKENS columns (O2) remain
the per-run evidence, and the report cites this file for the round total.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path


class BudgetExceededError(RuntimeError):
    """Raised before an LLM call that would push cumulative spend to/over the
    configured cap. Callers must fail loudly (HTTP 5xx), never skip silently."""


def _empty_state() -> dict:
    return {"total_usd": 0.0, "model_calls": 0, "tokens_in": 0, "tokens_out": 0, "updated_at": None}


class SpendGate:
    def __init__(self, state_path: str | Path, cap_usd: float) -> None:
        self.state_path = Path(state_path)
        self.cap_usd = float(cap_usd)
        if self.cap_usd <= 0:
            raise ValueError(f"budget_usd_cap must be positive, got {cap_usd!r}")

    # -- state ---------------------------------------------------------------

    def current(self) -> dict:
        """Read cumulative spend. A corrupt state file raises — silently
        resetting the counter would defeat the whole gate."""
        if not self.state_path.exists():
            return _empty_state()
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            return {
                "total_usd": float(state["total_usd"]),
                "model_calls": int(state["model_calls"]),
                "tokens_in": int(state["tokens_in"]),
                "tokens_out": int(state["tokens_out"]),
                "updated_at": state.get("updated_at"),
            }
        except Exception as exc:
            raise RuntimeError(
                f"spend state file {self.state_path} is corrupt — refusing to guess "
                f"cumulative spend (delete it consciously to reset the round budget): {exc}"
            ) from None

    def check(self) -> dict:
        """Raise when the cap is already reached; returns current state."""
        state = self.current()
        if state["total_usd"] >= self.cap_usd:
            raise BudgetExceededError(
                f"API budget cap reached: ${state['total_usd']:.4f} spent >= "
                f"${self.cap_usd:.2f} cap (state: {self.state_path}). "
                "Stopping — no further LLM calls are allowed this round."
            )
        return state

    def record(self, *, cost_usd: float, tokens_in: int, tokens_out: int) -> dict:
        """Accumulate one completed call and durably persist the state
        (write-to-temp + atomic rename)."""
        state = self.current()
        state["total_usd"] = round(state["total_usd"] + float(cost_usd), 8)
        state["model_calls"] += 1
        state["tokens_in"] += int(tokens_in)
        state["tokens_out"] += int(tokens_out)
        state["updated_at"] = datetime.now(tz=UTC).isoformat(timespec="seconds")
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.state_path.parent, prefix=".spend-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(state, fh, indent=2, sort_keys=True)
            os.replace(tmp, self.state_path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return state

    def remaining_usd(self) -> float:
        return max(0.0, self.cap_usd - self.current()["total_usd"])
