"""Verifiers: rollout scoring for RL and eval (docs/verifier.md, docs/verifier-trd.md).

Tinker-dependent: install with `uv sync --extra train`.
"""

from gdpsolve.verifier.checkers import support, verify
from gdpsolve.verifier.grounding import check_field
from gdpsolve.verifier.reward import reward

__all__ = ["check_field", "reward", "support", "verify"]
