"""reward(): the RL reward for one rollout (docs/verifier.md §6)."""

from gdpsolve.schemas import VerifyResult


def reward(v: VerifyResult, grounding: list[float], beta: float = 1.0) -> float | None:
    """mean_criteria + beta * all_pass + mean(grounding); None (skip the rollout) if any criterion is unscored.

    grounding: one value per field in [0, 1]: FieldCheck.ok for extracted/computed fields, mean
    P(supported) over claims for summary fields. An empty list adds nothing.
    """
    if v.all_pass is None or v.mean_criteria is None:
        return None
    g = sum(grounding) / len(grounding) if grounding else 0.0
    return v.mean_criteria + beta * v.all_pass + g
