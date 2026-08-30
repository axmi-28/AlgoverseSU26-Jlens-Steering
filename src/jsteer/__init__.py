"""jsteer -- when are J-Lens read directions valid write directions?

Research subcomponent 1: characterize how much the prompt-local Jacobian
``J_x`` varies around the averaged ``J_bar`` the lens actually uses, and
whether that variation predicts whether steering works.
"""

from jsteer.config import ModelConfig, available_configs, load_config
from jsteer.jacobian import (
    PullbackResult,
    averaged_pullback,
    jacobian_for_prompt,
    pullback_for_prompt,
    steering_direction,
)
from jsteer.loading import load_lens, load_model, single_token_id, unembedding_rows

__all__ = [
    "ModelConfig",
    "PullbackResult",
    "available_configs",
    "averaged_pullback",
    "jacobian_for_prompt",
    "load_config",
    "load_lens",
    "load_model",
    "pullback_for_prompt",
    "single_token_id",
    "steering_direction",
    "unembedding_rows",
]
