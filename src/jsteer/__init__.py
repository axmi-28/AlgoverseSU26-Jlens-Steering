"""jsteer -- when are J-Lens read directions valid write directions?

The J-Lens uses a single averaged Jacobian ``J_bar`` per layer to *read* the
residual stream. Steering with it treats that same object as a *write*
direction. Two research subcomponents ask whether that is licensed:

1. What is the distribution of prompt-local Jacobians ``J_x`` around the
   averaged ``J_bar``? How does that vary with the causal effect of steering?
2. What is the distribution of local pulled-back directions
   ``g_x = J_x^T u_y`` -- aligned, clustered, sign-inconsistent, dominated by
   rare prompts? How does that vary with the causal effect of steering?

Layout: :mod:`jsteer.jacobian` computes ``J_x`` and its pullbacks;
:mod:`jsteer.reduce` shrinks a Jacobian to something storable;
:mod:`jsteer.steering` is the only module that writes to a forward pass;
:mod:`jsteer.analysis` turns the stored clouds into statistics;
:mod:`jsteer.run` holds the conventions the sweeps must share.
"""

from jsteer.config import ModelConfig, available_configs, load_config
from jsteer.data import (
    BasePrompt,
    SwapTrial,
    all_answers,
    all_args,
    category_args,
    flexible_generalization_prompts,
    flexible_generalization_trials,
)
from jsteer.jacobian import (
    PullbackResult,
    averaged_pullback,
    jacobian_for_prompt,
    pullback_for_prompt,
    steering_direction,
)
from jsteer.loading import (
    first_token_id,
    load_lens,
    load_model,
    single_token_id,
    unembedding_rows,
)
from jsteer.reduce import JacobianDigest, RunningMean, digest_jacobian
from jsteer.steering import (
    ResidualEdit,
    TrialOutcome,
    additive_edit,
    grade,
    mean_residual_norms,
    next_token_logits,
    steered,
    swap_edit,
)

__all__ = [
    "BasePrompt",
    "JacobianDigest",
    "ModelConfig",
    "PullbackResult",
    "ResidualEdit",
    "RunningMean",
    "SwapTrial",
    "TrialOutcome",
    "additive_edit",
    "all_answers",
    "all_args",
    "available_configs",
    "averaged_pullback",
    "category_args",
    "digest_jacobian",
    "first_token_id",
    "flexible_generalization_prompts",
    "flexible_generalization_trials",
    "grade",
    "jacobian_for_prompt",
    "load_config",
    "load_lens",
    "load_model",
    "mean_residual_norms",
    "next_token_logits",
    "pullback_for_prompt",
    "single_token_id",
    "steered",
    "steering_direction",
    "swap_edit",
    "unembedding_rows",
]
