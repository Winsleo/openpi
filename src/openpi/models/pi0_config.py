import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    # Flexible freezing options: specify which components to freeze
    # Supported components: 'llm_base', 'action_expert_base', 'vision_encoder', 'action_proj'
    # If None, uses legacy LoRA-based freezing logic
    freeze_components: list[str] | None = None

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config.

        If freeze_components is specified, uses component-based freezing for more flexibility.
        Otherwise, falls back to legacy LoRA-based freezing logic.
        """
        # New flexible component-based freezing
        if self.freeze_components is not None:
            return self._get_component_freeze_filter()

        # Legacy LoRA-based freezing logic
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)

    def _get_component_freeze_filter(self) -> nnx.filterlib.Filter:
        """Create freeze filter based on specified components to freeze.

        Supported components:
        - 'llm_base': Base LLM parameters (llm.* but not lora and not _1)
        - 'action_expert_base': Action expert base parameters (llm.*_1.*)
        - 'vision_encoder': Vision encoder parameters (img.*)
        - 'action_proj': Action projection layers (action_in_proj, action_out_proj, etc.)
        """
        if not self.freeze_components:
            return nnx.Nothing

        filters = []

        for component in self.freeze_components:
            if component == "llm_base":
                # Freeze base LLM params: llm.* but exclude action expert (_1) and lora params
                filters.append(
                    nnx.All(
                        nnx_utils.PathRegex(".*llm.*"),
                        nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*")),  # Exclude action expert
                        nnx.Not(nnx_utils.PathRegex(".*lora.*")),     # Exclude lora params
                    )
                )
            elif component == "action_expert_base":
                # Freeze action expert base params: llm.*_1.* but exclude lora params
                filters.append(
                    nnx.All(
                        nnx_utils.PathRegex(".*llm.*_1.*"),
                        nnx.Not(nnx_utils.PathRegex(".*lora.*")),
                    )
                )
            elif component == "vision_encoder":
                # Freeze vision encoder params
                filters.append(nnx_utils.PathRegex(".*img.*"))
            elif component == "action_proj":
                # Freeze action projection layers
                action_proj_patterns = [
                    ".*action_in_proj.*",
                    ".*action_out_proj.*",
                    ".*action_time_mlp_in.*",
                    ".*action_time_mlp_out.*",
                    ".*state_proj.*"
                ]
                action_filters = [nnx_utils.PathRegex(pattern) for pattern in action_proj_patterns]
                filters.append(nnx.Any(*action_filters))
            elif component == "lora":
                filters.append(nnx_utils.PathRegex(".*lora.*"))
            elif component == "all":
                filters.append(nnx.Everything)
            else:
                raise ValueError(f"Unknown freeze component: {component}. "
                               f"Supported components: llm_base, action_expert_base, vision_encoder, action_proj")

        if not filters:
            return nnx.Nothing

        # Combine all component filters - a parameter is frozen if it matches ANY of the component filters
        return nnx.Any(*filters)
