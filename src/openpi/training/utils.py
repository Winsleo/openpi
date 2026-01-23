from collections.abc import Callable
from typing import Any

from flax import nnx
from flax import struct
import jax
import optax

from openpi.models import model as _model
from openpi.shared import array_typing as at


@at.typecheck
@struct.dataclass
class TrainState:
    step: at.Int[at.ArrayLike, ""]
    params: nnx.State
    model_def: nnx.GraphDef[_model.BaseModel]
    opt_state: optax.OptState
    tx: optax.GradientTransformation = struct.field(pytree_node=False)

    ema_decay: float | None = struct.field(pytree_node=False)
    ema_params: nnx.State | None = None


@at.typecheck
def tree_to_info(tree: at.PyTree, interp_func: Callable[[Any], str] = str) -> str:
    """Converts a PyTree into a human-readable string for logging. Optionally, `interp_func` can be provided to convert
    the leaf values to more meaningful strings.
    """
    tree, _ = jax.tree_util.tree_flatten_with_path(tree)
    return "\n".join(f"{jax.tree_util.keystr(path)}: {interp_func(value)}" for path, value in tree)


@at.typecheck
def array_tree_to_info(tree: at.PyTree) -> str:
    """Converts a PyTree of arrays into a human-readable string for logging."""
    return tree_to_info(tree, lambda x: f"{x.shape}@{x.dtype}")

@at.typecheck
def print_trainable_params_info(
    all_params: nnx.State, trainable_params: nnx.State, freeze_filter: nnx.filterlib.Filter
) -> None:
    """Print information about which parameters are trainable vs frozen."""

    def count_params(state: nnx.State) -> tuple[int, dict[str, int]]:
        """Count total parameters and parameters by category."""
        total_params = 0
        param_counts = {}

        for path, value in state.flat_state().items():
            if hasattr(value, 'value') and hasattr(value.value, 'size'):
                param_count = value.value.size
                total_params += param_count

                # Categorize by path components
                path_str = str(path)
                if 'llm' in path_str:
                    if 'lora' in path_str:
                        category = 'llm_lora'
                    elif '_1' in path_str:
                        category = 'action_expert_base'
                    else:
                        category = 'llm_base'
                elif 'img' in path_str:
                    category = 'vision_encoder'
                elif any(x in path_str for x in ['action_in_proj', 'action_out_proj', 'action_time_mlp_in', 'action_time_mlp_out', 'state_proj']):
                    category = 'action_proj'
                else:
                    category = 'other'

                param_counts[category] = param_counts.get(category, 0) + param_count

        return total_params, param_counts

    # Count all and trainable parameters
    total_all, all_counts = count_params(all_params)
    total_trainable, trainable_counts = count_params(trainable_params)

    print(f"Total parameters: {total_all:,}")
    print(f"Trainable parameters: {total_trainable:,}")
    print(f"Trainable percentage: {total_trainable / total_all * 100:.1f}%")

    print("\nParameter breakdown:")
    print(f"{'Category':<15} {'Total':>12} {'Trainable':>12} {'Frozen':>12} {'Status':<8}")
    print("-" * 70)

    all_categories = set(all_counts.keys()) | set(trainable_counts.keys())
    for category in sorted(all_categories):
        all_count = all_counts.get(category, 0)
        trainable_count = trainable_counts.get(category, 0)
        frozen_count = all_count - trainable_count
        status = "🟢 TRAIN" if trainable_count > 0 else "❌ FROZEN"

        print(f"{category:<15} {all_count:>12,} {trainable_count:>12,} {frozen_count:>12,} {status:<8}")

    # Print filter info
    print(f"\nFreeze filter: {freeze_filter}")

    # Special checks for common patterns
    print("\nQuick checks:")
    flat_all = all_params.flat_state()
    flat_trainable = trainable_params.flat_state()

    llm_params = [p for p in flat_all.keys() if 'llm' in str(p)]
    img_params = [p for p in flat_all.keys() if 'img' in str(p)]
    action_params = [p for p in flat_all.keys() if any(x in str(p) for x in ['action_in_proj', 'action_out_proj', 'action_time_mlp_in', 'action_time_mlp_out', 'state_proj'])]

    def check_category(name, params):
        trainable = sum(1 for p in params if p in flat_trainable)
        total = len(params)
        status = "🟢 TRAIN" if trainable > 0 else "❌ FROZEN"
        print(f"  {name}: {trainable}/{total} parameters trainable {status}")

    if llm_params:
        check_category("LLM parameters", llm_params)
    if img_params:
        check_category("Vision encoder", img_params)
    if action_params:
        check_category("Action network", action_params)