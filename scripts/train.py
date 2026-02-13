import dataclasses
import functools
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def _unwrap_batch(batch: tuple) -> tuple[str | None, tuple[_model.Observation, _model.Actions]]:
    """Unwrap a possibly source-tagged batch.

    Returns:
        (source_name, (observation, actions))
    """
    if (
        isinstance(batch, tuple)
        and len(batch) == 2
        and isinstance(batch[0], (str, int))
        and isinstance(batch[1], tuple)
        and len(batch[1]) == 2
    ):
        return str(batch[0]), batch[1]
    return None, batch

# ============== Validation Functions ==============

def _extract_training_norm_stats(train_data_loader: _data_loader.DataLoader | None) -> dict | None:
    """Extract normalization stats from the training data loader."""
    if train_data_loader is not None and hasattr(train_data_loader, "data_config"):
        training_data_config = train_data_loader.data_config()
        if hasattr(training_data_config, "norm_stats") and training_data_config.norm_stats is not None:
            return training_data_config.norm_stats
    return None


def _prepare_validation_config(
    config: _config.TrainConfig, training_norm_stats: dict | None
) -> tuple[_config.TrainConfig, str, bool, _config.DataConfig]:
    """
    Prepare validation configuration with training norm_stats.

    Returns:
        tuple: (validation_config, repo_id, use_norm_stats, actual_val_data_config)
    """
    val_config = dataclasses.replace(
        config,
        batch_size=config.val_batch_size or config.batch_size,
    )

    use_norm_stats = False
    
    # Determine the data factory to use for validation
    # When using composable_data, get the first dataset config as reference
    if config.composable_data is not None and config.composable_data.dataset_configs:
        data_factory = config.composable_data.dataset_configs[0]
    else:
        data_factory = val_config.data
    
    # Check if we have a valid repo_id source
    has_repo_id = config.val_repo_id or (hasattr(data_factory, "repo_id") and data_factory.repo_id)
    
    if has_repo_id:
        repo_id = config.val_repo_id or getattr(data_factory, "repo_id", None)
        if repo_id is None:
            raise ValueError("No validation repository ID could be determined")
        
        # Safely get val_episodes_index, defaulting to None if not present
        episodes_index = getattr(config, 'val_episodes_index', None)

        # Create validation data config by copying the data factory but changing repo_id
        # Check if base_config exists and is a dataclass
        base_config = getattr(data_factory, 'base_config', None)
        if base_config is not None and dataclasses.is_dataclass(base_config):
            val_base_config = dataclasses.replace(base_config, episodes_index=episodes_index)
            val_data_factory = dataclasses.replace(data_factory, repo_id=repo_id, base_config=val_base_config)
        else:
            # For data factories without base_config (e.g., SimpleDataConfig)
            val_data_factory = dataclasses.replace(data_factory, repo_id=repo_id)
        
        # Update val_config with the new data factory (disable composable_data for validation)
        val_config = dataclasses.replace(val_config, data=val_data_factory, composable_data=None)

        # Create the actual DataConfig using the factory and add norm_stats
        actual_val_data_config = val_config.data.create(val_config.assets_dirs, val_config.model)

        # Explicitly use norm_stats from training data loader for validation
        if training_norm_stats is not None:
            actual_val_data_config = dataclasses.replace(actual_val_data_config, norm_stats=training_norm_stats)
            logging.info("Copied norm_stats from training data loader to validation config")
            logging.info("Training norm_stats keys: %s", list(training_norm_stats.keys()))
            use_norm_stats = True
        else:
            logging.warning("No norm_stats found in training data loader - skipping normalization for validation")

        return val_config, repo_id, use_norm_stats, actual_val_data_config

    raise ValueError("No validation repository ID could be determined")


def _compute_validation_losses(
    val_loader: _data_loader.DataLoader,
    train_state: training_utils.TrainState,
    mesh: jax.sharding.Mesh,
    train_state_sharding: jax.sharding.NamedSharding,
    replicated_sharding: jax.sharding.NamedSharding,
    config: _config.TrainConfig,
) -> float | None:
    """Compute validation losses over multiple batches."""

    # Define validation loss function (similar to train_step structure)
    @at.typecheck
    def val_loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=False)
        return jnp.mean(chunked_loss)

    def validation_step(state, batch, rng):
        """Single validation step, aligned with train_step structure."""
        model = nnx.merge(state.model_def, state.params)
        model.eval()

        observation, actions = batch
        val_rng = jax.random.fold_in(rng, state.step)

        return val_loss_fn(model, val_rng, observation, actions)

    # JIT compile the validation step
    pvalidation_step = jax.jit(
        validation_step,
        in_shardings=(train_state_sharding, replicated_sharding, replicated_sharding),
        out_shardings=replicated_sharding,
    )

    val_iter = iter(val_loader)
    losses = []
    # Use a separate RNG for validation to avoid interference with training RNG,
    # the specific seed offset is arbitrary.
    val_rng = jax.random.key(config.seed + 1000)

    for batch_idx in range(config.val_num_batches):
        try:
            batch = next(val_iter)
        except StopIteration:
            break

        try:
            with sharding.set_mesh(mesh):
                loss = pvalidation_step(train_state, batch, val_rng)
            losses.append(jax.device_get(loss))
        except (RuntimeError, ValueError) as e:
            logging.warning("Error computing validation loss for batch %d: %s", batch_idx, e)
            continue

    if not losses:
        return None
    return float(jnp.mean(jnp.array(losses)))


def compute_validation_loss(
    config: _config.TrainConfig,
    train_state: training_utils.TrainState,
    mesh: jax.sharding.Mesh,
    train_state_sharding: jax.sharding.NamedSharding,
    replicated_sharding: jax.sharding.NamedSharding,
    train_data_loader: _data_loader.DataLoader | None = None,
) -> float | None:
    """Compute average validation loss over a few batches. Downloads validation dataset if missing locally."""
    # Extract training normalization stats
    training_norm_stats = _extract_training_norm_stats(train_data_loader)

    # Prepare validation configuration
    try:
        val_config, repo_id, use_norm_stats, actual_val_data_config = _prepare_validation_config(
            config, training_norm_stats
        )
    except ValueError:
        logging.warning("Could not determine validation repository ID, skipping validation loss.")
        return None

    # Create validation data loader (this will use local data or trigger download if needed)
    try:
        val_loader = _data_loader.create_data_loader(
            val_config,
            data_config=actual_val_data_config,
            sharding=replicated_sharding,
            shuffle=False,
            num_batches=val_config.val_num_batches,
            skip_norm_stats=not use_norm_stats,
        )
        logging.info(f"Validation dataset loaded: {repo_id}")
    except Exception as e:
        logging.warning(f"Failed to create validation data loader for {repo_id}: {e}")
        logging.warning("Skipping validation loss computation.")
        return None

    # Compute and return validation losses
    return _compute_validation_losses(val_loader, train_state, mesh, train_state_sharding, replicated_sharding, config)


# ============== Logging & Initialization ==============

def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def compute_grads(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[at.Params, dict[str, at.Array]]:
    """Compute gradients for a single batch without updating parameters.

    This function is used for gradient accumulation - it computes gradients
    that can be accumulated across multiple mini-batches before applying updates.
    """
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
    }
    return grads, info


@at.typecheck
def apply_grads(
    config: _config.TrainConfig,
    state: training_utils.TrainState,
    grads: at.Params,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """Apply accumulated gradients to update model parameters.

    This function is called after gradients have been accumulated across
    multiple mini-batches (for gradient accumulation).
    """
    model = nnx.merge(state.model_def, state.params)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """Single training step (compute gradients and apply updates).

    This is the original train_step for non-gradient-accumulation case.
    For gradient accumulation, use compute_grads() and apply_grads() separately.
    """
    grads, grad_info = compute_grads(config, rng, state, batch)
    new_state, apply_info = apply_grads(config, state, grads)

    info = {**grad_info, **apply_info}
    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
        num_batches=config.num_train_steps * config.batch_size // jax.device_count(),
    )
    data_iter = iter(data_loader)
    try:
        batch = next(data_iter)
    except StopIteration:
        raise ValueError("Data loader is empty - no data available for training")
    _, (init_obs, init_actions) = _unwrap_batch(batch)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info((init_obs, init_actions))}")

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in init_obs.images.values()], axis=1))
        for i in range(min(5, len(next(iter(init_obs.images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)
    
    trainable_params = train_state.params.filter(config.trainable_filter)
    logging.info("\n=== Trainable Parameters Summary ===")
    training_utils.print_trainable_params_info(train_state.params, trainable_params, config.trainable_filter)
    logging.info("=" * 50)
    
    # Setup gradient accumulation
    grad_accum_steps = getattr(config, 'gradient_accumulation_steps', 1)
    effective_batch_size = config.batch_size * grad_accum_steps
    logging.info(f"Gradient accumulation steps: {grad_accum_steps}")
    logging.info(f"Per-step batch size: {config.batch_size}, Effective batch size: {effective_batch_size}")

    if grad_accum_steps > 1:
        # JIT compile separate functions for gradient accumulation
        pcompute_grads = jax.jit(
            functools.partial(compute_grads, config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=(replicated_sharding, replicated_sharding),
        )
        papply_grads = jax.jit(
            functools.partial(apply_grads, config),
            in_shardings=(train_state_sharding, replicated_sharding),
            out_shardings=(train_state_sharding, replicated_sharding),
            donate_argnums=(0,),
        )
    else:
        # JIT compile single train_step for non-accumulation case
        ptrain_step = jax.jit(
            functools.partial(train_step, config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=(train_state_sharding, replicated_sharding),
            donate_argnums=(1,),
        )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    # Check if validation is enabled (requires val_log_interval > 0 and val_repo_id or data.repo_id)
    has_val_repo = config.val_repo_id is not None or hasattr(config.data, "repo_id")
    validation_enabled = getattr(config, 'val_log_interval', 0) > 0 and has_val_repo
    if validation_enabled:
        logging.info(f"Validation enabled: logging every {config.val_log_interval} steps")
    else:
        if not has_val_repo:
            logging.info("Validation disabled (no val_repo_id configured)")
        else:
            logging.info("Validation disabled (val_log_interval <= 0)")

    infos = []
    try:
        for step in pbar:
            source, (observation, actions) = _unwrap_batch(batch)
            if source is not None:
                pbar.write(f"Step {step}: data_source={source}")
            with sharding.set_mesh(mesh):
                if grad_accum_steps > 1:
                    # Gradient accumulation mode
                    accumulated_grads = None
                    accumulated_infos = []

                    for accum_step in range(grad_accum_steps):
                        grads, grad_info = pcompute_grads(train_rng, train_state, (observation, actions))
                        accumulated_infos.append(grad_info)

                        # Accumulate gradients (average across accumulation steps)
                        if accumulated_grads is None:
                            accumulated_grads = grads
                        else:
                            accumulated_grads = jax.tree.map(
                                lambda acc, g: acc + g, accumulated_grads, grads
                            )

                        # Get next batch for accumulation (except for last step)
                        if accum_step < grad_accum_steps - 1:
                            batch = next(data_iter)
                            source, (observation, actions) = _unwrap_batch(batch)
                            if source is not None:
                                pbar.write(f"Step {step}: data_source={source} (accumulation step {accum_step + 1}/{grad_accum_steps})")

                    # Average the accumulated gradients
                    accumulated_grads = jax.tree.map(
                        lambda g: g / grad_accum_steps, accumulated_grads
                    )

                    # Apply the accumulated gradients
                    train_state, apply_info = papply_grads(train_state, accumulated_grads)

                    # Combine info from all accumulation steps
                    stacked_accum_infos = common_utils.stack_forest(accumulated_infos)
                    info = {
                        "loss": jnp.mean(stacked_accum_infos["loss"]),
                        "grad_norm": jnp.mean(stacked_accum_infos["grad_norm"]),
                        **apply_info,
                    }
                else:
                    # Standard single-step training
                    train_state, info = ptrain_step(train_rng, train_state, (observation, actions))

            infos.append(info)
            if step % config.log_interval == 0:
                stacked_infos = common_utils.stack_forest(infos)
                reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
                info_str = ", ".join(f"{k}={float(v):.4f}" for k, v in reduced_info.items())
                pbar.write(f"Step {step}: {info_str}")
                wandb.log(reduced_info, step=step)
                infos = []

            # Validation loss computation (if enabled)
            if validation_enabled and step % config.val_log_interval == 0:
                val_loss = compute_validation_loss(
                    config, train_state, mesh, train_state_sharding, replicated_sharding, data_loader
                )
                if val_loss is not None:
                    wandb.log({"val_loss": val_loss}, step=step)
                    logging.info("Validation loss at step %d: %.4f", step, val_loss)

            batch = next(data_iter)

            if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
                _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)
    
    except StopIteration:
        # Data iterator exhausted - save final state and exit gracefully
        final_step = int(train_state.step)
        logging.info(f"Data iterator exhausted at step {final_step}. Saving final checkpoint and exiting gracefully.")
        
        # Log any remaining metrics
        if infos:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={float(v):.4f}" for k, v in reduced_info.items())
            logging.info(f"Final metrics: {info_str}")
            wandb.log(reduced_info, step=final_step)
        
        # Save final checkpoint
        _checkpoints.save_state(checkpoint_manager, train_state, data_loader, final_step)
        logging.info(f"Training completed early at step {final_step} due to data exhaustion.")

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
