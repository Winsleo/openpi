import flax.nnx as nnx
import jax

import openpi.models.pi0_config as _pi0_config


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)


def test_freeze_components_empty():
    """Test that empty freeze_components list freezes nothing."""
    config = _pi0_config.Pi0Config(freeze_components=[])
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_freeze_components_llm_base():
    """Test freezing only base LLM parameters."""
    config = _pi0_config.Pi0Config(freeze_components=["llm_base"])
    state = _get_frozen_state(config)
    # Should freeze base LLM params but not action expert or lora
    assert len(state) > 0
    assert all("llm" in str(p) for p in state)
    assert all("_1" not in str(p) for p in state)  # No action expert
    assert all("lora" not in str(p) for p in state)  # No lora


def test_freeze_components_action_expert_base():
    """Test freezing only action expert base parameters."""
    config = _pi0_config.Pi0Config(freeze_components=["action_expert_base"])
    state = _get_frozen_state(config)
    # Should freeze action expert base params but not lora
    assert len(state) > 0
    assert all("llm" in str(p) for p in state)
    assert all("_1" in str(p) for p in state)  # All should be action expert
    assert all("lora" not in str(p) for p in state)  # No lora


def test_freeze_components_vision_encoder():
    """Test freezing only vision encoder parameters."""
    config = _pi0_config.Pi0Config(freeze_components=["vision_encoder"])
    state = _get_frozen_state(config)
    assert len(state) > 0
    assert all("img" in str(p) for p in state)


def test_freeze_components_action_proj():
    """Test freezing only action projection parameters."""
    config = _pi0_config.Pi0Config(freeze_components=["action_proj"])
    state = _get_frozen_state(config)
    assert len(state) > 0
    action_proj_patterns = ["action_in_proj", "action_out_proj", "action_time_mlp_in", "action_time_mlp_out", "state_proj"]
    assert all(any(pattern in str(p) for pattern in action_proj_patterns) for p in state)


def test_freeze_components_multiple():
    """Test freezing multiple components."""
    config = _pi0_config.Pi0Config(freeze_components=["vision_encoder", "action_proj"])
    state = _get_frozen_state(config)
    assert len(state) > 0
    # Should contain both vision and action projection params
    has_vision = any("img" in str(p) for p in state)
    action_proj_patterns = ["action_in_proj", "action_out_proj", "action_time_mlp_in", "action_time_mlp_out", "state_proj"]
    has_action_proj = any(any(pattern in str(p) for pattern in action_proj_patterns) for p in state)
    assert has_vision and has_action_proj


def test_freeze_components_unknown_component():
    """Test that unknown components raise ValueError."""
    config = _pi0_config.Pi0Config(freeze_components=["unknown_component"])
    try:
        config.get_freeze_filter()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown freeze component" in str(e)
