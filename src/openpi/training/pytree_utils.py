"""
Efficient utilities for handling PyTrees with JAX, PyTorch, and NumPy arrays.

Features:
- Lazy JAX import (only imports when JAX arrays are detected)
- Support for Mapping/Sequence PyTree structures
- Type-safe sharding operations
- Optimized with caching and early-exit patterns
"""

from typing import Any
from collections.abc import Mapping, Sequence
import numpy as np


# ============================================================================
# Lazy Module Imports with Caching
# ============================================================================

_jax_module = None
_jax_array_type = None
_sharding_type = None
_torch_module = None
_torch_tensor_type = None


def _get_jax():
    """Lazy import JAX module."""
    global _jax_module
    if _jax_module is None:
        import jax
        _jax_module = jax
    return _jax_module


def _get_jax_array_type():
    """Lazy import JAX Array type."""
    global _jax_array_type
    if _jax_array_type is None:
        from jax import Array
        _jax_array_type = Array
    return _jax_array_type


def _get_sharding_type():
    """Lazy import JAX Sharding type."""
    global _sharding_type
    if _sharding_type is None:
        try:
            from jax.sharding import Sharding
            _sharding_type = Sharding
        except ImportError:
            _sharding_type = False  # Mark as unavailable
    return _sharding_type


def _get_torch():
    """Lazy import PyTorch module."""
    global _torch_module
    if _torch_module is None:
        try:
            import torch
            _torch_module = torch
        except ImportError:
            _torch_module = False  # Mark as unavailable
    return _torch_module


# ============================================================================
# Type Checking Functions
# ============================================================================

def is_jax_array(x) -> bool:
    """Check if x is a JAX array without importing JAX unless necessary.
    
    Uses cached type checking for optimal performance after first call.
    """
    # Fast path: cached type check
    if _jax_array_type is not None:
        return isinstance(x, _jax_array_type)
    
    # Module name check before importing JAX
    type_module = type(x).__module__
    if not type_module.startswith('jax'):
        return False
    
    # Import and cache JAX Array type
    try:
        Array = _get_jax_array_type()
        return isinstance(x, Array)
    except ImportError:
        return False


def _is_torch_tensor(x) -> bool:
    """Check if x is a PyTorch tensor with caching."""
    global _torch_tensor_type
    
    # Fast path: cached type check
    if _torch_tensor_type is not None:
        return isinstance(x, _torch_tensor_type)
    
    # Check if torch is unavailable
    if _torch_module is False:
        return False
    
    # Try to get torch
    torch = _get_torch()
    if torch is False:
        return False
    
    # Cache tensor type
    _torch_tensor_type = torch.Tensor
    return isinstance(x, _torch_tensor_type)


def _is_sharding_object(x) -> bool:
    """Check if x is a valid JAX Sharding object."""
    if x is None:
        return False
    
    Sharding = _get_sharding_type()
    if Sharding is False:
        return False
    
    return isinstance(x, Sharding)


# Pre-compile type tuples for faster isinstance checks
_CONTAINER_TYPES = (Mapping, Sequence)
_STRING_TYPES = (str, bytes)


def _is_array_like(x) -> bool:
    """Check if x is an array-like object (JAX/PyTorch/NumPy).
    
    Optimized order: NumPy (most common) -> JAX -> PyTorch
    """
    # NumPy check first (most common, fastest)
    if isinstance(x, np.ndarray):
        return True
    # Then JAX (avoid module check if possible)
    if is_jax_array(x):
        return True
    # Finally PyTorch (least common)
    return _is_torch_tensor(x)


# ============================================================================
# PyTree Traversal Utilities
# ============================================================================

def contains_jax_array(tree) -> bool:
    """Check if PyTree contains any JAX arrays.
    
    Uses early-exit optimization for performance on large trees.
    """
    # Quick type checks for common leaf types
    if isinstance(tree, np.ndarray):
        return False
    if is_jax_array(tree):
        return True
    if _is_torch_tensor(tree):
        return False

    # Container traversal with early exit
    if isinstance(tree, Mapping):
        return any(contains_jax_array(v) for v in tree.values())

    if isinstance(tree, Sequence) and not isinstance(tree, _STRING_TYPES):
        return any(contains_jax_array(item) for item in tree)

    return False


def _has_valid_sharding(tree) -> bool:
    """Check if PyTree contains any valid JAX Sharding objects.
    
    Only actual Sharding instances are considered valid, not arbitrary non-None values.
    
    Args:
        tree: PyTree potentially containing Sharding objects
        
    Returns:
        True if any leaf is a Sharding object, False otherwise
        
    Examples:
        >>> _has_valid_sharding(None)
        False
        >>> _has_valid_sharding({'a': None, 'b': None})
        False
        >>> _has_valid_sharding({'a': None, 'b': some_sharding})
        True
        >>> _has_valid_sharding({'a': 'not a sharding'})
        False
    """
    if tree is None:
        return False
    
    # Check if this leaf is a Sharding object
    if _is_sharding_object(tree):
        return True
    
    # Not a container and not a Sharding - this is an invalid leaf
    if not isinstance(tree, _CONTAINER_TYPES) or isinstance(tree, _STRING_TYPES):
        return False
    
    # Container - recursively check children with early exit
    if isinstance(tree, Mapping):
        return any(_has_valid_sharding(v) for v in tree.values())
    
    return any(_has_valid_sharding(item) for item in tree)


# ============================================================================
# Data Slicing
# ============================================================================

def slice_data(data, indices):
    """Slice data using indices, supporting various formats.
    
    Supports:
    - JAX arrays (native indexing)
    - PyTorch tensors (native indexing)
    - NumPy arrays (native indexing)
    - PyTrees with Mapping/Sequence nesting
    - None values (passed through)
    
    JAX is only imported if JAX arrays are detected.
    
    Args:
        data: Data to slice (array, tensor, or PyTree)
        indices: Indices to use for slicing (int, slice, array, etc.)
        
    Returns:
        Sliced data with same structure as input
        
    Examples:
        >>> # Single array
        >>> arr = np.arange(100)
        >>> slice_data(arr, slice(0, 10))  # array([0, 1, ..., 9])
        
        >>> # PyTree
        >>> data = {'obs': np.ones((100, 84, 84, 3)), 'actions': np.ones((100, 7))}
        >>> sliced = slice_data(data, slice(0, 32))
        >>> sliced['obs'].shape  # (32, 84, 84, 3)
    """
    if data is None:
        return None

    # Fast path: direct array slicing (most common case)
    if isinstance(data, np.ndarray):
        return data[indices]
    
    # Check other array types
    if is_jax_array(data):
        return data[indices]
    
    if _is_torch_tensor(data):
        return data[indices]

    # PyTree case - only import JAX tree utilities when needed
    jax = _get_jax()
    
    # Inline lambda for better performance
    return jax.tree_util.tree_map(
        lambda x: x[indices] if (x is not None and _is_array_like(x)) else x,
        data
    )


# ============================================================================
# Data Concatenation
# ============================================================================

def concat_data(data_list: Sequence[Any]):
    """Concatenate data from multiple sources, supporting various formats.
    
    Supports:
    - JAX arrays (jnp.concatenate)
    - PyTorch tensors (torch.cat)
    - NumPy arrays (np.concatenate)
    - PyTrees with Mapping/Sequence nesting
    - None values (filtered out)
    
    JAX is only imported if JAX arrays are detected.
    
    Args:
        data_list: Sequence of data items to concatenate
        
    Returns:
        Concatenated data with same structure as input items
        
    Raises:
        TypeError: If data type is not supported for concatenation
        ValueError: If data_list is empty or contains only None
        
    Examples:
        >>> # Single arrays
        >>> arrays = [np.ones((10, 5)), np.ones((20, 5))]
        >>> concat_data(arrays).shape  # (30, 5)
        
        >>> # PyTrees
        >>> data_list = [
        ...     {'obs': np.ones((10, 84)), 'actions': np.ones((10, 7))},
        ...     {'obs': np.ones((20, 84)), 'actions': np.ones((20, 7))}
        ... ]
        >>> result = concat_data(data_list)
        >>> result['obs'].shape  # (30, 84)
        
        >>> # Mixed with None
        >>> concat_data([None, np.ones((10, 5)), None])  # shape (10, 5)
    """
    if (not data_list) or (not isinstance(data_list, Sequence)) or (len(data_list) == 0):
        raise ValueError("data_list must be a non-empty sequence")
    
    # Early filtering of None values
    non_none_data = [d for d in data_list if d is not None]
    if not non_none_data:
        return None
    
    first = non_none_data[0]
    
    # Fast path: NumPy arrays (most common)
    if isinstance(first, np.ndarray):
        return np.concatenate(non_none_data, axis=0)
    
    # Fast path: JAX arrays
    if is_jax_array(first):
        jax = _get_jax()
        return jax.numpy.concatenate(non_none_data, axis=0)
    
    # Fast path: PyTorch tensors
    if _is_torch_tensor(first):
        torch = _get_torch()
        if torch is False:
            raise TypeError("PyTorch is not available")
        return torch.cat(non_none_data, dim=0)
    
    # PyTree case
    jax = _get_jax()
    
    # Pre-compile concat function for reuse
    def _concat_leaf(*items):
        """Concatenate leaf values, filtering None."""
        # Filter None at leaf level
        non_none_items = [x for x in items if x is not None]
        if not non_none_items:
            return None
        
        first_item = non_none_items[0]
        
        # Fast type dispatch for arrays
        if isinstance(first_item, np.ndarray):
            return np.concatenate(non_none_items, axis=0)
        if is_jax_array(first_item):
            return jax.numpy.concatenate(non_none_items, axis=0)
        if _is_torch_tensor(first_item):
            torch = _get_torch()
            if torch is False:
                raise TypeError("PyTorch is not available")
            return torch.cat(non_none_items, dim=0)
        
        # Non-array leaf - return first value (assuming all are same)
        return first_item
    
    try:
        return jax.tree_util.tree_map(_concat_leaf, *non_none_data)
    except Exception as e:
        raise TypeError(
            f"Cannot concatenate PyTree with first element type {type(first)}"
        ) from e


# ============================================================================
# Sharding Utilities
# ============================================================================

def get_sharding(data):
    """Extract sharding from JAX arrays in data (lazy JAX import).
    
    Returns a PyTree with the same structure where JAX array leaves are
    replaced by their sharding objects, and non-JAX leaves become None.
    
    Args:
        data: JAX array or PyTree with mixed leaves
        
    Returns:
        Sharding object, PyTree of sharding objects/None, or None if no JAX arrays
        
    Examples:
        >>> data = {'params': jax_array_with_sharding, 'metadata': 'info'}
        >>> sharding_tree = get_sharding(data)
        >>> # Returns: {'params': NamedSharding(...), 'metadata': None}
        
        >>> # Pure NumPy data - returns None without importing JAX
        >>> numpy_data = {'arr': np.ones(10)}
        >>> get_sharding(numpy_data)  # None
    """
    # Fast path: no JAX arrays means no sharding to extract
    if not contains_jax_array(data):
        return None
    
    # Only import JAX if we detected JAX arrays
    jax = _get_jax()
    
    # Use getattr for safe access with minimal overhead
    return jax.tree_map(
        lambda x: getattr(x, 'sharding', None) if is_jax_array(x) else None,
        data,
        is_leaf=is_jax_array
    )


def apply_sharding(data, sharding):
    """Apply sharding to JAX arrays in ``data`` (lazy JAX import).

    This function supports two sharding formats:

    1. **Single Sharding object**: the same sharding is applied to every JAX
       array leaf in ``data``.
    2. **PyTree of Sharding objects / None**: sharding is applied per-leaf,
       matching the structure of ``data``.

    Non‑JAX leaves are always returned unchanged.

    Args:
        data: A JAX array or a PyTree containing mixed leaf types.
        sharding: Either a single JAX ``Sharding`` object or a PyTree of
            ``Sharding`` / ``None`` with structure compatible with ``data``.

    Returns:
        The input ``data`` with sharding applied to JAX array leaves where
        applicable. If ``sharding`` is ``None`` or contains no valid sharding
        objects, ``data`` is returned unchanged.
    """
    # Fast path: nothing to apply
    if sharding is None:
        return data

    jax = _get_jax()

    # Case 1: a single, global Sharding object for all JAX leaves in `data`.
    if _is_sharding_object(sharding):
        return jax.tree_map(
            lambda d: jax.device_put(d, sharding) if is_jax_array(d) else d,
            data,
            is_leaf=is_jax_array,
        )

    # Case 2: `sharding` is a PyTree; only proceed if it actually contains
    # valid Sharding objects.
    if not _has_valid_sharding(sharding):
        return data

    return jax.tree_map(
        lambda d, s: jax.device_put(d, s) if (is_jax_array(d) and _is_sharding_object(s)) else d,
        data,
        sharding,
        is_leaf=is_jax_array,
    )


# ============================================================================
# Convenience Functions
# ============================================================================

def shard_like(data, reference):
    """Apply the sharding from reference to data.
    
    Useful for ensuring data matches expected sharding pattern.
    
    Args:
        data: Data to be sharded
        reference: Reference data with desired sharding
    
    Returns:
        Data with sharding matching reference
        
    Examples:
        >>> train_state = ...  # Has proper sharding
        >>> new_batch = ...    # From data loader, might not have sharding
        >>> new_batch = shard_like(new_batch, train_state)
    """
    reference_sharding = get_sharding(reference)
    return apply_sharding(data, reference_sharding)


def shard_batch(batch, data_sharding):
    """Apply data sharding to a batch from the data loader.
    
    Handles both source-tagged and regular batches efficiently.
    
    Args:
        batch: Batch data (possibly source-tagged as (source, (obs, actions)))
        data_sharding: Sharding to apply
        
    Returns:
        Batch with sharding applied
        
    Examples:
        >>> # For training loop
        >>> batch = next(data_iter)
        >>> batch = shard_batch(batch, data_sharding)
        >>> train_state, info = train_step(train_rng, train_state, batch)
    """
    # Fast unwrap check for source-tagged batch: (source, (observation, actions))
    if (isinstance(batch, tuple) and 
        len(batch) == 2 and 
        isinstance(batch[0], (str, int)) and 
        isinstance(batch[1], tuple) and 
        len(batch[1]) == 2):
        # Source-tagged batch
        source, (observation, actions) = batch
        observation = apply_sharding(observation, data_sharding)
        actions = apply_sharding(actions, data_sharding)
        return source, (observation, actions)
    
    # Regular batch: (observation, actions)
    if isinstance(batch, tuple) and len(batch) == 2:
        observation, actions = batch
        observation = apply_sharding(observation, data_sharding)
        actions = apply_sharding(actions, data_sharding)
        return observation, actions
    
    # Fallback: apply sharding directly to any structure
    return apply_sharding(batch, data_sharding)


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Type checking
    'is_jax_array',
    'contains_jax_array',
    # Data manipulation
    'slice_data',
    'concat_data',
    # Sharding
    'get_sharding',
    'apply_sharding',
    'shard_like',
    'shard_batch',
]