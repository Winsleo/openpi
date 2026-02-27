"""Loader tree identification for logging and source tagging.

Builds human-readable representations of composed DataLoader hierarchies,
including data_config info (repo_id, episodes_index, etc.) and instance ids.
"""

from collections.abc import Callable
from typing import Any, Optional


def _format_episodes_index(episodes_index: Any) -> Optional[str]:
    """Format episodes_index for display, e.g. '0..99' or '100..189'."""
    if episodes_index is None or not episodes_index:
        return None
    try:
        lst = list(episodes_index)
    except Exception:
        return None
    if not lst:
        return None
    if len(lst) == 1:
        return str(lst[0])
    # Check if contiguous
    if lst == list(range(lst[0], lst[-1] + 1)):
        return f"{lst[0]}..{lst[-1]}"
    return f"<{len(lst)} episodes>"


# Spec format: (attr, opts). List order = display order. Attribute name = display key.
#
# opts: dict with optional keys:
#   - formatter: Callable[[Any], str | None]. Converts raw attr value to display string.
#     Return None to omit the field. Use when str(value) is not suitable (e.g. list -> "0..99").
#   - skip_if: Callable[[Any, Any], bool]. (value, cfg) -> True to skip extraction.
#     Use to avoid redundant output (e.g. skip asset_id when it equals repo_id).
#
# Conventions:
#   - Empty opts {}: use str(value), include if value is not None.
#   - Combine formatter and skip_if when both needed: {"formatter": fn, "skip_if": fn}.
_DATA_CONFIG_EXTRACT_SPECS: list[tuple[str, dict]] = [
    ("repo_id", {}),
    ("asset_id", {"skip_if": lambda value, cfg: value == getattr(cfg, "repo_id", None)}),
    ("episodes_index", {"formatter": _format_episodes_index}),
    ("behavior_dataset_root", {}),
    ("rlds_data_dir", {}),
]


def _make_attr_extractor(
    attr: str,
    *,
    formatter: Optional[Callable[[Any], Optional[str]]] = None,
    skip_if: Optional[Callable[[Any, Any], bool]] = None,
) -> Callable[[Any], Optional[tuple[str, str]]]:
    """Build an extractor for a data_config attribute.

    Args:
        attr: Attribute name on the config object. Also used as the display key.
        formatter: Optional. Converts raw value to display string. Signature (value) -> str | None.
            If returns None, the field is omitted.
        skip_if: Optional. Skip extraction when true. Signature (value, cfg) -> bool.
            Use to suppress redundant info (e.g. skip asset_id when it equals repo_id).
            Parameters: value = attr value, cfg = data_config object.
    """
    def extract(cfg: Any) -> Optional[tuple[str, str]]:
        value = getattr(cfg, attr, None)
        if value is None:
            return None
        if skip_if is not None and skip_if(value, cfg):
            return None
        if formatter is not None:
            s = formatter(value)
        else:
            s = str(value)
        return (attr, s) if s else None
    return extract


_DATA_CONFIG_EXTRACTORS: list[Callable[[Any], Optional[tuple[str, str]]]] = [
    _make_attr_extractor(attr, **opts) for attr, opts in _DATA_CONFIG_EXTRACT_SPECS
]


def _extract_data_config_info(loader: Any) -> Optional[dict[str, str]]:
    """Extract data_config fields for display using _DATA_CONFIG_EXTRACT_SPECS."""
    cfg_attr = getattr(loader, "data_config", None)
    if cfg_attr is None:
        return None
    try:
        cfg = cfg_attr() if callable(cfg_attr) else cfg_attr
    except Exception:
        return None
    if cfg is None:
        return None
    out: dict[str, str] = {}
    for extractor in _DATA_CONFIG_EXTRACTORS:
        try:
            result = extractor(cfg)
            if result is not None:
                key, value = result
                out[key] = value
        except Exception:
            continue
    return out if out else None


# Cached types for _unwrap_transparent; populated on first use to avoid repeated imports.
_BaseDataLoaderAdapter: Optional[type] = None
_SingleLoaderWrapper: Optional[type] = None


def _unwrap_transparent(loader: Any) -> Any:
    """Unwrap SingleLoaderWrapper and BaseDataLoaderAdapter layers, collecting data_config info.

    Returns the first non-transparent loader (a MultiSourceDataLoader or a real leaf like
    TorchDataLoader / RLDSDataLoader). data_config info collected from adapter layers is
    attached as ``_transient_dc_parts`` on the returned loader (monkey-patched transiently).
    """
    global _BaseDataLoaderAdapter, _SingleLoaderWrapper
    if _BaseDataLoaderAdapter is None:
        from openpi.training.composable_dataloader import SingleLoaderWrapper
        from openpi.training.data_loader import BaseDataLoaderAdapter
        _BaseDataLoaderAdapter = BaseDataLoaderAdapter
        _SingleLoaderWrapper = SingleLoaderWrapper

    dc_parts: list[dict[str, str]] = []
    seen: set[int] = set()
    cur = loader
    while True:
        if id(cur) in seen:
            break
        seen.add(id(cur))

        if isinstance(cur, _BaseDataLoaderAdapter):
            dc = _extract_data_config_info(cur)
            if dc:
                dc_parts.append(dc)
            nxt = getattr(cur, "inner", None)
            if nxt is not None:
                cur = nxt
                continue
            break

        if isinstance(cur, _SingleLoaderWrapper):
            nxt = getattr(cur, "inner", None)
            if nxt is not None:
                cur = nxt
                continue
            break

        break

    cur._transient_dc_parts = dc_parts  # type: ignore[attr-defined]
    return cur


def _build_loader_tree(
    loader: Any,
    seen: set[int],
    *,
    indent: str = "",
    max_depth: int = 12,
) -> list[str]:
    """Recursively build a tree of meaningful loader nodes."""
    if max_depth <= 0 or id(loader) in seen:
        return []

    node = _unwrap_transparent(loader)
    dc_parts: list[dict[str, str]] = getattr(node, "_transient_dc_parts", [])

    if id(node) in seen:
        return []
    seen.add(id(node))

    instance_id = f"#{id(node) & 0xFFFF:04x}"
    cls_name = type(node).__name__
    children = getattr(node, "dataloaders", None)

    node_names: list[str] = list(
        getattr(node, "source_names", None) or getattr(node, "task_names", None) or []
    )

    if not children:
        merged: dict[str, str] = {}
        for d in dc_parts:
            merged.update(d)
        node_dc = _extract_data_config_info(node)
        if node_dc:
            merged.update(node_dc)
        detail_parts = [f"{attr}={merged[attr]}" for attr, _ in _DATA_CONFIG_EXTRACT_SPECS if attr in merged]
        dataset = getattr(node, "dataset", None)
        if dataset is not None:
            ds_hint = (
                getattr(dataset, "repo_id", None)
                or getattr(dataset, "dataset_id", None)
                or getattr(dataset, "name", None)
            )
            if ds_hint:
                detail_parts.append("dataset=" + str(ds_hint))
        detail = f"  [{', '.join(detail_parts)}]" if detail_parts else ""
        return [f"{indent}{cls_name} {instance_id}{detail}"]

    n = len(children)
    label = cls_name + " " + instance_id
    if node_names:
        label += " [" + ", ".join(node_names) + "]"
    lines: list[str] = [f"{indent}{label}"]
    child_indent = indent + "  "
    for i, child in enumerate(children):
        connector = "└─" if i == n - 1 else "├─"
        sub_lines = _build_loader_tree(child, seen, indent=child_indent + connector + " ", max_depth=max_depth - 1)
        lines.extend(sub_lines)
    return lines


def get_loader_ident(loader: Any) -> str:
    """Build a human-readable loader tree for refresh logs and source tagging.

    Shows only meaningful nodes:
    - MultiSourceDataLoader subclasses (RandomMixDataLoader, ProportionalMixDataLoader, …)
    - Real leaf loaders (TorchDataLoader, RLDSDataLoader)

    Transparent wrappers (SingleLoaderWrapper, BaseDataLoaderAdapter) are skipped;
    their data_config info is surfaced on the leaf nodes instead.
    """
    seen: set[int] = set()
    lines = _build_loader_tree(loader, seen)
    return "\n".join(lines) if lines else type(loader).__name__
