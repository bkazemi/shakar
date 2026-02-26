from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Union

from ..runtime import Frame, ShkValue, ShakarRuntimeError
from .mutation import set_field_value, set_index_value

WriteTargetKind = Literal["ident", "field", "index", "selector", "noop"]


@dataclass
class WriteTarget:
    """Resolved writeback destination captured once at context-build time."""

    kind: WriteTargetKind
    owner: Optional[ShkValue]
    name_or_index: Optional[Union[str, ShkValue]]
    frame: Frame
    create: bool = False


def apply_write_target(target: WriteTarget, new_value: ShkValue) -> ShkValue:
    """Write to a pre-resolved target without re-evaluating AST/index expressions."""
    kind = target.kind
    key = target.name_or_index

    if kind == "noop":
        return new_value

    if kind == "ident":
        if not isinstance(key, str):
            raise ShakarRuntimeError("Invalid identifier write target")
        try:
            target.frame.set(key, new_value)
        except ShakarRuntimeError:
            if target.create:
                target.frame.define(key, new_value)
            else:
                raise
    elif kind == "field":
        if not isinstance(key, str) or target.owner is None:
            raise ShakarRuntimeError("Invalid field write target")
        set_field_value(
            target.owner, key, new_value, target.frame, create=target.create
        )
    elif kind in {"index", "selector"}:
        if key is None or target.owner is None:
            raise ShakarRuntimeError("Invalid index write target")
        if isinstance(key, str):
            raise ShakarRuntimeError("Invalid index write target")
        set_index_value(target.owner, key, new_value, target.frame)
    else:
        raise ShakarRuntimeError(f"Unsupported write target kind '{kind}'")

    target.frame.dot = new_value
    return new_value
