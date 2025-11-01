from __future__ import annotations

from typing import Any

from shakar_runtime import (
    Env,
    Descriptor,
    ShkArray,
    ShkNull,
    ShkNumber,
    ShkObject,
    ShkSelector,
    ShkString,
    ShakarRuntimeError,
    ShakarTypeError,
    call_shkfn,
)
from eval.selector_eval import clone_selector_parts, apply_selectors_to_value

def set_field_value(recv: Any, name: str, value: Any, env: Env, *, create: bool) -> Any:
    match recv:
        case ShkObject(slots=slots):
            slot = slots.get(name)
            if isinstance(slot, Descriptor):
                setter = slot.setter
                if setter is None:
                    raise ShakarRuntimeError(f"Property '{name}' is read-only")
                call_shkfn(setter, [value], subject=recv, caller_env=env)
                return value
            slots[name] = value
            return value
        case _:
            raise ShakarTypeError(f"Cannot set field '{name}' on {type(recv).__name__}")

def set_index_value(recv: Any, index: Any, value: Any, env: Env) -> Any:
    match recv:
        case ShkArray(items=items):
            if isinstance(index, ShkNumber):
                idx = int(index.value)
            elif isinstance(index, int):
                idx = index
            else:
                raise ShakarTypeError("Array index must be an integer")
            items[idx] = value
            return value
        case ShkObject(slots=slots):
            key = _normalize_index_key(index)
            slots[key] = value
            return value
        case _:
            raise ShakarTypeError("Unsupported index assignment target")

def index_value(recv: Any, idx: Any, env: Env) -> Any:
    match recv:
        case ShkArray(items=items):
            if isinstance(idx, ShkSelector):
                cloned = clone_selector_parts(idx.parts, clamp=True)
                return apply_selectors_to_value(recv, cloned, env)
            if isinstance(idx, ShkNumber):
                return items[int(idx.value)]
            if isinstance(idx, int):
                return items[idx]
            raise ShakarTypeError("Array index must be a number")
        case ShkString(value=s):
            if isinstance(idx, ShkSelector):
                cloned = clone_selector_parts(idx.parts, clamp=True)
                return apply_selectors_to_value(recv, cloned, env)
            if isinstance(idx, ShkNumber):
                return ShkString(s[int(idx.value)])
            if isinstance(idx, int):
                return ShkString(s[idx])
            raise ShakarTypeError("String index must be a number")
        case ShkObject(slots=slots):
            key = _normalize_index_key(idx)
            if key in slots:
                val = slots[key]
                if isinstance(val, Descriptor):
                    getter = val.getter
                    if getter is None:
                        return ShkNull()
                    return call_shkfn(getter, [], subject=recv, caller_env=env)
                return val
            raise ShakarRuntimeError(f"Key '{key}' not found")
        case _:
            raise ShakarTypeError("Unsupported index operation")

def _normalize_index_key(idx: Any) -> str:
    if isinstance(idx, ShkString):
        return idx.value
    if isinstance(idx, ShkNumber):
        return str(int(idx.value))
    if isinstance(idx, (int, bool)):
        return str(int(idx))
    return str(idx)
