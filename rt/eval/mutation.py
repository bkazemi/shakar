from __future__ import annotations

from typing import Any

from shakar_runtime import (
    Env,
    BoundMethod,
    Descriptor,
    ShkArray,
    ShkFn,
    ShkNull,
    ShkNumber,
    ShkObject,
    ShkSelector,
    ShkString,
    ShakarRuntimeError,
    ShakarTypeError,
    call_shkfn,
)
from eval.selector import clone_selector_parts, apply_selectors_to_value

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
            raise ShakarTypeError("Array index must be a number")
        case ShkString(value=s):
            if isinstance(idx, ShkSelector):
                cloned = clone_selector_parts(idx.parts, clamp=True)
                return apply_selectors_to_value(recv, cloned, env)
            if isinstance(idx, ShkNumber):
                return ShkString(s[int(idx.value)])
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

def slice_value(recv: Any, start: int | None, stop: int | None, step: int | None) -> Any:
    s = slice(start, stop, step)
    match recv:
        case ShkArray(items=items):
            return ShkArray(items[s])
        case ShkString(value=sval):
            return ShkString(sval[s])
        case _:
            raise ShakarTypeError("Slice only supported on arrays/strings")

def get_field_value(recv: Any, name: str, env: Env) -> Any:
    match recv:
        case ShkObject(slots=slots):
            if name in slots:
                slot = slots[name]
                if isinstance(slot, Descriptor):
                    getter = slot.getter
                    if getter is None:
                        return ShkNull()
                    return call_shkfn(getter, [], subject=recv, caller_env=env)
                if isinstance(slot, ShkFn):
                    return BoundMethod(slot, recv)
                return slot
            raise ShakarRuntimeError(f"Key '{name}' not found")
        case ShkArray(items=items):
            if name == "len":
                return ShkNumber(float(len(items)))
            raise ShakarTypeError(f"Array has no field '{name}'")
        case ShkString(value=value):
            if name == "len":
                return ShkNumber(float(len(value)))
            raise ShakarTypeError(f"String has no field '{name}'")
        case ShkFn():
            raise ShakarTypeError("Function has no fields")
        case _:
            raise ShakarTypeError(f"Unsupported field access on {type(recv).__name__}")

def _normalize_index_key(idx: Any) -> str:
    if isinstance(idx, ShkString):
        return idx.value
    if isinstance(idx, ShkNumber):
        return str(int(idx.value))
    raise ShakarTypeError("Object index must be a string or number value")
