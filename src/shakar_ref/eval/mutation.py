from __future__ import annotations

from typing import Callable, Optional

from ..runtime import (
    BoundMethod,
    BuiltinMethod,
    Builtins,
    Descriptor,
    Frame,
    ShkArray,
    ShkCommand,
    ShkFn,
    ShkNull,
    ShkNumber,
    ShkObject,
    ShkSelector,
    SelectorIndex,
    SelectorSlice,
    ShkRegex,
    ShkString,
    ShkValue,
    ShakarIndexError,
    ShakarKeyError,
    ShakarRuntimeError,
    ShakarTypeError,
    call_shkfn,
)
from .selector import (
    clone_selector_parts,
    apply_selectors_to_value,
    _selector_index_to_int,
    _selector_slice_to_slice,
)

def set_field_value(recv: ShkValue, name: str, value: ShkValue, frame: Frame, *, create: bool) -> ShkValue:
    """Assign `recv.name = value`, honoring descriptors and creation semantics."""
    match recv:
        case ShkObject(slots=slots):
            slot = slots.get(name)

            if isinstance(slot, Descriptor):
                # property slot: defer to its setter so user code can enforce invariants.
                setter = slot.setter
                if setter is None:
                    raise ShakarRuntimeError(f"Property '{name}' is read-only")

                call_shkfn(setter, [value], subject=recv, caller_frame=frame)
                return value

            if slot is None and not create:
                raise ShakarRuntimeError(f"Field '{name}' is undefined; use ':=' to create it")

            slots[name] = value
            return value
        case _:
            raise ShakarTypeError(f"Cannot set field '{name}' on {type(recv).__name__}")

def set_index_value(recv: ShkValue, index: ShkValue, value: ShkValue, frame: Frame) -> ShkValue:
    """Assign `recv[index] = value` for arrays/objects with minimal coercions."""
    match recv:
        case ShkArray(items=items):
            if isinstance(index, ShkNumber):
                idx = int(index.value)
                items[idx] = value
                return value

            if isinstance(index, ShkSelector):
                _assign_selector_into_array(items, index, value)
                return value

            raise ShakarTypeError("Array index must be an integer or selector")
        case ShkObject(slots=slots):
            # objects store arbitrary keys; normalize to string for consistency.
            key = _normalize_index_key(index)
            slot = slots.get(key)

            if isinstance(slot, Descriptor):
                setter = slot.setter

                if setter is None:
                    raise ShakarRuntimeError(f"Property '{key}' is read-only")

                call_shkfn(setter, [value], subject=recv, caller_frame=frame)
                return value

            slots[key] = value
            return value
        case _:
            raise ShakarTypeError("Unsupported index assignment target")


def _assign_selector_into_array(items: list[ShkValue], selector: ShkSelector, value: ShkValue) -> None:
    """Broadcast assignment of `value` across indices/slices specified by selector."""
    length = len(items)

    for part in selector.parts:
        if isinstance(part, SelectorIndex):
            idx = _selector_index_to_int(part.value)
            pos = idx + length if idx < 0 else idx

            if pos < 0 or pos >= length:
                raise ShakarIndexError("Array index out of bounds")

            items[pos] = value
            continue

        if isinstance(part, SelectorSlice):
            slice_obj = _selector_slice_to_slice(part, length)
            for pos in range(*slice_obj.indices(length)):
                items[pos] = value
            continue

        raise ShakarTypeError("Unsupported selector component for array assignment")

def index_value(recv: ShkValue, idx: ShkValue, frame: Frame, default_thunk: Optional[Callable[[], ShkValue]]=None) -> ShkValue:
    """Read `recv[idx]`, supporting selectors, descriptors, and builtins."""
    match recv:
        case ShkArray(items=items):
            if default_thunk is not None:
                raise ShakarTypeError("Array index does not accept default")
            if isinstance(idx, ShkSelector):
                # cloning prevents later selectors from mutating the shared object.
                cloned = clone_selector_parts(idx.parts, clamp=True)
                return apply_selectors_to_value(recv, cloned)

            if isinstance(idx, ShkNumber):
                try:
                    return items[int(idx.value)]
                except IndexError:
                    raise ShakarIndexError("Array index out of bounds")
            raise ShakarTypeError("Array index must be a number")
        case ShkString(value=s):
            if default_thunk is not None:
                raise ShakarTypeError("String index does not accept default")
            if isinstance(idx, ShkSelector):
                cloned = clone_selector_parts(idx.parts, clamp=True)
                return apply_selectors_to_value(recv, cloned)

            if isinstance(idx, ShkNumber):
                try:
                    return ShkString(s[int(idx.value)])
                except IndexError:
                    raise ShakarIndexError("String index out of bounds")
            raise ShakarTypeError("String index must be a number")
        case ShkObject(slots=slots):
            key = _normalize_index_key(idx)

            if key in slots:
                val = slots[key]

                if isinstance(val, Descriptor):
                    # getter-only descriptor behaves like a computed property.
                    getter = val.getter
                    if getter is None:
                        return ShkNull()

                    return call_shkfn(getter, [], subject=recv, caller_frame=frame)

                return val
            if default_thunk is not None:
                return default_thunk()
            raise ShakarKeyError(key)
        case _:
            if default_thunk is not None:
                raise ShakarTypeError("Default index expects an object receiver")

            raise ShakarTypeError(
                f"Unsupported index operation on {type(recv).__name__} with {type(idx).__name__}"
            )

def slice_value(recv: ShkValue, start: Optional[int], stop: Optional[int], step: Optional[int]) -> ShkValue:
    """Return a shallow slice of an array/string (selector extraction)."""
    s = slice(start, stop, step)
    match recv:
        case ShkArray(items=items):
            return ShkArray(items[s])
        case ShkString(value=sval):
            return ShkString(sval[s])
        case _:
            raise ShakarTypeError("Slice only supported on arrays/strings")

def get_field_value(recv: ShkValue, name: str, frame: Frame) -> ShkValue:
    """Fetch `recv.name`, resolving descriptors and builtin method sugar."""
    match recv:
        case ShkObject(slots=slots):
            if name in slots:
                slot = slots[name]

                if isinstance(slot, Descriptor):
                    # defer to descriptor getter; absence returns nil to mirror go-style access.
                    getter = slot.getter

                    if getter is None:
                        return ShkNull()
                    return call_shkfn(getter, [], subject=recv, caller_frame=frame)

                if isinstance(slot, ShkFn):
                    # methods capture the receiver via BoundMethod to keep dot semantics.
                    return BoundMethod(slot, recv)

                return slot
            raise ShakarKeyError(name)
        case ShkArray(items=items):
            if name == "len":
                return ShkNumber(float(len(items)))
            if name == "high":
                size = len(items)
                return ShkNumber(size - 1) if size > 0 else ShkNumber(-1)
            if name in Builtins.array_methods:
                return BuiltinMethod(name=name, subject=recv)
            raise ShakarTypeError(f"Array has no field '{name}'")
        case ShkString(value=value):
            if name == "len":
                return ShkNumber(float(len(value)))
            if name == "high":
                size = len(value)
                return ShkNumber(size - 1) if size > 0 else ShkNumber(-1)

            if name in Builtins.string_methods:
                return BuiltinMethod(name=name, subject=recv)
            raise ShakarTypeError(f"String has no field '{name}'")
        case ShkRegex():
            if name in Builtins.regex_methods:
                return BuiltinMethod(name=name, subject=recv)
            raise ShakarTypeError(f"Regex has no field '{name}'")
        case ShkNumber():
            raise ShakarTypeError(f"Number has no field '{name}'")
        case ShkCommand():
            if name in Builtins.command_methods:
                return BuiltinMethod(name=name, subject=recv)
            raise ShakarTypeError(f"Command has no field '{name}'")
        case ShkFn():
            raise ShakarTypeError("Function has no fields")
        case _:
            raise ShakarTypeError(f"Unsupported field access on {type(recv).__name__}")

def _normalize_index_key(idx: ShkValue) -> str:
    """Map object index operands to canonical slot keys."""

    if isinstance(idx, ShkString):
        return idx.value

    if isinstance(idx, ShkNumber):
        return str(int(idx.value))

    raise ShakarTypeError("Object index must be a string or number value")
