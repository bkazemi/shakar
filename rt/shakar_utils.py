from __future__ import annotations

from typing import Any, Iterable, List, Optional

from shakar_runtime import (
    ShkNull,
    ShkNumber,
    ShkString,
    ShkBool,
    ShkArray,
    ShkObject,
    ShkFn,
    Descriptor,
    ShakarRuntimeError,
)

def value_in_list(seq: List[Any], value: Any) -> bool:
    for existing in seq:
        if shk_equals(existing, value):
            return True
    return False

def shk_equals(lhs: Any, rhs: Any) -> bool:
    if type(lhs) is not type(rhs):
        return False
    match lhs:
        case ShkNull():
            return True
        case ShkNumber(value=a):
            return a == rhs.value
        case ShkString(value=a):
            return a == rhs.value
        case ShkBool(value=a):
            return a == rhs.value
        case ShkArray(items=items):
            rhs_items = rhs.items
            if len(items) != len(rhs_items):
                return False
            return all(shk_equals(a, b) for a, b in zip(items, rhs_items))
        case ShkObject(slots=slots):
            rhs_slots = rhs.slots
            if slots.keys() != rhs_slots.keys():
                return False
            return all(shk_equals(slots[k], rhs_slots[k]) for k in slots)
        case ShkFn():
            return lhs is rhs
        case Descriptor():
            return lhs is rhs
        case _:
            return lhs is rhs

def is_sequence_value(value: Any) -> bool:
    return isinstance(value, (ShkArray, list, tuple))

def sequence_items(value: Any) -> List[Any]:
    if isinstance(value, ShkArray):
        return list(value.items)
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return []

def coerce_sequence(value: Any, expected_len: Optional[int]) -> Optional[List[Any]]:
    if not is_sequence_value(value):
        return None
    items = sequence_items(value)
    if expected_len is not None and len(items) != expected_len:
        raise ShakarRuntimeError("Destructure arity mismatch")
    return items

def fanout_values(value: Any, count: int) -> List[Any]:
    if isinstance(value, ShkArray) and len(value.items) == count:
        return list(value.items)
    if isinstance(value, list) and len(value) == count:
        return list(value)
    return [value] * count

def replicate_empty_sequence(value: Any, count: int) -> List[Any]:
    if isinstance(value, ShkArray) and len(value.items) == 0:
        return [ShkArray([]) for _ in range(count)]
    if isinstance(value, list) and len(value) == 0:
        return [[] for _ in range(count)]
    if isinstance(value, tuple) and len(value) == 0:
        return [tuple() for _ in range(count)]
    return [value] * count

def normalize_object_key(value: Any) -> str:
    match value:
        case ShkString(value=s):
            return s
        case ShkNumber(value=num):
            return str(int(num)) if num.is_integer() else str(num)
        case ShkBool(value=b):
            return 'true' if b else 'false'
        case ShkNull():
            return 'null'
        case _:
            return str(value)
