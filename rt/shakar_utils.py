from __future__ import annotations

from typing import Any, List

from shakar_runtime import (
    ShkNull,
    ShkNumber,
    ShkString,
    ShkBool,
    ShkArray,
    ShkObject,
    ShkFn,
    Descriptor,
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
