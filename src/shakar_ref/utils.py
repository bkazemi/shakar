from __future__ import annotations

import os as _os
import math
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Optional, Tuple

from .types import (
    ShkValue,
    DecoratorConfigured,
    DecoratorContinuation,
    ShkEnvVar,
    ShkNil,
    ShkNumber,
    ShkString,
    ShkBool,
    ShkDuration,
    ShkSize,
    ShkArray,
    ShkSet,
    ShkFan,
    ShkObject,
    ShkPath,
    ShkFn,
    ShkDecorator,
    ShkCommand,
    Descriptor,
    ShakarRuntimeError,
    ShakarTypeError,
)


def value_in_list(seq: List[ShkValue], value: ShkValue) -> bool:
    for existing in seq:
        if shk_equals(existing, value):
            return True

    return False


def _normalized_number_key(value: float) -> Tuple[int, float]:
    """Produce a deterministic sort key for floating-point numbers."""
    if math.isnan(value):
        return (1, 0.0)

    if value == 0:
        return (0, 0.0)

    return (0, value)


def _set_sort_key(value: ShkValue) -> Tuple[object, ...]:
    """Return a total-order key for deterministic set ordering."""
    if isinstance(value, ShkNil):
        return (0, 0)

    if isinstance(value, ShkBool):
        return (1, 1 if value.value else 0)

    if isinstance(value, ShkNumber):
        return (2, _normalized_number_key(value.value))

    if isinstance(value, ShkDuration):
        return (3, value.nanos)

    if isinstance(value, ShkSize):
        return (4, value.byte_count)

    if isinstance(value, ShkString):
        return (5, value.value)

    if isinstance(value, ShkPath):
        return (6, value.value)

    if isinstance(value, ShkEnvVar):
        return (7, value.name)

    if isinstance(value, ShkArray):
        return (8, tuple(_set_sort_key(item) for item in value.items))

    if isinstance(value, ShkSet):
        normalized_items = normalize_set_items(value.items)
        return (9, tuple(_set_sort_key(item) for item in normalized_items))

    if isinstance(value, ShkFan):
        return (10, tuple(_set_sort_key(item) for item in value.items))

    if isinstance(value, ShkObject):
        slots = tuple(
            (key, _set_sort_key(slot_value))
            for key, slot_value in sorted(value.slots.items())
        )
        return (11, slots)

    return (99, type(value).__name__, repr(value))


def normalize_set_items(items: List[ShkValue]) -> List[ShkValue]:
    """Deduplicate and deterministically sort set elements."""
    deduped: List[ShkValue] = []

    for item in items:
        if not value_in_list(deduped, item):
            deduped.append(item)

    return sorted(deduped, key=_set_sort_key)


def envvar_is_nil(env: ShkEnvVar) -> bool:
    """Check if an env var has no value (doesn't exist)."""
    return env.name not in _os.environ


def envvar_value(env: ShkEnvVar) -> Optional[str]:
    """Get the current value of an env var, or None if missing."""
    return _os.environ.get(env.name)


def envvar_value_by_name(name: str) -> Optional[str]:
    """Get the current value of an env var by name, or None if missing."""
    return _os.environ.get(name)


def debug_py_trace_enabled() -> bool:
    """Check if Python traceback capture is enabled (debug-only)."""
    raw = envvar_value_by_name("SHAKAR_DEBUG_PY_TRACE")
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def is_nil_like(value: ShkValue) -> bool:
    """Check if a value is nil or an env var with missing value."""
    if isinstance(value, ShkNil):
        return True
    if isinstance(value, ShkEnvVar):
        return envvar_is_nil(value)
    return False


def shk_equals(lhs: ShkValue, rhs: ShkValue) -> bool:
    match (lhs, rhs):
        case (ShkNil(), ShkNil()):
            return True
        # EnvVar compared with nil: true if env var doesn't exist
        case (ShkEnvVar() as env, ShkNil()) | (ShkNil(), ShkEnvVar() as env):
            return envvar_is_nil(env)
        # EnvVar compared with EnvVar: compare resolved values (nil if missing)
        case (ShkEnvVar() as env_a, ShkEnvVar() as env_b):
            val_a = envvar_value(env_a)
            val_b = envvar_value(env_b)
            if val_a is None and val_b is None:
                return True
            return val_a == val_b
        # EnvVar compared with string: compare values
        case (ShkEnvVar() as env, ShkString(value=s)):
            env_val = envvar_value(env)
            return env_val is not None and env_val == s
        case (ShkString(value=s), ShkEnvVar() as env):
            env_val = envvar_value(env)
            return env_val is not None and s == env_val
        case (ShkNumber(value=a), ShkNumber(value=b)):
            return a == b
        case (ShkString(value=a), ShkString(value=b)):
            return a == b
        case (ShkBool(value=a), ShkBool(value=b)):
            return a == b
        case (ShkDuration(nanos=a), ShkDuration(nanos=b)):
            return a == b
        case (ShkSize(byte_count=a), ShkSize(byte_count=b)):
            return a == b
        case (ShkArray(items=items_a), ShkArray(items=items_b)):
            return len(items_a) == len(items_b) and all(
                shk_equals(a, b) for a, b in zip(items_a, items_b)
            )
        case (ShkSet(items=items_a), ShkSet(items=items_b)):
            # Sets are equal if they have same length and every element in a is in b
            if len(items_a) != len(items_b):
                return False
            return all(any(shk_equals(a, b) for b in items_b) for a in items_a)
        case (ShkFan(items=items_a), ShkFan(items=items_b)):
            return len(items_a) == len(items_b) and all(
                shk_equals(a, b) for a, b in zip(items_a, items_b)
            )
        case (ShkObject(slots=slots_a), ShkObject(slots=slots_b)):
            return slots_a.keys() == slots_b.keys() and all(
                shk_equals(slots_a[k], slots_b[k]) for k in slots_a
            )
        case (ShkPath(value=a), ShkPath(value=b)):
            return a == b
        case (
            (ShkFn(), ShkFn())
            | (ShkDecorator(), ShkDecorator())
            | (DecoratorConfigured(), DecoratorConfigured())
            | (DecoratorContinuation(), DecoratorContinuation())
            | (Descriptor(), Descriptor())
        ):
            return lhs is rhs
        case _:
            return False


def is_sequence_value(value: ShkValue) -> bool:
    return isinstance(value, (ShkArray, ShkSet, ShkFan))


def sequence_items(value: ShkValue) -> List[ShkValue]:
    if isinstance(value, (ShkArray, ShkSet, ShkFan)):
        return list(value.items)
    raise ShakarTypeError("Expected array, set, or fan for sequence operations")


def coerce_sequence(
    value: ShkValue, expected_len: Optional[int]
) -> Optional[List[ShkValue]]:
    if not is_sequence_value(value):
        return None

    items = sequence_items(value)

    if expected_len is not None and len(items) != expected_len:
        raise ShakarRuntimeError("Destructure arity mismatch")

    return items


def fanout_values(value: ShkValue, count: int) -> List[ShkValue]:
    if isinstance(value, (ShkArray, ShkSet, ShkFan)) and len(value.items) == count:
        return list(value.items)

    return [value] * count


def replicate_empty_sequence(value: ShkValue, count: int) -> List[ShkValue]:
    if isinstance(value, (ShkArray, ShkSet, ShkFan)) and len(value.items) == 0:
        if isinstance(value, ShkFan):
            empty: ShkValue = ShkFan([])
        elif isinstance(value, ShkSet):
            empty = ShkSet([])
        else:
            empty = ShkArray([])
        return [empty for _ in range(count)]

    return [value] * count


def normalize_object_key(value: ShkValue) -> str:
    match value:
        case ShkString(value=s):
            return s
        case ShkNumber(value=num):
            return str(int(num)) if num.is_integer() else str(num)
        case ShkBool(value=b):
            return "true" if b else "false"
        case ShkNil():
            return "nil"
        case ShkEnvVar() as env:
            env_val = envvar_value(env)
            if env_val is None:
                raise ShakarTypeError(
                    f"Env var '{env.name}' has no value for object key"
                )
            return env_val
        case _:
            raise ShakarTypeError(
                "Object key must be a Shakar string, number, bool, null, or env var"
            )


def stringify(value: Optional[ShkValue]) -> str:
    if isinstance(value, ShkPath):
        return str(value)

    if isinstance(value, ShkEnvVar):
        env_val = envvar_value_by_name(value.name)
        return env_val if env_val is not None else "nil"

    if isinstance(value, ShkString):
        return value.value

    if isinstance(value, ShkNumber):
        return str(value)

    if isinstance(value, ShkBool):
        return "true" if value.value else "false"

    if isinstance(value, ShkNil) or value is None:
        return "nil"

    if isinstance(value, ShkSet):
        inner = ", ".join(stringify(item) for item in value.items)
        return f"set{{{inner}}}"

    if isinstance(value, ShkCommand):
        return value.render()

    return str(value)


def parse_compound_literal(
    raw: str, unit_values: Dict[str, int], kind: str
) -> Tuple[int, Tuple[str, ...]]:
    """Parse a compound literal string and return (total_value, units_tuple).

    Reusable by both the lexer (for source literals) and stdlib conversion
    functions (for runtime string=>duration/size parsing).
    Raises ShakarTypeError on malformed input.
    """
    unit_keys = sorted(unit_values.keys(), key=len, reverse=True)
    parts: List[Tuple[str, str]] = []
    pos = 0

    while pos < len(raw):
        start = pos

        # Consume digits (with underscores)
        while pos < len(raw) and (raw[pos].isdigit() or raw[pos] == "_"):
            pos += 1

        # Optional fractional part
        if pos < len(raw) and raw[pos] == ".":
            pos += 1
            while pos < len(raw) and (raw[pos].isdigit() or raw[pos] == "_"):
                pos += 1

        # Optional exponent
        if pos < len(raw) and raw[pos] in {"e", "E"}:
            pos += 1
            if pos < len(raw) and raw[pos] in {"+", "-"}:
                pos += 1
            while pos < len(raw) and (raw[pos].isdigit() or raw[pos] == "_"):
                pos += 1

        num_str = raw[start:pos]
        if not num_str:
            raise ShakarTypeError(f"Cannot convert string to {kind}: malformed literal")

        # Match unit suffix
        unit = None
        for u in unit_keys:
            if raw[pos : pos + len(u)] == u:
                unit = u
                pos += len(u)
                break

        if unit is None:
            raise ShakarTypeError(f"Cannot convert string to {kind}: malformed literal")

        parts.append((num_str, unit))

    if not parts:
        raise ShakarTypeError(f"Cannot convert string to {kind}: empty literal")

    # Validate compound literals: no decimals, no duplicate units
    if len(parts) > 1:
        if any("." in n or "e" in n.lower() for n, _ in parts):
            raise ShakarTypeError(
                f"Cannot convert string to {kind}: decimal in compound literal"
            )
        seen: set[str] = set()
        for _, u in parts:
            if u in seen:
                raise ShakarTypeError(
                    f"Cannot convert string to {kind}: duplicate unit '{u}'"
                )
            seen.add(u)

    total = Decimal(0)
    for num_str, unit in parts:
        try:
            num = Decimal(num_str)
        except InvalidOperation:
            raise ShakarTypeError(f"Cannot convert string to {kind}: malformed literal")
        total += num * Decimal(unit_values[unit])

    if total != total.to_integral_value():
        raise ShakarTypeError(
            f"Cannot convert string to {kind}: not representable as integer"
        )

    # Sort largest-first to match _format_places expectation in types.py
    units = tuple(
        sorted((u for _, u in parts), key=lambda u: unit_values[u], reverse=True)
    )

    return int(total), units
