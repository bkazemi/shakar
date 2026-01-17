from __future__ import annotations

from typing import List, Optional

from .types import (
    ShkValue,
    DecoratorConfigured,
    DecoratorContinuation,
    ShkEnvVar,
    ShkNull,
    ShkNumber,
    ShkString,
    ShkBool,
    ShkDuration,
    ShkSize,
    ShkArray,
    ShkObject,
    ShkPath,
    ShkFn,
    ShkDecorator,
    Descriptor,
    ShakarRuntimeError,
    ShakarTypeError,
)


def value_in_list(seq: List[ShkValue], value: ShkValue) -> bool:
    for existing in seq:
        if shk_equals(existing, value):
            return True

    return False


import os as _os


def envvar_is_nil(env: ShkEnvVar) -> bool:
    """Check if an env var has no value (doesn't exist)."""
    return env.name not in _os.environ


def envvar_value(env: ShkEnvVar) -> Optional[str]:
    """Get the current value of an env var, or None if missing."""
    return _os.environ.get(env.name)


def envvar_value_by_name(name: str) -> Optional[str]:
    """Get the current value of an env var by name, or None if missing."""
    return _os.environ.get(name)


def is_nil_like(value: ShkValue) -> bool:
    """Check if a value is nil or an env var with missing value."""
    if isinstance(value, ShkNull):
        return True
    if isinstance(value, ShkEnvVar):
        return envvar_is_nil(value)
    return False


def shk_equals(lhs: ShkValue, rhs: ShkValue) -> bool:
    match (lhs, rhs):
        case (ShkNull(), ShkNull()):
            return True
        # EnvVar compared with nil: true if env var doesn't exist
        case (ShkEnvVar() as env, ShkNull()) | (ShkNull(), ShkEnvVar() as env):
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
    return isinstance(value, ShkArray)


def sequence_items(value: ShkValue) -> List[ShkValue]:
    if isinstance(value, ShkArray):
        return list(value.items)
    raise ShakarTypeError("Expected ShkArray for sequence operations")


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
    if isinstance(value, ShkArray) and len(value.items) == count:
        return list(value.items)

    return [value] * count


def replicate_empty_sequence(value: ShkValue, count: int) -> List[ShkValue]:
    if isinstance(value, ShkArray) and len(value.items) == 0:
        return [ShkArray([]) for _ in range(count)]

    return [value] * count


def normalize_object_key(value: ShkValue) -> str:
    match value:
        case ShkString(value=s):
            return s
        case ShkNumber(value=num):
            return str(int(num)) if num.is_integer() else str(num)
        case ShkBool(value=b):
            return "true" if b else "false"
        case ShkNull():
            return "null"
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
