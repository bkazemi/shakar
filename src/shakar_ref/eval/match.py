"""
Structural Match Logic for the ~ Operator

Implements recursive structural matching as specified:
- Object matching: RHS keys must exist in LHS with recursive match
- Type matching: LHS must be instance of RHS type
- Selector matching: LHS must be in RHS selector
- Value matching: fallback to equality
"""

from typing import Optional

from ..types import (
    ShkValue,
    ShkArray,
    ShkObject,
    ShkType,
    ShkSelector,
    ShkNumber,
    ShkOptional,
    ShkRepeatSchema,
    ShkUnion,
    ShkEnvVar,
    ShkNil,
    ShkString,
    ShakarRuntimeError,
)
from ..utils import shk_equals, envvar_value_by_name


def _validate_schema(
    schema: ShkValue,
    _seen: Optional[set] = None,
    _in_array_tail: bool = False,
) -> None:
    """Recursively validate that repeat schemas only appear in final array position.

    Walks the full RHS schema tree so malformed nested schemas are always
    rejected regardless of lhs shape (e.g. empty arrays that would
    short-circuit before reaching the invalid inner schema).

    Tracks visited nodes by identity to handle cyclic runtime objects.
    _in_array_tail is True only when visiting the last element of an array
    schema, the sole position where ShkRepeatSchema is permitted.
    """
    if _seen is None:
        _seen = set()

    node_id = id(schema)

    if isinstance(schema, ShkRepeatSchema):
        if not _in_array_tail:
            raise ShakarRuntimeError(
                "Repeat schema (...) is only valid in the final position"
                " of an array schema"
            )
        # Context-independent: inner only needs checking once
        if node_id not in _seen:
            _seen.add(node_id)
            _validate_schema(schema.inner, _seen)
        return

    # Non-repeat nodes: context doesn't affect validity, skip if seen
    if node_id in _seen:
        return
    _seen.add(node_id)

    if isinstance(schema, ShkArray):
        items = schema.items
        for i, item in enumerate(items):
            is_tail = i == len(items) - 1
            _validate_schema(item, _seen, _in_array_tail=is_tail)

    elif isinstance(schema, ShkOptional):
        _validate_schema(schema.inner, _seen)

    elif isinstance(schema, ShkObject):
        for val in schema.slots.values():
            _validate_schema(val, _seen)

    elif isinstance(schema, ShkUnion):
        for alt in schema.alternatives:
            _validate_schema(alt, _seen)


def match_structure(lhs: ShkValue, rhs: ShkValue) -> bool:
    """
    Structural match: lhs ~ rhs

    Rules:
    1. If rhs is ShkUnion, check if lhs matches any alternative
    2. If rhs is ShkObject, check object structure (subset match)
    3. If rhs is ShkType, check type membership
    4. If rhs is ShkSelector, check membership
    5. Otherwise, check value equality
    """
    # Validate full schema tree upfront so malformed schemas always error
    # regardless of lhs shape (avoids data-dependent validation).
    _validate_schema(rhs)

    return _match_value(lhs, rhs)


def _match_value(lhs: ShkValue, rhs: ShkValue) -> bool:
    """Inner match logic — called after schema validation."""
    # Coerce env vars to their string value (or nil) before matching
    if isinstance(lhs, ShkEnvVar):
        env_val = envvar_value_by_name(lhs.name)
        lhs = ShkNil() if env_val is None else ShkString(env_val)
    if isinstance(rhs, ShkUnion):
        return any(_match_value(lhs, alt) for alt in rhs.alternatives)

    if isinstance(rhs, ShkType):
        return isinstance(lhs, rhs.mapped_type)

    if isinstance(rhs, ShkSelector):
        if not isinstance(lhs, ShkNumber):
            return False
        from .selector import selector_contains

        return selector_contains(rhs, lhs)

    if isinstance(rhs, ShkArray):
        if not isinstance(lhs, ShkArray):
            return False

        rhs_items = rhs.items
        n = len(rhs_items)

        # Empty schema: only empty arrays match
        if n == 0:
            return len(lhs.items) == 0

        # Check if last element is a repeat schema
        if isinstance(rhs_items[-1], ShkRepeatSchema):
            tail_schema = rhs_items[-1].inner
            positional = rhs_items[:-1]
            pos_count = len(positional)

            # LHS must have at least pos_count elements
            if len(lhs.items) < pos_count:
                return False

            # Check positional elements
            for i in range(pos_count):
                if not _match_value(lhs.items[i], positional[i]):
                    return False

            # Check remaining elements against tail schema
            for i in range(pos_count, len(lhs.items)):
                if not _match_value(lhs.items[i], tail_schema):
                    return False

            return True

        # Positional (no repeat): exact length + per-element match
        if len(lhs.items) != n:
            return False

        return all(_match_value(lhs.items[i], rhs_items[i]) for i in range(n))

    if isinstance(rhs, ShkObject):
        if not isinstance(lhs, ShkObject):
            return False

        for key in rhs.slots:
            schema_value = rhs.slots[key]

            # Handle optional fields
            if isinstance(schema_value, ShkOptional):
                if key not in lhs.slots:
                    continue  # Missing optional key is OK
                # Key present, check inner schema
                if not _match_value(lhs.slots[key], schema_value.inner):
                    return False
            else:
                # Non-optional field: must exist
                if key not in lhs.slots:
                    return False
                if not _match_value(lhs.slots[key], schema_value):
                    return False

        return True

    return shk_equals(lhs, rhs)
