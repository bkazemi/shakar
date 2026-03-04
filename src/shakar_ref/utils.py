from __future__ import annotations

import os as _os
import math
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Optional, Set, Tuple

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
    ShkSelector,
    SelectorIndex,
    SelectorSlice,
    SelectorPart,
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


def _selector_part_sort_key(
    part: "SelectorIndex | SelectorSlice",
) -> Tuple[object, ...]:
    """Return a sort key for a single selector part (index or slice).

    Ordering: index < slice.
    - Index parts delegate to _set_sort_key on their ShkValue.
    - Slice parts compare fields directly with integer/sentinel logic:
      start: None sorts before concrete values.
      stop: None sorts after concrete values (unbounded upper range).
      step: None sorts before concrete values.
      exclusive_stop: False before True.
      clamp: False before True.
    """
    if isinstance(part, SelectorIndex):
        # kind 0 => index sorts before slice
        return (0, _set_sort_key(part.value))

    # SelectorSlice
    # Sentinels: for start/step, None sorts first => (-inf).
    # For stop, None sorts last => (+inf).
    NEG_INF = float("-inf")
    POS_INF = float("inf")

    start_key = NEG_INF if part.start is None else part.start
    stop_key = POS_INF if part.stop is None else part.stop
    step_key = NEG_INF if part.step is None else part.step

    return (
        1,
        start_key,
        stop_key,
        step_key,
        0 if not part.exclusive_stop else 1,
        0 if not part.clamp else 1,
    )


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

    if isinstance(value, ShkSelector):
        return (12, tuple(_selector_part_sort_key(p) for p in value.parts))

    return (99, type(value).__name__, repr(value))


def normalize_set_items(items: List[ShkValue]) -> List[ShkValue]:
    """Deduplicate and deterministically sort set elements."""
    deduped: List[ShkValue] = []

    for item in items:
        if not value_in_list(deduped, item):
            deduped.append(item)

    return sorted(deduped, key=_set_sort_key)


def compact_selector_parts(parts: List[SelectorPart]) -> List[SelectorPart]:
    """Merge selector parts into a minimal union-preserving representation."""
    canonical_parts: List[SelectorSlice] = []
    passthrough_parts: List[SelectorPart] = []
    # Track which integer positions originated from index selectors, so we
    # can collapse them back after merge without converting real slices.
    index_origins: Set[int] = set()

    for part in parts:
        if isinstance(part, SelectorIndex):
            idx = _selector_compact_index_value(part.value)
            index_origins.add(idx)
            canonical_parts.append(_selector_compact_index_to_slice(part))
            continue

        if isinstance(part, SelectorSlice):
            canonical_part = _selector_compact_canonicalize_slice(part)

            if canonical_part is None:
                passthrough_parts.append(part)
            else:
                canonical_parts.append(canonical_part)

    merged_parts = _selector_compact_merge_canonical(canonical_parts)
    all_parts: List[SelectorPart] = []
    all_parts.extend(merged_parts)
    all_parts.extend(passthrough_parts)

    deduped_parts = _selector_compact_dedupe(all_parts)
    deduped_parts.sort(key=_selector_compact_sort_key)

    return [_selector_compact_minimize(p, index_origins) for p in deduped_parts]


def _selector_compact_index_to_slice(part: SelectorIndex) -> SelectorSlice:
    idx = _selector_compact_index_value(part.value)
    return SelectorSlice(
        start=idx,
        stop=idx,
        step=1,
        clamp=False,
        exclusive_stop=False,
    )


def _selector_compact_index_value(value: ShkValue) -> int:
    if not isinstance(value, ShkNumber):
        raise ShakarTypeError("compact() expects numeric selector indices")

    num = value.value
    if not float(num).is_integer():
        raise ShakarTypeError("compact() expects integral selector indices")

    return int(num)


def _selector_compact_canonicalize_slice(
    part: SelectorSlice,
) -> Optional[SelectorSlice]:
    step = part.step

    # Non-unit steps are length-dependent and are intentionally excluded from merge math.
    if step is not None and step != 1:
        return None

    stop = part.stop
    if part.exclusive_stop and stop is not None:
        stop -= 1

    return SelectorSlice(
        start=part.start,
        stop=stop,
        step=1,
        clamp=part.clamp,
        exclusive_stop=False,
    )


def _selector_compact_merge_canonical(parts: List[SelectorSlice]) -> List[SelectorPart]:
    grouped: Dict[Tuple[bool, str], List[SelectorSlice]] = {}
    passthrough_mixed: List[SelectorPart] = []

    for part in parts:
        sign_partition = _selector_compact_sign_partition(part)

        if sign_partition == "mixed":
            passthrough_mixed.append(part)
            continue

        key = (part.clamp, sign_partition)
        group = grouped.get(key)
        if group is None:
            group = []
            grouped[key] = group
        group.append(part)

    merged: List[SelectorPart] = []
    for group in grouped.values():
        merged.extend(_selector_compact_merge_group(group))

    merged.extend(passthrough_mixed)
    return merged


def _selector_compact_sign_partition(part: SelectorSlice) -> str:
    start = part.start
    stop = part.stop

    start_non_negative = start is None or start >= 0
    stop_non_negative = stop is None or stop >= 0

    if start_non_negative and stop_non_negative:
        return "non_negative"

    # Fully negative ranges are length-dependent after normalization, so
    # merging them with interval math is not union-preserving. Treat them
    # as unmergeable (same as mixed-sign).
    return "mixed"


def _selector_compact_merge_group(parts: List[SelectorSlice]) -> List[SelectorPart]:
    ordered = sorted(parts, key=_selector_compact_group_sort_key)
    if not ordered:
        return []

    merged: List[SelectorPart] = [ordered[0]]

    for part in ordered[1:]:
        current = merged[-1]
        assert isinstance(current, SelectorSlice)

        if _selector_compact_can_merge(current, part):
            merged[-1] = _selector_compact_merge_pair(current, part)
        else:
            merged.append(part)

    return merged


def _selector_compact_group_sort_key(part: SelectorSlice) -> Tuple[object, ...]:
    start_key = 0 if part.start is None else part.start
    none_start_key = 0 if part.start is None else 1
    stop_key = float("inf") if part.stop is None else part.stop
    return (start_key, none_start_key, stop_key)


def _selector_compact_can_merge(left: SelectorSlice, right: SelectorSlice) -> bool:
    left_stop = left.stop
    if left_stop is None:
        return True

    right_start = 0 if right.start is None else right.start
    return right_start <= left_stop + 1


def _selector_compact_merge_pair(
    left: SelectorSlice, right: SelectorSlice
) -> SelectorSlice:
    if left.start is None or right.start is None:
        merged_start = None
    else:
        merged_start = min(left.start, right.start)

    if left.stop is None or right.stop is None:
        merged_stop = None
    else:
        merged_stop = max(left.stop, right.stop)

    return SelectorSlice(
        start=merged_start,
        stop=merged_stop,
        step=1,
        clamp=left.clamp,
        exclusive_stop=False,
    )


def _selector_compact_dedupe(parts: List[SelectorPart]) -> List[SelectorPart]:
    deduped: List[SelectorPart] = []
    seen: Set[Tuple[object, ...]] = set()

    for part in parts:
        # Sort key is injective (unique per distinct part), so it doubles as identity.
        key = _selector_part_sort_key(part)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(part)

    return deduped


def _selector_compact_sort_key(part: SelectorPart) -> Tuple[object, ...]:
    if isinstance(part, SelectorIndex):
        start_key = _selector_compact_index_value(part.value)
        none_start_key = 1
    else:
        if part.start is None:
            start_key = 0
            none_start_key = 0
        else:
            start_key = part.start
            none_start_key = 1

    return (start_key, none_start_key, _selector_part_sort_key(part))


def _selector_compact_minimize(
    part: SelectorPart, index_origins: Set[int]
) -> SelectorPart:
    if not isinstance(part, SelectorSlice):
        return part

    # Normalize step=1 to step=None (the default representation).
    normalized_step: Optional[int] = None if part.step == 1 else part.step

    if normalized_step != part.step:
        normalized = SelectorSlice(
            start=part.start,
            stop=part.stop,
            step=normalized_step,
            clamp=part.clamp,
            exclusive_stop=part.exclusive_stop,
        )
    else:
        normalized = part

    # Only collapse singleton slices back to index if they originated from
    # an index selector. Real slices (e.g. `7:7`) must stay slices because
    # slices clamp on OOB while indices throw — changing kind changes semantics.
    can_collapse = (
        not normalized.clamp
        and not normalized.exclusive_stop
        and normalized.step is None
        and normalized.start is not None
        and normalized.stop is not None
        and normalized.start == normalized.stop
        and normalized.start in index_origins
    )
    if can_collapse:
        return SelectorIndex(ShkNumber(float(normalized.start)))

    return normalized


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
        case (ShkSelector(parts=parts_a), ShkSelector(parts=parts_b)):
            if len(parts_a) != len(parts_b):
                return False
            for pa, pb in zip(parts_a, parts_b):
                if type(pa) is not type(pb):
                    return False
                if isinstance(pa, SelectorIndex):
                    assert isinstance(pb, SelectorIndex)
                    if not shk_equals(pa.value, pb.value):
                        return False
                else:
                    assert isinstance(pa, SelectorSlice) and isinstance(
                        pb, SelectorSlice
                    )
                    if (pa.start, pa.stop, pa.step, pa.exclusive_stop, pa.clamp) != (
                        pb.start,
                        pb.stop,
                        pb.step,
                        pb.exclusive_stop,
                        pb.clamp,
                    ):
                        return False
            return True
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
