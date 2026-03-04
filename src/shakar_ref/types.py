from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
import math
import pathlib
import re
import threading
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    FrozenSet,
    List,
    Literal,
    NamedTuple,
    Optional,
    Tuple,
    TypeVar,
    Union,
    TYPE_CHECKING,
)
from typing_extensions import Protocol, TypeAlias, TypeGuard

UfcsStyle = Literal["subject", "prepend"]
from .tree import Node

if TYPE_CHECKING:
    from .eval.bind import RebindContext, FanContext
    from .eval.common import DestructFields

# ---------- Value Model (only Sh* => Shk*) ----------


@dataclass
class ShkNil:
    def __repr__(self) -> str:
        return "nil"


@dataclass
class ShkNumber:
    value: float

    def __repr__(self) -> str:
        v = float(self.value)
        # Always show .0 for integer values to match expected output
        return f"{int(v)}.0" if v.is_integer() else str(v)

    def __str__(self) -> str:
        # For stringify: render integral floats without .0
        v = float(self.value)
        return str(int(v)) if v.is_integer() else str(v)


DURATION_NANOS: Dict[str, int] = {
    "nsec": 1,
    "usec": 1_000,
    "msec": 1_000_000,
    "sec": 1_000_000_000,
    "min": 60_000_000_000,
    "hr": 3_600_000_000_000,
    "day": 86_400_000_000_000,
    "wk": 604_800_000_000_000,
}

SIZE_BYTES: Dict[str, int] = {
    "b": 1,
    "kb": 1_000,
    "mb": 1_000_000,
    "gb": 1_000_000_000,
    "tb": 1_000_000_000_000,
    "kib": 1_024,
    "mib": 1_048_576,
    "gib": 1_073_741_824,
    "tib": 1_099_511_627_776,
}


MAX_REPEATING_FRACTION_DIGITS = 20


def _has_terminating_decimal(remainder: int, per_unit: int) -> bool:
    """Check whether remainder/per_unit has a finite decimal expansion."""
    denominator = per_unit // math.gcd(remainder, per_unit)

    while denominator % 2 == 0:
        denominator //= 2
    while denominator % 5 == 0:
        denominator //= 5

    return denominator == 1


def _format_single(abs_val: int, unit: str, per_unit: int) -> str:
    """Format a non-negative value with a single unit."""
    if per_unit == 1:
        return f"{abs_val}{unit}"

    whole, remainder = divmod(abs_val, per_unit)
    if remainder == 0:
        return f"{whole}{unit}"

    # Exact rational formatting using integer long-division.
    # For terminating decimals, emit the full exact expansion.
    # For repeating decimals, cap output to keep rendering bounded.
    frac_digits: List[str] = []
    r = remainder
    if _has_terminating_decimal(remainder, per_unit):
        while r:
            r *= 10
            d, r = divmod(r, per_unit)
            frac_digits.append(str(d))
    else:
        for _ in range(MAX_REPEATING_FRACTION_DIGITS):
            r *= 10
            d, r = divmod(r, per_unit)
            frac_digits.append(str(d))
            if r == 0:
                break

    # Preserve literal shape for very tiny values in large units.
    frac = "".join(frac_digits) or "0"
    return f"{whole}.{frac}{unit}"


def _format_places(
    abs_val: int,
    places: Tuple[str, ...],
    unit_map: Dict[str, int],
) -> str:
    """Decompose a non-negative value across ordered unit places.
    Places are ordered largest-first. Zero-valued places are omitted.
    The last place absorbs the remainder (possibly fractional)."""
    if len(places) == 1:
        single = places[0]
        return _format_single(abs_val, single, unit_map[single])

    # Compound literals require integer components. If the tail would be
    # fractional in the smallest stored place, render as a single-unit value.
    tail = abs_val
    for u in places[:-1]:
        _, tail = divmod(tail, unit_map[u])
    smallest_per = unit_map[places[-1]]
    if tail % smallest_per != 0:
        primary = places[0]
        return _format_single(abs_val, primary, unit_map[primary])

    parts: List[str] = []
    remaining = abs_val
    for i, u in enumerate(places):
        per = unit_map[u]
        if i < len(places) - 1:
            count, remaining = divmod(remaining, per)
            if count:
                parts.append(f"{count}{u}")
        else:
            # Last place absorbs whatever is left
            if remaining or not parts:
                count, _ = divmod(remaining, per)
                parts.append(f"{count}{u}")

    return "".join(parts)


def _format_value(
    raw_value: int,
    places: Optional[Tuple[str, ...]],
    unit_map: Dict[str, int],
    fallback_unit: str,
) -> str:
    """Format a raw value (nanos or bytes) using stored unit places."""
    if not places:
        places = (fallback_unit,)

    sign = "-" if raw_value < 0 else ""
    return sign + _format_places(abs(raw_value), places, unit_map)


def merge_units(
    a: Optional[Tuple[str, ...]],
    b: Optional[Tuple[str, ...]],
    unit_map: Dict[str, int],
) -> Optional[Tuple[str, ...]]:
    """Merge two unit tuples, returning the union sorted largest-first."""
    if a is None:
        return b
    if b is None:
        return a

    combined = set(a) | set(b)
    return tuple(sorted(combined, key=lambda u: unit_map[u], reverse=True))


@dataclass
class ShkDuration:
    nanos: int
    units: Optional[Tuple[str, ...]] = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ShkDuration):
            return False
        return self.nanos == other.nanos

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return _format_value(self.nanos, self.units, DURATION_NANOS, "nsec")


@dataclass
class ShkSize:
    byte_count: int
    units: Optional[Tuple[str, ...]] = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ShkSize):
            return False
        return self.byte_count == other.byte_count

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return _format_value(self.byte_count, self.units, SIZE_BYTES, "b")


@dataclass
class ShkString:
    value: str

    def __repr__(self) -> str:
        return f'"{self.value}"'


@dataclass
class ShkRegex:
    pattern: str
    flags: str
    include_full: bool
    compiled: re.Pattern[str]

    def __repr__(self) -> str:
        suffix = f"/{self.flags}" if self.flags else ""
        return f'r"{self.pattern}"{suffix}'


@dataclass
class ShkBool:
    value: bool

    def __repr__(self) -> str:
        return "true" if self.value else "false"


@dataclass
class ShkArray:
    items: List["ShkValue"]

    def __repr__(self) -> str:
        return "[" + ", ".join(repr(x) for x in self.items) + "]"


@dataclass
class ShkSet:
    items: List["ShkValue"]

    def __post_init__(self) -> None:
        # Keep set invariants centralized at construction sites.
        from .utils import normalize_set_items

        self.items = normalize_set_items(self.items)

    def __repr__(self) -> str:
        return "set{" + ", ".join(repr(x) for x in self.items) + "}"


@dataclass
class ShkFan:
    items: List["ShkValue"]

    def __repr__(self) -> str:
        return "fan { " + ", ".join(repr(x) for x in self.items) + " }"


@dataclass
class ShkCommand:
    segments: List[str]

    def render(self) -> str:
        return "".join(self.segments)

    def __repr__(self) -> str:
        return f"sh<{self.render()}>"


@dataclass
class ShkPath:
    value: str

    def as_path(self) -> pathlib.Path:
        return pathlib.Path(self.value)

    def __repr__(self) -> str:
        return f'p"{self.value}"'

    def __str__(self) -> str:
        return self.value


@dataclass
class ShkEnvVar:
    name: str

    def __repr__(self) -> str:
        return f'env"{self.name}"'

    def __str__(self) -> str:
        import os

        return os.environ.get(self.name, "")


@dataclass(frozen=True)
class ShkType:
    """Runtime type representation for structural matching."""

    name: str
    mapped_type: type

    def __repr__(self) -> str:
        return f"<type {self.name}>"


@dataclass(frozen=True)
class ShkOptional:
    """Optional field wrapper for structural matching schemas."""

    inner: "ShkValue"

    def __repr__(self) -> str:
        return f"Optional({self.inner!r})"


@dataclass(frozen=True)
class ShkRepeatSchema:
    """Marks a repeating element-type schema in array structural matching."""

    inner: "ShkValue"

    def __repr__(self) -> str:
        return f"{self.inner!r}..."


@dataclass(frozen=True)
class ShkUnion:
    """Union type for structural matching - matches any of the alternatives."""

    alternatives: Tuple["ShkValue", ...]

    def __repr__(self) -> str:
        alts = ", ".join(repr(a) for a in self.alternatives)
        return f"Union({alts})"


@dataclass
class ShkObject:
    slots: Dict[str, "ShkValue"]

    def __repr__(self) -> str:
        pairs = []

        for k, v in self.slots.items():
            pairs.append(f"{k}: {repr(v)}")

        return "{ " + ", ".join(pairs) + " }"


@dataclass
class ShkModule(ShkObject):
    name: Optional[str] = None

    def __repr__(self) -> str:
        if self.name:
            return f"<module {self.name}>"
        return "<module>"

    def __setitem__(self, _key: str, _value: "ShkValue") -> None:
        raise ShakarImportError("Modules are immutable")


class CancelToken:
    """Cooperative cancellation token for spawn tasks.

    Supports registering Condition objects to be notified on cancellation,
    eliminating the need for polling in blocking waits.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._conditions: list[threading.Condition] = []

    def register_condition(self, cond: threading.Condition) -> None:
        with self._lock:
            if cond not in self._conditions:
                self._conditions.append(cond)

    def unregister_condition(self, cond: threading.Condition) -> None:
        with self._lock:
            if cond in self._conditions:
                self._conditions.remove(cond)

    def cancel(self) -> None:
        self._event.set()
        with self._lock:
            for cond in self._conditions:
                with cond:
                    cond.notify_all()

    def cancelled(self) -> bool:
        return self._event.is_set()


@dataclass
class RecvResult:
    """Unified result type for channel receive operations."""

    status: str  # "ready", "blocked", "closed", "cancelled"
    value: Optional["ShkValue"] = None

    @property
    def ok(self) -> bool:
        return self.status == "ready"


@dataclass
class _ResultItem:
    """Wrapper for spawn result channel payloads."""

    value: Optional["ShkValue"] = None
    error: Optional[Exception] = None


class ShkChannel:
    """Thread-safe channel for inter-task communication.

    Supports both buffered (capacity > 0) and unbuffered (capacity = 0) modes.
    Unbuffered channels use synchronous rendezvous semantics.
    """

    def __init__(self, capacity: int = 0) -> None:
        if capacity < 0:
            raise ShakarRuntimeError("Channel capacity cannot be negative")

        self.capacity = capacity
        self._buffer: Deque["ShkValue"] = deque()
        self._cond = threading.Condition()
        self._closed = False
        self._select_waiters: set[threading.Event] = set()

        # Unbuffered channel state
        self._unbuffered_value: Optional["ShkValue"] = None
        self._has_unbuffered = False
        self._waiting_receivers = 0

    def __repr__(self) -> str:
        with self._cond:
            if self._closed:
                state = "closed"
            elif self.capacity > 0:
                state = f"len={len(self._buffer)}/{self.capacity}"
            else:
                state = "unbuffered"
            return f"<channel {state}>"

    # ---- Select support ----

    def register_waiter(self, event: threading.Event) -> None:
        with self._cond:
            self._select_waiters.add(event)

    def unregister_waiter(self, event: threading.Event) -> None:
        with self._cond:
            self._select_waiters.discard(event)

    def _notify_waiters(self) -> None:
        self._cond.notify_all()
        for event in list(self._select_waiters):
            event.set()

    # ---- Status ----

    def is_closed(self) -> bool:
        with self._cond:
            return self._closed

    def is_result_channel(self) -> bool:
        return False

    def close(self) -> None:
        with self._cond:
            if self._closed:
                raise ShakarRuntimeError("Channel already closed")
            self._closed = True
            self._notify_waiters()

    # ---- Send ----

    def send(self, value: "ShkValue") -> bool:
        return self.send_with_cancel(value, None)

    def send_with_cancel(
        self, value: "ShkValue", cancel_token: Optional[CancelToken]
    ) -> bool:
        result = self._send(value, block=True, cancel_token=cancel_token)
        return result.ok

    def try_send(self, value: "ShkValue") -> tuple[bool, bool]:
        """Non-blocking send. Returns (sent, was_closed)."""
        result = self._send(value, block=False, cancel_token=None)
        if result.status == "blocked":
            return False, False
        if result.status == "closed":
            return False, True
        return True, False

    def _send(
        self,
        value: "ShkValue",
        *,
        block: bool,
        cancel_token: Optional[CancelToken],
    ) -> RecvResult:
        if self.capacity > 0:
            return self._send_buffered(value, block, cancel_token)
        return self._send_unbuffered(value, block, cancel_token)

    def _send_buffered(
        self,
        value: "ShkValue",
        block: bool,
        cancel_token: Optional[CancelToken],
    ) -> RecvResult:
        with self._cond:
            self._check_cancel(cancel_token)

            if self._closed:
                return RecvResult("closed")

            if len(self._buffer) >= self.capacity:
                if not block:
                    return RecvResult("blocked")
                self._wait_for(
                    lambda: len(self._buffer) < self.capacity or self._closed,
                    cancel_token,
                )
                if self._closed:
                    return RecvResult("closed")

            self._buffer.append(value)
            self._notify_waiters()
            return RecvResult("ready")

    def _send_unbuffered(
        self,
        value: "ShkValue",
        block: bool,
        cancel_token: Optional[CancelToken],
    ) -> RecvResult:
        with self._cond:
            self._check_cancel(cancel_token)

            if self._closed:
                return RecvResult("closed")

            # Non-blocking: only succeed if a receiver is already waiting
            if not block:
                if self._has_unbuffered or self._waiting_receivers == 0:
                    return RecvResult("blocked")
                self._unbuffered_value = value
                self._has_unbuffered = True
                self._notify_waiters()
                return RecvResult("ready")

            # Blocking: wait for slot, place value, wait for consumption
            self._wait_for(
                lambda: not self._has_unbuffered or self._closed, cancel_token
            )
            if self._closed:
                return RecvResult("closed")

            self._unbuffered_value = value
            self._has_unbuffered = True
            self._notify_waiters()

            # Wait for receiver to take it
            self._wait_for(
                lambda: not self._has_unbuffered or self._closed, cancel_token
            )
            if self._closed and self._has_unbuffered:
                self._has_unbuffered = False
                self._unbuffered_value = None
                return RecvResult("closed")

            return RecvResult("ready")

    # ---- Receive ----

    def recv(
        self, *, cancel_token: Optional[CancelToken] = None
    ) -> tuple["ShkValue", bool]:
        """Blocking receive. Returns (value, ok)."""
        result = self._recv(block=True, cancel_token=cancel_token)
        if result.ok:
            return result.value if result.value else ShkNil(), True
        return ShkNil(), False

    def recv_value(self, *, cancel_token: Optional[CancelToken] = None) -> "ShkValue":
        """Blocking receive that returns value directly (nil if closed)."""
        result = self._recv(block=True, cancel_token=cancel_token)
        return result.value if result.value else ShkNil()

    def recv_with_ok(
        self, *, cancel_token: Optional[CancelToken] = None
    ) -> tuple["ShkValue", bool]:
        """Blocking receive. Returns (value, ok)."""
        return self.recv(cancel_token=cancel_token)

    def try_recv_value(self) -> tuple[str, Optional["ShkValue"]]:
        """Non-blocking receive. Returns (status, value)."""
        result = self._recv(block=False, cancel_token=None)
        return result.status, result.value

    def _recv(self, *, block: bool, cancel_token: Optional[CancelToken]) -> RecvResult:
        if self.capacity > 0:
            return self._recv_buffered(block, cancel_token)
        return self._recv_unbuffered(block, cancel_token)

    def _recv_buffered(
        self, block: bool, cancel_token: Optional[CancelToken]
    ) -> RecvResult:
        with self._cond:
            self._check_cancel(cancel_token)

            if self._buffer:
                value = self._buffer.popleft()
                self._notify_waiters()
                return RecvResult("ready", value)

            if self._closed:
                return RecvResult("closed")

            if not block:
                return RecvResult("blocked")

            self._wait_for(lambda: bool(self._buffer) or self._closed, cancel_token)

            if self._buffer:
                value = self._buffer.popleft()
                self._notify_waiters()
                return RecvResult("ready", value)

            return RecvResult("closed")

    def _recv_unbuffered(
        self, block: bool, cancel_token: Optional[CancelToken]
    ) -> RecvResult:
        with self._cond:
            self._check_cancel(cancel_token)

            if self._has_unbuffered:
                value = self._unbuffered_value
                self._unbuffered_value = None
                self._has_unbuffered = False
                self._notify_waiters()
                return RecvResult("ready", value)

            if self._closed:
                return RecvResult("closed")

            if not block:
                return RecvResult("blocked")

            self._waiting_receivers += 1
            try:
                self._wait_for(
                    lambda: self._has_unbuffered or self._closed, cancel_token
                )
            finally:
                self._waiting_receivers -= 1

            if self._has_unbuffered:
                value = self._unbuffered_value
                self._unbuffered_value = None
                self._has_unbuffered = False
                self._notify_waiters()
                return RecvResult("ready", value)

            return RecvResult("closed")

    # ---- Internal helpers ----

    def _check_cancel(self, cancel_token: Optional[CancelToken]) -> None:
        if cancel_token and cancel_token.cancelled():
            raise ShakarCancelledError("Spawn task cancelled")

    def _wait_for(
        self,
        predicate: Callable[[], bool],
        cancel_token: Optional[CancelToken],
    ) -> None:
        """Wait until predicate is true, checking cancellation."""
        if cancel_token:
            cancel_token.register_condition(self._cond)
        try:
            while not predicate():
                # Re-check after registration to close the race window where
                # cancellation fires between the caller's _check_cancel and
                # register_condition above.
                self._check_cancel(cancel_token)
                self._cond.wait()
                self._check_cancel(cancel_token)
        finally:
            if cancel_token:
                cancel_token.unregister_condition(self._cond)

    # ---- Result channel stubs (overridden in subclass) ----

    def send_result(self, value: Optional["ShkValue"]) -> bool:
        raise ShakarRuntimeError("send_result only valid for spawn channels")

    def send_error(self, exc: Exception) -> bool:
        raise ShakarRuntimeError("send_error only valid for spawn channels")


class ShkResultChannel(ShkChannel):
    """One-shot channel for spawn task results.

    Receiving unwraps the result or re-raises any captured exception.
    Closing the channel signals cancellation to the spawned task.
    """

    def __init__(self, cancel_token: CancelToken) -> None:
        super().__init__(capacity=1)
        self._cancel_token = cancel_token
        self._cancelled = False
        self._result_delivered = False

    def __repr__(self) -> str:
        with self._cond:
            if self._cancelled:
                state = "cancelled"
            elif self._result_delivered:
                state = "completed"
            elif self._closed:
                state = "closed"
            else:
                state = "pending"
            return f"<result-channel {state}>"

    def is_result_channel(self) -> bool:
        return True

    def close(self) -> None:
        with self._cond:
            if self._closed:
                if self._result_delivered:
                    return
                raise ShakarRuntimeError("Channel already closed")
            self._closed = True
            self._cancelled = True
            self._cancel_token.cancel()
            self._notify_waiters()

    def send(self, value: "ShkValue") -> bool:
        raise ShakarRuntimeError("Cannot send to spawn result channel")

    def send_with_cancel(
        self, value: "ShkValue", cancel_token: Optional[CancelToken]
    ) -> bool:
        raise ShakarRuntimeError("Cannot send to spawn result channel")

    def try_send(self, value: "ShkValue") -> tuple[bool, bool]:
        raise ShakarRuntimeError("Cannot send to spawn result channel")

    def send_result(self, value: Optional["ShkValue"]) -> bool:
        return self._deliver(_ResultItem(value=value))

    def send_error(self, exc: Exception) -> bool:
        return self._deliver(_ResultItem(error=exc))

    def _deliver(self, item: _ResultItem) -> bool:
        with self._cond:
            if self._closed or self._cancelled:
                return False
            if self._buffer:
                raise ShakarRuntimeError("Spawn result channel already has a value")
            self._buffer.append(item)  # type: ignore[arg-type]
            self._result_delivered = True
            self._closed = True
            self._notify_waiters()
            return True

    def recv_value(self, *, cancel_token: Optional[CancelToken] = None) -> "ShkValue":
        result = self._recv(block=True, cancel_token=cancel_token)
        return self._unwrap(result)

    def recv_with_ok(
        self, *, cancel_token: Optional[CancelToken] = None
    ) -> tuple["ShkValue", bool]:
        result = self._recv(block=True, cancel_token=cancel_token)
        return self._unwrap(result), True

    def try_recv_value(self) -> tuple[str, Optional["ShkValue"]]:
        result = self._recv(block=False, cancel_token=None)
        if result.status == "blocked":
            return "blocked", None
        if result.status == "closed":
            if self._cancelled:
                return "cancelled", None
            return "closed", None
        return "ready", self._unwrap(result)

    def _unwrap(self, result: RecvResult) -> "ShkValue":
        if not result.ok:
            if self._cancelled and self._closed:
                raise ShakarCancelledError("Spawn task cancelled")
            return ShkNil()

        value = result.value
        if value is None:
            raise ShakarRuntimeError("Spawn result missing value")

        if isinstance(value, _ResultItem):
            if value.error:
                raise value.error
            return value.value if value.value else ShkNil()

        return value


@dataclass
class SelectorIndex:
    value: "ShkValue"


@dataclass
class SelectorSlice:
    start: Optional[int]
    stop: Optional[int]
    step: Optional[int]
    clamp: bool
    exclusive_stop: bool = False


SelectorPart = Union[SelectorIndex, SelectorSlice]


@dataclass
class ShkSelector:
    parts: List[SelectorPart]

    def __repr__(self) -> str:
        descr = []

        for part in self.parts:
            if isinstance(part, SelectorIndex):
                descr.append(str(part.value))
            else:
                bits = []
                bits.append("" if part.start is None else str(part.start))
                bits.append(
                    ""
                    if part.stop is None
                    else (
                        "<" + str(part.stop) if part.exclusive_stop else str(part.stop)
                    )
                )

                if part.step is not None:
                    bits.append(str(part.step))

                while len(bits) < 3:
                    bits.append("")
                descr.append(":".join(bits[:3]).rstrip(":"))

        return "selector{" + ", ".join(descr) + "}"


@dataclass
class ShkDecorator:
    params: Optional[List[str]]
    body: Node
    frame: "Frame"
    vararg_indices: Optional[List[int]] = None
    param_defaults: Optional[List[Optional[Node]]] = None


@dataclass
class DecoratorConfigured:
    decorator: ShkDecorator
    args: List["ShkValue"]


@dataclass
class ShkFn:
    params: Optional[List[str]]  # None for subject-only amp-lambda
    body: Node  # AST node
    frame: "Frame"  # Closure frame
    decorators: Optional[Tuple[DecoratorConfigured, ...]] = None
    kind: str = "fn"
    name: Optional[str] = None
    return_contract: Optional[Node] = None  # AST node for return type contract
    vararg_indices: Optional[List[int]] = None
    param_defaults: Optional[List[Optional[Node]]] = None
    destruct_fields: Optional[List[Optional["DestructFields"]]] = None

    def __repr__(self) -> str:
        body_label = getattr(self.body, "data", type(self.body).__name__)
        label = "amp-fn" if self.kind == "amp" else self.kind

        if self.params is None:
            param_desc = "subject"
        else:
            param_desc = ", ".join(self.params) if self.params else "nullary"

        return f"<{label} params={param_desc} body={body_label}>"


@dataclass
class BoundMethod:
    fn: ShkFn
    subject: "ShkValue"
    name: Optional[str] = None


@dataclass
class BoundCallable:
    target: "ShkValue"
    subject: "ShkValue"
    style: UfcsStyle
    name: Optional[str] = None


@dataclass
class BuiltinMethod:
    name: str
    subject: "ShkValue"


@dataclass
class Descriptor:
    getter: Optional[ShkFn] = None
    setter: Optional[ShkFn] = None


@dataclass
class DecoratorContinuation:
    fn: ShkFn
    decorators: Tuple[DecoratorConfigured, ...]
    index: int
    subject: Optional["ShkValue"]
    caller_frame: "Frame"

    def invoke(self, args_value: ShkValue) -> "ShkValue":
        from . import runtime

        args = runtime._coerce_decorator_args(args_value)
        return runtime._run_decorator_chain(
            self.fn, self.decorators, self.index, args, self.subject, self.caller_frame
        )


@dataclass
class DeferEntry:
    """Scheduled deferred action plus optional metadata for dependency ordering."""

    action: "DeferredAction"


@dataclass
class DeferredAction:
    """Deferred call/block payload with schedule-time execution context."""

    kind: Literal["call", "block"]
    payload_node: Node
    origin_frame: "Frame"
    saved_dot: "DotValue"
    saved_source: Optional[str]
    saved_source_path: Optional[str]
    deps: List[str] = field(default_factory=list)
    label: Optional[str] = None


StdlibFn: TypeAlias = Callable[
    ["Frame", Optional["ShkValue"], List["ShkValue"], Optional[Dict[str, "ShkValue"]]],
    "ShkValue",
]


@dataclass(frozen=True)
class StdlibFunction:
    fn: StdlibFn
    arity: Optional[int] = None
    accepts_named: bool = False
    name: Optional[str] = None


OnceSiteStatus = Literal["uninitialized", "evaluating", "initialized", "failed"]


@dataclass
class OnceSiteState:
    """Explicit state machine for a once expression site.

    Status transitions:
        uninitialized => evaluating => initialized  (success)
        uninitialized => evaluating => failed        (error cached)
    Re-entry while evaluating raises circular initialization error.
    Both 'initialized' and 'failed' are terminal — subsequent access
    replays the cached value or error without re-evaluation."""

    status: OnceSiteStatus = "uninitialized"
    value: Optional["ShkValue"] = None
    error: Optional["ShakarRuntimeError"] = None
    is_block: bool = False
    # Names introduced by a block-form once body, in declaration order, plus
    # the frame where the block actually ran. On rematerialization (e.g.
    # static once in a new call), live values are read from
    # block_source_frame so closures and outer bindings stay coherent.
    block_binding_names: Optional[List[str]] = None
    block_source_frame: Optional["Frame"] = None
    # Re-entrant so recursive self-reads can report circular initialization.
    lock: threading.RLock = field(default_factory=threading.RLock)


@dataclass
class OnceBinding:
    """Metadata for a once expression: the unevaluated body, its defining
    frame, and whether it uses static (global) or lazy (per-invocation) caching."""

    node_id: int
    body: Node
    def_frame: "Frame"
    fn_frame: "Frame"
    is_static: bool
    is_block: bool = False


_STATIC_ONCE_CELLS: Dict[int, OnceSiteState] = {}
_STATIC_ONCE_CELLS_LOCK = threading.Lock()


ShkValue: TypeAlias = Union[
    ShkNil,
    ShkNumber,
    ShkDuration,
    ShkSize,
    ShkString,
    ShkRegex,
    ShkBool,
    ShkArray,
    ShkSet,
    ShkFan,
    ShkObject,
    ShkModule,
    ShkChannel,
    ShkFn,
    ShkDecorator,
    DecoratorConfigured,
    DecoratorContinuation,
    Descriptor,
    ShkSelector,
    ShkCommand,
    ShkPath,
    ShkEnvVar,
    BoundMethod,
    BoundCallable,
    BuiltinMethod,
    StdlibFunction,
    ShkType,
    ShkOptional,
    ShkRepeatSchema,
    ShkUnion,
]

# Internal evaluator result type - includes assignment contexts
EvalResult: TypeAlias = Union[ShkValue, "RebindContext", "FanContext"]

DotValue: TypeAlias = Optional[ShkValue]


@dataclass
class LazyOnceThunk:
    """Lazy once-walrus binding: body is deferred until the name is first read."""

    resolve: Callable[[], ShkValue]


class MissingBindingLookup(NamedTuple):
    """Binding not found in the current frame."""

    pass


class ValueBindingLookup(NamedTuple):
    """Binding resolved to a concrete value."""

    value: ShkValue


class LazyOnceLookup(NamedTuple):
    """Binding is a lazy-once thunk that hasn't been evaluated yet."""

    thunk: LazyOnceThunk


LocalBindingLookup: TypeAlias = Union[
    MissingBindingLookup,
    ValueBindingLookup,
    LazyOnceLookup,
]

_MISSING_BINDING_LOOKUP = MissingBindingLookup()


class CallSite(NamedTuple):
    name: str
    line: int
    column: int
    path: Optional[str]


@dataclass(frozen=True)
class SourceSpanInfo:
    line: int
    column: int
    end_line: Optional[int]
    end_column: Optional[int]


@dataclass
class ErrorContext:
    span: Optional[SourceSpanInfo] = None
    call_stack: Optional[List[CallSite]] = None
    py_trace: Optional[Any] = None
    enriched: bool = False


def retain_value(value: "ShkValue") -> "ShkValue":
    """Ownership hook: retain one reference when storing a value."""
    return value


def release_value(value: "ShkValue") -> None:
    """Ownership hook: release one owned reference."""
    _ = value


def set_mapping_item(
    mapping: Dict[str, "ShkValue"], key: str, value: "ShkValue"
) -> None:
    """Replace-or-insert mapping entry with explicit retain/release semantics."""
    retained = retain_value(value)
    if key in mapping:
        old = mapping[key]
        mapping[key] = retained
        release_value(old)
        return
    mapping[key] = retained


def delete_mapping_item(mapping: Dict[str, "ShkValue"], key: str) -> None:
    """Delete mapping entry and release owned storage when present."""
    if key not in mapping:
        return
    old = mapping.pop(key)
    release_value(old)


def set_sequence_item(items: List["ShkValue"], index: int, value: "ShkValue") -> None:
    """Replace sequence element with explicit retain/release semantics."""
    old = items[index]
    items[index] = retain_value(value)
    release_value(old)


def replace_sequence(items: List["ShkValue"], values: List["ShkValue"]) -> None:
    """Replace sequence contents while releasing previous owned entries."""
    for old in items:
        release_value(old)
    items[:] = [retain_value(value) for value in values]


@dataclass
class FrameLexicalState:
    vars: Dict[str, ShkValue] = field(default_factory=dict)
    builtins: Dict[str, ShkValue] = field(default_factory=dict)
    lazy_once: Dict[str, LazyOnceThunk] = field(default_factory=dict)
    lazy_once_site_ids: Dict[str, int] = field(default_factory=dict)
    once_cells: Dict[int, OnceSiteState] = field(default_factory=dict)
    once_cells_lock: threading.Lock = field(default_factory=threading.Lock)
    block_aliases: Dict[str, "Frame"] = field(default_factory=dict)
    let_scopes: List[Dict[str, ShkValue]] = field(default_factory=list)
    captured_let_scopes: List[Dict[str, ShkValue]] = field(default_factory=list)
    frozen_scope_names: Optional[FrozenSet[str]] = None


@dataclass
class FrameExecState:
    dot: DotValue = None
    emit_target: Optional[ShkValue] = None
    pending_anchor_override: Optional[ShkValue] = None
    source: Optional[str] = None
    source_path: Optional[str] = None


@dataclass
class FrameControlState:
    defer_stack: List[List[DeferEntry]] = field(default_factory=list)
    is_function_frame: bool = False
    active_error: Optional["ShakarRuntimeError"] = None
    call_stack: List[CallSite] = field(default_factory=list)
    hoisted_names: Optional[set[str]] = None
    cancel_token: Optional[CancelToken] = None


@dataclass
class TempBindingRecord:
    scope: Dict[str, ShkValue]
    name: str
    existed: bool
    previous_value: Optional[ShkValue] = None


class Frame:
    def __init__(
        self,
        parent: Optional["Frame"] = None,
        dot: DotValue = None,
        emit_target: Optional["ShkValue"] = None,
        cancel_token: Optional[CancelToken] = None,
        source: Optional[str] = None,
        source_path: Optional[str] = None,
        call_stack: Optional[List[CallSite]] = None,
    ):
        self.parent = parent
        self._lexical = FrameLexicalState()
        self._exec = FrameExecState(dot=dot, emit_target=emit_target)
        self._control = FrameControlState()

        if cancel_token:
            self._control.cancel_token = cancel_token
        elif parent:
            self._control.cancel_token = parent.cancel_token
        else:
            self._control.cancel_token = None

        if parent is None:
            for name, std in Builtins.stdlib_functions.items():
                self._lexical.builtins[name] = retain_value(std)
            for name, typ in Builtins.type_constants.items():
                self._lexical.builtins[name] = retain_value(typ)

        if source is not None:
            self._exec.source = source
        elif parent:
            self._exec.source = parent.source
        else:
            self._exec.source = None

        if source_path is not None:
            self._exec.source_path = source_path
        elif parent:
            self._exec.source_path = parent.source_path
        else:
            self._exec.source_path = None

        # NOTE: call_stack is shared by reference across synchronous child
        # frames so that push/pop in call_shkfn stays balanced.  Spawned
        # threads MUST snapshot (list(frame.call_stack)) to avoid races.
        if call_stack is not None:
            self._control.call_stack = call_stack
        elif parent:
            self._control.call_stack = parent.call_stack
        else:
            self._control.call_stack = []

    @property
    def vars(self) -> Dict[str, ShkValue]:
        return self._lexical.vars

    @property
    def builtins(self) -> Dict[str, ShkValue]:
        return self._lexical.builtins

    @property
    def lazy_once(self) -> Dict[str, LazyOnceThunk]:
        return self._lexical.lazy_once

    @property
    def lazy_once_site_ids(self) -> Dict[str, int]:
        return self._lexical.lazy_once_site_ids

    @property
    def once_cells(self) -> Dict[int, OnceSiteState]:
        return self._lexical.once_cells

    @property
    def once_cells_lock(self) -> threading.Lock:
        return self._lexical.once_cells_lock

    @property
    def _block_aliases(self) -> Dict[str, "Frame"]:
        return self._lexical.block_aliases

    @property
    def _let_scopes(self) -> List[Dict[str, ShkValue]]:
        return self._lexical.let_scopes

    @property
    def _captured_let_scopes(self) -> List[Dict[str, ShkValue]]:
        return self._lexical.captured_let_scopes

    @property
    def dot(self) -> DotValue:
        return self._exec.dot

    @dot.setter
    def dot(self, value: DotValue) -> None:
        self._exec.dot = value

    @property
    def emit_target(self) -> Optional[ShkValue]:
        return self._exec.emit_target

    @emit_target.setter
    def emit_target(self, value: Optional[ShkValue]) -> None:
        self._exec.emit_target = value

    @property
    def pending_anchor_override(self) -> Optional[ShkValue]:
        return self._exec.pending_anchor_override

    @pending_anchor_override.setter
    def pending_anchor_override(self, value: Optional[ShkValue]) -> None:
        self._exec.pending_anchor_override = value

    @property
    def source(self) -> Optional[str]:
        return self._exec.source

    @source.setter
    def source(self, value: Optional[str]) -> None:
        self._exec.source = value

    @property
    def source_path(self) -> Optional[str]:
        return self._exec.source_path

    @source_path.setter
    def source_path(self, value: Optional[str]) -> None:
        self._exec.source_path = value

    @property
    def _defer_stack(self) -> List[List[DeferEntry]]:
        return self._control.defer_stack

    @property
    def _is_function_frame(self) -> bool:
        return self._control.is_function_frame

    @_is_function_frame.setter
    def _is_function_frame(self, value: bool) -> None:
        self._control.is_function_frame = value

    @property
    def _active_error(self) -> Optional["ShakarRuntimeError"]:
        return self._control.active_error

    @_active_error.setter
    def _active_error(self, value: Optional["ShakarRuntimeError"]) -> None:
        self._control.active_error = value

    @property
    def call_stack(self) -> List[CallSite]:
        return self._control.call_stack

    @call_stack.setter
    def call_stack(self, value: List[CallSite]) -> None:
        self._control.call_stack = value

    @property
    def cancel_token(self) -> Optional[CancelToken]:
        return self._control.cancel_token

    @cancel_token.setter
    def cancel_token(self, value: Optional[CancelToken]) -> None:
        self._control.cancel_token = value

    @property
    def hoisted_names(self) -> Optional[set[str]]:
        return self._control.hoisted_names

    @hoisted_names.setter
    def hoisted_names(self, value: Optional[set[str]]) -> None:
        self._control.hoisted_names = value

    @property
    def frozen_scope_names(self) -> Optional[FrozenSet[str]]:
        return self._lexical.frozen_scope_names

    @frozen_scope_names.setter
    def frozen_scope_names(self, value: Optional[FrozenSet[str]]) -> None:
        self._lexical.frozen_scope_names = value

    def _store_scope_value(
        self,
        scope: Dict[str, ShkValue],
        name: str,
        val: ShkValue,
    ) -> None:
        retained = retain_value(val)
        if name in scope:
            old = scope[name]
            scope[name] = retained
            release_value(old)
            return
        scope[name] = retained

    def _store_var(self, name: str, val: ShkValue) -> None:
        self._store_scope_value(self._lexical.vars, name, val)

    def define(self, name: str, val: ShkValue) -> None:
        # define() is used by import/mixin-style flows that may intentionally
        # overwrite existing bindings; clear lazy once placeholders and
        # rematerialized aliases so the concrete value wins on reads.
        self._lexical.lazy_once.pop(name, None)
        self._lexical.block_aliases.pop(name, None)
        self._store_var(name, val)

    def ensure_name_available_in_current_scope(self, name: str) -> None:
        """Enforce duplicate-definition rules shared by walrus bindings.

        Invariant: vars-level definitions cannot collide with any visible let
        name in this frame, nor with existing vars or lazy once bindings.
        """
        if self.has_let_name(name):
            raise ShakarRuntimeError(f"Name '{name}' already defined in a let scope")

        if name in self._lexical.vars:
            raise ShakarRuntimeError(f"Name '{name}' already defined in this scope")

        if name in self._lexical.block_aliases:
            raise ShakarRuntimeError(f"Name '{name}' already defined in this scope")

        if name in self._lexical.lazy_once:
            raise ShakarRuntimeError(f"Name '{name}' already defined in this scope")

    def define_new_ident(self, name: str, val: ShkValue) -> None:
        """Define a new vars-level identifier after duplicate checks."""
        self.ensure_name_available_in_current_scope(name)
        self._store_var(name, val)

    def push_let_scope(self) -> None:
        self._lexical.let_scopes.append({})

    def pop_let_scope(self) -> Dict[str, ShkValue]:
        if not self._lexical.let_scopes:
            return {}

        return self._lexical.let_scopes.pop()

    def capture_let_scopes(self, scopes: List[Dict[str, ShkValue]]) -> None:
        self._lexical.captured_let_scopes = list(scopes)

    def all_let_scopes(self) -> List[Dict[str, ShkValue]]:
        return self._lexical.captured_let_scopes + self._lexical.let_scopes

    def has_let_name(self, name: str) -> bool:
        return self._find_let_scope(name) is not None

    def has_let_in_current_scope(self, name: str) -> bool:
        return bool(self._lexical.let_scopes and name in self._lexical.let_scopes[-1])

    def name_exists(self, name: str) -> bool:
        current: Optional["Frame"] = self
        while current:
            if (
                current.has_let_name(name)
                or name in current._lexical.vars
                or name in current._lexical.block_aliases
                or name in current._lexical.lazy_once
            ):
                return True
            current = current.parent
        return False

    def define_let(self, name: str, val: ShkValue) -> None:
        if not self._lexical.let_scopes:
            self._lexical.let_scopes.append({})
        target_scope = self._lexical.let_scopes[-1]
        if name in target_scope:
            raise ShakarRuntimeError(f"Name '{name}' already defined in this scope")
        self._store_scope_value(target_scope, name, val)

    def _find_let_scope(self, name: str) -> Optional[Dict[str, ShkValue]]:
        # Search live scopes (innermost first), then captured scopes.
        for scope in reversed(self._lexical.let_scopes):
            if name in scope:
                return scope
        for scope in reversed(self._lexical.captured_let_scopes):
            if name in scope:
                return scope

        return None

    def _lookup_scoped_binding(self, name: str) -> LocalBindingLookup:
        """Lookup nearest let-scoped binding in this frame."""
        for scope in reversed(self._lexical.let_scopes):
            if name in scope:
                return ValueBindingLookup(scope[name])

        for scope in reversed(self._lexical.captured_let_scopes):
            if name in scope:
                return ValueBindingLookup(scope[name])

        return _MISSING_BINDING_LOOKUP

    def _lookup_local_binding(self, name: str) -> LocalBindingLookup:
        """Lookup nearest local binding in this frame with explicit precedence."""
        scoped = self._lookup_scoped_binding(name)
        if not isinstance(scoped, MissingBindingLookup):
            return scoped

        if name in self._lexical.vars:
            return ValueBindingLookup(self._lexical.vars[name])
        if name in self._lexical.block_aliases:
            return self._lexical.block_aliases[name]._lookup_local_binding(name)
        if name in self._lexical.lazy_once:
            return LazyOnceLookup(self._lexical.lazy_once[name])
        if name in self._lexical.builtins:
            return ValueBindingLookup(self._lexical.builtins[name])

        return _MISSING_BINDING_LOOKUP

    def _materialize_lazy_once(self, name: str, thunk: LazyOnceThunk) -> ShkValue:
        value = thunk.resolve()
        self._lexical.lazy_once.pop(name, None)
        self._store_var(name, value)
        return value

    def _raw_get(self, name: str) -> ShkValue:
        """Lookup a visible binding, resolving lazy once bindings as needed."""
        current: Optional["Frame"] = self
        while current:
            scope = current._find_let_scope(name)
            if scope:
                return scope[name]

            if name in current._lexical.vars:
                return current._lexical.vars[name]

            alias_frame = current._lexical.block_aliases.get(name)
            if alias_frame:
                return alias_frame._raw_get(name)

            thunk = current._lexical.lazy_once.get(name)
            if thunk:
                return current._materialize_lazy_once(name, thunk)

            if name in current._lexical.builtins:
                return current._lexical.builtins[name]

            current = current.parent

        raise ShakarRuntimeError(f"Name '{name}' not found")

    def get(self, name: str) -> ShkValue:
        current: Optional["Frame"] = self
        while current:
            local = current._lookup_local_binding(name)
            if isinstance(local, ValueBindingLookup):
                return local.value

            if isinstance(local, LazyOnceLookup):
                return current._materialize_lazy_once(name, local.thunk)

            current = current.parent

        raise ShakarRuntimeError(f"Name '{name}' not found")

    def set(self, name: str, val: ShkValue) -> None:
        current: Optional["Frame"] = self
        while current:
            scoped = current._lookup_scoped_binding(name)
            if isinstance(scoped, ValueBindingLookup):
                target_scope = current._find_let_scope(name)
                if not target_scope:
                    raise ShakarRuntimeError("let scope lookup mismatch")
                current._store_scope_value(target_scope, name, val)
                return

            if name in current._lexical.block_aliases:
                current._lexical.block_aliases[name].set(name, val)
                return

            if name in current._lexical.lazy_once:
                # Assignment before first read should materialize a normal vars
                # binding and suppress deferred once resolution.
                current._lexical.lazy_once.pop(name, None)
                current._store_var(name, val)
                return

            if name in current._lexical.vars:
                current._store_var(name, val)
                return

            if name in current._lexical.builtins:
                raise ShakarRuntimeError(
                    f"Cannot assign to builtin '{name}'; use := to shadow it"
                )

            current = current.parent

        raise ShakarRuntimeError(f"Name '{name}' not found")

    def push_defer_frame(self) -> None:
        self._control.defer_stack.append([])

    def pop_defer_frame(self) -> List[DeferEntry]:
        if not self._control.defer_stack:
            return []

        return self._control.defer_stack.pop()

    def current_defer_frame(self) -> List[DeferEntry]:
        if not self._control.defer_stack:
            raise ShakarRuntimeError("Cannot use defer outside of a block")

        return self._control.defer_stack[-1]

    def has_defer_frame(self) -> bool:
        return bool(self._control.defer_stack)

    def mark_function_frame(self) -> None:
        self._control.is_function_frame = True

    def is_function_frame(self) -> bool:
        return self._control.is_function_frame

    def get_function_frame(self) -> "Frame":
        """Walk up to the nearest function frame; root falls back to self."""
        current = self

        while current.parent and not current._control.is_function_frame:
            current = current.parent

        return current

    def get_emit_target(self) -> ShkValue:
        current: Optional["Frame"] = self
        while current:
            if current._exec.emit_target:
                return current._exec.emit_target
            current = current.parent
        raise ShakarRuntimeError("No emit target available for '>'")

    def local_vars(self) -> Dict[str, ShkValue]:
        return self._lexical.vars

    def local_var_names(self) -> List[str]:
        return list(self._lexical.vars.keys())

    def local_var_items(self) -> List[Tuple[str, ShkValue]]:
        return list(self._lexical.vars.items())

    def has_local_var(self, name: str) -> bool:
        return name in self._lexical.vars

    def has_local_lazy_once(self, name: str) -> bool:
        return name in self._lexical.lazy_once

    def local_lazy_once_site_id(self, name: str) -> Optional[int]:
        return self._lexical.lazy_once_site_ids.get(name)

    def register_lazy_once(self, name: str, thunk: LazyOnceThunk) -> None:
        self._lexical.lazy_once[name] = thunk

    def register_lazy_once_site(self, name: str, site_id: int) -> None:
        self._lexical.lazy_once_site_ids[name] = site_id

    def local_lazy_once_names(self) -> List[str]:
        return list(self._lexical.lazy_once.keys())

    def get_block_alias(self, name: str) -> Optional["Frame"]:
        return self._lexical.block_aliases.get(name)

    def set_block_alias(self, name: str, source: "Frame") -> None:
        self._lexical.block_aliases[name] = source

    def block_alias_names(self) -> List[str]:
        return list(self._lexical.block_aliases.keys())

    def get_or_create_once_site(self, site_id: int, is_block: bool) -> OnceSiteState:
        site = self._lexical.once_cells.get(site_id)
        if site:
            return site

        with self._lexical.once_cells_lock:
            site = self._lexical.once_cells.get(site_id)
            if site is None:
                site = OnceSiteState(is_block=is_block)
                self._lexical.once_cells[site_id] = site

        return site

    def push_call_site(self, site: CallSite) -> None:
        self._control.call_stack.append(site)

    def pop_call_site(self) -> None:
        self._control.call_stack.pop()

    def call_stack_snapshot(self) -> List[CallSite]:
        return list(self._control.call_stack)

    def set_active_error(self, error: Optional["ShakarRuntimeError"]) -> None:
        self._control.active_error = error

    def active_error(self) -> Optional["ShakarRuntimeError"]:
        return self._control.active_error

    def apply_temporary_bindings(
        self, bindings: Dict[str, ShkValue]
    ) -> List[TempBindingRecord]:
        records: List[TempBindingRecord] = []
        scope = (
            self._lexical.let_scopes[-1]
            if self._lexical.let_scopes
            else self._lexical.vars
        )

        for name, value in bindings.items():
            if name in scope:
                previous = retain_value(scope[name])
                records.append(
                    TempBindingRecord(
                        scope=scope,
                        name=name,
                        existed=True,
                        previous_value=previous,
                    )
                )
            else:
                records.append(TempBindingRecord(scope=scope, name=name, existed=False))

            self._store_scope_value(scope, name, value)

        return records

    def restore_temporary_bindings(self, records: List[TempBindingRecord]) -> None:
        for record in reversed(records):
            if record.existed:
                previous = record.previous_value
                if previous is None:
                    raise ShakarRuntimeError("temporary binding restore mismatch")
                current = record.scope.get(record.name)
                if current is not None:
                    release_value(current)
                record.scope[record.name] = previous
                continue

            delete_mapping_item(record.scope, record.name)


# ---------- Exceptions (keep Shakar* canonical) ----------


class ShakarRuntimeError(Exception):
    shk_type: Optional[str]
    shk_data: Optional[ShkValue]
    shk_payload: Optional[ShkValue]
    context: ErrorContext

    def __init__(self, message: str):
        super().__init__(message)
        self.shk_type = None
        self.shk_data = None
        self.shk_payload = None
        self.context = ErrorContext()

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        msg = super().__str__()
        meta = self.context.span
        if meta is None:
            rendered = msg
        else:
            rendered = f"{msg} (line {meta.line}, col {meta.column})"

        stack = self.context.call_stack
        if not stack:
            return rendered

        lines = [rendered]
        for site in reversed(stack):
            location = f"line {site.line}, col {site.column}"
            if site.path:
                location = f"{location} ({site.path})"
            lines.append(f"  in {site.name} called at {location}")
        return "\n".join(lines)


class ShakarImportError(ShakarRuntimeError):
    pass


class ShakarTypeError(ShakarRuntimeError):
    pass


class ShakarFieldNotFoundError(ShakarTypeError):
    """Raised when a field/property is not found on a value."""

    pass


class ShakarArityError(ShakarRuntimeError):
    pass


class ShakarKeyError(ShakarRuntimeError):
    def __init__(self, key: str):
        super().__init__(f"Key '{key}' not found")
        self.key = key


class ShakarIndexError(ShakarRuntimeError):
    def __init__(self, message: str = "Index out of bounds"):
        super().__init__(message)


class ShakarMethodNotFound(ShakarRuntimeError):
    def __init__(self, recv: ShkValue, name: str):
        super().__init__(f"{type(recv).__name__} has no builtin method '{name}'")
        self.receiver = recv
        self.name = name


class ShakarMatchError(ShakarRuntimeError):
    pass


class CommandError(ShakarRuntimeError):
    def __init__(self, cmd: str, code: int, stdout: str, stderr: str):
        super().__init__(f"Command failed with exit code {code}: {cmd}")
        self.cmd = cmd
        self.code = code
        self.stdout = stdout
        self.stderr = stderr
        self.shk_payload = ShkObject(
            {
                "cmd": ShkString(cmd),
                "code": ShkNumber(float(code)),
                "stdout": ShkString(stdout),
                "stderr": ShkString(stderr),
            }
        )
        self.shk_type = "CommandError"


class ShakarAssertionError(ShakarRuntimeError):
    pass


class ShakarCancelledError(ShakarRuntimeError):
    pass


class ShakarChannelClosedEmpty(ShakarRuntimeError):
    pass


class ShakarAllChannelsClosed(ShakarRuntimeError):
    pass


class ShakarReturnSignal(Exception):
    """Internal control-flow exception used to implement `return`."""

    def __init__(self, value: ShkValue):
        self.value = value


class ShakarBreakSignal(Exception):
    """Internal control flow for `break`."""


class ShakarContinueSignal(Exception):
    """Internal control flow for `continue`."""


_SHK_VALUE_TYPES: Tuple[type, ...] = (
    ShkNil,
    ShkNumber,
    ShkDuration,
    ShkSize,
    ShkString,
    ShkRegex,
    ShkBool,
    ShkArray,
    ShkSet,
    ShkFan,
    ShkObject,
    ShkModule,
    ShkChannel,
    ShkFn,
    ShkDecorator,
    DecoratorConfigured,
    DecoratorContinuation,
    Descriptor,
    ShkSelector,
    ShkCommand,
    ShkPath,
    ShkEnvVar,
    BoundMethod,
    BoundCallable,
    BuiltinMethod,
    StdlibFunction,
    ShkType,
    ShkOptional,
    ShkRepeatSchema,
    ShkUnion,
)


def is_shk_value(value: ShkValue | Node) -> TypeGuard[ShkValue]:
    return isinstance(value, _SHK_VALUE_TYPES)


def _ensure_shk_value(value: ShkValue | Node) -> ShkValue:
    if value is None:
        return ShkNil()
    if is_shk_value(value):
        return value
    raise ShakarTypeError(f"Unexpected value type {type(value).__name__}")


R_contra = TypeVar("R_contra", bound="ShkValue", contravariant=True)


class Method(Protocol[R_contra]):
    def __call__(
        self, frame: "Frame", recv: R_contra, args: List["ShkValue"]
    ) -> "ShkValue": ...


MethodRegistry = Dict[str, Method[ShkValue]]


class Builtins:
    array_methods: MethodRegistry = {}
    set_methods: MethodRegistry = {}
    string_methods: MethodRegistry = {}
    regex_methods: MethodRegistry = {}
    object_methods: MethodRegistry = {}
    command_methods: MethodRegistry = {}
    path_methods: MethodRegistry = {}
    envvar_methods: MethodRegistry = {}
    channel_methods: MethodRegistry = {}
    stdlib_functions: Dict[str, StdlibFunction] = {}
    type_constants: Dict[str, "ShkType"] = {}
