from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
import pathlib
import re
import threading
import types as _pytypes
from typing import (
    Callable,
    Deque,
    Dict,
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

# ---------- Value Model (only Sh* -> Shk*) ----------


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


@dataclass
class ShkDuration:
    nanos: int
    display: Optional[str] = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ShkDuration):
            return False
        return self.nanos == other.nanos

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        text = self.display if self.display is not None else f"{self.nanos}nsec"
        if self.nanos < 0 and not text.startswith("-"):
            return "-" + text
        return text


@dataclass
class ShkSize:
    byte_count: int
    display: Optional[str] = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ShkSize):
            return False
        return self.byte_count == other.byte_count

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        text = self.display if self.display is not None else f"{self.byte_count}b"
        if self.byte_count < 0 and not text.startswith("-"):
            return "-" + text
        return text


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
            return result.value if result.value is not None else ShkNil(), True
        return ShkNil(), False

    def recv_value(self, *, cancel_token: Optional[CancelToken] = None) -> "ShkValue":
        """Blocking receive that returns value directly (nil if closed)."""
        result = self._recv(block=True, cancel_token=cancel_token)
        return result.value if result.value is not None else ShkNil()

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
        if cancel_token is not None and cancel_token.cancelled():
            raise ShakarCancelledError("Spawn task cancelled")

    def _wait_for(
        self,
        predicate: Callable[[], bool],
        cancel_token: Optional[CancelToken],
    ) -> None:
        """Wait until predicate is true, checking cancellation."""
        if cancel_token is not None:
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
            if cancel_token is not None:
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
            if value.error is not None:
                raise value.error
            return value.value if value.value is not None else ShkNil()

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
    """Scheduled defer thunk plus optional metadata for dependency ordering."""

    thunk: Callable[[], None]
    label: Optional[str] = None
    deps: List[str] = field(default_factory=list)


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


ShkValue: TypeAlias = Union[
    ShkNil,
    ShkNumber,
    ShkDuration,
    ShkSize,
    ShkString,
    ShkRegex,
    ShkBool,
    ShkArray,
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
    ShkUnion,
]

# Internal evaluator result type - includes assignment contexts
EvalResult: TypeAlias = Union[ShkValue, "RebindContext", "FanContext"]

DotValue: TypeAlias = Optional[ShkValue]


class CallSite(NamedTuple):
    name: str
    line: int
    column: int
    path: Optional[str]


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
        self.vars: Dict[str, ShkValue] = {}
        self._let_scopes: List[Dict[str, ShkValue]] = []
        self._captured_let_scopes: List[Dict[str, ShkValue]] = []
        self.dot: DotValue = dot
        self.emit_target: Optional[ShkValue] = emit_target
        self._defer_stack: List[List[DeferEntry]] = []
        self._is_function_frame = False
        self._active_error: Optional[ShakarRuntimeError] = None
        self.pending_anchor_override: Optional[ShkValue] = None
        self.call_stack: List[CallSite]
        self.cancel_token: Optional[CancelToken]
        if cancel_token is not None:
            self.cancel_token = cancel_token
        elif parent is not None:
            self.cancel_token = parent.cancel_token
        else:
            self.cancel_token = None
        self.source: Optional[str]
        self.source_path: Optional[str]

        if parent is None:
            for name, std in Builtins.stdlib_functions.items():
                self.vars[name] = std
            for name, typ in Builtins.type_constants.items():
                self.vars[name] = typ

        if source is not None:
            self.source = source
        elif parent is not None and hasattr(parent, "source"):
            self.source = parent.source
        else:
            self.source = None

        if source_path is not None:
            self.source_path = source_path
        elif parent is not None and hasattr(parent, "source_path"):
            self.source_path = parent.source_path
        else:
            self.source_path = None

        # NOTE: call_stack is shared by reference across synchronous child
        # frames so that push/pop in call_shkfn stays balanced.  Spawned
        # threads MUST snapshot (list(frame.call_stack)) to avoid races.
        if call_stack is not None:
            self.call_stack = call_stack
        elif parent is not None and hasattr(parent, "call_stack"):
            self.call_stack = parent.call_stack
        else:
            self.call_stack = []

    def define(self, name: str, val: ShkValue) -> None:
        self.vars[name] = val

    def push_let_scope(self) -> None:
        self._let_scopes.append({})

    def pop_let_scope(self) -> Dict[str, ShkValue]:
        if not self._let_scopes:
            return {}
        return self._let_scopes.pop()

    def capture_let_scopes(self, scopes: List[Dict[str, ShkValue]]) -> None:
        self._captured_let_scopes = list(scopes)

    def all_let_scopes(self) -> List[Dict[str, ShkValue]]:
        return self._captured_let_scopes + self._let_scopes

    def has_let_name(self, name: str) -> bool:
        return self._find_let_scope(name) is not None

    def has_let_in_current_scope(self, name: str) -> bool:
        return bool(self._let_scopes and name in self._let_scopes[-1])

    def name_exists(self, name: str) -> bool:
        if self.has_let_name(name) or name in self.vars:
            return True
        if self.parent is not None:
            return self.parent.name_exists(name)
        return False

    def define_let(self, name: str, val: ShkValue) -> None:
        if not self._let_scopes:
            self._let_scopes.append({})
        if name in self._let_scopes[-1]:
            raise ShakarRuntimeError(f"Name '{name}' already defined in this scope")
        self._let_scopes[-1][name] = val

    def _find_let_scope(self, name: str) -> Optional[Dict[str, ShkValue]]:
        for scope in reversed(self.all_let_scopes()):
            if name in scope:
                return scope
        return None

    def get(self, name: str) -> ShkValue:
        scope = self._find_let_scope(name)
        if scope is not None:
            return scope[name]

        if name in self.vars:
            return self.vars[name]

        if self.parent is not None:
            return self.parent.get(name)

        raise ShakarRuntimeError(f"Name '{name}' not found")

    def set(self, name: str, val: ShkValue) -> None:
        scope = self._find_let_scope(name)
        if scope is not None:
            scope[name] = val
            return

        if name in self.vars:
            self.vars[name] = val
            return

        if self.parent is not None:
            self.parent.set(name, val)
            return

        raise ShakarRuntimeError(f"Name '{name}' not found")

    def push_defer_frame(self) -> None:
        self._defer_stack.append([])

    def pop_defer_frame(self) -> List[DeferEntry]:
        if not self._defer_stack:
            return []

        return self._defer_stack.pop()

    def current_defer_frame(self) -> List[DeferEntry]:
        if not self._defer_stack:
            raise ShakarRuntimeError("Cannot use defer outside of a block")

        return self._defer_stack[-1]

    def has_defer_frame(self) -> bool:
        return bool(self._defer_stack)

    def mark_function_frame(self) -> None:
        self._is_function_frame = True

    def is_function_frame(self) -> bool:
        return self._is_function_frame

    def get_emit_target(self) -> ShkValue:
        if self.emit_target is not None:
            return self.emit_target
        if self.parent is not None:
            return self.parent.get_emit_target()
        raise ShakarRuntimeError("No emit target available for '>'")


# ---------- Exceptions (keep Shakar* canonical) ----------


class ShakarRuntimeError(Exception):
    shk_type: Optional[str]
    shk_data: Optional[ShkValue]
    shk_payload: Optional[ShkValue]
    shk_meta: Optional[object]
    shk_py_trace: Optional[_pytypes.TracebackType]
    shk_call_stack: Optional[List["CallSite"]]

    def __init__(self, message: str):
        super().__init__(message)
        self.shk_type = None
        self.shk_data = None
        self.shk_payload = None
        self.shk_meta = None
        self.shk_py_trace = None
        self.shk_call_stack = None

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        msg = super().__str__()

        meta = getattr(self, "shk_meta", None)
        if meta is None:
            rendered = msg
        else:
            line = getattr(meta, "line", None)
            col = getattr(meta, "column", None)

            if line is None:
                rendered = msg
            elif col is None:
                rendered = f"{msg} (line {line})"
            else:
                rendered = f"{msg} (line {line}, col {col})"

        stack = getattr(self, "shk_call_stack", None)
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
    ShkFan,
    ShkObject,
    ShkModule,
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
    string_methods: MethodRegistry = {}
    regex_methods: MethodRegistry = {}
    object_methods: MethodRegistry = {}
    command_methods: MethodRegistry = {}
    path_methods: MethodRegistry = {}
    envvar_methods: MethodRegistry = {}
    channel_methods: MethodRegistry = {}
    stdlib_functions: Dict[str, StdlibFunction] = {}
    type_constants: Dict[str, "ShkType"] = {}
