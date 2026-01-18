from __future__ import annotations

from dataclasses import dataclass, field
import pathlib
import re
from typing import Callable, Dict, List, Optional, Tuple, TypeVar, Union, TYPE_CHECKING
from typing_extensions import Protocol, TypeAlias, TypeGuard
from .tree import Node

if TYPE_CHECKING:
    from .eval.bind import RebindContext, FanContext

# ---------- Value Model (only Sh* -> Shk*) ----------


@dataclass
class ShkNull:
    def __repr__(self) -> str:
        return "null"


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


StdlibFn = Callable[["Frame", List["ShkValue"]], "ShkValue"]


@dataclass(frozen=True)
class StdlibFunction:
    fn: StdlibFn
    arity: Optional[int] = None


ShkValue: TypeAlias = Union[
    ShkNull,
    ShkNumber,
    ShkDuration,
    ShkSize,
    ShkString,
    ShkRegex,
    ShkBool,
    ShkArray,
    ShkObject,
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
    BuiltinMethod,
    StdlibFunction,
    ShkType,
    ShkOptional,
    ShkUnion,
]

# Internal evaluator result type - includes assignment contexts
EvalResult: TypeAlias = Union[ShkValue, "RebindContext", "FanContext"]

DotValue: TypeAlias = Optional[ShkValue]


class Frame:
    def __init__(
        self,
        parent: Optional["Frame"] = None,
        dot: DotValue = None,
        emit_target: Optional["ShkValue"] = None,
        source: Optional[str] = None,
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
        self.source: Optional[str]

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

    def __init__(self, message: str):
        super().__init__(message)
        self.shk_type = None
        self.shk_data = None
        self.shk_payload = None
        self.shk_meta = None

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        msg = super().__str__()

        meta = getattr(self, "shk_meta", None)
        if meta is None:
            return msg

        line = getattr(meta, "line", None)
        col = getattr(meta, "column", None)

        if line is None:
            return msg

        if col is None:
            return f"{msg} (line {line})"

        return f"{msg} (line {line}, col {col})"


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


class ShakarReturnSignal(Exception):
    """Internal control-flow exception used to implement `return`."""

    def __init__(self, value: ShkValue):
        self.value = value


class ShakarBreakSignal(Exception):
    """Internal control flow for `break`."""


class ShakarContinueSignal(Exception):
    """Internal control flow for `continue`."""


_SHK_VALUE_TYPES: Tuple[type, ...] = (
    ShkNull,
    ShkNumber,
    ShkDuration,
    ShkSize,
    ShkString,
    ShkRegex,
    ShkBool,
    ShkArray,
    ShkObject,
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
        return ShkNull()
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
    stdlib_functions: Dict[str, StdlibFunction] = {}
    type_constants: Dict[str, "ShkType"] = {}
