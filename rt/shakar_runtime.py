from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# ---------- Value Model (only Sh* -> Shk*) ----------

@dataclass
class ShkNull:
    def __repr__(self) -> str:
        return "null"

@dataclass
class ShkNumber:
    value: float
    def __repr__(self) -> str:
        v = self.value
        return str(int(v)) if v.is_integer() else str(v)

@dataclass
class ShkString:
    value: str
    def __repr__(self) -> str:
        return f'"{self.value}"'

@dataclass
class ShkBool:
    value: bool
    def __repr__(self) -> str:
        return "true" if self.value else "false"

@dataclass
class ShkArray:
    items: List[Any]
    def __repr__(self) -> str:
        return "[" + ", ".join(repr(x) for x in self.items) + "]"

@dataclass
class ShkObject:
    slots: Dict[str, Any]
    def __repr__(self) -> str:
        pairs = []
        for k, v in self.slots.items():
            pairs.append(f"{k}: {repr(v)}")
        return "{ " + ", ".join(pairs) + " }"

@dataclass
class SelectorIndex:
    value: Any

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
                bits.append("" if part.stop is None else ("<" + str(part.stop) if part.exclusive_stop else str(part.stop)))
                if part.step is not None:
                    bits.append(str(part.step))
                while len(bits) < 3:
                    bits.append("")
                descr.append(":".join(bits[:3]).rstrip(":"))
        return "selector{" + ", ".join(descr) + "}"

@dataclass
class ShkFn:
    params: Optional[List[str]]  # None for subject-only amp-lambda
    body: Any                    # AST node
    env: 'Env'                   # Closure env

@dataclass
class BoundMethod:
    fn: ShkFn
    subject: Any

@dataclass
class BuiltinMethod:
    name: str
    subject: Any

@dataclass
class Descriptor:
    getter: Any = None  # ShkFn or None
    setter: Any = None  # ShkFn or None

class Env:
    def __init__(self, parent: Optional['Env']=None, dot: Any=None, source: Optional[str]=None):
        self.parent = parent
        self.vars: Dict[str, Any] = {}
        self.dot = dot
        if source is not None:
            self.source = source
        elif parent is not None and hasattr(parent, 'source'):
            self.source = parent.source
        else:
            self.source = None

    def define(self, name: str, val: Any) -> None:
        self.vars[name] = val

    def get(self, name: str) -> Any:
        if name in self.vars:
            return self.vars[name]
        if self.parent is not None:
            return self.parent.get(name)
        raise ShakarRuntimeError(f"Name '{name}' not found")

    def set(self, name: str, val: Any) -> None:
        if name in self.vars:
            self.vars[name] = val
            return
        if self.parent is not None:
            self.parent.set(name, val)
            return
        raise ShakarRuntimeError(f"Name '{name}' not found")

# ---------- Exceptions (keep Shakar* canonical) ----------

class ShakarRuntimeError(Exception):
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
    def __init__(self, recv: Any, name: str):
        super().__init__(f"{type(recv).__name__} has no builtin method '{name}'")
        self.receiver = recv
        self.name = name

# ---------- Built-in method registry ----------

class Builtins:
    array_methods: Dict[str, Callable[['Env', 'ShkArray', List[Any]], Any]] = {}
    string_methods: Dict[str, Callable[['Env', 'ShkString', List[Any]], Any]] = {}
    object_methods: Dict[str, Callable[['Env', 'ShkObject', List[Any]], Any]] = {}

    known_arity: Dict[Tuple[str, str], Tuple[str, int]] = {
        ("array","map"):     ("exact", 1),
        ("array","filter"):  ("exact", 1),
        ("array","zipWith"): ("exact", 2),
        ("string","len"):    ("exact", 0),
        ("string","trim"):   ("exact", 0),
        ("string","lower"):  ("exact", 0),
        ("string","upper"):  ("exact", 0),
        ("string","hasPrefix"): ("exact", 1),
        ("string","hasSuffix"): ("exact", 1),
        ("string","isAscii"): ("exact", 0),
        ("object","items"):  ("exact", 0),
    }

def register_array(name: str):
    def dec(fn):
        Builtins.array_methods[name] = fn
        return fn
    return dec

def register_string(name: str):
    def dec(fn):
        Builtins.string_methods[name] = fn
        return fn
    return dec

def register_object(name: str):
    def dec(fn):
        Builtins.object_methods[name] = fn
        return fn
    return dec

def _string_expect_arity(method: str, args: List[Any], expected: int) -> None:
    if len(args) != expected:
        raise ShakarArityError(f"string.{method} expects {expected} argument(s); got {len(args)}")

def _string_arg(method: str, arg: Any) -> str:
    if isinstance(arg, ShkString):
        return arg.value
    raise ShakarTypeError(f"string.{method} expects a string argument")

@register_string("trim")
def _string_trim(env: Env, recv: ShkString, args: List[Any]) -> ShkString:
    _string_expect_arity("trim", args, 0)
    return ShkString(recv.value.strip())

@register_string("lower")
def _string_lower(env: Env, recv: ShkString, args: List[Any]) -> ShkString:
    _string_expect_arity("lower", args, 0)
    return ShkString(recv.value.lower())

@register_string("upper")
def _string_upper(env: Env, recv: ShkString, args: List[Any]) -> ShkString:
    _string_expect_arity("upper", args, 0)
    return ShkString(recv.value.upper())

@register_string("hasPrefix")
def _string_has_prefix(env: Env, recv: ShkString, args: List[Any]) -> ShkBool:
    _string_expect_arity("hasPrefix", args, 1)
    prefix = _string_arg("hasPrefix", args[0])
    return ShkBool(recv.value.startswith(prefix))

@register_string("hasSuffix")
def _string_has_suffix(env: Env, recv: ShkString, args: List[Any]) -> ShkBool:
    _string_expect_arity("hasSuffix", args, 1)
    suffix = _string_arg("hasSuffix", args[0])
    return ShkBool(recv.value.endswith(suffix))

@register_string("isAscii")
def _string_is_ascii(env: Env, recv: ShkString, args: List[Any]) -> ShkBool:
    _string_expect_arity("isAscii", args, 0)
    return ShkBool(recv.value.isascii())

def call_builtin_method(recv: Any, name: str, args: List[Any], env: 'Env') -> Any:
    if isinstance(recv, ShkArray):
        fn = Builtins.array_methods.get(name)
        if fn: return fn(env, recv, args)
    if isinstance(recv, ShkString):
        fn = Builtins.string_methods.get(name)
        if fn: return fn(env, recv, args)
    if isinstance(recv, ShkObject):
        fn = Builtins.object_methods.get(name)
        if fn: return fn(env, recv, args)
    raise ShakarMethodNotFound(recv, name)

def call_shkfn(fn: ShkFn, positional: List[Any], subject: Any, caller_env: 'Env') -> Any:
    """
    Subjectful call semantics:
    - subject is available to callee as env.dot
    - If fn.params is None, treat as unary subject-lambda: ignore positional, evaluate body with dot bound.
    - Else: arity must match len(fn.params); bind params; dot=subject.
    """
    from shakar_eval import eval_node  # local import to avoid cycle
    callee_env = Env(parent=fn.env, dot=subject)
    if fn.params is None:
        return eval_node(fn.body, callee_env)
    if len(positional) != len(fn.params):
        raise ShakarArityError(f"Function expects {len(fn.params)} args; got {len(positional)}")
    for name, val in zip(fn.params, positional):
        callee_env.define(name, val)
    return eval_node(fn.body, callee_env)
