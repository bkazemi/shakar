from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = (BASE_DIR / "src").resolve()

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from shakar_ref.ast_transforms import Prune
from shakar_ref.lexer_rd import LexError, Lexer
from shakar_ref.lower import lower
from shakar_ref.parser_rd import ParseError, parse_source as parse_rd
from shakar_ref.runner import run as run_program
from shakar_ref.runtime import (
    CommandError,
    ShakarArityError,
    ShakarAssertionError,
    ShakarImportError,
    ShakarMatchError,
    ShakarRuntimeError,
    ShakarTypeError,
    ShkArray,
    ShkBool,
    ShkCommand,
    ShkDuration,
    ShkFan,
    ShkNil,
    ShkNumber,
    ShkSize,
    ShkString,
)

RuntimeExpectation = Optional[Tuple[str, object]]
ParserCase = Tuple[str, str, str]

KEYWORDS = Lexer.KEYWORDS


@dataclass(frozen=True)
class LimitResult:
    """Holds the sampled size plus an optional truncation note."""

    size: int
    note: Optional[str]


@dataclass(frozen=True)
class KeywordPlan:
    """Captures keyword sampling decisions for parser sweeps."""

    sample: List[str]
    variants: int
    ident_limit: int
    notes: List[str]


PREFIX_SNIPPET_TEMPLATES = [
    "{ident} = 1",
    "a.{ident}()",
    "{ident}(1,2,3)",
    "x = {ident} + 2",
]
KEYWORD_SUFFIXES = ["ing", "ful", "_x", "Then", "Else", "Valid", "able"]
KEYWORD_PREFIXES = ["my", "pre", "x"]


def parse_pipeline(code: str, use_indenter: bool) -> object:
    """Run parse_rd -> prune -> lower and return the lowered tree."""
    tree = parse_rd(code, use_indenter=use_indenter)
    tree = Prune().transform(tree)
    return lower(tree)


def parse_both_modes(code: str, start: str) -> None:
    """Parse code according to the case start mode."""
    match start:
        case "both":
            parse_pipeline(code, use_indenter=False)
            parse_pipeline(code, use_indenter=True)
        case "noindent":
            parse_pipeline(code, use_indenter=False)
        case "indented":
            parse_pipeline(code, use_indenter=True)
        case _:
            raise AssertionError(f"unknown parser start mode {start!r}")


def verify_result(value: object, kind: str, expected: object) -> None:
    """Assert runtime result shape/value compatibility with legacy expectations."""
    match kind:
        case "string":
            assert isinstance(
                value, ShkString
            ), f"expected ShkString, got {type(value).__name__}"
            assert (
                value.value == expected
            ), f"expected {expected!r}, got {value.value!r}"
            return
        case "number":
            assert isinstance(
                value, ShkNumber
            ), f"expected number, got {type(value).__name__}"
            assert (
                abs(value.value - float(expected)) <= 1e-9
            ), f"expected {expected}, got {value.value}"
            return
        case "bool":
            assert isinstance(
                value, ShkBool
            ), f"expected bool, got {type(value).__name__}"
            assert bool(value.value) == bool(
                expected
            ), f"expected {expected}, got {value.value}"
            return
        case "null":
            assert isinstance(
                value, ShkNil
            ), f"expected ShkNil, got {type(value).__name__}"
            return
        case "command":
            assert isinstance(
                value, ShkCommand
            ), f"expected ShkCommand, got {type(value).__name__}"
            rendered = value.render()
            assert rendered == expected, f"expected {expected!r}, got {rendered!r}"
            return
        case "array":
            assert isinstance(
                value, ShkArray
            ), f"expected ShkArray, got {type(value).__name__}"
            actual_items = [
                item.value if hasattr(item, "value") else item for item in value.items
            ]
            assert (
                actual_items == expected
            ), f"expected {expected!r}, got {actual_items!r}"
            return
        case "fan":
            assert isinstance(
                value, ShkFan
            ), f"expected ShkFan, got {type(value).__name__}"
            actual_items = [
                item.value if hasattr(item, "value") else item for item in value.items
            ]
            assert (
                actual_items == expected
            ), f"expected {expected!r}, got {actual_items!r}"
            return
        case "duration":
            assert isinstance(
                value, ShkDuration
            ), f"expected ShkDuration, got {type(value).__name__}"
            assert value.nanos == expected, f"expected {expected}, got {value.nanos}"
            return
        case "size":
            assert isinstance(
                value, ShkSize
            ), f"expected ShkSize, got {type(value).__name__}"
            assert (
                value.byte_count == expected
            ), f"expected {expected}, got {value.byte_count}"
            return
        case _:
            raise AssertionError(f"unknown expectation kind {kind}")


def run_runtime_case(
    source: str,
    expectation: RuntimeExpectation,
    expected_exc: Optional[type],
) -> None:
    """Execute one runtime scenario with optional expected exception."""
    if expected_exc is not None:
        with pytest.raises(expected_exc):
            run_program(source)
        return

    result = run_program(source)
    if expectation is not None:
        verify_result(result, expectation[0], expectation[1])


def _limit_from_env(env_var: str, default: int, total: int, label: str) -> LimitResult:
    raw = os.getenv(env_var)
    if raw is None:
        if default >= total:
            return LimitResult(size=total, note=None)
        note = (
            f"[INFO] {label} truncated to {default}/{total} "
            f"(default; set {env_var}=full for full sweep)"
        )
        return LimitResult(size=default, note=note)

    raw = raw.strip()
    if raw.lower() in {"full", "all", "*"}:
        return LimitResult(size=total, note=None)

    try:
        value = max(0, int(raw))
    except ValueError:
        value = default

    if value >= total:
        return LimitResult(size=total, note=None)

    note = (
        f"[INFO] {label} truncated to {value}/{total} "
        f"({env_var}={raw}; use 'full' for complete set)"
    )
    return LimitResult(size=value, note=note)


def _keyword_sample(
    all_keywords: Sequence[str], default_limit: int
) -> Tuple[List[str], List[str]]:
    words = sorted(all_keywords)
    notes: List[str] = []
    raw = os.getenv("SANITY_KEYWORD_LIMIT")
    if not words:
        return [], notes

    if raw is None:
        if default_limit >= len(words):
            return words, notes
        notes.append(
            f"[INFO] keyword prefix cases truncated to {default_limit}/{len(words)} "
            f"(default limit {default_limit}; set SANITY_KEYWORD_LIMIT=full for full sweep)"
        )
        return words[:default_limit], notes

    raw = raw.strip()
    if raw.lower() in {"full", "all", "*"}:
        return words, notes

    try:
        limit = int(raw)
    except ValueError:
        limit = default_limit

    if limit <= 0:
        notes.append(
            f"[INFO] keyword prefix cases disabled (SANITY_KEYWORD_LIMIT={raw})"
        )
        return [], notes

    if limit >= len(words):
        return words, notes

    notes.append(
        f"[INFO] keyword prefix cases truncated to {limit}/{len(words)} "
        f"(SANITY_KEYWORD_LIMIT={raw}; use 'full' for complete set)"
    )
    return words[:limit], notes


def build_keyword_plan() -> KeywordPlan:
    sample, notes = _keyword_sample(KEYWORDS.keys(), default_limit=12)
    variant_limit = _limit_from_env(
        "SANITY_KEYWORD_VARIANTS",
        default=2,
        total=len(PREFIX_SNIPPET_TEMPLATES),
        label="keyword prefix contexts",
    )
    ident_limit = _limit_from_env(
        "SANITY_KEYWORD_IDENT_LIMIT",
        default=5,
        total=len(KEYWORD_SUFFIXES) + len(KEYWORD_PREFIXES),
        label="keyword identifier variants",
    )
    all_notes = notes[:]
    all_notes.extend(filter(None, (variant_limit.note, ident_limit.note)))
    return KeywordPlan(
        sample=sample,
        variants=variant_limit.size,
        ident_limit=ident_limit.size,
        notes=all_notes,
    )


def _identifier_variants(keyword: str, limit: int) -> List[str]:
    ids = [f"{keyword}{suffix}" for suffix in KEYWORD_SUFFIXES]
    ids += [f"{prefix}{keyword}" for prefix in KEYWORD_PREFIXES]
    return ids[:limit]


def build_keyword_cases(plan: KeywordPlan) -> List[ParserCase]:
    cases: List[ParserCase] = []
    templates = PREFIX_SNIPPET_TEMPLATES[: plan.variants]
    for kw in plan.sample:
        for ident in _identifier_variants(kw, plan.ident_limit):
            for idx, template in enumerate(templates):
                code = template.format(ident=ident)
                cases.append((f"ident-{kw}-{idx}", code, "both"))
    return cases


def check_zipwith(ast: object) -> Optional[str]:
    try:
        stmtlist = ast.children[0]
        chain = stmtlist.children[0]
        call = chain.children[1]
        lam = call.children[0].children[0]
    except Exception as exc:
        return f"structure mismatch: {exc}"

    if getattr(lam, "data", None) != "amp_lambda":
        return "lambda node missing"
    if not lam.children or getattr(lam.children[0], "data", None) != "paramlist":
        return "paramlist not inferred"

    params = [tok.value for tok in lam.children[0].children]
    if params != ["left", "right"]:
        return f"unexpected params {params}"
    return None


def check_map(ast: object) -> Optional[str]:
    stmtlist = ast.children[0]
    chain = stmtlist.children[0]
    call = chain.children[1]
    lam = call.children[0].children[0]

    if getattr(lam, "data", None) != "amp_lambda":
        return "lambda node missing"
    if len(lam.children) != 1:
        return "subject lambda should remain unary"
    if getattr(lam.children[0], "data", None) != "implicit_chain":
        return "subject lambda body altered"
    return None


def check_holes(ast: object) -> Optional[str]:
    stmtlist = ast.children[0]
    lam = stmtlist.children[0]
    if getattr(lam, "data", None) != "amp_lambda":
        return "top-level lambda missing"
    if len(lam.children) < 2:
        return "hole lambda missing paramlist"

    params = [tok.value for tok in lam.children[0].children]
    if params != ["_hole0", "_hole1"]:
        return f"unexpected params {params}"
    return None


def check_hook_inline(ast: object) -> Optional[str]:
    stmtlist = ast.children[0]
    hook = stmtlist.children[0]

    if getattr(hook, "data", None) != "hook":
        return "hook node missing"
    if not hook.children or hook.children[0].value != '"warn"':
        return "hook name missing"

    lam = hook.children[1] if len(hook.children) > 1 else None
    if getattr(lam, "data", None) != "amp_lambda":
        return "hook handler missing amp_lambda"

    body = lam.children[0] if lam.children else None
    if getattr(body, "data", None) != "body":
        return "hook body not body"
    return None


def check_decorator_def(ast: object) -> Optional[str]:
    stmtlist = ast.children[0]
    deco = stmtlist.children[0]
    if getattr(deco, "data", None) != "decorator_def":
        return "decorator node missing"

    name = deco.children[0] if deco.children else None
    if getattr(name, "value", None) != "logger":
        return "decorator name mismatch"

    body = next(
        (ch for ch in deco.children if getattr(ch, "data", None) == "body"),
        None,
    )
    if body is None:
        return "decorator body missing"
    return None


def check_decorated_fn(ast: object) -> Optional[str]:
    stmtlist = ast.children[0]
    fn_node = stmtlist.children[0]
    if getattr(fn_node, "data", None) != "fndef":
        return "fndef node missing"

    deco_list = next(
        (
            ch
            for ch in fn_node.children
            if getattr(ch, "data", None) == "decorator_list"
        ),
        None,
    )
    if deco_list is None:
        return "decorator list missing"
    if len(deco_list.children) != 1:
        return "decorator list size mismatch"
    return None


def _check_channel_cancel_race() -> Optional[str]:
    from shakar_ref.types import CancelToken, ShakarCancelledError, ShkChannel

    class ImmediateCancelToken(CancelToken):
        def register_condition(self, cond: threading.Condition) -> None:
            super().register_condition(cond)
            self.cancel()

    channel = ShkChannel()
    token = ImmediateCancelToken()
    done = threading.Event()
    outcome: Dict[str, Optional[object]] = {"status": None, "error": None}

    def worker() -> None:
        try:
            channel.recv_with_ok(cancel_token=token)
            outcome["status"] = "ready"
        except ShakarCancelledError:
            outcome["status"] = "cancelled"
        except Exception as exc:
            outcome["error"] = exc
        finally:
            done.set()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    if not done.wait(timeout=0.5):
        return "blocked recv after immediate cancellation"
    if outcome["error"] is not None:
        exc = outcome["error"]
        return f"unexpected {type(exc).__name__}: {exc}"
    if outcome["status"] != "cancelled":
        return "expected ShakarCancelledError"
    return None


def _check_dot_continuation_invalid() -> Optional[str]:
    src = """user := {profile: {name: "Ada"}}
name := user.profile
  .name
  oops
"""
    try:
        parse_rd(src, use_indenter=True)
    except ParseError:
        return None
    except Exception as exc:
        return f"unexpected {type(exc).__name__}: {exc}"
    return "expected ParseError for invalid dot continuation"


def check_channel_cancel_race() -> None:
    err = _check_channel_cancel_race()
    assert err is None, err


def check_dot_continuation_invalid() -> None:
    err = _check_dot_continuation_invalid()
    assert err is None, err
