"""
sanity_check_basic.py — regression driver for the "basic" lexer pipeline.

The suite exercises three layers:
1. Grammar sanity: each sample parses under the appropriate start symbol.
2. AST shape checks: key amp-lambda and decorator transforms.
3. Runtime behaviour: canned programs run through the interpreter.

The script prints a textual report and exits non-zero on failure.
"""

from __future__ import annotations

import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))
SRC_DIR = (BASE_DIR / "src").resolve()
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from shakar_ref.runner import run as run_program
from shakar_ref.runtime import (
    CommandError,
    ShkCommand,
    ShkNull,
    ShkNumber,
    ShkString,
    ShkBool,
    ShakarArityError,
    ShakarAssertionError,
    ShakarRuntimeError,
    ShakarTypeError,
)
from shakar_ref import parse_auto as runner

# Optional RD parser import
TEST_RD_PARSER = os.getenv("TEST_RD_PARSER", "").lower() in {"1", "true", "yes"}
if TEST_RD_PARSER:
    from shakar_ref.parser_rd import parse_source as parse_rd, ParseError
    from shakar_ref.lexer_rd import LexError
    from shakar_ref.parse_auto import Prune
    from shakar_ref.lower import lower
    from lark import Tree, Token

GRAMMAR_PATH = Path("grammar.lark")
GRAMMAR_TEXT = GRAMMAR_PATH.read_text(encoding="utf-8")

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Case:
    """Parser sample used for grammar sanity checks."""
    name: str
    code: str
    start: str  # "noindent", "indented", or "both"

@dataclass(frozen=True)
class AstScenario:
    """Validates AST transforms (amp lambdas, decorators, etc.)."""
    name: str
    code: str
    checker: Optional[Callable[[object], Optional[str]]]
    expected_exc: Optional[type]

@dataclass(frozen=True)
class RuntimeScenario:
    """Runs a source string end-to-end through the interpreter."""
    name: str
    source: str
    expectation: Optional[Tuple[str, object]]
    expected_exc: Optional[type]

@dataclass(frozen=True)
class LimitResult:
    """Holds the sampled size plus an optional truncation note."""
    size: int
    note: Optional[str]

@dataclass(frozen=True)
class KeywordPlan:
    """Captures keyword sampling decisions for the parser sweeps."""
    sample: List[str]
    variants: int
    ident_limit: int
    notes: List[str]

# ---------------------------------------------------------------------------
# Limit computation helpers
# ---------------------------------------------------------------------------

def _limit_from_env(env_var: str, default: int, total: int, label: str) -> LimitResult:
    raw = os.getenv(env_var)
    if raw is None:
        if default >= total:
            return LimitResult(size=total, note=None)
        note = f"[INFO] {label} truncated to {default}/{total} (default; set {env_var}=full for full sweep)"
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
    note = f"[INFO] {label} truncated to {value}/{total} ({env_var}={raw}; use 'full' for complete set)"
    return LimitResult(size=value, note=note)

def _keyword_sample(all_keywords: Sequence[str], default_limit: int) -> Tuple[List[str], List[str]]:
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
        notes.append(f"[INFO] keyword prefix cases disabled (SANITY_KEYWORD_LIMIT={raw})")
        return [], notes
    if limit >= len(words):
        return words, notes
    notes.append(
        f"[INFO] keyword prefix cases truncated to {limit}/{len(words)} "
        f"(SANITY_KEYWORD_LIMIT={raw}; use 'full' for complete set)"
    )
    return words[:limit], notes

def build_keyword_plan() -> KeywordPlan:
    sample, notes = _keyword_sample(runner.KEYWORDS.keys(), default_limit=12)
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

# ---------------------------------------------------------------------------
# Parser management
# ---------------------------------------------------------------------------

class ParserBundle:
    """Caches the no-indent and indented parser variants for reuse."""
    def __init__(self, grammar_text: str):
        self.parsers = {
            "noindent": runner.build_parser(
                grammar_text,
                parser_kind="earley",
                use_indenter=False,
                start_sym="start_noindent",
            ),
            "indented": runner.build_parser(
                grammar_text,
                parser_kind="earley",
                use_indenter=True,
                start_sym="start_indented",
            ),
        }
        self.test_rd = TEST_RD_PARSER

    def _trees_match(self, rd_tree, lark_tree, path="") -> Tuple[bool, str]:
        """Compare RD and Lark trees for equivalence."""
        # Both are tokens
        if isinstance(rd_tree, Token) and isinstance(lark_tree, Token):
            if rd_tree.type != lark_tree.type:
                return False, f"{path}: Token type mismatch: {rd_tree.type} vs {lark_tree.type}"
            if rd_tree.value != lark_tree.value:
                return False, f"{path}: Token value mismatch: {rd_tree.value!r} vs {lark_tree.value!r}"
            return True, ""

        # Both are trees
        if isinstance(rd_tree, Tree) and isinstance(lark_tree, Tree):
            if rd_tree.data != lark_tree.data:
                return False, f"{path}: Tree data mismatch: {rd_tree.data} vs {lark_tree.data}"

            if len(rd_tree.children) != len(lark_tree.children):
                return False, f"{path}: Child count mismatch: {len(rd_tree.children)} vs {len(lark_tree.children)}"

            for i, (rd_child, lark_child) in enumerate(zip(rd_tree.children, lark_tree.children)):
                match, err = self._trees_match(rd_child, lark_child, f"{path}/{rd_tree.data}[{i}]")
                if not match:
                    return False, err

            return True, ""

        # Type mismatch
        return False, f"{path}: Type mismatch: {type(rd_tree).__name__} vs {type(lark_tree).__name__}"

    def parse(self, start: str, code: str) -> Tuple[bool, Optional[str]]:
        """Parse using the requested start symbol, returning (ok, error_message)."""
        parser = self.parsers[start]
        label = f"start_{start}"

        # Parse with Lark
        try:
            lark_tree = parser.parse(code)
        except Exception as exc:  # pragma: no cover - defensive
            return False, f"{label} (Lark): {exc}"

        # If RD testing is enabled, compare
        if self.test_rd:
            try:
                # Parse with RD (use indenter for indented mode)
                use_indenter = (start == "indented")
                rd_tree = parse_rd(code, use_indenter=use_indenter)
                # Apply same transformations as Lark
                rd_tree = Prune().transform(rd_tree)
                rd_tree = lower(rd_tree)

                # Also apply transformations to Lark tree for fair comparison
                lark_tree_transformed = Prune().transform(lark_tree)
                lark_tree_transformed = lower(lark_tree_transformed)

                # Compare trees
                match, error = self._trees_match(rd_tree, lark_tree_transformed)
                if not match:
                    return False, f"{label} (RD vs Lark): AST mismatch: {error}"

            except (ParseError, LexError) as exc:
                return False, f"{label} (RD): {exc}"
            except Exception as exc:
                return False, f"{label} (RD): {exc}"

        return True, None

# ---------------------------------------------------------------------------
# Case builders
# ---------------------------------------------------------------------------

PREFIX_SNIPPET_TEMPLATES = [
    "{ident} = 1",
    "a.{ident}()",
    "{ident}(1,2,3)",
    "x = {ident} + 2",
]
KEYWORD_SUFFIXES = ["ing", "ful", "_x", "Then", "Else", "Valid", "able"]
KEYWORD_PREFIXES = ["my", "pre", "x"]

CASE_BUILDERS: List[Callable[[KeywordPlan], List[Case]]] = []

def case_builder(func: Callable[[KeywordPlan], List[Case]]) -> Callable[[KeywordPlan], List[Case]]:
    CASE_BUILDERS.append(func)
    return func

def _identifier_variants(keyword: str, limit: int) -> List[str]:
    ids = [f"{keyword}{suffix}" for suffix in KEYWORD_SUFFIXES]
    ids += [f"{prefix}{keyword}" for prefix in KEYWORD_PREFIXES]
    return ids[:limit]

@case_builder
def build_keyword_cases(plan: KeywordPlan) -> List[Case]:
    cases: List[Case] = []
    templates = PREFIX_SNIPPET_TEMPLATES[:plan.variants]
    for kw in plan.sample:
        for ident in _identifier_variants(kw, plan.ident_limit):
            for idx, template in enumerate(templates):
                code = template.format(ident=ident)
                cases.append(Case(name=f"ident-{kw}-{idx}", code=code, start="both"))
    return cases

@case_builder
def build_operator_cases(_: KeywordPlan) -> List[Case]:
    samples = [
        "1 and 2 or 3",
        "x and y and z or w",
        "a and $b and .c",
        "not 1",
        "not (1 and 2)",
        "a is b",
        "a is not b",
        "a !is b",
        "a in b",
        "a not in b",
        "a !in b",
        "7 // 2",
    ]
    return [Case(name=f"op-{i}", code=src, start="both") for i, src in enumerate(samples)]

@case_builder
def build_postfix_if_cases(_: KeywordPlan) -> List[Case]:
    samples = ["1 if 0", "foo() if bar", "(a+b) if c", "x.y if z"]
    return [Case(name=f"postfixif-{i}", code=src, start="both") for i, src in enumerate(samples)]

@case_builder
def build_await_cases(_: KeywordPlan) -> List[Case]:
    samples = [
        "await f()",
        "x = await g(1,2)",
        "h(await k())",
    ]
    return [Case(name=f"await-{i}", code=src, start="both") for i, src in enumerate(samples)]

@case_builder
def build_block_cases(_: KeywordPlan) -> List[Case]:
    samples = [
        "if 1:\n  a\n  b\n",
        "if a:\n  b\nelif c:\n  d\nelse:\n  e\n",
        "if a:\n  if b:\n    c\n  else:\n    d\n",
    ]
    return [Case(name=f"block-{i}", code=src, start="indented") for i, src in enumerate(samples)]

@case_builder
def build_misc_cases(_: KeywordPlan) -> List[Case]:
    samples = [
        "a.b[c](d, e).f",
        "a.b; c.d; e",
        "x = (a.b + c[d]) * e.f(g)",
    ]
    return [Case(name=f"misc-{i}", code=src, start="both") for i, src in enumerate(samples)]

@case_builder
def build_fn_cases(_: KeywordPlan) -> List[Case]:
    samples = [
        "fn add(x, y): x + y",
        "fn greet(name): { dbg(name) }",
    ]
    return [Case(name=f"fn-{i}", code=src, start="both") for i, src in enumerate(samples)]

# ---------------------------------------------------------------------------
# AST + runtime scenarios
# ---------------------------------------------------------------------------

AST_SCENARIOS: List[AstScenario] = []
RUNTIME_SCENARIOS: List[RuntimeScenario] = []

def runtime_scenario(func: Callable[[], RuntimeScenario]) -> None:
    RUNTIME_SCENARIOS.append(func())

# AST check helpers

def _check_zipwith(ast) -> Optional[str]:
    try:
        stmtlist = ast.children[0]
        chain = stmtlist.children[0]
        call = chain.children[1]
        lam = call.children[0].children[0]
    except Exception as exc:  # pragma: no cover - defensive
        return f"structure mismatch: {exc}"
    if getattr(lam, "data", None) != "amp_lambda":
        return "lambda node missing"
    if not lam.children or getattr(lam.children[0], "data", None) != "paramlist":
        return "paramlist not inferred"
    params = [tok.value for tok in lam.children[0].children]
    if params != ["left", "right"]:
        return f"unexpected params {params}"
    return None

def _check_map(ast) -> Optional[str]:
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

def _check_holes(ast) -> Optional[str]:
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

def _check_hook_inline(ast) -> Optional[str]:
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
    if getattr(body, "data", None) != "inlinebody":
        return "hook body not inlinebody"
    return None

def _check_decorator_def(ast) -> Optional[str]:
    stmtlist = ast.children[0]
    deco = stmtlist.children[0]
    if getattr(deco, "data", None) != "decorator_def":
        return "decorator node missing"
    name = deco.children[0] if deco.children else None
    if getattr(name, "value", None) != "logger":
        return "decorator name mismatch"
    body = next((ch for ch in deco.children if getattr(ch, "data", None) in {"inlinebody", "indentblock"}), None)
    if body is None:
        return "decorator body missing"
    return None

def _check_decorated_fn(ast) -> Optional[str]:
    stmtlist = ast.children[0]
    fn_node = stmtlist.children[0]
    if getattr(fn_node, "data", None) != "fndef":
        return "fndef node missing"
    deco_list = next((ch for ch in fn_node.children if getattr(ch, "data", None) == "decorator_list"), None)
    if deco_list is None:
        return "decorator list missing"
    if len(deco_list.children) != 1:
        return "decorator list size mismatch"
    return None

AST_SCENARIOS.extend([
    AstScenario("lambda-infer-zipwith", 'zipWith&(left + right)(xs, ys)', _check_zipwith, None),
    AstScenario("lambda-respect-subject", 'map&(.trim())', _check_map, None),
    AstScenario("lambda-hole-desugar", 'blend(?, ?, 0.25)', _check_holes, None),
    AstScenario("lambda-dot-mix-error", 'map&(value + .trim())', None, SyntaxError),
    AstScenario("hook-inline-body", 'hook "warn": .trim()', _check_hook_inline, None),
    AstScenario("decorator-ast-def", 'decorator logger(msg): args', _check_decorator_def, None),
    AstScenario("decorated-fn-ast", """@noop
fn hi(): 1""", _check_decorated_fn, None),
])

# Runtime scenarios

def _rt(name: str, source: str, expectation: Optional[Tuple[str, object]], expected_exc: Optional[type]) -> RuntimeScenario:
    return RuntimeScenario(name, source, expectation, expected_exc)

runtime_scenario(
    lambda: _rt(
        "decorator-pass-through",
        """decorator noop(): args
@noop
fn hi(name): name
hi("Ada")""",
        ("string", "Ada"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "decorator-arg-mutate",
        """decorator double(): args[0] = args[0] * 2
@double
fn sum(x, y): x + y
sum(3, 4)""",
        ("number", 10),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "decorator-return-shortcut",
        """decorator always_nil(): return nil
@always_nil
fn value(): 42
value()""",
        ("null", None),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "number-len-typeerror",
        "x := 1\nx.len",
        None,
        ShakarTypeError,
    )
)
runtime_scenario(
    lambda: _rt(
        "object-assign-requires-existing-field",
        "o := {}\no.x = 1",
        None,
        ShakarRuntimeError,
    )
)
runtime_scenario(
    lambda: _rt(
        "assign-requires-existing-var",
        "x = 1",
        None,
        ShakarRuntimeError,
    )
)
runtime_scenario(
    lambda: _rt(
        "int-builtin",
        "int(3) + int(\"4\")",
        ("number", 7),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "array-concat",
        "([1,2] + [3]).len",
        ("number", 3),
        None,
    )
)
runtime_scenario(lambda: _rt("join-array", '", ".join(["a", "b"])', ("string", "a, b"), None))
runtime_scenario(lambda: _rt("join-varargs", '"-".join("a", "b")', ("string", "a-b"), None))
runtime_scenario(lambda: _rt("join-mixed", '"|".join("a", 1, true)', ("string", "a|1|true"), None))
runtime_scenario(
    lambda: _rt(
        "decorator-chain-order",
        """decorator mark(label): args[0] = args[0] * 10 + label
@mark(3)
@mark(2)
fn encode(x): x
encode(1)""",
        ("number", 123),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "decorator-call-twice",
        """decorator double_call(): return f(args) + f(args)
@double_call
fn point(): 2
point()""",
        ("number", 4),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "decorator-params",
        """decorator add_offset(delta): args[0] = args[0] + delta
@add_offset(5)
fn bump(x): x
bump(3)""",
        ("number", 8),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "using-enter-exit",
        """flag := { value: 0 }
resource := {
  enter: fn():
    flag.value = flag.value + 1
    return "ok"
  exit: fn(err):
    flag.value = flag.value + 1
}
using[f] resource:
  flag.value = flag.value + 1
flag.value""",
        ("number", 3),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "using-enter-only",
        """flag := { value: 0 }
resource := {
  enter: fn(): flag.value = flag.value + 1
}
using resource:
  flag.value = flag.value + 1
flag.value""",
        ("number", 2),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "using-binder",
        """resource := {
  enter: fn(): { x: 1 }
}
using resource bind r:
  r.x""",
        ("number", 1),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "using-handle-default-binder",
        """resource := { enter: fn(): 7 }
using[r] resource:
  r + 1""",
        ("number", 8),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "using-using_exit-suppresses",
        """resource := {
  using_enter: fn(): 1
  using_exit: fn(err): true
}
using resource:
  throw "boom"
1""",
        ("number", 1),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "using-exit-err-propagates",
        """resource := {
  exit: fn(err): err
}
using resource:
  throw "boom" """,
        ("null", None),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "using-inlinebody",
        """resource := { enter: fn(): 5 }
using resource: resource + 1""",
        ("number", 6),
        None,
    )
)
runtime_scenario(lambda: _rt("lambda-subject-direct-call", 'a := &(.trim()); a(" B")', ("string", "B"), None))
runtime_scenario(lambda: _rt("floor-div-basic", '7 // 2', ("number", 3), None))
runtime_scenario(lambda: _rt("floor-div-negative", '-7 // 2', ("number", -4), None))
runtime_scenario(lambda: _rt("compound-assign-number", 'a := 1; a += 2; a', ("number", 3), None))
runtime_scenario(lambda: _rt("compound-assign-string", 's := "a"; s += "b"; s', ("string", "ab"), None))
runtime_scenario(lambda: _rt("compound-assign-mod", 'a := 10; a %= 3; a', ("number", 1), None))
runtime_scenario(lambda: _rt("compound-assign-minus", 'a := 10; a -= 4; a', ("number", 6), None))
runtime_scenario(lambda: _rt("compound-assign-mul", 'a := 3; a *= 4; a', ("number", 12), None))
runtime_scenario(lambda: _rt("compound-assign-div", 'a := 9; a /= 2; a', ("number", 4.5), None))
runtime_scenario(lambda: _rt("compound-assign-floordiv", 'a := 9; a //= 2; a', ("number", 4), None))
runtime_scenario(lambda: _rt("fanout-block-basic", 'state := {cur: 1, next: 2, x: 0}; state{ .cur = .next; .x += 5 }; state.cur + state.x', ("number", 7), None))
runtime_scenario(lambda: _rt("fanout-block-apply", 's := {name: " Ada ", greet: ""}; s{ .name .= .trim(); .greet = .name }; s.greet', ("string", "Ada"), None))
runtime_scenario(lambda: _rt("fanout-block-dup-error", 'state := {a: 1}; state{ .a = 1; .a = 2 }', None, ShakarRuntimeError))
runtime_scenario(lambda: _rt("fanout-value-array", 'state := {a: 1, b: 2}; arr := state.{a, b}; arr[0] + arr[1]', ("number", 3), None))
runtime_scenario(lambda: _rt("fanout-call-spread", 'state := {a: 1, b: 2}; fn add(x, y): x + y; add(state.{a, b})', ("number", 3), None))
runtime_scenario(lambda: _rt("fanout-value-call-item", 'state := {a: fn():3, b: 2}; vals := state.{a(), b}; vals[0] + vals[1]', ("number", 5), None))
runtime_scenario(lambda: _rt("fanout-named-arg-no-spread", 'state := {a: 1, b: 2}; fn wrap(x): x[1]; wrap(named: state.{a, b})', ("number", 2), None))
runtime_scenario(lambda: _rt("all-varargs", 'all(true, 1, "x")', ("bool", True), None))
runtime_scenario(lambda: _rt("all-varargs-short", 'all(true, false, 1)', ("bool", False), None))
runtime_scenario(lambda: _rt("all-iterable", 'all([true, 1, "x"])', ("bool", True), None))
runtime_scenario(lambda: _rt("all-empty-iterable", 'all([])', ("bool", True), None))
runtime_scenario(lambda: _rt("all-zero-args-error", 'all()', None, ShakarRuntimeError))
runtime_scenario(lambda: _rt("any-varargs", 'any(false, 0, "x")', ("bool", True), None))
runtime_scenario(lambda: _rt("any-iterable", 'any([false, 0, ""])', ("bool", False), None))
runtime_scenario(lambda: _rt("raw-string-basic", 'raw"hi {name}\\n"', ("string", "hi {name}\\n"), None))
runtime_scenario(lambda: _rt("raw-hash-string", 'raw#"path "C:\\\\tmp"\\file"#', ("string", 'path "C:\\\\tmp"\\file'), None))
runtime_scenario(lambda: _rt("shell-string-quote", 'path := "file name.txt"; sh"cat {path}"', ("command", "cat 'file name.txt'"), None))
runtime_scenario(lambda: _rt("shell-string-array", 'files := ["a.txt", "b 1.txt"]; sh"ls {files}"', ("command", "ls a.txt 'b 1.txt'"), None))
runtime_scenario(lambda: _rt("shell-string-raw-splice", 'flag := "-n 2"; file := "log 1.txt"; sh"head {{flag}} {file}"', ("command", "head -n 2 'log 1.txt'"), None))
runtime_scenario(lambda: _rt("shell-run-stdout", 'msg := "hi"; res := (sh"printf {msg}").run(); res', ("string", "hi"), None))
runtime_scenario(lambda: _rt("shell-run-code", '(sh"false").run()', None, CommandError))
runtime_scenario(lambda: _rt("shell-run-catch-code", 'val := (sh"false").run() catch err: err.code', ("number", 1), None))
runtime_scenario(
    lambda: _rt(
        "listcomp-filter",
        """src := [1, 2, 3, 4]
odds := [ n over src if n % 2 == 1 ]
odds[1]""",
        ("number", 3),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "listcomp-binder",
        """pairs := [[i, v] over [i, v] [[0, "a"], [1, "b"]]]
pairs[1][1]""",
        ("string", "b"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "setliteral-sum",
        """vals := set{1, 2, 1}
total := 0
for v in vals: total += v
total""",
        ("number", 3),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "setcomp-overspec",
        """src := [[0, 2], [1, 3]]
sums := set{ i + v over [i, v] src }
total := 0
for v in sums: total += v
total""",
        ("number", 6),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "dictcomp-basic",
        """items := ["a", "b"]
obj := { k: k + "!" over items bind k }
obj["b"]""",
        ("string", "b!"),
        None,
    )
)
runtime_scenario(lambda: _rt("fn-definition", 'fn add(x, y): x + y; add(2, 3)', ("number", 5), None))
runtime_scenario(lambda: _rt("fn-closure", 'y := 5; fn addY(x): x + y; addY(2)', ("number", 7), None))
runtime_scenario(
    lambda: _rt(
        "fn-return-value",
        """fn addOne(x): { return x + 1 }
addOne(4)""",
        ("number", 5),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "fn-return-default-null",
        """fn noop(): { return }
noop()""",
        ("null", None),
        None,
    )
)
runtime_scenario(lambda: _rt("anon-fn-expression", 'inc := fn(x): { x + 1 }; inc(5)', ("number", 6), None))
runtime_scenario(
    lambda: _rt(
        "anon-fn-block",
        """make := fn(): { value := 2; value + 3 }
make()""",
        ("number", 5),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "anon-fn-auto",
        """result := fn(()) : { tmp := 4; tmp + 1 }
result""",
        ("number", 5),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "anon-fn-inline",
        """inline := fn(): 40 + 2
inline()""",
        ("number", 42),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "anon-fn-inline-store",
        """fnRef := fn(): print("awd")
"done\"""",
        ("string", "done"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "await-value",
        """fn id(x): x
await id(5)""",
        ("number", 5),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "await-stmt-body",
        """val := 0
await sleep(10): { val = . }
val""",
        ("null", None),
        None,
    )
)
runtime_scenario(lambda: _rt("await-any-trailing-body", 'await [any]( fast: sleep(10), slow: sleep(50) ): winner', ("string", "fast"), None))
runtime_scenario(lambda: _rt("await-any-inline-body", 'await [any]( fast: sleep(10): "done" )', ("string", "done"), None))
runtime_scenario(lambda: _rt("await-all-trailing-body", 'await [all]( first: sleep(10), second: sleep(20) ): "ok"', ("string", "ok"), None))
runtime_scenario(
    lambda: _rt(
        "no-anchor-preserves-dot",
        """user := { name: "Ada" }
user and $"skip" and .name""",
        ("string", "Ada"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "and-literal-anchor",
        """a := "a"
a and 1 and .upper()""",
        ("string", "A"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "anchor-law",
        """user := { id: "outer" }
other := { id: "inner" }
user and (other and .id) and .id""",
        ("string", "outer"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "selector-base-anchor",
        """users := ["aa", "bb"]
seen := users.len
users[seen - 1]
"" + seen""",
        ("string", "2.0"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "group-anchor",
        """user := { friend: { name: "bob" } }
(user.friend and .name)""",
        ("string", "bob"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "leading-dot-chain-law",
        """user := { profile: { name: "  Ada " }, id: "ID" }
user and (.profile.name.trim()) and .id""",
        ("string", "ID"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "statement-subject-locality",
        """outer := { size: 42 }
u := "  hi  "
outer and (=u .trim()) and .size""",
        ("number", 42),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "statement-subject-grouped-anchor",
        """user := { profile: { contact: { name: "  Ada " } } }
=(user.profile.contact.name).trim()
user.profile.contact.name""",
        ("string", "Ada"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "statement-subject-retarget-ident",
        """user := { profile: { contact: { name: "  Ada " } } }
=(user).profile.contact.name.trim()
user""",
        ("string", "Ada"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "statement-subject-grouped-ident-tail",
        """a := { b: "s" }
=(a).b
a""",
        ("string", "s"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "statement-subject-missing-tail",
        """user := { name: "Ada" }
=user.name""",
        None,
        ShakarRuntimeError,
    )
)
runtime_scenario(
    lambda: _rt(
        "applyassign-chain",
        """profile := { name: "  Ada " }
profile.name .= .trim()
profile.name""",
        ("string", "Ada"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "nullish-chain",
        """nil ?? "guest" ?? "fallback" """,
        ("string", "guest"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "ternary-expr",
        """flag := true
flag ? "yes" : "no" """,
        ("string", "yes"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "string-interp",
        """user := { name: "Ada", score: 5 }
msg := "Name: {user.name}, score: {user.score}"
msg""",
        ("string", "Name: Ada, score: 5"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "string-interp-braces",
        """value := 10
text := "set {{value}} = {value}"
text""",
        ("string", "set {value} = 10"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "string-interp-single-quote",
        """user := { name: "Ada" }
text := 'hi {user.name}!'
text""",
        ("string", "hi Ada!"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "ccc-runtime",
        """temp := 50
verdict := "fail"
if temp > 40, < 60, and != 55: verdict = "ok"
verdict""",
        ("string", "ok"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "selector-literal-pick",
        """sel := `0, 2`
arr := [10, 20, 30]
picked := arr[sel]
picked[0] + picked[1]""",
        ("number", 40),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "selector-slice",
        """arr := [10, 20, 30, 40]
slice := arr[1:3]
slice[0] + slice[1]""",
        ("number", 50),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "selector-literal-interp",
        """start := 1
stop := 3
sel := `{start}:{stop}`
arr := [4, 5, 6, 7]
sum := arr[sel]
sum[0] + sum[1]""",
        ("number", 11),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "object-index-default",
        """cfg := { db: { host: "db" } }
calls := 0
fn fallback():
  calls += 1
  return "localhost"
found := cfg["db", default: {}]["host", default: fallback()]
missing := cfg["port", default: fallback()]
assert calls == 1
"{found}-{missing}"
""",
        ("string", "db-localhost"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "array-index-default-error",
        """arr := [1, 2, 3]
arr[0, default: 9]""",
        None,
        ShakarTypeError,
    )
)
runtime_scenario(
    lambda: _rt(
        "postfix-if-walrus-true",
        """calls := 0
fn bump(): { calls += 1; "run" }
result := bump() if true
assert calls == 1
result""",
        ("string", "run"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "postfix-if-walrus-false",
        """calls := 0
fn bump(): { calls += 1; "run" }
result := bump() if false
assert calls == 0
assert result == nil
result""",
        ("null", None),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "postfix-if-assign-noop",
        """value := 1
value = 2 if false
assert value == 1
value""",
        ("number", 1),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "postfix-unless-walrus",
        """calls := 0
fn bump(): { calls += 1; "run" }
result := bump() unless true
assert calls == 0
fallback := bump() unless false
assert calls == 1
fallback""",
        ("string", "run"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "guard-oneline",
        """result := "unset"
true: result = "hit" | false: result = "miss" |: result = "else"
result""",
        ("string", "hit"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "while-basic",
        """i := 0
while i < 3: { i += 1 }
i""",
        ("number", 3),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "while-break-continue",
        """i := 0
acc := 0
while true:
  i += 1
  if i == 5: break
  if i % 2 == 0: continue
  acc += i
acc""",
        ("number", 4),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "object-getter-setter",
        """obj := {
  name: "Ada"
  total: 12
  get label(): .name
  set label(next):
    owner := .
    owner.name = next.trim().upper()
  get double(): .total * 2
  set double(next):
    owner := .
    owner.total = next / 2
  greet(name): "hi " + name
  ("dyn" + "Key"): 42
}
before := obj.double
obj.double = 10
after_dot := obj.double
obj["double"] = 18
after_index := obj.double
label_before := obj.label
obj.label = "  grace  "
label_after := obj.label
greet := obj.greet(label_after)
dyn := obj["dynKey"]
label_before + "|" + greet + "|" + ("" + dyn) + "|" + ("" + before) + "|" + ("" + after_dot) + "|" + ("" + after_index) + "|" + label_after""",
        ("string", "Ada|hi GRACE|42.0|24.0|10.0|18.0|GRACE"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "walrus-duplicate-name",
        """a := 1
a := 2""",
        None,
        ShakarRuntimeError,
    )
)
runtime_scenario(
    lambda: _rt(
        "catch-expr-binder",
        """value := (missingVar catch err: err.type)
value""",
        ("string", "ShakarRuntimeError"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "catch-expr-dot",
        """fallback := missingVar @@: .message
fallback""",
        ("string", "Name 'missingVar' not found"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "catch-stmt-binder",
        """msg := ""
missingVar catch err: { msg = err.message }
msg""",
        ("string", "Name 'missingVar' not found"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "catch-stmt-dot",
        """typ := ""
missingVar catch: { typ = .type }
typ""",
        ("string", "ShakarRuntimeError"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "catch-type-match",
        """value := (missingVar catch (ShakarRuntimeError) bind err: err.type)
value""",
        ("string", "ShakarRuntimeError"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "catch-typed-bind",
        """fn risky(tag): { throw error(tag, "bad") }
value := (risky("ValidationError") catch (ValidationError, Missing) bind err: err.type)
value""",
        ("string", "ValidationError"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "return-if",
        """fn first_even(xs): { for n in xs: { if n % 2 == 0: { ?ret n } }; "none" }
first_even([1, 3, 5, 6, 8])""",
        ("number", 6),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "for-in-sum",
        """sum := 0
for x in [1, 2, 3]: sum = sum + x
sum""",
        ("number", 6),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "for-subject-dot",
        """acc := ""
for ["a", "b"]: acc = acc + .upper()
acc""",
        ("string", "AB"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "for-indexed",
        """logs := ""
items := ["a", "b"]
for[i] items: logs = logs + ("" + i)
logs""",
        ("string", "0.01.0"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "for-map-key-value",
        """obj := { "a": 1, "b": 2 }
keys := ""
sum := 0
for[k, v] obj: { keys = keys + k; sum = sum + v }
keys + ":" + ("" + sum)""",
        ("string", "ab:3.0"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "for-map-destructure",
        """obj := { "a": 1, "b": 2 }
keys := ""
sum := 0
for k, v in obj: { keys = keys + k; sum = sum + v }
keys + ":" + ("" + sum)""",
        ("string", "ab:3.0"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "for-hoist-index",
        """for[^idx] [10, 20]: idx = idx
idx""",
        ("number", 1),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "for-break",
        """sum := 0
for x in [1, 2, 3]: { if x == 2: break; sum = sum + x }
sum""",
        ("number", 1),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "for-continue",
        """sum := 0
for x in [1, 2, 3]: { if x == 2: continue; sum = sum + x }
sum""",
        ("number", 4),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "fn-return-in-defer",
        """fn choose(): { defer finish: { return 2 }; return 1 }
choose()""",
        ("number", 2),
        None,
    )
)
runtime_scenario(lambda: _rt("stdlib-print", 'print(1, "a")', ("null", None), None))
runtime_scenario(
    lambda: _rt(
        "defer-runs",
        """flag := 0
fn setFlag(): { flag = 2 }
fn run(): { defer cleanup: setFlag() ; flag = 1 }
run()
flag""",
        ("number", 2),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "defer-after-order",
        """log := ""
fn run(): { defer second after first: { log = log + "2" }; defer first: { log = log + "1" } }
run()
log""",
        ("string", "12"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "defer-simplecall-after",
        """log := ""
fn push(ch): { log = log + ch }
fn run(): { defer cleanup: { push("1") }; defer push("2") after cleanup }
run()
log""",
        ("string", "12"),
        None,
    )
)
runtime_scenario(lambda: _rt("defer-unknown-handle", 'defer cleanup: pass', None, ShakarRuntimeError))
runtime_scenario(
    lambda: _rt(
        "defer-cycle-detected",
        """log := []
fn run(): { defer second after first: pass; defer first after second: pass }
run()""",
        None,
        ShakarRuntimeError,
    )
)
runtime_scenario(
    lambda: _rt(
        "defer-duplicate-handle",
        """fn run(): { defer tag: pass; defer tag: pass }
run()""",
        None,
        ShakarRuntimeError,
    )
)
runtime_scenario(lambda: _rt("assert-pass", 'assert 1 == 1', ("null", None), None))
runtime_scenario(lambda: _rt("assert-fail", 'assert false, "boom"', None, ShakarAssertionError))
runtime_scenario(lambda: _rt("throw-new-error", 'throw error("boom")', None, ShakarRuntimeError))
runtime_scenario(
    lambda: _rt(
        "throw-custom-catch",
        """fn risky(): { throw error("ValidationError", "bad", 123) }
value := (risky() catch err: err.message)
value""",
        ("string", "bad"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "throw-custom-guard-miss",
        """fn risky(): { throw error("TypeError", "bad") }
risky() catch (ValidationError): 1""",
        None,
        ShakarRuntimeError,
    )
)
runtime_scenario(
    lambda: _rt(
        "throw-rethrow",
        """fn risky(): { throw error("TypeError", "bad") }
risky() catch err: throw""",
        None,
        ShakarRuntimeError,
    )
)
runtime_scenario(lambda: _rt("return-outside-fn", 'return 1', None, ShakarRuntimeError))
runtime_scenario(lambda: _rt("lambda-subject-missing-arg", 'a := &(.trim()); a()', None, ShakarArityError))
runtime_scenario(
    lambda: _rt(
        "lambda-hole-runtime",
        """fn blend(a, b, ratio): a + (b - a) * ratio
partial := blend(?, ?, 0.25)
partial(0, 16)""",
        ("number", 4),
        None,
    )
)
runtime_scenario(lambda: _rt("lambda-hole-iifc", 'blend(?, ?, 0.25)()', None, SyntaxError))

# ---------------------------------------------------------------------------
# Suite execution
# ---------------------------------------------------------------------------

class CaseRunner:
    """Runs parser-only cases across both start symbols and captures errors."""
    def __init__(self, parsers: ParserBundle):
        self.parsers = parsers

    def run(self, case: Case) -> Tuple[bool, List[str]]:
        """Return success flag + per-mode error strings if any."""
        modes = []
        if case.start in {"both", "noindent"}:
            modes.append("noindent")
        if case.start in {"both", "indented"}:
            modes.append("indented")
        ok = True
        errors: List[str] = []
        for mode in modes:
            success, message = self.parsers.parse(mode, case.code)
            if not success:
                ok = False
                if message:
                    errors.append(message.replace("\n", "\\n"))
        return ok, errors

class SanitySuite:
    """Coordinates parser, AST, and runtime checks then emits a text report."""
    def __init__(self, plan: KeywordPlan, filters: Optional[Sequence[str]] = None):
        self.plan = plan
        self.filters = list(filters) if filters else []
        self.parsers = ParserBundle(GRAMMAR_TEXT)
        self.case_runner = CaseRunner(self.parsers)
        self.ast_parser = runner
        self.cases = self._build_cases()

    def _matches_filter(self, name: str) -> bool:
        if not self.filters:
            return True
        return any(token in name for token in self.filters)

    def _build_cases(self) -> List[Case]:
        cases: List[Case] = []
        for builder in CASE_BUILDERS:
            cases.extend(builder(self.plan))
        seen = set()
        deduped: List[Case] = []
        # Keyword prefix builders can collide (same snippet under different keywords),
        # so keep only the first occurrence to avoid redundant parser work.
        for case in cases:
            key = (case.name, case.code, case.start)
            if key not in seen:
                seen.add(key)
                if self._matches_filter(case.name):
                    deduped.append(case)
        return deduped

    def execute(self) -> Tuple[str, int]:
        lines: List[str] = []
        total = 0
        failed = 0
        lines.extend(self.plan.notes)
        # parser-only cases
        for case in self.cases:
            total += 1
            ok, errs = self.case_runner.run(case)
            status = "PASS" if ok else "FAIL"
            if not ok:
                failed += 1
            lines.append(f"[{status}] {case.name}: {case.code!r}")
            for err in errs:
                lines.append(f"    {err}")
        # ast transform checks
        amp_lines, amp_total, amp_failed = self._run_ast_scenarios()
        if amp_lines:
            lines.append("")
            lines.extend(amp_lines)
        total += amp_total
        failed += amp_failed
        # interpreter runs
        runtime_lines, runtime_total, runtime_failed = self._run_runtime_scenarios()
        lines.extend("")
        lines.extend(runtime_lines)
        total += runtime_total
        failed += runtime_failed
        lines.append("")
        lines.append(f"Total cases: {total}")
        lines.append(f"Failures: {failed}")
        return "\n".join(lines), failed

    def _parse_ast(self, source: str):
        return self.ast_parser.parse_to_ast(source, GRAMMAR_TEXT)

    def _run_ast_scenarios(self) -> Tuple[List[str], int, int]:
        lines: List[str] = []
        total = 0
        failed = 0
        for scenario in AST_SCENARIOS:
            if not self._matches_filter(scenario.name):
                continue
            total += 1
            try:
                ast = self._parse_ast(scenario.code)
                if scenario.expected_exc is not None:
                    lines.append(f"[FAIL] {scenario.name}: expected {scenario.expected_exc.__name__}")
                    failed += 1
                    continue
                error = scenario.checker(ast) if scenario.checker else None
                if error:
                    lines.append(f"[FAIL] {scenario.name}: {error}")
                    failed += 1
                else:
                    lines.append(f"[PASS] {scenario.name}: {scenario.code!r}")
            except Exception as exc:
                if scenario.expected_exc and isinstance(exc, scenario.expected_exc):
                    lines.append(f"[PASS] {scenario.name}: raised {scenario.expected_exc.__name__}")
                else:
                    lines.append(f"[FAIL] {scenario.name}: {exc}")
                    failed += 1
        return lines, total, failed

    def _run_runtime_scenarios(self) -> Tuple[List[str], int, int]:
        lines: List[str] = []
        total = 0
        failed = 0
        for scenario in RUNTIME_SCENARIOS:
            if not self._matches_filter(scenario.name):
                continue
            total += 1
            try:
                result = run_program(scenario.source)
                if scenario.expected_exc is not None:
                    lines.append(f"[FAIL] {scenario.name}: expected {scenario.expected_exc.__name__}")
                    failed += 1
                    continue
                err = self._verify_runtime_result(result, scenario.expectation)
                if err:
                    lines.append(f"[FAIL] {scenario.name}: {err}")
                    failed += 1
                else:
                    lines.append(self._pass_line(scenario.name, result))
            except Exception as exc:
                if scenario.expected_exc and isinstance(exc, scenario.expected_exc):
                    lines.append(f"[PASS] {scenario.name}: raised {scenario.expected_exc.__name__}")
                else:
                    lines.append(f"[FAIL] {scenario.name}: {exc}")
                    failed += 1
        return lines, total, failed

    def _verify_runtime_result(self, value: object, expectation: Optional[Tuple[str, object]]) -> Optional[str]:
        if expectation is None:
            return None
        kind, expected = expectation
        if kind == "string":
            if not isinstance(value, ShkString):
                return f"expected ShkString, got {type(value).__name__}"
            if value.value != expected:
                return f"expected {expected!r}, got {value.value!r}"
            return None
        if kind == "number":
            if not isinstance(value, ShkNumber):
                return f"expected number, got {type(value).__name__}"
            if abs(value.value - float(expected)) > 1e-9:
                return f"expected {expected}, got {value.value}"
            return None
        if kind == "bool":
            if not isinstance(value, ShkBool):
                return f"expected bool, got {type(value).__name__}"
            if bool(value.value) != bool(expected):
                return f"expected {expected}, got {value.value}"
            return None
        if kind == "null":
            if not isinstance(value, ShkNull):
                return f"expected ShkNull, got {type(value).__name__}"
            return None
        if kind == "command":
            if not isinstance(value, ShkCommand):
                return f"expected ShkCommand, got {type(value).__name__}"
            rendered = value.render()
            if rendered != expected:
                return f"expected {expected!r}, got {rendered!r}"
            return None
        return f"unknown expectation kind {kind}"

    def _pass_line(self, name: str, value: object) -> str:
        if isinstance(value, ShkString):
            desc = f"produced {value.value!r}"
        elif isinstance(value, ShkNumber):
            desc = f"produced {int(value.value) if value.value.is_integer() else value.value}"
        elif isinstance(value, ShkNull):
            desc = "produced null"
        elif isinstance(value, ShkCommand):
            desc = f"produced sh<{value.render()}>"
        else:
            desc = f"produced {value}"
        return f"[PASS] {name}: {desc}"

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def _selected_filters(argv: Sequence[str]) -> List[str]:
    env = os.getenv("SANITY_FILTER")
    filters: List[str] = []
    if env:
        filters.extend([tok for tok in env.split(",") if tok.strip()])
    if argv:
        filters.extend(list(argv))
    return filters

def run(argv: Sequence[str] = ()) -> Tuple[str, int]:
    plan = build_keyword_plan()
    suite = SanitySuite(plan, filters=_selected_filters(argv))
    return suite.execute()

if __name__ == "__main__":
    report, failures = run(sys.argv[1:])
    out_path = Path("sanity_report.txt")
    out_path.write_text(report, encoding="utf-8")
    print(report)
    if failures:
        raise SystemExit(1)
r_runtime_ops = [
    ("compound-mod", 'a := 10; a %= 3; a', ("number", 1)),
    ("compound-minus", 'a := 10; a -= 4; a', ("number", 6)),
    ("compound-mul", 'a := 3; a *= 4; a', ("number", 12)),
    ("compound-div", 'a := 9; a /= 2; a', ("number", 4.5)),
    ("compound-floordiv", 'a := 9; a //= 2; a', ("number", 4)),
]
for name, source, expect in r_runtime_ops:
    runtime_scenario(lambda n=name, src=source, exp=expect: _rt(n, src, exp, None))

# runtime_scenario(
#     lambda: _rt(
#         "nullsafe-chain",
#         """arr := nil
# fallback := [1, 2]
# value := arr ??(arr[0]) ?? fallback[1]
# value""",
#         ("number", 2),
#         None,
