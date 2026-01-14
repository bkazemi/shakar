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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = (BASE_DIR / "src").resolve()


def _ensure_src_on_path() -> None:
    if str(BASE_DIR) not in sys.path:
        sys.path.append(str(BASE_DIR))
    if str(SRC_DIR) not in sys.path:
        sys.path.append(str(SRC_DIR))


def _load_shakar_modules():
    _ensure_src_on_path()

    from shakar_ref.runner import run as run_program
    from shakar_ref.runtime import (
        CommandError,
        ShkCommand,
        ShkNull,
        ShkNumber,
        ShkString,
        ShkBool,
        ShkArray,
        ShkDuration,
        ShkSize,
        ShakarArityError,
        ShakarAssertionError,
        ShakarRuntimeError,
        ShakarTypeError,
    )

    # RD parser only (Lark deprecated)
    from shakar_ref.parser_rd import parse_source as parse_rd, ParseError
    from shakar_ref.lexer_rd import LexError, Lexer
    from shakar_ref.ast_transforms import Prune
    from shakar_ref.lower import lower

    return (
        run_program,
        CommandError,
        ShkCommand,
        ShkNull,
        ShkNumber,
        ShkString,
        ShkBool,
        ShkArray,
        ShkDuration,
        ShkSize,
        ShakarArityError,
        ShakarAssertionError,
        ShakarRuntimeError,
        ShakarTypeError,
        parse_rd,
        ParseError,
        LexError,
        Lexer,
        Prune,
        lower,
    )


(
    run_program,
    CommandError,
    ShkCommand,
    ShkNull,
    ShkNumber,
    ShkString,
    ShkBool,
    ShkArray,
    ShkDuration,
    ShkSize,
    ShakarArityError,
    ShakarAssertionError,
    ShakarRuntimeError,
    ShakarTypeError,
    parse_rd,
    ParseError,
    LexError,
    Lexer,
    Prune,
    lower,
) = _load_shakar_modules()

# Get KEYWORDS for test generation
KEYWORDS = Lexer.KEYWORDS

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


# ---------------------------------------------------------------------------
# Parser management
# ---------------------------------------------------------------------------


class ParserBundle:
    """Simplified parser runner using RD parser only."""

    def __init__(self, grammar_text: str):
        pass  # No initialization needed for RD-only mode

    def parse(self, start: str, code: str) -> Tuple[bool, Optional[str]]:
        """Parse using the requested start symbol, returning (ok, error_message)."""
        label = f"start_{start}"
        use_indenter = start == "indented"

        try:
            rd_tree = parse_rd(code, use_indenter=use_indenter)
            rd_tree = Prune().transform(rd_tree)
            rd_tree = lower(rd_tree)
            return True, None
        except (ParseError, LexError) as exc:
            return False, f"{label} (RD): {exc}"
        except Exception as exc:
            return False, f"{label} (RD): {exc}"


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


def case_builder(
    func: Callable[[KeywordPlan], List[Case]],
) -> Callable[[KeywordPlan], List[Case]]:
    CASE_BUILDERS.append(func)
    return func


def _identifier_variants(keyword: str, limit: int) -> List[str]:
    ids = [f"{keyword}{suffix}" for suffix in KEYWORD_SUFFIXES]
    ids += [f"{prefix}{keyword}" for prefix in KEYWORD_PREFIXES]
    return ids[:limit]


@case_builder
def build_keyword_cases(plan: KeywordPlan) -> List[Case]:
    cases: List[Case] = []
    templates = PREFIX_SNIPPET_TEMPLATES[: plan.variants]
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
    return [
        Case(name=f"op-{i}", code=src, start="both") for i, src in enumerate(samples)
    ]


@case_builder
def build_noanchor_segment_cases(_: KeywordPlan) -> List[Case]:
    samples = [
        "state.$lines",
        "arr$[idx]",
        "obj.$method()",
        "state.foo.$bar.baz",
        ".$field and .other",
    ]
    return [
        Case(name=f"noanchor-seg-{i}", code=src, start="both")
        for i, src in enumerate(samples)
    ]


@case_builder
def build_postfix_if_cases(_: KeywordPlan) -> List[Case]:
    samples = ["1 if 0", "foo() if bar", "(a+b) if c", "x.y if z"]
    return [
        Case(name=f"postfixif-{i}", code=src, start="both")
        for i, src in enumerate(samples)
    ]


@case_builder
def build_await_cases(_: KeywordPlan) -> List[Case]:
    samples = [
        "await f()",
        "x = await g(1,2)",
        "h(await k())",
    ]
    return [
        Case(name=f"await-{i}", code=src, start="both") for i, src in enumerate(samples)
    ]


@case_builder
def build_block_cases(_: KeywordPlan) -> List[Case]:
    samples = [
        "if 1:\n  a\n  b\n",
        "if a:\n  b\nelif c:\n  d\nelse:\n  e\n",
        "if a:\n  if b:\n    c\n  else:\n    d\n",
    ]
    return [
        Case(name=f"block-{i}", code=src, start="indented")
        for i, src in enumerate(samples)
    ]


@case_builder
def build_misc_cases(_: KeywordPlan) -> List[Case]:
    samples = [
        "a.b[c](d, e).f",
        "a.b; c.d; e",
        "x = (a.b + c[d]) * e.f(g)",
    ]
    return [
        Case(name=f"misc-{i}", code=src, start="both") for i, src in enumerate(samples)
    ]


@case_builder
def build_destructure_inline_cases(_: KeywordPlan) -> List[Case]:
    samples = [
        "if true: a, b := 1, 2",
    ]
    return [
        Case(name="inline-destructure-walrus", code=src, start="both")
        for src in samples
    ]


@case_builder
def build_fn_cases(_: KeywordPlan) -> List[Case]:
    samples = [
        "fn add(x, y): x + y",
        "fn greet(name): { dbg(name) }",
    ]
    return [
        Case(name=f"fn-{i}", code=src, start="both") for i, src in enumerate(samples)
    ]


@case_builder
def build_param_contract_cases(_: KeywordPlan) -> List[Case]:
    samples = [
        "fn f(a, b, c ~ Int): a",
        "fn f(a, b, (c ~ Int)): a",
        "fn f((a), b, c ~ Int): a",
        "fn f(a, (b ~ Str), c ~ Int): a",
        "fn f((a, b) ~ Int, c ~ Str): a",
        "fn f(a, b, ...rest ~ Int): rest",
        "&[a, b ~ Int](a + b)",
    ]
    return [
        Case(name=f"param-contract-{i}", code=src, start="both")
        for i, src in enumerate(samples)
    ]


@case_builder
def build_power_cases(_: KeywordPlan) -> List[Case]:
    samples = [
        "2 ** 3",
        "x ** 2 + y ** 2",
        "a ** b ** c",
    ]
    return [
        Case(name=f"power-{i}", code=src, start="both") for i, src in enumerate(samples)
    ]


@case_builder
def build_unary_incr_cases(_: KeywordPlan) -> List[Case]:
    samples = [
        "++x",
        "--y",
        "++(a.b)",
    ]
    return [
        Case(name=f"unary-incr-{i}", code=src, start="both")
        for i, src in enumerate(samples)
    ]


@case_builder
def build_postfix_incr_cases(_: KeywordPlan) -> List[Case]:
    samples = [
        "x++",
        "y--",
        "arr[i]++",
        "obj.count--",
    ]
    return [
        Case(name=f"postfix-incr-{i}", code=src, start="both")
        for i, src in enumerate(samples)
    ]


@case_builder
def build_valuefan_cases(_: KeywordPlan) -> List[Case]:
    samples = [
        "state.{a, b}",
        "obj.{x(), y}",
        "[state.{a, b, c}]",
    ]
    return [
        Case(name=f"valuefan-{i}", code=src, start="both")
        for i, src in enumerate(samples)
    ]


@case_builder
def build_dbg_cases(_: KeywordPlan) -> List[Case]:
    samples = [
        "dbg(x)",
    ]
    return [
        Case(name=f"dbg-{i}", code=src, start="both") for i, src in enumerate(samples)
    ]


@case_builder
def build_computed_key_cases(_: KeywordPlan) -> List[Case]:
    samples = [
        '{ ("key" + "1"): 10 }',
        "{ (a + b): value }",
    ]
    return [
        Case(name=f"computed-key-{i}", code=src, start="both")
        for i, src in enumerate(samples)
    ]


@case_builder
def build_anon_fn_expr_cases(_: KeywordPlan) -> List[Case]:
    samples = [
        "x := fn(): 42",
        "arr.map(fn(x): x + 1)",
        "result := fn(()): { tmp := 1; tmp + 2 }",
    ]
    return [
        Case(name=f"anon-fn-expr-{i}", code=src, start="both")
        for i, src in enumerate(samples)
    ]


@case_builder
def build_pattern_destructure_cases(_: KeywordPlan) -> List[Case]:
    samples = [
        "a, b = 1, 2",
        "a, b := get_pair()",
        "x, y, z := 1, 2, 3",
    ]
    return [
        Case(name=f"pattern-destructure-{i}", code=src, start="both")
        for i, src in enumerate(samples)
    ]


@case_builder
def build_deepmerge_cases(_: KeywordPlan) -> List[Case]:
    samples = [
        "a +> b",
        "obj1 +> obj2 +> obj3",
    ]
    return [
        Case(name=f"deepmerge-{i}", code=src, start="both")
        for i, src in enumerate(samples)
    ]


@case_builder
def build_assignor_cases(_: KeywordPlan) -> List[Case]:
    samples = [
        "config.key or= default",
        "obj.field or= fallback",
    ]
    return [
        Case(name=f"assignor-{i}", code=src, start="both")
        for i, src in enumerate(samples)
    ]


@case_builder
def build_slice_cases(_: KeywordPlan) -> List[Case]:
    samples = [
        "arr[1:3]",
        "arr[:5]",
        "arr[2:]",
        "arr[::2]",
        "arr[1:3:2]",
    ]
    return [
        Case(name=f"slice-{i}", code=src, start="both") for i, src in enumerate(samples)
    ]


@case_builder
def build_comp_for_cases(_: KeywordPlan) -> List[Case]:
    samples = [
        "[x for x in arr]",
        "[x + 1 for x in nums if x > 0]",
        "set{ x for x in vals }",
        "{ k: v for [k, v] items }",
    ]
    return [
        Case(name=f"comp-for-{i}", code=src, start="both")
        for i, src in enumerate(samples)
    ]


@case_builder
def build_nullsafe_cases(_: KeywordPlan) -> List[Case]:
    samples = [
        "??(x)",
        "??(arr[0])",
        "??(obj.field)",
    ]
    return [
        Case(name=f"nullsafe-{i}", code=src, start="both")
        for i, src in enumerate(samples)
    ]


@case_builder
def build_postfix_unless_cases(_: KeywordPlan) -> List[Case]:
    samples = [
        "x = 1 unless false",
        "return 5 unless cond",
    ]
    return [
        Case(name=f"postfix-unless-{i}", code=src, start="both")
        for i, src in enumerate(samples)
    ]


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
    body = next(
        (
            ch
            for ch in deco.children
            if getattr(ch, "data", None) in {"inlinebody", "indentblock"}
        ),
        None,
    )
    if body is None:
        return "decorator body missing"
    return None


def _check_decorated_fn(ast) -> Optional[str]:
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


AST_SCENARIOS.extend(
    [
        AstScenario(
            "lambda-infer-zipwith",
            "zipWith&(left + right)(xs, ys)",
            _check_zipwith,
            None,
        ),
        AstScenario("lambda-respect-subject", "map&(.trim())", _check_map, None),
        AstScenario("lambda-hole-desugar", "blend(?, ?, 0.25)", _check_holes, None),
        AstScenario("lambda-dot-mix-error", "map&(value + .trim())", None, SyntaxError),
        AstScenario(
            "hook-inline-body", 'hook "warn": .trim()', _check_hook_inline, None
        ),
        AstScenario(
            "decorator-ast-def",
            "decorator logger(msg): args",
            _check_decorator_def,
            None,
        ),
        AstScenario(
            "decorated-fn-ast",
            """@noop
fn hi(): 1""",
            _check_decorated_fn,
            None,
        ),
        AstScenario("emit-outside-call", "> 1", None, ParseError),
        AstScenario(
            "param-group-nested-contract",
            "fn f((a ~ Int, b) ~ Int): a",
            None,
            ParseError,
        ),
    ]
)

# Runtime scenarios


def _rt(
    name: str,
    source: str,
    expectation: Optional[Tuple[str, object]],
    expected_exc: Optional[type],
) -> RuntimeScenario:
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
        "param-contract-grouped-ok",
        """fn f(a, b ~ Int): a + b
f(1, 2)""",
        ("number", 3),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "param-contract-grouped-fail",
        """fn f(a, b ~ Int): a + b
f("x", 2)""",
        None,
        ShakarAssertionError,
    )
)
runtime_scenario(
    lambda: _rt(
        "param-contract-isolated-inner",
        """fn f(a, (b ~ Int)): a
f("x", 2)""",
        ("string", "x"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "param-contract-isolated-bare",
        """fn f((a), b ~ Int): a
f("x", 2)""",
        ("string", "x"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "param-contract-spread",
        """fn f(...rest ~ Int): rest.len
f(1, "x", 3)""",
        None,
        ShakarAssertionError,
    )
)
runtime_scenario(
    lambda: _rt(
        "param-default-basic",
        """fn f(a = 1, b = 2): a + b
f()""",
        ("number", 3),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "param-default-partial",
        """fn f(a, b = 2): a + b
f(1)""",
        ("number", 3),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "amp-lambda-contract-ok",
        """f := &[a, b ~ Int](a + b)
f(1, 2)""",
        ("number", 3),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "amp-lambda-contract-fail",
        """f := &[a, b ~ Int](a + b)
f("x", 2)""",
        None,
        ShakarAssertionError,
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
        "call-basic-emit",
        """count := 0
fn bump(x): count += x
call bump:
  > 1
  > 2
count""",
        ("number", 3),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "call-emit-capture",
        """log := ""
fn out(x): log += x
fn wrap(x): log += "[" + x + "]"
call out:
  helper := fn(): > "a"
  call wrap:
    > "b"
    helper()
log""",
        ("string", "[b]a"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "call-emit-postfix-unless",
        """log := ""
fn emit(x): log += x
call emit:
  > "a" unless false
  > "b" unless true
log""",
        ("string", "a"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "call-emit-named-args",
        """log := ""
fn emit(a, b): log += a + ":" + b
call emit:
  > a: "x", b: "y"
log""",
        ("string", "x:y"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "call-emit-spread",
        """count := 0
fn emit(...args): count += args.len
call emit:
  > ...[1, 2, 3]
count""",
        ("number", 3),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "call-nested-shadow",
        """log := ""
fn outer(x): log += "o" + x
fn inner(x): log += "i" + x
call outer:
  > "1"
  call inner:
    > "2"
  > "3"
log""",
        ("string", "o1i2o3"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "call-non-callable-error",
        """x := 42
call x:
  > 1""",
        None,
        ShakarRuntimeError,
    )
)
runtime_scenario(
    lambda: _rt(
        "call-emit-trailing-comma",
        """log := ""
fn emit(a, b): log += a + b
call emit:
  > "x", "y",
log""",
        ("string", "xy"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "noanchor-segment-field",
        """state := { lines: 6, level: 2 }
state.$lines >= .level * 3""",
        ("bool", True),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "noanchor-segment-index",
        """arr := [{x: 1}, {x: 2}]
arr$[0].x + .len""",
        ("number", 3),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "noanchor-segment-multiple-error",
        "state.$foo.$bar",
        None,
        ParseError,
    )
)
runtime_scenario(
    lambda: _rt(
        "noanchor-segment-in-noanchor-error",
        "$state.$lines",
        None,
        SyntaxError,
    )
)
runtime_scenario(
    lambda: _rt(
        "noanchor-segment-no-leak",
        """state := { lines: 6 }
x := state.$lines
obj := { val: 10 }
obj and .val""",
        ("number", 10),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "noanchor-segment-no-leak-grouped",
        """state := { lines: 6, val: 99 }
obj := { val: 10 }
(state.$lines) and obj and .val""",
        ("number", 10),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "noanchor-segment-no-leak-call-arg",
        """state := { lines: 6, val: 99 }
fn make(x): ({ val: x })
make(state.$lines) and .val""",
        ("number", 6),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "noanchor-segment-no-leak-array-literal",
        """state := { lines: 6 }
[state.$lines, 1] and .len""",
        ("number", 2),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "noanchor-segment-no-leak-object-literal",
        """state := { lines: 6, val: 99 }
{ val: 9, x: state.$lines } and .val""",
        ("number", 9),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "noanchor-segment-no-leak-nullish",
        """state := { obj: { val: 5 }, val: 99 }
state.$obj ?? { val: 0 } and .val""",
        ("number", 5),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "noanchor-segment-no-leak-ternary",
        """state := { obj: { val: 5 }, val: 99 }
[true ? state.$obj : { val: 0 }] and .len""",
        ("number", 1),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "noanchor-segment-no-leak-selector",
        """state := { start: 1, val: 99 }
stop := 3
arr := [1,2,3,4,5,6,7]
arr[`{state.$start}:{stop}`] and .len""",
        ("number", 3),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "noanchor-segment-postfix-incr",
        """obj := { count: 5, val: 10 }
obj.$count++ and .val""",
        ("number", 10),
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
        'int(3) + int("4")',
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
runtime_scenario(
    lambda: _rt(
        "array-push",
        "arr := [1]; arr.push(2); arr.len",
        ("number", 2),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "array-append",
        "arr := [1]; arr.append(2); arr[1]",
        ("number", 2),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "array-pop",
        "arr := [1, 2, 3]; arr.pop() + arr.len",
        ("number", 5),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "array-pop-empty",
        "[].pop()",
        None,
        ShakarRuntimeError,
    )
)
runtime_scenario(
    lambda: _rt(
        "array-high",
        """xs := [10, 20, 30]
xs.high""",
        ("number", 2),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "array-high-empty",
        " [].high",
        ("number", -1),
        None,
    )
)
runtime_scenario(
    lambda: _rt("join-array", '", ".join(["a", "b"])', ("string", "a, b"), None)
)
runtime_scenario(
    lambda: _rt("join-varargs", '"-".join("a", "b")', ("string", "a-b"), None)
)
runtime_scenario(
    lambda: _rt("join-mixed", '"|".join("a", 1, true)', ("string", "a|1|true"), None)
)
runtime_scenario(
    lambda: _rt(
        "string-split",
        '"a,b,c".split(",")',
        ("array", ["a", "b", "c"]),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "string-split-empty",
        '"abc".split("")',
        None,
        ShakarRuntimeError,
    )
)
runtime_scenario(
    lambda: _rt(
        "regex-match-captures",
        """date := "2024-01-02"
year, month, day := date ~~ r"(\\d{4})-(\\d{2})-(\\d{2})"
[year, month, day]""",
        ("array", ["2024", "01", "02"]),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "regex-match-full-flag",
        """s := "ab"
full, first := s ~~ r"(a)b"/f
[full, first]""",
        ("array", ["ab", "a"]),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "regex-invalid-flag",
        'r"foo"/z',
        None,
        LexError,
    )
)
runtime_scenario(
    lambda: _rt(
        "regex-methods",
        """rx := r"(\\d+)"
ok := rx.test("a1")
match := rx.match("b22")
repl := rx.replace("c3", "x")
if ok and match:
  [match[0], repl]
else:
  ["", ""]""",
        ("array", ["22", "cx"]),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "string-high",
        '"hello".high',
        ("number", 4),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "string-high-empty",
        ' "".high',
        ("number", -1),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "object-len",
        "obj := {a: 1, b: 2}; obj.len",
        ("number", 2),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "object-keys",
        "{a: 1, b: 2}.keys()",
        ("array", ["a", "b"]),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "object-values",
        "{a: 1, b: 2}.values()",
        ("array", [1, 2]),
        None,
    )
)
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
runtime_scenario(
    lambda: _rt(
        "lambda-subject-direct-call", 'a := &(.trim()); a(" B")', ("string", "B"), None
    )
)
runtime_scenario(lambda: _rt("floor-div-basic", "7 // 2", ("number", 3), None))
runtime_scenario(lambda: _rt("floor-div-negative", "-7 // 2", ("number", -4), None))
runtime_scenario(
    lambda: _rt("compound-assign-number", "a := 1; a += 2; a", ("number", 3), None)
)
runtime_scenario(
    lambda: _rt(
        "compound-assign-string", 's := "a"; s += "b"; s', ("string", "ab"), None
    )
)
runtime_scenario(
    lambda: _rt("compound-assign-mod", "a := 10; a %= 3; a", ("number", 1), None)
)
runtime_scenario(
    lambda: _rt("compound-assign-minus", "a := 10; a -= 4; a", ("number", 6), None)
)
runtime_scenario(
    lambda: _rt("compound-assign-mul", "a := 3; a *= 4; a", ("number", 12), None)
)
runtime_scenario(
    lambda: _rt("compound-assign-div", "a := 9; a /= 2; a", ("number", 4.5), None)
)
runtime_scenario(
    lambda: _rt("compound-assign-floordiv", "a := 9; a //= 2; a", ("number", 4), None)
)
runtime_scenario(
    lambda: _rt(
        "fanout-block-basic",
        "state := {cur: 1, next: 2, x: 0}; state{ .cur = .next; .x += 5 }; state.cur + state.x",
        ("number", 7),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "fanout-block-indented",
        "state := {cur: 1, next: 2, x: 0}\n"
        "state{\n"
        "  .cur = .next\n"
        "  .x += 5\n"
        "}\n"
        "state.cur + state.x",
        ("number", 7),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "fieldfan-chain-assign",
        "state := {a: {c: 0}, b: {c: 1}}; state.{a, b}.c = 5; state.a.c + state.b.c",
        ("number", 10),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "fieldfan-chain-apply",
        "state := {a: {c: 1}, b: {c: 3}}; state.{a, b}.c .= . + 1; state.a.c + state.b.c",
        ("number", 6),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "selector-assign-broadcast",
        "arr := [0, 1]; o := {a: arr}; o.a[0,1] = 2; o.a[0] + o.a[1]",
        ("number", 4),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "selectorliteral-assign-broadcast",
        "arr := [0, 1, 2]; arr[`0:1`] = 5; arr[0] + arr[1] + arr[2]",
        ("number", 12),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "fanout-single-clause-implicit",
        "state := {cur: 1, next: 2}; state{ .cur = .next }; state.cur",
        ("number", 2),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "fanout-single-clause-literal-error",
        "state := {a: 1}; state{ .a = 5 }",
        None,
        ParseError,
    )
)
runtime_scenario(
    lambda: _rt(
        "slice-negative-start-positive-stop-error",
        "arr := [0, 1, 2]; arr[-1:2]",
        None,
        ShakarRuntimeError,
    )
)
runtime_scenario(
    lambda: _rt(
        "slice-positive-start-negative-stop",
        "arr := [0, 1, 2, 3, 4]; result := arr[0:-1]; result[0] + result[3]",
        ("number", 3),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "slice-both-negative",
        "arr := [0, 1, 2, 3, 4]; result := arr[-3:-1]; result[0] + result[1]",
        ("number", 5),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "fieldfan-chain-apply-return",
        "state := {a: {c: 1}, b: {c: 3}}; result := state.{a, b}.c .= . + 1; result[0] + result[1]",
        ("number", 6),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "fanout-block-slice-selector",
        "state := {rows: [{v: 1}, {v: 3}, {v: 5}]}; state{ .rows[1:3].v = 0 }; state.rows[0].v + state.rows[1].v + state.rows[2].v",
        ("number", 1),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "fanout-block-bracketed",
        "state := {rows: [{v: 1}, {v: 3}, {v: 5}]}; state{ .rows[1].v += 4; .rows[0] = {v: state.rows[0].v + 2} }; state.rows[0].v + state.rows[1].v",
        ("number", 10),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "fanout-block-multi-index-selector",
        "state := {rows: [[{v: 1}], [{v: 3}]]}; state{ .rows[1][0].v = 8 }; state.rows[0][0].v + state.rows[1][0].v",
        None,
        ParseError,
    )
)
runtime_scenario(
    lambda: _rt(
        "fanout-block-apply",
        's := {name: " Ada ", greet: ""}; s{ .name .= .trim(); .greet = .name }; s.greet',
        ("string", "Ada"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "fanout-block-dup-error",
        "state := {a: 1}; state{ .a = 1; .a = 2 }",
        None,
        ShakarRuntimeError,
    )
)
runtime_scenario(
    lambda: _rt(
        "fanout-value-array",
        "state := {a: 1, b: 2}; arr := state.{a, b}; arr[0] + arr[1]",
        ("number", 3),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "fanout-call-spread",
        "state := {a: 1, b: 2}; fn add(x, y): x + y; add(state.{a, b})",
        ("number", 3),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "fanout-value-call-item",
        "state := {a: fn():3, b: 2}; vals := state.{a(), b}; vals[0] + vals[1]",
        ("number", 5),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "fanout-named-arg-no-spread",
        "state := {a: 1, b: 2}; fn wrap(x): x[1]; wrap(named: state.{a, b})",
        ("number", 2),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "spread-array-literal",
        "arr := [1, ...[2, 3], 4]; arr[0] + arr[3]",
        ("number", 5),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "spread-object-literal",
        "base := {a: 1}; obj := { ...base, b: 2, a: 3 }; obj.a + obj.b",
        ("number", 5),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "spread-call-array",
        "fn add(a, b, c): a + b + c; add(...[1, 2, 3])",
        ("number", 6),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "spread-call-object",
        "fn pair(a, b): a * 10 + b; pair(...{a: 1, b: 2})",
        ("number", 12),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "spread-params-multi",
        "fn capture(a, ...mid, b, ...tail, c): [a, mid.len, b, tail.len, c]; res := capture(1, 2, 3, 4, 5, 6, 7); res[0] + res[1] + res[2] + res[3] + res[4]",
        None,
        SyntaxError,
    )
)
runtime_scenario(
    lambda: _rt(
        "spread-object-dot-space",
        "state := {config: {a: 1}, cfg: {}}; state{ .cfg = { ... .config } }; state.cfg.a",
        ("number", 1),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "spread-decorator-params",
        """decorator bump(...xs):
  args[0] = args[0] + xs.len
@bump(1, 2, 3)
fn id(x): x
id(10)""",
        ("number", 13),
        None,
    )
)
runtime_scenario(lambda: _rt("all-varargs", 'all(true, 1, "x")', ("bool", True), None))
runtime_scenario(
    lambda: _rt("all-varargs-short", "all(true, false, 1)", ("bool", False), None)
)
runtime_scenario(
    lambda: _rt("all-iterable", 'all([true, 1, "x"])', ("bool", True), None)
)
runtime_scenario(lambda: _rt("all-empty-iterable", "all([])", ("bool", True), None))
runtime_scenario(lambda: _rt("all-zero-args-error", "all()", None, ShakarRuntimeError))
runtime_scenario(lambda: _rt("any-varargs", 'any(false, 0, "x")', ("bool", True), None))
runtime_scenario(
    lambda: _rt("any-iterable", 'any([false, 0, ""])', ("bool", False), None)
)
runtime_scenario(
    lambda: _rt(
        "raw-string-basic", 'raw"hi {name}\\n"', ("string", "hi {name}\\n"), None
    )
)
runtime_scenario(
    lambda: _rt(
        "raw-hash-string",
        'raw#"path "C:\\\\tmp"\\file"#',
        ("string", 'path "C:\\\\tmp"\\file'),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "shell-string-quote",
        'path := "file name.txt"; sh"cat {path}"',
        ("command", "cat 'file name.txt'"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "shell-string-array",
        'files := ["a.txt", "b 1.txt"]; sh"ls {files}"',
        ("command", "ls a.txt 'b 1.txt'"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "shell-string-raw-splice",
        'flag := "-n 2"; file := "log 1.txt"; sh"head {{flag}} {file}"',
        ("command", "head -n 2 'log 1.txt'"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "shell-run-stdout",
        'msg := "hi"; res := (sh"printf {msg}").run(); res',
        ("string", "hi"),
        None,
    )
)
runtime_scenario(lambda: _rt("shell-run-code", '(sh"false").run()', None, CommandError))
runtime_scenario(
    lambda: _rt(
        "shell-run-catch-code",
        'val := (sh"false").run() catch err: err.code',
        ("number", 1),
        None,
    )
)
runtime_scenario(
    lambda: _rt("path-literal-exists", 'p"README.md".exists', ("bool", True), None)
)
runtime_scenario(
    lambda: _rt(
        "path-join-exists",
        '(p"docs" / "shakar-design-notes.md").exists',
        ("bool", True),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "path-interp-read",
        'name := "README.md"\n(p"{name}").read().len > 0',
        ("bool", True),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "path-glob-contains",
        'names := [ .name over p"*.md" ]\n"README.md" in names',
        ("bool", True),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "path-glob-empty",
        '[ .name over p"__shakar_no_match__*.zzz" ].len == 0',
        ("bool", True),
        None,
    )
)
runtime_scenario(
    lambda: _rt("base-prefix-binary", "0b1010_0011", ("number", 163), None)
)
runtime_scenario(lambda: _rt("base-prefix-octal", "0o755", ("number", 493), None))
runtime_scenario(
    lambda: _rt("base-prefix-hex", "0xdead_beef", ("number", 3735928559), None)
)
runtime_scenario(
    lambda: _rt(
        "base-prefix-binary-63bit",
        "0b111111111111111111111111111111111111111111111111111111111111111",
        ("number", 9223372036854775807),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "base-prefix-hex-max",
        "0x7fffffffffffffff",
        ("number", 9223372036854775807),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "base-prefix-hex-min-neg",
        "-0x8000000000000000",
        ("number", -9223372036854775808),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "decimal-int64-min-neg",
        "-9223372036854775808",
        ("number", -9223372036854775808),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "decimal-int64-overflow",
        "9223372036854775808",
        None,
        LexError,
    )
)
runtime_scenario(
    lambda: _rt(
        "base-prefix-binary-overflow",
        "0b1000000000000000000000000000000000000000000000000000000000000000",
        None,
        LexError,
    )
)
runtime_scenario(
    lambda: _rt(
        "base-prefix-octal-overflow",
        "0o1000000000000000000000",
        None,
        LexError,
    )
)
runtime_scenario(
    lambda: _rt("base-prefix-hex-overflow", "0x8000000000000000", None, LexError)
)
runtime_scenario(lambda: _rt("base-prefix-invalid-bin", "0b102", None, LexError))
runtime_scenario(lambda: _rt("base-prefix-invalid-oct", "0o9", None, LexError))
runtime_scenario(lambda: _rt("base-prefix-invalid-hex", "0xG", None, LexError))
runtime_scenario(lambda: _rt("base-prefix-incomplete", "0b", None, LexError))
runtime_scenario(lambda: _rt("base-prefix-uppercase", "0X10", None, LexError))
runtime_scenario(lambda: _rt("base-prefix-underscore-start", "0b_101", None, LexError))
runtime_scenario(lambda: _rt("base-prefix-underscore-end", "0b101_", None, LexError))
runtime_scenario(
    lambda: _rt("base-prefix-underscore-double", "0b10__01", None, LexError)
)
runtime_scenario(lambda: _rt("base-prefix-duration", "0x10ms", None, LexError))
runtime_scenario(lambda: _rt("base-prefix-duration-bin", "0b10s", None, LexError))
runtime_scenario(lambda: _rt("base-prefix-size", "0o755kb", None, LexError))
runtime_scenario(
    lambda: _rt(
        "base-prefix-duration-mul", "0x10 * 1msec", ("duration", 16000000), None
    )
)
runtime_scenario(
    lambda: _rt("duration-underscore", "1_000msec", ("duration", 1000000000), None)
)
runtime_scenario(
    lambda: _rt(
        "duration-compound-underscore",
        "1sec500_000usec",
        ("duration", 1500000000),
        None,
    )
)
runtime_scenario(lambda: _rt("size-underscore", "1_000kb", ("size", 1000000), None))
runtime_scenario(
    lambda: _rt("size-compound-underscore", "1mb500_000b", ("size", 1500000), None)
)

# Decimal underscore tests
runtime_scenario(lambda: _rt("decimal-underscore", "1_000", ("number", 1000), None))
runtime_scenario(
    lambda: _rt("decimal-underscore-multi", "1_000_000", ("number", 1000000), None)
)
runtime_scenario(lambda: _rt("decimal-underscore-trailing", "100_", None, LexError))
runtime_scenario(lambda: _rt("decimal-underscore-double", "1__0", None, LexError))
# Float underscore tests
runtime_scenario(
    lambda: _rt("float-underscore-int", "1_000.5", ("number", 1000.5), None)
)
runtime_scenario(
    lambda: _rt("float-underscore-frac", "1.000_001", ("number", 1.000001), None)
)
runtime_scenario(
    lambda: _rt("float-underscore-both", "1_000.000_5", ("number", 1000.0005), None)
)
runtime_scenario(lambda: _rt("float-underscore-leading-frac", "1._5", None, LexError))
runtime_scenario(lambda: _rt("float-underscore-trailing-frac", "1.5_", None, LexError))
# Float exponent tests
runtime_scenario(lambda: _rt("float-exponent", "1e10", ("number", 1e10), None))
runtime_scenario(lambda: _rt("float-exponent-upper", "1E10", ("number", 1e10), None))
runtime_scenario(lambda: _rt("float-exponent-neg", "1.5e-3", ("number", 1.5e-3), None))
runtime_scenario(lambda: _rt("float-exponent-pos", "1E+5", ("number", 1e5), None))
runtime_scenario(
    lambda: _rt("float-exponent-underscore", "1_000e2", ("number", 1000e2), None)
)
runtime_scenario(
    lambda: _rt("float-exponent-underscore-exp", "1e1_0", ("number", 1e10), None)
)
runtime_scenario(
    lambda: _rt("float-exponent-leading-underscore", "1e_5", None, LexError)
)
runtime_scenario(lambda: _rt("float-dot-no-frac", "1.e5", None, LexError))
runtime_scenario(lambda: _rt("float-trailing-dot", "1e5.", None, LexError))
runtime_scenario(
    lambda: _rt(
        "duration-total-nsec",
        "5min30sec.total_nsec",
        ("number", 330000000000),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "duration-unit-sec",
        "(1min + 30sec).sec",
        ("number", 90),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "size-total-bytes",
        "2gb500mb.total_bytes",
        ("number", 2500000000),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "size-unit-gb",
        "(1gb * 2).gb",
        ("number", 2),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "duration-compound-multi",
        "1hr30min15sec.sec",
        ("number", 5415),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "duration-div-ratio",
        "5sec / 2sec",
        ("number", 2.5),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "size-div-ratio",
        "6gb / 2gb",
        ("number", 3),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "duration-compare-gt",
        "5min > 3min",
        ("bool", True),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "duration-compare-lte",
        "2sec <= 2sec",
        ("bool", True),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "size-compare-lt",
        "1mb < 2mb",
        ("bool", True),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "size-compare-gte",
        "5gb >= 4gb",
        ("bool", True),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "duration-negate",
        "(-5sec).total_nsec",
        ("number", -5000000000),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "size-negate",
        "(-2kb).total_bytes",
        ("number", -2000),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "duration-sub",
        "(10sec - 3sec).sec",
        ("number", 7),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "size-sub",
        "(5mb - 2mb).mb",
        ("number", 3),
        None,
    )
)
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
runtime_scenario(
    lambda: _rt("fn-definition", "fn add(x, y): x + y; add(2, 3)", ("number", 5), None)
)
runtime_scenario(
    lambda: _rt("fn-closure", "y := 5; fn addY(x): x + y; addY(2)", ("number", 7), None)
)
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
runtime_scenario(
    lambda: _rt(
        "anon-fn-expression", "inc := fn(x): { x + 1 }; inc(5)", ("number", 6), None
    )
)
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
runtime_scenario(
    lambda: _rt(
        "await-any-trailing-body",
        "await [any]( fast: sleep(10), slow: sleep(50) ): winner",
        ("string", "fast"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "await-any-inline-body",
        'await [any]( fast: sleep(10): "done" )',
        ("string", "done"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "await-all-trailing-body",
        'await [all]( first: sleep(10), second: sleep(20) ): "ok"',
        ("string", "ok"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "hook-raw-string",
        'hook raw"event": 1',
        ("null", None),
        None,
    )
)
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
        ("string", "2"),
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
        "selector-literal-open-start",
        """sel := `:3`
arr := [10, 20, 30, 40]
picked := arr[sel]
picked[0] + picked[1] + picked[2]""",
        ("number", 60),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "selector-literal-open-stop",
        """sel := `2:`
arr := [1, 2, 3, 4, 5]
picked := arr[sel]
picked[0] + picked[1] + picked[2]""",
        ("number", 12),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "selector-literal-open-stop-neg-step",
        """sel := `5::-2`
arr := [0, 1, 2, 3, 4, 5]
picked := arr[sel]
picked.len""",
        # Clamped selector lists normalize open-stop negative steps to empty.
        ("number", 0),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "selector-literal-open-start-neg-step",
        """sel := `:1:-2`
arr := [0, 1, 2, 3, 4, 5]
picked := arr[sel]
picked[0] + picked[1] + picked[2]""",
        ("number", 9),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "selector-literal-slice-step",
        """sel := `1:7:2`
arr := [0, 1, 2, 3, 4, 5, 6, 7]
picked := arr[sel]
picked[0] + picked[1] + picked[2]""",
        ("number", 9),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "selector-literal-slice-step-interp-neg",
        """step := -1
sel := `5:1:{step}`
arr := [0, 1, 2, 3, 4, 5]
picked := arr[sel]
picked[0] + picked[1] + picked[2] + picked[3]""",
        ("number", 14),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "selector-literal-hex-overflow",
        "arr := [0]\narr[`0x8000000000000000`]",
        None,
        LexError,
    )
)
runtime_scenario(
    lambda: _rt(
        "selector-literal-dec-overflow",
        "arr := [0]\narr[`9223372036854775808`]",
        None,
        LexError,
    )
)
runtime_scenario(
    lambda: _rt(
        "selector-literal-slice-stop-overflow",
        "arr := [0]\narr[`0:0x8000000000000000`]",
        None,
        LexError,
    )
)
runtime_scenario(
    lambda: _rt(
        "selector-literal-slice-step-overflow",
        "arr := [0]\narr[`0:1:0x8000000000000000`]",
        None,
        LexError,
    )
)
runtime_scenario(
    lambda: _rt(
        "selector-literal-slice-step-neg-overflow",
        "arr := [0]\narr[`0:1:-0x8000000000000001`]",
        None,
        LexError,
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
        "compare-anchor-object",
        "a := {b: 1, c: 2}; a >= .b",
        None,
        ShakarTypeError,
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
        "guard-nullsafe-index",
        """fn rotate(shape):
  ??(!shape[0]): return []
  return shape

rotate([])""",
        None,
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
        ("string", "Ada|hi GRACE|42|24|10|18|GRACE"),
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
        "for-subject-bare-ident",
        """acc := ""
items := ["x", "y"]
for items: acc = acc + .
acc""",
        ("string", "xy"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "for-subject-number-int",
        """sum := 0
for 3: sum += .
sum""",
        ("number", 3),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "for-subject-number-float",
        """for 2.5: print(.)""",
        None,
        ShakarTypeError,
    )
)
runtime_scenario(
    lambda: _rt(
        "for-subject-number-negative",
        """for -2: print(.)""",
        None,
        ShakarTypeError,
    )
)
runtime_scenario(
    lambda: _rt(
        "for-subject-in-call-block",
        """log := ""
fn out(s): log = log + s
items := ["a", "b"]
call out:
  for items: > .
log""",
        ("string", "ab"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "await-bare-ident",
        """x := 99
await x""",
        ("number", 99),
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
        ("string", "01"),
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
        ("string", "ab:3"),
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
        ("string", "ab:3"),
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
runtime_scenario(
    lambda: _rt("defer-unknown-handle", "defer cleanup: pass", None, ShakarRuntimeError)
)
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
runtime_scenario(lambda: _rt("assert-pass", "assert 1 == 1", ("null", None), None))
runtime_scenario(
    lambda: _rt("assert-fail", 'assert false, "boom"', None, ShakarAssertionError)
)
runtime_scenario(
    lambda: _rt("throw-new-error", 'throw error("boom")', None, ShakarRuntimeError)
)
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
runtime_scenario(lambda: _rt("return-outside-fn", "return 1", None, ShakarRuntimeError))
runtime_scenario(
    lambda: _rt(
        "lambda-subject-missing-arg", "a := &(.trim()); a()", None, ShakarArityError
    )
)
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
runtime_scenario(
    lambda: _rt("lambda-hole-iifc", "blend(?, ?, 0.25)()", None, SyntaxError)
)
runtime_scenario(lambda: _rt("power-basic", "2 ** 3", ("number", 8), None))
runtime_scenario(lambda: _rt("power-precedence", "2 ** 3 ** 2", ("number", 512), None))
runtime_scenario(lambda: _rt("power-negative", "(-2) ** 3", ("number", -8), None))
runtime_scenario(lambda: _rt("power-assign", "x := 2; x **= 3; x", ("number", 8), None))
runtime_scenario(
    lambda: _rt("postfix-incr-basic", "a := 5; a++; a", ("number", 6), None)
)
runtime_scenario(
    lambda: _rt("postfix-decr-basic", "a := 5; a--; a", ("number", 4), None)
)
runtime_scenario(
    lambda: _rt("prefix-incr-basic", "a := 5; ++a; a", ("number", 6), None)
)
runtime_scenario(
    lambda: _rt("prefix-decr-basic", "a := 5; --a; a", ("number", 4), None)
)

# CCC disambiguation tests
runtime_scenario(
    lambda: _rt(
        "ccc-function-args", "fn f(a, b, c): a + b + c; f(1, 2, 3)", ("number", 6), None
    )
)
runtime_scenario(
    lambda: _rt(
        "ccc-array-elements", "a := [1, 2, 3]; a[0] + a[1] + a[2]", ("number", 6), None
    )
)
runtime_scenario(
    lambda: _rt(
        "ccc-array-with-parens", "a := [(5 == 5, == 5)]; a[0]", ("bool", True), None
    )
)
runtime_scenario(
    lambda: _rt("ccc-destructure-pack", "a, b := 10, 20; a + b", ("number", 30), None)
)
runtime_scenario(
    lambda: _rt(
        "ccc-statement-allowed", "x := 5; assert x == 5, < 10; x", ("number", 5), None
    )
)

# CCC in comprehensions - parser and evaluator now support it
runtime_scenario(
    lambda: _rt(
        "ccc-comprehension-filter",
        "data := [1, 5, 10]; [x for x in data if x == 1, < 6]",
        ("array", [1.0]),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "ccc-comprehension-explicit",
        "data := [1, 5, 10]; [x for x in data if x > 0, and < 8]",
        ("array", [1.0, 5.0]),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "ccc-comprehension-or",
        "data := [1, 5, 10]; [x for x in data if x == 1, or == 10]",
        ("array", [1.0, 10.0]),
        None,
    )
)

# Structural match (~) operator
runtime_scenario(lambda: _rt("struct-match-type-int", "5 ~ Int", ("bool", True), None))
runtime_scenario(
    lambda: _rt("struct-match-type-str", '"hello" ~ Str', ("bool", True), None)
)
runtime_scenario(
    lambda: _rt("struct-match-type-bool", "true ~ Bool", ("bool", True), None)
)
runtime_scenario(
    lambda: _rt("struct-match-type-array", "[1, 2] ~ Array", ("bool", True), None)
)
runtime_scenario(
    lambda: _rt("struct-match-type-object", "{a: 1} ~ Object", ("bool", True), None)
)
runtime_scenario(
    lambda: _rt(
        "struct-match-object-subset",
        "obj := {a: 1, b: 2, c: 3}; obj ~ {a: Int}",
        ("bool", True),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "struct-match-object-multi",
        "obj := {a: 1, b: 2}; obj ~ {a: Int, b: Int}",
        ("bool", True),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "struct-match-object-missing",
        "{a: 1} ~ {a: Int, b: Int}",
        ("bool", False),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "struct-match-nested",
        'u := {name: "Alice", profile: {role: "admin"}}; u ~ {profile: {role: Str}}',
        ("bool", True),
        None,
    )
)
runtime_scenario(lambda: _rt("struct-match-value", "5 ~ 5", ("bool", True), None))

# Type contracts in function parameters
runtime_scenario(
    lambda: _rt(
        "contract-int-valid",
        """fn add(a ~ Int, b ~ Int):
    a + b
add(1, 2)""",
        ("number", 3),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "contract-str-valid",
        """fn greet(name ~ Str):
    "Hello, " + name
greet("Bob")""",
        ("string", "Hello, Bob"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "contract-object-valid",
        """fn getname(u ~ {name: Str}):
    u.name
getname({name: "Alice", age: 30})""",
        ("string", "Alice"),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "contract-int-invalid",
        """fn add(a ~ Int):
    a
add("wrong")""",
        None,
        ShakarAssertionError,
    )
)
runtime_scenario(
    lambda: _rt(
        "contract-object-invalid",
        """fn f(u ~ {x: Int}):
    u
f({y: 1})""",
        None,
        ShakarAssertionError,
    )
)

# Ensure unary ~ is rejected (only binary ~ is supported)
runtime_scenario(lambda: _rt("unary-tilde-rejected", "~5", None, ParseError))

# Inline functions with contracts should work correctly
runtime_scenario(
    lambda: _rt(
        "contract-inline-fn", "fn inc(x ~ Int): x + 1; inc(5)", ("number", 6), None
    )
)
runtime_scenario(
    lambda: _rt(
        "contract-inline-fn-invalid",
        'fn inc(x ~ Int): x + 1; inc("bad")',
        None,
        ShakarAssertionError,
    )
)
runtime_scenario(
    lambda: _rt(
        "contract-inline-fn-defer",
        """fn test(x ~ Int):
    defer: x + 1
    x * 2
test(5)""",
        ("number", 10),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "contract-inline-multiple",
        "fn add(a ~ Int, b ~ Int): a + b; add(3, 4)",
        ("number", 7),
        None,
    )
)

# Optional fields with Optional() function
runtime_scenario(
    lambda: _rt(
        "optional-fn-missing-ok", "{} ~ {a: Optional(Int)}", ("bool", True), None
    )
)
runtime_scenario(
    lambda: _rt(
        "optional-fn-present-valid", "{a: 1} ~ {a: Optional(Int)}", ("bool", True), None
    )
)
runtime_scenario(
    lambda: _rt(
        "optional-fn-present-invalid",
        '{a: "no"} ~ {a: Optional(Int)}',
        ("bool", False),
        None,
    )
)

# Optional fields with key?: syntax
runtime_scenario(
    lambda: _rt("optional-syntax-missing-ok", "{} ~ {a?: Int}", ("bool", True), None)
)
runtime_scenario(
    lambda: _rt(
        "optional-syntax-present-valid",
        "{a: 1, b: 2} ~ {a: Int, b?: Int}",
        ("bool", True),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "optional-syntax-present-invalid",
        '{a: "bad"} ~ {a?: Int}',
        ("bool", False),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "optional-syntax-mixed",
        '{name: "Alice"} ~ {name: Str, age?: Int}',
        ("bool", True),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "optional-syntax-required-missing",
        "{age: 30} ~ {name: Str, age?: Int}",
        ("bool", False),
        None,
    )
)

# Union types
runtime_scenario(
    lambda: _rt("union-basic-int", "5 ~ Union(Int, Str)", ("bool", True), None)
)
runtime_scenario(
    lambda: _rt("union-basic-str", '"hi" ~ Union(Int, Str)', ("bool", True), None)
)
runtime_scenario(
    lambda: _rt("union-basic-fail", "true ~ Union(Int, Str)", ("bool", False), None)
)
runtime_scenario(
    lambda: _rt("union-with-nil", "nil ~ Union(Int, Nil)", ("bool", True), None)
)
runtime_scenario(
    lambda: _rt(
        "union-reassign",
        'Schema := Union(Int, Str); x := 5; x = "hello"; x ~ Schema',
        ("bool", True),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "union-in-object", "{age: 30} ~ {age: Union(Int, Str)}", ("bool", True), None
    )
)
runtime_scenario(
    lambda: _rt(
        "union-in-object-str",
        '{age: "30"} ~ {age: Union(Int, Str)}',
        ("bool", True),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "union-in-object-fail",
        "{age: true} ~ {age: Union(Int, Str)}",
        ("bool", False),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "union-with-optional",
        '{name: "Alice"} ~ {name: Str, age?: Union(Int, Str)}',
        ("bool", True),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "union-contract-valid",
        """fn process(value ~ Union(Int, Str)):
    value
process(42)""",
        ("number", 42),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "union-contract-invalid",
        """fn process(value ~ Union(Int, Str)):
    value
process(true)""",
        None,
        ShakarAssertionError,
    )
)

# Return type contracts
runtime_scenario(
    lambda: _rt(
        "return-contract-basic",
        """fn double(x ~ Int) ~ Int:
    x * 2
double(5)""",
        ("number", 10),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "return-contract-union",
        """fn safe_divide(a ~ Int, b ~ Int) ~ Union(Float, Nil):
    if b == 0:
        nil
    else:
        a / b
safe_divide(10, 2)""",
        ("number", 5.0),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "return-contract-union-nil",
        """fn safe_divide(a ~ Int, b ~ Int) ~ Union(Float, Nil):
    if b == 0:
        nil
    else:
        a / b
safe_divide(10, 0)""",
        ("null", None),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "return-contract-inline",
        """fn inc(x ~ Int) ~ Int: x + 1
inc(5)""",
        ("number", 6),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "return-contract-anon",
        """square := fn(x ~ Int) ~ Int: x * x
square(4)""",
        ("number", 16),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "return-contract-object",
        """PersonSchema := {name: Str, age: Int}
fn make_person(name ~ Str, age ~ Int) ~ PersonSchema:
    {name: name, age: age}
p := make_person("Bob", 30)
p.age""",
        ("number", 30),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "return-contract-fail",
        """fn bad_return(x ~ Int) ~ Str:
    x * 2
bad_return(5)""",
        None,
        ShakarTypeError,
    )
)

# Destructure contracts
runtime_scenario(
    lambda: _rt(
        "destructure-contract-single",
        """a ~ Int := 42
a""",
        ("number", 42),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "destructure-nested",
        """a, (b, (c, d)) := [1, [2, [3, 4]]]
a + b + c + d""",
        ("number", 10),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "destructure-mixed-contract-nested",
        """a ~ Int, (b, c) := [10, [20, 30]]
a + b + c""",
        ("number", 60),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "destructure-contract-basic",
        """a ~ Int, b ~ Str := 10, "hello"
a""",
        ("number", 10),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "destructure-contract-partial",
        """x ~ Int, y, z ~ Int := 5, "mid", 15
z - x""",
        ("number", 10),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "destructure-contract-broadcast",
        """m ~ Int, n ~ Int := 42
m + n""",
        ("number", 84),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "destructure-contract-array",
        """id ~ Int, name ~ Str := [100, "Alice"]
id""",
        ("number", 100),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "destructure-contract-fail",
        """a ~ Int, b ~ Str := 10, 20
a""",
        None,
        ShakarAssertionError,
    )
)

# Lookahead paren_depth leak tests - ensure layout parsing works after lookahead with parens
runtime_scenario(
    lambda: _rt(
        "lookahead-destructure-paren",
        """a ~ (Int), b := 10, 20
x := 1
y := 2
x + y""",
        ("number", 3),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "lookahead-forin-paren",
        """sum := 0
for x in [1, 2, 3]:
    sum = sum + x
sum""",
        ("number", 6),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "lookahead-guard-continue",
        """x := 5
x > 3:
    print("big")
| x > 0:
    print("small")
y := 42
y""",
        ("number", 42),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "lookahead-ccc-paren",
        """a, b := (1 > 0), (2 > 1)
x := 99
x""",
        ("number", 99),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "lookahead-dictcomp-paren",
        """d := {x: x * 2 for x in [1, 2, 3]}
y := 42
y""",
        ("number", 42),
        None,
    )
)
runtime_scenario(
    lambda: _rt(
        "lookahead-slice-literal",
        """arr := [10, 20, 30, 40]
s := arr`1:3`
z := 100
z""",
        ("number", 100),
        None,
    )
)

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
        self._parsers: Optional[ParserBundle] = None
        self._case_runner: Optional[CaseRunner] = None
        self.cases = self._build_cases()

    @property
    def parsers(self) -> ParserBundle:
        """Lazy-load parsers only when needed."""
        if self._parsers is None:
            self._parsers = ParserBundle(GRAMMAR_TEXT)
        return self._parsers

    @property
    def case_runner(self) -> CaseRunner:
        """Lazy-load case runner only when needed."""
        if self._case_runner is None:
            self._case_runner = CaseRunner(self.parsers)
        return self._case_runner

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
        # Use RD parser pipeline for AST scenarios
        tree = parse_rd(source, use_indenter=False)
        tree = Prune().transform(tree)
        tree = lower(tree)
        return tree

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
                    lines.append(
                        f"[FAIL] {scenario.name}: expected {scenario.expected_exc.__name__}"
                    )
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
                    lines.append(
                        f"[PASS] {scenario.name}: raised {scenario.expected_exc.__name__}"
                    )
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
                    lines.append(
                        f"[FAIL] {scenario.name}: expected {scenario.expected_exc.__name__}"
                    )
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
                    lines.append(
                        f"[PASS] {scenario.name}: raised {scenario.expected_exc.__name__}"
                    )
                else:
                    lines.append(f"[FAIL] {scenario.name}: {exc}")
                    failed += 1
        return lines, total, failed

    def _verify_runtime_result(
        self, value: object, expectation: Optional[Tuple[str, object]]
    ) -> Optional[str]:
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
        if kind == "array":
            if not isinstance(value, ShkArray):
                return f"expected ShkArray, got {type(value).__name__}"
            actual_items = [
                item.value if hasattr(item, "value") else item for item in value.items
            ]
            if actual_items != expected:
                return f"expected {expected!r}, got {actual_items!r}"
            return None
        if kind == "duration":
            if not isinstance(value, ShkDuration):
                return f"expected ShkDuration, got {type(value).__name__}"
            if value.nanos != expected:
                return f"expected {expected}, got {value.nanos}"
            return None
        if kind == "size":
            if not isinstance(value, ShkSize):
                return f"expected ShkSize, got {type(value).__name__}"
            if value.byte_count != expected:
                return f"expected {expected}, got {value.byte_count}"
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
        elif isinstance(value, ShkArray):
            items = [
                item.value if hasattr(item, "value") else item for item in value.items
            ]
            desc = f"produced {items!r}"
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
    ("compound-mod", "a := 10; a %= 3; a", ("number", 1)),
    ("compound-minus", "a := 10; a -= 4; a", ("number", 6)),
    ("compound-mul", "a := 3; a *= 4; a", ("number", 12)),
    ("compound-div", "a := 9; a /= 2; a", ("number", 4.5)),
    ("compound-floordiv", "a := 9; a //= 2; a", ("number", 4)),
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
