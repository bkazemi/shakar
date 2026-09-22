from pathlib import Path

import pytest

from shakar_ref.lexer_rd import LexError
from shakar_ref.parser_rd import ParseError
from shakar_ref.runner import eval_in_env, run, run_with_env
from shakar_ref.runtime import ShakarRuntimeError, ShakarTypeError, ShkNumber
from tests.support.harness import run_runtime_case


@pytest.mark.parametrize(
    "source, expected",
    [
        ("a := 0\nif true:\n  a += 1\nif true:\n  a += 2\na", ("number", 3)),
        (
            "fn f():\n  a := 0\n  if true:\n    a += 1\n  if false:\n    a += 2\n  a\nf()",
            ("number", 1),
        ),
        ("a := 0\nif true: a += 1\nif true: a += 2\na", ("number", 3)),
        ("a := 0; a += 1 if false; a", ("number", 0)),
        ("n := 2; {ready: (n == 2, 2), issues: []}.ready", ("bool", True)),
        ("n := 2; {ok: n == 1, 2}.ok", ("bool", False)),
        ("n := 2; other := 2; obj := {ok: n == 2, other}; obj.len", ("number", 1)),
        ("n := 0; {ready: (n == 0), issues: []}.ready", ("bool", True)),
        (
            'xs := [" A ", " ", " B "]; xs .= .map&(.trim()).filter&(.len > 0); xs',
            ("array", ["A", "B"]),
        ),
        ("xs := [1, 2]; xs .= .map&[x](x + 1); xs", ("array", [2, 3])),
        ("offset := 10; [1, 2].map&(. + offset)", ("array", [11, 12])),
        ("policy := {min: 2}; [1, 2, 3].filter&(. >= $policy.min)", ("array", [2, 3])),
        (
            "fn add(x, y): x + y\noffset := 10\n[1, 2].map&(add(., offset))",
            ("array", [11, 12]),
        ),
        ("offset := 10; [1, 2].map&(item + offset)", ("array", [11, 12])),
        ("[1, 2].map&(str(item))", ("array", ["1", "2"])),
        ("item := 100; [1, 2].map&[item](item + 1)", ("array", [2, 3])),
        ("offset := 10; [1, 2].map&(offset)", ("array", [10, 10])),
        (
            "fn add(offset): [1, 2].map&(item + offset)\nassert add(10) == [11, 12]; assert add(20) == [21, 22]; true",
            ("bool", True),
        ),
        (
            "out := []; for [10, 20]: { x := 1; x .= . + 1; out.append(.) }; out",
            ("array", [10, 20]),
        ),
        ('outer := {n: 7}; x := " A "; outer and (x .= .trim()) and .n', ("number", 7)),
        (
            'xs := [" A "]; out := []; for xs: { x := " B "; =x.trim(); out.append(.) }; out',
            ("array", [" A "]),
        ),
    ],
)
def test_language_interactions(source: str, expected: tuple[str, object]) -> None:
    run_runtime_case(source, expected, None)


@pytest.mark.parametrize("update", ["x .= .trim()", "=x.trim()", 'x += "!"'])
@pytest.mark.parametrize("separator", ["\n", "; "])
def test_subject_does_not_escape_statement(update: str, separator: str) -> None:
    with pytest.raises(ShakarRuntimeError, match="No subject available"):
        run(separator.join(['x := " A "', update, "."]))


@pytest.mark.parametrize(
    "source",
    [
        "[1].map&(. + missing)",
        "[].map&(. + missing)",
        "callback := &(. + missing)",
        "callback := &(. + later); later := 10",
        "[0].map&(. and missing)",
    ],
)
def test_subject_lambda_rejects_unbound_candidates_at_creation(source: str) -> None:
    with pytest.raises(
        ShakarTypeError, match="Cannot mix subject.*implicit parameters"
    ):
        run(source)


@pytest.mark.parametrize("imported", [False, True])
def test_invalid_indentation_never_falls_back_to_top_level(
    tmp_path: Path, imported: bool
) -> None:
    # A no-indent retry used to accept this and execute x += 1 at top level.
    source = "fn f():\n  x := 1\n x += 1\nf()"
    if imported:
        (tmp_path / "bad.shk").write_text(source)
        source = 'import "./bad.shk"'
    with pytest.raises((LexError, ParseError), match="[Ii]ndent|[Dd]edent"):
        run(source, source_path=str(tmp_path / "main.shk"))


def test_inferred_capture_does_not_force_lazy_initializer() -> None:
    source = """calls := 0
fn make(): { calls += 1; 10 }
once[lazy]: offset := make()
f := &(item + offset)
assert calls == 0
value := f(2)
assert calls == 1
value
"""
    run_runtime_case(source, ("number", 12), None)


def test_multiline_literal_can_have_same_line_postfix_guard() -> None:
    run_runtime_case('fn f(): return "a\nb" if true\nf()', ("string", "a\nb"), None)


def test_subject_restored_after_caught_exception() -> None:
    source = """out := []
for [10, 20]:
  x := 0
  try:
    x .= . + 1
    throw "oops"
  catch:
    nil
  out.append(.)
out
"""
    run_runtime_case(source, ("array", [10, 20]), None)


def test_nested_subject_lambdas_capture_outer_name() -> None:
    source = "offset := 10; result := [[1], [2]].map&(.map&(. + offset)); result == [[11], [12]]"
    run_runtime_case(source, ("bool", True), None)


@pytest.mark.parametrize(
    "source, expected",
    [
        ("fn add(x, amount): x + amount\n[1].map&(add(., amount: 2))", ("array", [3])),
        ("[1, 2].map&(str(.))", ("array", ["1", "2"])),
        ("offset := 0; [1].map&(. + offset)", ("array", [1])),
        ("offset := nil; [1].map&(offset ?? .)", ("array", [1])),
        ("offset := false; [1].map&(. > 0 and not offset)", ("array", [True])),
        (
            "fn make(offset): &(. + offset)\ncallback := make(10)\ncallback(2)",
            ("number", 12),
        ),
        ("callbacks := [1].map&(fn(x): x + 1); callbacks[0](2)", ("number", 3)),
        ("records := [1].map&({value: .}); records[0].value", ("number", 1)),
        (
            "calls := 0\nfn make(): { calls += 1; 10 }\nonce[lazy]: offset := make()\ncallback := &(. + offset)\nassert calls == 0\nresult := callback(2)\nassert calls == 1\nresult",
            ("number", 12),
        ),
    ],
)
def test_subject_capture_validation(source: str, expected: tuple[str, object]) -> None:
    run_runtime_case(source, expected, None)


@pytest.mark.parametrize("first", ["true", "false"])
@pytest.mark.parametrize("second", ["true", "false"])
@pytest.mark.parametrize("gap", ["", "\n", "# comment\n"])
def test_adjacent_if_blocks_are_independent(first: str, second: str, gap: str) -> None:
    source = f"out := []\nif {first}:\n  out.append(1)\n{gap}if {second}:\n  out.append(2)\nout"
    expected = []
    if first == "true":
        expected.append(1)
    if second == "true":
        expected.append(2)
    run_runtime_case(source, ("array", expected), None)


@pytest.mark.parametrize(
    "source, expected",
    [
        (
            "out := [];\nfor [1, 2]:\n  out.append(.)\nif true:\n  out.append(3)\nout",
            ("array", [1, 2, 3]),
        ),
        ("n := 0\nwhile n < 2:\n  n += 1\nif true:\n  n += 3\nn", ("number", 5)),
        ("fn f():\n  10\nif true:\n  result := f()\nresult", ("number", 10)),
        (
            "out := []\nif false:\n  out.append(1)\nelif true:\n  out.append(2)\nelse:\n  out.append(3)\nif true:\n  out.append(4)\nout",
            ("array", [2, 4]),
        ),
        (
            "out := []; out.append(1) if true; out.append(2) unless true; out",
            ("array", [1]),
        ),
        ("out := []; if true: { out.append(1) } if false; out", ("array", [])),
    ],
)
def test_postfix_boundary_preserves_other_control_flow(
    source: str, expected: tuple[str, object]
) -> None:
    run_runtime_case(source, expected, None)


@pytest.mark.parametrize("imported", [False, True])
@pytest.mark.parametrize(
    "source, expected",
    [
        (" value := 3", 3),
        ("value := 0\nif true: { value += 1; value += 2 }\n", 3),
        ("fn f():\n  x := 1\n  x += 2\n  x\nvalue := f()", 3),
        ("# leading comment\nvalue := (\n  1\n  + 2\n)", 3),
    ],
)
def test_parse_mode_preserves_valid_sources(
    tmp_path: Path, imported: bool, source: str, expected: int
) -> None:
    if imported:
        (tmp_path / "valid.shk").write_text(source)
        program = 'import "./valid.shk" bind module\nmodule.value'
        value = run(program, source_path=str(tmp_path / "main.shk"))
    else:
        # Keep single-line snippets single-line when adding the result expression.
        separator = "\n" if "\n" in source else "; "
        value = run(source + separator + "value")
    assert isinstance(value, ShkNumber)
    assert value.value == expected


def test_parse_error_prevents_any_execution() -> None:
    frame = run_with_env("flag := 0")
    source = "flag += 1\nfn f():\n  x := 1\n x += 1"
    with pytest.raises((LexError, ParseError)):
        eval_in_env(source, frame)
    flag = frame.get("flag")
    assert isinstance(flag, ShkNumber)
    assert flag.value == 0
