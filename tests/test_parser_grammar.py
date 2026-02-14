from __future__ import annotations

from textwrap import dedent

import pytest

from tests.support.harness import check_dot_continuation_invalid, parse_both_modes
from tests.support.harness import ParseError, parse_pipeline
from shakar_ref.tree import Tree

PARSER_GRAMMAR_CASES = [
    ("op-0", "1 and 2 or 3", "both"),
    ("op-1", "x and y and z or w", "both"),
    ("op-2", "a and $b and .c", "both"),
    ("op-3", "not 1", "both"),
    ("op-4", "not (1 and 2)", "both"),
    ("op-5", "a is b", "both"),
    ("op-6", "a is not b", "both"),
    ("op-7", "a !is b", "both"),
    ("op-8", "a in b", "both"),
    ("op-9", "a not in b", "both"),
    ("op-10", "a !in b", "both"),
    ("op-11", "7 // 2", "both"),
    ("noanchor-seg-0", "state.$lines", "both"),
    ("noanchor-seg-1", "arr$[idx]", "both"),
    ("noanchor-seg-2", "obj.$method()", "both"),
    ("noanchor-seg-3", "state.foo.$bar.baz", "both"),
    ("noanchor-seg-4", ".$field and .other", "both"),
    ("postfixif-0", "1 if 0", "both"),
    ("postfixif-1", "foo() if bar", "both"),
    ("postfixif-2", "(a+b) if c", "both"),
    ("postfixif-3", "x.y if z", "both"),
    ("wait-0", "wait spawn f()", "both"),
    ("wait-1", "x = wait spawn g(1,2)", "both"),
    ("wait-2", "h(wait spawn k())", "both"),
    ("wait-3", "wait[all](tasks)", "both"),
    ("wait-4", "wait[group]([spawn f(), spawn g()])", "both"),
    ("wait-5", "wait[all] tasks", "both"),
    ("wait-6", "wait[group] tasks", "both"),
    ("wait-7", "spawn [f, g]", "both"),
    ("wait-8", "spawn fan {f, g}", "both"),
    ("wait-9", "x := <-ch", "both"),
    ("wait-10", "1 -> ch", "both"),
    ("wait-11", "wait[foo] tasks", "both"),
    (
        "wait-12",
        dedent(
            """\
            wait[foo]:
              spawn task()

        """
        ),
        "indented",
    ),
    (
        "wait-13",
        dedent(
            """\
            group := 1
            wait[group]:
              spawn work()

        """
        ),
        "indented",
    ),
    (
        "wait-block-0",
        dedent(
            """\
            wait[any]:
              x := <-ch: x
              default: 0
            
        """
        ),
        "indented",
    ),
    (
        "wait-block-1",
        dedent(
            """\
            wait[any]:
              1 -> ch: ok
              timeout 1sec: err
            
        """
        ),
        "indented",
    ),
    (
        "wait-block-2",
        dedent(
            """\
            wait[all]:
              a: f()
              b: g(1,2)
            
        """
        ),
        "indented",
    ),
    (
        "wait-block-3",
        dedent(
            """\
            wait[group]:
              log()
              fn(): { x := 1; x }
            
        """
        ),
        "indented",
    ),
    (
        "block-0",
        dedent(
            """\
            if 1:
              a
              b
            
        """
        ),
        "indented",
    ),
    (
        "block-1",
        dedent(
            """\
            if a:
              b
            elif c:
              d
            else:
              e
            
        """
        ),
        "indented",
    ),
    (
        "block-2",
        dedent(
            """\
            if a:
              if b:
                c
              else:
                d
            
        """
        ),
        "indented",
    ),
    (
        "match-0",
        dedent(
            """\
            match x:
              1: 2
              else: 3
            
        """
        ),
        "indented",
    ),
    (
        "match-1",
        dedent(
            """\
            match key:
              "a" | "b": 1
              "c": 2
            
        """
        ),
        "indented",
    ),
    (
        "match-2",
        dedent(
            """\
            match[lt] score:
              90: "A"
              80: "B"
              else: "F"
            
        """
        ),
        "indented",
    ),
    (
        "match-3",
        dedent(
            """\
            match[==] x:
              1: 2
              else: 3
            
        """
        ),
        "indented",
    ),
    (
        "match-4",
        dedent(
            """\
            match[!=] x:
              1: 2
              else: 3
            
        """
        ),
        "indented",
    ),
    (
        "match-5",
        dedent(
            """\
            match[<] x:
              1: 2
              else: 3
            
        """
        ),
        "indented",
    ),
    (
        "match-6",
        dedent(
            """\
            match[>=] x:
              1: 2
              else: 3
            
        """
        ),
        "indented",
    ),
    (
        "match-7",
        dedent(
            """\
            match[in] x:
              1: 2
              else: 3
            
        """
        ),
        "indented",
    ),
    (
        "match-8",
        dedent(
            """\
            match[!in] x:
              1: 2
              else: 3
            
        """
        ),
        "indented",
    ),
    (
        "match-9",
        dedent(
            """\
            match[not in] x:
              1: 2
              else: 3
            
        """
        ),
        "indented",
    ),
    (
        "match-10",
        dedent(
            """\
            match[~~] x:
              r"a": 1
              else: 2
            
        """
        ),
        "indented",
    ),
    ("misc-0", "a.b[c](d, e).f", "both"),
    ("misc-1", "a.b; c.d; e", "both"),
    ("misc-2", "x = (a.b + c[d]) * e.f(g)", "both"),
    ("inline-destructure-walrus", "if true: a, b := 1, 2", "both"),
    ("fn-0", "fn add(x, y): x + y", "both"),
    ("fn-1", "fn greet(name): { dbg(name) }", "both"),
    ("param-contract-0", "fn f(a, b, c ~ Int): a", "both"),
    ("param-contract-1", "fn f(a, b, (c ~ Int)): a", "both"),
    ("param-contract-2", "fn f((a), b, c ~ Int): a", "both"),
    ("param-contract-3", "fn f(a, (b ~ Str), c ~ Int): a", "both"),
    ("param-contract-4", "fn f((a, b) ~ Int, c ~ Str): a", "both"),
    ("param-contract-5", "fn f(a, b, ...rest ~ Int): rest", "both"),
    ("param-contract-6", "&[a, b ~ Int](a + b)", "both"),
    ("power-0", "2 ** 3", "both"),
    ("power-1", "x ** 2 + y ** 2", "both"),
    ("power-2", "a ** b ** c", "both"),
    ("unary-incr-0", "++x", "both"),
    ("unary-incr-1", "--y", "both"),
    ("unary-incr-2", "++(a.b)", "both"),
    ("postfix-incr-0", "x++", "both"),
    ("postfix-incr-1", "y--", "both"),
    ("postfix-incr-2", "arr[i]++", "both"),
    ("postfix-incr-3", "obj.count--", "both"),
    (
        "dot-continuation-0",
        dedent(
            """\
            user := {profile: {contact: {name: "Ada"}}}
            name := user.profile
              .contact
              .name
        """
        ),
        "indented",
    ),
    (
        "dot-continuation-1",
        dedent(
            """\
            value := "  hi "
              .trim()
              .upper()
        """
        ),
        "indented",
    ),
    ("valuefan-0", "state.{a, b}", "both"),
    ("valuefan-1", "obj.{x(), y}", "both"),
    ("valuefan-2", "[state.{a, b, c}]", "both"),
    ("fan-literal-0", "fan { 1, 2 }", "both"),
    (
        "fan-literal-1",
        dedent(
            """\
            fan { 1,
              2,
             }
        """
        ),
        "both",
    ),
    ("fan-literal-2", "fan { }", "both"),
    ("fan-literal-3", "fan[par] { 1 }", "both"),
    ("dbg-0", "dbg(x)", "both"),
    ("computed-key-0", '{ ("key" + "1"): 10 }', "both"),
    ("computed-key-1", "{ (a + b): value }", "both"),
    ("anon-fn-expr-0", "x := fn(): 42", "both"),
    ("anon-fn-expr-1", "arr.map(fn(x): x + 1)", "both"),
    ("anon-fn-expr-2", "result := fn(()): { tmp := 1; tmp + 2 }", "both"),
    ("pattern-destructure-0", "a, b = 1, 2", "both"),
    ("pattern-destructure-1", "a, b := get_pair()", "both"),
    ("pattern-destructure-2", "x, y, z := 1, 2, 3", "both"),
    ("deepmerge-0", "a +> b", "both"),
    ("deepmerge-1", "obj1 +> obj2 +> obj3", "both"),
    ("assignor-0", "config.key or= default", "both"),
    ("assignor-1", "obj.field or= fallback", "both"),
    ("slice-0", "arr[1:3]", "both"),
    ("slice-1", "arr[:5]", "both"),
    ("slice-2", "arr[2:]", "both"),
    ("slice-3", "arr[::2]", "both"),
    ("slice-4", "arr[1:3:2]", "both"),
    ("comp-for-0", "[x for x in arr]", "both"),
    ("comp-for-1", "[x + 1 for x in nums if x > 0]", "both"),
    ("comp-for-2", "set{ x for x in vals }", "both"),
    ("comp-for-3", "{ k: v for [k, v] items }", "both"),
    ("nullsafe-0", "??(x)", "both"),
    ("nullsafe-1", "??(arr[0])", "both"),
    ("nullsafe-2", "??(obj.field)", "both"),
    ("postfix-unless-0", "x = 1 unless false", "both"),
    ("postfix-unless-1", "return 5 unless cond", "both"),
    ("indent-corner-0", "x := fan { 1, 2 }", "both"),
    (
        "indent-corner-1",
        dedent(
            """\
            x := fan {
              1,
              2
            }
        """
        ),
        "both",
    ),
    (
        "indent-corner-2",
        dedent(
            """\
            x := fan {
              # Comment
              1,
              # Another
              2
            }
        """
        ),
        "both",
    ),
    (
        "indent-corner-3",
        dedent(
            """\
            x := fan {
              fan { 1, 2 },
              fan { 3, 4 }
            }
        """
        ),
        "both",
    ),
    (
        "indent-corner-4",
        dedent(
            """\
            obj := {
              f: fan {
                1,
                2
              }
            }
        """
        ),
        "both",
    ),
    (
        "indent-corner-5",
        dedent(
            """\
            x := fan {
              { a: 1 },
              { b: 2 }
            }
        """
        ),
        "both",
    ),
    (
        "indent-corner-6",
        dedent(
            """\
            obj := {
              method: fn():
                return 1
            }
        """
        ),
        "both",
    ),
    (
        "indent-corner-7",
        dedent(
            """\
            obj := {
              method: fn(): 1
            }
        """
        ),
        "both",
    ),
    (
        "indent-corner-8",
        dedent(
            """\
            root := {
              l1: {
                l2: fan {
                  {
                    deep_method: fn():
                      if true:
                        return fan { 1 }
                  }
                }
              }
            }
        """
        ),
        "both",
    ),
    (
        "indent-corner-9",
        dedent(
            """\
            fn complex(
              a,
              b
            ):
              return a + b
        """
        ),
        "both",
    ),
    (
        "indent-corner-10",
        dedent(
            """\
            my_call(
              arg1,
              arg2
            )
        """
        ),
        "both",
    ),
    (
        "indent-corner-11",
        dedent(
            """\
            arr := [
              1,
              2,
              3
            ]
        """
        ),
        "both",
    ),
    (
        "indent-corner-12",
        dedent(
            """\
            x := fan {
              1, 2,
              3,
              4
            }
        """
        ),
        "both",
    ),
    (
        "indent-corner-13",
        dedent(
            """\
            obj := {
              m1: fn():
                return 1
              m2: fn():
                return 2
            }
        """
        ),
        "both",
    ),
    (
        "indent-corner-14",
        dedent(
            """\
            obj := {
              m1: fn():
                return 1
              ,
              m2: fn():
                return 2
            }
        """
        ),
        "both",
    ),
    (
        "indent-corner-15",
        dedent(
            """\
            obj := {
              a: 1
              b: 2
              c: 3
            }
        """
        ),
        "both",
    ),
    (
        "indent-corner-16",
        dedent(
            """\
            obj := {
              a: 1, b: 2
              c: 3, d: 4
              e: 5
            }
        """
        ),
        "both",
    ),
    (
        "indent-corner-17",
        dedent(
            """\
            config := {
              debug: true
              retries: 3
              timeout: 5sec
            }
        """
        ),
        "both",
    ),
    (
        "indent-corner-18",
        dedent(
            """\
            root := {
              child1: {
                name: "c1"
                age: 10
              }
              child2: {
                name: "c2"
                age: 20
              }
            }
        """
        ),
        "both",
    ),
    (
        "indent-corner-19",
        dedent(
            """\
            data := {
              list: [1, 2, 3]
              func: fn(x): x * 2
              map: { k: "v" }
            }
        """
        ),
        "both",
    ),
    (
        "indent-corner-20",
        dedent(
            """\
            headers := {
              "Content-Type": "application/json"
              "Authorization": "Bearer token"
            }
        """
        ),
        "both",
    ),
    (
        "indent-corner-21",
        dedent(
            """\
            flags := {
              (1 + 1): "two"
              (2 + 2): "four"
            }
        """
        ),
        "both",
    ),
    (
        "indent-corner-22",
        dedent(
            """\
            classy := {
              get name(): "Bond"
              set name(n): nil
              greet(who): "Hello " + who
            }
        """
        ),
        "both",
    ),
    (
        "indent-corner-23",
        dedent(
            """\
            spaced := {
              a: 1
            
              # This is b
              b: 2
            
              c: 3
            }
        """
        ),
        "both",
    ),
    (
        "indent-corner-24",
        dedent(
            """\
            trailing := {
              a: 1
              b: 2
            }
        """
        ),
        "both",
    ),
]


@pytest.mark.parametrize(
    "code, start",
    [pytest.param(code, start, id=name) for name, code, start in PARSER_GRAMMAR_CASES],
)
def test_parser_grammar(code: str, start: str) -> None:
    parse_both_modes(code, start)


def test_dot_continuation_invalid() -> None:
    check_dot_continuation_invalid()


@pytest.mark.parametrize(
    "source",
    [
        "wait[]:",
        "wait[1]:",
        'wait["any"]:',
        "wait[any all]:",
    ],
)
def test_wait_modifier_slot_parse_errors(source: str) -> None:
    for use_indenter in (False, True):
        with pytest.raises(ParseError):
            parse_pipeline(source, use_indenter=use_indenter)


PARSE_ERROR_LOCATION_CASES = [
    ("missing-rpar", "f(1, 2", 1, 7, "Expected RPAR"),
    ("bad-fn-arg", "fn 123()", 1, 4, "Expected COLON"),
    ("lone-colon", ":", 1, 1, "Unexpected token"),
    ("missing-body", "if true:", 1, 9, "Unexpected token"),
    # Multi-line: error on later line
    ("line2-missing-rpar", "x = 1\nf(1, 2", 2, 7, "Expected RPAR"),
    ("line3-bad-fn", "a = 1\nb = 2\nfn 99()", 3, 4, "Expected COLON"),
    ("line2-unexpected", "x = 1\n:", 2, 1, "Unexpected token"),
]


@pytest.mark.parametrize(
    "name,source,exp_line,exp_col,msg",
    PARSE_ERROR_LOCATION_CASES,
    ids=[c[0] for c in PARSE_ERROR_LOCATION_CASES],
)
def test_parse_error_location(
    name: str, source: str, exp_line: int, exp_col: int, msg: str
) -> None:
    with pytest.raises(ParseError) as exc_info:
        parse_pipeline(source, use_indenter=False)

    err = exc_info.value
    assert msg in err.message
    assert err.line == exp_line, f"expected line {exp_line}, got {err.line}"
    assert err.column == exp_col, f"expected col {exp_col}, got {err.column}"
    # end span must be present and at least as wide as start
    assert err.end_line is not None
    assert err.end_column is not None
    assert err.end_line >= err.line
    if err.end_line == err.line:
        assert err.end_column > err.column


def _collect_tree_labels(node: object) -> set[str]:
    labels: set[str] = set()
    if not isinstance(node, Tree):
        return labels

    stack = [node]
    while stack:
        current = stack.pop()
        labels.add(current.data)
        for child in current.children:
            if isinstance(child, Tree):
                stack.append(child)
    return labels


def test_wait_unknown_call_uses_generic_wait_modifier_node() -> None:
    ast = parse_pipeline("wait[foo] tasks", use_indenter=False)
    labels = _collect_tree_labels(ast)
    assert "waitmodifiercall" in labels
    assert "waitgroupcall" not in labels


def test_wait_unknown_block_uses_generic_wait_modifier_node() -> None:
    ast = parse_pipeline(
        dedent(
            """\
            wait[foo]:
              spawn work()
        """
        ),
        use_indenter=True,
    )
    labels = _collect_tree_labels(ast)
    assert "waitmodifierblock" in labels
    assert "waitgroupblock" not in labels
