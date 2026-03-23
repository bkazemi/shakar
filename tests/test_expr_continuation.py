"""Tests for indented expression continuation.

Covers the feature described in docs/features/expression_continuation.md.
"""

import pytest

from tests.support.harness import parse_pipeline, run_runtime_case
from shakar_ref.lexer_rd import LexError
from shakar_ref.parser_rd import ParseError, parse_expr_fragment, parse_source
from shakar_ref.runner import run


# ========================================================================
# Positive tests
# ========================================================================


class TestAssignmentContinuation:
    """Section 4.1: continuation after assignment operators."""

    def test_walrus_newline_rhs(self):
        result = run(
            """
x :=
  42
x
"""
        )
        assert result.value == 42

    def test_assign_newline_rhs(self):
        result = run(
            """
x := 0
x =
  42
x
"""
        )
        assert result.value == 42

    def test_compound_assign_continuation(self):
        result = run(
            """
x := 10
x +=
  5
x
"""
        )
        assert result.value == 15

    def test_minus_eq_continuation(self):
        result = run(
            """
x := 10
x -=
  3
x
"""
        )
        assert result.value == 7

    def test_star_eq_continuation(self):
        result = run(
            """
x := 3
x *=
  4
x
"""
        )
        assert result.value == 12

    def test_slash_eq_continuation(self):
        result = run(
            """
x := 10
x /=
  2
x
"""
        )
        assert result.value == 5

    def test_walrus_to_match_expr(self):
        """Assignment to match expression across newline."""
        result = run(
            """
key := "a"
result :=
  match key:
    "a": 1
    else: 0
result
"""
        )
        assert result.value == 1

    def test_walrus_to_once_expr(self):
        """Assignment to once expression across newline."""
        result = run(
            """
config :=
  once:
    42
config
"""
        )
        assert result.value == 42

    def test_parity_same_line_vs_continuation(self):
        """Continuation does not change RHS legality."""
        r1 = run(
            """
x := 1 + 2
x
"""
        )
        r2 = run(
            """
x :=
  1 + 2
x
"""
        )
        assert r1.value == r2.value == 3

    def test_field_walrus_continuation_then_next_stmt(self):
        parse_source(
            """
obj := {x: 0}
obj.x :=
  1
next := 2
""",
            use_indenter=True,
        )


class TestInfixContinuation:
    """Section 4.2: continuation after infix operators."""

    def test_multiline_addition(self):
        result = run(
            """
x :=
  1
  + 2
  + 3
x
"""
        )
        assert result.value == 6

    def test_multiline_subtraction(self):
        result = run(
            """
x :=
  10
  - 3
  - 2
x
"""
        )
        assert result.value == 5

    def test_multiline_mixed_arith(self):
        """Precedence respected across continuation lines."""
        result = run(
            """
x :=
  1
  + 2
  * 3
x
"""
        )
        assert result.value == 7  # 1 + (2 * 3)

    def test_multiline_nullish_chain(self):
        result = run(
            """
x := nil
y := nil
z :=
  x
  ?? y
  ?? "guest"
z
"""
        )
        assert result.value == "guest"

    def test_multiline_logical_or(self):
        result = run(
            """
a := false
b := true
x :=
  a
  || b
x
"""
        )
        assert hasattr(result, "value") and result.value is True

    def test_multiline_logical_and(self):
        result = run(
            """
a := true
b := true
x :=
  a
  && b
x
"""
        )
        assert hasattr(result, "value") and result.value is True

    def test_multiline_comparison(self):
        result = run(
            """
x := 5
y :=
  x
  == 5
y
"""
        )
        assert result.value is True

    def test_multiline_addition_across_comment_only_line(self):
        result = run(
            """
x :=
  1
  # still the same continued expression
  + 2
x
"""
        )
        assert result.value == 3


class TestTernaryContinuation:
    """Section 4.3: multiline ternary."""

    def test_multiline_ternary(self):
        result = run(
            """
cond := true
value :=
  cond
  ? 10
  : 20
value
"""
        )
        assert result.value == 10

    def test_multiline_ternary_false(self):
        result = run(
            """
cond := false
value :=
  cond
  ? 10
  : 20
value
"""
        )
        assert result.value == 20

    def test_nested_ternary_then_branch(self):
        result = run(
            """
outer := true
inner := false
value := outer ? inner ? 1 : 2 : 3
value
"""
        )
        assert result.value == 2


class TestTrailingOperatorContinuation:
    """Trailing binary/ternary operators: op ends the line, RHS on next indented line."""

    def test_trailing_plus(self):
        result = run(
            """
x := 1 +
  2
x
"""
        )
        assert result.value == 3

    def test_trailing_nullish(self):
        result = run(
            """
x := nil
result := x ??
  42
result
"""
        )
        assert result.value == 42

    def test_trailing_comparison_in_if(self):
        result = run(
            """
score := 5
bonus := 3
limit := 7
if score +
  bonus >
  limit:
  "over"
else:
  "under"
"""
        )
        assert result.value == "over"

    def test_trailing_qmark(self):
        """Trailing ? with then-arm on next indented line."""
        result = run(
            """
value := true ?
  1
  : 2
value
"""
        )
        assert result.value == 1

    def test_trailing_colon_in_ternary(self):
        """Trailing : with else-arm on next indented line."""
        result = run(
            """
value := true ? 1 :
  2
value
"""
        )
        assert result.value == 1

    def test_trailing_star(self):
        result = run(
            """
x := 3 *
  4
x
"""
        )
        assert result.value == 12

    def test_trailing_pow(self):
        result = run(
            """
x := 2 **
  3
x
"""
        )
        assert result.value == 8

    def test_trailing_and(self):
        result = run(
            """
x := true &&
  false
x
"""
        )
        assert result.value is False

    def test_trailing_or(self):
        result = run(
            """
x := false ||
  true
x
"""
        )
        assert result.value is True

    def test_trailing_op_inside_continuation(self):
        """Trailing op when already inside a continuation block."""
        result = run(
            """
x :=
  1 +
  2
x
"""
        )
        assert result.value == 3


class TestPrefixHeadContinuation:
    """Section 4.4: prefix statement heads with expression operands."""

    def test_return_continuation(self):
        result = run(
            """
fn f():
  return
    42
f()
"""
        )
        assert result.value == 42

    def test_return_match_continuation(self):
        result = run(
            """
fn build():
  return
    match "a":
      "a": 99
      else: 0
build()
"""
        )
        assert result.value == 99

    def test_bare_return_no_continuation(self):
        """Bare return followed by new statement on non-indented line."""
        result = run(
            """
fn f():
  return
x := 42
x
"""
        )
        assert result.value == 42

    def test_throw_continuation(self):
        result = run(
            """
fn f():
  throw
    "bad"
f() catch: "caught"
"""
        )
        assert result.value == "caught"

    def test_throw_expr_continuation_in_assignment(self):
        parse_source(
            """
x := throw
  "bad"
""",
            use_indenter=True,
        )

    def test_throw_expr_continuation_after_nullish(self):
        parse_source(
            """
x := nil
  ?? throw
    "bad"
""",
            use_indenter=True,
        )

    def test_bare_throw_no_continuation(self):
        """Bare throw without indented line."""
        result = run(
            """
fn f():
  throw
f() catch: 99
"""
        )
        assert result.value == 99

    def test_assert_continuation(self):
        """Assert with continuation."""
        run(
            """
assert
  1 + 1 == 2
"""
        )

    def test_assert_continuation_with_message(self):
        run(
            """
assert
  true, "ok"
"""
        )

    def test_assert_continuation_with_multiline_message(self):
        run(
            """
assert
  true,
  "ok"
"""
        )

    def test_assert_same_line_then_multiline_message(self):
        result = run(
            """
assert true,
  "ok"
next := 1
next
"""
        )
        assert result.value == 1

    def test_dbg_continuation(self):
        """dbg with continuation parses correctly."""
        parse_source(
            """
dbg
  42
""",
            use_indenter=True,
        )

    def test_dbg_continuation_with_multiline_second_operand(self):
        parse_source(
            """
dbg
  1,
  2
""",
            use_indenter=True,
        )

    def test_dbg_same_line_then_multiline_second_operand(self):
        parse_source(
            """
dbg 1,
  2
next := 1
""",
            use_indenter=True,
        )

    def test_unary_minus_continuation(self):
        result = run(
            """
x :=
  -
    1
x
"""
        )
        assert result.value == -1

    def test_unary_not_continuation(self):
        result = run(
            """
x :=
  not
    false
x
"""
        )
        assert result.value is True

    def test_recv_continuation(self):
        parse_source(
            """
x :=
  <-
    ch
""",
            use_indenter=True,
        )

    def test_wait_continuation(self):
        parse_source(
            """
x :=
  wait
    ch
""",
            use_indenter=True,
        )

    def test_wait_modifier_continuation(self):
        parse_source(
            """
x :=
  wait[all]
    tasks
""",
            use_indenter=True,
        )

    def test_spawn_continuation(self):
        parse_source(
            """
x :=
  spawn
    work()
""",
            use_indenter=True,
        )


class TestBlockExpressionContinuation:
    """Section 4.5: block-valued expressions after assignment."""

    def test_match_continuation(self):
        result = run(
            """
result :=
  match "a":
    "a": 1
    else: 0
result
"""
        )
        assert result.value == 1

    def test_once_continuation(self):
        result = run(
            """
config :=
  once:
    42
config
"""
        )
        assert result.value == 42


class TestDotChainContinuation:
    """Section 6: existing dot-chain still works."""

    def test_existing_dot_chain(self):
        result = run(
            """
user := {profile: {contact: {name: "Ada"}}}
name := user.profile
  .contact
  .name
name
"""
        )
        assert result.value == "Ada"


class TestGroupedDelimiters:
    """Section 4.6 / 7: grouped delimiters with multiline content."""

    def test_multiline_call_args(self):
        result = run(
            """
fn f(a, b, c): a + b + c
f(
  1,
  2,
  3,
)
"""
        )
        assert result.value == 6

    def test_multiline_array(self):
        result = run(
            """
items := [
  1,
  2,
  3,
]
items[0]
"""
        )
        assert result.value == 1

    def test_multiline_object(self):
        result = run(
            """
obj := {
  name: "ada",
  score: 10,
}
obj.name
"""
        )
        assert result.value == "ada"

    def test_grouped_if_condition_inside_function_block(self):
        result = run(
            """
fn f():
  a := true
  b := true
  out := 0

  if (a and
    b):
    out = 42
  out
f()
"""
        )
        assert result.value == 42

    def test_object_field_fn_body_inside_group(self):
        parse_source(
            """
obj := {
  f: fn():
    1
}
""",
            use_indenter=True,
        )

    def test_object_field_fn_body_inside_group_and_function(self):
        parse_source(
            """
fn outer():
  obj := {
    f: fn():
      1
  }
""",
            use_indenter=True,
        )

    def test_named_function_return_contract_continuation(self):
        parse_source(
            """
fn f() ~
  Int:
  1
""",
            use_indenter=True,
        )

    def test_anonymous_function_return_contract_continuation(self):
        parse_source(
            """
f := fn() ~
  Int:
  1
""",
            use_indenter=True,
        )

    def test_anonymous_function_return_contract_continuation_in_call_arg(self):
        parse_source(
            """
wrap(
  fn() ~
    Int:
    1
)
""",
            use_indenter=True,
        )

    def test_object_field_nested_block_inside_group_with_sibling_field(self):
        parse_source(
            """
obj := {
  method: fn():
    if true:
      return 1
  sibling: 2
}
""",
            use_indenter=True,
        )

    def test_object_field_grouped_body_after_leading_comment_and_blank(self):
        parse_source(
            """
obj := {
  f: fn():
    # note

    a := 1
    b := 2
}
""",
            use_indenter=True,
        )

    def test_object_field_grouped_body_after_header_indent_comment_and_blank(self):
        parse_source(
            """
obj := {
  f: fn():
  # note

    a := 1
    b := 2
}
""",
            use_indenter=True,
        )


class TestContinuationBlock:
    """Section 5: continuation block model."""

    def test_nested_block_in_continuation(self):
        """Nested : block inside continuation."""
        result = run(
            """
result :=
  match 1:
    1: "one"
    else: "other"
result
"""
        )
        assert result.value == "one"

    def test_expression_after_continuation(self):
        """Statement after continuation block ends correctly."""
        result = run(
            """
x :=
  5
y := x + 1
y
"""
        )
        assert result.value == 6

    def test_match_arm_pattern_continuation_keeps_else_visible(self):
        result = run(
            """
x := match 3:
  1
    + 1: 10
  else: 20
x
"""
        )
        assert result.value == 20


class TestStructuralContinuators:
    """Section 6.5 / 13.4: structural continuators."""

    def test_catch_in_continuation(self):
        """Catch at same indent as continuation expression."""
        result = run(
            """
fn risky():
  throw "oops"
x := risky() catch err: "recovered"
x
"""
        )
        assert result.value == "recovered"

    def test_catch_after_comment_only_line_still_attaches(self):
        result = run(
            """
fn risky():
  throw "oops"
value :=
  risky()
  # explain recovery path
  catch err: "recovered"
value
"""
        )
        assert result.value == "recovered"

    def test_noindent_catch_after_comment_only_line_still_attaches(self):
        parse_source(
            """
risky()
# explain recovery path
catch err: 1
""",
            use_indenter=False,
        )

    def test_blank_line_breaks_catch_attachment(self):
        with pytest.raises(ParseError):
            parse_source(
                """
fn risky():
  throw "oops"
value :=
  risky()

  catch err: "recovered"
value
""",
                use_indenter=True,
            )

    def test_whitespace_only_line_breaks_catch_attachment(self):
        with pytest.raises(ParseError):
            parse_source(
                """
fn risky():
  throw "oops"
value :=
  risky()
   
  catch err: "recovered"
value
""",
                use_indenter=True,
            )

    def test_multiline_chain_catch_statement_reparse(self):
        result = run(
            """
fn risky(x):
  throw "oops"
obj := {f: risky}
out := "miss"
obj
  .f(1)
  catch err: out = "caught"
out
"""
        )
        assert result.value == "caught"

    def test_multiline_chain_catch_statement_before_next_stmt(self):
        parse_source(
            """
fn risky(x):
  throw "oops"
obj := {f: risky}
obj
  .f(1)
  catch err: 1
next := 2
""",
            use_indenter=True,
        )


class TestDestructureContinuation:
    """Destructure RHS continuation."""

    def test_destructure_walrus_continuation(self):
        result = run(
            """
a, b :=
  1, 2
a + b
"""
        )
        assert result.value == 3

    def test_destructure_assign_continuation(self):
        result = run(
            """
a := 0
b := 0
a, b =
  3, 4
a + b
"""
        )
        assert result.value == 7


# ========================================================================
# Negative tests
# ========================================================================


class TestNegativeCases:
    """Sections 9.x: invalid patterns."""

    def test_no_indent_after_walrus(self):
        """9.1: continuation without indentation is forbidden."""
        with pytest.raises(ParseError):
            parse_source(
                """
x :=
foo()
""",
                use_indenter=True,
            )

    def test_same_indent_operator(self):
        """9.2: same-indent operator continuation is forbidden."""
        with pytest.raises(ParseError):
            parse_source(
                """
score :=
+ bonus
""",
                use_indenter=True,
            )

    def test_ccc_leg_no_newline_split(self):
        """9.7: CCC legs do not split across newline."""
        with pytest.raises(ParseError):
            parse_source(
                """
ok :=
  x < 10,
  > 0
""",
                use_indenter=True,
            )

    def test_fresh_line_call_rejected(self):
        """9.6: fresh-line call suffix is not valid continuation."""
        with pytest.raises(ParseError):
            parse_source(
                """
x :=
  f
  (1, 2)
""",
                use_indenter=True,
            )

    def test_fresh_line_index_rejected(self):
        """9.6: fresh-line index suffix is not valid continuation."""
        with pytest.raises(ParseError):
            parse_source(
                """
x :=
  arr
  [0]
""",
                use_indenter=True,
            )

    def test_blank_line_spliced_infix_rejected(self):
        with pytest.raises(ParseError):
            parse_source(
                """
x :=
  1

  + 2
""",
                use_indenter=True,
            )

    def test_blank_line_spliced_block_head_infix_rejected(self):
        with pytest.raises(ParseError):
            parse_source(
                """
if 1

  + 1 == 2:
    42
""",
                use_indenter=True,
            )

    def test_blank_line_breaks_ternary_else_arm(self):
        """Blank line before `:` must not attach the else arm."""
        with pytest.raises(ParseError):
            parse_source(
                """
x := true
  ? 1

  : 2
""",
                use_indenter=True,
            )

    def test_blank_line_breaks_chain_continuation(self):
        """Blank line before `.method` must not continue a dot chain."""
        with pytest.raises(ParseError):
            parse_source(
                """
x := [1, 2]
  .map(fn(x): x + 1)

  .map(fn(x): x * 2)
""",
                use_indenter=True,
            )

    def test_blank_line_breaks_pattern_contract_continuation(self):
        """Blank line before `~ Type` must not attach the contract."""
        with pytest.raises(ParseError):
            parse_source("a\n\n  ~ Int, b := 1, 2\n", use_indenter=True)

    def test_blank_line_breaks_return_operand(self):
        """Blank line after bare return must not absorb the next indented expr."""
        tree = parse_source("fn f():\n  return\n\n  1\nf()\n", use_indenter=True)
        # stmtlist > stmt > fnstmt > body > stmt(0) = returnstmt
        fn_stmt = tree.children[0].children[0].children[0]
        body = fn_stmt.children[2]  # body tree
        ret_stmt = body.children[1].children[0]  # first stmt in body
        assert ret_stmt.data == "returnstmt"
        assert len(ret_stmt.children) == 1  # bare return, no operand

    def test_blank_line_breaks_throw_operand(self):
        """Blank line after bare throw must not absorb the next indented expr."""
        tree = parse_source('fn f():\n  throw\n\n  "err"\n', use_indenter=True)
        fn_stmt = tree.children[0].children[0].children[0]
        body = fn_stmt.children[2]
        throw_stmt = body.children[1].children[0]
        assert throw_stmt.data == "throwstmt"
        assert len(throw_stmt.children) == 0  # bare throw, no operand


class TestContinuationExitCleanup:
    """Verify continuation blocks are properly exited."""

    def test_continuation_then_next_stmt(self):
        """After continuation, next statement at base indent works."""
        result = run(
            """
x :=
  10
y := 20
x + y
"""
        )
        assert result.value == 30

    def test_continuation_in_loop(self):
        """Continuation inside loop body."""
        result = run(
            """
x := 0
for i in [1, 2, 3]:
  x +=
    i
x
"""
        )
        assert result.value == 6

    def test_continuation_in_if(self):
        """Continuation inside if body."""
        result = run(
            """
x := 0
if true:
  x =
    42
x
"""
        )
        assert result.value == 42

    def test_multiple_continuations(self):
        """Multiple continuation statements in sequence."""
        result = run(
            """
a :=
  1
b :=
  2
a + b
"""
        )
        assert result.value == 3


class TestDirectInfixContinuation:
    """Regression: operators entering continuation without outer assignment wrapper.

    Covers the off-by-one where entered depth was captured after the first
    _try_infix_continuation had already incremented it.
    """

    def test_nullish_direct_continuation(self):
        """?? on indented line after plain assignment RHS."""
        result = run(
            """
x := 1
result := x
  ?? 2
result
"""
        )
        assert result.value == 1

    def test_or_direct_continuation(self):
        """|| on indented line after plain assignment RHS."""
        result = run(
            """
x := false
result := x
  || true
result
"""
        )
        assert result.value is True

    def test_and_direct_continuation(self):
        """&& on indented line after plain assignment RHS."""
        result = run(
            """
x := true
result := x
  && false
result
"""
        )
        assert result.value is False

    def test_ternary_direct_continuation(self):
        """? : on indented lines after plain assignment RHS."""
        result = run(
            """
x := true
result := x
  ? 10
  : 20
result
"""
        )
        assert result.value == 10

    def test_compare_direct_continuation(self):
        """== on indented line after plain assignment RHS."""
        result = run(
            """
x := 5
result := x
  == 5
result
"""
        )
        assert result.value is True

    def test_add_direct_continuation(self):
        """+ on indented line after plain assignment RHS."""
        result = run(
            """
x := 1
result := x
  + 2
result
"""
        )
        assert result.value == 3


class TestDirectBlockHeadContinuation:
    """Regression: direct infix continuations before ':'-headed bodies."""

    def test_if_condition_direct_continuation(self):
        result = run(
            """
value := 0
if 1
  + 1 == 2:
    value = 42
value
"""
        )
        assert result.value == 42

    def test_while_condition_direct_continuation(self):
        result = run(
            """
i := 0
count := 0
while i
  < 3:
    count += 1
    i += 1
count
"""
        )
        assert result.value == 3

    def test_match_subject_direct_continuation(self):
        result = run(
            """
value := match 1
  + 1:
    2: "ok"
    else: "bad"
value
"""
        )
        assert result.value == "ok"


class TestDotChainInfixHandoff:
    """Regression: multiline dot-chains can feed later continuation operators."""

    def test_dot_chain_then_add(self):
        result = run(
            """
obj := {x: 3}
value := obj
  .x
  + 2
value
"""
        )
        assert result.value == 5


class TestOuterContinuationOwnership:
    """Regression: nested bodies must not consume an outer continuation block."""

    def test_match_expr_then_outer_add(self):
        result = run(
            """
x :=
  match 1:
    1: 10
    else: 20
  + 2
x
"""
        )
        assert result.value == 12

    def test_once_block_then_outer_add(self):
        result = run(
            """
x :=
  once:
    1
  + 2
x
"""
        )
        assert result.value == 3

    def test_once_block_with_return_cont_then_outer_add(self):
        result = run(
            """
fn f():
  x :=
    once:
      if true:
        return
          10
      0
    + 2
  x
f()
"""
        )
        assert result.value == 10

    def test_once_block_with_throw_cont_then_outer_add(self):
        result = run(
            """
fn f():
  x :=
    once:
      if true:
        throw
          "bad"
      0
    + 2
  x
f() catch: "caught"
"""
        )
        assert result.value == "caught"

    def test_dot_chain_then_nullish(self):
        result = run(
            """
obj := {nested: nil}
value := obj
  .nested
  ?? 5
value
"""
        )
        assert result.value == 5

    def test_blank_line_breaks_catch_after_dedented_expression(self):
        with pytest.raises(ParseError):
            parse_source(
                """
value :=
  risky()

catch err: 1
""",
                use_indenter=True,
            )

    def test_blank_line_breaks_catch_after_multiline_chain(self):
        with pytest.raises(ParseError):
            parse_source(
                """
obj
  .f()

catch err: 1
""",
                use_indenter=True,
            )


class TestReturnPackContinuation:
    """Regression: multiline return must still support packs."""

    def test_return_pack_same_line(self):
        result = run(
            """
fn f():
  return 1, 2
a, b := f()
a + b
"""
        )
        assert result.value == 3

    def test_return_pack_continuation(self):
        """return followed by indented 1, 2 should produce a pack."""
        result = run(
            """
fn f():
  return
    1, 2
a, b := f()
a + b
"""
        )
        assert result.value == 3

    def test_return_pack_continuation_after_comma_newline(self):
        result = run(
            """
fn f():
  return
    1,
    2
a, b := f()
a + b
"""
        )
        assert result.value == 3

    def test_return_single_continuation(self):
        result = run(
            """
fn f():
  return
    42
f()
"""
        )
        assert result.value == 42

    def test_destructure_pack_continuation_after_comma_newline(self):
        result = run(
            """
a, b :=
  1,
  2
a + b
"""
        )
        assert result.value == 3

    def test_destructure_pack_same_line_then_continuation_after_comma(self):
        result = run(
            """
a, b := 1,
  2
a + b
"""
        )
        assert result.value == 3

    def test_fan_assign_pack_rhs_continuation_after_comma(self):
        result = run(
            """
obj := {x: 0, y: 0}
obj.{x, y} =
  1,
  2
obj.x + obj.y
"""
        )
        assert result.value == 3


class TestPowContinuation:
    """Regression: ** must participate in indented continuation."""

    def test_pow_in_continuation_block(self):
        result = run(
            """
x :=
  2
  ** 3
x
"""
        )
        assert result.value == 8

    def test_pow_direct_continuation(self):
        result = run(
            """
x := 2
  ** 3
x
"""
        )
        assert result.value == 8

    def test_pow_same_line(self):
        result = run(
            """
x := 2 ** 3
x
"""
        )
        assert result.value == 8


class TestApplyAssignContinuation:
    """Regression: .= on indented continuation line must parse."""

    def test_apply_assign_continuation_parses(self):
        """Parser accepts .= on an indented continuation line."""
        parse_source(
            """
fn double(x): x * 2
y := 5
y
  .= double
y
""",
            use_indenter=True,
        )

    def test_return_assign_expr_continuation(self):
        result = run(
            """
fn f():
  x := 1
  return
    x
    = 7
f()
"""
        )
        assert result.value == 7

    def test_return_compound_assign_expr_continuation(self):
        result = run(
            """
fn f():
  x := 1
  return
    x
    += 2
f()
"""
        )
        assert result.value == 3


class TestTildeContinuation:
    """Regression: ~ must be included in comparison continuation ops."""

    def test_tilde_continuation(self):
        result = run(
            """
x := 5
ok :=
  x
  ~ Int
ok
"""
        )
        assert result.value is True

    def test_tilde_same_line(self):
        result = run(
            """
x := 5
ok := x ~ Int
ok
"""
        )
        assert result.value is True


# ========================================================================
# Grouped-delimiter layout rejection tests (Phase 4)
# ========================================================================


class TestGroupedDelimiterRejection:
    """Reject flat or under-indented layouts inside grouped delimiters."""

    def test_flat_call_args(self):
        with pytest.raises(ParseError):
            parse_source(
                """
f(
a,
b,
)
""",
                use_indenter=True,
            )

    def test_flat_array_elements(self):
        with pytest.raises(ParseError):
            parse_source(
                """
[
a,
b,
]
""",
                use_indenter=True,
            )

    def test_flat_object_fields(self):
        with pytest.raises(ParseError):
            parse_source(
                """
{
x: 1,
y: 2,
}
""",
                use_indenter=True,
            )

    def test_under_indented_continuation_in_call(self):
        with pytest.raises(ParseError):
            parse_source(
                """
int(
  cols / 2
- offset
)
""",
                use_indenter=True,
            )

    def test_comma_led_call_line_must_be_indented(self):
        with pytest.raises(ParseError):
            parse_source(
                """
f(
  a
, b
)
""",
                use_indenter=True,
            )

    def test_comma_led_array_line_must_be_indented(self):
        with pytest.raises(ParseError):
            parse_source(
                """
[
  1
, 2
]
""",
                use_indenter=True,
            )

    def test_content_at_opener_column_not_closing_delimiter(self):
        with pytest.raises(ParseError):
            parse_source(
                """
f(
a
)
""",
                use_indenter=True,
            )

    def test_under_indented_in_array(self):
        with pytest.raises(ParseError):
            parse_source(
                """
items := [
  1,
2,
]
""",
                use_indenter=True,
            )

    def test_under_indented_in_nested_group(self):
        with pytest.raises(ParseError):
            parse_source(
                """
f(
  g(
  x,
  ),
)
""",
                use_indenter=True,
            )

    def test_single_line_groups_unaffected(self):
        """Single-line groups have no newlines => no enforcement."""
        parse_source("f(a, b)", use_indenter=True)
        parse_source("[1, 2, 3]", use_indenter=True)
        parse_source("{x: 1}", use_indenter=True)

    def test_closing_delimiter_at_opener_column(self):
        """Closing delimiters may appear at the opener's column."""
        parse_source(
            """
f(
  a,
  b,
)
""",
            use_indenter=True,
        )

    def test_closing_delimiter_at_outer_column(self):
        """Closing delimiters may appear at any outer level."""
        parse_source(
            """
result := f(
  a,
  b,
)
""",
            use_indenter=True,
        )

    def test_over_indented_closing_paren_rejected(self):
        with pytest.raises(LexError):
            parse_source(
                """
f(
  a
   )
""",
                use_indenter=True,
            )

    def test_over_indented_closing_brace_rejected(self):
        with pytest.raises(LexError):
            parse_source(
                """
{
  a: 1
   }
""",
                use_indenter=True,
            )

    def test_nested_groups_independent_baselines(self):
        """Each group enforces its own opener column."""
        parse_source(
            """
f(
  g(
    x,
  ),
  y,
)
""",
            use_indenter=True,
        )


class TestGroupedDelimiterMultilineSites:
    """Multiline layout inside grouped delimiters for all parse sites."""

    def test_method_call_multiline_args(self):
        result = run(
            """
obj := {add: fn(a, b): a + b}
obj.add(
  3,
  4,
)
""",
        )
        assert result.value == 7

    def test_wait_multiline(self):
        parse_source(
            """
wait(
  ch
)
""",
            use_indenter=True,
        )

    def test_spawn_multiline(self):
        parse_source(
            """
spawn(
  f
)
""",
            use_indenter=True,
        )

    def test_catch_filter_multiline(self):
        parse_source(
            """
try:
  throw "err"
catch(
  Str
) bind e:
  e
""",
            use_indenter=True,
        )

    def test_defer_after_multiline(self):
        parse_source(
            """
defer after(
  dep
):
  1
""",
            use_indenter=True,
        )

    def test_brace_body_multiline(self):
        result = run(
            """
fn f(): {
  x := 1
  x + 1
}
f()
""",
        )
        assert result.value == 2

    def test_brace_body_operator_led_line_not_merged(self):
        """Operator-led lines in brace bodies are separate statements, not continuations."""
        result = run(
            """
fn f(x): {
  x
  -1
}
f(5)
""",
        )
        assert result.value == -1

    def test_import_destructure_multiline(self):
        parse_source(
            """
import [
  a,
  b,
] "mod"
""",
            use_indenter=True,
        )

    def test_rebind_grouped_multiline(self):
        parse_source(
            """
=(
  x
)
""",
            use_indenter=True,
        )

    def test_nullsafe_multiline(self):
        parse_source(
            """
??(
  x
)
""",
            use_indenter=True,
        )

    def test_index_multiline(self):
        result = run(
            """
arr := [10, 20, 30]
arr[
  1
]
""",
        )
        assert result.value == 20

    def test_fieldfan_multiline(self):
        parse_source(
            """
obj.{
  x,
  y,
}
""",
            use_indenter=True,
        )

    def test_group_closer_does_not_pop_enclosing_block(self):
        """Closing delimiter aligned to outer column must not unwind enclosing indent."""
        result = run(
            """
fn f(x): x
fn g():
  f(
    1
  )
  42
g()
""",
        )
        assert result.value == 42

    def test_grouped_params_multiline(self):
        """Grouped contract sugar with multiline parameter names."""
        result = run(
            """
fn f((
  a,
  b,
) ~ Int):
  a + b
f(1, 2)
""",
        )
        assert result.value == 3

    def test_destructured_params_multiline(self):
        """Destructured parameter list with multiline fields."""
        result = run(
            """
fn f({
  a,
  b,
}):
  a + b
f({a: 1, b: 2})
""",
        )
        assert result.value == 3

    def test_index_default_multiline(self):
        """Multiline index with default arm must not parse as slice."""
        result = run(
            """
cfg := {a: 1}
cfg[
  "b",
  default: 99
]
""",
        )
        assert result.value == 99

    def test_listcomp_multiline_for(self):
        """Comprehension whose 'for' starts on the next indented line."""
        parse_source(
            """
[
  x * 2
  for x in items
]
""",
            use_indenter=True,
        )

    def test_setcomp_multiline_for(self):
        """Set comprehension with 'for' on next line."""
        parse_source(
            """
set{
  x
  for x in [1, 2]
}
""",
            use_indenter=True,
        )

    def test_dictcomp_multiline_for(self):
        """Dict comprehension with 'for' on next line."""
        parse_source(
            """
{
  x: x * 2
  for x in items
}
""",
            use_indenter=True,
        )

    def test_obj_member_body_brace_is_expression(self):
        """Object member body starting with { must parse as expression, not block."""
        result = run(
            """
obj := { m(): {x: 1} }
obj.m().x
""",
        )
        assert result.value == 1

    def test_obj_member_body_postfix_if(self):
        """Inline obj member body supports postfix-if."""
        result = run("obj := { m(): 1 if true }\nobj.m()")
        assert result.value == 1

    def test_obj_member_body_postfix_unless(self):
        """Inline obj member body supports postfix-unless."""
        result = run("obj := { m(): 1 unless false }\nobj.m()")
        assert result.value == 1

    def test_obj_member_body_destructure(self):
        """Inline obj member body supports destructuring assignment."""
        result = run("pair := [1, 2]\nobj := { m(): a, b := pair }\nobj.m()")
        assert [v.value for v in result.items] == [1.0, 2.0]

    def test_obj_member_body_comma_separated(self):
        """Comma after inline obj member body does not misparse."""
        result = run("obj := { m(): 5, y: 2 }\nobj.y")
        assert result.value == 2

    def test_obj_member_body_ident_comma(self):
        """Ident body followed by comma does not trigger destructure detection."""
        result = run("obj := { m(): 5, y: 2 }\nobj.y")
        assert result.value == 2

    def test_anon_fn_body_destructure(self):
        """Anonymous fn inline body supports destructuring assignment."""
        result = run("pair := [1, 2]\nf := fn(): a, b := pair\nf()")
        assert [v.value for v in result.items] == [1.0, 2.0]

    def test_anon_fn_body_ident_comma_in_array(self):
        """Anon fn inline body with ident, followed by comma in array."""
        result = run("x := 5\nresult := [fn(): x, 2]\nresult[1]")
        assert result.value == 2

    def test_indent_mismatch_inside_group_rejected(self):
        """Real indentation errors inside grouped constructs must still be caught."""
        with pytest.raises(Exception):
            parse_source(
                """
obj := {
  m():
    if true:
      1
     2
}
""",
                use_indenter=True,
            )

    def test_for_bracket_multiline(self):
        """for[...] binder list with multiline layout."""
        parse_source(
            """
for[
  i
] items:
  i
""",
            use_indenter=True,
        )

    def test_overspec_bracket_multiline(self):
        """Comprehension binder list with multiline layout."""
        parse_source(
            """
[x for[
  i
] items]
""",
            use_indenter=True,
        )

    def test_named_arg_value_multiline(self):
        """Named argument whose value starts on the next line inside (...)."""
        result = run(
            """
fn f(x): x
f(
  x:
    1
)
""",
        )
        assert result.value == 1

    def test_index_default_value_multiline(self):
        """Index default arm whose value starts on the next line inside [...]."""
        result = run(
            """
cfg := {a: 1}
cfg[
  "b",
  default:
    99
]
""",
        )
        assert result.value == 99

    def test_overspec_bracket_multiline_multi_binders(self):
        """Comprehension binder list with multiple binders across lines."""
        parse_source(
            """
[x for[
  a,
  b,
] items]
""",
            use_indenter=True,
        )

    def test_nested_body_not_merged_inside_group(self):
        """Statement body inside a grouped delimiter must not use column continuation.

        arr := [fn(x):\n  x\n  -1\n] — the body is two statements (x; -1),
        not a single expression (x - 1).
        """
        result = run(
            """
arr := [fn(x):
  x
  -1
]
arr[0](5)
""",
        )
        assert result.value == -1

    def test_brace_body_not_merged_inside_group(self):
        """Brace body inside a grouped delimiter must not use column continuation.

        arr := [fn(x): { x; -1 }] — the body has two statements,
        so arr[0](5) returns -1, not x - 1 = 4.
        """
        result = run(
            """
arr := [fn(x): {
  x
  -1
}]
arr[0](5)
""",
        )
        assert result.value == -1

    def test_setcomp_multiline_for_brace(self):
        """Set comprehension with iterator on next indented line inside set{...}."""
        parse_source(
            """
set{
  x
  for x in items
}
""",
            use_indenter=True,
        )

    def test_setcomp_multiline_if_brace(self):
        """Set comprehension with if-clause on next indented line inside set{...}."""
        parse_source(
            """
set{
  x
  for x in items
  if x > 0
}
""",
            use_indenter=True,
        )

    def test_dictcomp_multiline_value_brace(self):
        """Dict comprehension with value on next line after : inside {...}."""
        parse_source(
            """
{
  k:
    v
  for k in items
}
""",
            use_indenter=True,
        )

    def test_dictcomp_multiline_for_brace(self):
        """Dict comprehension with for on next indented line inside {...}."""
        parse_source(
            """
{
  k: v
  for k in items
}
""",
            use_indenter=True,
        )

    def test_no_indenter_multiline_parens(self):
        """parse_expr_fragment (use_indenter=False) must handle multiline (...)."""
        tree = parse_expr_fragment("(1 +\n  2)")
        # Should parse as (1 + 2) = 3
        assert tree is not None

    def test_no_indenter_multiline_call(self):
        """parse_expr_fragment (use_indenter=False) must handle multiline call args."""
        tree = parse_expr_fragment("foo(\n  1 +\n  2\n)")
        assert tree is not None


class TestGroupedBlankLineBreak:
    """Blank lines must not splice together disconnected expressions inside groups.

    Per spec Section 5: "blank lines do not splice together otherwise
    disconnected expressions."
    """

    def test_blank_line_breaks_continuation_in_parens(self):
        """Blank line between complete expr and operator inside () must reject."""
        with pytest.raises(ParseError):
            parse_source(
                "(\n  1\n\n  + 2\n)",
                use_indenter=True,
            )

    def test_blank_line_breaks_continuation_in_brackets(self):
        """Blank line between complete expr and operator inside [] must reject."""
        with pytest.raises(ParseError):
            parse_source(
                "[\n  1\n\n  + 2\n]",
                use_indenter=True,
            )

    def test_blank_line_between_comma_items_ok(self):
        """Blank lines between comma-separated items are fine."""
        parse_source(
            """
f(
  a,

  b,
)
""",
            use_indenter=True,
        )

    def test_blank_line_between_array_items_ok(self):
        """Blank lines between comma-separated array elements are fine."""
        parse_source(
            """
[
  1,

  2,
]
""",
            use_indenter=True,
        )

    def test_blank_line_before_closing_delimiter_ok(self):
        """Blank line before closing delimiter is fine."""
        parse_source(
            """
(
  1

)
""",
            use_indenter=True,
        )

    def test_blank_line_breaks_dot_chain_in_parens(self):
        """Blank line between complete expr and dot-chain inside () must reject."""
        with pytest.raises(ParseError):
            parse_source(
                "(\n  x\n\n  .y\n)",
                use_indenter=True,
            )

    def test_blank_line_breaks_array_comprehension_clause(self):
        """Blank line before a later array-comprehension clause must reject."""
        with pytest.raises(ParseError):
            parse_source(
                "[\n  x\n\n  for x in items\n]",
                use_indenter=True,
            )

    def test_blank_line_breaks_array_over_clause(self):
        """Blank line before a later array-comprehension over-clause must reject."""
        with pytest.raises(ParseError):
            parse_source(
                "[\n  x\n\n  over items\n]",
                use_indenter=True,
            )

    def test_blank_line_breaks_set_comprehension_clause(self):
        """Blank line before a later set-comprehension clause must reject."""
        with pytest.raises(ParseError):
            parse_source(
                "set{\n  x\n\n  for x in items\n}",
                use_indenter=True,
            )

    def test_blank_line_breaks_dict_comprehension_clause(self):
        """Blank line before a later dict-comprehension clause must reject."""
        with pytest.raises(ParseError):
            parse_source(
                "{\n  k: v\n\n  for k in items\n}",
                use_indenter=True,
            )


class TestGroupedActiveContBaseline:
    """Active continuation baseline enforcement inside groups.

    Once a continuation establishes a column, later lines in the same
    expression cannot regress below that column.
    """

    def test_under_indented_continuation_in_parens(self):
        """Deeper continuation then shallower must reject in ()."""
        with pytest.raises(ParseError):
            parse_source(
                "(\n  1 +\n    2\n  - 3\n)",
                use_indenter=True,
            )

    def test_same_indent_continuation_in_parens_ok(self):
        """Same-column continuation is fine."""
        parse_source(
            """
(
  1 +
  2 -
  3
)
""",
            use_indenter=True,
        )

    def test_deeper_then_deeper_ok(self):
        """Progressively deeper continuation is fine."""
        parse_source(
            """
(
  1 +
    2 +
      3
)
""",
            use_indenter=True,
        )

    def test_baseline_resets_after_comma(self):
        """Each comma-separated item has independent baseline."""
        parse_source(
            """
f(
  1 +
    2,
  3 +
  4,
)
""",
            use_indenter=True,
        )

    def test_under_indented_continuation_in_braces(self):
        """Deeper continuation then shallower must reject in {}."""
        with pytest.raises(ParseError):
            parse_source(
                "{\n  k: 1 +\n    2\n  - 3\n}",
                use_indenter=True,
            )

    def test_array_comp_resets_baseline_before_for_clause(self):
        """A completed multiline element must not poison a later for-clause."""
        parse_source(
            "[\n  x +\n    y\n  for z in items\n]",
            use_indenter=True,
        )

    def test_array_comp_resets_baseline_before_if_clause(self):
        """A completed multiline iterable must not poison a later if-clause."""
        parse_source(
            "[\n  x\n  for z in a +\n    b\n  if z > 0\n]",
            use_indenter=True,
        )


class TestBraceComprehensionGroupedLayout:
    """Brace comprehensions (set{}, dict {}) must consume grouped layout
    at the same boundaries as list comprehensions in [...].

    Brace groups disable auto-skip in advance(), so explicit
    _consume_grouped_layout() calls are required after for/over/if keywords.
    """

    def test_set_comp_for_on_next_line(self):
        """set{ expr for\\n  var in items } must parse."""
        result = run("items := [1, 2, 3]\n" "set{ x for\n" "  x in items }")
        assert sorted(i.value for i in result.items) == [1.0, 2.0, 3.0]

    def test_set_comp_if_on_next_line(self):
        """set{ expr for var in items\\n  if\\n    pred } must parse."""
        result = run(
            "items := [1, 2, 3]\n" "set{ x for x in items\n" "  if\n" "    x > 1 }"
        )
        assert sorted(i.value for i in result.items) == [2.0, 3.0]

    def test_dict_comp_value_on_next_line(self):
        """{ k:\\n    v\\n  for k in items } must parse."""
        result = run(
            "items := [1, 2]\n" "{\n" "  k:\n" "    k\n" "  for k in items\n" "}"
        )
        assert {k: v.value for k, v in result.slots.items()} == {"1": 1.0, "2": 2.0}

    def test_dict_comp_for_on_next_line(self):
        """{ k: v\\n  for\\n    k in items } must parse."""
        result = run(
            "items := [1, 2]\n" "{\n" "  k: k\n" "  for\n" "    k in items\n" "}"
        )
        assert {k: v.value for k, v in result.slots.items()} == {"1": 1.0, "2": 2.0}

    def test_dict_comp_if_on_next_line(self):
        """Dict comprehension with if-clause on next line."""
        result = run(
            "items := [1, 2, 3]\n"
            "{\n"
            "  k: k\n"
            "  for k in items\n"
            "  if\n"
            "    k > 1\n"
            "}"
        )
        assert {k: v.value for k, v in result.slots.items()} == {"2": 2.0, "3": 3.0}


class TestNoIndenterGroupedLayout:
    """In no-indenter mode (use_indenter=False), grouped multiline forms
    must parse without column validation.
    """

    def test_call_no_indenter(self):
        """f(\\n1,\\n2\\n) must parse in no-indenter mode."""
        parse_expr_fragment("f(\n1,\n2\n)")

    def test_paren_no_indenter(self):
        """(\\n1\\n) must parse in no-indenter mode."""
        parse_expr_fragment("(\n1\n)")

    def test_object_no_indenter(self):
        """{\\na: 1\\n} must parse in no-indenter mode."""
        parse_expr_fragment("{\na: 1\n}")

    def test_array_no_indenter(self):
        """[\\n1,\\n2\\n] must parse in no-indenter mode."""
        parse_expr_fragment("[\n1,\n2\n]")


class TestObjectNewlineSeparatedBaseline:
    """Continuation baseline must reset between newline-separated object items,
    not only on commas.
    """

    def test_newline_separated_items_after_continuation(self):
        """Each newline-separated object item has independent baseline."""
        result = run("obj := { a: 1 +\n" "      2\n" "  b: 3 +\n" "    4 }\n" "obj")
        assert result.slots["a"].value == 3.0
        assert result.slots["b"].value == 7.0

    def test_comma_separated_items_after_continuation(self):
        """Comma-separated items also have independent baseline (existing behavior)."""
        result = run("obj := { a: 1 +\n" "      2,\n" "  b: 3 +\n" "    4 }\n" "obj")
        assert result.slots["a"].value == 3.0
        assert result.slots["b"].value == 7.0


class TestDestructuredFieldGroupedLayout:
    """Destructured parameter field metadata (defaults and contracts) must
    consume grouped layout after `=` and `~` inside brace groups.
    """

    def test_default_on_next_line(self):
        """fn f({ a =\\n    1 }): ... must parse."""
        result = run("fn f({ a =\n" "    1 }): a\n" "f({})")
        assert result.value == 1.0

    def test_contract_on_next_line(self):
        """fn f({ a ~\\n    Int }): ... must parse."""
        parse_source(
            "fn f({ a ~\n" "    Int }): a",
            use_indenter=True,
        )

    def test_default_and_contract_on_next_lines(self):
        """fn f({ a =\\n    1 ~\\n    Int }): ... must parse."""
        result = run("fn f({ a =\n" "    1 ~\n" "    Int }): a\n" "f({})")
        assert result.value == 1.0


class TestModifierBracketSingleLine:
    """Atomic modifier brackets (wait[all], once[static], fan[par], etc.)
    must stay single-line — they are not expression containers.
    """

    def test_wait_modifier_rejects_multiline(self):
        with pytest.raises(ParseError):
            parse_source("wait[\n  all\n] 1", use_indenter=True)

    def test_once_modifier_rejects_multiline(self):
        with pytest.raises(ParseError):
            parse_source("once[\n  static\n]: 1", use_indenter=True)

    def test_wait_modifier_single_line_ok(self):
        parse_source("wait[all] 1", use_indenter=True)

    def test_once_modifier_single_line_ok(self):
        parse_source("once[static]: 1", use_indenter=True)

    def test_using_modifier_rejects_multiline(self):
        with pytest.raises(ParseError):
            parse_source("using[\n  h\n] resource:\n  h\n", use_indenter=True)

    def test_call_modifier_rejects_multiline(self):
        with pytest.raises(ParseError):
            parse_source("call[\n  ctx\n] target:\n  >\n", use_indenter=True)

    def test_fan_modifier_rejects_multiline(self):
        with pytest.raises(ParseError):
            parse_source("fan[\n  par\n] {1}\n", use_indenter=True)

    def test_using_modifier_single_line_ok(self):
        parse_source("using[h] resource:\n  h\n", use_indenter=True)

    def test_call_modifier_single_line_ok(self):
        parse_source("call[ctx] target:\n  >\n", use_indenter=True)

    def test_fan_modifier_single_line_ok(self):
        parse_source("fan[par] {1}\n", use_indenter=True)


class TestBinderListVsListLiteral:
    """Multiline list literals in comprehension specs must not be
    misparsed as binder lists.
    """

    def test_multiline_list_literal_not_stolen(self):
        """[x for [\\n  a,\\n  b,\\n]] must parse as list-literal iterable."""
        parse_source(
            "a := 1\n" "b := 2\n" "[x for [\n" "  a,\n" "  b,\n" "]]",
            use_indenter=True,
        )

    def test_multiline_binder_list_still_works(self):
        """for[\\n  i\\n] items must still parse as a binder list."""
        parse_source(
            "[x for[\n" "  i\n" "] items]",
            use_indenter=True,
        )

    def test_single_line_binder_list_ok(self):
        """for[i] items must parse as a binder list."""
        parse_source("[x for[i] items]", use_indenter=True)

    def test_multiline_over_list_literal_not_stolen(self):
        """[x over [\n  a,\n  b,\n]] must parse as list-literal iterable."""
        parse_source(
            "a := 1\n" "b := 2\n" "[x over [\n" "  a,\n" "  b,\n" "]]",
            use_indenter=True,
        )

    def test_multiline_over_binder_list_still_works(self):
        """over[\n  i\n] items must still parse as a binder list."""
        parse_source(
            "[x over[\n" "  i\n" "] items]",
            use_indenter=True,
        )

    def test_single_line_over_binder_list_ok(self):
        """over[i] items must parse as a binder list."""
        parse_source("[x over[i] items]", use_indenter=True)
