"""Tests for indented expression continuation.

Covers the feature described in docs/features/expression_continuation.md.
"""

import pytest

from tests.support.harness import parse_pipeline, run_runtime_case
from shakar_ref.parser_rd import ParseError, parse_source
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
