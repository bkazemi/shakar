from __future__ import annotations

from textwrap import dedent

import pytest

from tests.support.harness import (
    LexError,
    ShakarRuntimeError,
    ShakarTypeError,
    run_runtime_case,
)

SCENARIOS = [
    pytest.param(
        dedent(
            """\
            o := {}
            o.x = 1
        """
        ),
        None,
        ShakarRuntimeError,
        id="object-assign-requires-existing-field",
    ),
    pytest.param(
        "([1,2] + [3]).len",
        ("number", 3),
        None,
        id="array-concat",
    ),
    pytest.param(
        "arr := [1]; arr.push(2); arr.len",
        ("number", 2),
        None,
        id="array-push",
    ),
    pytest.param(
        "arr := [1]; arr.append(2); arr[1]",
        ("number", 2),
        None,
        id="array-append",
    ),
    pytest.param(
        "arr := [1, 2, 3]; arr.pop() + arr.len",
        ("number", 5),
        None,
        id="array-pop",
    ),
    pytest.param(
        "[].pop()",
        None,
        ShakarRuntimeError,
        id="array-pop-empty",
    ),
    pytest.param(
        dedent(
            """\
            xs := [10, 20, 30]
            xs.high
        """
        ),
        ("number", 2),
        None,
        id="array-high",
    ),
    pytest.param(
        " [].high",
        ("number", -1),
        None,
        id="array-high-empty",
    ),
    pytest.param(
        "obj := {a: 1, b: 2}; obj.len",
        ("number", 2),
        None,
        id="object-len",
    ),
    pytest.param(
        "{a: 1, b: 2}.keys()",
        ("array", ["a", "b"]),
        None,
        id="object-keys",
    ),
    pytest.param(
        "{a: 1, b: 2}.values()",
        ("array", [1, 2]),
        None,
        id="object-values",
    ),
    pytest.param(
        "[1, 2, 3].map&(. * 2)",
        ("array", [2, 4, 6]),
        None,
        id="array-map-basic",
    ),
    pytest.param(
        "[1, 2, 3, 4].filter&(. > 1)",
        ("array", [2, 3, 4]),
        None,
        id="array-filter-basic",
    ),
    pytest.param(
        '[0, 1, "", "hi", nil].filter&(.)',
        ("array", [1, "hi"]),
        None,
        id="array-filter-truthy",
    ),
    pytest.param(
        "arr := [0, 1]; o := {a: arr}; o.a[0,1] = 2; o.a[0] + o.a[1]",
        ("number", 4),
        None,
        id="selector-assign-broadcast",
    ),
    pytest.param(
        "arr := [0, 1, 2]; arr[`0:1`] = 5; arr[0] + arr[1] + arr[2]",
        ("number", 12),
        None,
        id="selectorliteral-assign-broadcast",
    ),
    pytest.param(
        "arr := [0, 1, 2]; arr[-1:2]",
        None,
        ShakarRuntimeError,
        id="slice-negative-start-positive-stop-error",
    ),
    pytest.param(
        "arr := [0, 1, 2, 3, 4]; result := arr[0:-1]; result[0] + result[3]",
        ("number", 3),
        None,
        id="slice-positive-start-negative-stop",
    ),
    pytest.param(
        "arr := [0, 1, 2, 3, 4]; result := arr[-3:-1]; result[0] + result[1]",
        ("number", 5),
        None,
        id="slice-both-negative",
    ),
    pytest.param(
        "set{1, 2, 1}",
        ("set", [1, 2]),
        None,
        id="setliteral-dedup",
    ),
    pytest.param(
        "set{3, 1, 2}",
        ("set", [1, 2, 3]),
        None,
        id="setliteral-sorted",
    ),
    pytest.param(
        "set{}",
        ("set", []),
        None,
        id="setliteral-empty",
    ),
    pytest.param(
        dedent(
            """\
            vals := set{1, 2, 1}
            total := 0
            for v in vals: total += v
            total
        """
        ),
        ("number", 3),
        None,
        id="setliteral-sum",
    ),
    pytest.param(
        "2 in set{1, 2, 3}",
        ("bool", True),
        None,
        id="set-in-membership",
    ),
    pytest.param(
        "5 in set{1, 2, 3}",
        ("bool", False),
        None,
        id="set-not-in-membership",
    ),
    pytest.param(
        "set{1, 2} + set{2, 3}",
        ("set", [1, 2, 3]),
        None,
        id="set-union",
    ),
    pytest.param(
        "set{1, 2, 3} - set{2}",
        ("set", [1, 3]),
        None,
        id="set-difference",
    ),
    pytest.param(
        "s := set{1, 2}; s.add(3); s.add(2); s",
        ("set", [1, 2, 3]),
        None,
        id="set-add",
    ),
    pytest.param(
        "s := set{1, 2, 3}; s[0] = 4; s",
        ("set", [2, 3, 4]),
        None,
        id="set-index-assign-resort",
    ),
    pytest.param(
        "s := set{1, 2}; s[0] = 2; s",
        ("set", [2]),
        None,
        id="set-index-assign-dedup",
    ),
    pytest.param(
        "s := set{1, 2, 3}; s[`0,1`] = 3; s",
        ("set", [3]),
        None,
        id="set-selector-assign-dedup",
    ),
    pytest.param(
        "s := set{1, 2, 3}; s.remove(2); s",
        ("set", [1, 3]),
        None,
        id="set-remove",
    ),
    pytest.param(
        "set{1, 2, 3}.remove(5)",
        None,
        ShakarRuntimeError,
        id="set-remove-absent",
    ),
    pytest.param(
        "set{1, 2, 3}.has(2)",
        ("bool", True),
        None,
        id="set-has-true",
    ),
    pytest.param(
        "set{1, 2, 3}.has(5)",
        ("bool", False),
        None,
        id="set-has-false",
    ),
    pytest.param(
        "set{1, 2, 3}.map&(. * 2)",
        ("set", [2, 4, 6]),
        None,
        id="set-map",
    ),
    pytest.param(
        "set{1, 2, 2, 3}.map&(. // 2)",
        ("set", [0, 1]),
        None,
        id="set-map-dedup",
    ),
    pytest.param(
        "set{1, 2, 3, 4}.filter&(. > 2)",
        ("set", [3, 4]),
        None,
        id="set-filter",
    ),
    pytest.param(
        "s := set{1, 2, 3, 4}; s.keep&(. > 2); s",
        ("set", [3, 4]),
        None,
        id="set-keep",
    ),
    pytest.param(
        "s := set{1, 2, 3}; s.update&(. * 10); s",
        ("set", [10, 20, 30]),
        None,
        id="set-update",
    ),
    pytest.param(
        "set{10, 20, 30}[1]",
        ("number", 20),
        None,
        id="set-index",
    ),
    pytest.param(
        "set{10, 20, 30}[`0,0`]",
        ("set", [10]),
        None,
        id="set-selector-dedup",
    ),
    pytest.param(
        "set{10, 20, 30}.len",
        ("number", 3),
        None,
        id="set-len",
    ),
    pytest.param(
        "set{n * n over [1, 2, 3, 2]}",
        ("set", [1, 4, 9]),
        None,
        id="set-comprehension",
    ),
    pytest.param(
        "set{1, 2, 3} ~ Set",
        ("bool", True),
        None,
        id="set-typeof",
    ),
    pytest.param(
        "if set{1}: 1 else: 0",
        ("number", 1),
        None,
        id="set-truthy-non-empty",
    ),
    pytest.param(
        "if set{}: 1 else: 0",
        ("number", 0),
        None,
        id="set-truthy-empty",
    ),
    pytest.param(
        "toSet([1, 2, 1])",
        ("set", [1, 2]),
        None,
        id="set-from-array",
    ),
    pytest.param(
        dedent(
            """\
            users := ["aa", "bb"]
            seen := users.len
            users[seen - 1]
            "" + seen
        """
        ),
        ("string", "2"),
        None,
        id="selector-base-anchor",
    ),
    pytest.param(
        dedent(
            """\
            myObj := {myArr: [10, 20, 30]}
            myObj and .myArr[.len-1]
        """
        ),
        ("number", 30),
        None,
        id="selector-local-dot-in-chain",
    ),
    pytest.param(
        dedent(
            """\
            state := {arr: [10, 20, 30], out: 0}
            state.out .= state.arr[.len-1]
            state.out
        """
        ),
        ("number", 30),
        None,
        id="selector-local-dot-apply-assign-rhs",
    ),
    pytest.param(
        dedent(
            """\
            v1 := 1 == `1, 2`
            v2 := 1 != `2, 3`
            v3 := 1 != `1, 2`
            [v1, v2, v3]
        """
        ),
        ("array", [True, True, False]),
        None,
        id="selector-compare-eq-any",
    ),
    pytest.param(
        dedent(
            """\
            sel := `0, 2`
            arr := [10, 20, 30]
            picked := arr[sel]
            picked[0] + picked[1]
        """
        ),
        ("number", 40),
        None,
        id="selector-literal-pick",
    ),
    pytest.param(
        dedent(
            """\
            sel := `:3`
            arr := [10, 20, 30, 40]
            picked := arr[sel]
            picked[0] + picked[1] + picked[2]
        """
        ),
        ("number", 60),
        None,
        id="selector-literal-open-start",
    ),
    pytest.param(
        dedent(
            """\
            sel := `2:`
            arr := [1, 2, 3, 4, 5]
            picked := arr[sel]
            picked[0] + picked[1] + picked[2]
        """
        ),
        ("number", 12),
        None,
        id="selector-literal-open-stop",
    ),
    pytest.param(
        dedent(
            """\
            sel := `5::-2`
            arr := [0, 1, 2, 3, 4, 5]
            picked := arr[sel]
            picked.len
        """
        ),
        ("number", 3),
        None,
        id="selector-literal-open-stop-neg-step",
    ),
    pytest.param(
        dedent(
            """\
            sel := `:1:-2`
            arr := [0, 1, 2, 3, 4, 5]
            picked := arr[sel]
            picked[0] + picked[1] + picked[2]
        """
        ),
        ("number", 9),
        None,
        id="selector-literal-open-start-neg-step",
    ),
    pytest.param(
        dedent(
            """\
            sel := `1:7:2`
            arr := [0, 1, 2, 3, 4, 5, 6, 7]
            picked := arr[sel]
            picked[0] + picked[1] + picked[2]
        """
        ),
        ("number", 9),
        None,
        id="selector-literal-slice-step",
    ),
    pytest.param(
        dedent(
            """\
            step := -1
            sel := `5:1:{step}`
            arr := [0, 1, 2, 3, 4, 5]
            picked := arr[sel]
            picked[0] + picked[1] + picked[2] + picked[3]
        """
        ),
        ("number", 14),
        None,
        id="selector-literal-slice-step-interp-neg",
    ),
    pytest.param(
        dedent(
            """\
            arr := [0]
            arr[`0x8000000000000000`]
        """
        ),
        None,
        LexError,
        id="selector-literal-hex-overflow",
    ),
    pytest.param(
        dedent(
            """\
            arr := [0]
            arr[`9223372036854775808`]
        """
        ),
        None,
        LexError,
        id="selector-literal-dec-overflow",
    ),
    pytest.param(
        dedent(
            """\
            arr := [0]
            arr[`0:0x8000000000000000`]
        """
        ),
        None,
        LexError,
        id="selector-literal-slice-stop-overflow",
    ),
    pytest.param(
        dedent(
            """\
            arr := [0]
            arr[`0:1:0x8000000000000000`]
        """
        ),
        None,
        LexError,
        id="selector-literal-slice-step-overflow",
    ),
    pytest.param(
        dedent(
            """\
            arr := [0]
            arr[`0:1:-0x8000000000000001`]
        """
        ),
        None,
        LexError,
        id="selector-literal-slice-step-neg-overflow",
    ),
    pytest.param(
        dedent(
            """\
            arr := [10, 20, 30, 40]
            slice := arr[1:3]
            slice[0] + slice[1]
        """
        ),
        ("number", 50),
        None,
        id="selector-slice",
    ),
    pytest.param(
        dedent(
            """\
            start := 1
            stop := 3
            sel := `{start}:{stop}`
            arr := [4, 5, 6, 7]
            sum := arr[sel]
            sum[0] + sum[1]
        """
        ),
        ("number", 11),
        None,
        id="selector-literal-interp",
    ),
    pytest.param(
        dedent(
            """\
            cfg := { db: { host: "db" } }
            calls := 0
            fn fallback():
              calls += 1
              return "localhost"
            found := cfg["db", default: {}]["host", default: fallback()]
            missing := cfg["port", default: fallback()]
            assert calls == 1
            "{found}-{missing}"
            
        """
        ),
        ("string", "db-localhost"),
        None,
        id="object-index-default",
    ),
    pytest.param(
        dedent(
            """\
            arr := [1, 2, 3]
            arr[0, default: 9]
        """
        ),
        None,
        ShakarTypeError,
        id="array-index-default-error",
    ),
    pytest.param(
        dedent(
            """\
            obj := {
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
            label_before + "|" + greet + "|" + ("" + dyn) + "|" + ("" + before) + "|" + ("" + after_dot) + "|" + ("" + after_index) + "|" + label_after
        """
        ),
        ("string", "Ada|hi GRACE|42|24|10|18|GRACE"),
        None,
        id="object-getter-setter",
    ),
    pytest.param(
        "o := {(sum(a, b)): a + b}; o.sum(2, 3)",
        ("number", 5),
        None,
        id="object-key-expr-method-sugar-simple-ident-args",
    ),
    pytest.param(
        dedent(
            """\
            fn g(x): "k" + x
            fn f(v): v
            x := "1"
            o := {(f(g(x))): 7}
            o["k1"]
        """
        ),
        ("number", 7),
        None,
        id="object-key-expr-nested-call-not-method-sugar",
    ),
    pytest.param(
        dedent(
            """\
            arr := [1, 2]
            arr.repeat(2)
        """
        ),
        ("array", [1, 2, 1, 2]),
        None,
        id="array-repeat",
    ),
    pytest.param(
        dedent(
            """\
            arr := [1, 2]
            arr.repeat(0)
        """
        ),
        ("array", []),
        None,
        id="array-repeat-zero",
    ),
    pytest.param(
        "a := [1, 2, 3]; a.update&(. * 2); a",
        ("array", [2, 4, 6]),
        None,
        id="array-update-basic",
    ),
    pytest.param(
        "a := [1, 2].update&(. + 1); a",
        ("array", [2, 3]),
        None,
        id="array-update-chaining",
    ),
    pytest.param(
        "a := [1, 2, 3, 4]; a.keep&(. > 2); a",
        ("array", [3, 4]),
        None,
        id="array-keep-basic",
    ),
    pytest.param(
        "o := {a: 1, b: 2}; o.update&(. + 10); o.a",
        ("number", 11),
        None,
        id="object-update-basic",
    ),
    # Object punning
    pytest.param(
        "x := 42; {x}.x",
        ("number", 42),
        None,
        id="obj-pun-single",
    ),
    pytest.param(
        "x := 1; y := 2; o := {x, y}; o.x + o.y",
        ("number", 3),
        None,
        id="obj-pun-multi",
    ),
    pytest.param(
        "x := 1; {x, y: 2}.y",
        ("number", 2),
        None,
        id="obj-pun-mixed",
    ),
    pytest.param(
        "x := 1; base := {y: 2}; {x, ...base}.y",
        ("number", 2),
        None,
        id="obj-pun-with-spread",
    ),
    pytest.param(
        dedent(
            """\
            x := 1
            y := 2
            {
              x
              y
            }.x
        """
        ),
        ("number", 1),
        None,
        id="obj-pun-newline-sep",
    ),
    pytest.param(
        "x := 1; {x,}.x",
        ("number", 1),
        None,
        id="obj-pun-trailing-comma",
    ),
    # Expression punning
    pytest.param(
        "score := 3; {score ** 2}.score",
        ("number", 9),
        None,
        id="obj-expr-pun-binop",
    ),
    pytest.param(
        "count := 10; {count + 1}.count",
        ("number", 11),
        None,
        id="obj-expr-pun-add",
    ),
    pytest.param(
        'name := "hello"; {name.upper()}.name',
        ("string", "HELLO"),
        None,
        id="obj-expr-pun-method-call",
    ),
    pytest.param(
        'score := 3; {score ** 2, label: "result", count: 5}.score',
        ("number", 9),
        None,
        id="obj-expr-pun-mixed-with-explicit",
    ),
    pytest.param(
        "a := 2; b := 3; o := {a * 10, b + 1}; o.a + o.b",
        ("number", 24),
        None,
        id="obj-expr-pun-multi",
    ),
    pytest.param(
        dedent(
            """\
            x := 5
            y := 2
            {
              x * 10
              y + 1
            }.x
        """
        ),
        ("number", 50),
        None,
        id="obj-expr-pun-multiline",
    ),
    pytest.param(
        "f := fn(x): x * 10; {f(1)}.f",
        ("number", 10),
        None,
        id="obj-expr-pun-direct-call",
    ),
    pytest.param(
        "cond := true; {cond ? 1 : 2}.cond",
        ("number", 1),
        None,
        id="obj-expr-pun-ternary",
    ),
    pytest.param(
        dedent(
            """\
            x := 5
            {
              x
              + 1
            }.x
        """
        ),
        ("number", 6),
        None,
        id="obj-expr-pun-multiline-continuation",
    ),
]


@pytest.mark.parametrize("source, expectation, expected_exc", SCENARIOS)
def test_collections(source: str, expectation, expected_exc) -> None:
    run_runtime_case(source, expectation, expected_exc)


# ---------------------------------------------------------------------------
# Selector set sorting & dedup (direct Python tests)
# ---------------------------------------------------------------------------

from shakar_ref.types import (
    ShkNumber,
    ShkSelector,
    SelectorIndex,
    SelectorSlice,
)
from shakar_ref.utils import normalize_set_items, shk_equals


def _idx(n: float) -> SelectorIndex:
    return SelectorIndex(value=ShkNumber(n))


def _slc(
    start: int | None = None,
    stop: int | None = None,
    step: int | None = None,
    exclusive_stop: bool = False,
    clamp: bool = False,
) -> SelectorSlice:
    return SelectorSlice(
        start=start,
        stop=stop,
        step=step,
        clamp=clamp,
        exclusive_stop=exclusive_stop,
    )


def _sel(*parts: SelectorIndex | SelectorSlice) -> ShkSelector:
    return ShkSelector(parts=list(parts))


class TestSelectorSetSorting:
    """Selector elements in sets should be deduplicated and sorted."""

    def test_index_selectors_sorted(self) -> None:
        """set{`3`, `1`, `2`} => [sel(1), sel(2), sel(3)]"""
        items = [_sel(_idx(3)), _sel(_idx(1)), _sel(_idx(2))]
        result = normalize_set_items(items)
        assert len(result) == 3
        assert result[0] == _sel(_idx(1))
        assert result[1] == _sel(_idx(2))
        assert result[2] == _sel(_idx(3))

    def test_duplicate_selectors_deduped(self) -> None:
        """set{`0:2`, `0:2`} => [sel(0:2)]"""
        s = _sel(_slc(0, 2))
        items = [s, _sel(_slc(0, 2))]
        result = normalize_set_items(items)
        assert len(result) == 1

    def test_slice_sorted_by_stop(self) -> None:
        """set{`0:5`, `0:2`, `0:4`} => sorted by stop ascending."""
        items = [_sel(_slc(0, 5)), _sel(_slc(0, 2)), _sel(_slc(0, 4))]
        result = normalize_set_items(items)
        assert len(result) == 3
        assert result[0] == _sel(_slc(0, 2))
        assert result[1] == _sel(_slc(0, 4))
        assert result[2] == _sel(_slc(0, 5))

    def test_none_start_before_concrete(self) -> None:
        """set{`:2`, `0:2`} => [sel(:2), sel(0:2)] — None start sorts first."""
        items = [_sel(_slc(0, 2)), _sel(_slc(None, 2))]
        result = normalize_set_items(items)
        assert len(result) == 2
        assert result[0] == _sel(_slc(None, 2))
        assert result[1] == _sel(_slc(0, 2))

    def test_multipart_shorter_first(self) -> None:
        """Shorter selector sorts before longer when prefix matches."""
        sel_short = _sel(_slc(0, 2))
        sel_long = _sel(_slc(0, 2), _slc(1, 5))
        items = [sel_long, sel_short]
        result = normalize_set_items(items)
        assert len(result) == 2
        assert result[0] == sel_short
        assert result[1] == sel_long

    def test_multipart_sorted_by_second_part(self) -> None:
        """set{`0:2, 3`, `0:2, 1:5`, `0:2, 1`}
        => [sel(0:2, 1), sel(0:2, 3), sel(0:2, 1:5)]
        Second part ordering: index < slice, so idx(1) < idx(3) < slc(1:5).
        """
        items = [
            _sel(_slc(0, 2), _idx(3)),
            _sel(_slc(0, 2), _slc(1, 5)),
            _sel(_slc(0, 2), _idx(1)),
        ]
        result = normalize_set_items(items)
        assert len(result) == 3
        assert result[0] == _sel(_slc(0, 2), _idx(1))
        assert result[1] == _sel(_slc(0, 2), _idx(3))
        assert result[2] == _sel(_slc(0, 2), _slc(1, 5))

    def test_index_before_slice(self) -> None:
        """Index parts sort before slice parts."""
        items = [_sel(_slc(0, 2)), _sel(_idx(0))]
        result = normalize_set_items(items)
        assert len(result) == 2
        assert result[0] == _sel(_idx(0))
        assert result[1] == _sel(_slc(0, 2))


class TestSelectorEquality:
    """ShkSelector equality in shk_equals."""

    def test_equal_index_selectors(self) -> None:
        assert shk_equals(_sel(_idx(1)), _sel(_idx(1)))

    def test_unequal_index_selectors(self) -> None:
        assert not shk_equals(_sel(_idx(1)), _sel(_idx(2)))

    def test_equal_slice_selectors(self) -> None:
        assert shk_equals(_sel(_slc(0, 5, 2)), _sel(_slc(0, 5, 2)))

    def test_unequal_slice_exclusive_stop(self) -> None:
        a = _sel(_slc(0, 5, exclusive_stop=False))
        b = _sel(_slc(0, 5, exclusive_stop=True))
        assert not shk_equals(a, b)

    def test_unequal_slice_clamp(self) -> None:
        a = _sel(_slc(0, 5, clamp=False))
        b = _sel(_slc(0, 5, clamp=True))
        assert not shk_equals(a, b)

    def test_different_part_count(self) -> None:
        assert not shk_equals(_sel(_idx(1)), _sel(_idx(1), _idx(2)))

    def test_different_part_kind(self) -> None:
        assert not shk_equals(_sel(_idx(0)), _sel(_slc(0, 1)))

    def test_none_vs_zero_slice_start(self) -> None:
        """None start and 0 start are distinct."""
        assert not shk_equals(_sel(_slc(None, 2)), _sel(_slc(0, 2)))
