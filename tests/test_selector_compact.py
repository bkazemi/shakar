from __future__ import annotations

import pytest

from tests.support.harness import ShakarTypeError, run_runtime_case
from shakar_ref.types import ShkNumber, SelectorIndex, SelectorSlice
from shakar_ref.utils import compact_selector_parts


SCENARIOS = [
    pytest.param(
        "str(compact([`0:2`, `3:5`, `7`]))",
        ("string", "selector{0:5, 7}"),
        None,
        id="compact-basic-merge-and-index",
    ),
    pytest.param(
        "str(compact([`0:2, 7`, `3:5`]))",
        ("string", "selector{0:5, 7}"),
        None,
        id="compact-flattens-multipart-selectors",
    ),
    pytest.param(
        "str(compact([`0:3`, `0:3`]))",
        ("string", "selector{0:3}"),
        None,
        id="compact-collapses-duplicates",
    ),
    pytest.param(
        "str(compact([`1`, `2`, `3`]))",
        ("string", "selector{1:3}"),
        None,
        id="compact-indexes-merge-to-range",
    ),
    pytest.param(
        "str(compact(set{`0:5`, `0:2`, `3:7`}))",
        ("string", "selector{0:7}"),
        None,
        id="compact-accepts-set-input",
    ),
    pytest.param(
        "str(compact([`:3`, `2:5`]))",
        ("string", "selector{:5}"),
        None,
        id="compact-open-start-merges",
    ),
    pytest.param(
        "str(compact([`0:3`, `-5:-1`]))",
        ("string", "selector{-5:-1, 0:3}"),
        None,
        id="compact-sign-partitions",
    ),
    pytest.param(
        "str(compact([`0:3`, `0:6:2`]))",
        ("string", "selector{0:3, 0:6:2}"),
        None,
        id="compact-stepped-slice-passthrough",
    ),
    pytest.param(
        "str(compact([`0:6:2`, `0:6:2`]))",
        ("string", "selector{0:6:2}"),
        None,
        id="compact-stepped-duplicate-collapse",
    ),
    pytest.param(
        "str(compact([`0:<3`, `2:5`]))",
        ("string", "selector{0:5}"),
        None,
        id="compact-exclusive-stop-normalization",
    ),
    pytest.param(
        "compact([`0:2`, 1])",
        None,
        ShakarTypeError,
        id="compact-non-selector-element-error",
    ),
    pytest.param(
        "str(compact([]))",
        ("string", "selector{}"),
        None,
        id="compact-empty-input",
    ),
]


@pytest.mark.parametrize("source, expectation, expected_exc", SCENARIOS)
def test_compact_scenarios(source: str, expectation, expected_exc) -> None:
    run_runtime_case(source, expectation, expected_exc)


def _idx(n: int) -> SelectorIndex:
    return SelectorIndex(value=ShkNumber(float(n)))


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


class TestCompactSelectorParts:
    def test_clamp_partitions_do_not_merge(self) -> None:
        out = compact_selector_parts([_slc(0, 2), _slc(3, 5, clamp=True)])
        assert len(out) == 2
        assert isinstance(out[0], SelectorSlice)
        assert isinstance(out[1], SelectorSlice)
        assert out[0].clamp is False
        assert out[1].clamp is True

    def test_mixed_sign_slice_passes_through(self) -> None:
        out = compact_selector_parts([_slc(-2, 3), _slc(4, 6)])
        assert len(out) == 2
        assert isinstance(out[0], SelectorSlice)
        assert isinstance(out[1], SelectorSlice)
        assert (out[0].start, out[0].stop) == (-2, 3)
        assert (out[1].start, out[1].stop) == (4, 6)

    def test_open_start_merges_as_zero(self) -> None:
        out = compact_selector_parts([_slc(None, 2), _slc(3, 5)])
        assert len(out) == 1
        assert isinstance(out[0], SelectorSlice)
        assert out[0].start is None
        assert out[0].stop == 5
        assert out[0].step is None
        assert out[0].exclusive_stop is False

    def test_exclusive_stop_normalized_before_merge(self) -> None:
        out = compact_selector_parts([_slc(0, 3, exclusive_stop=True), _slc(2, 5)])
        assert len(out) == 1
        assert isinstance(out[0], SelectorSlice)
        assert (out[0].start, out[0].stop) == (0, 5)
        assert out[0].exclusive_stop is False

    def test_singleton_slice_stays_slice(self) -> None:
        # Real slices must not collapse to index — different OOB semantics.
        out = compact_selector_parts([_slc(7, 7)])
        assert len(out) == 1
        assert isinstance(out[0], SelectorSlice)
        assert (out[0].start, out[0].stop) == (7, 7)
        assert out[0].step is None

    def test_lone_index_stays_index(self) -> None:
        # An index that doesn't merge into a range should round-trip as index.
        out = compact_selector_parts([_idx(7)])
        assert len(out) == 1
        assert out[0] == _idx(7)

    def test_negative_slices_do_not_merge(self) -> None:
        # Negative indices are length-dependent; merging with interval
        # math is not union-preserving after normalization.
        out = compact_selector_parts([_slc(-5, -3), _slc(-3, -1)])
        assert len(out) == 2
        assert isinstance(out[0], SelectorSlice)
        assert isinstance(out[1], SelectorSlice)
        assert (out[0].start, out[0].stop) == (-5, -3)
        assert (out[1].start, out[1].stop) == (-3, -1)

    def test_stepped_duplicates_collapse(self) -> None:
        out = compact_selector_parts([_slc(0, 6, step=2), _slc(0, 6, step=2)])
        assert len(out) == 1
        assert isinstance(out[0], SelectorSlice)
        assert (out[0].start, out[0].stop, out[0].step) == (0, 6, 2)
