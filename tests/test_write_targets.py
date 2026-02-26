from __future__ import annotations

from textwrap import dedent
from typing import List

from shakar_ref.eval.bind import _rebind_segment
from shakar_ref.eval.blocks import run_deferred_action, schedule_defer
from shakar_ref.eval.write_targets import WriteTarget, apply_write_target
from shakar_ref.runtime import (
    DeferredAction,
    Frame,
    ShkArray,
    ShkNil,
    ShkNumber,
)
from shakar_ref.tree import Tree, tree_label
from tests.support.harness import run_runtime_case


def test_noop_write_target_skips_dot_update() -> None:
    frame = Frame()
    frame.dot = ShkNumber(1.0)
    target = WriteTarget(
        kind="noop",
        owner=None,
        name_or_index=None,
        frame=frame,
        create=False,
    )
    updated = ShkNumber(2.0)

    result = apply_write_target(target, updated)

    assert result is updated
    assert isinstance(frame.dot, ShkNumber)
    assert frame.dot.value == 1.0


def test_index_target_is_captured_once_at_context_build_time() -> None:
    frame = Frame()
    arr = ShkArray([ShkNumber(10.0), ShkNumber(20.0)])
    index_evals = 0

    def fake_eval_index(_op, _frame, _eval_fn):
        nonlocal index_evals
        index_evals += 1
        return ShkNumber(1.0)

    context = _rebind_segment(
        arr,
        Tree("lv_index", []),
        "lv_index",
        frame,
        fake_eval_index,
        lambda _node, _frame: ShkNil(),
    )

    assert index_evals == 1
    apply_write_target(context.target, ShkNumber(30.0))
    apply_write_target(context.target, ShkNumber(40.0))
    assert index_evals == 1
    assert isinstance(arr.items[1], ShkNumber)
    assert arr.items[1].value == 40.0


def test_rebind_chain_field_index_noanchor() -> None:
    source = dedent(
        """\
        state := {arr: [{n: 1}, {n: 2}], idx: 1}
        =(state.$arr[state.idx]).n += 3
        state.arr[1].n
        """
    )
    run_runtime_case(source, ("number", 5), None)


def test_fan_writeback_updates_targets_and_preserves_dot_semantics() -> None:
    source = "state := {a: 1, b: 2}; =(state).{a, b} += 1; [., state.a, state.b]"
    run_runtime_case(source, ("array", [3.0, 2.0, 3.0]), None)


def test_run_deferred_action_call_preserves_context_and_nested_scope() -> None:
    origin = Frame(dot=ShkNumber(9.0), source="src", source_path="/tmp/demo.shk")
    seen: List[tuple[str, Frame]] = []
    payload = Tree("call_payload", [])
    nested_payload = Tree("nested_payload", [])

    def eval_fn(node, frame):
        label = tree_label(node)
        if label == "call_payload":
            seen.append(("call", frame))
            nested = DeferredAction(
                kind="call",
                payload_node=nested_payload,
                origin_frame=frame,
                saved_dot=frame.dot,
                saved_source=frame.source,
                saved_source_path=frame.source_path,
            )
            schedule_defer(frame, nested)
            return ShkNil()
        if label == "nested_payload":
            seen.append(("nested", frame))
            return ShkNil()
        raise AssertionError(f"Unexpected payload {label}")

    action = DeferredAction(
        kind="call",
        payload_node=payload,
        origin_frame=origin,
        saved_dot=origin.dot,
        saved_source=origin.source,
        saved_source_path=origin.source_path,
    )
    run_deferred_action(origin, action, eval_fn)

    assert [name for name, _ in seen] == ["call", "nested"]
    call_frame = seen[0][1]
    nested_frame = seen[1][1]
    assert call_frame.parent is origin
    assert nested_frame.parent is call_frame
    assert isinstance(call_frame.dot, ShkNumber)
    assert call_frame.dot.value == 9.0
    assert call_frame.source == "src"
    assert call_frame.source_path == "/tmp/demo.shk"


def test_run_deferred_action_block_preserves_context() -> None:
    origin = Frame(dot=ShkNumber(4.0), source="src2", source_path="/tmp/block.shk")
    seen: List[Frame] = []
    block_body = Tree("body", [Tree("block_payload", [])], attrs={"inline": True})

    def eval_fn(node, frame):
        if tree_label(node) != "block_payload":
            raise AssertionError("Unexpected block payload")
        seen.append(frame)
        return ShkNil()

    action = DeferredAction(
        kind="block",
        payload_node=block_body,
        origin_frame=origin,
        saved_dot=origin.dot,
        saved_source=origin.source,
        saved_source_path=origin.source_path,
    )
    run_deferred_action(origin, action, eval_fn)

    assert len(seen) == 1
    frame = seen[0]
    assert frame.parent is origin
    assert isinstance(frame.dot, ShkNumber)
    assert frame.dot.value == 4.0
    assert frame.source == "src2"
    assert frame.source_path == "/tmp/block.shk"
