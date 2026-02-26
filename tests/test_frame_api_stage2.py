from __future__ import annotations

from typing import List, Tuple

from shakar_ref import types as shk_types
from shakar_ref.eval.mutation import set_field_value, set_index_value
from shakar_ref.runtime import Frame, ShkArray, ShkNumber, ShkObject, ShkValue


def _install_ownership_counters(monkeypatch) -> Tuple[List[ShkValue], List[ShkValue]]:
    retained: List[ShkValue] = []
    released: List[ShkValue] = []

    def _retain(value: ShkValue) -> ShkValue:
        retained.append(value)
        return value

    def _release(value: ShkValue) -> None:
        released.append(value)

    monkeypatch.setattr(shk_types, "retain_value", _retain)
    monkeypatch.setattr(shk_types, "release_value", _release)

    return retained, released


def test_frame_shadowing_let_vars_builtins_precedence() -> None:
    frame = Frame()

    frame.define_new_ident("print", ShkNumber(10.0))
    first = frame.get("print")
    assert isinstance(first, ShkNumber)
    assert first.value == 10.0

    frame.push_let_scope()
    frame.define_let("print", ShkNumber(20.0))
    second = frame.get("print")
    assert isinstance(second, ShkNumber)
    assert second.value == 20.0

    frame.pop_let_scope()
    third = frame.get("print")
    assert isinstance(third, ShkNumber)
    assert third.value == 10.0


def test_frame_define_get_set_ownership_contract(monkeypatch) -> None:
    parent = Frame()
    retained, released = _install_ownership_counters(monkeypatch)
    frame = Frame(parent=parent)

    first = ShkNumber(1.0)
    frame.define("x", first)
    assert retained == [first]
    assert released == []

    # get() is a borrowed read and must not transfer ownership.
    assert frame.get("x") is first
    assert retained == [first]
    assert released == []

    second = ShkNumber(2.0)
    frame.set("x", second)
    assert frame.get("x") is second
    assert retained == [first, second]
    assert released == [first]


def test_container_replacement_writes_use_ownership_hooks(monkeypatch) -> None:
    parent = Frame()
    retained, released = _install_ownership_counters(monkeypatch)
    frame = Frame(parent=parent)

    old_field = ShkNumber(1.0)
    obj = ShkObject({"a": old_field})
    new_field = ShkNumber(2.0)
    set_field_value(obj, "a", new_field, frame, create=False)

    old_item = ShkNumber(7.0)
    arr = ShkArray([old_item])
    new_item = ShkNumber(8.0)
    set_index_value(arr, ShkNumber(0.0), new_item, frame)

    assert obj.slots["a"] is new_field
    assert arr.items[0] is new_item
    assert new_field in retained
    assert new_item in retained
    assert old_field in released
    assert old_item in released
