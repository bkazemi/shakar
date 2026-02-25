import pytest

pytest.importorskip("prompt_toolkit")

from shakar_ref.repl_highlight import GROUP_STYLE, _highlight_line


def test_once_keyword_highlighted_in_repl() -> None:
    spans = _highlight_line("once: 1")
    keyword_style = GROUP_STYLE["keyword"]

    assert any(style == keyword_style and text == "once" for style, text in spans)


@pytest.mark.parametrize(
    "source,modifier",
    [("once[lazy]: 1", "lazy"), ("once[lazy, static]: 1", "static")],
)
def test_once_modifiers_highlighted_in_repl(source: str, modifier: str) -> None:
    spans = _highlight_line(source)
    keyword_style = GROUP_STYLE["keyword"]

    assert any(style == keyword_style and text == modifier for style, text in spans)
