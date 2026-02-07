from __future__ import annotations

from pathlib import Path

from textwrap import dedent

import pytest

from tests.support.harness import (
    ShakarImportError,
    run_runtime_case,
)

SCENARIOS = [
    pytest.param(
        dedent(
            """\
            import "io"
            io.len
        """
        ),
        ("number", 7),
        None,
        id="import-io-binding",
    ),
    pytest.param(
        dedent(
            """\
            read_key := 1
            import[*] "io"
        """
        ),
        None,
        ShakarImportError,
        id="import-mixin-collision",
    ),
    pytest.param(
        'import "./tests/missing_import.shk"',
        None,
        ShakarImportError,
        id="import-file-missing",
    ),
]


@pytest.mark.parametrize("source, expectation, expected_exc", SCENARIOS)
def test_imports(source: str, expectation, expected_exc) -> None:
    run_runtime_case(source, expectation, expected_exc)


def test_import_file_basic(tmp_path: Path) -> None:
    module_path = tmp_path / "import_basic.shk"
    module_path.write_text("value := 41\n", encoding="utf-8")
    source = dedent(
        f"""\
        import "{module_path}"
        import_basic.value + 1
    """
    )
    run_runtime_case(source, ("number", 42), None)
