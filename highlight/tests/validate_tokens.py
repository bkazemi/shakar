#!/usr/bin/env python3
"""Validate C lexer output against Python lexer output.

Compares token types, lines, and columns between the two implementations.
Requires the native build: make -C highlight native
"""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from shakar_ref.lexer_rd import Lexer as PyLexer, LexError
from shakar_ref.token_types import TT

# Map Python TT enum names to C TT_ values (by position in the C enum).
# Build from the Python enum order — C enum mirrors it.
TT_NAME_TO_C: dict[str, int] = {}
_c_idx = 0
for member in TT:
    TT_NAME_TO_C[member.name] = _c_idx
    _c_idx += 1


def find_native_lib() -> Path:
    """Locate the compiled native shared library or executable."""
    base = Path(__file__).resolve().parent.parent
    exe = base / "shakar_hl_native"
    if exe.exists():
        return exe
    raise FileNotFoundError("Native binary not found. Run: make -C highlight native")


def tokenize_c(source: str, lib_path: Path) -> list[tuple[int, int, int, int, int]]:
    """Tokenize using the C lexer via subprocess (JSON protocol).

    Returns list of (type, line, col, start, len) tuples.
    """
    import subprocess
    import json

    result = subprocess.run(
        [str(lib_path)],
        input=source,
        capture_output=True,
        text=True,
        timeout=10,
    )

    if result.returncode != 0:
        return []

    tokens = []
    for line in result.stdout.strip().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            tokens.append(tuple(int(x) for x in parts[:5]))

    return tokens


def tokenize_python(source: str) -> list[tuple[int, int, int]]:
    """Tokenize using Python lexer.

    Returns list of (type_c_id, line, col) tuples.
    """
    try:
        lexer = PyLexer(source, track_indentation=False, emit_comments=True)
        tokens = lexer.tokenize()
    except LexError:
        return []

    result = []
    for tok in tokens:
        c_id = TT_NAME_TO_C.get(tok.type.name)
        if c_id is not None:
            result.append((c_id, tok.line, tok.column))

    return result


def collect_test_files() -> list[Path]:
    """Collect .sk test files."""
    files: list[Path] = []

    tests_dir = REPO_ROOT / "tests"
    if tests_dir.is_dir():
        files.extend(sorted(tests_dir.glob("**/*.sk")))

    web_dir = REPO_ROOT / "web"
    if web_dir.is_dir():
        files.extend(sorted(web_dir.glob("*.sk")))

    return files


def main() -> int:
    lib_path = find_native_lib()
    test_files = collect_test_files()

    if not test_files:
        print("No .sk test files found")
        return 1

    total = 0
    passed = 0
    failed_files: list[str] = []

    for path in test_files:
        source = path.read_text()
        total += 1

        py_tokens = tokenize_python(source)
        c_tokens = tokenize_c(source, lib_path)

        if not py_tokens and not c_tokens:
            passed += 1
            continue

        # Compare type, line, col
        py_tl = [(t, l, c) for t, l, c in py_tokens]
        c_tl = [(t, l, c) for t, l, c, *_ in c_tokens]

        if py_tl == c_tl:
            passed += 1
        else:
            failed_files.append(str(path.relative_to(REPO_ROOT)))
            # Show first mismatch
            for i in range(max(len(py_tl), len(c_tl))):
                py_tok = py_tl[i] if i < len(py_tl) else None
                c_tok = c_tl[i] if i < len(c_tl) else None
                if py_tok != c_tok:
                    print(f"  MISMATCH at token {i}:")
                    print(f"    Python: {py_tok}")
                    print(f"    C:      {c_tok}")
                    break

    print(f"\n{passed}/{total} files match")
    if failed_files:
        print("Failed:")
        for f in failed_files:
            print(f"  {f}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
