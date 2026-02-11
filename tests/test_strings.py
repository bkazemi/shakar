from __future__ import annotations

from textwrap import dedent

import pytest

from tests.support.harness import (
    CommandError,
    LexError,
    ShakarRuntimeError,
    ShakarTypeError,
    run_runtime_case,
)

SCENARIOS = [
    pytest.param(
        '", ".join(["a", "b"])',
        ("string", "a, b"),
        None,
        id="join-array",
    ),
    pytest.param(
        '"-".join("a", "b")',
        ("string", "a-b"),
        None,
        id="join-varargs",
    ),
    pytest.param(
        '"|".join("a", 1, true)',
        ("string", "a|1|true"),
        None,
        id="join-mixed",
    ),
    pytest.param(
        '"a,b,c".split(",")',
        ("array", ["a", "b", "c"]),
        None,
        id="string-split",
    ),
    pytest.param(
        '"abc".split("")',
        None,
        ShakarRuntimeError,
        id="string-split-empty",
    ),
    pytest.param(
        dedent(
            """\
            date := "2024-01-02"
            year, month, day := date ~~ r"(\\d{4})-(\\d{2})-(\\d{2})"
            [year, month, day]
        """
        ),
        ("array", ["2024", "01", "02"]),
        None,
        id="regex-match-captures",
    ),
    pytest.param(
        dedent(
            """\
            s := "ab"
            full, first := s ~~ r"(a)b"/f
            [full, first]
        """
        ),
        ("array", ["ab", "a"]),
        None,
        id="regex-match-full-flag",
    ),
    pytest.param(
        'r"foo"/z',
        None,
        LexError,
        id="regex-invalid-flag",
    ),
    pytest.param(
        dedent(
            """\
            rx := r"(\\d+)"
            ok := rx.test("a1")
            m := rx.search("b22")
            repl := rx.replace("c3", "x")
            if ok and m:
              [m[0], repl]
            else:
              ["", ""]
        """
        ),
        ("array", ["22", "cx"]),
        None,
        id="regex-methods",
    ),
    pytest.param(
        dedent(
            """\
            text := "foo
            bar"
            text ~~ r"foo
            bar"
            
        """
        ),
        ("array", ["foo\nbar"]),
        None,
        id="regex-multiline-literal",
    ),
    pytest.param(
        '"hello".high',
        ("number", 4),
        None,
        id="string-high",
    ),
    pytest.param(
        ' "".high',
        ("number", -1),
        None,
        id="string-high-empty",
    ),
    pytest.param(
        'raw"hi {name}\\n"',
        ("string", "hi {name}\\n"),
        None,
        id="raw-string-basic",
    ),
    pytest.param(
        dedent(
            """\
            s := "
              alpha
              beta
            "
            s
        """
        ),
        ("string", "alpha\nbeta\n"),
        None,
        id="string-multiline-dedent",
    ),
    pytest.param(
        'raw#"path "C:\\\\tmp"\\file"#',
        ("string", 'path "C:\\\\tmp"\\file'),
        None,
        id="raw-hash-string",
    ),
    pytest.param(
        'path := "file name.txt"; sh"cat {path}"',
        ("command", "cat 'file name.txt'"),
        None,
        id="shell-string-quote",
    ),
    pytest.param(
        'files := ["a.txt", "b 1.txt"]; sh"ls {files}"',
        ("command", "ls a.txt 'b 1.txt'"),
        None,
        id="shell-string-array",
    ),
    pytest.param(
        'flag := "-n 2"; file := "log 1.txt"; sh"head {{flag}} {file}"',
        ("command", "head -n 2 'log 1.txt'"),
        None,
        id="shell-string-raw-splice",
    ),
    pytest.param(
        'sh"echo \\"hi\\""',
        ("command", 'echo \\"hi\\"'),
        None,
        id="shell-string-escape-quote",
    ),
    pytest.param(
        dedent(
            """\
            cmd := sh"
              echo hi
              echo bye
            "
            cmd
        """
        ),
        ("command", "echo hi\necho bye\n"),
        None,
        id="shell-string-multiline-dedent",
    ),
    pytest.param(
        dedent(
            """\
            cmd := sh_raw"
              echo hi
              echo bye
            "
            cmd
        """
        ),
        ("command", "\n  echo hi\n  echo bye\n"),
        None,
        id="shell-string-raw",
    ),
    pytest.param(
        'msg := "hi"; res := (sh"printf {msg}").run(); res',
        ("string", "hi"),
        None,
        id="shell-run-stdout",
    ),
    pytest.param(
        'msg := "hi"; sh!"printf {msg}"',
        ("string", "hi"),
        None,
        id="shell-bang-stdout",
    ),
    pytest.param(
        '(sh"false").run()',
        None,
        CommandError,
        id="shell-run-code",
    ),
    pytest.param(
        'val := (sh"false").run() catch err: err.code',
        ("number", 1),
        None,
        id="shell-run-catch-code",
    ),
    pytest.param(
        'p"README.md".exists',
        ("bool", True),
        None,
        id="path-literal-exists",
    ),
    pytest.param(
        '(p"docs" / "shakar-design-notes.md").exists',
        ("bool", True),
        None,
        id="path-join-exists",
    ),
    pytest.param(
        dedent(
            """\
            name := "README.md"
            (p"{name}").read().len > 0
        """
        ),
        ("bool", True),
        None,
        id="path-interp-read",
    ),
    pytest.param(
        dedent(
            """\
            names := [ .name over p"*.md" ]
            "README.md" in names
        """
        ),
        ("bool", True),
        None,
        id="path-glob-contains",
    ),
    pytest.param(
        '[ .name over p"__shakar_no_match__*.zzz" ].len == 0',
        ("bool", True),
        None,
        id="path-glob-empty",
    ),
    pytest.param(
        'env"PATH" != nil',
        ("bool", True),
        None,
        id="env-exists",
    ),
    pytest.param(
        'env"SHAKAR_TEST_NONEXISTENT_VAR_12345" == nil',
        ("bool", True),
        None,
        id="env-missing-is-nil",
    ),
    pytest.param(
        'env"SHAKAR_TEST_NONEXISTENT_VAR_12345" ?? "fallback"',
        ("string", "fallback"),
        None,
        id="env-coalesce",
    ),
    pytest.param(
        'x := "PATH"; env"{x}" != nil',
        ("bool", True),
        None,
        id="env-interp",
    ),
    pytest.param(
        'env"PATH".name',
        ("string", "PATH"),
        None,
        id="env-name",
    ),
    pytest.param(
        'env"PATH".exists',
        ("bool", True),
        None,
        id="env-exists-prop",
    ),
    pytest.param(
        'env"SHAKAR_TEST_NONEXISTENT_VAR_12345".exists',
        ("bool", False),
        None,
        id="env-exists-false",
    ),
    pytest.param(
        'env"PATH".value != nil',
        ("bool", True),
        None,
        id="env-value-prop",
    ),
    pytest.param(
        'env"SHAKAR_TEST_VAR".assign("hello"); v := env"SHAKAR_TEST_VAR".value; env"SHAKAR_TEST_VAR".unset(); v',
        ("string", "hello"),
        None,
        id="env-assign-unset",
    ),
    pytest.param(
        'env"SHAKAR_TEST_VAR".assign("foo"); obj := {foo: 42}; v := obj[env"SHAKAR_TEST_VAR"]; env"SHAKAR_TEST_VAR".unset(); v',
        ("number", 42),
        None,
        id="env-as-object-key",
    ),
    pytest.param(
        'o := {(env"SHAKAR_TEST_NONEXISTENT_VAR_12345"): 1}; o[""]',
        None,
        ShakarTypeError,
        id="env-missing-object-literal-key-error",
    ),
    pytest.param(
        'env"SHAKAR_TEST_VAR".assign("world"); r := "world" in env"SHAKAR_TEST_VAR"; env"SHAKAR_TEST_VAR".unset(); r',
        ("bool", True),
        None,
        id="env-in-string",
    ),
    pytest.param(
        'env"SHAKAR_TEST_VAR".assign("hello world"); r := "wor" in env"SHAKAR_TEST_VAR"; env"SHAKAR_TEST_VAR".unset(); r',
        ("bool", True),
        None,
        id="env-substring-in-env",
    ),
    pytest.param(
        'env"SHAKAR_TEST_VAR".assign("hello"); v := env"SHAKAR_TEST_VAR"[0:3]; env"SHAKAR_TEST_VAR".unset(); v',
        ("string", "hel"),
        None,
        id="env-slicing",
    ),
    pytest.param(
        'env"SHAKAR_TEST_VAR".assign("hello"); v := env"SHAKAR_TEST_VAR".len; env"SHAKAR_TEST_VAR".unset(); v',
        ("number", 5),
        None,
        id="env-len-prop",
    ),
    pytest.param(
        'env"SHAKAR_TEST_A".assign("same"); env"SHAKAR_TEST_B".assign("same"); r := env"SHAKAR_TEST_A" == env"SHAKAR_TEST_B"; env"SHAKAR_TEST_A".unset(); env"SHAKAR_TEST_B".unset(); r',
        ("bool", True),
        None,
        id="env-compare-envs",
    ),
    pytest.param(
        'env"SHAKAR_TEST_A".assign("one"); env"SHAKAR_TEST_B".assign("two"); r := env"SHAKAR_TEST_A" != env"SHAKAR_TEST_B"; env"SHAKAR_TEST_A".unset(); env"SHAKAR_TEST_B".unset(); r',
        ("bool", True),
        None,
        id="env-compare-envs-diff",
    ),
    pytest.param(
        'env"SHAKAR_TEST_A".assign("hello"); env"SHAKAR_TEST_B".assign("world"); r := env"SHAKAR_TEST_A" + env"SHAKAR_TEST_B"; env"SHAKAR_TEST_A".unset(); env"SHAKAR_TEST_B".unset(); r',
        ("string", "helloworld"),
        None,
        id="env-concat-envs",
    ),
    pytest.param(
        'env"SHAKAR_TEST_VAR".assign("hello"); r := env"SHAKAR_TEST_VAR" + " world"; env"SHAKAR_TEST_VAR".unset(); r',
        ("string", "hello world"),
        None,
        id="env-concat-string",
    ),
    pytest.param(
        "env := 1; env + 1",
        ("number", 2),
        None,
        id="ident-env",
    ),
    pytest.param(
        "sh := 2; sh + 1",
        ("number", 3),
        None,
        id="ident-sh",
    ),
    pytest.param(
        "raw := 3; raw + 1",
        ("number", 4),
        None,
        id="ident-raw",
    ),
    pytest.param(
        "p := 4; p + 1",
        ("number", 5),
        None,
        id="ident-p",
    ),
    pytest.param(
        "r := 5; r + 1",
        ("number", 6),
        None,
        id="ident-r",
    ),
    pytest.param(
        'hook raw"event": 1',
        ("null", None),
        None,
        id="hook-raw-string",
    ),
    pytest.param(
        dedent(
            """\
            user := { name: "Ada", score: 5 }
            msg := "Name: {user.name}, score: {user.score}"
            msg
        """
        ),
        ("string", "Name: Ada, score: 5"),
        None,
        id="string-interp",
    ),
    pytest.param(
        dedent(
            """\
            value := 10
            text := "set {{value}} = {value}"
            text
        """
        ),
        ("string", "set {value} = 10"),
        None,
        id="string-interp-braces",
    ),
    pytest.param(
        dedent(
            """\
            user := { name: "Ada" }
            text := 'hi {user.name}!'
            text
        """
        ),
        ("string", "hi Ada!"),
        None,
        id="string-interp-single-quote",
    ),
    pytest.param(
        dedent(
            """\
            text := "line1\\nline2\\tend"
            text
        """
        ),
        ("string", "line1\nline2\tend"),
        None,
        id="string-escapes-basic",
    ),
    pytest.param(
        dedent(
            """\
            text := "smile: \\u{263A}"
            text
        """
        ),
        ("string", "smile: ☺"),
        None,
        id="string-escapes-unicode",
    ),
    pytest.param(
        dedent(
            """\
            text := "hex: \\x41\\x42"
            text
        """
        ),
        ("string", "hex: AB"),
        None,
        id="string-escapes-hex",
    ),
    pytest.param(
        dedent(
            """\
            text := "a\\b\\f\\0z"
            text
        """
        ),
        ("string", "a\x08\x0c\x00z"),
        None,
        id="string-escapes-control",
    ),
    pytest.param(
        dedent(
            """\
            s := "abc"
            s.repeat(3)
        """
        ),
        ("string", "abcabcabc"),
        None,
        id="string-repeat",
    ),
    pytest.param(
        dedent(
            """\
            s := "abc"
            s.repeat(0)
        """
        ),
        ("string", ""),
        None,
        id="string-repeat-zero",
    ),
]


@pytest.mark.parametrize("source, expectation, expected_exc", SCENARIOS)
def test_strings(source: str, expectation, expected_exc) -> None:
    run_runtime_case(source, expectation, expected_exc)
