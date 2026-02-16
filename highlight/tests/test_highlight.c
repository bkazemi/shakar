/*
 * Highlight regression tests.
 *
 * Build: cc -O0 -Wall -I../src -o test_highlight test_highlight.c ../src/lexer.c ../src/highlight.c
 * Run:   ./test_highlight
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "highlight.h"
#include "lexer.h"

static int g_pass = 0;
static int g_fail = 0;

/* Find span at (line, col_start). Returns NULL if not found. */
static HlSpan *find_span(HlBuf *hl, int line, int col) {
    for (int i = 0; i < hl->count; i++) {
        HlSpan *s = &hl->spans[i];
        if (s->line == line && s->col_start == col)
            return s;
    }
    return NULL;
}

/* Run highlight on source and return the HlBuf.
 * Caller must call hlbuf_free + lexer_free on the out params. */
static void run_hl(const char *src, Lexer *lex, HlBuf *hl) {
    int len = (int)strlen(src);
    lexer_init(lex, src, len, 0, 1);
    lexer_tokenize(lex);
    hlbuf_init(hl);
    highlight(src, len, &lex->tokens, hl);
}

static void check_span(const char *test, const char *src, int line, int col, HlGroup expected) {
    Lexer lex;
    HlBuf hl;
    run_hl(src, &lex, &hl);

    HlSpan *s = find_span(&hl, line, col);
    if (!s) {
        printf("FAIL %s: no span at line=%d col=%d\n", test, line, col);
        g_fail++;
    } else if (s->group != expected) {
        printf("FAIL %s: line=%d col=%d expected=%s got=%s\n", test, line, col,
               hl_group_name(expected), hl_group_name(s->group));
        g_fail++;
    } else {
        g_pass++;
    }

    hlbuf_free(&hl);
    lexer_free(&lex);
}

/* ================================================================ */
/* Tests                                                             */
/* ================================================================ */

/* Standalone implicit-subject dot should not leak TT_DOT as prev_sig
 * to the next statement's identifier. */
static void test_standalone_dot_no_leak(void) {
    const char *src = "sum := 0\nfor [1, 2, 3]:\n    sum += .\nprint(sum)";
    check_span("standalone_dot: '.' is implicit_subject", src, 2, 11, HL_IMPLICIT_SUBJECT);
    check_span("standalone_dot: 'print' is function", src, 3, 0, HL_FUNCTION);
}

/* .method() after dot => function, not property. */
static void test_dot_method_call(void) {
    check_span("dot_method: inline .method()", "x = .method()", 0, 5, HL_FUNCTION);
    check_span("dot_method: inline .prop", "x = .prop", 0, 5, HL_PROPERTY);
    check_span("dot_method: obj.method()", "obj.method()", 0, 4, HL_FUNCTION);
    check_span("dot_method: obj.prop", "obj.prop", 0, 4, HL_PROPERTY);
}

/* .prop should not become a method call across statement boundaries. */
static void test_dot_method_call_statement_boundary(void) {
    check_span("dot_method_boundary: newline split", "obj.prop\n(foo)", 0, 4, HL_PROPERTY);
    check_span("dot_method_boundary: semicolon split", "obj.prop; (foo)", 0, 4, HL_PROPERTY);
}

/* Newline before call in parenthesized grouping is still a method call. */
static void test_dot_method_call_paren_continuation(void) {
    const char *src = "x = (\n  obj.prop\n  (foo)\n)";
    check_span("dot_method_paren: multiline paren call", src, 1, 6, HL_FUNCTION);
}

/* .d.e.f chain: only leading dot is implicit_subject, rest is
 * property/punctuation. */
static void test_chain_no_override(void) {
    const char *src = "fn myFn(sum):\n  .d.e.f";
    check_span("chain: '.' is implicit_subject", src, 1, 2, HL_IMPLICIT_SUBJECT);
    check_span("chain: 'd' is property", src, 1, 3, HL_PROPERTY);
    check_span("chain: second '.' is punctuation", src, 1, 4, HL_PUNCTUATION);
    check_span("chain: 'e' is property", src, 1, 5, HL_PROPERTY);
    check_span("chain: third '.' is punctuation", src, 1, 6, HL_PUNCTUATION);
    check_span("chain: 'f' is property", src, 1, 7, HL_PROPERTY);
}

/* Same-indent dot after expr_end is implicit_subject, not
 * continuation (continuation requires greater indent). */
static void test_same_indent_dot_is_subject(void) {
    const char *src = "a()\n.b()";
    check_span("same_indent: '.' is implicit_subject", src, 1, 0, HL_IMPLICIT_SUBJECT);
}

/* Greater-indent dot after expr_end is continuation (punctuation). */
static void test_indented_dot_is_continuation(void) {
    const char *src = "a()\n  .b()";
    check_span("continuation: '.' is punctuation", src, 1, 2, HL_PUNCTUATION);
    check_span("continuation: 'b' is function", src, 1, 3, HL_FUNCTION);
}

/* Bracket-led statements must reset continuation state for later line-start dots. */
static void test_bracket_led_stmt_resets_continuation(void) {
    const char *src = "a()\n  .b()\n  (c)\n  .d()";
    check_span("bracket_reset: first continuation dot is punctuation", src, 1, 2, HL_PUNCTUATION);
    check_span("bracket_reset: post-bracket dot is implicit_subject", src, 3, 2,
               HL_IMPLICIT_SUBJECT);
}

/* Dot after non-expr-end (e.g. colon) is implicit_subject. */
static void test_dot_after_colon_is_subject(void) {
    const char *src = "for [1,2,3]:\n  .name";
    check_span("after_colon: '.' is implicit_subject", src, 1, 2, HL_IMPLICIT_SUBJECT);
    check_span("after_colon: 'name' is property", src, 1, 3, HL_PROPERTY);
}

/* Assignment followed by dot at same indent => implicit_subject. */
static void test_assign_then_dot(void) {
    const char *src = "a := 1\n.b()";
    check_span("assign_dot: '.' is implicit_subject", src, 1, 0, HL_IMPLICIT_SUBJECT);
}

/* Multi-line chained method calls with indented continuation dots. */
static void test_multiline_chain(void) {
    const char *src = "fn f():\n  a().b\n    .c()\n    .d()";
    /* a().b on line 1 */
    check_span("multiline: 'a' is function", src, 1, 2, HL_FUNCTION);
    check_span("multiline: '.' after a() is punctuation", src, 1, 5, HL_PUNCTUATION);
    check_span("multiline: 'b' is property", src, 1, 6, HL_PROPERTY);
    /* .c() on line 2 — indented continuation */
    check_span("multiline: '.' before c is punctuation", src, 2, 4, HL_PUNCTUATION);
    check_span("multiline: 'c' is function", src, 2, 5, HL_FUNCTION);
    /* .d() on line 3 — same-indent continuation */
    check_span("multiline: '.' before d is punctuation", src, 3, 4, HL_PUNCTUATION);
    check_span("multiline: 'd' is function", src, 3, 5, HL_FUNCTION);
}

int main(void) {
    test_standalone_dot_no_leak();
    test_dot_method_call();
    test_dot_method_call_statement_boundary();
    test_dot_method_call_paren_continuation();
    test_chain_no_override();
    test_same_indent_dot_is_subject();
    test_indented_dot_is_continuation();
    test_bracket_led_stmt_resets_continuation();
    test_dot_after_colon_is_subject();
    test_assign_then_dot();
    test_multiline_chain();

    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
