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

/* Find the line/col of the nth occurrence of a character in src.
 * nth is zero-based. Returns 1 on success, 0 if not found. */
static int find_nth_char_pos(const char *src, char ch, int nth, int *line_out, int *col_out) {
    int line = 0;
    int col = 0;
    int seen = 0;

    for (int i = 0; src[i] != '\0'; i++) {
        char cur = src[i];
        if (cur == ch) {
            if (seen == nth) {
                *line_out = line;
                *col_out = col;
                return 1;
            }
            seen++;
        }

        if (cur == '\n') {
            line++;
            col = 0;
        } else {
            col++;
        }
    }

    return 0;
}

/* Resolve nth character position then check the highlight group at that span. */
static void check_nth_char_span(const char *test, const char *src, char ch, int nth,
                                HlGroup expected) {
    int line = 0;
    int col = 0;
    if (!find_nth_char_pos(src, ch, nth, &line, &col)) {
        printf("FAIL %s: could not find %d occurrence(s) of '%c'\n", test, nth + 1, ch);
        g_fail++;
        return;
    }
    check_span(test, src, line, col, expected);
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

/* Guard branch pipes should be keywords in both multiline and inline forms. */
static void test_guard_pipe_keywords(void) {
    const char *multiline = "fn temp_status(t):\n"
                            "    t > 100:\n"
                            "        return \"boiling\"\n"
                            "    | t > 50:\n"
                            "        return \"hot\"\n"
                            "    |:\n"
                            "        return \"cold\"";

    check_span("guard_pipe: branch '|' is keyword", multiline, 3, 4, HL_KEYWORD);
    check_span("guard_pipe: else '|' is keyword", multiline, 5, 4, HL_KEYWORD);
    check_span("guard_pipe: else ':' is keyword", multiline, 5, 5, HL_KEYWORD);

    check_span("guard_pipe: inline branch '|' is keyword", "a: b | c: d", 0, 5, HL_KEYWORD);
}

/* Match pattern separators use operator-style pipes, not guard keywords. */
static void test_match_pipe_not_guard_keyword(void) {
    check_span("match_pipe: '|' stays operator", "\"a\" | \"b\": x", 0, 4, HL_OPERATOR);
}

/* Colons nested inside [] must not trigger guard-pipe keywording. */
static void test_pipe_with_nested_colons_not_guard_keyword(void) {
    const char *src = "arr := [1, 2, 3]\n"
                      "x := [1]\n"
                      "match x:\n"
                      "    arr[0:1] | arr[1:2]: 1\n"
                      "    else: 0";
    check_span("nested_colon_slice_match: '|' stays operator", src, 3, 13, HL_OPERATOR);
}

/* Guard pipe keywording must never apply inside grouping depth > 0. */
static void test_grouped_pipe_not_guard_keyword(void) {
    check_span("grouped_paren_pipe: '|' stays operator", "head: (a | b): c", 0, 9, HL_OPERATOR);
    check_span("grouped_paren_colon: ':' stays punctuation", "head: (a | b): c", 0, 13,
               HL_PUNCTUATION);

    check_span("grouped_bracket_pipe: '|' stays operator", "head: [a | b: c]: d", 0, 9,
               HL_OPERATOR);
    check_span("grouped_bracket_colon: ':' stays punctuation", "head: [a | b: c]: d", 0, 12,
               HL_PUNCTUATION);

    check_span("grouped_brace_pipe: '|' stays operator", "head: {k: a | b: c}: d", 0, 12,
               HL_OPERATOR);
    check_span("grouped_brace_colon: ':' stays punctuation", "head: {k: a | b: c}: d", 0, 15,
               HL_PUNCTUATION);

    check_span("grouped_pipe_colon_pair: '|' stays operator", "head: (a |: b): c", 0, 9,
               HL_OPERATOR);
    check_span("grouped_pipe_colon_pair: ':' stays punctuation", "head: (a |: b): c", 0, 10,
               HL_PUNCTUATION);
}

/* Exhaustive matrix for grouped pipes: all grouped variants stay operator. */
static void test_grouped_pipe_matrix_not_guard_keyword(void) {
    const char *case_paren = "head: (a | b): c";
    const char *case_bracket = "head: [a | b: c]: d";
    const char *case_brace = "head: {k: a | b: c}: d";
    const char *case_paren_pipe_colon = "head: (a |: b): c";
    const char *case_bracket_pipe_colon = "head: [a |: b]: c";
    const char *case_brace_pipe_colon = "head: {k: a |: b}: c";
    const char *case_nested = "head: ((a | b)): c";
    const char *case_mixed_inline = "head: left | (a | b): right";
    const char *case_mixed_multiline = "head:\n    left\n| (a | b):\n    right";

    check_nth_char_span("matrix: paren '|'", case_paren, '|', 0, HL_OPERATOR);
    check_nth_char_span("matrix: bracket '|'", case_bracket, '|', 0, HL_OPERATOR);
    check_nth_char_span("matrix: brace '|'", case_brace, '|', 0, HL_OPERATOR);

    check_nth_char_span("matrix: paren '|:' pipe", case_paren_pipe_colon, '|', 0, HL_OPERATOR);
    check_nth_char_span("matrix: paren '|:' colon", case_paren_pipe_colon, ':', 2, HL_PUNCTUATION);
    check_nth_char_span("matrix: bracket '|:' pipe", case_bracket_pipe_colon, '|', 0, HL_OPERATOR);
    check_nth_char_span("matrix: bracket '|:' colon", case_bracket_pipe_colon, ':', 2,
                        HL_PUNCTUATION);
    check_nth_char_span("matrix: brace '|:' pipe", case_brace_pipe_colon, '|', 0, HL_OPERATOR);
    check_nth_char_span("matrix: brace '|:' colon", case_brace_pipe_colon, ':', 3, HL_PUNCTUATION);

    check_nth_char_span("matrix: nested '|'", case_nested, '|', 0, HL_OPERATOR);

    /* Mixed inline: first pipe is a guard continuation at depth 0, inner pipe is grouped. */
    check_nth_char_span("matrix: mixed inline outer '|'", case_mixed_inline, '|', 0, HL_KEYWORD);
    check_nth_char_span("matrix: mixed inline inner '|'", case_mixed_inline, '|', 1, HL_OPERATOR);

    /* Mixed multiline: same expectation as inline form. */
    check_nth_char_span("matrix: mixed multiline outer '|'", case_mixed_multiline, '|', 0,
                        HL_KEYWORD);
    check_nth_char_span("matrix: mixed multiline inner '|'", case_mixed_multiline, '|', 1,
                        HL_OPERATOR);
}

/* Match separators must stay operator even when prior patterns include ':' expressions. */
static void test_match_separator_with_prior_colon_expr_not_guard_keyword(void) {
    const char *ternary_case = "match x:\n"
                               "    a ? 1 : 2 | b: y\n"
                               "    else: z";
    const char *catch_case = "match x:\n"
                             "    a catch e: b | c: y\n"
                             "    else: z";
    const char *catch_sugar_case = "match x:\n"
                                   "    a @@ e: b | c: y\n"
                                   "    else: z";
    const char *selector_case = "match x:\n"
                                "    `0:1` | c: y\n"
                                "    else: z";
    const char *once_case = "match x:\n"
                            "    once: a | c: y\n"
                            "    else: z";
    const char *nested_once_case = "match x:\n"
                                   "    once: once: a | c: y\n"
                                   "    else: z";

    check_nth_char_span("match_ternary: '|' stays operator", ternary_case, '|', 0, HL_OPERATOR);
    check_nth_char_span("match_catch: '|' stays operator", catch_case, '|', 0, HL_OPERATOR);
    check_nth_char_span("match_catch_sugar: '|' stays operator", catch_sugar_case, '|', 0,
                        HL_OPERATOR);
    check_nth_char_span("match_selector: '|' stays operator", selector_case, '|', 0, HL_OPERATOR);
    check_nth_char_span("match_once: '|' stays operator", once_case, '|', 0, HL_OPERATOR);
    check_nth_char_span("match_nested_once: '|' stays operator", nested_once_case, '|', 0,
                        HL_OPERATOR);
}

/* Inline guard continuation remains keyword when prior branch body has ':' expressions. */
static void test_inline_guard_with_colon_expr_in_body_stays_keyword(void) {
    const char *ternary_body = "true: x ? 1 : 2 | false: y";
    const char *catch_body = "true: a catch e: b | false: y";
    const char *catch_sugar_body = "true: a @@ e: b | false: y";
    const char *selector_body = "true: `0:1` | false: y";
    const char *once_body = "true: once: a | false: y";
    const char *nested_once_body = "true: once: once: a | false: y";

    check_nth_char_span("guard_body_ternary: '|' is keyword", ternary_body, '|', 0, HL_KEYWORD);
    check_nth_char_span("guard_body_catch: '|' is keyword", catch_body, '|', 0, HL_KEYWORD);
    check_nth_char_span("guard_body_catch_sugar: '|' is keyword", catch_sugar_body, '|', 0,
                        HL_KEYWORD);
    check_nth_char_span("guard_body_selector: '|' is keyword", selector_body, '|', 0, HL_KEYWORD);
    check_nth_char_span("guard_body_once: '|' is keyword", once_body, '|', 0, HL_KEYWORD);
    check_nth_char_span("guard_body_nested_once: '|' is keyword", nested_once_body, '|', 0,
                        HL_KEYWORD);
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
    test_guard_pipe_keywords();
    test_match_pipe_not_guard_keyword();
    test_pipe_with_nested_colons_not_guard_keyword();
    test_grouped_pipe_not_guard_keyword();
    test_grouped_pipe_matrix_not_guard_keyword();
    test_match_separator_with_prior_colon_expr_not_guard_keyword();
    test_inline_guard_with_colon_expr_in_body_stays_keyword();

    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
