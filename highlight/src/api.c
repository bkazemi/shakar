/*
 * WASM-exported API for Shakar lexer + highlighter.
 *
 * Single static instance (worker is single-threaded).
 * Exported functions prefixed with shk_.
 */

#include "highlight.h"
#include "lexer.h"
#include <string.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#define EXPORT EMSCRIPTEN_KEEPALIVE
#else
#define EXPORT
#endif

/* Static instances — one per worker. */
static Lexer g_lexer;
static HlBuf g_hlbuf;
static int   g_initialized = 0;

/* Source buffer for WASM: JS writes source here, then calls shk_highlight. */
#define MAX_SRC (1024 * 1024) /* 1 MiB */
static char g_src[MAX_SRC];
static int  g_src_len = 0;

/* ======================================================================== */
/* Source buffer                                                              */
/* ======================================================================== */

EXPORT char *shk_src_ptr(void) {
    return g_src;
}

EXPORT void shk_set_src_len(int len) {
    g_src_len = (len > MAX_SRC) ? MAX_SRC : len;
}

/* ======================================================================== */
/* Lexer API                                                                 */
/* ======================================================================== */

EXPORT int shk_lex(int track_indent, int emit_comments) {
    if (g_initialized)
        lexer_free(&g_lexer);
    g_initialized = 1;

    lexer_init(&g_lexer, g_src, g_src_len, track_indent, emit_comments);
    return lexer_tokenize(&g_lexer);
}

EXPORT int shk_tok_count(void) {
    return g_lexer.tokens.count;
}

EXPORT int shk_tok_type(int i) {
    if (i < 0 || i >= g_lexer.tokens.count)
        return -1;
    return (int)g_lexer.tokens.toks[i].type;
}

EXPORT int shk_tok_line(int i) {
    if (i < 0 || i >= g_lexer.tokens.count)
        return -1;
    return g_lexer.tokens.toks[i].line;
}

EXPORT int shk_tok_col(int i) {
    if (i < 0 || i >= g_lexer.tokens.count)
        return -1;
    return g_lexer.tokens.toks[i].col;
}

EXPORT int shk_tok_start(int i) {
    if (i < 0 || i >= g_lexer.tokens.count)
        return -1;
    return g_lexer.tokens.toks[i].start;
}

EXPORT int shk_tok_len(int i) {
    if (i < 0 || i >= g_lexer.tokens.count)
        return -1;
    return g_lexer.tokens.toks[i].len;
}

EXPORT const char *shk_error(void) {
    if (g_lexer.has_error)
        return g_lexer.error_msg;
    return "";
}

/* Error location (1-based line/col, byte offset). */
EXPORT int shk_error_line(void) {
    if (g_lexer.has_error)
        return g_lexer.error_line;
    return 0;
}

EXPORT int shk_error_col(void) {
    if (g_lexer.has_error)
        return g_lexer.error_col;
    return 0;
}

EXPORT int shk_error_pos(void) {
    if (g_lexer.has_error)
        return g_lexer.error_pos;
    return 0;
}

/* ======================================================================== */
/* Highlight API                                                             */
/* ======================================================================== */

EXPORT int shk_highlight(void) {
    /* Lex first (no indent tracking, emit comments for highlighting) */
    if (g_initialized)
        lexer_free(&g_lexer);
    g_initialized = 1;

    lexer_init(&g_lexer, g_src, g_src_len, 0, 1);
    int r = lexer_tokenize(&g_lexer);
    if (r < 0)
        return -1;

    hlbuf_free(&g_hlbuf);
    hlbuf_init(&g_hlbuf);

    highlight(g_src, g_src_len, &g_lexer.tokens, &g_hlbuf);
    return g_hlbuf.count;
}

EXPORT int *shk_hl_spans_ptr(void) {
    return (int *)g_hlbuf.spans;
}

EXPORT int shk_hl_count(void) {
    return g_hlbuf.count;
}

EXPORT int shk_hl_line(int i) {
    if (i < 0 || i >= g_hlbuf.count)
        return -1;
    return g_hlbuf.spans[i].line;
}

EXPORT int shk_hl_col_start(int i) {
    if (i < 0 || i >= g_hlbuf.count)
        return -1;
    return g_hlbuf.spans[i].col_start;
}

EXPORT int shk_hl_col_end(int i) {
    if (i < 0 || i >= g_hlbuf.count)
        return -1;
    return g_hlbuf.spans[i].col_end;
}

EXPORT int shk_hl_group(int i) {
    if (i < 0 || i >= g_hlbuf.count)
        return -1;
    return (int)g_hlbuf.spans[i].group;
}

EXPORT const char *shk_hl_group_name(int i) {
    if (i < 0 || i >= g_hlbuf.count)
        return "";
    return hl_group_name(g_hlbuf.spans[i].group);
}

/* ======================================================================== */
/* Diagnostic API                                                            */
/* ======================================================================== */

static DiagBuf g_diagbuf;
static int     g_diag_ready = 0;

/* Run structural diagnostics on the current source.
 * Requires a separate lex pass with indent tracking for colon detection.
 * Returns diagnostic count. */
EXPORT int shk_diagnostics(void) {
    diagbuf_free(&g_diagbuf);
    diagbuf_init(&g_diagbuf);
    g_diag_ready = 0;

    /* Lex with indent tracking + comments for structural analysis. */
    Lexer lex;
    lexer_init(&lex, g_src, g_src_len, 1, 1);
    int r = lexer_tokenize(&lex);
    if (r < 0) {
        /* Lex error — report via the existing shk_error API; no structural
         * diagnostics possible on a failed lex. */
        lexer_free(&lex);
        g_diag_ready = 1;
        return 0;
    }

    /* Run structural pass for diagnostics only (skip full highlight). */
    HlBuf hl;
    hlbuf_init(&hl);
    structural_highlight(g_src, g_src_len, &lex.tokens, &hl, &g_diagbuf);

    hlbuf_free(&hl);
    lexer_free(&lex);
    g_diag_ready = 1;
    return g_diagbuf.count;
}

EXPORT int shk_diag_count(void) {
    return g_diag_ready ? g_diagbuf.count : 0;
}

EXPORT int shk_diag_line(int i) {
    if (!g_diag_ready || i < 0 || i >= g_diagbuf.count)
        return -1;
    return g_diagbuf.diags[i].line;
}

EXPORT int shk_diag_col_start(int i) {
    if (!g_diag_ready || i < 0 || i >= g_diagbuf.count)
        return -1;
    return g_diagbuf.diags[i].col_start;
}

EXPORT int shk_diag_col_end(int i) {
    if (!g_diag_ready || i < 0 || i >= g_diagbuf.count)
        return -1;
    return g_diagbuf.diags[i].col_end;
}

EXPORT int shk_diag_severity(int i) {
    if (!g_diag_ready || i < 0 || i >= g_diagbuf.count)
        return -1;
    return g_diagbuf.diags[i].severity;
}

EXPORT const char *shk_diag_message(int i) {
    if (!g_diag_ready || i < 0 || i >= g_diagbuf.count)
        return "";
    return g_diagbuf.diags[i].message;
}

/* ======================================================================== */
/* Native test main: reads stdin, tokenizes, prints tokens for validation.   */
/* Output: one line per token: type line col start len                       */
/* ======================================================================== */

#ifdef SHK_NATIVE_TEST
#include <stdio.h>

int main(void) {
    /* Read all of stdin into g_src */
    int len = 0;
    int ch;
    while ((ch = getchar()) != EOF && len < MAX_SRC) {
        g_src[len++] = (char)ch;
    }
    g_src_len = len;

    lexer_init(&g_lexer, g_src, g_src_len, 0, 1);
    g_initialized = 1;
    int count = lexer_tokenize(&g_lexer);

    if (count < 0) {
        fprintf(stderr, "Lex error: %s\n", g_lexer.error_msg);
        return 1;
    }

    for (int i = 0; i < count; i++) {
        Tok *t = &g_lexer.tokens.toks[i];
        printf("%d %d %d %d %d\n", (int)t->type, t->line, t->col, t->start, t->len);
    }

    lexer_free(&g_lexer);
    return 0;
}
#endif
