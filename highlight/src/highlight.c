/*
 * Shakar highlight — lexical + structural (token-stream) highlighting.
 *
 * Mirrors highlight_server.py: _tokens_to_highlights, _classify_identifier,
 * _classify_string, _duration_size_spans, _token_spans.
 */

#include "highlight.h"
#include <ctype.h>
#include <stdlib.h>
#include <string.h>

/* ======================================================================== */
/* HlBuf                                                                     */
/* ======================================================================== */

void hlbuf_init(HlBuf *b) {
    b->spans = 0;
    b->count = 0;
    b->capacity = 0;
}

void hlbuf_free(HlBuf *b) {
    free(b->spans);
    b->spans = 0;
    b->count = 0;
    b->capacity = 0;
}

static void hlbuf_push(HlBuf *b, int line, int cs, int ce, HlGroup g) {
    if (ce <= cs)
        return;
    if (b->count >= b->capacity) {
        int cap = b->capacity ? b->capacity * 2 : 256;
        b->spans = realloc(b->spans, cap * sizeof(HlSpan));
        b->capacity = cap;
    }
    HlSpan *s = &b->spans[b->count++];
    s->line = line;
    s->col_start = cs;
    s->col_end = ce;
    s->group = g;
}

/* ======================================================================== */
/* Group name table                                                          */
/* ======================================================================== */

static const char *GROUP_NAMES[] = {
    "", /* HL_NONE */
    "keyword", "boolean",  "constant",         "number",      "unit",       "string",   "regex",
    "path",    "comment",  "operator",         "punctuation", "identifier", "function", "decorator",
    "hook",    "property", "implicit_subject", "type",
};

const char *hl_group_name(HlGroup g) {
    if (g >= 0 && g < HL__COUNT)
        return GROUP_NAMES[g];
    return "";
}

/* ======================================================================== */
/* Token → base group mapping                                                */
/* ======================================================================== */

static HlGroup base_group(TT t) {
    switch (t) {
    /* Keywords */
    case TT_IF:
    case TT_ELIF:
    case TT_ELSE:
    case TT_UNLESS:
    case TT_WHILE:
    case TT_FOR:
    case TT_MATCH:
    case TT_IN:
    case TT_BREAK:
    case TT_CONTINUE:
    case TT_RETURN:
    case TT_FN:
    case TT_LET:
    case TT_WAIT:
    case TT_SPAWN:
    case TT_USING:
    case TT_CALL:
    case TT_DEFER:
    case TT_AFTER:
    case TT_THROW:
    case TT_CATCH:
    case TT_ASSERT:
    case TT_DBG:
    case TT_DECORATOR:
    case TT_GET:
    case TT_SET:
    case TT_HOOK:
    case TT_OVER:
    case TT_BIND:
    case TT_IMPORT:
    case TT_ANY:
    case TT_ALL:
    case TT_GROUP:
    case TT_FAN:
    case TT_NOT:
    case TT_AND:
    case TT_OR:
    case TT_IS:
        return HL_KEYWORD;

    case TT_TRUE:
    case TT_FALSE:
        return HL_BOOLEAN;
    case TT_NIL:
        return HL_CONSTANT;

    case TT_NUMBER:
    case TT_DURATION:
    case TT_SIZE:
        return HL_NUMBER;

    case TT_STRING:
    case TT_RAW_STRING:
    case TT_RAW_HASH_STRING:
    case TT_SHELL_STRING:
    case TT_SHELL_BANG_STRING:
    case TT_ENV_STRING:
        return HL_STRING;

    case TT_REGEX:
        return HL_REGEX;
    case TT_PATH_STRING:
        return HL_PATH;
    case TT_COMMENT:
        return HL_COMMENT;

    case TT_IDENT:
        return HL_IDENTIFIER;

    /* Operators */
    case TT_PLUS:
    case TT_MINUS:
    case TT_STAR:
    case TT_SLASH:
    case TT_FLOORDIV:
    case TT_MOD:
    case TT_POW:
    case TT_DEEPMERGE:
    case TT_CARET:
    case TT_EQ:
    case TT_NEQ:
    case TT_LT:
    case TT_LTE:
    case TT_GT:
    case TT_GTE:
    case TT_SEND:
    case TT_RECV:
    case TT_NEG:
    case TT_NULLISH:
    case TT_ASSIGN:
    case TT_WALRUS:
    case TT_APPLYASSIGN:
    case TT_PLUSEQ:
    case TT_MINUSEQ:
    case TT_STAREQ:
    case TT_SLASHEQ:
    case TT_FLOORDIVEQ:
    case TT_MODEQ:
    case TT_POWEQ:
    case TT_INCR:
    case TT_DECR:
    case TT_TILDE:
    case TT_REGEXMATCH:
    case TT_AMP:
    case TT_PIPE:
        return HL_OPERATOR;

    /* Punctuation */
    case TT_LPAR:
    case TT_RPAR:
    case TT_LSQB:
    case TT_RSQB:
    case TT_LBRACE:
    case TT_RBRACE:
    case TT_DOT:
    case TT_COMMA:
    case TT_COLON:
    case TT_SEMI:
    case TT_QMARK:
    case TT_AT:
    case TT_DOLLAR:
    case TT_BACKQUOTE:
        return HL_PUNCTUATION;

    /* Layout / special — no highlight */
    default:
        return HL_NONE;
    }
}

/* ======================================================================== */
/* Structural classification                                                 */
/* ======================================================================== */

static int is_layout(TT t) {
    return t == TT_NEWLINE || t == TT_INDENT || t == TT_DEDENT || t == TT_EOF || t == TT_SEMI;
}

static int is_line_blank_or_comment(const char *src, int start, int end) {
    int i = start;
    while (i < end) {
        char ch = src[i];
        if (ch == ' ' || ch == '\t') {
            i++;
            continue;
        }
        if (ch == '#' || ch == '\r' || ch == '\n')
            return 1;
        return 0;
    }
    return 1;
}

static int line_start_for_pos(const char *src, int pos) {
    int i = pos;
    while (i > 0) {
        char ch = src[i - 1];
        if (ch == '\n' || ch == '\r')
            break;
        i--;
    }
    return i;
}

static int line_indent_width(const char *src, int line_start, int src_len) {
    int i = line_start;
    int indent = 0;
    while (i < src_len) {
        char ch = src[i];
        if (ch == ' ')
            indent += 1;
        else if (ch == '\t')
            indent += 8;
        else
            break;
        i++;
    }
    return indent;
}

static int line_is_ws_before_pos(const char *src, int line_start, int pos) {
    for (int i = line_start; i < pos; i++) {
        char ch = src[i];
        if (ch != ' ' && ch != '\t')
            return 0;
    }
    return 1;
}

static int prev_nonblank_indent(const char *src, int src_len, int line_start) {
    int i = line_start;
    while (i > 0) {
        int j = i - 1;
        if (src[j] == '\n') {
            if (j > 0 && src[j - 1] == '\r')
                j--;
        } else if (src[j] == '\r') {
            j--;
        }
        int line_end = j + 1;
        int k = j;
        while (k > 0 && src[k - 1] != '\n' && src[k - 1] != '\r')
            k--;
        if (!is_line_blank_or_comment(src, k, line_end))
            return line_indent_width(src, k, src_len);
        i = k;
    }
    return -1;
}

static int is_expr_end(TT t) {
    switch (t) {
    case TT_IDENT:
    case TT_NUMBER:
    case TT_DURATION:
    case TT_SIZE:
    case TT_STRING:
    case TT_RAW_STRING:
    case TT_RAW_HASH_STRING:
    case TT_SHELL_STRING:
    case TT_SHELL_BANG_STRING:
    case TT_ENV_STRING:
    case TT_REGEX:
    case TT_PATH_STRING:
    case TT_TRUE:
    case TT_FALSE:
    case TT_NIL:
    case TT_RPAR:
    case TT_RSQB:
    case TT_RBRACE:
    case TT_INCR:
    case TT_DECR:
        return 1;
    default:
        return 0;
    }
}

/* Classify identifier based on prev/next significant token. */
static HlGroup classify_ident(const char *src, Tok *tok, Tok *next_sig, TT prev_sig) {
    if (prev_sig == TT_FN)
        return HL_FUNCTION;
    if (prev_sig == TT_DECORATOR || prev_sig == TT_AT)
        return HL_DECORATOR;
    if (prev_sig == TT_HOOK)
        return HL_HOOK;
    if (prev_sig == TT_GET || prev_sig == TT_SET || prev_sig == TT_DOT)
        return HL_PROPERTY;
    if (prev_sig == TT_TILDE)
        return HL_TYPE;

    /* Contextual keywords: timeout, default */
    if (tok->len == 7 && strncmp(src + tok->start, "timeout", 7) == 0)
        return HL_KEYWORD;
    if (tok->len == 7 && strncmp(src + tok->start, "default", 7) == 0)
        return HL_KEYWORD;

    /* Call-site: ident followed by ( */
    if (next_sig && next_sig->type == TT_LPAR)
        return HL_FUNCTION;

    return HL_IDENTIFIER;
}

/* Classify string based on previous significant token. */
static HlGroup classify_string(TT prev_sig) {
    if (prev_sig == TT_HOOK)
        return HL_HOOK;
    return HL_STRING;
}

/* ======================================================================== */
/* Duration/size sub-span splitting                                          */
/* ======================================================================== */

/* Emit number+unit sub-spans for duration/size tokens.
 * Returns 1 if spans were emitted, 0 if not. */
static int emit_dur_size_spans(const char *src, Tok *tok, HlBuf *out) {
    const char *text = src + tok->start;
    int         tlen = tok->len;
    int         line = tok->line - 1;
    int         base_col = tok->col - 1;

    int pos = 0;
    int emitted = 0;

    while (pos < tlen) {
        /* Skip leading digits (number part) */
        int num_start = pos;
        while (pos < tlen &&
               (isdigit((unsigned char)text[pos]) || text[pos] == '_' || text[pos] == '.' ||
                text[pos] == 'e' || text[pos] == 'E' || text[pos] == '+' || text[pos] == '-')) {
            /* Handle e/E only if preceded by digit context */
            if ((text[pos] == '+' || text[pos] == '-') && pos > 0 && text[pos - 1] != 'e' &&
                text[pos - 1] != 'E')
                break;
            if ((text[pos] == 'e' || text[pos] == 'E') && pos > 0 &&
                !isdigit((unsigned char)text[pos - 1]) && text[pos - 1] != '_')
                break;
            pos++;
        }
        int num_end = pos;

        /* Unit part: lowercase alpha */
        int unit_start = pos;
        while (pos < tlen && text[pos] >= 'a' && text[pos] <= 'z')
            pos++;
        int unit_end = pos;

        if (unit_end > unit_start && num_end > num_start) {
            hlbuf_push(out, line, base_col + num_start, base_col + num_end, HL_NUMBER);
            hlbuf_push(out, line, base_col + unit_start, base_col + unit_end, HL_UNIT);
            emitted = 1;
        } else {
            /* Safety: skip one char to avoid infinite loop */
            if (pos == num_start)
                pos++;
        }
    }

    return emitted;
}

/* ======================================================================== */
/* Token spans (multiline support)                                           */
/* ======================================================================== */

/* Emit highlight spans for a token, handling multiline tokens. */
static void emit_token_spans(const char *src, Tok *tok, HlGroup group, HlBuf *out) {
    const char *text = src + tok->start;
    int         tlen = tok->len;
    int         start_line = tok->line - 1;
    int         start_col = tok->col - 1;

    /* Fast path: single-line token */
    int has_newline = 0;
    for (int i = 0; i < tlen; i++) {
        if (text[i] == '\n' || text[i] == '\r') {
            has_newline = 1;
            break;
        }
    }

    if (!has_newline) {
        hlbuf_push(out, start_line, start_col, start_col + tlen, group);
        return;
    }

    /* Multiline: emit one span per line */
    int line_idx = start_line;
    int line_start = 0;
    for (int i = 0; i <= tlen; i++) {
        int at_end = (i == tlen);
        int at_nl = !at_end && (text[i] == '\n' || text[i] == '\r');

        if (at_end || at_nl) {
            int line_len = i - line_start;
            /* Strip trailing \r */
            if (line_len > 0 && text[line_start + line_len - 1] == '\r')
                line_len--;

            if (line_len > 0) {
                int cs = (line_idx == start_line) ? start_col : 0;
                int ce = cs + line_len;
                hlbuf_push(out, line_idx, cs, ce, group);
            }

            if (at_nl) {
                if (text[i] == '\r' && i + 1 < tlen && text[i + 1] == '\n')
                    i++;
                line_idx++;
                line_start = i + 1;
            }
        }
    }
}

/* ======================================================================== */
/* Main highlight function                                                   */
/* ======================================================================== */

void highlight(const char *src, int src_len, TokBuf *tokens, HlBuf *out) {
    (void)src_len;

    /* Build significant-token index for prev/next lookups */
    int  tc = tokens->count;
    Tok *toks = tokens->toks;

    /* We iterate only over significant (non-layout) tokens. */
    TT prev_sig = TT_EOF;

    /* Pre-scan: build array of sig token indices for next-lookahead. */
    int *sig_idx = malloc(tc * sizeof(int));
    int  sig_count = 0;
    /* Emit comment spans first (comments are transparent to structural logic). */
    for (int i = 0; i < tc; i++) {
        if (toks[i].type == TT_COMMENT)
            emit_token_spans(src, &toks[i], HL_COMMENT, out);
    }

    for (int i = 0; i < tc; i++) {
        TT t = toks[i].type;
        if (t == TT_COMMENT || is_layout(t))
            continue;
        sig_idx[sig_count] = i;
        sig_count++;
    }

    for (int si = 0; si < sig_count; si++) {
        Tok    *tok = &toks[sig_idx[si]];
        Tok    *next_sig_tok = (si + 1 < sig_count) ? &toks[sig_idx[si + 1]] : 0;
        TT      next_sig_type = next_sig_tok ? next_sig_tok->type : TT_EOF;
        HlGroup group = base_group(tok->type);
        if (group == HL_NONE) {
            prev_sig = tok->type;
            continue;
        }

        int subject_override = 0;
        if (tok->type == TT_DOT) {
            int line_start = line_start_for_pos(src, tok->start);
            int at_line_start = line_is_ws_before_pos(src, line_start, tok->start);
            if (!at_line_start) {
                if (!is_expr_end(prev_sig))
                    subject_override = 1;
            } else {
                if (!is_expr_end(prev_sig)) {
                    subject_override = 1;
                } else {
                    int prev_indent = prev_nonblank_indent(src, src_len, line_start);
                    int cur_indent = line_indent_width(src, line_start, src_len);
                    if (prev_indent < 0 || cur_indent <= prev_indent)
                        subject_override = 1;
                }
            }
        }

        /* Structural classification */
        if (subject_override) {
            group = HL_IMPLICIT_SUBJECT;
        } else if (tok->type == TT_IDENT) {
            group = classify_ident(src, tok, next_sig_tok, prev_sig);
        } else if (tok->type == TT_STRING || tok->type == TT_RAW_STRING ||
                   tok->type == TT_RAW_HASH_STRING || tok->type == TT_SHELL_STRING ||
                   tok->type == TT_SHELL_BANG_STRING || tok->type == TT_ENV_STRING) {
            group = classify_string(prev_sig);
        } else if (tok->type == TT_PIPE && next_sig_type == TT_COLON) {
            group = HL_KEYWORD; /* guard else: |: */
        } else if (tok->type == TT_COLON && prev_sig == TT_PIPE) {
            group = HL_KEYWORD; /* guard else: |: */
        }

        /* Duration/size sub-spans */
        if (tok->type == TT_DURATION || tok->type == TT_SIZE) {
            if (emit_dur_size_spans(src, tok, out)) {
                prev_sig = tok->type;
                continue;
            }
        }

        emit_token_spans(src, tok, group, out);
        if (tok->type != TT_COMMENT)
            prev_sig = tok->type;
    }

    free(sig_idx);
}
