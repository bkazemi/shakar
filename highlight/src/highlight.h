#ifndef SHK_HIGHLIGHT_H
#define SHK_HIGHLIGHT_H

#include "token_types.h"

/* Highlight group IDs — mapped to strings in JS wrapper. */
typedef enum {
    HL_NONE = 0,
    HL_KEYWORD,
    HL_BOOLEAN,
    HL_CONSTANT,
    HL_NUMBER,
    HL_UNIT,
    HL_STRING,
    HL_REGEX,
    HL_PATH,
    HL_COMMENT,
    HL_OPERATOR,
    HL_PUNCTUATION,
    HL_IDENTIFIER,
    HL_FUNCTION,
    HL_DECORATOR,
    HL_HOOK,
    HL_PROPERTY,
    HL_IMPLICIT_SUBJECT,
    HL_TYPE,
    HL__COUNT
} HlGroup;

typedef struct {
    int     line;      /* 0-based */
    int     col_start; /* 0-based */
    int     col_end;   /* 0-based, exclusive */
    HlGroup group;
} HlSpan;

typedef struct {
    HlSpan *spans;
    int     count;
    int     capacity;
} HlBuf;

void hlbuf_init(HlBuf *b);
void hlbuf_free(HlBuf *b);

/* ---- Diagnostics ---- */

typedef struct {
    int  line;      /* 0-based */
    int  col_start; /* 0-based */
    int  col_end;   /* 0-based, exclusive */
    int  severity;  /* 1=error, 2=warning, 3=info */
    char message[128];
} HlDiag;

typedef struct {
    HlDiag *diags;
    int     count;
    int     capacity;
} DiagBuf;

void diagbuf_init(DiagBuf *b);
void diagbuf_free(DiagBuf *b);

/* ---- Highlighting ---- */

/* Perform lexical + structural highlighting on the token stream.
 * Source is needed for multiline span computation. */
void highlight(const char *src, int src_len, TokBuf *tokens, HlBuf *out);

/* Structural highlight pass: implicit-subject chain detection, fanpath
 * overrides, bracket-matching and block-header-colon diagnostics.
 * Modifies hl in-place (structural groups override lexical groups).
 * If diags is non-NULL, collects diagnostics. */
void structural_highlight(const char *src, int src_len, TokBuf *tokens, HlBuf *hl, DiagBuf *diags);

/* Group ID to string name. */
const char *hl_group_name(HlGroup g);

#endif /* SHK_HIGHLIGHT_H */
