#ifndef SHK_LEXER_H
#define SHK_LEXER_H

#include "token_types.h"

#define SHK_MAX_INDENT_DEPTH 256

typedef struct {
    const char *src;
    int         src_len;
    int         pos;
    int         line;
    int         col;

    TokBuf tokens;

    /* Indentation */
    int track_indent;
    int emit_comments;
    int indent_stack[SHK_MAX_INDENT_DEPTH];
    int indent_count; /* depth of indent_stack */
    int at_line_start;
    int group_depth;
    int line_ended_with_colon;
    int indent_after_colon;
    int prev_line_indent; /* -1 = not set */

    /* Error state */
    int  has_error;
    char error_msg[256];
} Lexer;

void lexer_init(Lexer *L, const char *src, int len, int track_indent, int emit_comments);
int  lexer_tokenize(Lexer *L); /* returns token count, or -1 on error */
void lexer_free(Lexer *L);

#endif /* SHK_LEXER_H */
