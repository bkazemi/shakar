#ifndef SHK_LSP_SESSION_H
#define SHK_LSP_SESSION_H

#include "highlight.h"

typedef struct {
    char *uri;
    char *text;
    int   text_len;
    int   version;
} Document;

typedef struct {
    Document *docs;
    int       doc_count;
    int       doc_capacity;
} Session;

typedef struct {
    int start_line;
    int start_col;
    int end_line;
    int end_col;
} LspRange;

typedef struct {
    int *data;
    int  count;
    int  capacity;
} LspTokenBuf;

typedef struct {
    int  line;
    int  col_start;
    int  col_end;
    int  severity; /* 1=error, 2=warning, 3=info, 4=hint */
    char message[256];
} Diagnostic;

extern const char *LSP_TOKEN_TYPES[];
extern const int   LSP_TOKEN_TYPE_COUNT;
extern const char *LSP_TOKEN_MODIFIERS[];
extern const int   LSP_TOKEN_MOD_COUNT;

void      session_init(Session *s);
void      session_free(Session *s);
void      session_open(Session *s, const char *uri, int uri_len, const char *text, int text_len,
                       int version);
void      session_change(Session *s, const char *uri, int uri_len, const char *text, int text_len,
                         int version);
void      session_close(Session *s, const char *uri, int uri_len);
Document *session_get(Session *s, const char *uri, int uri_len);

void lsp_tokens_init(LspTokenBuf *b);
void lsp_tokens_free(LspTokenBuf *b);

int session_build_semantic_tokens(Session *s, const char *uri, int uri_len, const LspRange *range,
                                  LspTokenBuf *out);
int session_lex_diagnostics(Session *s, const char *uri, int uri_len, Diagnostic *out);

/* Structural diagnostics (bracket matching, missing colons).
 * Returns number of diagnostics written to out (up to max_out). */
int session_structural_diagnostics(Session *s, const char *uri, int uri_len, Diagnostic *out,
                                   int max_out);

#endif /* SHK_LSP_SESSION_H */
