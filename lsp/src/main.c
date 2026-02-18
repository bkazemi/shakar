#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "protocol.h"
#include "session.h"

static int read_message(FILE *in, char **out, int *out_len) {
    char line[512];
    int content_len = -1;

    while (fgets(line, sizeof(line), in)) {
        if (strcmp(line, "\n") == 0 || strcmp(line, "\r\n") == 0)
            break;
        if (strncmp(line, "Content-Length:", 15) == 0) {
            content_len = atoi(line + 15);
        }
    }

    if (content_len < 0)
        return 0;

    char *buf = malloc((size_t)content_len + 1);
    if (!buf)
        return 0;

    int read = 0;
    while (read < content_len) {
        size_t r = fread(buf + read, 1, (size_t)(content_len - read), in);
        if (r == 0) {
            free(buf);
            return 0;
        }
        read += (int)r;
    }

    buf[content_len] = '\0';
    *out = buf;
    *out_len = content_len;
    return 1;
}

static int read_text_document(const ProtocolMessage *msg, int *out_doc_tok) {
    if (msg->params < 0)
        return 0;
    int doc_tok = protocol_find_key(msg, msg->params, "textDocument");
    if (doc_tok < 0)
        return 0;
    *out_doc_tok = doc_tok;
    return 1;
}

static int read_uri(const ProtocolMessage *msg, int doc_tok, char **out_uri, int *out_len) {
    int uri_tok = protocol_find_key(msg, doc_tok, "uri");
    if (uri_tok < 0)
        return 0;
    return protocol_token_string_dup(msg, uri_tok, out_uri, out_len);
}

static int read_version(const ProtocolMessage *msg, int doc_tok, int *out_version) {
    int ver_tok = protocol_find_key(msg, doc_tok, "version");
    if (ver_tok < 0)
        return 0;
    return protocol_token_int(msg, ver_tok, out_version);
}

static int read_text(const ProtocolMessage *msg, int doc_tok, char **out_text, int *out_len) {
    int text_tok = protocol_find_key(msg, doc_tok, "text");
    if (text_tok < 0)
        return 0;
    return protocol_token_string_dup(msg, text_tok, out_text, out_len);
}

static int read_change_text(const ProtocolMessage *msg, char **out_text, int *out_len) {
    if (msg->params < 0)
        return 0;
    int changes_tok = protocol_find_key(msg, msg->params, "contentChanges");
    if (changes_tok < 0)
        return 0;
    int first_change = protocol_array_nth(msg, changes_tok, 0);
    if (first_change < 0)
        return 0;
    int text_tok = protocol_find_key(msg, first_change, "text");
    if (text_tok < 0)
        return 0;
    return protocol_token_string_dup(msg, text_tok, out_text, out_len);
}

/* Read a {line, character} position object into *line and *col. */
static int read_position(const ProtocolMessage *msg, int pos_tok, int *line, int *col) {
    int line_tok = protocol_find_key(msg, pos_tok, "line");
    int col_tok = protocol_find_key(msg, pos_tok, "character");

    return protocol_token_int(msg, line_tok, line) && protocol_token_int(msg, col_tok, col);
}

static int read_range(const ProtocolMessage *msg, LspRange *out) {
    if (msg->params < 0)
        return 0;
    int range_tok = protocol_find_key(msg, msg->params, "range");
    if (range_tok < 0)
        return 0;

    int start_tok = protocol_find_key(msg, range_tok, "start");
    int end_tok = protocol_find_key(msg, range_tok, "end");
    if (start_tok < 0 || end_tok < 0)
        return 0;

    return read_position(msg, start_tok, &out->start_line, &out->start_col) &&
           read_position(msg, end_tok, &out->end_line, &out->end_col);
}

/* Begin a JSON-RPC response buffer: {"jsonrpc":"2.0","id":<id> */
static void response_begin(StrBuf *buf, const ProtocolMessage *msg) {
    strbuf_init(buf);
    strbuf_append(buf, "{\"jsonrpc\":\"2.0\",\"id\":");
    protocol_append_id(buf, msg);
}

/* Finalize and send a JSON-RPC message, then free the buffer. */
static void response_send(FILE *out, StrBuf *buf) {
    protocol_write_message(out, buf->data, buf->len);
    strbuf_free(buf);
}

static void send_initialize_response(FILE *out, const ProtocolMessage *msg) {
    StrBuf buf;
    response_begin(&buf, msg);

    strbuf_append(&buf, ",\"result\":{\"capabilities\":{");
    strbuf_append(&buf, "\"textDocumentSync\":1,");
    strbuf_append(&buf, "\"semanticTokensProvider\":{");
    strbuf_append(&buf, "\"legend\":{");
    strbuf_append(&buf, "\"tokenTypes\":[");
    for (int i = 0; i < LSP_TOKEN_TYPE_COUNT; i++) {
        if (i > 0)
            strbuf_append(&buf, ",");
        strbuf_append_json_string(&buf, LSP_TOKEN_TYPES[i], (int)strlen(LSP_TOKEN_TYPES[i]));
    }
    strbuf_append(&buf, "],\"tokenModifiers\":[");
    for (int i = 0; i < LSP_TOKEN_MOD_COUNT; i++) {
        if (i > 0)
            strbuf_append(&buf, ",");
        strbuf_append_json_string(&buf, LSP_TOKEN_MODIFIERS[i],
                                  (int)strlen(LSP_TOKEN_MODIFIERS[i]));
    }
    strbuf_append(&buf, "]},\"full\":true,\"range\":true}},");
    strbuf_append(&buf, "\"serverInfo\":{\"name\":\"shakar-lsp\"}}}");

    response_send(out, &buf);
}

static void send_shutdown_response(FILE *out, const ProtocolMessage *msg) {
    StrBuf buf;
    response_begin(&buf, msg);
    strbuf_append(&buf, ",\"result\":null}");
    response_send(out, &buf);
}

static void send_semantic_tokens(FILE *out, const ProtocolMessage *msg, LspTokenBuf *tokens) {
    StrBuf buf;
    response_begin(&buf, msg);

    strbuf_append(&buf, ",\"result\":{\"data\":[");
    for (int i = 0; i < tokens->count; i++) {
        if (i > 0)
            strbuf_append(&buf, ",");
        strbuf_appendf(&buf, "%d", tokens->data[i]);
    }
    strbuf_append(&buf, "]}}");

    response_send(out, &buf);
}

static void emit_diag_json(StrBuf *buf, Diagnostic *d, const char *source, int severity) {
    strbuf_appendf(buf,
                   "{\"range\":{\"start\":{\"line\":%d,\"character\":%d},"
                   "\"end\":{\"line\":%d,\"character\":%d}},"
                   "\"severity\":%d,\"source\":",
                   d->line, d->col_start, d->line, d->col_end, severity);
    strbuf_append_json_string(buf, source, (int)strlen(source));
    strbuf_append(buf, ",\"message\":");
    strbuf_append_json_string(buf, d->message, (int)strlen(d->message));
    strbuf_append(buf, "}");
}

static void send_diagnostics(FILE *out, const char *uri, int uri_len, Diagnostic *diags,
                             int count) {
    StrBuf buf;
    strbuf_init(&buf);

    strbuf_append(
        &buf, "{\"jsonrpc\":\"2.0\",\"method\":\"textDocument/publishDiagnostics\",\"params\":{");
    strbuf_append(&buf, "\"uri\":");
    strbuf_append_json_string(&buf, uri, uri_len);
    strbuf_append(&buf, ",\"diagnostics\":[");

    for (int i = 0; i < count; i++) {
        if (i > 0)
            strbuf_append(&buf, ",");
        emit_diag_json(&buf, &diags[i], "shakar", diags[i].severity);
    }

    strbuf_append(&buf, "]}}");
    protocol_write_message(out, buf.data, buf.len);
    strbuf_free(&buf);
}

static void gather_and_send_diagnostics(FILE *out, Session *session, const char *uri, int uri_len) {
    Diagnostic all_diags[33];
    int count = 0;
    Diagnostic lex_diag;

    if (session_lex_diagnostics(session, uri, uri_len, &lex_diag))
        all_diags[count++] = lex_diag;
    count += session_structural_diagnostics(session, uri, uri_len, all_diags + count, 32 - count);
    send_diagnostics(out, uri, uri_len, all_diags, count);
}

int main(void) {
    Session session;
    session_init(&session);

    for (;;) {
        char *json = 0;
        int json_len = 0;
        if (!read_message(stdin, &json, &json_len))
            break;

        ProtocolMessage msg;
        char err[64] = {0};
        if (!protocol_parse_message(json, json_len, &msg, err, (int)sizeof(err))) {
            free(json);
            protocol_message_free(&msg);
            continue;
        }

        if (protocol_method_is(&msg, "initialize")) {
            send_initialize_response(stdout, &msg);
        } else if (protocol_method_is(&msg, "initialized")) {
            /* no-op */
        } else if (protocol_method_is(&msg, "shutdown")) {
            send_shutdown_response(stdout, &msg);
        } else if (protocol_method_is(&msg, "exit")) {
            protocol_message_free(&msg);
            free(json);
            break;
        } else if (protocol_method_is(&msg, "textDocument/didOpen") ||
                   protocol_method_is(&msg, "textDocument/didChange")) {
            int is_change = protocol_method_is(&msg, "textDocument/didChange");
            int doc_tok = 0;
            char *uri = 0;
            int uri_len = 0;
            char *text = 0;
            int text_len = 0;
            int version = 0;
            int got_text = 0;

            if (read_text_document(&msg, &doc_tok) && read_uri(&msg, doc_tok, &uri, &uri_len)) {
                got_text = is_change ? read_change_text(&msg, &text, &text_len)
                                     : read_text(&msg, doc_tok, &text, &text_len);
                if (got_text) {
                    read_version(&msg, doc_tok, &version);
                    session_open(&session, uri, uri_len, text, text_len, version);
                    gather_and_send_diagnostics(stdout, &session, uri, uri_len);
                }
            }

            free(uri);
            free(text);
        } else if (protocol_method_is(&msg, "textDocument/didClose")) {
            int doc_tok = 0;
            char *uri = 0;
            int uri_len = 0;
            if (read_text_document(&msg, &doc_tok) && read_uri(&msg, doc_tok, &uri, &uri_len)) {
                session_close(&session, uri, uri_len);
                send_diagnostics(stdout, uri, uri_len, NULL, 0);
            }
            free(uri);
        } else if (protocol_method_is(&msg, "textDocument/semanticTokens/full") ||
                   protocol_method_is(&msg, "textDocument/semanticTokens/range")) {
            int doc_tok = 0;
            char *uri = 0;
            int uri_len = 0;
            LspRange range;
            int has_range = read_range(&msg, &range);
            if (read_text_document(&msg, &doc_tok) && read_uri(&msg, doc_tok, &uri, &uri_len)) {
                LspTokenBuf tokens;
                lsp_tokens_init(&tokens);
                session_build_semantic_tokens(&session, uri, uri_len, has_range ? &range : 0,
                                              &tokens);
                send_semantic_tokens(stdout, &msg, &tokens);
                lsp_tokens_free(&tokens);
            }
            free(uri);
        }

        protocol_message_free(&msg);
        free(json);
    }

    session_free(&session);
    return 0;
}
