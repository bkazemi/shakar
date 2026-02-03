CC       = /usr/lib/emscripten/emcc
CFLAGS   = -O2 -Wall -Wextra -Wno-unused-parameter
SOURCES  = src/lexer.c src/highlight.c src/api.c

EXPORTED = _shk_src_ptr,_shk_set_src_len,\
           _shk_lex,_shk_tok_count,_shk_tok_type,_shk_tok_line,_shk_tok_col,_shk_tok_start,_shk_tok_len,\
           _shk_error,_shk_error_line,_shk_error_col,_shk_error_pos,\
           _shk_highlight,_shk_hl_spans_ptr,_shk_hl_count,_shk_hl_line,_shk_hl_col_start,_shk_hl_col_end,_shk_hl_group,_shk_hl_group_name,\
           _malloc,_free

EMFLAGS  = -s EXPORTED_FUNCTIONS='[$(EXPORTED)]' \
           -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","UTF8ToString","stringToUTF8","lengthBytesUTF8","HEAP32"]' \
           -s MODULARIZE=1 \
           -s EXPORT_NAME=ShakarHL \
           -s ALLOW_MEMORY_GROWTH=1 \
           -s INITIAL_MEMORY=2097152 \
           -s ENVIRONMENT=worker

.PHONY: all clean test native

all: js/shakar_hl.js

js/shakar_hl.js: $(SOURCES) src/lexer.h src/highlight.h src/token_types.h
	@mkdir -p js
	$(CC) $(CFLAGS) $(EMFLAGS) -o $@ $(SOURCES)

# Native build for testing with validate_tokens.py
native: shakar_hl_native

shakar_hl_native: $(SOURCES) src/lexer.h src/highlight.h src/token_types.h
	cc -O2 -Wall -Wextra -Wno-unused-parameter -DSHK_NATIVE_TEST -o $@ $(SOURCES)

test: shakar_hl_native
	python tests/validate_tokens.py

# Deploy WASM artifacts to web/ directory
deploy: js/shakar_hl.js
	cp js/shakar_hl.js ../web/shakar_hl_glue.js
	cp js/shakar_hl.wasm ../web/shakar_hl_glue.wasm

clean:
	rm -f js/shakar_hl.js js/shakar_hl.wasm shakar_hl_native
