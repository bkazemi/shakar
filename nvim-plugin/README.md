# Shakar Neovim LSP

This plugin starts the `shakar-lsp` language server and relies on Neovim's built-in LSP semantic tokens.

## Install

Add the plugin directory to your runtimepath:

```lua
vim.opt.rtp:append("/path/to/shakar/nvim-plugin")
```

## Build the server

From the repo root:

```
make -C lsp
```

Ensure `shakar-lsp` is on your PATH, or configure the command explicitly.

## Configure

```lua
require("shakar").setup({
  -- Optional: override server command (default: "shakar-lsp")
  cmd = { "shakar-lsp" },
  -- Optional: file patterns (default: {"*.shk", "*.sk"})
  pattern = { "*.shk", "*.sk" },
})
```

You can also point to the server via environment variable:

```
export SHAKAR_LSP_CMD=/absolute/path/to/shakar-lsp
```

## Notes

- Highlights are provided via LSP semantic tokens. Use `:Inspect` to see `@lsp.type.*.shakar` groups.
- The plugin sets filetype for `.shk` and `.sk` using `vim.filetype.add`.
