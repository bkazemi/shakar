if vim.g.loaded_shakar_lsp then
  return
end
vim.g.loaded_shakar_lsp = true

require("shakar").setup()
