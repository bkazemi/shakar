local M = {}

local state = {
  cmd = { "shakar-lsp" },
  pattern = { "*.shk", "*.sk" },
  root_dir = nil,
}

local function ensure_filetype()
  if vim.filetype and vim.filetype.add then
    vim.filetype.add({
      extension = {
        shk = "shakar",
        sk = "shakar",
      },
    })
  end
end

local function start_semantic_tokens(bufnr, client_id)
  if not client_id or type(client_id) ~= "number" then
    local clients = vim.lsp.get_clients({ bufnr = bufnr })
    for _, client in ipairs(clients) do
      if vim.tbl_get(client.server_capabilities, "semanticTokensProvider", "full") then
        client_id = client.id
        break
      end
    end
  end

  if not client_id or type(client_id) ~= "number" then
    return
  end

  if vim.lsp.semantic_tokens and vim.lsp.semantic_tokens.start then
    vim.lsp.semantic_tokens.start(bufnr, client_id)
    return
  end

  if vim.lsp.buf.semantic_tokens_full then
    pcall(vim.lsp.buf.semantic_tokens_full, { bufnr = bufnr })
  end
end

local function normalize_cmd(cmd)
  if type(cmd) == "string" then
    return { cmd }
  end
  return cmd
end

local function resolve_root(bufnr)
  local name = vim.api.nvim_buf_get_name(bufnr)
  if name and name ~= "" then
    local dir = vim.fs.dirname(name)
    if dir and dir ~= "" then
      return dir
    end
  end
  return vim.fn.getcwd()
end

function M.start(bufnr)
  bufnr = bufnr or vim.api.nvim_get_current_buf()
  local cmd = normalize_cmd(state.cmd)
  local root_dir = state.root_dir
  if type(root_dir) == "function" then
    root_dir = root_dir(bufnr)
  end
  if not root_dir or root_dir == "" then
    root_dir = resolve_root(bufnr)
  end

  local config = {
    name = "shakar-lsp",
    cmd = cmd,
    root_dir = root_dir,
    filetypes = { "shakar" },
    reuse_client = function(client, config)
      return client.name == config.name and client.config.root_dir == config.root_dir
    end,
    on_attach = function(client, attached_bufnr)
      if client.name == "shakar-lsp" then
        start_semantic_tokens(attached_bufnr, client.id)
      end
    end,
  }

  local ok, client_id = pcall(vim.lsp.start, config, { bufnr = bufnr })
  if not ok then
    vim.notify("[shakar] LSP start error: " .. tostring(client_id), vim.log.levels.ERROR)
    return
  end
  if not client_id then
    vim.notify("[shakar] failed to start shakar-lsp", vim.log.levels.WARN)
    return
  end
end

local function has_shakar_client(bufnr)
  local clients = vim.lsp.get_clients({ bufnr = bufnr })
  for _, client in ipairs(clients) do
    if client.name == "shakar-lsp" then
      return true
    end
  end
  return false
end

function M.ensure_started(bufnr)
  bufnr = bufnr or vim.api.nvim_get_current_buf()
  if not vim.api.nvim_buf_is_valid(bufnr) then
    return
  end
  if not has_shakar_client(bufnr) then
    M.start(bufnr)
  end
end

function M.setup(opts)
  opts = opts or {}
  if opts.cmd then
    state.cmd = opts.cmd
  elseif vim.env.SHAKAR_LSP_CMD and vim.env.SHAKAR_LSP_CMD ~= "" then
    state.cmd = { vim.env.SHAKAR_LSP_CMD }
  end
  state.pattern = opts.pattern or state.pattern
  state.root_dir = opts.root_dir

  ensure_filetype()

  local ok_number, number_hl = pcall(vim.api.nvim_get_hl, 0, { name = "Number", link = false })
  if not ok_number then
    number_hl = {}
  end
  number_hl.italic = true
  vim.api.nvim_set_hl(0, "@shakar.unit", number_hl)
  local function hl_defined(name)
    local ok, hl = pcall(vim.api.nvim_get_hl, 0, { name = name, link = false })
    if ok and next(hl) ~= nil then
      return true
    end
    local ok_link, hl_link = pcall(vim.api.nvim_get_hl, 0, { name = name, link = true })
    return ok_link and hl_link.link and hl_link.link ~= ""
  end

  if not hl_defined("@shakar.implicit_subject") then
    vim.api.nvim_set_hl(0, "@shakar.implicit_subject", { link = "@punctuation.delimiter" })
  end
  if not hl_defined("@shakar") then
    vim.api.nvim_set_hl(0, "@shakar", { link = "@shakar.implicit_subject" })
  end

  local links = {
    ["@lsp.type.keyword.shakar"] = "@keyword",
    ["@lsp.type.variable.shakar"] = "@variable",
    ["@lsp.type.number.shakar"] = "@number",
    ["@lsp.type.string.shakar"] = "@string",
    ["@lsp.type.comment.shakar"] = "@comment",
    ["@lsp.type.operator.shakar"] = "@operator",
    ["@lsp.type.function.shakar"] = "@function",
    ["@lsp.type.decorator.shakar"] = "@attribute",
    ["@lsp.type.macro.shakar"] = "@string.special",
    ["@lsp.type.property.shakar"] = "@property",
    ["@lsp.type.implicit_subject.shakar"] = "@shakar.implicit_subject",
    ["@lsp.type.type.shakar"] = "@type",
    ["@lsp.type.regexp.shakar"] = "@string.regexp",
    ["@lsp.type.punctuation.shakar"] = "@punctuation.delimiter",
    ["@lsp.typemod.keyword.defaultLibrary.shakar"] = "@boolean",
    ["@lsp.typemod.variable.defaultLibrary.shakar"] = "@constant.builtin",
    ["@lsp.typemod.keyword.modification.shakar"] = "@shakar.unit",
  }

  for hl, link in pairs(links) do
    vim.api.nvim_set_hl(0, hl, { link = link })
  end

  vim.api.nvim_create_autocmd({ "BufReadPost", "BufNewFile" }, {
    pattern = state.pattern,
    callback = function(args)
      M.ensure_started(args.buf)
    end,
  })

  vim.api.nvim_create_autocmd("FileType", {
    pattern = "shakar",
    callback = function(args)
      M.ensure_started(args.buf)
    end,
  })

  vim.api.nvim_create_autocmd({ "BufEnter", "BufWinEnter" }, {
    pattern = state.pattern,
    callback = function(args)
      if vim.bo[args.buf].filetype == "shakar" then
        M.ensure_started(args.buf)
      end
    end,
  })

  local bufnr = vim.api.nvim_get_current_buf()
  if vim.bo[bufnr].filetype == "shakar" then
    M.ensure_started(bufnr)
  end
end

return M
