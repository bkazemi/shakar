local ok, shakar = pcall(require, "shakar")
if not ok then
  return
end

shakar.ensure_started(0)
