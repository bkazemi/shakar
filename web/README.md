# Shakar Playground

Browser-based playground for Shakar using Pyodide.

## Quick Start

1. Generate the bundle:
   ```bash
   cd web
   python bundle.py > shakar_bundle.py
   ```

2. Serve the directory:
   ```bash
   python -m http.server 8000
   ```

3. Open http://localhost:8000

## How It Works

- **Pyodide** runs CPython in WebAssembly
- **bundle.py** packages all `shakar_ref` modules into a single file
- The bundle self-extracts and registers modules in `sys.modules`
- User code runs via `shk_run()` with stdout captured

## Production Deployment

For GitHub Pages, keep `shakar_bundle.py` committed and push changes under
`web/`. The repo's Pages workflow deploys on `web/` changes.

For other production builds, build a proper wheel and use micropip:

```bash
# Build wheel
pip install build
python -m build --wheel

# In main.js, load via micropip instead of shakar_bundle.py:
# await pyodide.loadPackage("micropip");
# await micropip.install("./dist/shakar_ref-0.1.0-py3-none-any.whl");
```

## Files

- `index.html` - Main page
- `style.css` - Styling
- `main.js` - Pyodide integration
- `bundle.py` - Bundle generator
- `shakar_bundle.py` - Generated bundle (committed for Pages)
