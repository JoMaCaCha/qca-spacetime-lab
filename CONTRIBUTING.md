# Contributing to QCA Spacetime Lab

Thank you for your interest in contributing to **QCA Spacetime Lab**! Independent numerical audits, physics questions, bug reports, documentation improvements, and code pull requests are all welcome.

## Ways to Contribute

- **Scientific & Numerical Audits:** Independently verify results in `scripts/` or `src/`, report sign/normalization/convergence discrepancies, or suggest stronger null controls.
- **Bug Reports:** Report browser/WebAssembly glitches, UI layout issues, or numerical edge cases using the Bug Report issue template.
- **Code & Performance Improvements:** Optimize Rust/WASM SIMD kernels, add automated unit tests, or improve accessibility and bilingual (`EN | ES`) UI coverage in `index.html`.
- **Documentation:** Clarify equations, references, or build instructions in `README.md`, `RESEARCH_SUMMARY.md`, or `ANALISIS.md`.

## Development & Verification Setup

### 1. Rust & Native Tests
```bash
# Fast targeted verification of research modules (Phases 18–20, ~20 s)
cargo test --release --lib tests::phase18
cargo test --release --lib tests::phase19
cargo test --release --lib tests::phase20

# Full automated test suite (195 unit tests)
cargo test --release
```

### 2. Independent Python / NumPy / SciPy Cross-Checks
```bash
pip install -r requirements-ci.txt
python scripts/verify_via_a_mlodinow_brun.py
python scripts/verify_via_c_schwinger_exact.py
```

### 3. WebAssembly SIMD Build & Local Preview
```bash
# `.cargo/config.toml` already configures `-C target-feature=+simd128` for wasm32-unknown-unknown
wasm-pack build --target web --release
python -m http.server 8000
# Open http://localhost:8000/ in a SIMD-capable browser (Chrome 91+, Firefox 89+, Safari 16.4+, Edge 91+)
```

## Pull Request Guidelines

1. Fork the repository and create a descriptive branch from `main`.
2. Keep changes focused and well-tested:
   - Run `cargo test --release` (or the relevant phase test modules) and confirm zero regressions.
   - If touching Python scripts or physics equations, re-run the corresponding `scripts/verify_*.py` script.
   - If modifying `index.html`, verify both **EN** and **ES** UI modes locally over HTTP.
3. Document any physical assumptions, tolerances, and explicit limitations clearly—scientific honesty and reproducibility take priority over cosmetic claims.
4. Follow our [Code of Conduct](CODE_OF_CONDUCT.md).
