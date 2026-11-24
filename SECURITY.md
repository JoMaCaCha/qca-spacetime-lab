# Security Policy

## Scope & Architecture

**QCA Spacetime Lab** is a static, client-side scientific web application (`index.html` + compiled Rust/WebAssembly SIMD in `pkg/`) paired with local native Rust CLI tools (`qsu_cli`) and standalone Python verification scripts.

- The web laboratory runs **100 % locally in the user's browser**.
- It has **no backend server, no authentication, no user accounts, no database, and no tracking/analytics**.
- Only a lightweight UI language preference (`qca_lang`: `en` or `es`) is stored in browser `localStorage`.

## Supported Versions

Security fixes are applied to the latest release on the `main` branch:

| Version | Supported |
| --- | --- |
| `1.0.x` (latest) | Yes |
| `< 1.0.0` | No |

## Reporting a Vulnerability

If you discover a security vulnerability (for example, in a Rust/WebAssembly dependency, GitHub Actions workflow, or XSS/injection vector in `index.html`), please report it privately rather than opening a public issue:

- **Email:** `johan.m.cabrera.ch@gmail.com`
- **Subject line:** `[Security] QCA Spacetime Lab — <short summary>`

Please include a clear description of the issue and minimal reproduction steps. You can expect an initial acknowledgment within 5 business days.
