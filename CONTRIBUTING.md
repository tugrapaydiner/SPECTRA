# Contributing

See [docs/development.md](docs/development.md) for setup, tests and CI.

- Keep pull requests focused, and include tests for behaviour changes.
- `ruff check .` and `python -m pytest` must pass; CI also runs sanitizers, Windows
  and the installed-wheel checks.
- Native code must stay explicit-build only, and every new answer path must be
  checked before it is returned (original-input check for SAT, independently
  replayed proof for UNSAT).
- Performance claims need the complete cost (preparation, execution, checking)
  measured against a strong baseline, with the benchmark committed under
  `benchmarks/`.
- Do not change the format of an existing receipt, packet or proof; add a new
  versioned format instead.
