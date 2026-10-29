# README chart provenance

These original SVG figures render **published, rounded summary values already
present in this repository**. They are not new benchmark measurements, a replay
of raw timing records, or outside researcher validation.

## Sources and scopes

The source snapshot is `b9d9f90d28a3e7071080be38555cd6ca09afcccf`.
Exact SHA-256 values and every plotted number are retained in
[chart-data.json](chart-data.json).

**Indexed search:** [efficiency guide](../../docs/EFFICIENCY_GUIDE.md), all six
size/family cells and both flip budgets. Bars are indexed/reference cold-call
ratios, including preparation. The 4.21x callout uses the separately reported
pooled primary ratio, not an unweighted average of the bars. All large cases
return UNKNOWN; small-case regressions and cold-allocation increases remain.

**Native comparison:** [matched native results](https://github.com/tugrapaydiner/SPECTRA/blob/757babb05bd631c89075f374ba765a197a1fb77b/experiments/native_baselines/RESULTS.md),
all seven models at batch 32. Each bar divides the published cost by that model's
LIBSVM cost. SPECTRA uses the fixed default profile; no per-model best profile is
selected. The JSON also retains the published binary-stream costs. Missing HAR
code generation is `null`, never zero latency or a defeated competitor. The
faster linear classifier is a different fitted model, not same-model compression.

Both figures show relative cost on a zero-based linear axis. The dashed line is
reference cost 1.00. Bars above it are regressions. No invented error bars,
synthetic precision, pooled cross-host speedups or altered scientific gates are
introduced. The native ratios derive from the published three-decimal timings;
they are descriptive, not recomputed from inaccessible raw measurements.

## Regenerate

Use the [complete historical source tree](https://github.com/tugrapaydiner/SPECTRA/tree/757babb05bd631c89075f374ba765a197a1fb77b)
and its `scripts/render_readme_charts.py --check` command with Python 3.10+.
That generator verifies both source hashes and every plotted cell before
comparing its output byte for byte. Its tests cover source drift, accessibility,
missing assets and the missing-comparator convention. The current SVG and JSON
assets retain those bytes and provenance; no benchmarks were rerun.
