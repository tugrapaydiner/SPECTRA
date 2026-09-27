# Recorded development deviations

The first focused invocation had 51 passes and three test-marker setup errors:
a positional `skipif` reason was corrected to `reason=...`. No runtime result was
accepted for those three tests until the corrected complete 54-case run passed.

After the first complete trace, review found a possible same-thread close between
the Python live-handle check and entry into the busy region. The busy flag was
moved inside try/finally and a second live-handle check added before entering C.
A trace-hook regression test closes the worker at that exact boundary. This is
lifetime hardening, not tuning from benchmark timing. The first trace, original
bound sources and results remain retained. A second complete run using the same
protocol is the accepted final-source performance result; they are not pooled.

The first full suite completed with 1965 passes before this Python-only hardening
and the additional regression test. Its receipt remains separately named. The
final source's focused/full results must be separately recorded, not inferred.
