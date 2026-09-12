# Cross-version summary arithmetic correction

Before merge, replaying a Python 3.10 CI report on Python 3.13 exposed 28
last-bit differences in descriptive statistics (largest absolute difference
1.14e-13 ms). Original witnesses, path digests and every scientific gate agreed.
No completed timing or original report was overwritten or retimed. The original
source snapshots and reports remain with their producing Python environments.

Summary schema v2 uses exact integer nanosecond totals over the balanced
formula/seed/round inventory, dividing only at final means and bootstrap ratios.
It evaluates the same frozen statistic and thresholds without version-dependent
floating summation. Unbalanced inventories are rejected. This is an analysis
portability correction, not a performance intervention or a loosened tolerance.
Final CI produces new, separately retained observations from the corrected source;
its Python 3.10 records are verified again by the Python 3.13 release packager.
