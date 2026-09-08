# M14 Attempt 5 result table

Evidence surface: **confirmation**

Candidate: original dim-64 dual-stream FP TRM with reference-free semantic early exit.

| config | semantic validity | median complete-solve ms | mean supervision steps | mean shared-block apps | early-exit fraction | role |
|---|---:|---:|---:|---:|---:|---|
| symbolic_exact | 1.000000 | 0.560654 |  |  |  | context |
| dual_stream_exit_k4 | 0.984570 | 1.215085 | 1.2484 | 2.4969 | 0.9846 | candidate |
| single_pass_fp | 0.833203 | 1.234590 |  | 2.0000 |  | primary baseline |
| single_pass_int8 | 0.833594 | 1.403155 |  | 2.0000 |  | context |

Complete-solve latency includes every executed dual-stream recurrence step and semantic validity check. Unmatched points are not called iso-budget.
