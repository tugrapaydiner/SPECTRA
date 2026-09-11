# M16 development serialization repair

Before opening confirmation, independent replay rejected the original fixed-pool NPZ exports. A Python dictionary used the flat names `validity`, `quality` and `improvement` both for truth arrays and predicted probabilities; the later probability mapping overwrote those exported truth fields.

The fitted heads and in-memory selection/calibration summaries used the correct `Pool` targets. Online solves were separate and unaffected. Nevertheless, the exported arrays did not satisfy their intended contract, so the original exports are retained under `development_repair_history/pool_export_v1` rather than presented as valid evidence.

The repair gives truth and predictions disjoint `label_*` and `prediction_*` names. It reconstructs each fixed pool from the exact frozen core, directions and dataset, verifies that every decoded candidate and every previously stored predicted probability is unchanged, and exports the recovered truth arrays. Old and new SHA-256 identities are recorded. No fitting, data selection, threshold or online prediction was changed. A regression test deliberately uses identical target/prediction names and checks that both survive export.

Independent reference-free replay then checked all 28,672 development/validation online answers and all 12,288 fixed-pool answers successfully. This is a replay of recorded predictions through an independent Python checker, not a claim of independently retraining the neural models. Confirmation remained unopened throughout the repair.

The final confirmation freeze includes this history and the corrected exporter. The historical M14/M15 archives are unchanged.
