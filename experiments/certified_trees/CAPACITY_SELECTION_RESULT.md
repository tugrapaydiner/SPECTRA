# Capacity selection result — frozen before official-test prediction

Complete locked selection: 72 fits = 4 tasks × 2 grouped splits × 9 recipes.
The execution-only amendments changed thread count and imposed a uniform fit cap;
all nine recipes completed in the final bounded run. No official test prediction
from these newly trained capacity models has been made before this record.

Selection follows the original pooled-accuracy, fewer tree-predicates, smaller
model, fixed-order tie rules.

| task | selected | pooled validation | 256x6 control | gain |
|---|---|---:|---:|---:|
| Letter | 512 trees, depth8, lr .06 | 6081/6401 = 95.0008% | 5932/6401 = 92.6730% | +2.3278 points |
| Pendigits | 512 trees, depth8, lr .06 | 2969/2997 = 99.0657% | 2965/2997 = 98.9323% | +0.1335 |
| Satellite | 512 trees, depth7, lr .06 | 1613/1774 = 90.9245% | 1602/1774 = 90.3044% | +0.6201 |
| OptDigits | 512 trees, depth7, lr .06 | 1496/1530 = 97.7778% | 1489/1530 = 97.3203% | +0.4575 |

The prospective >=0.75-point validation improvement on at least two tasks is NOT
met at selection time; only Letter clears it. This is retained regardless of the
later official-test outcomes.

Next: refit each selected recipe and each 256x6 control exactly once on its complete
official training partition using random_seed 20260929, save CBM/JSON/C++ identities,
then open official tests once. No post-test capacity, learning-rate or depth change.
