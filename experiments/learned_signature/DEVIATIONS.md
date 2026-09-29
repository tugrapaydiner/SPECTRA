# Development findings and recorded clarifications

The committed protocol's phrase "Seven radial scales" conflated the seven global
gamma multipliers with the five mixture components. The implementation committed
before development fits uses five components (0.25, 0.5, 1, 2, 4), for each of seven
global multipliers. All trials use that implementation; no trial is replaced.

After the first outer validation split, the separately committed pair-expert
amendment adds a nested selection control. It does not alter the original primary
hypothesis, data split, training limits or final-evaluation boundary.

The complete three-split validation experiment rejects the original promotion
hypothesis: no learned metric/alignment family improves on the tuned RBF family
on both tasks. The pair-specific secondary learner also fails to improve average
validation accuracy on either task. Final evaluation remains descriptive and
cannot reverse this failed validation gate. All families are retained for the
predeclared single final fit, not tuned again in response to final-test scores.

Two potential new datasets were unavailable because acquisition failed. No new
independent test set is claimed. The historical Letter and Pendigits final
partitions have not been used for fitting or model selection in this continuation,
but they were consumed in previous work and are not fresh confirmation.
