# Final pre-test freeze

September28,2026. No natural-data outer-holdout prediction has occurred at this point. All86 selected models are frozen:36 original kernel models,24 pre-test interaction extensions,24 external controls and2 explicitly synthetic parity models. The natural experiments have3 development splits per task, not3 independent holdouts.

The supplied PRETEST_FREEZE.json binds359 source/data/model/receipt members. Its SHA256 is `5bf5191027bcd1e286f81b832f14eb895692d4fb18ce4c5b4657b7b605737e97`.

- Original training receipt: `8154bb6a5b88497dd39cba852d54f9b211cba121e30ca0b0c76b6db4f5c64f24`.
- Interaction-extension receipt: `b39dbef92da3baef8d4c5abfdcdc9b2c17d814620b65abfe8ae25156ca80d165`.
- Controlled parity selection receipt: `cde6febf8134442c71fdecebdb3b00fd4d9b39744bb7ca32e1574858550f5682`.

Actual local pre-test checks:62 synthetic runtime/signature/ownership/spectral tests pass on the POPCNT and portable builds. All60 natural-data kernel models match their own selected sklearn predictions on12450 development model/input pairs. A separate coordinate-wise integer and ordered-score implementation matches all307530 development margins bit-for-bit. These are development correctness checks, not held-out accuracy.

The first projected metric and radial mixtures underperform on validation; the original primary remains unchanged. Additional models were defined in the pre-test amendments and retain their separate secondary status. The next evaluation verifies every bound member before reading either holdout and performs no fitting, reselection or change to the original acceptance thresholds. This receipt is our local prospective commitment, not independent preregistration or a correctness proof.