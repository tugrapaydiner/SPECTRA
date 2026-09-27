# Stronger matched-ISA control — before the second full timing run

The original protocol and first complete 1176-cell comparison remain unchanged and retained. It gave SPECTRA both portable and AVX2 builds while the unmodified LIBSVM and m2cgen baselines used portable strict O3 compilation. That supports a named-configuration result, not the strongest matched-ISA claim.

Build the SAME unmodified upstream LIBSVM3.37 source and SAME six already-generated C files with the additional -mavx2 flag, keeping -O3 -fno-fast-math -ffp-contract=off and the identical wrappers. Do not regenerate a different model, reassociate arithmetic, enable FMA contraction, remove failed observations or alter labels. The HAR generated-C timeout remains missing; this amendment does not reopen its budget.

Revalidate all new native competitor outputs against the already frozen sklearn predictions. Then repeat the ENTIRE original seven-model, three-batch-size, seven-repetition timing matrix with the same deterministic schedule. Keep all SPECTRA portable and AVX2 controls and the fixed linear HAR control. Run no other local numerical task concurrently. Do not mix rows between runs or select the better run per model. The final promoted table uses this stronger complete matched-ISA run; the first remains an explicit separate experiment.

The new quality result is not reopened: the fixed linear control was more accurate than the RBF classifier on the recorded HAR split. That application-level limitation remains regardless of an improved same-function SVM speed ratio. No runtime-default change or high hiring grade follows from either run.
