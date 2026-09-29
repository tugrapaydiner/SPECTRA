# Stronger deployment controls after the complete first measurement

All24 model artifacts and official predictions remain frozen. The first complete
1176-cell matrix is retained without deletion, pooling or selective retiming.
The local-prototype model has better observed accuracy than the selected MLP on
all four tasks, but its apparent speed advantage needs stronger implementation
controls: the first MLP runs ordered rowwise loops rather than a tuned batch BLAS,
and the SVM control uses the original full-precision SPECTRA engine rather than
its already developed finite-kernel executor.

Add a single-threaded OpenBLAS batch MLP on the SAME trained/scaled weights, including
raw-code normalization, scaling, packing, allocation, bias/ReLU, all layers and
fresh labels inside the timer. No numerical Python layer dispatch. Its reduction
order may differ; check every label and disclose it is not a bitwise same-score
control. Link the installed pinned OpenBLAS library, recording its hash/config,
not silently copy a new classifier. No portable no-library deployment claim for it.

Also give the SAME new SVC a raw-uint8 wrapper around the unchanged previous
finite/adaptive SPECTRA executor. Its domain detection and conditional numerical
contract remain inherited; charge normalization within the wrapper, never before
timing. Accept only if all frozen predictions agree; retain original native SVC.
This control is an existing engine, not new learning or a separate current baseline
model selected from the test.

Run the entire matrix again with all original14 arms plus mlp_blas and svc_finite:
four tasks, three batches1/32/256, seven fixed shuffled repetitions =1344 cells.
One pinned core and one BLAS thread, no simultaneous fitting/compilation. Use new
seed2026092912. No candidate or learned parameter changes from the first complete
run. Final reporting uses this stronger complete run and keeps the earlier run
separate. Original learning/layout gates and all failed outcomes stay unchanged.
The extra baseline work is triggered by implementation review, not model tuning.
