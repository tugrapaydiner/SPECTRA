# Native-compiler amendment, before any accepted latency measurement

The model freeze and chronological test outcomes are unchanged. The full-SVM application admission failed its90% quality target; no classifier is retuned because of that result.

Original m2cgen0.10.0 exported the frozen237-support SVC in59.606 seconds of generation (65.700 whole process), producing1,360,995 bytes. Its unmodified generated C reached the declared180-second GCC-O3 compile timeout. No linked model was produced. This failure remains part of the primary comparator inventory.

Inspection found that subprocess.run(timeout=...) killed the compiler driver but left its cc1plus child running. The orphan was explicitly killed at about190.8 seconds of child lifetime. Therefore the initial harness did NOT enforce a180-second whole-process-tree bound; do not describe that initial total resource use as exactly capped. The original builder, failure receipt and cleanup observation are retained. The corrected helper launches an isolated process group and kills the entire group on timeout, with a regression test for child cleanup.

Before the first accepted native timing matrix, add one explicitly named alternative: compile the SAME original m2cgen C with the installed clang++ at O3, AVX2, no fast math and no contraction, within180 seconds/4GiB address space. Compile unchanged LIBSVM and SPECTRA with that compiler too, so a matching-compiler SVM comparison is available. Keep original GCC controls and its failure. No generation source, fitted coefficients or final-test selection changes.

Report compiler identity and source/binary digests, each compilation outcome and all available native timing arms. A successful Clang artifact is not a retroactive GCC success. This addition was decided from compilation feasibility, not by inspecting latency cells. If it also fails, record the missing comparator; do not keep choosing compilers until a favorable result appears.

The linear/MLP/Nyström controls retain their separately recorded GCC builds. Numerical acceptance is unchanged: exact observed decisions, and the predeclared score tolerance for nonbitwise lowerings. No operational safety, novel compiler method, high grading score or deployment superiority is established by this amendment.