# Native-platform validation history

This records validation of the new build boundary, not a change to any historical
research outcome. Numerical source and model formats are unchanged.

## First native matrix: d846b7a

Run 36331528028 passed on native Linux ARM64 with Python3.13, including the
framework-free installed-wheel check. Windows3.11 and3.13 compiled the SVM and
preprocessor and executed the suite, but the oversized-JSON rejection case could
not enter its body: pytest embedded its megabyte-long parameter representation in
PYTEST_CURRENT_TEST, exceeding Windows' environment-variable size limit.

Commit61f10f2 replaces generated parameter names with bounded descriptive IDs.
The original oversized payload and all assertions remain. The malformed-model
fixture also retains its corrupted bytes in its test directory. No numerical
source or acceptance tolerance changes. The first failure is retained.

## Completed corrected matrix: 61f10f2

Run36331938182 completed successfully on all three native targets. Windows3.11
and3.13 each have433 passes and3 POSIX-only signal cases skipped. ARM64-3.13
has435 passes and1 Windows-only legacy-path case skipped. All three build the
wheel through its sdist, match145 packaged sources to their local checkout, then
pass48 installed-only checks without numerical frameworks. These are repeated
contracts across environments, not1301 unique tests or external replication.

Downloaded artifacts10935893076/10935699043/10935444794 have verified GitHub
digests, ZIP CRCs, manifests and exact source treee0638e5c5322974f839d19144e5682dfefb64268.
Inspection found Windows checkout converted text files to CRLF, so its wheel
matched its own checkout but not the LF source bytes on Linux. A final workflow
change disables checkout newline conversion before cloning. It does not relax
source comparison or change executable mathematics. Its exact-head outcome must
be recorded separately. Previous successful runs are not relabelled as that run.

All435 Linux-focused tests and the full2082-test local fast suite pass at the
first implementation tree, with1 Windows-only skip and the original16 slow
exclusions in the full suite. The later receipt-ID change is covered by its
49-case rerun. Final installed acceptance passes48 checks and145 source-member
comparisons. Those scopes overlap; they are not summed as a larger test corpus.

Windows tests run on Windows Server2022 x64, not a user's Windows11 desktop.
ARM tests run on native Linux ARM64, not emulation or macOS. No additional speed,
accuracy, production-interrupt, energy or whole-project portability claim follows.
