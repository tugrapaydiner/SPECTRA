# Native-platform acceptance

The SVM builders now recognize **64-bit little-endian Linux x86-64/ARM64** and
**Windows x64**. This changes build/link integration, not the learned model,
preprocessing equations, scheduling rules, or numerical source files.

## Build on the machine that will execute it

Linux requires a C++17 compiler and CPython development headers. On Windows use
GIL-enabled x64 CPython with its headers/import library and the **x64 Native Tools
Command Prompt for Visual Studio 2022** (MSVC). From that environment:

```python
from spectra.svm import build_runtime
from spectra.svm_preprocess_native import build_preprocessor

svm_library = build_runtime('native-build')
preprocessor_library = build_preprocessor('preprocessor-build')
```

Both directories must be new. Windows produces `spectra_svm.dll` and a `.pyd`
for the running Python ABI; Linux produces `.so` files. Builds are explicit, never
an install/import side effect. A missing compiler, compiler error, or timeout
leaves `build.json`, not an unannounced fallback binary. Only our own sources and
build outputs are included in receipts; no compiler/OS library is bundled.

`target='avx2'` is an explicit x86-64 option for a machine supporting those
instructions; there is no runtime CPU dispatch. ARM uses `portable`, not AVX2
emulation. Windows builds accept MSVC only and reject ambient CL/_CL_/LINK/_LINK_
flags that could silently override the recorded arithmetic options. MSVC uses
`/fp:strict`; Linux retains `-fno-fast-math -ffp-contract=off`. Contraction and
reassociation are not enabled to make a platform test pass.

The CPython extension is **not abi3**. Build it separately per interpreter ABI.
Free-threaded Python, subinterpreters, 32-bit hosts, big-endian hosts, macOS and
Windows ARM are not included. This is not blanket portability for historical
PyTorch-based research modules.

## File paths and ownership

Use `PreparedModel` or `PreparedPipeline` for Unicode model paths: Python reads
bounded model bytes before calling C++. Legacy `Session` opens a file through
its retained narrow C++ loader and therefore requires an ASCII model path on
Windows; it rejects a non-ASCII path explicitly rather than misdecoding it.
Source/build/library paths in the new builders may contain Unicode characters.
Existing request ownership, independent worker caches and close/reentry contracts
are unchanged.

## What a green platform job establishes

The new `svm-portability` matrix runs on native Windows x64 (Python3.11/3.13) and
Linux ARM64 (Python3.13). It executes the SVM suite, builds the wheel **through its
sdist**, compares every packaged source member, installs into a fresh venv without
numerical frameworks, and compiles/executes both native components there. The
existing Linux x86-64 matrix remains active.

A new C test probe reads the target's own FE_DOWNWARD constant instead of assuming
Linux/x86's numeric constant. Rounding rejection tests are executed, not skipped
because the constant differs. POSIX-signal cases remain platform-specific: they
are not Windows interrupt acceptance. Explicit skip counts belong in each receipt.

No cross-OS bitwise claim follows from preserving source arithmetic: system `exp`
implementations may differ. Compare every prediction on the target, and use the
independent ordered receipt checker under its stated per-host numerical contract.
These jobs are compatibility evidence, not independent researcher reproduction,
production load tests, or performance results on a user's particular computer.

## Primary references

- MSVC strict floating-point behavior: https://learn.microsoft.com/en-us/cpp/build/reference/fp-specify-floating-point-behavior
- CPython Windows extension linking: https://docs.python.org/3.13/extending/windows.html
- GitHub native runner availability: https://docs.github.com/en/actions/reference/runners/github-hosted-runners

Acceptance is the recorded outcome of a completed **exact-head** run, not this
workflow's presence in the repository. Initial failures and fixes must remain.
