# Offline streaming deployment

The runner provides an application boundary around `PreparedPipeline`: UTF-8 JSONL
raw rows in, a complete new file of model predictions out. No training, downloads,
network listener, subprocess per prediction, or automatic compilation occurs.
It preserves the model, preprocessing arithmetic, schedule and class mapping.

## Command

Build the SVM library and optional preprocessing extension explicitly, as described
in [native-platform support](SVM_PORTABILITY.md). Then run:

```bash
spectra svm run exported-model --library /path/to/runtime-library \
  --preprocessor /path/to/preprocessing-extension \
  --input rows.jsonl --output predictions.jsonl --batch-rows 128
```

On Windows, `py -3.13 -m spectra svm run ...` is equivalent after installing the
package into that interpreter. Use the actual `.dll` and matching `.pyd` paths.
Input `-` reads binary standard input. Output must be a new regular-file path;
there is deliberately no partially emitted stdout-prediction mode. The command
prints a completion summary to stdout and errors to stderr. Exit0 means completed;
exit2 means an input/settings/I/O failure. Interrupts use Python's ordinary interrupt
exit behavior. A published result remains complete if stdout later breaks.

To avoid the compiled preprocessor, omit `--preprocessor` and select
`--engine two_stage`. With the extension supplied, `two_stage` uses its regular
full-chunk path; `fused` uses its bounded native feature tile. No default schedule
changes: `beretta_cert` remains default; `--schedule binary_stream` is explicit.

## Raw-row wire format

Each line is one JSON array in the fitted plan's column order. For example, a plan
with columns `['age', 'category']` accepts `[31.5, "known-category"]` or
`[null, "known-category"]`. That is an illustration, not a universal schema.
The exported preprocessing plan supplies the actual schema and missing-value rules.

Only numbers, strings and null are accepted. Booleans, objects, nested arrays,
nonfinite numbers (including overflowing exponents), malformed UTF-8, lone
surrogates and blank lines fail. Category strings are bounded to4,096 UTF-8 bytes.
No implicit string-to-number coercion or schema reordering occurs. CRLF and a final
line without a newline are accepted. Numeric null uses the stored fitted fill;
category null remains invalid unless the existing plan supports it (currently not).
Domain feature extraction and original file-format parsing are caller responsibilities.

## Bounded memory, not bounded total output

Defaults are128 rows per chunk,1MiB per encoded line,8MiB of encoded rows per chunk,
10million total rows and256MiB of output. The actual row cap also respects the native
8million-element limit. Command flags can adjust these positive limits explicitly.
One extra encoded look-ahead line can coexist with a full chunk; decoded Python
objects can occupy more memory than their encoded text. For supported built-in rows,
fused transformed scratch retains its previous tile bound; binary-stream packing,
model data, worker caches and other interpreter storage are additional.

Long inputs are processed by successive independent chunks without retaining all
input rows or results in RAM. Source bytes are hashed as consumed. The output file
uses disk proportional to output size and has its own cap. No per-request wall-clock,
energy, queueing, concurrency or total-process-memory guarantee is implied.

## Complete-result publication

A private temporary file is created in the output directory. Only after EOF,
all inference and output writes, flush and file fsync does the runner link it to
the requested output name with create-if-absent semantics. Existing files, symlinks
and concurrently created output names are never replaced. A filesystem that does
not support this hard-link operation fails; there is no overwriting fallback.

The directory must be trusted and stable. Network filesystems, directory mutation,
private runtime tampering and hostile native libraries are outside the contract.
Temporary cleanup is best effort: a forced kill or cleanup failure can leave a
`.spectra-*.partial` file, which is NOT a completed output. Partial files may contain
sensitive inputs' predictions; protect the directory and delete abandoned files
according to the application's retention policy. No directory-fsync or crash-durable
commit claim is made. Once publication succeeds, a later interrupt does not undo it.

Invalid later rows and interrupts before publication do not return a partial final
file. Internal computation, source consumption and native counters are not rolled
back. No retained global answer cache or cross-input kernel reuse is introduced.

## Output and verification boundary

The first record contains `spectra.svm.stream.v1`, the model/plan hashes, columns,
labels, explicit float64 input convention, selected execution profile and limits.
Each prediction has a zero-based index and original label. The final `complete`
record includes row/byte counts, raw input SHA256 and the SHA256 of all preceding
output bytes. The stdout summary additionally binds the complete output bytes.

These records detect accidental corruption only when the expected identities are
trusted. They are not signatures, true-label proofs or independent SVM verification.
Use [decision receipts](SVM_RECEIPTS.md) when numerical replay is required. This
CLI deliberately avoids the cost and disclosure of a receipt on every row.

The [retained-corpus replay](../experiments/streaming/README.md) compares every
output with fixed expected predictions and checks actual transformed feature bytes.
It is automated compatibility evidence, not outside researcher reproduction or a
new held-out benchmark.

Primary implementation references: Python JSON parsing limits and `os.link`:
https://docs.python.org/3.13/library/json.html
https://docs.python.org/3.13/library/os.html#os.link
