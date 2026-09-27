"""Bounded offline JSONL deployment for an already fitted SVM pipeline.

No fitting, downloading, compilation or numerical-framework import. Output is
staged beside its destination and published without replacing an existing file.
The artifact is an execution record, NOT an independently verified proof.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import BinaryIO

FORMAT = 'spectra.svm.stream.v1'


@dataclass(frozen=True)
class Limits:
    batch_rows: int = 128
    line_bytes: int = 1_048_576
    batch_bytes: int = 8_388_608
    total_rows: int = 10_000_000
    output_bytes: int = 268_435_456

    def __post_init__(self):
        for name, value in vars(self).items():
            if type(value) is not int or not 1 <= value <= sys.maxsize - 1:
                raise ValueError(name + ' must be a positive bounded integer')
        if self.batch_rows > 65536:
            raise ValueError('batch_rows exceeds 65,536')
        if self.line_bytes > self.batch_bytes:
            raise ValueError('line_bytes must not exceed batch_bytes')


def _constant(_):
    raise ValueError('nonfinite number')


def _finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('number outside binary64 range')
    return result


def _object(_):
    # Objects are never part of the flat raw-row wire format, including nested ones.
    raise ValueError('JSON objects are not raw rows')


def _row(raw: bytes, columns: int, line: int) -> list:
    try:
        values = json.loads(raw.decode('utf-8'), parse_constant=_constant,
                            parse_float=_finite_float, object_pairs_hook=_object)
        if type(values) is not list or len(values) != columns:
            raise ValueError('schema width')
        for value in values:
            if value is None:
                continue
            if type(value) is str:
                if len(value.encode('utf-8')) > 4096:
                    raise ValueError('string byte limit')
            elif type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError('invalid scalar')
        return values
    except (ValueError, TypeError, UnicodeError, OverflowError, RecursionError) as error:
        # Do not expose potentially sensitive input values through exception text.
        raise ValueError(f'invalid raw JSON row at line {line}') from None


@contextmanager
def _staged_output(destination: Path):
    """Publish a completed file with link/create-if-absent, never os.replace.

    Requires a trusted stable output directory on a filesystem supporting hard
    links. No safe-looking fallback that could overwrite a concurrent writer.
    A killed process may leave a private .partial file; this is not a sandbox.
    """
    destination = Path(destination).absolute()
    if os.path.lexists(destination):
        raise FileExistsError('output already exists')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', prefix='.spectra-', suffix='.partial',
                                         dir=destination.parent, delete=False) as output:
            temporary = Path(output.name)
            yield output
            output.flush()
            os.fsync(output.fileno())
        # Both paths are on the same filesystem. Existing files/links are refused
        # atomically. Unsupported hard links fail rather than changing semantics.
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                # Cleanup is best effort. Never remove an already published result.
                pass


class _Writer:
    def __init__(self, stream: BinaryIO, limit: int):
        self.stream, self.limit = stream, limit
        self.bytes = 0
        self.digest = hashlib.sha256()

    def write(self, value):
        raw = (json.dumps(value, ensure_ascii=True, allow_nan=False,
                          separators=(',', ':')) + '\n').encode('ascii')
        if self.bytes + len(raw) > self.limit:
            raise ValueError('output byte limit exceeded')
        written = self.stream.write(raw)
        if written != len(raw):
            raise OSError('incomplete output write')
        self.bytes += len(raw)
        self.digest.update(raw)


def run_stream(source: BinaryIO, destination: str | Path, prepared, *,
               limits: Limits | None = None, engine: str = 'fused',
               schedule: str = 'beretta_cert') -> dict:
    """Run an ordered raw-row stream with bounded chunks and staged disk output.

    source must be a blocking binary stream; rows are strict UTF-8 JSON arrays.
    A result is published only after clean EOF and successful inference. Prefix
    bytes, raw values and outputs are not returned on failure. Internal computation
    and source consumption are not rolled back. Input objects and native libraries
    remain trusted. This does not independently verify the classifier's decision.
    """
    from .svm_pipeline import PreparedPipeline
    from .svm_shared import SHARED_SCHEDULES, MAX_ELEMENTS
    from .svm_preprocess_native import NativePreprocessor
    if type(prepared) is not PreparedPipeline:
        raise ValueError('a stock PreparedPipeline is required')
    if limits is None:
        limits = Limits()
    if type(limits) is not Limits:
        raise ValueError('limits must be a Limits instance')
    if type(engine) is not str or engine not in ('two_stage', 'fused'):
        raise ValueError('engine must be two_stage or fused')
    if type(schedule) is not str or schedule not in SHARED_SCHEDULES:
        raise ValueError('unsupported schedule')
    if engine == 'fused' and type(prepared.preprocessor) is not NativePreprocessor:
        raise ValueError('fused execution requires an explicitly compiled preprocessor')
    width = len(prepared.columns)
    cap = min(limits.batch_rows, MAX_ELEMENTS // max(width, prepared.preprocessor.features))
    input_hash = hashlib.sha256()
    consumed = input_bytes = completed = batches = batch_bytes = 0
    batch = []
    with _staged_output(Path(destination)) as output, prepared.session() as worker:
        writer = _Writer(output, limits.output_bytes)
        writer.write({'format': FORMAT, 'record': 'header',
                      'model_sha256': prepared.preprocessor.model_sha256,
                      'preprocessing_sha256': prepared.preprocessor.sha256,
                      'columns': list(prepared.columns), 'labels': list(prepared.labels),
                      'input_dtype': 'float64', 'engine': engine, 'schedule': schedule,
                      'limits': vars(limits), 'effective_batch_rows': cap})

        def flush():
            nonlocal completed, batches, batch_bytes
            if not batch:
                return
            labels = (worker.predict_fused(batch, schedule=schedule) if engine == 'fused'
                      else worker.predict_many(batch, schedule=schedule))
            if len(labels) != len(batch):
                raise RuntimeError('runtime output count mismatch')
            for label in labels:
                writer.write({'record': 'prediction', 'index': completed, 'label': label})
                completed += 1
            batches += 1
            batch.clear()
            batch_bytes = 0

        while True:
            raw = source.readline(limits.line_bytes + 1)
            if type(raw) is not bytes:
                raise ValueError('source must return bytes from bounded readline')
            if not raw:
                break
            if len(raw) > limits.line_bytes:
                raise ValueError(f'input line {consumed + 1} exceeds byte limit')
            if consumed >= limits.total_rows:
                raise ValueError('total input row limit exceeded')
            # Byte bound accounts for encoded rows, not decoder/interpreter overhead.
            # One additional encoded look-ahead line can coexist with a full chunk.
            if batch and batch_bytes + len(raw) > limits.batch_bytes:
                flush()
            row = _row(raw, width, consumed + 1)
            consumed += 1
            input_bytes += len(raw)
            input_hash.update(raw)
            batch.append(row)
            batch_bytes += len(raw)
            if len(batch) == cap:
                flush()
        flush()
        prefix_hash = writer.digest.hexdigest()
        footer = {'record': 'complete', 'rows': completed, 'input_bytes': input_bytes,
                  'input_sha256': input_hash.hexdigest(), 'output_prefix_sha256': prefix_hash,
                  'batches': batches}
        writer.write(footer)
        report = {'status': 'COMPLETE', 'format': FORMAT, **{k: v for k, v in footer.items() if k != 'record'},
                  'output_bytes': writer.bytes, 'output_sha256': writer.digest.hexdigest(),
                  'effective_batch_rows': cap}
    return report


def add_parser(commands):
    parser = commands.add_parser('svm', help='Offline execution of supported fitted SVM pipelines')
    actions = parser.add_subparsers(dest='svm_action', required=True)
    run = actions.add_parser('run', help='Strict raw-row JSONL to a complete new result file')
    run.add_argument('bundle', type=Path)
    run.add_argument('--library', type=Path, required=True)
    run.add_argument('--preprocessor', type=Path)
    run.add_argument('--input', default='-', help='UTF-8 JSONL input path, or - for binary stdin')
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--engine', choices=['two_stage', 'fused'], default='fused')
    run.add_argument('--schedule', default='beretta_cert')
    run.add_argument('--batch-rows', type=int, default=128)
    run.add_argument('--max-line-bytes', type=int, default=1_048_576)
    run.add_argument('--max-batch-bytes', type=int, default=8_388_608)
    run.add_argument('--max-rows', type=int, default=10_000_000)
    run.add_argument('--max-output-bytes', type=int, default=268_435_456)


def execute(args) -> dict:
    from contextlib import nullcontext
    from .svm_pipeline import PreparedPipeline
    limits = Limits(args.batch_rows, args.max_line_bytes, args.max_batch_bytes,
                    args.max_rows, args.max_output_bytes)
    # Refuse replacement before reading input or loading any executable component.
    if os.path.lexists(args.output):
        raise FileExistsError('output already exists')
    source = nullcontext(sys.stdin.buffer) if args.input == '-' else open(args.input, 'rb')
    with source as stream, PreparedPipeline(args.bundle, args.library,
                     preprocessor_library=args.preprocessor) as prepared:
        return run_stream(stream, args.output, prepared, limits=limits,
                          engine=args.engine, schedule=args.schedule)
