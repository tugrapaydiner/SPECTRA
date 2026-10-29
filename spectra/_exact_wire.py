"""Strict SBQ1 query-bank codec shared by the public and research wrappers."""
from __future__ import annotations

import struct
from typing import Iterable

QUERY_BANK_HEADER = struct.Struct('<4sIII')
QUERY_BANK_MAGIC = b'SBQ1'


def _checked_query_rows(n: int, rows: Iterable[tuple[tuple[int, int], ...]]
                        ) -> tuple[tuple[tuple[int, int], ...], ...]:
    if type(n) is not int or not 0 <= n <= 1_000_000:
        raise ValueError('n must be an exact bounded nonnegative integer')
    checked = []
    for row in rows:
        if type(row) is not tuple:
            raise ValueError('frozen query rows must be exact tuples')
        copied = []
        for pin in row:
            if type(pin) is not tuple or len(pin) != 2:
                raise ValueError('frozen query pins must be exact pairs')
            vertex, allowed = pin
            if (type(vertex) is not int or not 0 <= vertex < n or
                    type(allowed) is not int or not 0 <= allowed < 1 << 64):
                raise ValueError('invalid frozen query pin')
            copied.append((vertex, allowed))
        checked.append(tuple(copied))
    return tuple(checked)


def pack_frozen_query_bank(n: int,
                           rows: Iterable[tuple[tuple[int, int], ...]]) -> bytes:
    """Serialize already-frozen rows while independently rechecking pin syntax.

    This avoids rebuilding and sorting the fixed graph after :class:`Case` has
    already admitted it. It does not trust query contents: exact tuple geometry,
    integer types, uint64 masks, and original-vertex bounds are checked again.
    """
    rows = _checked_query_rows(n, rows)
    if len(rows) > 1_000_000:
        raise ValueError('query bank exceeds one million rows')
    pins = sum(map(len, rows))
    if pins > 1_000_000:
        raise ValueError('query bank exceeds one million pins')
    size = QUERY_BANK_HEADER.size + 4 * (len(rows) + 1) + 12 * pins
    payload = bytearray(size)
    QUERY_BANK_HEADER.pack_into(payload, 0, QUERY_BANK_MAGIC, len(rows), pins, 0)
    offset_at = QUERY_BANK_HEADER.size
    vertex_at = offset_at + 4 * (len(rows) + 1)
    mask_at = vertex_at + 4 * pins
    cursor = 0
    struct.pack_into('<I', payload, offset_at, 0)
    for index, row in enumerate(rows):
        for vertex, allowed in row:
            struct.pack_into('<I', payload, vertex_at + 4 * cursor, vertex)
            struct.pack_into('<Q', payload, mask_at + 8 * cursor, allowed)
            cursor += 1
        struct.pack_into('<I', payload, offset_at + 4 * (index + 1), cursor)
    if cursor != pins:
        raise AssertionError('query bank pin accounting differs')
    return bytes(payload)


def unpack_query_bank(payload: bytes, n: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Strict Python reference decoder used by tests and artifact inspection."""
    if type(payload) is not bytes or type(n) is not int or n < 0:
        raise ValueError('query bank requires immutable bytes and nonnegative n')
    if len(payload) < QUERY_BANK_HEADER.size + 4:
        raise ValueError('truncated query bank')
    magic, count, pins, reserved = QUERY_BANK_HEADER.unpack_from(payload)
    expected = QUERY_BANK_HEADER.size + 4 * (count + 1) + 12 * pins
    if magic != QUERY_BANK_MAGIC or reserved != 0 or expected != len(payload):
        raise ValueError('query bank header or length differs')
    if count > 1_000_000 or pins > 1_000_000:
        raise ValueError('query bank count exceeds supported bound')
    offset_at = QUERY_BANK_HEADER.size
    vertex_at = offset_at + 4 * (count + 1)
    mask_at = vertex_at + 4 * pins
    offsets = struct.unpack_from('<' + 'I' * (count + 1), payload, offset_at)
    if offsets[0] != 0 or offsets[-1] != pins or any(a > b for a, b in zip(offsets, offsets[1:])):
        raise ValueError('query bank offsets differ')
    rows = []
    for begin, end in zip(offsets, offsets[1:]):
        row = []
        for index in range(begin, end):
            vertex = struct.unpack_from('<I', payload, vertex_at + 4 * index)[0]
            allowed = struct.unpack_from('<Q', payload, mask_at + 8 * index)[0]
            if vertex >= n:
                raise ValueError('query bank vertex outside graph')
            row.append((vertex, allowed))
        rows.append(tuple(row))
    return tuple(rows)
