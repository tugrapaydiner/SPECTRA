"""Canonical SBC1 packing stays byte-stable while avoiding scalar pack loops."""
from __future__ import annotations

import struct

import pytest

from spectra.exact import (
    ExactProblem,
    pack_session_packet,
    pack_validated_session_packet,
    unpack_session_packet,
)

_HEADER = struct.Struct('<4s7I')


def _scalar_reference(problem: ExactProblem,
                      rows: tuple[tuple[tuple[int, int], ...], ...]) -> bytes:
    pins = sum(map(len, rows))
    size = (_HEADER.size + 8 * problem.n + 8 * len(problem.edges)
            + 4 * (len(rows) + 1) + 12 * pins)
    payload = bytearray(size)
    _HEADER.pack_into(payload, 0, b'SBC1', problem.n, problem.palette,
                      len(problem.edges), len(rows), pins, 0, 0)
    masks_at = _HEADER.size
    edges_at = masks_at + 8 * problem.n
    offsets_at = edges_at + 8 * len(problem.edges)
    vertices_at = offsets_at + 4 * (len(rows) + 1)
    allowed_at = vertices_at + 4 * pins
    for index, mask in enumerate(problem.masks):
        struct.pack_into('<Q', payload, masks_at + 8 * index, mask)
    for index, (left, right) in enumerate(problem.edges):
        struct.pack_into('<II', payload, edges_at + 8 * index, left, right)
    cursor = 0
    struct.pack_into('<I', payload, offsets_at, 0)
    for row_index, row in enumerate(rows):
        for vertex, allowed in row:
            struct.pack_into('<I', payload, vertices_at + 4 * cursor, vertex)
            struct.pack_into('<Q', payload, allowed_at + 8 * cursor, allowed)
            cursor += 1
        struct.pack_into('<I', payload, offsets_at + 4 * (row_index + 1), cursor)
    return bytes(payload)


def test_canonical_problem_keeps_the_admitted_edge_tuple():
    edges = ((0, 1), (0, 2), (1, 2))
    problem = ExactProblem.create(3, 3, edges, (0b011, 0b110, 0b101))
    assert problem.edges is edges

    normalized = ExactProblem.create(3, 3, ((2, 1), [0, 2], (1, 2)),
                                     (0b011, 0b110, 0b101))
    assert normalized.edges == ((0, 2), (1, 2))


def test_bulk_session_packet_is_byte_identical_to_scalar_reference():
    problem = ExactProblem.create(
        5, 4,
        ((0, 1), (0, 4), (1, 3), (2, 4)),
        (0b0011, 0b0110, 0b1100, 0b1001, 0b0101),
    )
    rows = (
        (),
        ((4, 0b0101), (0, 0b0001), (4, 0b0001)),
        ((2, (1 << 63) | 0b0100),),
        ((1, 0),),
    )
    expected = _scalar_reference(problem, rows)
    assert pack_validated_session_packet(problem, rows) == expected
    assert pack_session_packet(problem, rows) == expected
    decoded, decoded_rows = unpack_session_packet(expected)
    assert decoded == problem
    assert decoded_rows == rows


def test_validated_packer_still_rejects_noncanonical_problem_objects():
    malformed = ExactProblem(2, 2, ((1, 0),), (0b11, 0b11))
    with pytest.raises(ValueError, match='sorted unique'):
        pack_validated_session_packet(malformed, ((),))
