"""Fixed-universe ordered set with exact rank selection and bounded-size words.

The Fenwick tree indexes 256-bit blocks, not individual clauses. Selection uses
ascending integer rank, exactly matching sorted(set)[rank], without rebuilding a
sorted tuple. This is a classical order-statistic data structure, not a new
search policy. The implementation is private to the indexed search backend.
"""
from __future__ import annotations


class RankedSet:
    """Mutable integer set; O(log universe) updates/select, sorted iteration.

    Blocks bound Python big-integer operations to 256 bits. The prefix tree and
    total are updated only when membership changes. Independent instances never
    share storage. Public validation occurs before mutation.
    """
    __slots__ = ("_limit", "_blocks", "_tree", "_total", "_top")
    _SHIFT = 8
    _WIDTH = 1 << _SHIFT

    def __init__(self, limit: int, values=()):
        if type(limit) is not int or limit < 0:
            raise ValueError("limit must be a nonnegative integer")
        self._limit = limit
        self._blocks = [0] * ((limit + self._WIDTH - 1) >> self._SHIFT)
        for value in values:
            self._validate(value)
            self._blocks[value >> self._SHIFT] |= 1 << (value & (self._WIDTH - 1))
        self._tree = [0] + [b.bit_count() for b in self._blocks]
        for i in range(1, len(self._tree)):
            parent = i + (i & -i)
            if parent < len(self._tree):
                self._tree[parent] += self._tree[i]
        self._total = sum(b.bit_count() for b in self._blocks)
        self._top = 1 << (len(self._blocks).bit_length() - 1) if self._blocks else 0

    def _validate(self, value: int) -> None:
        if type(value) is not int or not 0 <= value < self._limit:
            raise ValueError("value must be an integer inside the fixed universe")

    def __len__(self) -> int:
        return self._total

    def __contains__(self, value) -> bool:
        if type(value) is not int or not 0 <= value < self._limit:
            return False
        return bool(self._blocks[value >> self._SHIFT] & (1 << (value & (self._WIDTH - 1))))

    def __iter__(self):
        for block, bits in enumerate(self._blocks):
            while bits:
                low = bits & -bits
                yield (block << self._SHIFT) + low.bit_length() - 1
                bits ^= low

    def _update(self, block: int, delta: int) -> None:
        self._total += delta
        index = block + 1
        while index < len(self._tree):
            self._tree[index] += delta
            index += index & -index

    def add(self, value: int) -> None:
        self._validate(value)
        block = value >> self._SHIFT
        bit = 1 << (value & (self._WIDTH - 1))
        if not self._blocks[block] & bit:
            self._blocks[block] |= bit
            self._update(block, 1)

    def discard(self, value: int) -> None:
        self._validate(value)
        block = value >> self._SHIFT
        bit = 1 << (value & (self._WIDTH - 1))
        if self._blocks[block] & bit:
            self._blocks[block] ^= bit
            self._update(block, -1)

    def remove(self, value: int) -> None:
        self._validate(value)
        if value not in self:
            raise KeyError(value)
        self.discard(value)

    def select(self, rank: int) -> int:
        """Return sorted(self)[rank] without materialization (no negative ranks)."""
        if type(rank) is not int or not 0 <= rank < self._total:
            raise IndexError("rank must be an integer inside the current set")
        index = 0
        step = self._top
        while step:
            nxt = index + step
            if nxt < len(self._tree) and self._tree[nxt] <= rank:
                rank -= self._tree[nxt]
                index = nxt
            step >>= 1
        bits = self._blocks[index]
        offset = 0
        width = self._WIDTH >> 1
        while width:
            mask = (1 << width) - 1
            lower = bits & mask
            count = lower.bit_count()
            if rank >= count:
                rank -= count
                offset += width
                bits >>= width
            else:
                bits = lower
            width >>= 1
        return (index << self._SHIFT) + offset
