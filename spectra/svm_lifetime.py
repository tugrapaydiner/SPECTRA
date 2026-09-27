"""Internal owning handles and request leases for synchronous native SVM calls.

An RLock serializes threads but does not prevent a Python signal handler on the
same thread from re-entering a session. Every pointer-using operation therefore
holds a strong reference to its native owner as well as the lock. Explicit close
is deferred while an operation is active. This is not protection from process
termination, repeated exceptions during cleanup, private-state tampering, or
arbitrary caller-supplied native pointers.
"""
from __future__ import annotations

from contextlib import contextmanager
import threading


class _Handle:
    """One destruction obligation; the library outlives its native allocation."""
    __slots__ = ('value', '_library', '_destroy')

    def __init__(self, value, library, destroy):
        self._library = library
        self._destroy = destroy
        self.value = value

    def __del__(self):
        # Detach BEFORE calling C: destruction can release the GIL or deliver a
        # signal on return. No second close may observe an owned live pointer.
        value = getattr(self, 'value', None)
        self.value = None
        if value:
            getattr(self._library, self._destroy)(value)


class _NativeOwner:
    """Mixin for legacy sessions, prepared-model owners, and shared workers."""
    def _init_lifetime(self, kind):
        self._lock = threading.RLock()
        self._resource = None
        self._native_active = False
        self._closing = False
        self._owner_kind = kind

    def _adopt(self, pointer, library, destroy):
        if not pointer or self._resource is not None or self._closing:
            raise ValueError('invalid native ownership transfer')
        self._resource = _Handle(pointer, library, destroy)

    @property
    def _handle(self):
        resource = self._resource
        return resource.value if resource is not None else None

    @contextmanager
    def _operation(self):
        """Lease the allocation through native execution AND result validation.

        The strong local reference is a separate lifetime guarantee from the
        reentrancy flag: a close between checking and entering cannot leave C
        holding only an unowned integer address. No lock is held while idle.
        """
        with self._lock:
            if self._native_active:
                raise ValueError('reentrant native operation')
            resource = self._resource
            if self._closing or resource is None:
                raise ValueError('closed ' + self._owner_kind)
            try:
                self._native_active = True
                # A signal may have closed this object before the assignment.
                if self._closing or self._resource is not resource:
                    raise ValueError('closed ' + self._owner_kind)
                yield resource.value
            finally:
                self._native_active = False
                if self._closing:
                    self._resource = None
                # `resource` stays alive until the context manager unwinds.

    def close(self):
        with self._lock:
            self._closing = True
            if not getattr(self, '_native_active', False):
                self._resource = None

    def __enter__(self):
        with self._lock:
            if self._closing or self._resource is None:
                raise ValueError('closed ' + self._owner_kind)
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        if hasattr(self, '_lock'):
            self.close()
