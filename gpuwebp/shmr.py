"""Shared-memory image ring: zero-copy handoff of decoded RGBA arrays from
the pipeline to the verification subprocesses.

The main process allocates a ring of R buffers (each max_image_bytes). The
decoder writes an image into a slot; the verify payload carries only the
slot index; the worker reads the array straight from shared memory. This
removes both the 4MB pickle (main-process GIL time) and the PNG re-decode
(subprocess CPU) that the previous designs paid.
"""
import ctypes
import multiprocessing.shared_memory as _shm

import numpy as np


class ShmRing:
    def __init__(self, name, count, capacity):
        self.name = name
        self.count = count
        self.capacity = capacity
        self.shm = _shm.SharedMemory(create=True, size=count * capacity)

    def buf(self, slot):
        return self.shm.buf[slot * self.capacity:(slot + 1) * self.capacity]

    def as_array(self, slot, h, w):
        # RGBA uint8 view of a slot
        arr = np.frombuffer(self.shm.buf, dtype=np.uint8,
                            count=h * w * 4,
                            offset=slot * self.capacity)
        return arr  # flat; caller reshapes


_ring = None            # main-process side
_attached = None        # worker-process side


def create_ring(count, capacity):
    global _ring
    _ring = ShmRing(None, count, capacity)
    return _ring


def ring_name():
    return _ring.shm.name if _ring else None


def attach_ring(name, count, capacity):
    """Worker-side attach (idempotent per process)."""
    global _attached
    if _attached is None or _attached.name != name:
        _attached = _shm.SharedMemory(name=name)
    return _attached


def read_slot(name, count, capacity, slot, h, w):
    """Worker-side: RGBA (h, w, 4) array view on the shared slot."""
    shm = attach_ring(name, count, capacity)
    arr = np.frombuffer(shm.buf, dtype=np.uint8, count=h * w * 4,
                        offset=slot * capacity)
    return arr.reshape(h, w, 4)
