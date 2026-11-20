"""Modelo de referencia bit-exacto del acelerador (NumPy)."""
import numpy as np


def requant(acc, bias, shift, relu):
    """Re-cuantización idéntica al hardware: bias, redondeo, ReLU, saturación."""
    x = np.asarray(acc, dtype=np.int64) + np.asarray(bias, dtype=np.int64)
    if shift:
        x = (x + (1 << (shift - 1))) >> shift
    if relu:
        x = np.maximum(x, 0)
    return np.clip(x, -128, 127).astype(np.int64)


def gemv(W, x, bias, shift, relu):
    """y = requant(W @ x + bias). Devuelve (acc int32 sin bias, y int8)."""
    acc = W.astype(np.int64) @ x.astype(np.int64)
    return acc, requant(acc, bias, shift, relu)


def pack_int8(vals):
    """Empaqueta valores int8 en palabras de 32 bits (little-endian, 4 por palabra)."""
    vals = np.asarray(vals, dtype=np.int64)
    assert len(vals) % 4 == 0
    words = []
    for i in range(0, len(vals), 4):
        w = 0
        for j in range(4):
            w |= (int(vals[i + j]) & 0xFF) << (8 * j)
        words.append(w)
    return words


def pack_int4(vals):
    """Empaqueta valores int4 (-8..7) en palabras de 32 bits (8 por palabra)."""
    vals = np.asarray(vals, dtype=np.int64)
    assert len(vals) % 8 == 0 and vals.min() >= -8 and vals.max() <= 7
    words = []
    for i in range(0, len(vals), 8):
        w = 0
        for j in range(8):
            w |= (int(vals[i + j]) & 0xF) << (4 * j)
        words.append(w)
    return words


def to_signed(v, bits):
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >> (bits - 1) else v
