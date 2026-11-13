"""Binary frame codec for array fields.

Layout: uint32 little-endian header length | UTF-8 JSON header, space-padded so
that the payload starts on a 4-byte boundary | raw little-endian float32 data.
The alignment lets a browser wrap the payload in a Float32Array without copying.
"""
from __future__ import annotations

import json
import struct
from typing import Any

import numpy as np


def encode_field(name: str, array: np.ndarray, **meta: Any) -> bytes:
    data = np.ascontiguousarray(array, dtype="<f4")
    header = json.dumps({"name": name, "shape": list(data.shape), "dtype": "float32", **meta}).encode()
    header += b" " * (-len(header) % 4)
    return struct.pack("<I", len(header)) + header + data.tobytes()


def decode_field(frame: bytes) -> tuple[dict[str, Any], np.ndarray]:
    (n,) = struct.unpack_from("<I", frame, 0)
    header = json.loads(frame[4:4 + n].decode())
    data = np.frombuffer(frame, dtype="<f4", offset=4 + n).reshape(header["shape"])
    return header, data
