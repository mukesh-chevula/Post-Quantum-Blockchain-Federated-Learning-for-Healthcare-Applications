from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from typing import Any


def blake3_256(data: bytes) -> bytes:
    """Return a 32-byte BLAKE3 digest, falling back to SHA-256 if blake3 isn't installed."""
    try:
        from blake3 import blake3  # type: ignore

        return blake3(data).digest(length=32)
    except Exception:
        return hashlib.sha256(data).digest()


def hash32(data: bytes) -> bytes:
    """Protocol-wide 32-byte hash (BLAKE3-256 preferred)."""
    return blake3_256(data)


def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def sha256_hex(data: bytes) -> str:
    return sha256(data).hex()


def sha256_bytes32(data: bytes) -> bytes:
    return sha256(data)[:32]


def to_bytes32_hex(data: bytes) -> str:
    return "0x" + sha256_bytes32(data).hex()


def json_dumps_canonical(obj: Any) -> str:
    def default(o: Any):
        if is_dataclass(o):
            return asdict(o)
        if isinstance(o, (bytes, bytearray)):
            return {"__bytes__": True, "hex": bytes(o).hex()}
        raise TypeError(f"Unsupported type: {type(o)!r}")

    return json.dumps(obj, default=default, separators=(",", ":"), sort_keys=True)


def json_loads_bytes(obj: Any) -> Any:
    if isinstance(obj, dict) and obj.get("__bytes__") is True:
        return bytes.fromhex(obj["hex"])
    if isinstance(obj, dict):
        return {k: json_loads_bytes(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_loads_bytes(v) for v in obj]
    return obj
