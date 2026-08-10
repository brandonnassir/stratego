"""Byte-level primitives for the compact trajectory codec.

Nothing here knows anything about Stratego. It exists so that
:mod:`stratego.training.trajectory` can describe records field by field without
repeating varint arithmetic, and so that the encoder and the decoder read the
same primitive list in the same order.

The format is deliberately boring:

- unsigned integers are LEB128 varints;
- signed integers are zigzag varints;
- an optional unsigned integer is `0` for `None` and `value + 1` otherwise;
- booleans are packed into flag bytes by the caller, or written as one byte;
- floats are little-endian IEEE-754 `float32`, so a stored probability decodes
  to exactly the `float32` that was stored;
- text is written through a per-record string table, so the repeated reason and
  event-type strings cost one varint each.

Only the standard library is used. `zlib` provides the compressed size baseline
that `reports/phase_3_data/agent_03_trajectory_reconstruction.json` reports
alongside the raw size.
"""

from __future__ import annotations

import struct
import zlib

SERIALIZATION_VERSION = "trajectory_codec_v1"

# zlib level 6 is the stdlib default: a middle setting that neither dominates
# collection time nor flatters the storage numbers.
DEFAULT_COMPRESSION_LEVEL = 6

_FLOAT32 = struct.Struct("<f")


class CodecError(ValueError):
    """A record could not be encoded or decoded."""


def to_float32(value: float) -> float:
    """Round a Python float to the nearest `float32`, as a Python float.

    Producers call this before storing a probability so that an in-memory
    record and its decoded form compare equal. Without it, every record would
    differ from its own round trip by a fraction of a bit and no exact
    storage-fidelity assertion would be possible.
    """
    return _FLOAT32.unpack(_FLOAT32.pack(float(value)))[0]


class StringTable:
    """Interning table shared by one encoded record.

    Reason strings, phase names, behaviour types and policy version strings
    repeat on nearly every piece and every decision. Writing each distinct
    string once and referring to it by index keeps the uncompressed record
    small, which matters because the raw size is what a memory-resident replay
    buffer actually pays.
    """

    __slots__ = ("_index", "_strings")

    def __init__(self, strings: "list[str] | None" = None) -> None:
        self._strings: list[str] = list(strings or [])
        self._index: dict[str, int] = {text: position for position, text in enumerate(self._strings)}

    def intern(self, text: str | None) -> int:
        """Index of `text`, adding it if new. `None` is index `0`."""
        if text is None:
            return 0
        position = self._index.get(text)
        if position is None:
            self._strings.append(text)
            position = len(self._strings)
            self._index[text] = position
        return position

    def resolve(self, position: int) -> str | None:
        """Inverse of :meth:`intern`; index `0` is `None`."""
        if position == 0:
            return None
        if not 1 <= position <= len(self._strings):
            raise CodecError(f"string table index {position} is out of range")
        return self._strings[position - 1]

    @property
    def strings(self) -> tuple[str, ...]:
        return tuple(self._strings)

    def __len__(self) -> int:
        return len(self._strings)


class ByteWriter:
    """Append-only binary writer."""

    __slots__ = ("_buffer",)

    def __init__(self) -> None:
        self._buffer = bytearray()

    # -- integers ---------------------------------------------------------

    def uvarint(self, value: int) -> None:
        number = int(value)
        if number < 0:
            raise CodecError(f"uvarint cannot hold a negative value: {number}")
        while True:
            chunk = number & 0x7F
            number >>= 7
            if number:
                self._buffer.append(chunk | 0x80)
            else:
                self._buffer.append(chunk)
                return

    def svarint(self, value: int) -> None:
        """Zigzag-encoded signed integer."""
        number = int(value)
        self.uvarint((number << 1) ^ (number >> 63) if number < 0 else number << 1)

    def optional_uvarint(self, value: int | None) -> None:
        self.uvarint(0 if value is None else int(value) + 1)

    # -- booleans and flags ----------------------------------------------

    def boolean(self, flag: bool) -> None:
        self._buffer.append(1 if flag else 0)

    def flags(self, values: "tuple[bool, ...] | list[bool]") -> None:
        """Pack up to eight booleans into one byte, least significant first."""
        if len(values) > 8:
            raise CodecError("a flag byte holds at most eight booleans")
        packed = 0
        for position, flag in enumerate(values):
            if flag:
                packed |= 1 << position
        self._buffer.append(packed)

    # -- floats -----------------------------------------------------------

    def float32(self, value: float) -> None:
        self._buffer.extend(_FLOAT32.pack(float(value)))

    def float32_sequence(self, values: "tuple[float, ...] | list[float]") -> None:
        self.uvarint(len(values))
        for value in values:
            self._buffer.extend(_FLOAT32.pack(float(value)))

    # -- composite --------------------------------------------------------

    def ascending_uvarints(self, values: "tuple[int, ...] | list[int]") -> None:
        """Length-prefixed strictly ascending integers, stored as deltas.

        Legal action identifiers arrive ascending from the reference engine, so
        the delta between neighbours is small and almost always fits in a single
        varint byte.
        """
        self.uvarint(len(values))
        previous = -1
        for value in values:
            number = int(value)
            if number <= previous:
                raise CodecError(
                    f"ascending sequence expected, got {number} after {previous}"
                )
            self.uvarint(number - previous - 1)
            previous = number

    def blob(self, payload: bytes) -> None:
        self.uvarint(len(payload))
        self._buffer.extend(payload)

    def text(self, value: str) -> None:
        """Inline text. Prefer a :class:`StringTable` index for repeated text."""
        encoded = value.encode("utf-8")
        self.uvarint(len(encoded))
        self._buffer.extend(encoded)

    def raw(self, payload: bytes) -> None:
        self._buffer.extend(payload)

    def to_bytes(self) -> bytes:
        return bytes(self._buffer)

    def __len__(self) -> int:
        return len(self._buffer)


class ByteReader:
    """Sequential reader mirroring :class:`ByteWriter` exactly."""

    __slots__ = ("_data", "_position")

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._position = 0

    # -- integers ---------------------------------------------------------

    def uvarint(self) -> int:
        result = 0
        shift = 0
        while True:
            if self._position >= len(self._data):
                raise CodecError("truncated varint")
            byte = self._data[self._position]
            self._position += 1
            result |= (byte & 0x7F) << shift
            if not byte & 0x80:
                return result
            shift += 7
            if shift > 70:
                raise CodecError("varint is implausibly long")

    def svarint(self) -> int:
        raw = self.uvarint()
        return (raw >> 1) ^ -(raw & 1)

    def optional_uvarint(self) -> int | None:
        raw = self.uvarint()
        return None if raw == 0 else raw - 1

    # -- booleans and flags ----------------------------------------------

    def boolean(self) -> bool:
        return self._byte() != 0

    def flags(self, count: int) -> tuple[bool, ...]:
        if count > 8:
            raise CodecError("a flag byte holds at most eight booleans")
        packed = self._byte()
        return tuple(bool(packed & (1 << position)) for position in range(count))

    # -- floats -----------------------------------------------------------

    def float32(self) -> float:
        chunk = self._take(4)
        return _FLOAT32.unpack(chunk)[0]

    def float32_sequence(self) -> tuple[float, ...]:
        count = self.uvarint()
        chunk = self._take(4 * count)
        return struct.unpack(f"<{count}f", chunk)

    # -- composite --------------------------------------------------------

    def ascending_uvarints(self) -> tuple[int, ...]:
        count = self.uvarint()
        values = []
        previous = -1
        for _ in range(count):
            previous = previous + 1 + self.uvarint()
            values.append(previous)
        return tuple(values)

    def blob(self) -> bytes:
        return self._take(self.uvarint())

    def text(self) -> str:
        return self._take(self.uvarint()).decode("utf-8")

    # -- position ---------------------------------------------------------

    def _byte(self) -> int:
        if self._position >= len(self._data):
            raise CodecError("unexpected end of record")
        byte = self._data[self._position]
        self._position += 1
        return byte

    def _take(self, count: int) -> bytes:
        end = self._position + count
        if end > len(self._data):
            raise CodecError("unexpected end of record")
        chunk = self._data[self._position : end]
        self._position = end
        return chunk

    @property
    def position(self) -> int:
        return self._position

    @property
    def exhausted(self) -> bool:
        return self._position >= len(self._data)

    def expect_exhausted(self) -> None:
        """Fail loudly when a record carries bytes the decoder did not read."""
        if not self.exhausted:
            raise CodecError(
                f"{len(self._data) - self._position} trailing bytes after decoding"
            )


# ---------------------------------------------------------------------------
# String table framing
# ---------------------------------------------------------------------------


def write_string_table(table: StringTable) -> bytes:
    """Serialise a completed table; the body is written separately."""
    writer = ByteWriter()
    writer.uvarint(len(table))
    for text in table.strings:
        writer.text(text)
    return writer.to_bytes()


def read_string_table(reader: ByteReader) -> StringTable:
    count = reader.uvarint()
    return StringTable([reader.text() for _ in range(count)])


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------


def compress(payload: bytes, level: int = DEFAULT_COMPRESSION_LEVEL) -> bytes:
    return zlib.compress(payload, level)


def decompress(payload: bytes) -> bytes:
    try:
        return zlib.decompress(payload)
    except zlib.error as error:  # pragma: no cover - corrupt input path
        raise CodecError(f"could not decompress record: {error}") from error


__all__ = [
    "DEFAULT_COMPRESSION_LEVEL",
    "SERIALIZATION_VERSION",
    "ByteReader",
    "ByteWriter",
    "CodecError",
    "StringTable",
    "compress",
    "decompress",
    "read_string_table",
    "to_float32",
    "write_string_table",
]
