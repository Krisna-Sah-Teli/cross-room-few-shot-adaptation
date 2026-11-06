"""Independent PCAP/CSI reader for Room A (Nexmon BCM4366c0 format).

The decoder logic follows the packed floating-point layout used by the public
Nexmon CSI tooling for the BCM4366c0 (seemoo-lab/nexmon_csi, `unpack_float.c`):
format=1, nman=12, nexp=6, 10-bit autoscaling, 256 CSI words per observation.
"""
from __future__ import annotations

import struct
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterator

import numpy as np

@dataclass
class CSIObservation:
    """One decoded CSI observation from a UDP packet."""
    timestamp_seconds: float
    amplitude: np.ndarray       # shape (256,), float32
    real: np.ndarray            # shape (256,), int32
    imag: np.ndarray            # shape (256,), int32
    source_mac: str
    sequence: int
    core: int
    stream: int
    chanspec: int
    chip_field: int
    archive_member: str
    packet_index: int


def _decode_pcap_global_header(header: bytes) -> tuple[str, str]:
    if len(header) != 24:
        raise ValueError("PCAP global header must be 24 bytes.")
    # Classic PCAP magic values identify byte order and timestamp precision.
    magic = header[:4]
    formats = {
        b"\xd4\xc3\xb2\xa1": ("<", "microseconds"),
        b"\xa1\xb2\xc3\xd4": (">", "microseconds"),
        b"\x4d\x3c\xb2\xa1": ("<", "nanoseconds"),
        b"\xa1\xb2\x3c\x4d": (">", "nanoseconds"),
    }
    if magic not in formats:
        raise ValueError(
            f"Unsupported PCAP magic {magic.hex(' ')}; expected classic PCAP."
        )
    endian, precision = formats[magic]
    version_major, version_minor = struct.unpack(endian + "HH", header[4:8])
    if (version_major, version_minor) != (2, 4):
        raise ValueError(
            f"Unexpected classic PCAP version {version_major}.{version_minor}."
        )
    network = struct.unpack(endian + "I", header[20:24])[0]
    if network != 1:
        raise ValueError(
            f"Expected Ethernet link type 1, got {network}; "
            "this reader currently handles Ethernet PCAPs."
        )
    return endian, precision


class TruncatedCaptureError(ValueError):
    """Raised when a packet record extends beyond the end of the file."""


def _iter_classic_pcap_frames(pcap_bytes: bytes) -> Iterator[tuple[float, bytes, int]]:
    """Yield (timestamp_seconds, captured_frame_bytes, packet_index).

    Raises TruncatedCaptureError at an incomplete trailing record, after all
    complete packets before it have been yielded.
    """
    if len(pcap_bytes) < 24:
        raise ValueError("File is too short to contain a PCAP global header.")
    endian, precision = _decode_pcap_global_header(pcap_bytes[:24])
    fraction_scale = 1_000_000 if precision == "microseconds" else 1_000_000_000
    offset = 24
    packet_index = 0
    while offset < len(pcap_bytes):
        if offset + 16 > len(pcap_bytes):
            raise TruncatedCaptureError(f"Truncated packet header at byte offset {offset}.")
        ts_sec, ts_fraction, incl_len, orig_len = struct.unpack(
            endian + "IIII", pcap_bytes[offset:offset + 16]
        )
        offset += 16
        if offset + incl_len > len(pcap_bytes):
            raise TruncatedCaptureError(f"Truncated packet data at packet {packet_index}.")
        frame = pcap_bytes[offset:offset + incl_len]
        offset += incl_len
        timestamp = ts_sec + ts_fraction / fraction_scale
        yield timestamp, frame, packet_index
        packet_index += 1


def _parse_ethernet_ipv4_udp(frame: bytes) -> tuple[bytes, str, str, int, int] | None:
    """Return UDP payload, source MAC, destination MAC, source port, destination port."""
    if len(frame) < 14:
        return None
    ethertype = int.from_bytes(frame[12:14], "big")
    if ethertype != 0x0800:  # IPv4 only
        return None
    if len(frame) < 14 + 20:
        return None
    ip_start = 14
    version_ihl = frame[ip_start]
    if version_ihl >> 4 != 4:
        return None
    ihl = (version_ihl & 0x0F) * 4
    if ihl < 20 or len(frame) < ip_start + ihl:
        return None
    if frame[ip_start + 9] != 17:  # UDP
        return None
    ip_total_len = int.from_bytes(frame[ip_start + 2:ip_start + 4], "big")
    ip_end = min(len(frame), ip_start + ip_total_len)
    udp_start = ip_start + ihl
    if udp_start + 8 > ip_end:
        return None
    source_port, dest_port, udp_len = struct.unpack(">HHH", frame[udp_start:udp_start + 6])
    if udp_len < 8:
        return None
    udp_end = min(ip_end, udp_start + udp_len)
    udp_payload = frame[udp_start + 8:udp_end]
    src_mac = ":".join(f"{b:02x}" for b in frame[6:12])
    dst_mac = ":".join(f"{b:02x}" for b in frame[0:6])
    return udp_payload, src_mac, dst_mac, source_port, dest_port


def decode_bcm4366c0_words(words: np.ndarray | list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Decode packed BCM4366c0 CSI words into int32 real and imaginary vectors.

    This reproduces the following format operations:
    nbits=10, autoscale=1, nman=12, nexp=6. Each uint32 word encodes I, Q,
    and a shared signed exponent. Returns (real, imag), each shape (n_words,).
    """
    raw = np.asarray(words, dtype=np.uint32).reshape(-1)
    nfft = len(raw)
    if nfft == 0:
        raise ValueError("No CSI words supplied.")

    nman = 12
    nexp = 6
    nbits = 10
    iq_mask = (1 << (nman - 1)) - 1
    exponent_mask = (1 << nexp) - 1
    exponent_pivot = 1 << (nexp - 1)
    real_sign_mask = 1 << (nexp + 2 * nman - 1)
    imag_sign_mask = real_sign_mask >> nman
    exponent_zero = -nman

    real_mantissas = np.empty(nfft, dtype=np.int64)
    imag_mantissas = np.empty(nfft, dtype=np.int64)
    exponents = np.empty(nfft, dtype=np.int64)
    maxbit = -exponent_pivot

    for i, item in enumerate(raw):
        word = int(item)
        vi = (word >> (nexp + nman)) & iq_mask
        vq = (word >> nexp) & iq_mask
        exponent = word & exponent_mask
        if exponent >= exponent_pivot:
            exponent -= exponent_pivot << 1

        # The reference stores the original signed exponent in He before
        # adjusting a local copy for the global autoscaling calculation.
        exponents[i] = exponent

        # Match the reference decoder's autoscaling scan.
        x = vi | vq
        if x:
            m = 0xFFFF0000
            b = 0xFFFF
            shift = 16
            while shift > 0:
                if x & m:
                    exponent += shift
                    x >>= shift
                shift >>= 1
                m = (m >> shift) & b
                b >>= shift
            if x:
                maxbit = max(maxbit, exponent)

        real_mantissas[i] = vi | (0x80000000 if word & real_sign_mask else 0)
        imag_mantissas[i] = vq | (0x80000000 if word & imag_sign_mask else 0)

    shift_all = nbits - maxbit

    def scale_component(encoded: int, exponent: int) -> int:
        negative = bool(encoded & 0x80000000)
        magnitude = encoded & 0x7FFFFFFF
        e = exponent + shift_all
        if e < exponent_zero:
            value = 0
        elif e < 0:
            value = magnitude >> (-e)
        else:
            value = magnitude << e
        return -value if negative else value

    real = np.empty(nfft, dtype=np.int32)
    imag = np.empty(nfft, dtype=np.int32)
    for i in range(nfft):
        real[i] = scale_component(int(real_mantissas[i]), int(exponents[i]))
        imag[i] = scale_component(int(imag_mantissas[i]), int(exponents[i]))
    return real, imag


def decode_csi_body(csi_body: bytes, expected_subcarriers: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a 1024-byte packed CSI body to (real, imag, amplitude)."""
    expected_bytes = expected_subcarriers * 4
    if len(csi_body) != expected_bytes:
        raise ValueError(
            f"Expected {expected_bytes} CSI-body bytes for {expected_subcarriers} "
            f"subcarriers, received {len(csi_body)}."
        )
    words = np.frombuffer(csi_body, dtype="<u4")
    real, imag = decode_bcm4366c0_words(words)
    amplitude = np.hypot(real.astype(np.float64), imag.astype(np.float64)).astype(np.float32)
    return real, imag, amplitude


def list_pcap_members(zip_path):
    """List actual PCAP captures, excluding macOS metadata files."""
    with zipfile.ZipFile(zip_path, "r") as archive:
        members = []

        for item in archive.infolist():
            if item.is_dir():
                continue

            name = item.filename.replace("\\", "/")
            path = PurePosixPath(name)

            # Exclude files inside any __MACOSX directory.
            if "__MACOSX" in path.parts:
                continue

            # Exclude macOS AppleDouble sidecar files such as ._fall10.pcap.
            if path.name.startswith("._"):
                continue

            if path.suffix.lower() == ".pcap":
                members.append(name)

    return sorted(members)


def iter_room_a_csi(
    zip_path: str,
    member_path: str,
    *,
    expected_subcarriers: int = 256,
    udp_port: int | None = 5500,
    keep_zero_observations: bool = False,
) -> Iterator[CSIObservation]:
    """Yield decoded CSI observations from one PCAP member in a ZIP archive.

    `member_path` must be the exact member path returned by `list_pcap_members`.
    Packet timestamps are retained as absolute PCAP seconds; subtract the first
    retained timestamp to obtain capture-relative time.
    """
    with zipfile.ZipFile(zip_path, "r") as archive:
        pcap_bytes = archive.read(member_path)

    for timestamp, frame, packet_index in _iter_classic_pcap_frames(pcap_bytes):
        parsed = _parse_ethernet_ipv4_udp(frame)
        if parsed is None:
            continue
        payload, src_mac, _dst_mac, source_port, dest_port = parsed
        if udp_port is not None and source_port != udp_port and dest_port != udp_port:
            continue

        # Legacy Nexmon payload observed in Experiment 2:
        # 4-byte magic + source MAC + sequence + core/stream + chanspec + chip field.
        if len(payload) < 18 or payload[:4] != b"\x11\x11\x11\x11":
            continue
        csi_body = payload[18:]
        if len(csi_body) != expected_subcarriers * 4:
            continue

        real, imag, amplitude = decode_csi_body(csi_body, expected_subcarriers)
        if not keep_zero_observations and not np.any(amplitude):
            continue

        sequence = int.from_bytes(payload[10:12], "little")
        core_stream = int.from_bytes(payload[12:14], "little")
        chanspec = int.from_bytes(payload[14:16], "little")
        chip_field = int.from_bytes(payload[16:18], "little")
        yield CSIObservation(
            timestamp_seconds=timestamp,
            amplitude=amplitude,
            real=real,
            imag=imag,
            source_mac=":".join(f"{b:02x}" for b in payload[4:10]),
            sequence=sequence,
            core=core_stream & 0x7,
            stream=(core_stream >> 3) & 0x7,
            chanspec=chanspec,
            chip_field=chip_field,
            archive_member=member_path,
            packet_index=packet_index,
        )


def read_room_a_capture(
    zip_path: str,
    member_path: str,
    *,
    expected_subcarriers: int = 256,
    udp_port: int | None = 5500,
    keep_zero_observations: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Read one capture and return (relative_timestamps_seconds, amplitudes).

    Returned amplitudes have shape (number_of_valid_observations, 256).
    """
    observations = list(iter_room_a_csi(
        zip_path,
        member_path,
        expected_subcarriers=expected_subcarriers,
        udp_port=udp_port,
        keep_zero_observations=keep_zero_observations,
    ))
    if not observations:
        raise ValueError(
            f"No valid CSI observations decoded from {member_path!r}. "
            "Check the archive member path, packet format, and UDP port."
        )
    timestamps = np.asarray([o.timestamp_seconds for o in observations], dtype=np.float64)
    timestamps -= timestamps[0]
    amplitudes = np.stack([o.amplitude for o in observations]).astype(np.float32)
    return timestamps, amplitudes


def decode_bcm4366c0_batch(words: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized equivalent of `decode_bcm4366c0_words` for many observations.

    `words` has shape (n_observations, n_subcarriers) of uint32. Autoscaling is
    computed per observation, exactly as in the per-word decoder. Returns
    (real, imag) as int64 arrays of the same shape.
    """
    raw = np.asarray(words, dtype=np.uint32)
    if raw.ndim != 2 or raw.shape[1] == 0:
        raise ValueError(f"Expected a 2D (observations, words) array, got {raw.shape}.")

    nman, nexp, nbits = 12, 6, 10
    iq_mask = (1 << (nman - 1)) - 1
    exponent_pivot = 1 << (nexp - 1)
    real_sign_mask = 1 << (nexp + 2 * nman - 1)
    imag_sign_mask = real_sign_mask >> nman
    exponent_zero = -nman

    w = raw.astype(np.int64)
    vi = (w >> (nexp + nman)) & iq_mask
    vq = (w >> nexp) & iq_mask
    exponent = w & ((1 << nexp) - 1)
    exponent = np.where(exponent >= exponent_pivot, exponent - 2 * exponent_pivot, exponent)

    # Autoscaling: the highest set bit of (vi | vq) plus the word exponent,
    # maximised over the observation. frexp gives the bit length exactly.
    x = vi | vq
    _, bit_length = np.frexp(x.astype(np.float64))
    scaled_exponent = np.where(x > 0, exponent + bit_length - 1, np.iinfo(np.int64).min)
    maxbit = np.maximum(scaled_exponent.max(axis=1, keepdims=True), -exponent_pivot)
    shift = exponent + (nbits - maxbit)

    def scale(mantissa):
        left = mantissa << np.clip(shift, 0, 40)
        right = mantissa >> np.clip(-shift, 0, 40)
        value = np.where(shift >= 0, left, right)
        return np.where(shift < exponent_zero, 0, value)

    real = scale(vi)
    imag = scale(vq)
    real = np.where(w & real_sign_mask, -real, real)
    imag = np.where(w & imag_sign_mask, -imag, imag)
    return real, imag


@dataclass
class RoomACapture:
    """All well-formed CSI observations of one PCAP plus a rejection tally."""
    amplitude: np.ndarray        # (n, 256) float32
    timestamp_seconds: np.ndarray  # (n,) float64, relative to the first kept observation
    sequence: np.ndarray         # (n,) int64, 802.11 sequence-control field (0xFFFF = none)
    packet_index: np.ndarray     # (n,) int64, record index in the PCAP
    counts: dict


def read_room_a_capture_qc(
    pcap_bytes: bytes,
    *,
    expected_subcarriers: int = 256,
    udp_port: int | None = 5500,
    salvage_truncated: bool = True,
    drop_consecutive_duplicates: bool = True,
) -> RoomACapture:
    """Decode one PCAP and account for every packet that is not kept.

    Rejection reasons: not IPv4/UDP, wrong UDP port, missing Nexmon magic,
    wrong CSI length, all-zero CSI, or an exact repeat of the previous
    observation's raw CSI words. Temporal order is preserved.
    """
    counts = {
        "packets": 0, "not_udp": 0, "wrong_port": 0, "bad_magic": 0,
        "bad_length": 0, "zero_csi": 0, "duplicate": 0, "truncated": False,
    }
    rows, stamps, seqs, indices = [], [], [], []
    expected_payload = 18 + 4 * expected_subcarriers

    frames = _iter_classic_pcap_frames(pcap_bytes)
    while True:
        try:
            timestamp, frame, packet_index = next(frames)
        except StopIteration:
            break
        except TruncatedCaptureError:
            if not salvage_truncated:
                raise
            counts["truncated"] = True
            break
        counts["packets"] += 1
        parsed = _parse_ethernet_ipv4_udp(frame)
        if parsed is None:
            counts["not_udp"] += 1
            continue
        payload, _src, _dst, source_port, dest_port = parsed
        if udp_port is not None and udp_port not in (source_port, dest_port):
            counts["wrong_port"] += 1
            continue
        if payload[:4] != b"\x11\x11\x11\x11":
            counts["bad_magic"] += 1
            continue
        if len(payload) != expected_payload:
            counts["bad_length"] += 1
            continue
        words = np.frombuffer(payload[18:], dtype="<u4")
        if not words.any():
            counts["zero_csi"] += 1
            continue
        if drop_consecutive_duplicates and rows and np.array_equal(words, rows[-1]):
            counts["duplicate"] += 1
            continue
        rows.append(words)
        stamps.append(timestamp)
        seqs.append(int.from_bytes(payload[10:12], "little"))
        indices.append(packet_index)

    if rows:
        real, imag = decode_bcm4366c0_batch(np.stack(rows))
        amplitude = np.hypot(real.astype(np.float64), imag.astype(np.float64)).astype(np.float32)
        stamps = np.asarray(stamps, dtype=np.float64)
        stamps -= stamps[0]
    else:
        amplitude = np.zeros((0, expected_subcarriers), dtype=np.float32)
        stamps = np.zeros(0, dtype=np.float64)
    counts["kept"] = len(rows)
    return RoomACapture(
        amplitude=amplitude,
        timestamp_seconds=stamps,
        sequence=np.asarray(seqs, dtype=np.int64),
        packet_index=np.asarray(indices, dtype=np.int64),
        counts=counts,
    )
