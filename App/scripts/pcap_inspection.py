
from pathlib import Path
from zipfile import ZipFile
import struct


def inspect_pcap_header(zip_path, activity="fall", packets_to_show=3):
    """
    Inspect the header and first few packet records of a classic PCAP file
    stored inside a ZIP archive. Does not extract the file.
    """
    zip_path = Path(zip_path)

    with ZipFile(zip_path) as archive:
        candidates = []

        for item in archive.infolist():
            if item.is_dir():
                continue

            path = item.filename.replace("\\", "/")
            parts = path.lower().split("/")

            if (
                path.lower().endswith(".pcap")
                and "__macosx" not in parts
                and "csidata" in parts
                and activity.lower() in parts
            ):
                candidates.append(path)

        candidates.sort(key=str.lower)

        if not candidates:
            raise FileNotFoundError(
                f"No PCAP file found for activity '{activity}'"
            )

        member_path = candidates[0]

        with archive.open(member_path) as f:
            header = f.read(24)

            if len(header) < 24:
                raise ValueError("File is too short for a PCAP global header.")

            magic = header[:4]

            formats = {
                b"\xd4\xc3\xb2\xa1": ("<", "microsecond"),
                b"\xa1\xb2\xc3\xd4": (">", "microsecond"),
                b"\x4d\x3c\xb2\xa1": ("<", "nanosecond"),
                b"\xa1\xb2\x3c\x4d": (">", "nanosecond"),
            }

            print("Capture:", member_path)
            print("First 24 bytes:", header.hex(" "))

            if magic == b"\x0a\x0d\x0d\x0a":
                print("Format: PCAPNG")
                print(
                    "This is a PCAPNG file; "
                    "we need a compatible parser to inspect packet records."
                )
                return {
                    "path": member_path,
                    "format": "pcapng",
                }

            if magic not in formats:
                print("Format: Unknown")
                print("Magic bytes:", magic.hex())
                return {
                    "path": member_path,
                    "format": "unknown",
                }

            endian, timestamp_precision = formats[magic]

            (
                major,
                minor,
                thiszone,
                sigfigs,
                snaplen,
                network,
            ) = struct.unpack(endian + "HHiIII", header[4:24])

            print("Format: Classic PCAP")
            print(f"PCAP version: {major}.{minor}")
            print("Byte order:", "little-endian" if endian == "<" else "big-endian")
            print("Timestamp precision:", timestamp_precision)
            print("Snapshot length:", snaplen)
            print("Link-layer type:", network)

            records = []

            for index in range(packets_to_show):
                packet_header = f.read(16)

                if len(packet_header) == 0:
                    break

                if len(packet_header) < 16:
                    print("Incomplete packet record header.")
                    break

                ts_sec, ts_fraction, incl_len, orig_len = struct.unpack(
                    endian + "IIII", packet_header
                )

                packet_data = f.read(incl_len)

                if len(packet_data) < incl_len:
                    print("Incomplete packet data.")
                    break

                record = {
                    "packet_index": index,
                    "timestamp_seconds": ts_sec,
                    "timestamp_fraction": ts_fraction,
                    "captured_length": incl_len,
                    "original_length": orig_len,
                    "first_bytes": packet_data[:24].hex(" "),
                }
                records.append(record)

            print(f"\nFirst {len(records)} packet record(s):")
            for record in records:
                print(record)

            return {
                "path": member_path,
                "format": "pcap",
                "version": f"{major}.{minor}",
                "byte_order": endian,
                "timestamp_precision": timestamp_precision,
                "link_type": network,
                "records": records,
            }


def inspect_first_packet_payload(zip_path, activity="fall", payload_bytes=128):
    """Inspect Ethernet, IPv4, UDP, and payload bytes of one PCAP packet."""
    from pathlib import Path
    from zipfile import ZipFile
    import struct

    zip_path = Path(zip_path)

    with ZipFile(zip_path) as archive:
        candidates = []

        for item in archive.infolist():
            if item.is_dir():
                continue

            path = item.filename.replace("\\", "/")
            parts = path.lower().split("/")

            if (
                path.lower().endswith(".pcap")
                and "__macosx" not in parts
                and "csidata" in parts
                and activity.lower() in parts
            ):
                candidates.append(path)

        if not candidates:
            raise FileNotFoundError(
                f"No PCAP found for activity: {activity}"
            )

        member_path = sorted(candidates, key=str.lower)[0]

        with archive.open(member_path) as f:
            global_header = f.read(24)
            if len(global_header) != 24:
                raise ValueError("Incomplete PCAP global header.")

            magic = global_header[:4]
            endian_map = {
                b"\xd4\xc3\xb2\xa1": "<",
                b"\xa1\xb2\xc3\xd4": ">",
                b"\x4d\x3c\xb2\xa1": "<",
                b"\xa1\xb2\x3c\x4d": ">",
            }

            if magic not in endian_map:
                raise ValueError(
                    "This helper currently supports classic PCAP only."
                )

            endian = endian_map[magic]
            packet_header = f.read(16)

            if len(packet_header) != 16:
                raise ValueError("No complete packet record found.")

            ts_sec, ts_frac, captured_len, original_len = struct.unpack(
                endian + "IIII", packet_header
            )
            packet = f.read(captured_len)

    if len(packet) < 14:
        raise ValueError("Packet is shorter than an Ethernet header.")

    dst_mac = ":".join(f"{b:02x}" for b in packet[0:6])
    src_mac = ":".join(f"{b:02x}" for b in packet[6:12])
    ethertype = struct.unpack("!H", packet[12:14])[0]

    print("Capture:", member_path)
    print("Captured packet bytes:", len(packet))
    print("Destination MAC:", dst_mac)
    print("Source MAC:", src_mac)
    print(f"EtherType: 0x{ethertype:04x}")

    if ethertype != 0x0800:
        print("Not a standard Ethernet IPv4 frame; stopping layer parsing.")
        return

    ip_start = 14
    if len(packet) < ip_start + 20:
        raise ValueError("Packet is too short for a minimum IPv4 header.")

    version_ihl = packet[ip_start]
    version = version_ihl >> 4
    ihl = (version_ihl & 0x0F) * 4

    if version != 4 or ihl < 20:
        raise ValueError("Invalid IPv4 version or header length.")

    ip_total_length = struct.unpack(
        "!H", packet[ip_start + 2:ip_start + 4]
    )[0]
    protocol = packet[ip_start + 9]
    src_ip = ".".join(str(b) for b in packet[ip_start + 12:ip_start + 16])
    dst_ip = ".".join(str(b) for b in packet[ip_start + 16:ip_start + 20])

    print("IPv4 source:", src_ip)
    print("IPv4 destination:", dst_ip)
    print("IPv4 header length:", ihl)
    print("IPv4 total length:", ip_total_length)
    print("IP protocol:", protocol, "(17 means UDP)")

    if protocol != 17:
        print("Not UDP; stopping payload parsing.")
        return

    udp_start = ip_start + ihl
    if len(packet) < udp_start + 8:
        raise ValueError("Packet is too short for a UDP header.")

    src_port, dst_port, udp_length, checksum = struct.unpack(
        "!HHHH", packet[udp_start:udp_start + 8]
    )

    payload_start = udp_start + 8
    payload_end = min(
        len(packet),
        udp_start + max(udp_length, 8),
    )
    payload = packet[payload_start:payload_end]

    print("UDP source port:", src_port)
    print("UDP destination port:", dst_port)
    print("UDP length:", udp_length)
    print("UDP payload length:", len(payload))
    print(f"\nFirst {payload_bytes} payload bytes (hex):")
    print(payload[:payload_bytes].hex(" "))

    print("\nASCII preview (non-printable bytes shown as dots):")
    print("".join(chr(b) if 32 <= b < 127 else "." for b in payload[:payload_bytes]))

    return {
        "path": member_path,
        "ethertype": ethertype,
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "protocol": protocol,
        "src_port": src_port,
        "dst_port": dst_port,
        "udp_length": udp_length,
        "payload": payload,
    }


def inspect_packet_sequence(zip_path, activity="fall", max_packets=10):
    """Compare the timestamps, lengths, and payload prefixes of PCAP packets."""
    from pathlib import Path
    from zipfile import ZipFile
    import struct

    zip_path = Path(zip_path)

    with ZipFile(zip_path) as archive:
        candidates = []

        for item in archive.infolist():
            if item.is_dir():
                continue

            path = item.filename.replace("\\", "/")
            parts = path.lower().split("/")

            if (
                path.lower().endswith(".pcap")
                and "__macosx" not in parts
                and "csidata" in parts
                and activity.lower() in parts
            ):
                candidates.append(path)

        if not candidates:
            raise FileNotFoundError(
                f"No PCAP found for activity '{activity}'"
            )

        member_path = sorted(candidates, key=str.lower)[0]

        with archive.open(member_path) as f:
            global_header = f.read(24)

            if len(global_header) != 24:
                raise ValueError("Incomplete PCAP global header.")

            magic = global_header[:4]
            formats = {
                b"\xd4\xc3\xb2\xa1": ("<", 1_000_000),
                b"\xa1\xb2\xc3\xd4": (">", 1_000_000),
                b"\x4d\x3c\xb2\xa1": ("<", 1_000_000_000),
                b"\xa1\xb2\x3c\x4d": (">", 1_000_000_000),
            }

            if magic not in formats:
                raise ValueError("Unsupported classic PCAP byte order or format.")

            endian, timestamp_scale = formats[magic]
            records = []

            for index in range(max_packets):
                packet_header = f.read(16)

                if not packet_header:
                    break
                if len(packet_header) != 16:
                    raise ValueError("Incomplete packet record header.")

                sec, fraction, captured_len, original_len = struct.unpack(
                    endian + "IIII", packet_header
                )
                packet = f.read(captured_len)

                if len(packet) != captured_len:
                    raise ValueError("Incomplete packet data.")

                # Ethernet header
                if len(packet) < 14:
                    continue

                ethertype = struct.unpack("!H", packet[12:14])[0]
                if ethertype != 0x0800 or len(packet) < 34:
                    continue

                ip_start = 14
                version_ihl = packet[ip_start]
                ip_version = version_ihl >> 4
                ip_header_len = (version_ihl & 0x0F) * 4

                if ip_version != 4 or ip_header_len < 20:
                    continue

                if packet[ip_start + 9] != 17:  # UDP
                    continue

                udp_start = ip_start + ip_header_len
                if len(packet) < udp_start + 8:
                    continue

                src_port, dst_port, udp_len, _ = struct.unpack(
                    "!HHHH", packet[udp_start:udp_start + 8]
                )

                if udp_len < 8:
                    continue

                payload_start = udp_start + 8
                payload_end = min(len(packet), udp_start + udp_len)
                payload = packet[payload_start:payload_end]

                timestamp = sec + fraction / timestamp_scale

                records.append({
                    "index": index,
                    "timestamp": timestamp,
                    "captured_len": captured_len,
                    "payload_len": len(payload),
                    "src_port": src_port,
                    "dst_port": dst_port,
                    "payload_prefix": payload[:32].hex(" "),
                    "payload": payload,
                })

    print("Capture:", member_path)
    print("UDP packets inspected:", len(records))

    if not records:
        print("No IPv4/UDP packets found in the inspected records.")
        return records

    first_time = records[0]["timestamp"]

    for record in records:
        delta_ms = (record["timestamp"] - first_time) * 1000
        print(
            f"\nPacket {record['index']}: "
            f"delta={delta_ms:.3f} ms, "
            f"captured={record['captured_len']} bytes, "
            f"payload={record['payload_len']} bytes, "
            f"ports={record['src_port']}->{record['dst_port']}"
        )
        print("  Payload prefix:", record["payload_prefix"])

    print("\nDistinct payload lengths:", sorted({
        record["payload_len"] for record in records
    }))
    print("Distinct first 4 payload bytes:", sorted({
        record["payload"][:4].hex(" ")
        for record in records
    }))

    return records
