"""
flow_extractor.py
=================
AI-Powered Network Traffic Analyzer — FLOW AGGREGATION

Converts individual Scapy packets into NF-UNSW-NB15-v3-compatible flow records.

A flow is identified by the 5-tuple:
    (src_ip, dst_ip, src_port, dst_port, protocol)

Flows are exported when:
  - A TCP FIN or RST flag is seen
  - The flow has been idle for IDLE_TIMEOUT_S seconds (no new packet)
  - The flow has run for longer than ACTIVE_TIMEOUT_S seconds total

Exported flows contain all 49 features defined in feature_schema.FEATURE_COLS,
used by the trained RF, XGBoost, and AE models.
"""

import logging
import time
import math
import struct
from typing import Dict, List, Optional, Tuple

from feature_schema import FEATURE_COLS  # noqa: F401  (re-exported for callers)

logger = logging.getLogger(__name__)

# ── Timeouts (reduced so live flows appear within seconds) ───────────────────
# A flow is exported as soon as it has been idle for this long.
# 10 s is enough to capture short HTTP/DNS/QUIC flows while not waiting forever.
IDLE_TIMEOUT_S   = 10.0   # export if no packet received for this many seconds
ACTIVE_TIMEOUT_S = 60.0   # always export long-running flows after this long

# ── L7 protocol inference from port number (best-effort) ─────────────────────
_PORT_TO_L7: Dict[int, int] = {
    20:   6,    # FTP-DATA
    21:   6,    # FTP
    22:   92,   # SSH
    23:   23,   # Telnet
    25:   18,   # SMTP
    53:   5,    # DNS
    67:   257,  # DHCP
    68:   257,  # DHCP
    80:   7,    # HTTP
    110:  110,  # POP3
    119:  144,  # NNTP
    123:  123,  # NTP
    143:  143,  # IMAP
    161:  161,  # SNMP
    194:  194,  # IRC
    389:  389,  # LDAP
    443:  91,   # SSL/TLS
    445:  91,   # SMB
    465:  91,   # SMTPS
    514:  514,  # Syslog
    587:  587,  # SMTP submission
    993:  993,  # IMAPS
    995:  995,  # POP3S
    1194: 1194, # OpenVPN
    1433: 1433, # MSSQL
    3306: 3306, # MySQL
    3389: 81,   # RDP
    5060: 81,   # SIP
    5900: 81,   # VNC
    6881: 6881, # BitTorrent
    8080: 7,    # HTTP-Alt
    8443: 91,   # HTTPS-Alt
}

# ── TCP flag bit constants ────────────────────────────────────────────────────
TCP_FIN = 0x01
TCP_SYN = 0x02
TCP_RST = 0x04
TCP_PSH = 0x08
TCP_ACK = 0x10
TCP_URG = 0x20
TCP_ECE = 0x40
TCP_CWR = 0x80


class _FlowRecord:
    """Internal mutable accumulator for one flow."""

    __slots__ = [
        # identity
        "src_ip", "dst_ip", "src_port", "dst_port", "protocol",
        # timing
        "t_first", "t_last",
        "t_first_in", "t_last_in",
        "t_first_out", "t_last_out",
        # counters
        "in_bytes", "in_pkts",
        "out_bytes", "out_pkts",
        # TCP flags
        "tcp_flags_all",
        "client_tcp_flags",
        "server_tcp_flags",
        # TTL
        "min_ttl", "max_ttl",
        # packet lengths
        "pkt_lengths",
        # retransmission tracking
        "seen_seq_in", "seen_seq_out",
        "retrans_in_bytes", "retrans_in_pkts",
        "retrans_out_bytes", "retrans_out_pkts",
        # TCP window
        "tcp_win_max_in", "tcp_win_max_out",
        # ICMP
        "icmp_type",
        # DNS  (first DNS packet only)
        "dns_query_id", "dns_query_type", "dns_ttl_answer",
        # FTP  (first FTP response only)
        "ftp_ret_code",
        # inter-arrival times  (lists of floats in seconds)
        "iat_in",    # src→dst IAT list
        "iat_out",   # dst→src IAT list
        # packet size buckets
        "npkts_0_128",
        "npkts_128_256",
        "npkts_256_512",
        "npkts_512_1024",
        "npkts_1024_1514",
    ]

    def __init__(self, src_ip, dst_ip, src_port, dst_port, protocol, now):
        self.src_ip   = src_ip
        self.dst_ip   = dst_ip
        self.src_port = src_port
        self.dst_port = dst_port
        self.protocol = protocol

        self.t_first = now
        self.t_last  = now
        self.t_first_in  = None
        self.t_last_in   = None
        self.t_first_out = None
        self.t_last_out  = None

        self.in_bytes  = 0
        self.in_pkts   = 0
        self.out_bytes = 0
        self.out_pkts  = 0

        self.tcp_flags_all    = 0
        self.client_tcp_flags = 0
        self.server_tcp_flags = 0

        self.min_ttl = 255
        self.max_ttl = 0

        self.pkt_lengths = []

        self.seen_seq_in   = set()
        self.seen_seq_out  = set()
        self.retrans_in_bytes  = 0
        self.retrans_in_pkts   = 0
        self.retrans_out_bytes = 0
        self.retrans_out_pkts  = 0

        self.tcp_win_max_in  = 0
        self.tcp_win_max_out = 0

        self.icmp_type = 0

        self.dns_query_id   = 0
        self.dns_query_type = 0
        self.dns_ttl_answer = 0

        self.ftp_ret_code = 0

        self.iat_in  = []
        self.iat_out = []

        self.npkts_0_128    = 0
        self.npkts_128_256  = 0
        self.npkts_256_512  = 0
        self.npkts_512_1024 = 0
        self.npkts_1024_1514 = 0


def _safe_std(lst: List[float]) -> float:
    if len(lst) < 2:
        return 0.0
    mean = sum(lst) / len(lst)
    variance = sum((x - mean) ** 2 for x in lst) / len(lst)
    return math.sqrt(variance)


def _parse_dns(payload: bytes) -> Tuple[int, int, int]:
    """
    Parse a raw DNS message and return (query_id, query_type, answer_ttl).
    Returns (0, 0, 0) on any parse error.
    """
    try:
        if len(payload) < 12:
            return 0, 0, 0
        txid    = struct.unpack("!H", payload[0:2])[0]
        qdcount = struct.unpack("!H", payload[4:6])[0]
        ancount = struct.unpack("!H", payload[6:8])[0]

        # Walk past the question section
        offset = 12
        qtype  = 0
        for _ in range(qdcount):
            while offset < len(payload) and payload[offset] != 0:
                label_len = payload[offset]
                if label_len & 0xC0 == 0xC0:   # pointer
                    offset += 2
                    break
                offset += 1 + label_len
            else:
                offset += 1   # null terminator
            if offset + 4 <= len(payload):
                qtype = struct.unpack("!H", payload[offset: offset + 2])[0]
                offset += 4   # skip qtype + qclass

        # Read TTL from first answer record
        ans_ttl = 0
        for _ in range(ancount):
            if offset >= len(payload):
                break
            if payload[offset] & 0xC0 == 0xC0:
                offset += 2
            else:
                while offset < len(payload) and payload[offset] != 0:
                    offset += 1 + payload[offset]
                offset += 1
            if offset + 10 > len(payload):
                break
            ans_ttl = struct.unpack("!I", payload[offset + 4: offset + 8])[0]
            break

        return txid, qtype, ans_ttl
    except Exception:
        return 0, 0, 0


def _parse_ftp_code(payload: bytes) -> int:
    """
    Return the 3-digit FTP response code from a raw TCP payload, or 0.
    """
    try:
        text = payload[:8].decode("ascii", errors="ignore").strip()
        if len(text) >= 3 and text[:3].isdigit():
            return int(text[:3])
    except Exception:
        pass
    return 0


def _infer_l7(src_port: int, dst_port: int, protocol: int) -> int:
    """Infer L7 protocol number from ports (best-effort)."""
    if protocol == 1:     # ICMP
        return 1
    for port in (dst_port, src_port):
        if port in _PORT_TO_L7:
            return _PORT_TO_L7[port]
    return 0


FlowKey = Tuple[str, str, int, int, int]


class FlowExtractor:
    """
    Flow accumulator. Call add_packet() from the sniff thread.
    Call flush_expired() periodically (from a dedicated timer thread) to get
    completed flows regardless of whether new packets arrive.
    """

    def __init__(
        self,
        idle_timeout_s: float = IDLE_TIMEOUT_S,
        active_timeout_s: float = ACTIVE_TIMEOUT_S,
    ):
        self._flows: Dict[FlowKey, _FlowRecord] = {}
        self._idle_t   = idle_timeout_s
        self._active_t = active_timeout_s

        # ── Diagnostic counters (read by live_capture for logging) ───────────
        self.ipv4_packets_seen = 0   # IPv4 packets successfully dispatched
        self.pkt_errors        = 0   # exceptions inside _process()

    # ── public API ────────────────────────────────────────────────────────────

    def active_flow_count(self) -> int:
        return len(self._flows)

    def add_packet(self, pkt) -> Optional[dict]:
        """
        Accept a Scapy packet.
        Returns a completed-flow feature dict only when a TCP FIN/RST closes a
        flow; returns None for every other packet.
        Any internal exception is logged (not silently swallowed).
        """
        try:
            return self._process(pkt)
        except Exception as exc:
            self.pkt_errors += 1
            # Log at DEBUG to avoid flooding; first 5 at WARNING so user notices.
            if self.pkt_errors <= 5:
                logger.warning("FlowExtractor._process error (#%d): %s",
                               self.pkt_errors, exc)
            else:
                logger.debug("FlowExtractor._process error: %s", exc)
            return None

    def flush_expired(self) -> List[dict]:
        """
        Export and remove all flows that have been idle or active too long.
        Must be called from a dedicated timer thread every few seconds so that
        flows are finalized even when no new packets arrive for them.
        """
        now     = time.time()
        expired = []
        for key in list(self._flows):
            flow   = self._flows[key]
            idle   = now - flow.t_last
            active = now - flow.t_first
            if idle >= self._idle_t or active >= self._active_t:
                rec = self._export(flow)
                if rec is not None:
                    expired.append(rec)
                del self._flows[key]
        return expired

    # ── internal ──────────────────────────────────────────────────────────────

    def _get_or_create(self, key: FlowKey, now: float) -> _FlowRecord:
        if key not in self._flows:
            src_ip, dst_ip, src_port, dst_port, protocol = key
            self._flows[key] = _FlowRecord(
                src_ip, dst_ip, src_port, dst_port, protocol, now
            )
        return self._flows[key]

    def _process(self, pkt) -> Optional[dict]:
        # ── Import only the layers we actually use ────────────────────────────
        from scapy.layers.inet import IP, TCP, UDP, ICMP
        from scapy.layers.l2  import Ether

        now = time.time()

        # ── Safety net: re-parse generic Packet objects ───────────────────────
        # When Scapy emits "Unable to guess datalink type", every captured frame
        # is wrapped in the base Packet class instead of Ether. The raw bytes are
        # intact; we just need to force the correct dissection.
        #
        # Three attempts in order:
        #   1. Already decoded (Ether/IP/TCP etc.) — fast path
        #   2. Re-parse as Ethernet frame (linktype=1, which is always the case
        #      on Wi-Fi and Ethernet adapters captured via Npcap on Windows)
        #   3. Re-parse as raw IPv4 (for L3-only captures)

        if not pkt.haslayer(IP):
            raw = bytes(pkt)

            # Attempt 2: Ethernet
            if len(raw) >= 14:  # minimum Ethernet header size
                try:
                    repkt = Ether(raw)
                    if repkt.haslayer(IP):
                        pkt = repkt
                except Exception:
                    pass

            # Attempt 3: Raw IPv4 (skip Ethernet header)
            if not pkt.haslayer(IP) and len(raw) >= 20:
                try:
                    # Check IP version nibble = 4
                    if (raw[0] >> 4) == 4:
                        repkt = IP(raw)
                        if repkt.haslayer(IP):
                            pkt = repkt
                except Exception:
                    pass

        # ── Still no IP layer → not an IPv4 packet (ARP, IPv6, etc.) ─────────
        if not pkt.haslayer(IP):
            return None   # skip silently, not an error

        self.ipv4_packets_seen += 1


        ip     = pkt[IP]
        src_ip = ip.src
        dst_ip = ip.dst
        proto  = ip.proto
        ttl    = ip.ttl
        ip_len = len(ip)

        src_port      = 0
        dst_port      = 0
        tcp_flags     = 0
        tcp_win       = 0
        tcp_seq       = None
        payload_bytes = b""

        # ── TCP ──────────────────────────────────────────────────────────────
        if pkt.haslayer(TCP):
            tcp       = pkt[TCP]
            src_port  = int(tcp.sport)
            dst_port  = int(tcp.dport)
            tcp_flags = int(tcp.flags)
            tcp_win   = int(tcp.window)
            tcp_seq   = int(tcp.seq)
            if tcp.payload:
                payload_bytes = bytes(tcp.payload)

        # ── UDP ──────────────────────────────────────────────────────────────
        elif pkt.haslayer(UDP):
            udp      = pkt[UDP]
            src_port = int(udp.sport)
            dst_port = int(udp.dport)
            if udp.payload:
                payload_bytes = bytes(udp.payload)

        # ── ICMP ─────────────────────────────────────────────────────────────
        icmp_type_val = 0
        if pkt.haslayer(ICMP):
            icmp_type_val = int(pkt[ICMP].type)

        # ── Flow key: try forward then reverse for bidirectional tracking ─────
        fwd_key = (src_ip, dst_ip, src_port, dst_port, proto)
        rev_key = (dst_ip, src_ip, dst_port, src_port, proto)

        is_reverse = False
        if fwd_key in self._flows:
            key = fwd_key
        elif rev_key in self._flows:
            key = rev_key
            is_reverse = True
        else:
            key = fwd_key   # new flow — always starts as forward

        flow = self._get_or_create(key, now)
        flow.t_last = now

        # ── TTL ──────────────────────────────────────────────────────────────
        flow.min_ttl = min(flow.min_ttl, ttl)
        flow.max_ttl = max(flow.max_ttl, ttl)

        # ── Packet size buckets ──────────────────────────────────────────────
        flow.pkt_lengths.append(ip_len)
        if   ip_len <= 128:   flow.npkts_0_128    += 1
        elif ip_len <= 256:   flow.npkts_128_256  += 1
        elif ip_len <= 512:   flow.npkts_256_512  += 1
        elif ip_len <= 1024:  flow.npkts_512_1024 += 1
        else:                 flow.npkts_1024_1514 += 1

        # ── Directional counters ─────────────────────────────────────────────
        if not is_reverse:   # src → dst  (IN direction per NF-UNSW)
            flow.in_bytes += ip_len
            flow.in_pkts  += 1

            # IAT in
            if flow.t_last_in is not None:
                flow.iat_in.append(now - flow.t_last_in)
            else:
                flow.t_first_in = now
            flow.t_last_in = now

            # TCP flags (client)
            flow.tcp_flags_all    |= tcp_flags
            flow.client_tcp_flags |= tcp_flags

            # TCP window
            if tcp_win > flow.tcp_win_max_in:
                flow.tcp_win_max_in = tcp_win

            # Retransmissions (TCP only)
            if tcp_seq is not None:
                if tcp_seq in flow.seen_seq_in:
                    flow.retrans_in_bytes += ip_len
                    flow.retrans_in_pkts  += 1
                else:
                    flow.seen_seq_in.add(tcp_seq)

            # DNS parsing (UDP port 53, raw payload — no Scapy DNS layer needed)
            if proto == 17 and (dst_port == 53 or src_port == 53) \
                    and flow.dns_query_id == 0 and len(payload_bytes) >= 12:
                qid, qtype, ans_ttl = _parse_dns(payload_bytes)
                flow.dns_query_id   = qid
                flow.dns_query_type = qtype
                flow.dns_ttl_answer = ans_ttl

            # FTP parsing (TCP port 21 server response)
            if proto == 6 and src_port == 21 \
                    and flow.ftp_ret_code == 0 and payload_bytes:
                flow.ftp_ret_code = _parse_ftp_code(payload_bytes)

        else:                # dst → src  (OUT direction)
            flow.out_bytes += ip_len
            flow.out_pkts  += 1

            # IAT out
            if flow.t_last_out is not None:
                flow.iat_out.append(now - flow.t_last_out)
            else:
                flow.t_first_out = now
            flow.t_last_out = now

            # TCP flags (server)
            flow.tcp_flags_all    |= tcp_flags
            flow.server_tcp_flags |= tcp_flags

            # TCP window
            if tcp_win > flow.tcp_win_max_out:
                flow.tcp_win_max_out = tcp_win

            # Retransmissions
            if tcp_seq is not None:
                if tcp_seq in flow.seen_seq_out:
                    flow.retrans_out_bytes += ip_len
                    flow.retrans_out_pkts  += 1
                else:
                    flow.seen_seq_out.add(tcp_seq)

            # DNS answer from server
            if proto == 17 and (dst_port == 53 or src_port == 53) \
                    and flow.dns_ttl_answer == 0 and len(payload_bytes) >= 12:
                _, _, ans_ttl = _parse_dns(payload_bytes)
                if ans_ttl:
                    flow.dns_ttl_answer = ans_ttl

            # FTP from server (reverse direction)
            if proto == 6 and dst_port == 21 \
                    and flow.ftp_ret_code == 0 and payload_bytes:
                flow.ftp_ret_code = _parse_ftp_code(payload_bytes)

        # ── ICMP type ────────────────────────────────────────────────────────
        if icmp_type_val:
            flow.icmp_type = icmp_type_val

        # ── Close on TCP FIN or RST ───────────────────────────────────────────
        if proto == 6 and (tcp_flags & (TCP_FIN | TCP_RST)):
            rec = self._export(flow)
            del self._flows[key]
            return rec

        return None

    def _export(self, flow: _FlowRecord) -> Optional[dict]:
        """
        Convert accumulated flow state into the 49-feature dict expected by
        the trained models, plus dashboard metadata fields.
        Returns None if the flow has no packets at all.
        """
        total_pkts = flow.in_pkts + flow.out_pkts
        if total_pkts < 1:
            return None

        dur_ms = max(1.0, (flow.t_last - flow.t_first) * 1000.0)
        dur_s  = dur_ms / 1000.0

        dur_in_ms  = 0.0
        dur_out_ms = 0.0
        if flow.t_first_in is not None and flow.t_last_in is not None:
            dur_in_ms  = max(0.0, (flow.t_last_in  - flow.t_first_in)  * 1000.0)
        if flow.t_first_out is not None and flow.t_last_out is not None:
            dur_out_ms = max(0.0, (flow.t_last_out - flow.t_first_out) * 1000.0)

        # Byte rates (bytes/s)
        src_to_dst_bytes_per_s = flow.in_bytes  / dur_s
        dst_to_src_bytes_per_s = flow.out_bytes / dur_s

        # Throughput (bits/s)
        dur_in_s  = max(0.001, dur_in_ms  / 1000.0)
        dur_out_s = max(0.001, dur_out_ms / 1000.0)
        src_avg_tp = (flow.in_bytes  * 8) / dur_in_s
        dst_avg_tp = (flow.out_bytes * 8) / dur_out_s

        # Packet lengths
        pkt_lens = flow.pkt_lengths
        longest  = max(pkt_lens) if pkt_lens else 0
        shortest = min(pkt_lens) if pkt_lens else 0

        # IAT stats helper
        def _iat_stats(lst):
            if not lst:
                return 0.0, 0.0, 0.0, 0.0
            mn  = min(lst)
            mx  = max(lst)
            avg = sum(lst) / len(lst)
            std = _safe_std(lst)
            return mn, mx, avg, std

        iat_in_min,  iat_in_max,  iat_in_avg,  iat_in_std  = _iat_stats(flow.iat_in)
        iat_out_min, iat_out_max, iat_out_avg, iat_out_std = _iat_stats(flow.iat_out)

        # Convert IAT seconds → milliseconds (matching NF-UNSW units)
        def ms(v): return v * 1000.0

        l7 = _infer_l7(flow.src_port, flow.dst_port, flow.protocol)

        _PROTO_NAME = {6: "TCP", 17: "UDP", 1: "ICMP"}
        proto_name  = _PROTO_NAME.get(flow.protocol, str(flow.protocol))
        pkt_rate    = total_pkts / max(0.001, dur_s)

        features = {
            # ── 49 model features (exact names matching FEATURE_COLS) ─────────
            "L4_SRC_PORT"               : float(flow.src_port),
            "L4_DST_PORT"               : float(flow.dst_port),
            "PROTOCOL"                  : float(flow.protocol),
            "L7_PROTO"                  : float(l7),

            "IN_BYTES"                  : float(flow.in_bytes),
            "IN_PKTS"                   : float(flow.in_pkts),
            "OUT_BYTES"                 : float(flow.out_bytes),
            "OUT_PKTS"                  : float(flow.out_pkts),

            "TCP_FLAGS"                 : float(flow.tcp_flags_all),
            "CLIENT_TCP_FLAGS"          : float(flow.client_tcp_flags),
            "SERVER_TCP_FLAGS"          : float(flow.server_tcp_flags),

            "FLOW_DURATION_MILLISECONDS": float(dur_ms),
            "DURATION_IN"               : float(dur_in_ms),
            "DURATION_OUT"              : float(dur_out_ms),

            "MIN_TTL"                   : float(flow.min_ttl),
            "MAX_TTL"                   : float(flow.max_ttl),

            "LONGEST_FLOW_PKT"          : float(longest),
            "SHORTEST_FLOW_PKT"         : float(shortest),
            "MIN_IP_PKT_LEN"            : float(shortest),
            "MAX_IP_PKT_LEN"            : float(longest),

            "SRC_TO_DST_SECOND_BYTES"   : float(src_to_dst_bytes_per_s),
            "DST_TO_SRC_SECOND_BYTES"   : float(dst_to_src_bytes_per_s),

            "RETRANSMITTED_IN_BYTES"    : float(flow.retrans_in_bytes),
            "RETRANSMITTED_IN_PKTS"     : float(flow.retrans_in_pkts),
            "RETRANSMITTED_OUT_BYTES"   : float(flow.retrans_out_bytes),
            "RETRANSMITTED_OUT_PKTS"    : float(flow.retrans_out_pkts),

            "SRC_TO_DST_AVG_THROUGHPUT" : float(src_avg_tp),
            "DST_TO_SRC_AVG_THROUGHPUT" : float(dst_avg_tp),

            "NUM_PKTS_UP_TO_128_BYTES"  : float(flow.npkts_0_128),
            "NUM_PKTS_128_TO_256_BYTES" : float(flow.npkts_128_256),
            "NUM_PKTS_256_TO_512_BYTES" : float(flow.npkts_256_512),
            "NUM_PKTS_512_TO_1024_BYTES": float(flow.npkts_512_1024),
            "NUM_PKTS_1024_TO_1514_BYTES": float(flow.npkts_1024_1514),

            "TCP_WIN_MAX_IN"            : float(flow.tcp_win_max_in),
            "TCP_WIN_MAX_OUT"           : float(flow.tcp_win_max_out),

            "ICMP_TYPE"                 : float(flow.icmp_type),
            "ICMP_IPV4_TYPE"            : float(flow.icmp_type),

            "DNS_QUERY_ID"              : float(flow.dns_query_id),
            "DNS_QUERY_TYPE"            : float(flow.dns_query_type),
            "DNS_TTL_ANSWER"            : float(flow.dns_ttl_answer),

            "FTP_COMMAND_RET_CODE"      : float(flow.ftp_ret_code),

            "SRC_TO_DST_IAT_MIN"        : float(ms(iat_in_min)),
            "SRC_TO_DST_IAT_MAX"        : float(ms(iat_in_max)),
            "SRC_TO_DST_IAT_AVG"        : float(ms(iat_in_avg)),
            "SRC_TO_DST_IAT_STDDEV"     : float(ms(iat_in_std)),

            "DST_TO_SRC_IAT_MIN"        : float(ms(iat_out_min)),
            "DST_TO_SRC_IAT_MAX"        : float(ms(iat_out_max)),
            "DST_TO_SRC_IAT_AVG"        : float(ms(iat_out_avg)),
            "DST_TO_SRC_IAT_STDDEV"     : float(ms(iat_out_std)),

            # ── Dashboard metadata (not model inputs, prefixed _) ─────────────
            "_src_ip"    : flow.src_ip,
            "_dst_ip"    : flow.dst_ip,
            "_src_port"  : flow.src_port,
            "_dst_port"  : flow.dst_port,
            "_protocol"  : proto_name,
            "_pkt_count" : total_pkts,
            "_byte_count": flow.in_bytes + flow.out_bytes,
            "_duration"  : dur_s,
            "_pkt_rate"  : pkt_rate,
            "_timestamp" : flow.t_first,
        }

        return features
