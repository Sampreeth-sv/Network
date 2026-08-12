"""
feature_schema.py
=================
Canonical 49-feature schema for the AI-Powered Network Intrusion Detection System.

This is the SINGLE source of truth for FEATURE_COLS.

All training scripts, inference engines, and the flow extractor import from here.
Do NOT define a second copy of this list elsewhere.

Feature names match the NF-UNSW-NB15-v3 dataset columns and the field names
produced by flow_extractor.py._export().
"""

FEATURE_COLS = [
    "L4_SRC_PORT",
    "L4_DST_PORT",
    "PROTOCOL",
    "L7_PROTO",

    "IN_BYTES",
    "IN_PKTS",
    "OUT_BYTES",
    "OUT_PKTS",

    "TCP_FLAGS",
    "CLIENT_TCP_FLAGS",
    "SERVER_TCP_FLAGS",

    "FLOW_DURATION_MILLISECONDS",
    "DURATION_IN",
    "DURATION_OUT",

    "MIN_TTL",
    "MAX_TTL",

    "LONGEST_FLOW_PKT",
    "SHORTEST_FLOW_PKT",
    "MIN_IP_PKT_LEN",
    "MAX_IP_PKT_LEN",

    "SRC_TO_DST_SECOND_BYTES",
    "DST_TO_SRC_SECOND_BYTES",

    "RETRANSMITTED_IN_BYTES",
    "RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_BYTES",
    "RETRANSMITTED_OUT_PKTS",

    "SRC_TO_DST_AVG_THROUGHPUT",
    "DST_TO_SRC_AVG_THROUGHPUT",

    "NUM_PKTS_UP_TO_128_BYTES",
    "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES",
    "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES",

    "TCP_WIN_MAX_IN",
    "TCP_WIN_MAX_OUT",

    "ICMP_TYPE",
    "ICMP_IPV4_TYPE",

    "DNS_QUERY_ID",
    "DNS_QUERY_TYPE",
    "DNS_TTL_ANSWER",

    "FTP_COMMAND_RET_CODE",

    "SRC_TO_DST_IAT_MIN",
    "SRC_TO_DST_IAT_MAX",
    "SRC_TO_DST_IAT_AVG",
    "SRC_TO_DST_IAT_STDDEV",

    "DST_TO_SRC_IAT_MIN",
    "DST_TO_SRC_IAT_MAX",
    "DST_TO_SRC_IAT_AVG",
    "DST_TO_SRC_IAT_STDDEV",
]

assert len(FEATURE_COLS) == 49, (
    f"FEATURE_COLS must have exactly 49 entries, got {len(FEATURE_COLS)}"
)
assert len(FEATURE_COLS) == len(set(FEATURE_COLS)), (
    "FEATURE_COLS contains duplicate names — check feature_schema.py"
)
