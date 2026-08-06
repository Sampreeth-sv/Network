"""
live_capture.py
===============
AI-Powered Network Traffic Analyzer — LIVE PACKET CAPTURE

Runs Scapy packet sniffing in a capture thread.
A SEPARATE dedicated flush thread periodically exports idle/expired flows
every FLUSH_INTERVAL_S seconds — independently of whether new packets arrive.

Key fix: the interface is resolved to a Scapy NetworkInterface object so that
Scapy knows the correct link-layer type and uses Ether/IP/TCP dissection instead
of returning raw generic Packet objects (which caused IPv4=0).

Usage (from app.py):
    from live_capture import start_capture, live_flows, capture_status
    start_capture(iface="Wi-Fi")     # display name, NPF path, or None
"""

import logging
import threading
import time
from collections import deque
from typing import Optional

from flow_extractor import FlowExtractor
from inference import engine as _engine

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
MAX_FLOWS        = 10_000   # rolling window kept in memory
FLUSH_INTERVAL_S = 2.0      # flush expired flows every N seconds (dedicated thread)
LOG_INTERVAL_S   = 10.0     # print pipeline diagnostics every N seconds

# ── Shared state (read by Dash callbacks) ─────────────────────────────────────
live_flows = deque(maxlen=MAX_FLOWS)

capture_status = {
    "running"           : False,
    "iface"             : None,
    "packets"           : 0,      # total raw packets from Scapy
    "ipv4_packets"      : 0,      # IPv4 packets reaching flow extractor
    "active_flows"      : 0,      # flows currently open
    "flows_finalized"   : 0,      # flows exported (FIN/RST or timeout)
    "flows_to_inference": 0,      # flows sent to inference
    "flows"             : 0,      # flows written to live_flows deque
    "validate_failures" : 0,      # feature validation failures
    "errors"            : 0,
    "last_error"        : "",
}

_extractor: Optional[FlowExtractor] = None

_flows_lock  = threading.Lock()
_status_lock = threading.Lock()
_stop_event  = threading.Event()


# ── Interface resolution ───────────────────────────────────────────────────────

def _get_scapy_ifaces_list():
    """
    Return a list of (index, scapy_iface_object_or_name, display_description)
    using scapy.interfaces.IFACES (the authoritative Scapy interface registry).
    Falls back to get_windows_if_list() for older Scapy versions.
    """
    # ── Primary: scapy.interfaces.IFACES (Scapy 2.4.5+) ─────────────────────
    try:
        from scapy.interfaces import IFACES
        result = []
        for i, (guid, intf) in enumerate(IFACES.items()):
            # Human-readable label shown to the user
            desc = (
                getattr(intf, "description",    "") or
                getattr(intf, "network_name",   "") or
                str(guid)
            )
            result.append((i, intf, desc))
        if result:
            return result
    except Exception as exc:
        logger.debug("scapy.interfaces.IFACES unavailable: %s", exc)

    # ── Fallback: get_windows_if_list() ───────────────────────────────────────
    try:
        from scapy.arch.windows import get_windows_if_list
        ifaces = get_windows_if_list()
        result = []
        for i, iface in enumerate(ifaces):
            name = iface.get("name", "")
            desc = iface.get("description", "") or name
            result.append((i, name, desc))
        return result
    except Exception as exc:
        logger.debug("get_windows_if_list unavailable: %s", exc)

    # ── Fallback: generic IFACES dict ─────────────────────────────────────────
    try:
        import scapy.interfaces as _si
        result = []
        for i, (name, intf) in enumerate(_si.ifaces.items()):
            desc = getattr(intf, "description", name)
            result.append((i, intf, desc))
        return result
    except Exception as exc:
        logger.debug("scapy.interfaces.ifaces unavailable: %s", exc)

    return []


def list_interfaces() -> list:
    """
    Return (index, name_string, description) tuples for display.
    name_string is what the user should type / what we store internally.
    """
    raw = _get_scapy_ifaces_list()
    result = []
    for i, iface_obj, desc in raw:
        # If iface_obj is already a string (fallback path), use it directly.
        # Otherwise use str() — NetworkInterface has a useful __str__.
        name_str = iface_obj if isinstance(iface_obj, str) else str(iface_obj)
        result.append((i, name_str, desc))
    return result


def _resolve_scapy_iface(selected: Optional[str]):
    """
    Given the string name the user chose (display name, NPF path, or None),
    return the matching Scapy NetworkInterface object so sniff() can correctly
    determine the link-layer type (Ether vs raw, etc.).

    If resolution fails, returns the original string so sniff() still runs.
    Returns None if selected is None (system default interface).
    """
    if selected is None:
        return None

    raw = _get_scapy_ifaces_list()
    for _i, iface_obj, desc in raw:
        if isinstance(iface_obj, str):
            # Fallback path: iface_obj is already a plain string name
            if selected in (iface_obj, desc):
                return iface_obj
        else:
            # Primary path: iface_obj is a Scapy NetworkInterface
            intf_name    = getattr(iface_obj, "name",         "")
            intf_desc    = getattr(iface_obj, "description",  "")
            intf_network = getattr(iface_obj, "network_name", "")
            candidates   = {intf_name, intf_desc, intf_network, str(iface_obj), desc}
            if selected in candidates:
                logger.info(
                    "Interface '%s' resolved → Scapy object '%s' (NPF: %s)",
                    selected, intf_desc or intf_network, intf_name,
                )
                return iface_obj   # ← the NetworkInterface object

    logger.warning(
        "Could not resolve '%s' to a Scapy NetworkInterface. "
        "Falling back to string — packets may not decode correctly.",
        selected,
    )
    return selected   # fallback: pass the string name as-is


def select_interface_interactive() -> Optional[str]:
    """
    Print available interfaces and prompt the user to select one.
    Returns the selected interface name string (stored and later resolved).
    """
    ifaces = list_interfaces()
    if not ifaces:
        print(
            "[WARN] Could not enumerate interfaces. "
            "Set CAPTURE_IFACE at the top of app.py manually."
        )
        return None

    print("\n" + "=" * 60)
    print("Available network interfaces:")
    print("=" * 60)
    for idx, name, desc in ifaces:
        label = desc if desc else name
        print(f"  [{idx}]  {label}")
        if desc and name != desc:
            print(f"         ({name})")
    print("=" * 60)

    while True:
        try:
            choice = input(
                "Enter interface number (or press ENTER for auto-select): "
            ).strip()
            if choice == "":
                # Auto-select: prefer Wi-Fi or Ethernet by description
                for idx, name, desc in ifaces:
                    label = (desc or name).lower()
                    if any(k in label for k in ("wi-fi", "wifi", "wireless",
                                                "ethernet", "eth")):
                        print(f"Auto-selected: [{idx}] {desc or name}")
                        return name
                idx, name, desc = ifaces[0]
                print(f"Auto-selected first interface: [{idx}] {desc or name}")
                return name

            idx = int(choice)
            if 0 <= idx < len(ifaces):
                _, name, desc = ifaces[idx]
                print(f"Selected: {desc or name}")
                return name
            else:
                print(f"Please enter a number between 0 and {len(ifaces) - 1}.")
        except (ValueError, EOFError):
            print("Invalid input.")


# ── Inference + write to live_flows ───────────────────────────────────────────

def _process_flow(feature_dict: dict):
    """Run inference on one completed flow and append to live_flows."""
    with _status_lock:
        capture_status["flows_to_inference"] += 1

    if not _engine._loaded:
        with _status_lock:
            capture_status["validate_failures"] += 1
        logger.warning("InferenceEngine not loaded — dropping flow.")
        return

    if not _engine.validate_features(feature_dict):
        with _status_lock:
            capture_status["validate_failures"] += 1
        return

    try:
        result = _engine.run_inference(feature_dict)
    except Exception as exc:
        with _status_lock:
            capture_status["errors"]    += 1
            capture_status["last_error"] = f"inference: {exc}"
        logger.warning("Inference error: %s", exc)
        return

    with _flows_lock:
        live_flows.append(result)

    with _status_lock:
        capture_status["flows"] += 1

    logger.debug(
        "Flow → rf=%d prob=%.3f ae_mse=%.4f score=%.4f  %s:%s→%s:%s",
        result["rf_pred"], result["rf_prob"],
        result["ae_mse"], result["score_combined"],
        result["src_ip"], result["src_port"],
        result["dst_ip"], result["dst_port"],
    )


# ── FLUSH THREAD ──────────────────────────────────────────────────────────────

def _run_flush(extractor: FlowExtractor):
    """
    Runs every FLUSH_INTERVAL_S seconds and exports idle/expired flows,
    completely independently of whether new packets are arriving.
    """
    last_log = time.time()

    while not _stop_event.is_set():
        time.sleep(FLUSH_INTERVAL_S)
        if _stop_event.is_set():
            break

        try:
            expired = extractor.flush_expired()
        except Exception as exc:
            logger.warning("flush_expired() error: %s", exc)
            expired = []

        n = len(expired)
        if n:
            with _status_lock:
                capture_status["flows_finalized"] += n
                capture_status["active_flows"]     = extractor.active_flow_count()
            logger.info("Flushed %d expired flow(s). Active: %d",
                        n, extractor.active_flow_count())
            for flow_feat in expired:
                _process_flow(flow_feat)
        else:
            with _status_lock:
                capture_status["active_flows"] = extractor.active_flow_count()

        # Periodic pipeline log
        now = time.time()
        if now - last_log >= LOG_INTERVAL_S:
            last_log = now
            s = capture_status
            logger.info(
                "=== Pipeline ===  pkts:%d  ipv4:%d  active:%d  "
                "finalized:%d  →infer:%d  dashboard:%d  "
                "val_fail:%d  errors:%d  pkt_err:%d",
                s["packets"], s["ipv4_packets"], s["active_flows"],
                s["flows_finalized"], s["flows_to_inference"],
                s["flows"], s["validate_failures"], s["errors"],
                extractor.pkt_errors,
            )

    logger.info("Flush thread stopped.")


# ── CAPTURE THREAD ────────────────────────────────────────────────────────────

def _run_capture(scapy_iface, extractor: FlowExtractor):
    """
    Packet-capture thread.  Uses sniff() with the resolved Scapy
    NetworkInterface object so that packets arrive as Ether/IP/TCP instead of
    generic undissected Packet objects.
    """
    from scapy.sendrecv import sniff

    with _status_lock:
        capture_status["running"] = True
        iface_label = (
            getattr(scapy_iface, "description", None) or
            getattr(scapy_iface, "network_name", None) or
            str(scapy_iface) if scapy_iface else "default"
        )
        capture_status["iface"] = iface_label

    logger.info("Capture thread started on: %s  (object type: %s)",
                iface_label, type(scapy_iface).__name__)

    # ── Debug: log first 5 packets to confirm decoding ────────────────────────
    _debug_logged = [0]
    _DEBUG_LIMIT  = 5

    def _handle_packet(pkt):
        # Count every raw packet
        with _status_lock:
            capture_status["packets"] += 1

        # ── Debug: show first 5 raw packets ──────────────────────────────────
        if _debug_logged[0] < _DEBUG_LIMIT:
            _debug_logged[0] += 1
            try:
                layers = [type(l).__name__ for l in pkt.layers()] \
                         if hasattr(pkt, "layers") else ["?"]
            except Exception:
                layers = ["?"]
            logger.info(
                "DEBUG pkt #%d: cls=%s  layers=%s  summary=%s",
                _debug_logged[0],
                type(pkt).__name__,
                layers,
                pkt.summary() if hasattr(pkt, "summary") else "?",
            )
            if _debug_logged[0] == _DEBUG_LIMIT:
                logger.info("(First %d packets logged; further debug suppressed.)",
                            _DEBUG_LIMIT)

        # ── Dispatch to flow extractor ────────────────────────────────────────
        try:
            result = extractor.add_packet(pkt)
        except Exception as exc:
            with _status_lock:
                capture_status["errors"]    += 1
                capture_status["last_error"] = f"add_packet: {exc}"
            return

        # Update IPv4 counter from extractor's own counter
        with _status_lock:
            capture_status["ipv4_packets"] = extractor.ipv4_packets_seen

        # TCP FIN/RST closed a flow immediately
        if result is not None:
            with _status_lock:
                capture_status["flows_finalized"] += 1
            _process_flow(result)

    def _stop_filter(_):
        return _stop_event.is_set()

    sniff_kwargs: dict = dict(
        prn=_handle_packet,
        store=False,
        stop_filter=_stop_filter,
    )
    if scapy_iface is not None:
        sniff_kwargs["iface"] = scapy_iface

    try:
        sniff(**sniff_kwargs)
    except Exception as exc:
        with _status_lock:
            capture_status["errors"]    += 1
            capture_status["last_error"] = f"sniff: {exc}"
        logger.error("Capture thread error: %s", exc)
    finally:
        with _status_lock:
            capture_status["running"] = False
        logger.info("Capture thread stopped.")


# ── Public API ────────────────────────────────────────────────────────────────

_capture_thread: Optional[threading.Thread] = None
_flush_thread:   Optional[threading.Thread] = None


def start_capture(iface: Optional[str] = None):
    """
    Resolve the interface, start the flush thread, then start the capture thread.

    Parameters
    ----------
    iface : str or None
        Interface display name (e.g. 'Wi-Fi'), NPF path, or None for default.
        Resolved to a Scapy NetworkInterface object internally.
    """
    global _capture_thread, _flush_thread, _extractor

    _stop_event.clear()
    _extractor = FlowExtractor()

    # Resolve the string name to a Scapy NetworkInterface object.
    # This ensures Scapy knows the link-layer type and uses Ether dissection.
    scapy_iface = _resolve_scapy_iface(iface)

    # ── Flush thread ──────────────────────────────────────────────────────────
    _flush_thread = threading.Thread(
        target=_run_flush,
        args=(_extractor,),
        daemon=True,
        name="FlowFlushThread",
    )
    _flush_thread.start()
    logger.info("Flush thread launched (interval=%.1fs).", FLUSH_INTERVAL_S)

    # ── Capture thread ────────────────────────────────────────────────────────
    _capture_thread = threading.Thread(
        target=_run_capture,
        args=(scapy_iface, _extractor),
        daemon=True,
        name="LiveCaptureThread",
    )
    _capture_thread.start()
    logger.info("Capture thread launched.")


def stop_capture():
    """Signal both threads to stop cleanly."""
    _stop_event.set()
    for t in (_capture_thread, _flush_thread):
        if t and t.is_alive():
            t.join(timeout=10)
    logger.info("Capture and flush threads stopped.")


def get_flows_snapshot() -> list:
    """Return a thread-safe copy of the current live_flows list."""
    with _flows_lock:
        return list(live_flows)
