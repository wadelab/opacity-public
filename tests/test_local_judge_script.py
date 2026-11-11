"""scripts/local_judge.sh: its tunnel must bind, and it must not report success on unjudged logs.

On 7 November 2026 the random local port it picked (19649) was held by another
process on d2-1. The tunnel never bound, the reachability check failed, nothing was
judged, and the script still exited 0, so the deception pipeline computed gates on
unjudged logs. The port now comes from the OS, the tunnel exits on a failed forward,
and the script exits 1 while any log lacks `judged_alt`."""

from __future__ import annotations

import re
import socket
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "local_judge.sh"


def test_script_parses():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_default_port_is_free_even_when_a_19xxx_port_is_taken():
    line = next(l for l in SCRIPT.read_text().splitlines() if l.startswith("LPORT="))
    with socket.socket() as held:
        held.bind(("127.0.0.1", 0))
        held.listen()
        for _ in range(20):
            out = subprocess.run(["bash", "-c", f"unset LPORT; {line}; echo $LPORT"],
                                 capture_output=True, text=True, check=True).stdout.strip()
            port = int(out)
            assert port != held.getsockname()[1]
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", port))


def test_tunnel_exits_on_a_failed_forward():
    tunnel = next(l for l in SCRIPT.read_text().splitlines() if " -N -L " in l)
    assert "ExitOnForwardFailure=yes" in tunnel


def test_exits_nonzero_while_a_log_is_unjudged():
    text = SCRIPT.read_text()
    tail = text[text.index('echo "all done'):]
    assert re.search(r"UNJUDGED.*missing=1", tail, re.S) and tail.rstrip().endswith("exit $missing")
