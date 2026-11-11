"""A local relay that adds a thinking budget to every chat completion.

safe-tre-agent's model client sends only the model, the temperature and the
messages, and vLLM takes `thinking_token_budget` per request only (0.32 has no
server-side default), so the sizing study's "1k think" cannot reach an
end-to-end Chimp run any other way. This relay sits between the question bank
and the tunnel on 127.0.0.1, sets the budget and turns thinking on in each
POST to .../chat/completions, and passes everything else through unchanged,
headers (the serving key) and error statuses included.

    python scripts/think_budget_proxy.py <listen port> <upstream base, e.g. http://127.0.0.1:18123> <budget>
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import error, request

TIMEOUT = 1800  # s; longer than the question bank's own SAFETRE_LLM_TIMEOUT


def with_budget(body: bytes, budget: int) -> bytes:
    payload = json.loads(body)
    payload["thinking_token_budget"] = budget
    payload.setdefault("chat_template_kwargs", {})["enable_thinking"] = True
    return json.dumps(payload).encode()


def make_handler(upstream: str, budget: int):
    class Relay(BaseHTTPRequestHandler):
        def _relay(self, body: bytes | None):
            headers = {k: v for k, v in self.headers.items()
                       if k.lower() in ("authorization", "content-type", "accept")}
            req = request.Request(upstream + self.path, data=body, headers=headers, method=self.command)
            try:
                with request.urlopen(req, timeout=TIMEOUT) as resp:
                    status, ctype, data = resp.status, resp.headers.get("Content-Type"), resp.read()
            except error.HTTPError as e:
                status, ctype, data = e.code, e.headers.get("Content-Type"), e.read()
            self.send_response(status)
            if ctype:
                self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._relay(None)

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if self.path.rstrip("/").endswith("/chat/completions"):
                body = with_budget(body, budget)
            self._relay(body)

        def log_message(self, *args):  # the question bank's own log is the record
            pass

    return Relay


def serve(port: int, upstream: str, budget: int) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(upstream.rstrip("/"), budget))


if __name__ == "__main__":
    port, upstream, budget = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    if budget <= 0:
        sys.exit("budget must be a positive number of tokens")
    serve(port, upstream, budget).serve_forever()
