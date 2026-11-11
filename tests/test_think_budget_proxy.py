"""scripts/think_budget_proxy.py: the budget reaches every chat completion,
and nothing else is changed on the way through."""

import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib import error, request

import pytest


def _script():
    spec = importlib.util.spec_from_file_location(
        "think_budget_proxy", Path(__file__).parents[1] / "scripts" / "think_budget_proxy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _script()


class Upstream(BaseHTTPRequestHandler):
    """Echoes what it received; /fail answers 429 as a busy server would."""
    seen: list = []

    def _answer(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._answer(200, {"path": self.path, "auth": self.headers.get("Authorization")})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Upstream.seen.append((self.path, body, self.headers.get("Authorization")))
        self._answer(429 if self.path.endswith("/fail") else 200, {"ok": True})

    def log_message(self, *args):
        pass


def _start(server):
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture
def relay():
    Upstream.seen = []
    up = _start(ThreadingHTTPServer(("127.0.0.1", 0), Upstream))
    px = _start(P.serve(0, f"http://127.0.0.1:{up.server_port}", 1024))
    yield f"http://127.0.0.1:{px.server_port}"
    px.shutdown()
    up.shutdown()


def _post(url, payload, key="k"):
    req = request.Request(url, data=json.dumps(payload).encode(), method="POST",
                          headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with request.urlopen(req, timeout=10) as r:
        return r.status


def test_chat_completions_carry_the_budget_and_thinking_on(relay):
    msg = {"model": "m", "temperature": 0.0, "messages": [{"role": "user", "content": "hi"}]}
    assert _post(relay + "/v1/chat/completions", msg) == 200
    path, body, auth = Upstream.seen[0]
    assert path == "/v1/chat/completions" and auth == "Bearer k"
    assert body["thinking_token_budget"] == 1024
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert {k: body[k] for k in msg} == msg


def test_other_requests_pass_through_unchanged(relay):
    with request.urlopen(request.Request(relay + "/v1/models", headers={"Authorization": "Bearer k"}), timeout=10) as r:
        assert json.load(r) == {"path": "/v1/models", "auth": "Bearer k"}
    assert _post(relay + "/v1/completions", {"prompt": "x"}) == 200
    assert Upstream.seen[0][1] == {"prompt": "x"}


def test_an_upstream_error_status_reaches_the_client(relay):
    with pytest.raises(error.HTTPError) as e:
        _post(relay + "/v1/fail", {"x": 1})
    assert e.value.code == 429
