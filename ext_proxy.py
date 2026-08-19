"""Usage-recording pass-through proxy for the trace re-collection in
REPORT.md section 7.

Sits between the MidScene agent runner and https://api.anthropic.com,
recording each call's token usage to JSONL for per-model cost tracking and
the collection spend cap. Applied identically to every model, bridge control
included, so the localhost overhead is uniform. For streamed requests,
stream_options include_usage is injected so the final SSE chunk carries
usage.

Usage:
  .venv/bin/python ext_proxy.py [--port 8399] [--log data/ext_usage.jsonl]
Point the runner at http://127.0.0.1:8399/v1 instead of
https://api.anthropic.com/v1.
"""

import argparse
import json
import ssl
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

import certifi

UPSTREAM = "https://api.anthropic.com"
# Claude 5-family models 400 on sampling params; older models accept them and
# received them in the authors' collection. Strip only where the API forces
# it. This model-surface difference is documented in REPORT.md.
STRIP_SAMPLING_FOR = ("claude-opus-5", "claude-sonnet-5", "claude-fable-5",
                      "claude-opus-4-7", "claude-opus-4-8")
HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
              "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
              "accept-encoding"}

log_lock = Lock()
log_path = Path("data/ext_usage.jsonl")
SSL_CTX = ssl.create_default_context(cafile=certifi.where())


def record(row):
    with log_lock:
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")


class Proxy(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_POST(self):
        t0 = time.time()
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        model, streamed = "", False
        stripped = []
        try:
            payload = json.loads(body)
            model = payload.get("model", "")
            streamed = bool(payload.get("stream"))
            changed = False
            if streamed:
                opts = payload.setdefault("stream_options", {})
                opts.setdefault("include_usage", True)
                changed = True
            if model.startswith(STRIP_SAMPLING_FOR):
                for k in ("temperature", "top_p", "top_k"):
                    if k in payload:
                        payload.pop(k)
                        stripped.append(k)
                        changed = True
            if changed:
                body = json.dumps(payload).encode()
        except Exception:
            pass

        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP}
        usage, status = {}, 502
        try:
            req = urllib.request.Request(UPSTREAM + self.path, data=body, headers=headers, method="POST")
            try:
                upstream = urllib.request.urlopen(req, timeout=600, context=SSL_CTX)
                status, resp_headers, content = upstream.status, upstream.headers, upstream.read()
            except urllib.error.HTTPError as he:
                status, resp_headers, content = he.code, he.headers, he.read()

            self.send_response(status)
            for k, v in resp_headers.items():
                if k.lower() not in HOP_BY_HOP:
                    self.send_header(k, v)
            self.send_header("Connection", "close")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

            if streamed:
                for line in content.split(b"\n"):
                    if line.startswith(b"data: ") and b'"usage"' in line:
                        try:
                            u = json.loads(line[6:]).get("usage") or {}
                            if u.get("total_tokens") or u.get("completion_tokens"):
                                usage = u
                        except Exception:
                            pass
            else:
                try:
                    usage = json.loads(content).get("usage") or {}
                except Exception:
                    pass
        except Exception as e:
            try:
                self.send_response(502)
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(str(e).encode())
            except Exception:
                pass

        record({
            "ts": time.time(),
            "path": self.path,
            "model": model,
            "status": status,
            "streamed": streamed,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "latency_s": round(time.time() - t0, 3),
            "stripped": stripped,
        })

    do_GET = do_POST


def main():
    global log_path
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8399)
    ap.add_argument("--log", default="data/ext_usage.jsonl")
    args = ap.parse_args()
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Proxy)
    print(f"proxy on 127.0.0.1:{args.port} -> {UPSTREAM}, usage -> {log_path}")
    server.serve_forever()


if __name__ == "__main__":
    main()
