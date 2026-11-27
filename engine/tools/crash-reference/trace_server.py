#!/usr/bin/env python3
"""A tiny local server for trace.html: correct the reference frames' tracings by hand.

    python3 engine/tools/crash-reference/trace_server.py [--port 8770]
        [--frames ~/Development/rubble-artifacts/crash-reference]
    then open http://127.0.0.1:8770/

Serves the page, the tracings (`engine/docs/crash-reference/tracings/*.json`: read, and
written back when the page saves) and the reference frames (third-party footage, read
from the artifacts folder, never from or into the repository). Listens on 127.0.0.1 only.
Standard library only.
"""
import argparse
import http.server
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRACINGS = HERE.parents[1] / 'docs' / 'crash-reference' / 'tracings'
NAME = re.compile(r'^[A-Za-z0-9_.-]+\.json$')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8770)
    ap.add_argument('--frames', default=str(Path.home() / 'Development' / 'rubble-artifacts' / 'crash-reference'))
    a = ap.parse_args()
    frames = Path(a.frames).expanduser().resolve()

    class H(http.server.BaseHTTPRequestHandler):
        def send(self, code, body, ctype='application/json'):
            b = body if isinstance(body, bytes) else body.encode()
            self.send_response(code)
            self.send_header('content-type', ctype)
            self.send_header('cache-control', 'no-store')
            self.send_header('content-length', str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            p = self.path.split('?')[0]
            if p in ('/', '/trace.html'):
                return self.send(200, (HERE / 'trace.html').read_bytes(), 'text/html; charset=utf-8')
            if p == '/api/list':
                return self.send(200, json.dumps(sorted(x.name for x in TRACINGS.glob('*.json'))))
            if p.startswith('/api/tracing/'):
                name = p.rsplit('/', 1)[1]
                f = TRACINGS / name
                if NAME.match(name) and f.exists():
                    return self.send(200, f.read_bytes())
                return self.send(404, '{}')
            if p.startswith('/frames/'):
                f = (frames / p[len('/frames/'):]).resolve()
                if f.is_file() and frames in f.parents:
                    ctype = {'.webp': 'image/webp', '.png': 'image/png', '.jpg': 'image/jpeg'}.get(f.suffix.lower(), 'application/octet-stream')
                    return self.send(200, f.read_bytes(), ctype)
                return self.send(404, 'not found', 'text/plain')
            self.send(404, 'not found', 'text/plain')

        def do_POST(self):
            p = self.path.split('?')[0]
            name = p.rsplit('/', 1)[1]
            if not (p.startswith('/api/tracing/') and NAME.match(name)):
                return self.send(404, '{}')
            n = int(self.headers.get('content-length', 0))
            try:
                doc = json.loads(self.rfile.read(n))
                assert isinstance(doc.get('halves'), dict) and 'frame' in doc
            except Exception as e:  # a malformed save never replaces a tracing
                return self.send(400, json.dumps({'error': str(e)}))
            (TRACINGS / name).write_text(json.dumps(doc, indent=1) + '\n')
            self.send(200, json.dumps({'saved': name}))

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(('127.0.0.1', a.port), H)
    print(f'http://127.0.0.1:{a.port}/  (tracings: {TRACINGS}; frames: {frames})')
    srv.serve_forever()


if __name__ == '__main__':
    main()
