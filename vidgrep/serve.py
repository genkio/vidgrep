import argparse
import json
import shutil
import socket
import subprocess
import sys
import threading
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from vidgrep.common import DEFAULT_DB, fmt_time, open_db, search_shots
from vidgrep.cut import ffmpeg_cut, make_embedder

CACHE_DIR = Path.home() / ".vidgrep" / "serve-cache"


class State:
    def __init__(self, db, embed, local_map, cache: Path, pad: float, k: int):
        self.db = db
        self.embed = embed
        self.local_map = local_map
        self.cache = cache
        self.pad = pad
        self.k = k
        self.lock = threading.Lock()  # guards db + embed (sqlite conn shared across threads)
        self.file_locks: dict[Path, threading.Lock] = defaultdict(threading.Lock)

    def source(self, shot_id: int):
        with self.lock:
            row = self.db.execute(
                "SELECT v.path, s.start_s, s.end_s FROM shots s"
                " JOIN videos v ON v.id = s.video_id WHERE s.id = ?",
                (shot_id,),
            ).fetchone()
        if row is None:
            return None
        path, start, end = row
        if self.local_map is not None:
            path = self.local_map.get(Path(path).name, path)
        return Path(path), start, end

    def clip(self, shot_id: int) -> Path | None:
        info = self.source(shot_id)
        if info is None:
            return None
        src, start, end = info
        dest = self.cache / f"clip_{shot_id}_p{self.pad:g}.mp4"
        return self._ensure(dest, lambda: ffmpeg_cut(src, start, end, self.pad, dest), src)

    def thumb(self, shot_id: int) -> Path | None:
        info = self.source(shot_id)
        if info is None:
            return None
        src, start, end = info
        dest = self.cache / f"thumb_{shot_id}.jpg"
        return self._ensure(dest, lambda: _ffmpeg_thumb(src, (start + end) / 2, dest), src)

    # generate once, then reuse; per-file lock lets distinct clips render in parallel
    def _ensure(self, dest: Path, build, src: Path) -> Path | None:
        if dest.exists():
            return dest
        with self.file_locks[dest]:
            if dest.exists():
                return dest
            if not src.exists():
                return None
            try:
                build()
            except subprocess.CalledProcessError:
                return None
        return dest


def _ffmpeg_thumb(src: Path, t: float, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{t:.3f}", "-i", str(src), "-frames:v", "1",
            "-vf", "scale=-2:240", str(dest),
        ],
        check=True,
    )


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # quiet; the terminal already shows the startup banner

    @property
    def state(self) -> State:
        return self.server.state

    def do_GET(self):
        url = urlparse(self.path)
        route = url.path
        if route == "/":
            page = PAGE.replace("__K__", str(self.state.k))
            self._send_bytes(page.encode(), "text/html; charset=utf-8")
        elif route == "/search":
            self._search(parse_qs(url.query))
        elif route.startswith("/clip/"):
            self._send_media(self.state.clip(_id_from(route)), "video/mp4")
        elif route.startswith("/thumb/"):
            self._send_media(self.state.thumb(_id_from(route)), "image/jpeg")
        else:
            self.send_error(404)

    def _search(self, qs: dict):
        query = (qs.get("q") or [""])[0].strip()
        if not query:
            self._send_bytes(b'{"results":[]}', "application/json")
            return
        k = int((qs.get("k") or [self.state.k])[0])
        with self.state.lock:
            hits = search_shots(self.state.db, self.state.embed(query), k)
        results = [
            {
                "id": sid,
                "video": Path(path).name,
                "range": f"{fmt_time(start)}-{fmt_time(end)}",
                "score": round(score, 3),
            }
            for sid, path, start, end, score in hits
        ]
        self._send_bytes(json.dumps({"results": results}).encode(), "application/json")

    def _send_bytes(self, body: bytes, content_type: str):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # HTML5 <video> needs byte-range replies (206) to seek and, on some browsers, to play
    def _send_media(self, path: Path | None, content_type: str):
        if path is None or not path.exists():
            self.send_error(404)
            return
        size = path.stat().st_size
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        if rng and rng.startswith("bytes="):
            lo, _, hi = rng[6:].partition("-")
            start = int(lo) if lo else 0
            end = int(hi) if hi else size - 1
            end = min(end, size - 1)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = f.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return  # player seeked/closed mid-stream
                remaining -= len(chunk)


def _id_from(route: str) -> int:
    stem = route.rsplit("/", 1)[-1].split(".")[0]
    return int(stem) if stem.isdigit() else -1


def _lan_ips() -> list[str]:
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    return sorted(ips)


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve a local web UI to search and clip indexed videos.")
    ap.add_argument("--host", default="0.0.0.0", help="bind address (0.0.0.0 = reachable over LAN/Tailscale)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("-k", type=int, default=12, help="results per search")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--pad", type=float, default=0.5, help="seconds added before/after each clip")
    ap.add_argument("--encoder", type=Path, help="exported encoder bundle; serve without PyTorch")
    ap.add_argument("--videos", type=Path, help="remap indexed paths to local files by filename")
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found, install with: brew install ffmpeg")

    db = open_db(args.db, check_same_thread=False)
    if db.execute("SELECT count(*) FROM shots").fetchone()[0] == 0:
        sys.exit("index is empty, run vidgrep index first")

    local_map = None
    if args.videos:
        from vidgrep.portable import local_video_map

        local_map = local_video_map(args.videos)

    # model load can take 15-30s; say so before blocking, or the terminal looks hung
    print("loading model ...", flush=True)
    embed = make_embedder(args.encoder)
    cache = CACHE_DIR / args.db.expanduser().stem
    cache.mkdir(parents=True, exist_ok=True)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.state = State(db, embed, local_map, cache, args.pad, args.k)
    server.daemon_threads = True

    print(f"vidgrep serving on port {args.port}:")
    print(f"  http://localhost:{args.port}")
    for ip in _lan_ips():
        print(f"  http://{ip}:{args.port}   (LAN / Tailscale)")
    print("Ctrl-C to stop", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>vidgrep</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 15px/1.4 -apple-system, system-ui, sans-serif;
         background: #0f1115; color: #e6e6e6; }
  header { position: sticky; top: 0; background: #0f1115; padding: 16px;
           border-bottom: 1px solid #232733; }
  form { display: flex; gap: 8px; max-width: 900px; margin: 0 auto; }
  input { flex: 1; padding: 10px 14px; border-radius: 8px; border: 1px solid #313747;
          background: #171a21; color: #e6e6e6; font-size: 16px; }
  button { padding: 10px 18px; border: 0; border-radius: 8px; background: #3b82f6;
           color: #fff; font-size: 15px; cursor: pointer; }
  button:hover { background: #2f6fe0; }
  #status { max-width: 900px; margin: 10px auto 0; color: #8b93a7; font-size: 13px; }
  main { display: grid; gap: 16px; padding: 16px;
         grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); max-width: 1400px;
         margin: 0 auto; }
  .card { background: #171a21; border: 1px solid #232733; border-radius: 10px;
          overflow: hidden; }
  .card video { width: 100%; display: block; background: #000; aspect-ratio: 16/9; }
  .meta { display: flex; justify-content: space-between; align-items: center;
          padding: 8px 12px; gap: 8px; }
  .meta .name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .meta .score { color: #8b93a7; font-variant-numeric: tabular-nums; }
  .actions { display: flex; gap: 8px; padding: 0 12px 12px; }
  .actions a, .actions button { flex: 1; text-align: center; text-decoration: none;
          padding: 7px; border-radius: 7px; background: #232733; color: #e6e6e6;
          font-size: 13px; border: 0; cursor: pointer; }
  .actions a:hover, .actions button:hover { background: #2c3242; }
</style>
</head>
<body>
<header>
  <form id="f">
    <input id="q" placeholder="describe a scene, e.g. two people kissing" autofocus autocomplete="off">
    <button>Search</button>
  </form>
  <div id="status"></div>
</header>
<main id="grid"></main>
<script>
const DEFAULT_K = __K__;
const f = document.getElementById('f'), q = document.getElementById('q');
const grid = document.getElementById('grid'), status = document.getElementById('status');

f.addEventListener('submit', async (e) => {
  e.preventDefault();
  const query = q.value.trim();
  if (!query) return;
  status.textContent = 'searching...';
  grid.innerHTML = '';
  const res = await fetch('/search?k=' + DEFAULT_K + '&q=' + encodeURIComponent(query));
  const { results } = await res.json();
  status.textContent = results.length ? results.length + ' results' : 'no matches';
  for (const r of results) grid.appendChild(card(r));
});

function card(r) {
  const el = document.createElement('div');
  el.className = 'card';
  const clip = '/clip/' + r.id + '.mp4';
  el.innerHTML = `
    <video controls preload="none" poster="/thumb/${r.id}.jpg" playsinline>
      <source src="${clip}" type="video/mp4">
    </video>
    <div class="meta">
      <span class="name" title="${r.video}">${r.video} ${r.range}</span>
      <span class="score">${r.score}</span>
    </div>
    <div class="actions">
      <a href="${clip}" download="${r.video}_${r.range}.mp4">Save</a>
      <button type="button">Fullscreen</button>
    </div>`;
  const video = el.querySelector('video');
  video.addEventListener('click', () => { if (video.paused) video.play(); });
  el.querySelector('button').addEventListener('click', () => {
    (video.requestFullscreen || video.webkitEnterFullscreen).call(video);
  });
  return el;
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
