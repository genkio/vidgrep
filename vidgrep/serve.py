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
CANDIDATE_POOL = 400  # global shots scanned to decide which videos appear in results
PER_VIDEO_MORE = 100  # scoped clips fetched when a group's "more" button is clicked


class State:
    def __init__(self, db, embed, local_map, cache: Path, pad: float, per_video, pool: int):
        self.db = db
        self.embed = embed
        self.local_map = local_map
        self.cache = cache
        self.pad = pad
        self.per_video = per_video  # None = uncapped (--all)
        self.pool = pool
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
            self._send_bytes(PAGE.encode(), "text/html; charset=utf-8")
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
            self._send_bytes(b'{"groups":[]}', "application/json")
            return
        st = self.state
        video = qs.get("video")
        with st.lock:
            vec = st.embed(query)
            if video:
                hits = search_shots(st.db, vec, PER_VIDEO_MORE, [int(video[0])])
                ids = None
            else:
                hits = search_shots(st.db, vec, st.pool)
                ids = _video_ids(st.db, [h[1] for h in hits])
        if ids is None:
            payload = {"items": [_item(h) for h in hits]}
        else:
            payload = {"groups": _group(hits, ids, st.per_video)}
        self._send_bytes(json.dumps(payload).encode(), "application/json")

    def _send_bytes(self, body: bytes, content_type: str):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")  # UI/JSON are dynamic; avoid stale mobile cache
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


def _item(hit) -> dict:
    sid, _path, start, end, score = hit
    return {"id": sid, "range": f"{fmt_time(start)}-{fmt_time(end)}", "score": round(score, 3)}


def _video_ids(db, paths) -> dict:
    uniq = list(dict.fromkeys(paths))
    if not uniq:
        return {}
    ph = ",".join("?" * len(uniq))
    return {p: i for p, i in db.execute(f"SELECT path, id FROM videos WHERE path IN ({ph})", uniq)}


# "more" flags videos with matches beyond the shown cap, so the UI can lazily fetch the rest
def _group(hits, ids: dict, per_video) -> list:
    groups, by_path = [], {}
    for hit in hits:
        path = hit[1]
        g = by_path.get(path)
        if g is None:
            g = {"video": Path(path).name, "video_id": ids.get(path), "items": [], "count": 0}
            by_path[path] = g
            groups.append(g)
        g["count"] += 1
        if per_video is None or len(g["items"]) < per_video:
            g["items"].append(_item(hit))
    for g in groups:
        g["more"] = per_video is not None and g["count"] > len(g["items"])
        del g["count"]
    return groups


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
    ap.add_argument("-k", type=int, default=5, help="clips shown per video before the 'more' button")
    ap.add_argument("--all", action="store_true", help="show every match per video, no cap (heavier)")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--pad", type=float, default=0.5, help="seconds added before/after each clip")
    ap.add_argument("--encoder", type=Path, help="exported encoder bundle; serve without PyTorch")
    ap.add_argument("--videos", type=Path, help="remap indexed paths to local files by filename")
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found, install with: brew install ffmpeg")

    db = open_db(args.db, check_same_thread=False)
    total = db.execute("SELECT count(*) FROM shots").fetchone()[0]
    if total == 0:
        sys.exit("index is empty, run vidgrep index first")
    per_video = None if args.all else args.k
    pool = total if args.all else min(total, CANDIDATE_POOL)

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
    server.state = State(db, embed, local_map, cache, args.pad, per_video, pool)
    server.daemon_threads = True

    print(f"vidgrep serving on port {args.port}:")
    print(f"  http://localhost:{args.port}")
    for ip in _lan_ips():
        print(f"  http://{ip}:{args.port}   (LAN / Tailscale)")
    print("every match per video" if args.all else f"{args.k} clips per video, 'more' loads the rest")
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
  #history { display: flex; flex-wrap: wrap; gap: 6px; max-width: 900px; margin: 8px auto 0; }
  .chip { padding: 5px 10px; border-radius: 999px; border: 1px solid #313747; background: #171a21;
          color: #cdd3e0; font-size: 13px; cursor: pointer; max-width: 240px; overflow: hidden;
          text-overflow: ellipsis; white-space: nowrap; }
  .chip:hover { background: #232733; }
  .chip.clear { color: #8b93a7; border-style: dashed; }
  main { padding: 10px; max-width: 1400px; margin: 0 auto; }
  .group { margin-bottom: 10px; }
  .group-head { display: flex; align-items: center; gap: 8px; padding: 8px 4px;
                cursor: pointer; user-select: none; border-bottom: 1px solid #232733; }
  .group-head .chev { display: inline-block; color: #8b93a7; transition: transform .12s; }
  .group.collapsed .chev { transform: rotate(-90deg); }
  .group-head .gname { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
                       font-weight: 600; }
  .group-head .gcount { color: #8b93a7; font-variant-numeric: tabular-nums; font-size: 13px; }
  .group-body { display: grid; gap: 10px; margin: 10px 0 4px;
                grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); }
  .group.collapsed .group-body { display: none; }
  .group.collapsed .more { display: none; }
  .more { width: 100%; margin-top: 8px; padding: 8px; border: 1px dashed #313747;
          border-radius: 8px; background: transparent; color: #8b93a7; cursor: pointer;
          font-size: 13px; }
  .more:hover { background: #171a21; color: #cdd3e0; }
  .card { background: #171a21; border: 1px solid #232733; border-radius: 10px;
          overflow: hidden; }
  .card video { width: 100%; display: block; background: #000; aspect-ratio: 16/9; }
  .meta { display: flex; justify-content: space-between; align-items: center;
          padding: 6px 8px; gap: 6px; font-size: 12px; }
  .meta .name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .meta .score { color: #8b93a7; font-variant-numeric: tabular-nums; }
  .actions { display: flex; gap: 6px; padding: 0 8px 8px; }
  .actions a, .actions button { flex: 1; text-align: center; text-decoration: none;
          padding: 6px; border-radius: 6px; background: #232733; color: #e6e6e6;
          font-size: 12px; border: 0; cursor: pointer; }
  .actions a:hover, .actions button:hover { background: #2c3242; }
  #overlay { position: fixed; inset: 0; z-index: 10; background: rgba(8,10,14,.94);
             display: flex; flex-direction: column; }
  #overlay[hidden] { display: none; }
  #overlay .bar { display: flex; align-items: center; gap: 10px; padding: 10px 14px;
                  font-size: 13px; color: #cdd3e0; }
  #overlay .bar .title { flex: 1; overflow: hidden; text-overflow: ellipsis;
                         white-space: nowrap; }
  #overlay .bar .pos { color: #8b93a7; font-variant-numeric: tabular-nums; }
  #overlay .bar button { padding: 6px 12px; background: #232733; font-size: 13px; }
  #overlay .bar button:hover { background: #2c3242; }
  #player { width: 100%; flex: 1; min-height: 0; background: #000; }
</style>
</head>
<body>
<header>
  <form id="f">
    <input id="q" placeholder="describe a scene, e.g. two people kissing" autofocus autocomplete="off">
    <button>Search</button>
  </form>
  <div id="status"></div>
  <div id="history"></div>
</header>
<main id="grid"></main>
<div id="overlay" hidden>
  <div class="bar">
    <span class="title" id="ov-title"></span>
    <span class="pos" id="ov-pos"></span>
    <button type="button" id="ov-prev">&#8249; prev</button>
    <button type="button" id="ov-next">next &#8250;</button>
    <button type="button" id="ov-close">&#10005;</button>
  </div>
  <video id="player" controls playsinline></video>
</div>
<script>
const HKEY = 'vidgrep-history', HMAX = 15;
const f = document.getElementById('f'), q = document.getElementById('q');
const grid = document.getElementById('grid'), status = document.getElementById('status');
const hist = document.getElementById('history');
let currentQuery = '';

f.addEventListener('submit', (e) => { e.preventDefault(); runSearch(q.value); });

async function runSearch(query) {
  query = query.trim();
  if (!query) return;
  q.value = query; currentQuery = query;
  status.textContent = 'searching...';
  grid.innerHTML = '';
  const res = await fetch('/search?q=' + encodeURIComponent(query));
  const { groups } = await res.json();
  addHistory(query);
  if (!groups.length) { status.textContent = 'no matches'; return; }
  const shown = groups.reduce((n, g) => n + g.items.length, 0);
  status.textContent = shown + ' clips across ' + groups.length +
                       (groups.length > 1 ? ' videos' : ' video');
  for (const g of groups) grid.appendChild(groupEl(g));
}

// sessionStorage: history lives for the tab's lifetime, gone when it closes
function loadHistory() {
  try { return JSON.parse(sessionStorage.getItem(HKEY)) || []; } catch { return []; }
}
function addHistory(query) {
  const h = [query, ...loadHistory().filter((x) => x !== query)].slice(0, HMAX);
  sessionStorage.setItem(HKEY, JSON.stringify(h));
  renderHistory();
}
function renderHistory() {
  const h = loadHistory();
  hist.innerHTML = '';
  for (const query of h) {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'chip';
    chip.textContent = query;
    chip.addEventListener('click', () => runSearch(query));
    hist.appendChild(chip);
  }
  if (h.length) {
    const clear = document.createElement('button');
    clear.type = 'button';
    clear.className = 'chip clear';
    clear.textContent = 'clear';
    clear.addEventListener('click', () => { sessionStorage.removeItem(HKEY); renderHistory(); });
    hist.appendChild(clear);
  }
}
renderHistory();

function groupEl(g) {
  const sec = document.createElement('section');
  sec.className = 'group';
  const head = document.createElement('div');
  head.className = 'group-head';
  head.innerHTML = `<span class="chev">▾</span>
    <span class="gname" title="${g.video}">${g.video}</span>
    <span class="gcount">${g.items.length}${g.more ? '+' : ''}</span>`;
  head.addEventListener('click', () => sec.classList.toggle('collapsed'));
  const body = document.createElement('div');
  body.className = 'group-body';
  for (const r of g.items) body.appendChild(card(r, g.video));
  sec.append(head, body);
  if (g.more) {
    const more = document.createElement('button');
    more.type = 'button';
    more.className = 'more';
    more.textContent = 'more from this video';
    more.addEventListener('click', async () => {
      more.textContent = 'loading...';
      const res = await fetch('/search?video=' + g.video_id + '&q=' + encodeURIComponent(currentQuery));
      const { items } = await res.json();
      body.innerHTML = '';
      for (const r of items) body.appendChild(card(r, g.video));
      head.querySelector('.gcount').textContent = items.length;
      more.remove();
    });
    sec.append(more);
  }
  return sec;
}

function card(r, video) {
  const el = document.createElement('div');
  el.className = 'card';
  const clip = '/clip/' + r.id + '.mp4';
  el.innerHTML = `
    <video controls preload="none" poster="/thumb/${r.id}.jpg" playsinline
           data-id="${r.id}" data-name="${video}" data-range="${r.range}">
      <source src="${clip}" type="video/mp4">
    </video>
    <div class="meta">
      <span class="name">${r.range}</span>
      <span class="score">${r.score}</span>
    </div>
    <div class="actions">
      <a href="${clip}" download="${video}_${r.range}.mp4">Save</a>
      <button type="button">Play all</button>
    </div>`;
  const v = el.querySelector('video');
  v.addEventListener('click', () => { if (v.paused) v.play(); });
  el.querySelector('.actions button').addEventListener('click', () => openPlayer(v.dataset.id));
  return el;
}

// overlay player: watch results back-to-back across group boundaries. In-page
// rather than the fullscreen API: native fullscreen players ignore our src swap
// on 'ended' on some setups.
const overlay = document.getElementById('overlay');
const player = document.getElementById('player');
let queue = [], qi = -1;

function openPlayer(startId) {
  queue = [...grid.querySelectorAll('.card video')].map((v) => ({
    id: v.dataset.id, name: v.dataset.name, range: v.dataset.range,
  }));
  overlay.hidden = false;
  playAt(queue.findIndex((c) => c.id === startId));
}

function playAt(i) {
  if (i < 0 || i >= queue.length) return;
  qi = i;
  const c = queue[i];
  document.getElementById('ov-title').textContent = c.name + '  ' + c.range;
  document.getElementById('ov-pos').textContent = (i + 1) + ' / ' + queue.length;
  player.src = '/clip/' + c.id + '.mp4';
  player.play();
  warm(queue[i + 1] && queue[i + 1].id);
}

function closePlayer() {
  overlay.hidden = true;
  player.pause();
  player.removeAttribute('src');
  player.load();
}

// 1-byte range fetch makes the server cut the next clip during playback, so the
// switch doesn't stall on ffmpeg
function warm(id) {
  if (id) fetch('/clip/' + id + '.mp4', { headers: { Range: 'bytes=0-0' } }).catch(() => {});
}

player.addEventListener('ended', () => playAt(qi + 1));
document.getElementById('ov-prev').addEventListener('click', () => playAt(qi - 1));
document.getElementById('ov-next').addEventListener('click', () => playAt(qi + 1));
document.getElementById('ov-close').addEventListener('click', closePlayer);
document.addEventListener('keydown', (e) => {
  if (!overlay.hidden && e.key === 'Escape') closePlayer();
});
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
