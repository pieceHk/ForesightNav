#!/usr/bin/env python3
"""
online_es_provider.py  — Phase 4 OnlineESProvider (with timing instrumentation).
"""

import base64, json, socket, os, gzip, time
from typing import Dict, List, Optional
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_INTEG_ROOT = _SCRIPT_DIR.parent
_REPO_ROOT = _INTEG_ROOT.parent
_LAGMEMO_ROOT = Path(os.environ.get('LAGMEMO_ROOT', str(_REPO_ROOT / 'LagMemo')))

_lagmemo_src = str(_LAGMEMO_ROOT / 'src' / 'lagmemo')
if _lagmemo_src not in sys.path:
    sys.path.insert(0, _lagmemo_src)
from lagmemo.agent.lagmemo_agent.candidate_provider import CandidateProvider

SOCKET_PATH = '/tmp/es_mapper.sock'
CONTENT_ROOT = str(_LAGMEMO_ROOT / 'data' / 'datasets' / 'goat' / 'hm3d' / '3_episode_data' / 'content')


class OnlineESProvider(CandidateProvider):
    """CandidateProvider backed by the live ES Mapper Server (IPC)."""

    def __init__(self, socket_path: str = SOCKET_PATH,
                 top_k: int = 10,
                 connect_timeout: float = 60.0,
                 content_root: str = CONTENT_ROOT):
        self.socket_path = socket_path
        self.top_k = top_k
        self._connect_timeout = connect_timeout
        self._content_root = content_root
        self._sock: Optional[socket.socket] = None
        self._buf = b''
        self._scene_id: Optional[str] = None
        self._episode_id: Optional[str] = None
        self._tasks: List = []
        self._goals_cache: Dict = {}
        self._candidate_records_cache: Dict = {}
        # ── timing state ────────────────────────────────────────────────
        self._t_push_ms: float = 0.0       # last push_frame round-trip
        self._t_query_ms: float = 0.0      # last get_goals round-trip
        self._t_get_all_goals_ms: float = 0.0  # last get_all_goals total
        self.current_snap: str = ''         # snap basename after last query
        self.n_live_frames: int = 0         # frames sent via push_live_frame
        self._live_mode: bool = False        # True when using Phase 5 server

    # ── connection ──────────────────────────────────────────────────────

    def _ensure_connected(self):
        if self._sock is not None:
            return
        deadline = time.time() + self._connect_timeout
        while time.time() < deadline:
            if os.path.exists(self.socket_path):
                try:
                    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    s.connect(self.socket_path)
                    self._sock = s
                    self._buf = b''
                    print(f'[OnlineESProvider] Connected to {self.socket_path}', flush=True)
                    return
                except ConnectionRefusedError:
                    pass
            time.sleep(0.5)
        raise RuntimeError(f'ES Mapper Server not available at {self.socket_path}')

    def _send_recv(self, msg: dict) -> dict:
        self._ensure_connected()
        payload = json.dumps(msg, separators=(',', ':')).encode() + b'\n'
        self._sock.sendall(payload)
        del payload
        while b'\n' not in self._buf:
            chunk = self._sock.recv(65536)
            if not chunk:
                raise RuntimeError('ES Mapper Server closed connection')
            self._buf += chunk
        line, self._buf = self._buf.split(b'\n', 1)
        return json.loads(line.decode())

    # ── content file helper ────────────────────────────────────────────

    def _load_tasks_for_episode(self, scene_id: str, episode_id: str) -> List:
        content_path = os.path.join(self._content_root, f'{scene_id}.json.gz')
        if not os.path.exists(content_path):
            print(f'[OnlineESProvider] WARNING: content file not found: {content_path}', flush=True)
            return []
        with gzip.open(content_path, 'rt') as f:
            data = json.load(f)
        episodes = data if isinstance(data, list) else data.get('episodes', [])
        # Habitat reindexes filtered content to 0..N-1 while the retained
        # GOAT records keep source episode IDs.  When the content is visibly
        # reindexed, runtime IDs can collide with a different source ID, so
        # the deterministic content order must take precedence.
        try:
            runtime_index = int(episode_id)
        except (TypeError, ValueError):
            runtime_index = -1
        has_reindexed_ids = any(
            str(ep.get('episode_id', '')) != str(index)
            for index, ep in enumerate(episodes)
        )
        if has_reindexed_ids and 0 <= runtime_index < len(episodes):
            matched = episodes[runtime_index]
            source_episode_id = matched.get('episode_id', '')
            tasks = matched.get('tasks', [])
            print(
                '[OnlineESProvider] episode ID reindex mapping: '
                f'runtime={episode_id} content={source_episode_id} '
                f'({len(tasks)} tasks)',
                flush=True,
            )
            return tasks
        for ep in episodes:
            if str(ep.get('episode_id', '')) == str(episode_id):
                return ep.get('tasks', [])
        print(f'[OnlineESProvider] WARNING: episode {episode_id} not found', flush=True)
        return []

    # ── CandidateProvider interface ────────────────────────────────────

    def reset_episode(self, scene_id: str, episode_id: str) -> None:
        self._scene_id = scene_id
        self._episode_id = episode_id
        self._tasks = self._load_tasks_for_episode(scene_id, episode_id)
        self._goals_cache = {}
        self._candidate_records_cache = {}
        resp = self._send_recv({'cmd': 'reset_episode',
                                'scene_id': scene_id, 'episode_id': episode_id})
        if not resp.get('ok'):
            raise RuntimeError(f'reset_episode failed: {resp}')
        self.n_live_frames = 0
        print(f'[OnlineESProvider] Episode reset: {scene_id}/{episode_id} '
              f'({len(self._tasks)} tasks)', flush=True)


    # ── live-mode frame push ────────────────────────────────────────────────

    def push_live_frame(self, rgb: 'np.ndarray', depth: 'np.ndarray',
                        globalpose: 'np.ndarray') -> None:
        """Send one Habitat observation frame to the ES server for live mapping.

        rgb        : (H, W, 3) uint8
        depth      : (H, W)    float32, metres
        globalpose : (7,) float32 [qw, qx, qy, qz, px, py, pz]
        """
        import math as _math, io as _io
        import numpy as _np
        from PIL import Image as _PILImage

        # ── build c2w from globalpose (matches snapshot_mapper.py convention) ──
        _RUB2RDF = _np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=_np.float64)
        qw, qx, qy, qz = globalpose[:4]
        px, py, pz = globalpose[4:7]
        x, y, z, w = qw, qx, qy, qz
        R = _np.array([
            [1-2*y*y-2*z*z, 2*x*y-2*z*w,   2*x*z+2*y*w],
            [2*x*y+2*z*w,   1-2*x*x-2*z*z,  2*y*z-2*x*w],
            [2*x*z-2*y*w,   2*y*z+2*x*w,   1-2*x*x-2*y*y],
        ], dtype=_np.float64)
        c2w = _np.eye(4, dtype=_np.float64)
        c2w[:3,:3] = R @ _RUB2RDF
        c2w[:3, 3] = _RUB2RDF @ _np.array([px, py, pz])

        # ── serialise rgb as JPEG bytes ──
        buf = _io.BytesIO()
        _PILImage.fromarray(rgb.astype(_np.uint8)).save(buf, format='JPEG', quality=92)
        rgb_bytes = base64.b64encode(buf.getvalue()).decode('ascii')
        buf.close()

        # ── serialise depth as raw float32 bytes ──
        H, W = depth.shape[:2]
        depth_raw = _np.ascontiguousarray(depth, dtype=_np.float32).tobytes()
        depth_bytes = base64.b64encode(depth_raw).decode('ascii')
        del depth_raw

        depth_h, depth_w = depth.shape[:2]
        t0 = time.perf_counter()
        resp = self._send_recv({
            'cmd':        'push_live_frame',
            'frame_id':   self.n_live_frames,
            'rgb_jpeg':   rgb_bytes,
            'depth_bytes': depth_bytes,
            'binary_encoding': 'base64',
            'depth_h':    depth_h,
            'depth_w':    depth_w,
            'extrinsic':  c2w.tolist(),
            'intrinsic':  [[1.0, 0.0, 0.0],  # placeholder (server uses internal K)
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0]],
        })
        self._t_push_ms = (time.perf_counter() - t0) * 1000
        self.n_live_frames += 1
        if not resp.get('ok'):
            print(f'[OnlineESProvider] push_live_frame warning: {resp}', flush=True)

    def get_goals(self, task_idx: int, category: str = None) -> List:
        if category is None:
            if task_idx < len(self._tasks):
                task = self._tasks[task_idx]
                category = task[0] if isinstance(task, (list, tuple)) else task.get('object_category', '')
        if not category:
            return []
        if task_idx in self._goals_cache:
            return self._goals_cache[task_idx]
        t0 = time.perf_counter()
        resp = self._send_recv({'cmd': 'query', 'task_idx': task_idx,
                                'category': category, 'top_k': self.top_k})
        self._t_query_ms = (time.perf_counter() - t0) * 1000
        if not resp.get('ok'):
            return []
        candidates = resp.get('candidates', [])
        self._goals_cache[task_idx] = candidates
        self._candidate_records_cache[task_idx] = resp.get(
            'candidate_records', []
        )
        self.current_snap = resp.get('snap', '')
        return candidates

    def _task_category(self, task) -> str:
        if isinstance(task, (list, tuple)):
            return task[0]
        elif isinstance(task, dict):
            return task.get('object_category', task.get('category', ''))
        return str(task)

    def get_all_goals(self) -> Dict[int, List]:
        """Query server for all task candidates using batch_query (single feat_dict load).

        Falls back to sequential per-task queries on protocol error.
        """
        if not self._tasks:
            return {}

        task_categories = [self._task_category(t) for t in self._tasks]
        unique_cats = list(dict.fromkeys(c for c in task_categories if c))

        t_start = time.perf_counter()
        n_unique = len(unique_cats)
        n_tasks  = len(self._tasks)
        print(f'[OnlineESProvider] get_all_goals: batch querying {n_unique} unique categories ({n_tasks} tasks)...', flush=True)

        try:
            resp = self._send_recv({'cmd': 'batch_query', 'categories': unique_cats, 'top_k': self.top_k})
        except Exception as e:
            print(f'[OnlineESProvider] batch_query failed ({e}), falling back to sequential', flush=True)
            return self._get_all_goals_sequential()

        if not resp.get('ok'):
            print(f'[OnlineESProvider] batch_query not ok, falling back to sequential', flush=True)
            return self._get_all_goals_sequential()

        cat_results = resp.get('results', {})
        cat_records = resp.get('candidate_records_by_category', {})
        self.current_snap = resp.get('snap', self.current_snap)

        result = {}
        for task_idx, cat in enumerate(task_categories):
            candidates = cat_results.get(cat, [])
            result[task_idx] = candidates
            self._goals_cache[task_idx] = candidates
            self._candidate_records_cache[task_idx] = cat_records.get(cat, [])
            # P6_FIX11: log actual coordinates (not just count) so an offline
            # script can cross-reference against GOAT GT object positions to
            # tell apart "candidate points to wrong place" (error failure) vs
            # "candidate is fine but nav/verification fails to use it" (missed
            # failure). Coordinates are rounded to keep log lines short.
            _coords_str = [[round(v, 3) for v in c] for c in candidates]
            print(f'  task {task_idx} [{cat}]: {len(candidates)} candidates '
                  f'{_coords_str}', flush=True)

        self._t_get_all_goals_ms = (time.perf_counter() - t_start) * 1000
        return result

    def get_all_candidate_records(self) -> Dict[int, List[dict]]:
        """Return optional metadata aligned with the most recent goal query."""
        return {
            int(task_idx): [dict(record) for record in records]
            for task_idx, records in self._candidate_records_cache.items()
        }

    def _get_all_goals_sequential(self) -> Dict[int, List]:
        """Fallback: sequential per-task queries (original behavior)."""
        result = {}
        for task_idx, task in enumerate(self._tasks):
            category = self._task_category(task)
            if not category:
                continue
            candidates = self.get_goals(task_idx, category)
            result[task_idx] = candidates
            print(f'  task {task_idx} [{category}]: {len(candidates)} candidates', flush=True)
        # Update snap via status
        try:
            st = self._send_recv({'cmd': 'status'})
            self.current_snap = st.get('snap', self.current_snap)
        except Exception:
            pass
        return result

    def push_frame(self, frame_id: int, **kwargs) -> None:
        """Notify server of new nav step. IPC round-trip is ~1-3 ms."""
        t0 = time.perf_counter()
        resp = self._send_recv({'cmd': 'push_frame', 'frame_id': frame_id})
        self._t_push_ms = (time.perf_counter() - t0) * 1000

    def status(self) -> dict:
        return self._send_recv({'cmd': 'status'})

    def flush_episode(self) -> dict:
        """Synchronously finalize the current online ES map and its artifacts."""
        t0 = time.perf_counter()
        resp = self._send_recv({'cmd': 'flush'})
        self._t_flush_ms = (time.perf_counter() - t0) * 1000
        if not resp.get('ok', False):
            raise RuntimeError(f"ES flush_episode failed: {resp}")
        return resp.get('live', {})

    def close(self):
        if self._sock:
            self._sock.close()
            self._sock = None
