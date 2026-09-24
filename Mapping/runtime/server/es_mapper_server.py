#!/usr/bin/env python3
# ==== HF_HOME 永久路径注入 (P6_WEIGHTS) ====
import os as _os
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_TDONAV_ROOT = Path(_os.environ.get("TDONAV_ROOT", _SCRIPT_DIR.parents[3])).resolve()
_DEFAULT_HF_HOME = str(_TDONAV_ROOT / "Exp" / "assets" / "weights" / "hf_home")
_os.environ.setdefault("HF_HOME", _DEFAULT_HF_HOME)
_os.environ.setdefault("HF_HUB_CACHE", str(Path(_DEFAULT_HF_HOME) / "hub"))
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
# ============================================

"""
es_mapper_server.py  — Phase 4/5 ES Mapper Server.

Phase 4 (snapshot mode):  --progressive  (default)
Phase 5 (live mode):      --live_mode    (stateful OnlineESMapper)
"""

import base64, sys, os, socket, json, threading, time, io
import numpy as np

SOCKET_PATH = '/tmp/es_mapper.sock'
_ES_ROOT = Path(os.environ.get('ES_ROOT', str(_TDONAV_ROOT / 'Mapping' / 'runtime' / 'embodied_splat')))

for _p in (_SCRIPT_DIR, _ES_ROOT, _ES_ROOT / 'src', _ES_ROOT / 'src' / 'third_party'):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)


class ESMapperServer:
    def __init__(self, snap_base, local_pos_path,
                 top_k=10, device='cuda', progressive=True,
                 live_mode=False, live_only=False, bev_dir=None):
        if live_only and not live_mode:
            raise ValueError('live_only requires live_mode')
        self.top_k = top_k
        self._live_mode = live_mode
        self._live_only = live_only
        self._live_mapper = None
        self._mapper = None

        if not live_only:
            from snapshot_mapper import SnapshotMapper
            self._mapper = SnapshotMapper(
                snap_base=snap_base,
                local_pos_path=local_pos_path,
                top_k=top_k,
                device=device,
                progressive=progressive,
            )
            print(f'[ESMapperServer] SnapshotMapper ready. progressive={progressive}', flush=True)
        else:
            print('[ESMapperServer] live_only=True: SnapshotMapper disabled.', flush=True)

        if live_mode:
            print('[ESMapperServer] live_mode=True: loading OnlineESMapper ...', flush=True)
            from online_es_mapper import OnlineESMapper
            warmup_keyframes = int(os.environ.get('ES_WARMUP_KEYFRAMES', '30'))
            rebuild_interval = int(os.environ.get('ES_REBUILD_INTERVAL', '10'))
            if warmup_keyframes < 1 or rebuild_interval < 1:
                raise ValueError('ES_WARMUP_KEYFRAMES and ES_REBUILD_INTERVAL must be positive')
            self._live_mapper = OnlineESMapper(
                device=device, top_k=top_k, local_pos_path=None,
                warmup_n=warmup_keyframes,
                rebuild_interval=rebuild_interval,
                bev_dir=bev_dir
            )
            print(f'[ESMapperServer] OnlineESMapper warmup_keyframes={warmup_keyframes}, '
                  f'rebuild_interval={rebuild_interval}.', flush=True)
            print('[ESMapperServer] Live coordinates use absolute Habitat world; '
                  'offline local_pos anchor disabled.', flush=True)
            print('[ESMapperServer] OnlineESMapper ready.', flush=True)

    # ── command handlers ──────────────────────────────────────────────────────

    def handle_reset_episode(self, scene_id, episode_id, **kwargs):
        if self._mapper is not None:
            self._mapper.reset_episode(scene_id, episode_id)
        if self._live_mode and self._live_mapper is not None:
            self._live_mapper.reset_episode(scene_id, episode_id)
        return {'ok': True, 'message': f'Episode reset: {scene_id}/{episode_id}'}

    def handle_push_frame(self, frame_id, **kwargs):
        if self._mapper is None:
            return {'ok': False, 'error': 'snapshot mapper disabled in live_only mode'}
        self._mapper.push_frame(frame_id)
        return {'ok': True, 'frame_id': frame_id, 'nav_steps': frame_id + 1}

    def handle_push_live_frame(self, frame_id, rgb_jpeg=None, depth_bytes=None,
                                extrinsic=None, intrinsic=None, **kwargs):
        """Receive one live nav frame and add to the online mapper."""
        if not self._live_mode or self._live_mapper is None:
            return {'ok': False, 'error': 'server not in live_mode'}

        from PIL import Image as PILImage

        rgb_jpeg = _decode_binary_payload(rgb_jpeg)
        depth_bytes = _decode_binary_payload(depth_bytes)

        if rgb_jpeg is None or depth_bytes is None or extrinsic is None or intrinsic is None:
            return {'ok': False, 'error': 'missing frame data'}

        rgb = np.array(PILImage.open(io.BytesIO(rgb_jpeg)))
        depth_h = int(kwargs.get('depth_h', 480))
        depth_w = int(kwargs.get('depth_w', 640))
        depth = np.frombuffer(depth_bytes, dtype=np.float32).reshape(depth_h, depth_w).copy()
        ext = np.array(extrinsic, dtype=np.float32).reshape(4, 4)
        intr = np.array(intrinsic, dtype=np.float32).reshape(3, 3)

        self._live_mapper.add_live_frame(rgb, depth, ext, intr)
        st = self._live_mapper.get_status()
        return {'ok': True, 'frame_id': frame_id, **st}

    def handle_query(self, task_idx, category, top_k=None, **kwargs):
        if self._live_mode and self._live_mapper is not None:
            candidates = self._live_mapper.query(category, top_k)
            status = self._live_mapper.get_status()
        else:
            if self._mapper is None:
                return {'ok': False, 'error': 'no mapper available'}
            candidates = self._mapper.query(task_idx, category, top_k)
            status = self._mapper.get_status()
        return {
            'ok': True,
            'candidates': candidates,
            'num_candidates': len(candidates),
            'snap': status.get('snap', 'live'),
            'nav_steps': status.get('nav_steps', 0),
        }

    def handle_batch_query(self, categories, top_k=None, **kwargs):
        """Batch query all categories: feat_dict loaded once for all."""
        if top_k is None:
            top_k = self.top_k
        if self._live_mode and self._live_mapper is not None:
            results = self._live_mapper.query_batch(categories, top_k)
            status = self._live_mapper.get_status()
        else:
            if self._mapper is None:
                return {'ok': False, 'error': 'no mapper available'}
            # Fallback: individual queries via snapshot mapper
            results = {cat: self._mapper.query(None, cat, top_k) for cat in categories}
            status = self._mapper.get_status()
        return {
            'ok': True,
            'results': results,
            'snap': status.get('snap', 'live'),
        }

    def handle_status(self):
        status = self._mapper.get_status() if self._mapper is not None else {}
        if self._live_mode and self._live_mapper is not None:
            live_st = self._live_mapper.get_status()
            status['live'] = live_st
            if self._live_only:
                status.update(live_st)
                status['snap'] = 'live'
        return {'ok': True, **status}

    def handle_flush(self):
        if not self._live_mode or self._live_mapper is None:
            return {'ok': False, 'error': 'server not in live_mode'}
        return {'ok': True, 'live': self._live_mapper.flush_episode()}


# ── socket server ──────────────────────────────────────────────────────────────

def serve(server, socket_path=SOCKET_PATH):
    if os.path.exists(socket_path):
        os.unlink(socket_path)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(socket_path)
    sock.listen(5)
    os.chmod(socket_path, 0o777)
    print(f'[ESMapperServer] Listening on {socket_path}', flush=True)

    while True:
        conn, _ = sock.accept()
        threading.Thread(target=_handle_conn, args=(conn, server), daemon=True).start()


def _decode_binary_payload(payload):
    """Decode compact base64 payloads while retaining list compatibility."""
    if isinstance(payload, str):
        return base64.b64decode(payload.encode('ascii'), validate=True)
    if isinstance(payload, (list, tuple)):
        return bytes(payload)
    return payload


def _handle_conn(conn, server):
    stream = None
    try:
        stream = conn.makefile('rb')
        while True:
            line = stream.readline()
            if not line:
                break
            cmd = ''
            try:
                msg = json.loads(line.decode())
                cmd = msg.get('cmd', '')
                if cmd == 'reset_episode':
                    resp = server.handle_reset_episode(**msg)
                elif cmd == 'push_frame':
                    resp = server.handle_push_frame(**{k: v for k, v in msg.items() if k != 'cmd'})
                elif cmd == 'push_live_frame':
                    resp = server.handle_push_live_frame(**{k: v for k, v in msg.items() if k != 'cmd'})
                elif cmd == 'query':
                    resp = server.handle_query(**{k: v for k, v in msg.items() if k != 'cmd'})
                elif cmd == 'batch_query':
                    resp = server.handle_batch_query(**{k: v for k, v in msg.items() if k != 'cmd'})
                elif cmd == 'status':
                    resp = server.handle_status()
                elif cmd == 'flush':
                    resp = server.handle_flush()
                elif cmd == 'shutdown':
                    try:
                        server.handle_flush()
                    except Exception as flush_exc:
                        print(f'[ESMapperServer] shutdown flush failed: {flush_exc}', flush=True)
                    conn.sendall(json.dumps({'ok': True, 'message': 'shutting down'}).encode() + b'\n')
                    conn.close()
                    os._exit(0)
                else:
                    resp = {'ok': False, 'message': f'Unknown command: {cmd}'}
                conn.sendall(json.dumps(resp, separators=(',', ':')).encode() + b'\n')
                del msg, resp, line
            except Exception as e:
                import traceback
                print(f'[ESMapperServer] cmd={cmd!r} ERROR: {e}', flush=True)
                traceback.print_exc(file=__import__("sys").stdout)
                conn.sendall(json.dumps({'ok': False, 'message': str(e),
                                         'traceback': traceback.format_exc()}).encode() + b'\n')
    finally:
        if stream is not None:
            stream.close()
        conn.close()


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--socket_path', default=SOCKET_PATH)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--progressive', action='store_true', default=True)
    ap.add_argument('--no_progressive', dest='progressive', action='store_false')
    ap.add_argument('--live_mode', action='store_true',
                    help='Phase 5: use OnlineESMapper with live nav frames')
    ap.add_argument('--live_only', action='store_true',
                    help='Use only OnlineESMapper; do not load an offline snapshot map.')
    ap.add_argument('--snap_base',
                    default=str(_ES_ROOT / 'outputs_semantic' /
                                'scannet_fast_gtdepth_gtdepth' / '5cd_f0_ep0'))
    ap.add_argument('--local_pos',
                    default='/data/GOAT_bench/5cd_f0/0/extrinsic/local_pos.txt')
    ap.add_argument('--bev_dir', default=None,
                    help='P6_VIS: dir to save ES map BEV per rebuild')
    args = ap.parse_args()

    server = ESMapperServer(
        snap_base=args.snap_base,
        local_pos_path=args.local_pos,
        top_k=args.top_k,
        device=args.device,
        progressive=args.progressive,
        live_mode=args.live_mode,
        live_only=args.live_only,
        bev_dir=args.bev_dir,
    )
    serve(server, args.socket_path)
