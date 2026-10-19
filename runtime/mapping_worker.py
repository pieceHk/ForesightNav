"""Run the online mapping service with explicit backend ownership."""

import argparse
import io
import os
from pathlib import Path
import signal
import threading

from runtime.bootstrap import configure_runtime_threads
from runtime.protocol import decode_binary_payload as _decode_binary_payload, serve


class MappingServer:
    """Adapt mapping operations to the shared socket command contract."""

    def __init__(self, mapper, *, top_k=10, snapshot_mapper=None):
        self.top_k = top_k
        self._live_mapper = mapper
        self._mapper = snapshot_mapper
        self._live_mode = mapper is not None
        self._live_only = snapshot_mapper is None
        self._closed = False

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

    def handle_push_live_frame(
        self,
        frame_id,
        rgb_jpeg=None,
        depth_bytes=None,
        extrinsic=None,
        intrinsic=None,
        **kwargs,
    ):
        """Receive one live nav frame and add to the online mapper."""
        if not self._live_mode or self._live_mapper is None:
            return {'ok': False, 'error': 'server not in live_mode'}

        import numpy as np
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
            if hasattr(self._live_mapper, 'query_with_records'):
                candidates, candidate_records = self._live_mapper.query_with_records(
                    category, top_k
                )
            else:
                candidates = self._live_mapper.query(category, top_k)
                candidate_records = []
            status = self._live_mapper.get_status()
        else:
            if self._mapper is None:
                return {'ok': False, 'error': 'no mapper available'}
            candidates = self._mapper.query(task_idx, category, top_k)
            candidate_records = []
            status = self._mapper.get_status()
        return {
            'ok': True,
            'candidates': candidates,
            'candidate_records': candidate_records,
            'num_candidates': len(candidates),
            'snap': status.get('snap', 'live'),
            'nav_steps': status.get('nav_steps', 0),
        }

    def handle_batch_query(self, categories, top_k=None, **kwargs):
        """Query all categories against one published map snapshot."""
        if top_k is None:
            top_k = self.top_k
        if self._live_mode and self._live_mapper is not None:
            if hasattr(self._live_mapper, 'query_batch_with_records'):
                results, records = self._live_mapper.query_batch_with_records(categories, top_k)
            else:
                results = self._live_mapper.query_batch(categories, top_k)
                records = {}
            status = self._live_mapper.get_status()
        else:
            if self._mapper is None:
                return {'ok': False, 'error': 'no mapper available'}
            # Fallback: individual queries via snapshot mapper
            results = {cat: self._mapper.query(None, cat, top_k) for cat in categories}
            records = {}
            status = self._mapper.get_status()
        return {
            'ok': True,
            'results': results,
            'candidate_records_by_category': records,
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

    def close(self):
        if self._closed:
            return
        self._closed = True
        errors = []
        for mapper in (self._live_mapper, self._mapper):
            if mapper is not None and hasattr(mapper, "close"):
                try:
                    mapper.close()
                except BaseException as exc:
                    errors.append(exc)
        if errors:
            raise errors[0]


def run_worker(server, socket_path, *, stop_event=None, on_ready=None):
    try:
        serve(server, socket_path, stop_event=stop_event, on_ready=on_ready)
    finally:
        server.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-plan", type=Path)
    parser.add_argument("--socket_path", default="/tmp/es_mapper.sock")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--live_mode", action="store_true", default=True)
    parser.add_argument("--live_only", action="store_true", default=True)
    parser.add_argument("--bev_dir", default=None)
    args = parser.parse_args(argv)

    if args.run_plan is not None:
        from runtime.plan import load_plan
        from runtime.bootstrap import configure_run_process
        from runtime.factories import create_mapper
        plan = load_plan(args.run_plan)
        configure_run_process(plan, "mapping")
        mapper = create_mapper(plan, device=args.device)
        args.socket_path = str(plan.socket_path)
        args.top_k = plan.experiment.method.query.top_k
    else:
        root = Path(__file__).resolve().parents[1]
        os.environ.setdefault("TDONAV_ROOT", str(root))
        os.environ.setdefault("ES_ROOT", str(root / "Mapping/runtime/embodied_splat"))
        os.environ.setdefault("HF_HOME", str(root / "Exp/assets/weights/hf_home"))
        os.environ.setdefault("HF_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        configure_runtime_threads()
        from runtime.legacy_mapping import create_mapper
        mapper = create_mapper(device=args.device, top_k=args.top_k, bev_dir=args.bev_dir)
    server = MappingServer(mapper, top_k=args.top_k)
    stopped = threading.Event()
    previous = {}
    try:
        for name in (signal.SIGINT, signal.SIGTERM):
            previous[name] = signal.signal(name, lambda *_: stopped.set())
        run_worker(server, args.socket_path, stop_event=stopped)
    finally:
        server.close()
        for name, handler in previous.items():
            signal.signal(name, handler)


if __name__ == "__main__":
    main()
