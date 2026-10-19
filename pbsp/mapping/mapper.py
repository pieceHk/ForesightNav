"""Online keyframe admission, asynchronous fusion and versioned map publication."""

from __future__ import annotations

import collections
import threading
import time

import numpy as np

from pbsp.settings import MappingSettings
from .geometry import habitat_pose_to_c2w
from .keyframes import tensor_frame_change_ratio
from .memory import (_release_cuda_cache, _move_tree_to_device, _log_cuda_memory,
                     _release_encoder_transient_caches, release_host_memory)
from .model import OnlineMappingModel, preprocess_frame
from .query import query_category, query_categories


class OnlineMapper:
    """Keep one persistent map per episode and share it across ordered tasks."""

    def __init__(self, *, settings: MappingSettings, model: OnlineMappingModel,
                 device: str, artifacts=None, host_memory_hook=None, camera_intrinsic=None):
        self.settings = settings
        self.model = model
        self.device = device
        self.top_k = settings.top_k
        self.warmup_n = settings.warmup_keyframes
        self.rebuild_interval = settings.rebuild_interval
        self._artifacts = artifacts
        self._host_memory_hook = host_memory_hook or release_host_memory
        self._closed = False
        self.camera_intrinsic = None if camera_intrinsic is None else np.asarray(camera_intrinsic).copy()
        # Rejected raw frames are retained only briefly. Accepted keyframes remain
        # on CPU because the incremental encoder requires stable frame indexing.
        self._frames: list = []
        self._n_total_frames: int = 0
        self._buf_lock = threading.Lock()

        self._keyframes: list = []
        self._admission_counts = collections.Counter()
        self._last_frame_change = 1.0

        self._online_state = None

        self._globals     = None
        self._globals_lock = threading.Lock()
        self._model_lock = threading.RLock()
        self.n_frames     = 0
        self.n_gaussians  = 0

        self._rebuild_thread    = None
        self._rebuild_lock      = threading.Lock()
        self._rebuild_active    = False
        self._rebuild_pending   = False
        self._resetting         = False
        self._last_rebuild_n    = 0
        self._last_fused_n      = 0

        self._scene_id   = ''
        self._episode_id = ''
        self._episode_artifacts_exported = False



    def _export_final_map(self):
        if self._artifacts is None or not self._scene_id or self._episode_artifacts_exported:
            return []
        outputs = []
        with self._globals_lock:
            try:
                outputs = self._artifacts.on_finish(
                    scene_id=self._scene_id, episode_id=self._episode_id,
                    globals_state=self._globals, n_keyframes=len(self._keyframes),
                    n_raw_frames=self._n_total_frames,
                )
            except Exception as exc:
                print(f'[OnlineMapper] Final map export failed: {exc}', flush=True)
        self._episode_artifacts_exported = bool(outputs)
        return outputs


    def flush_episode(self):
        """Synchronously fuse and optionally export the completed episode map."""
        for _ in range(3):
            with self._buf_lock:
                n_kf = len(self._keyframes)
            if n_kf < self.warmup_n or self._last_fused_n >= n_kf:
                break
            self._maybe_start_rebuild(force=True)
            with self._rebuild_lock:
                rebuild_thread = self._rebuild_thread
            if rebuild_thread is None:
                break
            rebuild_thread.join()
        if self._last_fused_n < n_kf:
            print(f'[OnlineESMapper] Flush incomplete: fused={self._last_fused_n}, '
                  f'accepted={n_kf}', flush=True)
        artifacts = self._export_final_map()
        status = self.get_status()
        status['final_artifacts'] = artifacts
        return status


    def reset_episode(self, scene_id: str, episode_id: str):
        if self._closed:
            raise RuntimeError("Mapper is closed")
        if self._scene_id:
            self.flush_episode()
        with self._rebuild_lock:
            self._resetting = True
            self._rebuild_pending = False

        if self._rebuild_thread and self._rebuild_thread.is_alive():
            self._rebuild_thread.join()

        with self._model_lock:
            with self._buf_lock:
                old_frames = self._frames
                old_keyframes = self._keyframes
                self._frames = []
                self._keyframes = []
                self._admission_counts.clear()
                self._last_frame_change = 1.0
                self._n_total_frames = 0
                self._last_rebuild_n = 0
                self._last_fused_n = 0
                self._episode_artifacts_exported = False
            with self._globals_lock:
                old_globals = self._globals
                self._globals      = None
                self.n_gaussians   = 0
                self.n_frames      = 0

            old_online_state = self._online_state
            self._online_state = None
            del old_frames, old_keyframes, old_globals, old_online_state
        _release_cuda_cache(self.device, 'episode reset')
        self._host_memory_hook('ES mapper episode reset')
        self._scene_id   = scene_id
        self._episode_id = episode_id
        with self._rebuild_lock:
            self._rebuild_active = False
            self._rebuild_pending = False
            self._resetting = False

        print(f'[OnlineESMapper] Episode reset: {scene_id}/{episode_id}', flush=True)


    def _consider_keyframe(self, frame: dict):
        """Compare the current RGB frame only with the last accepted keyframe."""
        if self.settings.max_gaussians > 0 and self.n_gaussians >= self.settings.max_gaussians:
            return False, 'gaussian_limit', 0.0
        if self.settings.max_keyframes > 0 and len(self._keyframes) >= self.settings.max_keyframes:
            return False, 'keyframe_limit', 0.0
        if not self._keyframes:
            return True, 'first_frame', 1.0
        change = tensor_frame_change_ratio(
            frame['image'],
            self._keyframes[-1]['image'], settings=self.settings,
        )
        if change >= self.settings.keyframe_threshold:
            return True, 'frame_change', change
        return False, 'frame_similarity', change


    def _maybe_start_rebuild(self, force: bool = False):
        with self._buf_lock:
            n_kf = len(self._keyframes)
        if n_kf < self.warmup_n:
            return

        with self._rebuild_lock:
            due = (
                force
                or self._last_rebuild_n == 0
                or n_kf - self._last_rebuild_n >= self.rebuild_interval
            )
            if not due or self._resetting:
                return
            if self._rebuild_active:
                self._rebuild_pending = True
                return
            self._rebuild_active = True
            self._rebuild_pending = False
            self._last_rebuild_n = n_kf
            self._rebuild_thread = threading.Thread(
                target=self._rebuild_worker,
                daemon=True,
                name=f'es_rebuild_kf{n_kf}',
            )
            self._rebuild_thread.start()


    def add_live_frame(self, rgb: np.ndarray, depth: np.ndarray,
                       extrinsic: np.ndarray, intrinsic: np.ndarray):
        """
        Ingest one Habitat navigation frame.

        rgb       : (H, W, 3) uint8
        depth     : (H, W)    float32, metres
        extrinsic : (4, 4)    c2w float32 (from globalpose_to_c2w)
        intrinsic : (3, 3)    normalized camera intrinsic matrix
        """
        # Frame history stays on CPU; rebuild transfers its snapshot to the model.
        if self._closed:
            raise RuntimeError("Mapper is closed")
        intrinsic = self.camera_intrinsic if self.camera_intrinsic is not None else intrinsic
        frame = preprocess_frame(rgb, depth, extrinsic, intrinsic,
                                 input_spec=self.model.input_spec, device='cpu')
        with self._buf_lock:
            self._frames.append(frame)
            self._n_total_frames += 1
            n = self._n_total_frames
            if self.settings.raw_frame_buffer <= 0:
                self._frames.clear()
            elif len(self._frames) > self.settings.raw_frame_buffer:
                del self._frames[:-self.settings.raw_frame_buffer]
        with self._globals_lock:
            self.n_frames = n

        with self._buf_lock:
            admitted, reason, change = self._consider_keyframe(frame)
            self._last_frame_change = change
            self._admission_counts[reason] += 1
            if admitted:
                self._keyframes.append(frame)
            n_kf = len(self._keyframes)

        if admitted:
            if n_kf == 1 or n_kf % 25 == 0:
                print(f'[OnlineESMapper] Keyframes: {n_kf}/{n} raw '
                      f'(change={change:.3f}, threshold={self.settings.keyframe_threshold:.2f})',
                      flush=True)
            self._maybe_start_rebuild()


    def _rebuild_worker(self):
        t0 = time.perf_counter()
        context = None
        results = None
        new_globals = None
        next_online_state = None
        old_globals = None
        old_online_state = None
        resume_state = None
        frames_snap = None
        # A result may only be published to the episode that produced its input.
        _ep_scene = self._scene_id
        _ep_id    = self._episode_id
        self._model_lock.acquire()
        try:
            with self._buf_lock:
                # Append-only ordering preserves the encoder's frame indices.
                frames_snap = list(self._keyframes)
                # Pair the map snapshot with its navigation observation counter.
                n_total_frames_snap = self._n_total_frames

            n = len(frames_snap)
            print(f'[OnlineESMapper] Rebuild start: {n} keyframes / '
                  f'raw_frame={n_total_frames_snap} '
                  f'(n_gaussians={self.n_gaussians})...', flush=True)

            context = self._build_context(frames_snap)

            # Episode identifier for ES save path (must end with '/')
            scene_ep = f'{self._scene_id}_{self._episode_id}/'

            # Keep the persistent state on CPU between rebuilds. The ES encoder
            # mutates it in place, so it only lives on GPU during this forward.
            resume_state = _move_tree_to_device(self._online_state, self.device)
            _log_cuda_memory(self.device, 'before encoder forward')

            results = self.model.update(
                context, scene_key=scene_ep, resume_state=resume_state,
            )
            _log_cuda_memory(self.device, 'after encoder forward')

            next_online_state = _move_tree_to_device(results.get('online_state'), 'cpu')
            new_globals = _move_tree_to_device(results.get('globals'), 'cpu')
            if new_globals is not None:
                n_valid = int(new_globals.valid[0].sum().item())
                # Discard results if the episode changed during inference.
                if self._scene_id != _ep_scene or self._episode_id != _ep_id:
                    print(f'[OnlineESMapper] Rebuild discarded: episode changed '
                          f'({_ep_scene}/{_ep_id} -> {self._scene_id}/{self._episode_id})',
                          flush=True)
                else:
                    # Persist CPU-resident online state for the next incremental
                    # rebuild, and drop stale references so CUDA tensors can die.
                    old_online_state = self._online_state
                    self._online_state = next_online_state
                    if old_online_state is next_online_state:
                        old_online_state = None
                    with self._globals_lock:
                        old_globals = self._globals
                        self._globals    = new_globals
                        self.n_gaussians = n_valid
                    self._last_fused_n = n
                    if old_globals is new_globals:
                        old_globals = None
                    dt = (time.perf_counter() - t0) * 1000
                    print(f'[OnlineESMapper] Rebuild done: {n} keyframes, '
                          f'{n_valid} Gaussians, {dt:.0f} ms', flush=True)
                if self.settings.clip_coverage_diagnostic:
                    try:
                        _idx = new_globals.clip_features_2d['idx'][
                            :, new_globals.valid[0], :3
                        ]
                        _has_clip = int((_idx != -1).any(dim=-1).sum())
                        print(f'[OnlineESMapper] CLIP coverage: {_has_clip}/{n_valid} '
                              f'({100*_has_clip/max(n_valid,1):.1f}%)', flush=True)
                    except Exception:
                        pass
                if self._artifacts is not None:
                    try:
                        self._artifacts.on_rebuild(
                            scene_id=self._scene_id, episode_id=self._episode_id,
                            globals_state=new_globals, n_keyframes=n,
                            n_raw_frames=n_total_frames_snap,
                        )
                    except Exception as exc:
                        print(f'[OnlineMapper] Map snapshot export failed: {exc}', flush=True)

            else:
                print(f'[OnlineESMapper] Rebuild returned no globals', flush=True)

        except Exception as e:
            import traceback
            print(f'[OnlineESMapper] Rebuild ERROR: {e}', flush=True)
            traceback.print_exc()
        finally:
            # Drop transient CUDA-heavy rebuild objects before marking the worker
            # idle. Current self._globals/self._online_state are CPU-resident by
            # design; context/results/resume_state should not pin GPU memory.
            _release_encoder_transient_caches(self.model.encoder, offload=self.settings.offload_fastsam_after_rebuild)
            del context, results, new_globals, next_online_state, old_globals, old_online_state, resume_state, frames_snap
            _release_cuda_cache(self.device, 'rebuild cleanup')
            self._model_lock.release()
            with self._rebuild_lock:
                self._rebuild_active = False
                restart = self._rebuild_pending and not self._resetting
                self._rebuild_pending = False
            if restart:
                self._maybe_start_rebuild()


    def _build_context(self, frames):
        return self.model.build_context(frames, device=self.device)


    def query(self, category: str, top_k: int = None) -> list:
        """Return top-k Habitat-world positions for `category`."""
        if self._closed:
            raise RuntimeError("Mapper is closed")
        with self._model_lock:
            with self._globals_lock:
                g = self._globals
            if g is None:
                return []
            top_k = top_k or self.top_k
            return query_category(
                self.model.encoder, g, category, top_k, self.device, settings=self.settings
            )


    def query_with_records(self, category: str, top_k: int = None):
        """Return legacy positions plus a backward-compatible metadata sidecar."""
        if self._closed:
            raise RuntimeError("Mapper is closed")
        with self._model_lock:
            with self._globals_lock:
                g = self._globals
            if g is None:
                return [], []
            top_k = top_k or self.top_k
            return query_category(
                self.model.encoder,
                g,
                category,
                top_k,
                self.device, settings=self.settings,
                return_records=True,
                map_revision=self._last_fused_n,
            )


    def query_batch(self, categories: list, top_k: int = None) -> dict:
        """Batch query all categories in one feat_dict pass."""
        if self._closed:
            raise RuntimeError("Mapper is closed")
        with self._model_lock:
            with self._globals_lock:
                g = self._globals
            if g is None:
                return {cat: [] for cat in categories}
            top_k = top_k or self.top_k
            return query_categories(
                self.model.encoder, g, categories, top_k, self.device, settings=self.settings
            )


    def query_batch_with_records(self, categories: list, top_k: int = None):
        if self._closed:
            raise RuntimeError("Mapper is closed")
        with self._model_lock:
            with self._globals_lock:
                g = self._globals
            if g is None:
                empty = {cat: [] for cat in categories}
                return empty, dict(empty)
            top_k = top_k or self.top_k
            return query_categories(
                self.model.encoder,
                g,
                categories,
                top_k,
                self.device, settings=self.settings,
                return_records=True,
                map_revision=self._last_fused_n,
            )


    def get_status(self) -> dict:
        with self._globals_lock:
            n_g = self.n_gaussians
            n_f = self.n_frames
        with self._rebuild_lock:
            rebuilding = self._rebuild_active
            pending = self._rebuild_pending
        with self._buf_lock:
            n_kf = len(self._keyframes)
            counts = dict(self._admission_counts)
            last_change = self._last_frame_change
        return {
            'n_frames':    n_f,
            'n_raw_frames': self._n_total_frames,
            'n_keyframes': n_kf,
            'last_fused_keyframes': self._last_fused_n,
            'n_gaussians': n_g,
            'rebuilding':  rebuilding,
            'rebuild_pending': pending,
            'last_frame_change': last_change,
            'admission_counts': counts,
            'scene_id':    self._scene_id,
            'episode_id':  self._episode_id,
        }


    def observe(self, observation):
        self.add_live_frame(observation.rgb, observation.depth_m,
                            habitat_pose_to_c2w(observation.habitat_pose),
                            observation.normalized_intrinsics)

    def finish_episode(self):
        return self.flush_episode()

    def close(self):
        if self._closed:
            return
        self.reset_episode('', '')
        self._closed = True
        self.model.close()
        _release_cuda_cache(self.device, 'mapper close')
