"""Versioned mapping queries and RGB-D streaming over a local socket."""

import base64, json, socket, os, time
from copy import deepcopy
from numbers import Integral
from typing import Dict, List, Mapping, Optional, Sequence

from pbsp.types import (
    BatchValidationStatus,
    CandidateBatch,
    CandidateRecord,
    FallbackReason,
    SCSPContractError,
)
from runtime.protocol import decode_message, receive_line, send_message


class MappingClient:
    """Versioned mapping candidates and RGB-D streaming over local IPC."""

    def __init__(
        self, socket_path, *, task_reader, top_k=10, connect_timeout=60.0, live_mode=True
    ):
        self.socket_path = socket_path
        self.top_k = top_k
        self._connect_timeout = connect_timeout
        self._task_reader = task_reader
        self._sock: Optional[socket.socket] = None
        self._buf = b''
        self._scene_id: Optional[str] = None
        self._episode_id: Optional[str] = None
        self._tasks: List = []
        self._goals_cache: Dict = {}
        self._candidate_records_cache: Dict[int, List[dict]] = {}
        self._candidate_batch: Optional[CandidateBatch] = None
        self._query_serial: int = 0
        self._last_fused_keyframes: Optional[int] = None
        self._last_version_gate: Optional[dict] = None
        self._query_success_by_task: Dict[int, bool] = {}
        self._query_error_by_task: Dict[int, str] = {}
        # ── timing state ────────────────────────────────────────────────
        self._t_push_ms: float = 0.0  # last push_frame round-trip
        self._t_query_ms: float = 0.0  # last get_goals round-trip
        self._t_get_all_goals_ms: float = 0.0  # last get_all_goals total
        self.current_snap: str = ''  # snap basename after last query
        self.n_live_frames: int = 0  # frames sent via push_live_frame
        self._live_mode: bool = live_mode

    # ── connection ──────────────────────────────────────────────────────

    def _ensure_connected(self):
        if self._sock is not None:
            return
        deadline = time.time() + self._connect_timeout
        while time.time() < deadline:
            if os.path.exists(self.socket_path):
                connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                try:
                    connection.connect(self.socket_path)
                except ConnectionRefusedError:
                    connection.close()
                except BaseException:
                    connection.close()
                    raise
                else:
                    self._sock = connection
                    self._buf = b''
                    print(f'[OnlineESProvider] Connected to {self.socket_path}', flush=True)
                    return
            time.sleep(0.5)
        raise RuntimeError(f'ES Mapper Server not available at {self.socket_path}')

    def _send_recv(self, msg: dict) -> dict:
        self._ensure_connected()
        send_message(self._sock, msg, versioned=True)
        line, self._buf = receive_line(self._sock, self._buf)
        return decode_message(line)

    # ── content file helper ────────────────────────────────────────────

    def _load_tasks_for_episode(self, scene_id: str, episode_id: str) -> List:
        return self._task_reader(scene_id, episode_id)

    # ── CandidateProvider interface ────────────────────────────────────

    def invalidate_cache(self) -> None:
        """Invalidate coordinates and metadata as one provider snapshot."""

        self._goals_cache = {}
        self._candidate_records_cache = {}
        self._candidate_batch = None
        self._last_fused_keyframes = None
        self._last_version_gate = None
        self._query_success_by_task = {}
        self._query_error_by_task = {}

    def reset_episode(self, scene_id: str, episode_id: str) -> None:
        self._scene_id = scene_id
        self._episode_id = episode_id
        self._tasks = self._load_tasks_for_episode(scene_id, episode_id)
        self.invalidate_cache()
        resp = self._send_recv(
            {'cmd': 'reset_episode', 'scene_id': scene_id, 'episode_id': episode_id}
        )
        if not resp.get('ok'):
            raise RuntimeError(f'reset_episode failed: {resp}')
        self.n_live_frames = 0
        print(
            f'[OnlineESProvider] Episode reset: {scene_id}/{episode_id} '
            f'({len(self._tasks)} tasks)',
            flush=True,
        )

    # ── live-mode frame push ────────────────────────────────────────────────

    def push_live_frame(
        self, rgb: 'np.ndarray', depth: 'np.ndarray', globalpose: 'np.ndarray'
    ) -> None:
        """Send one Habitat observation frame to the ES server for live mapping.

        rgb        : (H, W, 3) uint8
        depth      : (H, W)    float32, metres
        globalpose : (7,) float32 [qw, qx, qy, qz, px, py, pz]
        """
        import math as _math, io as _io
        import numpy as _np
        from PIL import Image as _PILImage

        # ── build c2w from globalpose (matches snapshot_mapper.py convention) ──
        _RUB2RDF = _np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=_np.float64)
        qw, qx, qy, qz = globalpose[:4]
        px, py, pz = globalpose[4:7]
        x, y, z, w = qw, qx, qy, qz
        R = _np.array(
            [
                [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
                [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
            ],
            dtype=_np.float64,
        )
        c2w = _np.eye(4, dtype=_np.float64)
        c2w[:3, :3] = R @ _RUB2RDF
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
        resp = self._send_recv(
            {
                'cmd': 'push_live_frame',
                'frame_id': self.n_live_frames,
                'rgb_jpeg': rgb_bytes,
                'depth_bytes': depth_bytes,
                'binary_encoding': 'base64',
                'depth_h': depth_h,
                'depth_w': depth_w,
                'extrinsic': c2w.tolist(),
                'intrinsic': [
                    [1.0, 0.0, 0.0],  # placeholder (server uses internal K)
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            }
        )
        self._t_push_ms = (time.perf_counter() - t0) * 1000
        self.n_live_frames += 1
        if not resp.get('ok'):
            print(f'[OnlineESProvider] push_live_frame warning: {resp}', flush=True)

    def get_goals(self, task_idx: int, category: str = None) -> List:
        if category is None:
            if task_idx < len(self._tasks):
                task = self._tasks[task_idx]
                category = (
                    task[0]
                    if isinstance(task, (list, tuple))
                    else task.get('object_category', '')
                )
        if not category:
            self._query_success_by_task[int(task_idx)] = False
            self._query_error_by_task[int(task_idx)] = 'missing task category'
            return []
        if task_idx in self._goals_cache:
            return self._goals_cache[task_idx]
        # A partial per-task refresh cannot leave a previous complete batch
        # looking current.
        self._candidate_batch = None
        self._last_fused_keyframes = None
        self._last_version_gate = None
        t0 = time.perf_counter()
        try:
            resp = self._send_recv(
                {
                    'cmd': 'query',
                    'task_idx': task_idx,
                    'category': category,
                    'top_k': self.top_k,
                }
            )
        except Exception as exc:
            self._t_query_ms = (time.perf_counter() - t0) * 1000
            self._query_success_by_task[int(task_idx)] = False
            self._query_error_by_task[int(task_idx)] = f'{type(exc).__name__}: {exc}'
            return []
        self._t_query_ms = (time.perf_counter() - t0) * 1000
        if not isinstance(resp, Mapping) or not resp.get('ok'):
            self._query_success_by_task[int(task_idx)] = False
            self._query_error_by_task[int(task_idx)] = str(
                resp.get('error', 'query response was not ok')
                if isinstance(resp, Mapping)
                else 'query response must be a mapping'
            )
            return []
        self._query_serial += 1
        candidates = resp.get('candidates', [])
        self._goals_cache[task_idx] = candidates
        raw_records = resp.get('candidate_records')
        if 'candidate_records' not in resp:
            self._candidate_records_cache[task_idx] = []
            self._query_error_by_task[int(task_idx)] = (
                'query response is missing candidate_records'
            )
        elif not self._is_record_sequence(raw_records):
            self._candidate_records_cache[task_idx] = []
            self._query_error_by_task[int(task_idx)] = (
                'candidate_records must be a sequence of mappings'
            )
        else:
            self._candidate_records_cache[task_idx] = deepcopy(list(raw_records))
            self._query_error_by_task.pop(int(task_idx), None)
        self._query_success_by_task[int(task_idx)] = True
        self.current_snap = resp.get('snap', '')
        return candidates

    def _task_category(self, task) -> str:
        if isinstance(task, (list, tuple)):
            return task[0]
        elif isinstance(task, dict):
            return task.get('object_category', task.get('category', ''))
        return str(task)

    def get_all_goals(self) -> Dict[int, List]:
        """Query distinct categories together and retain ordered task identity.

        Falls back to sequential per-task queries on protocol error.
        """
        if not self._tasks:
            return {}

        task_categories = [self._task_category(t) for t in self._tasks]
        unique_cats = list(dict.fromkeys(c for c in task_categories if c))

        t_start = time.perf_counter()
        pre_status, pre_status_error = self._capture_status_view()
        n_unique = len(unique_cats)
        n_tasks = len(self._tasks)
        print(
            f'[OnlineESProvider] get_all_goals: batch querying {n_unique} unique categories ({n_tasks} tasks)...',
            flush=True,
        )

        try:
            resp = self._send_recv(
                {'cmd': 'batch_query', 'categories': unique_cats, 'top_k': self.top_k}
            )
        except Exception as e:
            print(
                f'[OnlineESProvider] batch_query failed ({e}), falling back to sequential',
                flush=True,
            )
            return self._get_all_goals_sequential(
                pre_status=pre_status,
                pre_status_error=pre_status_error,
            )

        if not isinstance(resp, Mapping) or not resp.get('ok'):
            print(
                f'[OnlineESProvider] batch_query not ok, falling back to sequential', flush=True
            )
            return self._get_all_goals_sequential(
                pre_status=pre_status,
                pre_status_error=pre_status_error,
            )

        self._query_serial += 1
        query_serial = self._query_serial
        cat_results = resp.get('results')
        cat_records = resp.get('candidate_records_by_category')
        payload_errors = []
        if not isinstance(cat_results, Mapping):
            payload_errors.append('results must be a category mapping')
            cat_results = {}
        if not isinstance(cat_records, Mapping):
            payload_errors.append('candidate_records_by_category must be a category mapping')
            cat_records = {}
        self.current_snap = resp.get('snap', self.current_snap)

        result = {}
        records_by_task = {}
        for task_idx, cat in enumerate(task_categories):
            if cat not in cat_results:
                payload_errors.append(f'results is missing category {cat!r}')
            if cat not in cat_records:
                payload_errors.append(
                    f'candidate_records_by_category is missing category {cat!r}'
                )
            candidates = cat_results.get(cat, [])
            raw_records = cat_records.get(cat, [])
            if not self._is_candidate_sequence(candidates):
                payload_errors.append(f'candidates for category {cat!r} must be a sequence')
            result[task_idx] = candidates
            self._goals_cache[task_idx] = candidates
            if self._is_record_sequence(raw_records):
                copied_records = deepcopy(list(raw_records))
            else:
                copied_records = []
                payload_errors.append(
                    f'candidate records for category {cat!r} must be a ' 'sequence of mappings'
                )
            records_by_task[task_idx] = copied_records
            self._candidate_records_cache[task_idx] = copied_records
            # Retain rounded candidate coordinates in experiment diagnostics.
            try:
                _coords_str = [[round(v, 3) for v in c] for c in candidates]
                candidate_count = len(candidates)
            except (OverflowError, TypeError, ValueError):
                _coords_str = '<invalid-coordinate-payload>'
                candidate_count = '<invalid>'
            print(
                f'  task {task_idx} [{cat}]: {candidate_count} candidates ' f'{_coords_str}',
                flush=True,
            )

        has_atomic_record_revision = any(records_by_task.values())
        self._publish_candidate_batch(
            task_categories=task_categories,
            positions_by_task=result,
            records_by_task=records_by_task,
            query_serial=query_serial,
            payload_error='; '.join(dict.fromkeys(payload_errors)) or None,
            pre_status=pre_status,
            pre_status_error=pre_status_error,
            require_stable_status=not has_atomic_record_revision,
        )
        self._t_get_all_goals_ms = (time.perf_counter() - t_start) * 1000
        return result

    @staticmethod
    def _is_record_sequence(value) -> bool:
        return (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes))
            and all(isinstance(record, Mapping) for record in value)
        )

    @staticmethod
    def _is_candidate_sequence(value) -> bool:
        return isinstance(value, Sequence) and not isinstance(value, (str, bytes))

    @staticmethod
    def _nonnegative_int(value, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f'{name} must be a non-negative integer')
        result = int(value)
        if result < 0:
            raise ValueError(f'{name} must be a non-negative integer')
        return result

    @staticmethod
    def _status_view(response: dict) -> dict:
        if not isinstance(response, Mapping) or not response.get('ok', False):
            raise ValueError('mapper status response is unavailable')
        view = dict(response)
        live = response.get('live')
        if isinstance(live, Mapping):
            view.update(live)
        return view

    def _read_status_view(self) -> dict:
        return self._status_view(self.status())

    def _capture_status_view(self):
        try:
            return self._read_status_view(), None
        except Exception as exc:
            return None, f'{type(exc).__name__}: {exc}'

    def _validate_status_identity(self, status: Mapping) -> None:
        for name, expected in (
            ('scene_id', self._scene_id),
            ('episode_id', self._episode_id),
        ):
            observed = status.get(name)
            if observed is None:
                raise ValueError(f'mapper status is missing {name}')
            if str(observed) != str(expected):
                raise RuntimeError(f'{name} mismatch: expected {expected!r}, got {observed!r}')

    def _invalid_candidate_batch(
        self,
        task_categories: Sequence[str],
        query_serial: int,
        reason: str,
        revision: Optional[int] = None,
    ) -> CandidateBatch:
        safe_categories = tuple(
            str(category).strip() or f'<invalid-task-{task_idx}>'
            for task_idx, category in enumerate(task_categories)
        )
        safe_revision = 0 if revision is None else int(revision)
        scene_id = str(self._scene_id).strip() if self._scene_id is not None else ''
        episode_id = str(self._episode_id).strip() if self._episode_id is not None else ''
        return CandidateBatch(
            scene_id=scene_id or '<unset-scene>',
            episode_id=episode_id or '<unset-episode>',
            map_revision=safe_revision,
            last_fused_keyframes=safe_revision,
            query_serial=query_serial,
            ordered_categories=safe_categories,
            positions_by_task=tuple((task_idx, ()) for task_idx in range(len(safe_categories))),
            records_by_task=tuple((task_idx, ()) for task_idx in range(len(safe_categories))),
            validation_status=BatchValidationStatus.INVALID,
            validation_reason=reason,
        )

    def _publish_candidate_batch(
        self,
        *,
        task_categories: Sequence[str],
        positions_by_task: Mapping[int, Sequence],
        records_by_task: Mapping[int, Sequence[Mapping]],
        query_serial: int,
        payload_error: Optional[str] = None,
        pre_status: Optional[Mapping] = None,
        pre_status_error: Optional[str] = None,
        require_stable_status: bool = False,
    ) -> CandidateBatch:
        revision = None
        self._last_fused_keyframes = None
        try:
            status = self._read_status_view()
            self.current_snap = status.get('snap', self.current_snap)
            revision = self._nonnegative_int(
                status.get('last_fused_keyframes'), 'last_fused_keyframes'
            )
            self._last_fused_keyframes = revision
            try:
                self._validate_status_identity(status)
            except RuntimeError as exc:
                raise RuntimeError(f'episode_mismatch: {exc}') from exc
        except RuntimeError as exc:
            if str(exc).startswith('episode_mismatch:'):
                reason = str(exc)
            else:
                reason = f'provider_status_unavailable: {exc}'
            self._candidate_batch = self._invalid_candidate_batch(
                task_categories, query_serial, reason, revision
            )
            return self._candidate_batch
        except Exception as exc:
            self._candidate_batch = self._invalid_candidate_batch(
                task_categories,
                query_serial,
                f'provider_status_unavailable: {exc}',
                revision,
            )
            return self._candidate_batch

        if require_stable_status:
            if pre_status is None:
                self._candidate_batch = self._invalid_candidate_batch(
                    task_categories,
                    query_serial,
                    'provider_status_unavailable: pre-query status unavailable'
                    + (f' ({pre_status_error})' if pre_status_error else ''),
                    revision,
                )
                return self._candidate_batch
            try:
                pre_revision = self._nonnegative_int(
                    pre_status.get('last_fused_keyframes'),
                    'pre_query.last_fused_keyframes',
                )
                try:
                    self._validate_status_identity(pre_status)
                except RuntimeError as exc:
                    raise RuntimeError(f'episode_mismatch: {exc}') from exc
            except RuntimeError as exc:
                self._candidate_batch = self._invalid_candidate_batch(
                    task_categories, query_serial, str(exc), revision
                )
                return self._candidate_batch
            except Exception as exc:
                self._candidate_batch = self._invalid_candidate_batch(
                    task_categories,
                    query_serial,
                    f'provider_status_unavailable: {exc}',
                    revision,
                )
                return self._candidate_batch
            if pre_revision != revision:
                self._candidate_batch = self._invalid_candidate_batch(
                    task_categories,
                    query_serial,
                    'stale_version_token: mapper revision changed across query '
                    f'({pre_revision} -> {revision})',
                    revision,
                )
                return self._candidate_batch

        try:
            if payload_error:
                raise SCSPContractError(payload_error)
            positions = []
            records = []
            for task_idx, category in enumerate(task_categories):
                task_positions = positions_by_task[task_idx]
                raw_records = records_by_task[task_idx]
                positions.append((task_idx, tuple(task_positions)))
                converted = []
                for record_idx, raw_record in enumerate(raw_records):
                    if 'task_idx' in raw_record and raw_record['task_idx'] != task_idx:
                        raise SCSPContractError(
                            f'task {task_idx} record {record_idx} has mismatched task_idx'
                        )
                    if 'category' in raw_record and str(raw_record['category']) != str(
                        category
                    ):
                        raise SCSPContractError(
                            f'task {task_idx} record {record_idx} has mismatched category'
                        )
                    missing = [
                        name
                        for name in (
                            'position',
                            'raw_rank',
                            'semantic_score',
                            'support_count',
                            'support_score',
                            'map_revision',
                        )
                        if name not in raw_record
                    ]
                    if missing:
                        raise SCSPContractError(
                            f'task {task_idx} record {record_idx} is missing {missing}'
                        )
                    converted.append(
                        CandidateRecord(
                            task_idx=task_idx,
                            category=str(category),
                            position=raw_record['position'],
                            raw_rank=raw_record['raw_rank'],
                            semantic_score=raw_record['semantic_score'],
                            support_count=raw_record['support_count'],
                            support_score=raw_record['support_score'],
                            map_revision=raw_record['map_revision'],
                        )
                    )
                records.append((task_idx, tuple(converted)))
            self._candidate_batch = CandidateBatch(
                scene_id=str(self._scene_id),
                episode_id=str(self._episode_id),
                map_revision=revision,
                last_fused_keyframes=revision,
                query_serial=query_serial,
                ordered_categories=tuple(str(value) for value in task_categories),
                positions_by_task=tuple(positions),
                records_by_task=tuple(records),
            )
        except (KeyError, OverflowError, TypeError, ValueError) as exc:
            self._candidate_batch = self._invalid_candidate_batch(
                task_categories,
                query_serial,
                f'invalid_candidate_batch: {exc}',
                revision,
            )
        return self._candidate_batch

    def get_all_candidate_records(self) -> Dict[int, List[dict]]:
        """Return metadata only when the latest complete batch is trustworthy."""

        if (
            self._candidate_batch is None
            or self._candidate_batch.validation_status is BatchValidationStatus.INVALID
        ):
            return {}
        return deepcopy(self._candidate_records_cache)

    def get_candidate_batch(self) -> Optional[CandidateBatch]:
        """Return the immutable batch associated with the latest full query."""

        return self._candidate_batch

    def get_ordered_categories(self):
        return tuple(self._task_category(task) for task in self._tasks)

    @property
    def query_serial(self) -> int:
        return self._query_serial

    @property
    def last_fused_keyframes(self) -> Optional[int]:
        return self._last_fused_keyframes

    def check_candidate_batch_status(
        self,
        batch: Optional[CandidateBatch] = None,
        *,
        scene_id: Optional[str] = None,
        episode_id: Optional[str] = None,
        ordered_categories: Optional[Sequence[str]] = None,
        status_response: Optional[dict] = None,
    ) -> dict:
        """Check snapshot identity and map revision before candidate dispatch."""

        candidate_batch = self._candidate_batch if batch is None else batch
        event = {
            'would_invalidate': True,
            'reason': FallbackReason.INVALID_CANDIDATE_BATCH.value,
            'scene_id': str(scene_id if scene_id is not None else self._scene_id),
            'episode_id': str(episode_id if episode_id is not None else self._episode_id),
            'query_serial': None,
            'batch_map_revision': None,
            'status_last_fused_keyframes': None,
            'status_rebuilding': None,
        }
        if (
            not isinstance(candidate_batch, CandidateBatch)
            or candidate_batch.validation_status is BatchValidationStatus.INVALID
        ):
            self._last_version_gate = event
            return dict(event)

        event['query_serial'] = candidate_batch.query_serial
        event['batch_map_revision'] = candidate_batch.map_revision
        try:
            status = self._status_view(
                self.status() if status_response is None else status_response
            )
            revision = self._nonnegative_int(
                status.get('last_fused_keyframes'), 'last_fused_keyframes'
            )
            rebuilding = status.get('rebuilding')
            if not isinstance(rebuilding, bool):
                raise ValueError('rebuilding must be a bool')
            for name in ('scene_id', 'episode_id'):
                if status.get(name) is None:
                    raise ValueError(f'mapper status is missing {name}')
        except Exception:
            event['reason'] = FallbackReason.PROVIDER_STATUS_UNAVAILABLE.value
            self._last_version_gate = event
            return dict(event)

        event['status_last_fused_keyframes'] = revision
        event['status_rebuilding'] = rebuilding
        expected_scene = event['scene_id']
        expected_episode = event['episode_id']
        expected_categories = (
            candidate_batch.ordered_categories
            if ordered_categories is None
            else tuple(str(value) for value in ordered_categories)
        )
        if rebuilding:
            event['reason'] = FallbackReason.REBUILDING.value
        elif (
            candidate_batch.scene_id != expected_scene
            or candidate_batch.episode_id != expected_episode
            or candidate_batch.ordered_categories != expected_categories
            or str(status.get('scene_id')) != expected_scene
            or str(status.get('episode_id')) != expected_episode
        ):
            event['reason'] = FallbackReason.EPISODE_MISMATCH.value
        elif (
            self._candidate_batch is None
            or self._candidate_batch.validation_status is BatchValidationStatus.INVALID
            or self._candidate_batch.version_token != candidate_batch.version_token
        ):
            event['reason'] = FallbackReason.PROVIDER_TOKEN_CHANGED.value
        elif revision != candidate_batch.map_revision:
            event['reason'] = FallbackReason.STALE_VERSION_TOKEN.value
        else:
            event['would_invalidate'] = False
            event['reason'] = FallbackReason.NONE.value
        self._last_version_gate = event
        return dict(event)

    def get_last_version_gate(self) -> Optional[dict]:
        return None if self._last_version_gate is None else dict(self._last_version_gate)

    def _get_all_goals_sequential(
        self,
        *,
        pre_status: Optional[Mapping] = None,
        pre_status_error: Optional[str] = None,
    ) -> Dict[int, List]:
        """Fallback: sequential per-task queries (original behavior)."""
        t_start = time.perf_counter()
        self.invalidate_cache()
        if pre_status is None and pre_status_error is None:
            pre_status, pre_status_error = self._capture_status_view()
        result = {}
        for task_idx, task in enumerate(self._tasks):
            category = self._task_category(task)
            if not category:
                continue
            candidates = self.get_goals(task_idx, category)
            result[task_idx] = candidates
            candidate_count = (
                len(candidates) if self._is_candidate_sequence(candidates) else '<invalid>'
            )
            print(
                f'  task {task_idx} [{category}]: ' f'{candidate_count} candidates',
                flush=True,
            )
        payload_errors = [
            f'task {task_idx} query failed: '
            f'{self._query_error_by_task.get(task_idx, "unknown error")}'
            for task_idx in range(len(self._tasks))
            if not self._query_success_by_task.get(task_idx, False)
        ]
        payload_errors.extend(
            self._query_error_by_task[task_idx]
            for task_idx in sorted(self._query_error_by_task)
            if self._query_success_by_task.get(task_idx, False)
        )
        has_unversioned_empty_query = any(
            not self._candidate_records_cache.get(task_idx)
            for task_idx in range(len(self._tasks))
        )
        self._publish_candidate_batch(
            task_categories=[self._task_category(task) for task in self._tasks],
            positions_by_task=result,
            records_by_task=self._candidate_records_cache,
            query_serial=self._query_serial,
            payload_error='; '.join(payload_errors) or None,
            pre_status=pre_status,
            pre_status_error=pre_status_error,
            require_stable_status=has_unversioned_empty_query,
        )
        self._t_get_all_goals_ms = (time.perf_counter() - t_start) * 1000
        return result

    def push_frame(self, frame_id: int, **kwargs) -> None:
        """Notify the snapshot backend of a replay frame."""
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
