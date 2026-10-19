# Mapping runtime

The launcher starts [runtime.mapping_worker](../runtime/mapping_worker.py) in the mapping environment. Navigation uses [MappingClient](../runtime/mapping_client.py) to stream RGB-D frames and query the published map. Both use [runtime.protocol](../runtime/protocol.py).

## Backend ownership

`MappingServer(mapper, top_k=...)` receives a prepared mapping backend. `run_worker(server, socket_path)` owns the listener and client connections, then closes the backend after active requests finish. The default CLI assembles the current EmbodiedSplat mapper through the retained resource factory.

A mapping backend supplies `reset_episode`, `add_live_frame`, `query_with_records`, `query_batch_with_records`, `get_status`, `flush_episode` and `close`. Candidate coordinates and records follow the [query contract](components.md#query).

`MappingClient` receives a socket path and `task_reader(scene_id, episode_id)`. The released data reader is [load_runtime_tasks](../experiments/dataset.py); another dataset can supply its own reader. The client retains ordered task indices when duplicate categories share one query result.

Importing the client, worker or protocol does not load models or change environment variables. Worker startup applies [thread controls](../runtime/bootstrap.py) and model cache paths before model imports. Navigation startup and the root CLI remain to be migrated.

## Messages

Each frame is one UTF-8 JSON object followed by a newline. Native requests and responses include `protocol_version: 1`; unversioned requests remain compatible with the retained launcher. An explicit unsupported version returns an error. Responses include `ok` and the operation's payload.

| Command | Request fields | Result |
| --- | --- | --- |
| `reset_episode` | `scene_id`, `episode_id` | Reset episode-local map state |
| `push_live_frame` | Frame ID, RGB-D data, pose and intrinsics | Accept a frame and return map status |
| `query` | Task index, category, optional `top_k` | Coordinates and versioned candidate records |
| `batch_query` | Distinct categories, optional `top_k` | Results and records by category |
| `status` | None | Episode identity, map revision and rebuild state |
| `flush` | None | Complete pending fusion and export final artifacts |
| `shutdown` | None | Flush, acknowledge, stop serving and release resources |

The retained `push_frame` command requires a supplied snapshot backend. The default worker provides online mapping.

RGB uses base64 JPEG at quality 92. Depth uses base64 float32 bytes with explicit `depth_h` and `depth_w`. Extrinsics and intrinsics are JSON arrays. The current pose/axis conversion and frame payloads preserve the validated implementation; their numerical parity checks use the frozen source.

## Completion and failures

A shutdown request lets the worker return through normal cleanup. SIGINT and SIGTERM also stop the listener and close the mapper. The worker waits for active connection handlers and removes its socket. It refuses to replace an active listener or a regular file at that path.

The launcher records `mapping_process.json` with the command, PID and exit code. If the experiment body completes but the mapping worker exits with an error, the launcher reports failure. This record complements navigation results and map artifacts.

Socket tests cover actual client/server requests, fragmented messages, duplicate categories, stale revisions, sequential query fallback and process termination. A real two-environment episode verifies the simulator-to-mapper path separately; see [validation](migration.md).
