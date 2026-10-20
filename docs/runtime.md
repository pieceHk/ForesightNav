# Runtime workers

## Root launcher

[run.py](../run.py) reads an experiment preset and `runtime.local.yaml`. [The run planner](../runtime/plan.py) validates the complete dataset and its prepared identity index before selecting scenes and episodes. `--limit-episodes` limits complete ordered episodes across the selection. `--episode` uses actual runtime IDs within one `--scene`.

`--check` verifies inputs and resources without creating an output directory or loading models. A run freezes shared method/evaluation settings, model input/camera specifications and selected identities in `launch.json`; `resolved_config.yaml` provides a readable copy. [The launcher](../runtime/launch.py) directly starts mapping, navigation and resource-monitor processes, records their exit codes and aggregates the selected complete episodes. A failed worker prevents a completed status.

Both workers accept `--run-plan`. [The native factories](../runtime/factories.py) compose services from that saved configuration. Process startup supplies owned import paths, cache locations, GPU visibility and seed controls. Ambient settings from the retained experiment launcher do not override the native method configuration. Existing run directories cannot be resumed in place.

The launcher copies both environment locks into the run directory. Each native worker writes an [environment inventory](../runtime/environment.py) with its actual interpreter, package versions and locations, source search paths and selected GPU. Python user-site packages are disabled at process startup. These files describe installed metadata; the [installation diagnostic](installation.md#verify) separately checks native module origins and operations.

The launcher starts [runtime.mapping_worker](../runtime/mapping_worker.py) in the mapping environment. Navigation uses [MappingClient](../runtime/mapping_client.py) to stream RGB-D frames and query the published map. Both use [runtime.protocol](../runtime/protocol.py).

## Backend ownership

`MappingServer(mapper, top_k=...)` receives a prepared mapping backend. `run_worker(server, socket_path)` owns the listener and client connections, then closes the backend after active requests finish. The default CLI assembles the current EmbodiedSplat mapper through the retained resource factory.

A mapping backend supplies `reset_episode`, `add_live_frame`, `query_with_records`, `query_batch_with_records`, `get_status`, `flush_episode` and `close`. Candidate coordinates and records follow the [query contract](components.md#query).

`MappingClient` receives a socket path and `task_reader(scene_id, episode_id)`. The released data reader is [load_runtime_tasks](../experiments/dataset.py); another dataset can supply its own reader. The client retains ordered task indices when duplicate categories share one query result.

Importing the client, worker or protocol does not load models or change environment variables. Worker startup applies [thread controls](../runtime/bootstrap.py) and model cache paths before model imports. The [installation guide](installation.md) covers both independent environments and explicit resource preparation.

## Navigation assembly

The launcher runs [runtime.navigation_worker](../runtime/navigation_worker.py) directly in the navigation environment. `NavigationRequest` supplies dataset/output paths, explicit model resources, navigation settings, episode range and identity-bound planning settings. `run_navigation(request)` composes the mapping client, Habitat adapter, Agent, cooldown and planning service, then calls the native Runner.

[The Habitat factory](../adapters/goat.py) receives categories and Detic resources. [The configuration adapter](../adapters/habitat_config.py) combines the retained sensor and Agent presets. They share the same observation and evaluation boundaries described in [components](components.md).

The worker registers each constructed resource for cleanup. It closes the Agent's navigation backend, Habitat and mapping connection on completion or failure. SIGINT and SIGTERM unwind this scope. The launcher owns the navigation process group and records its command, PID, scene and exit code in `navigation_<scene>_process.json`.

[TrajectoryEnvironment](../experiments/trajectory.py) observes the existing environment calls around the Runner. It adds no sensor reads and returns the underlying observation/action results. Each accepted action records its preceding Habitat world pose, episode/task/step identity, action ID/name and STOP reason under `results_*/.../episodes/`. Episode IDs are reversibly encoded as path components; their original values remain in `identity.json`. A failed step leaves the accepted-action prefix for diagnosis.

The retained command-line flags and environment controls are translated once by [legacy_navigation_run](../runtime/legacy_navigation_run.py). The root launcher's `--run-plan` path bypasses that resolver. The native request, service assembly and Runner receive explicit settings and model resources. Habitat and SuperGlue sources are prepared separately and checked before worker startup.

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
