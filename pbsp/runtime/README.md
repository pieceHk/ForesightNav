# Runtime

`run.py` resolves the shared experiment and machine configuration into a run plan. `plan.py` validates dataset identities and selection. `resources.py` locates the explicit model and dependency inputs. `factories.py` assembles the default mapping, query and navigation components.

`launch.py` starts a mapping worker and a navigation worker using their separate interpreters. `bootstrap.py` sets each process's search paths, device and caches. `mapping_client.py` and `protocol.py` exchange versioned observations and candidates through a local socket. Both worker entries require the saved run plan.

The navigation worker creates the environment, Agent, evaluator and experiment Runner. The mapping worker owns the online map and query service. Episode reset clears the map; task reset keeps it. Worker failures propagate to the launcher and leave an incomplete status with logs.

`environment.py`, `source_snapshot.py` and the model/source manifests record the inputs used for the run. The monitor records device and process resources. Result layout and metric validation are described in [experiments](../../experiments/README.md).
