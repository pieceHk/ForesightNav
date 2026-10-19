"""Compatibility entrypoint for the PBSP mapping worker."""

from runtime.bootstrap import configure_runtime_threads


def main():
    from runtime.mapping_worker import main as run_worker

    run_worker()


if __name__ == "__main__":
    main()
