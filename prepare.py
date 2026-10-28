#!/usr/bin/env python3
"""Prepare PBSP data, environments, models and machine configuration."""

import argparse
import runpy
import sys

COMMANDS = {
    "data": "prepare_data",
    "objectnav": "prepare_objectnav",
    "envs": "setup_envs",
    "models": "prepare_models",
    "runtime": "prepare_runtime",
    "check": "check_install",
    "offline": "export_offline",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    args = parser.parse_args(sys.argv[1:2])
    remaining = sys.argv[2:]
    sys.argv = [f"prepare.py {args.command}", *remaining]
    runpy.run_module(f"experiments.{COMMANDS[args.command]}", run_name="__main__")


if __name__ == "__main__":
    main()
