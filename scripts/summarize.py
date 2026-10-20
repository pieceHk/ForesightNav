#!/usr/bin/env python3
"""Recompute and compare completed native PBSP runs, or merge disjoint shards."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.metrics import (
    FRAME_COUNT_KEY, FRAME_TIME_SUM_KEY, STEPS_KEY,
    merge_episode_metric_files, summarize_episode, summarize_records,
)
from pbsp.types import EpisodeIdentity


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def read_json(path):
    def reject(value):
        raise ValueError(f"Non-finite JSON value in {path}: {value}")
    return json.loads(Path(path).read_text(), parse_constant=reject)


def timing_records(paths, identities):
    grouped, samples, seen = {}, [], set()
    for path in sorted(paths):
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                key = (row["scene_id"], row["episode_id"])
                task, frame, elapsed = int(row["subtask_index"]), int(row["frame_index"]), float(row["dt_total_ms"])
                identity = identities.get(key)
                if (identity is None or not 0 <= task < len(identity.ordered_categories)
                        or frame < 1 or not math.isfinite(elapsed) or elapsed < 0):
                    raise ValueError(f"Timing row has an invalid task/frame identity: {path}")
                token = (*key, frame)
                if token in seen:
                    raise ValueError(f"Duplicate timing frame: {token}")
                seen.add(token)
                grouped.setdefault(key, []).append((frame, task, elapsed))
                samples.append(elapsed)
    return grouped, samples


@dataclass
class ValidatedRun:
    directory: Path
    identities: dict
    records: dict
    frame_times: list
    protocol: dict
    row: dict


def coverage(identities, dataset, manifest):
    observed = {}
    for scene, episode in identities:
        observed[scene] = observed.get(scene, 0) + 1
    expected = {row["scene_id"]: row["mon_episode_count"] for row in manifest["scene_summary"]}
    if not set(observed) <= set(expected) or any(count > expected[scene] for scene, count in observed.items()):
        raise ValueError("Selected episode coverage exceeds the dataset manifest")
    full = observed == expected and len(identities) == dataset["expected_episodes"]
    return {"kind": "full_dataset" if full else "selected_episodes", "completed_scenes": len(observed),
            "completed_episodes": len(identities), "dataset_episodes": dataset["expected_episodes"]}


def validate_run(directory):
    directory = Path(directory).resolve(strict=True)
    status = read_json(directory / "status.json")
    if status["status"] != "completed":
        raise ValueError(f"Run is not completed: {directory}")
    plan, provenance = read_json(directory / "launch.json"), read_json(directory / "provenance.json")
    if plan["schema_version"] != "pbsp_launch_v1":
        raise ValueError("Expected a native PBSP run plan")
    config = plan["configuration"]
    config_hash = digest({"experiment": config, "mapping_model": plan["model_specification"]})
    if config_hash != plan["configuration_sha256"] or config_hash != provenance["configuration_sha256"]:
        raise ValueError("Saved run configuration hash differs")
    identity_hash = digest(plan["identities"])
    if identity_hash != plan["identity_sha256"] or identity_hash != provenance["identity_sha256"]:
        raise ValueError("Saved task identity hash differs")
    dataset = config["dataset"]
    manifest_path = directory / "dataset_manifest.json"
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != dataset["manifest_sha256"]:
        raise ValueError("Saved dataset manifest differs")
    manifest = read_json(manifest_path)
    identities = {}
    for record in plan["identities"]:
        identity = EpisodeIdentity(**record)
        key = (identity.scene_id, identity.runtime_episode_id)
        if key in identities or identity.dataset_manifest_sha256 != dataset["manifest_sha256"]:
            raise ValueError(f"Duplicate or mismatched selected identity: {key}")
        if len(identity.ordered_categories) != dataset["expected_tasks_per_episode"]:
            raise ValueError(f"Selected task count differs: {key}")
        identities[key] = identity
    selection = {(row["scene_id"], str(episode)) for row in plan["scenes"] for episode in row["episode_ids"]}
    if not identities or selection != set(identities):
        raise ValueError("Selected scenes and episode identities differ")
    if any(Path(scene).name != scene for scene, episode in identities):
        raise ValueError("Invalid scene identifier")
    expected_processes = {"mapping_process.json", "monitor_process.json"}
    expected_processes.update(f"navigation_{scene}_process.json" for scene, episode in identities)
    actual_processes = {path.name for path in directory.glob("*_process.json")}
    if actual_processes != expected_processes:
        raise ValueError("Worker exit record inventory differs from the selected scenes")
    for name in expected_processes:
        code = read_json(directory / name)["returncode"]
        if type(code) is not int or code != 0:
            raise ValueError(f"Worker did not exit successfully: {name}")
    if provenance["commit"] != plan["source_commit"]:
        raise ValueError("Run plan and source provenance revisions differ")
    patch = (directory / "source.patch").read_bytes()
    checks = provenance.get("resource_checks", {})
    model_proof = checks.get("models")
    if model_proof is not None:
        model_manifest = directory / "model_manifest.json"
        if model_proof.get("status") != "passed" or hashlib.sha256(model_manifest.read_bytes()).hexdigest() != model_proof["manifest_sha256"]:
            raise ValueError("Saved model manifest differs from its checked identity")
    protocol = {key: config[key] for key in ("method", "dataset", "evaluation", "seed")}
    protocol.update({"model_specification": plan["model_specification"], "source_commit": plan["source_commit"],
                     "models": model_proof, "habitat_revision": checks.get("habitat_revision"),
                     "superglue_source": checks.get("superglue_source")})
    raw = merge_episode_metric_files(directory.glob("results_*/**/per_episode_metrics.json"))
    expected_keys = {f"{scene}_{episode}": (scene, episode) for scene, episode in identities}
    if len(expected_keys) != len(identities) or set(raw) != set(expected_keys):
        raise ValueError("Completed metric episodes differ from the selected identities")
    times, frame_times = timing_records(directory.glob("results_*/**/timing_log.csv"), identities)
    records = {}
    for key, record in raw.items():
        identity_key = expected_keys[key]
        identity = identities[identity_key]
        categories = [task["category"] for task in record["tasks"]]
        if categories != list(identity.ordered_categories):
            raise ValueError(f"Recorded task order differs: {key}")
        frames = sorted(times.get(identity_key, []))
        if [row[0] for row in frames] != list(range(1, record["total_num_steps"] + 1)):
            raise ValueError(f"Timing frame coverage differs: {key}")
        if [row[1] for row in frames] != sorted(row[1] for row in frames):
            raise ValueError(f"Timing task order differs: {key}")
        for task, metric in enumerate(record["metrics"]):
            samples = [elapsed for frame, index, elapsed in frames if index == task]
            if len(samples) != metric[STEPS_KEY] or len(samples) != metric[FRAME_COUNT_KEY] or sum(samples) != metric[FRAME_TIME_SUM_KEY]:
                raise ValueError(f"Subtask timing telemetry differs: {key}, task {task}")
        rebuilt = summarize_episode(record["metrics"], total_steps=record["total_num_steps"],
                                    stop_reasons=record["stop_reasons"], frame_times_ms=[row[2] for row in frames],
                                    categories=categories)
        if any(record.get(field) != value for field, value in rebuilt.items()):
            raise ValueError(f"Recorded episode metrics differ from raw measurements: {key}")
        records[key] = {"metrics": record["metrics"], **rebuilt}
    metrics = summarize_records(records, frame_times_ms=frame_times)
    if metrics != read_json(directory / "mon_metrics.json"):
        raise ValueError("Saved run aggregate differs from raw episode and timing records")
    if (status["completed_episodes"] != len(identities) or provenance["selected_episodes"] != len(identities)
            or provenance["selected_tasks"] != len(identities) * dataset["expected_tasks_per_episode"]):
        raise ValueError("Run completion counts differ from the selected task identities")
    row = {"run": directory.name, "directory": str(directory), "dataset": dataset["name"],
           "source_commit": plan["source_commit"], "source_dirty": bool(provenance["status"] or patch.strip()),
           "protocol_sha256": digest(protocol), "identity_sha256": identity_hash,
           "coverage": coverage(identities, dataset, manifest), "metrics": metrics,
           "model_identity_checked": model_proof is not None, "runtime": config["runtime"]}
    return ValidatedRun(directory, identities, records, frame_times, protocol, row)


def summarize_runs(directories, *, merge=False):
    paths = sorted(Path(path).resolve() for path in directories)
    if not paths or len(set(paths)) != len(paths) or any(a in b.parents for a in paths for b in paths if a != b):
        raise ValueError("Select distinct, non-nested run directories")
    runs = [validate_run(path) for path in paths]
    rows = [run.row for run in runs]
    if merge:
        protocol = runs[0].protocol
        if any(run.protocol != protocol for run in runs):
            raise ValueError("Cannot merge runs with different code, method, data, seed, model or evaluation identities")
        if any(run.row["source_dirty"] or not run.row["model_identity_checked"] for run in runs):
            raise ValueError("Merge requires clean source records and checked model identities")
        identities, records, times = {}, {}, []
        for run in runs:
            if set(identities) & set(run.identities):
                raise ValueError("Cannot merge overlapping episode identities")
            identities.update(run.identities)
            records.update(run.records)
            times.extend(run.frame_times)
        manifest = read_json(runs[0].directory / "dataset_manifest.json")
        rows = [{"run": "merged", "dataset": runs[0].row["dataset"], "source_commit": protocol["source_commit"],
                 "source_dirty": False, "protocol_sha256": digest(protocol),
                 "coverage": coverage(identities, protocol["dataset"], manifest),
                 "metrics": summarize_records(records, frame_times_ms=times),
                 "source_runs": [run.row for run in runs]}]
    return {"schema_version": "pbsp_result_summary_v1", "mode": "merge" if merge else "compare", "rows": rows}


def markdown(report):
    def escape(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| Run | Dataset | Episodes | Scope | SR | SPL | PSR | PeSPL | Code |",
             "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |"]
    for row in report["rows"]:
        values, counts = row["metrics"]["mon_metrics"], row["coverage"]
        code = row["source_commit"][:8] + ("*" if row["source_dirty"] else "")
        scope = "full" if counts["kind"] == "full_dataset" else f"selection of {counts['dataset_episodes']}"
        cells = [row["run"], row["dataset"], counts["completed_episodes"], scope,
                 *(f"{values[key]:.6f}" for key in ("sr", "spl", "psr", "pespl")), code]
        lines.append("| " + " | ".join(escape(cell) for cell in cells) + " |")
    if any(row["source_dirty"] for row in report["rows"]):
        lines.extend(["", "An asterisk marks a run launched with working-tree changes."])
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--merge", action="store_true", help="Combine disjoint runs with identical checked protocol identities")
    parser.add_argument("--output", type=Path, help="Write summary.json and summary.md into a new directory")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        raise FileExistsError(args.output)
    report = summarize_runs(args.runs, merge=args.merge)
    table = markdown(report)
    if args.output is not None:
        destination = args.output.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".pbsp-summary-", dir=destination.parent) as temporary:
            staging = Path(temporary) / "summary"
            staging.mkdir()
            (staging / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
            (staging / "summary.md").write_text(table)
            staging.replace(destination)
    print(table, end="")


if __name__ == "__main__":
    main()
