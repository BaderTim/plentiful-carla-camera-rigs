#!/usr/bin/env python3
"""Standardize raw cross-rig evaluation logs into per-result JSON files."""

import argparse
import ast
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import stdev


SUMMARY_METRICS = ("mAP", "mATE", "mASE", "mAOE", "mAVE", "mAAE", "NDS")
RIG_PATTERN = re.compile(r"^R\d+(?:-(?:c6|c10|f|r|t))?$")


def normalize_rig_name(name):
    normalized = name.strip()
    if normalized.startswith("train_"):
        normalized = normalized[len("train_") :]
    if normalized.endswith("_test"):
        normalized = normalized[: -len("_test")]

    normalized = normalized.upper().replace("S", "R")
    for suffix in ("-C10", "-C6", "-F", "-R", "-T"):
        normalized = normalized.replace(suffix, suffix.lower())
    return normalized


def rig_sort_key(rig_name):
    match = re.fullmatch(r"R(\d+)(?:-(.*))?", normalize_rig_name(rig_name))
    if not match:
        return (float("inf"), rig_name)

    suffix_order = {
        "": 0,
        "c10": 1,
        "c6": 2,
        "f": 3,
        "r": 4,
        "t": 5,
    }
    suffix = match.group(2) or ""
    return (int(match.group(1)), suffix_order.get(suffix, 99), suffix)


def extract_test_rig_from_name(name):
    patterns = (
        r"test_latest_on_(.+?)_test$",
        r"test_epoch_\d+_on_(.+?)(?:_test)?$",
        r".*tested_on_(.+?)\.log$",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, name)
        if match:
            return normalize_rig_name(match.group(1))
    return None


def extract_metrics_dict(content):
    metrics = {}
    for match in re.finditer(r"\{[^{}]*(?:object|pts_bbox_NuScenes)[^{}]*\}", content):
        try:
            value = ast.literal_eval(match.group(0))
        except (SyntaxError, ValueError):
            continue
        if isinstance(value, dict):
            metrics.update(value)
    return metrics


def extract_summary_metrics(content):
    summary = {}
    for metric in SUMMARY_METRICS:
        match = re.search(rf"^{re.escape(metric)}:\s+([0-9.]+)", content, re.MULTILINE)
        if match:
            summary[metric] = float(match.group(1))
    return summary


def first_present(*values):
    for value in values:
        if value is not None:
            return value
    return None


def read_result(log_path, model, train_rig, test_rig, raw_root):
    content = log_path.read_text(errors="replace")
    summary = extract_summary_metrics(content)
    raw_metrics = extract_metrics_dict(content)

    map_value = first_present(
        summary.get("mAP"),
        raw_metrics.get("object/map"),
        raw_metrics.get("pts_bbox_NuScenes/mAP"),
    )
    nds_value = first_present(
        summary.get("NDS"),
        raw_metrics.get("object/nds"),
        raw_metrics.get("object/NDS"),
        raw_metrics.get("pts_bbox_NuScenes/NDS"),
    )

    return {
        "model": model,
        "trained_on": train_rig,
        "tested_on": test_rig,
        "metrics": {
            "mAP": map_value,
            "NDS": nds_value,
            "summary": summary,
            "raw": raw_metrics,
        },
        "source": {
            "log": str(log_path.relative_to(raw_root)),
        },
    }


def iter_result_logs(raw_root):
    for model_dir in sorted(path for path in raw_root.iterdir() if path.is_dir()):
        model = model_dir.name
        for train_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
            train_rig = normalize_rig_name(train_dir.name)
            if not RIG_PATTERN.fullmatch(train_rig):
                continue

            for test_dir in sorted(path for path in train_dir.iterdir() if path.is_dir()):
                test_rig = extract_test_rig_from_name(test_dir.name)
                log_path = test_dir / "test.log"
                if test_rig and log_path.exists():
                    yield model, train_rig, test_rig, log_path

            petr_logs = train_dir / "test_logs"
            if petr_logs.exists():
                for log_path in sorted(petr_logs.glob("*tested_on_*.log")):
                    test_rig = extract_test_rig_from_name(log_path.name)
                    if test_rig:
                        yield model, train_rig, test_rig, log_path


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def add_numeric(values, key, value):
    if isinstance(value, (int, float)):
        values[key].append(float(value))


def average_numeric_dicts(dicts):
    values = defaultdict(list)
    for data in dicts:
        for key, value in data.items():
            add_numeric(values, key, value)
    return {
        key: sum(items) / len(items)
        for key, items in sorted(values.items())
        if items
    }


def std_numeric_dicts(dicts):
    values = defaultdict(list)
    for data in dicts:
        for key, value in data.items():
            add_numeric(values, key, value)
    return {
        key: sample_std(items)
        for key, items in sorted(values.items())
        if items
    }


def sample_std(values):
    if len(values) < 2:
        return 0.0
    return stdev(values)


def build_model_average(model, results):
    same_rig_results = [
        result
        for result in results
        if result.get("trained_on") == result.get("tested_on")
    ]
    metric_values = defaultdict(list)
    summary_dicts = []
    raw_dicts = []

    for result in same_rig_results:
        metrics = result.get("metrics", {})
        add_numeric(metric_values, "mAP", metrics.get("mAP"))
        add_numeric(metric_values, "NDS", metrics.get("NDS"))
        summary_dicts.append(metrics.get("summary", {}))
        raw_dicts.append(metrics.get("raw", {}))

    metric_counts = {key: len(values) for key, values in sorted(metric_values.items())}
    averages = {
        key: sum(values) / len(values)
        for key, values in sorted(metric_values.items())
        if values
    }
    averages["summary"] = average_numeric_dicts(summary_dicts)
    averages["raw"] = average_numeric_dicts(raw_dicts)
    std_values = {
        key: sample_std(values)
        for key, values in sorted(metric_values.items())
        if values
    }
    std_values["summary"] = std_numeric_dicts(summary_dicts)
    std_values["raw"] = std_numeric_dicts(raw_dicts)

    return {
        "model": model,
        "aggregation": "average_over_same_rig_results",
        "num_results": len(same_rig_results),
        "num_source_results": len(results),
        "metric_counts": metric_counts,
        "metrics": averages,
        "std": std_values,
    }


def standardize_results(raw_root, output_root, write_per_test=False):
    results_by_train = {}
    results_by_model = defaultdict(list)
    total = 0
    skipped_without_map = 0

    for model, train_rig, test_rig, log_path in iter_result_logs(raw_root):
        result = read_result(log_path, model, train_rig, test_rig, raw_root)
        if result["metrics"]["mAP"] is None:
            skipped_without_map += 1

        if write_per_test:
            result_path = output_root / model / f"trained_on_{train_rig}" / f"tested_on_{test_rig}.json"
            write_json(result_path, result)

        results_by_train.setdefault((model, train_rig), {})[test_rig] = result
        results_by_model[model].append(result)
        total += 1

    for (model, train_rig), rig_results in sorted(results_by_train.items()):
        ordered_results = {
            rig: rig_results[rig]
            for rig in sorted(rig_results, key=rig_sort_key)
        }
        aggregate = {
            "model": model,
            "trained_on": train_rig,
            "results": ordered_results,
        }
        aggregate_path = output_root / model / f"trained_on_{train_rig}.json"
        write_json(aggregate_path, aggregate)

    for model, model_results in sorted(results_by_model.items()):
        average_path = output_root / model / "average_over_all_rigs.json"
        write_json(average_path, build_model_average(model, model_results))

    manifest = {
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "num_results": total,
        "num_train_runs": len(results_by_train),
        "num_model_average_files": len(results_by_model),
        "num_results_without_map": skipped_without_map,
        "write_per_test_files": write_per_test,
    }
    write_json(output_root / "manifest.json", manifest)
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert raw evaluation logs into standardized JSON result files."
    )
    parser.add_argument(
        "--raw-root",
        default=Path("raw_results"),
        type=Path,
        help="Raw results directory. Defaults to raw_results, falling back to results_raw.",
    )
    parser.add_argument(
        "--output-root",
        default="results",
        type=Path,
        help="Destination directory. Defaults to results.",
    )
    parser.add_argument(
        "--write-per-test",
        action="store_true",
        help="Also write nested tested_on_<rig>.json files for every test result.",
    )
    return parser.parse_args()


def resolve_raw_root(raw_root):
    if raw_root.exists():
        return raw_root
    if raw_root == Path("raw_results") and Path("results_raw").exists():
        return Path("results_raw")
    return raw_root


def main():
    args = parse_args()
    raw_root = resolve_raw_root(args.raw_root)
    if not raw_root.exists():
        raise SystemExit(f"Raw results directory does not exist: {args.raw_root}")

    manifest = standardize_results(raw_root, args.output_root, args.write_per_test)
    print(
        "Standardized {num_results} results across {num_train_runs} train runs "
        "to {output_root}.".format(**manifest)
    )
    if manifest["num_results_without_map"]:
        print(f"Warning: {manifest['num_results_without_map']} results did not contain mAP.")


if __name__ == "__main__":
    main()
