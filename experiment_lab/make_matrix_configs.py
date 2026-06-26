#!/usr/bin/env python3
"""Generate C3Box config files for model x dataset experiment matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from check_environment import DATASETS, MODELS, ROOT, _split_csv


EPOCH_KEYS = (
    "tuned_epoch",
    "init_epochs",
    "later_epochs",
    "init_epoch",
    "epochs",
    "boosting_epochs",
    "compression_epochs",
    "crct_epochs",
    "FB_epoch",
    "FB_epoch_inc",
    "visual_clsf_epochs",
    "balance_epochs",
)


def task_count(total_classes: int, init_cls: int, increment: int) -> int:
    if init_cls >= total_classes:
        return 1
    remaining = total_classes - init_cls
    return 1 + (remaining + increment - 1) // increment


def load_base_config(model: str) -> dict:
    config_path = ROOT / "exps" / MODELS[model].config
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def apply_common_overrides(config: dict, model: str, dataset: str, device: str | None) -> dict:
    total_classes = DATASETS[dataset].classes
    config = dict(config)
    config["dataset"] = dataset
    config["total_class_num"] = total_classes
    config["nb_classes"] = total_classes

    if "class_num" in config:
        config["class_num"] = total_classes
    if device is not None:
        config["device"] = [device]

    init_cls = min(int(config.get("init_cls", 10)), total_classes)
    increment = min(int(config.get("increment", 10)), max(1, total_classes - init_cls or 1))
    config["init_cls"] = init_cls
    config["increment"] = increment
    config["nb_tasks"] = task_count(total_classes, init_cls, increment)
    config["prefix"] = f"matrix_{model}_{dataset}"
    return config


def apply_smoke_overrides(config: dict, smoke_classes: int, smoke_increment: int, smoke_memory_per_class: int) -> dict:
    total_classes = int(config["total_class_num"])
    init_cls = min(smoke_classes, total_classes)
    increment = min(smoke_increment, max(1, total_classes - init_cls or 1))
    config["init_cls"] = init_cls
    config["increment"] = increment
    config["nb_tasks"] = task_count(total_classes, init_cls, increment)
    config["max_tasks"] = 1
    config["prefix"] = "smoke_" + str(config.get("prefix", "matrix"))

    if "batch_size" in config:
        config["batch_size"] = min(int(config["batch_size"]), 8)
    for key in EPOCH_KEYS:
        if key in config:
            config[key] = 1

    if int(config.get("memory_size", 0)) > 0:
        config["memory_per_class"] = min(int(config.get("memory_per_class", smoke_memory_per_class)), smoke_memory_per_class)
        config["memory_size"] = min(int(config["memory_size"]), total_classes * smoke_memory_per_class)
    return config


def write_config(config: dict, out_dir: Path, model: str, dataset: str, preset: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{model}__{dataset}__{preset}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=4)
        handle.write("\n")
    return path


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="simplecil,zs_clip,finetune,ease,tuna,proof,engine,bofa,clg_cbm", help="Comma-separated model names, or 'all'.")
    parser.add_argument("--datasets", default="cifar224,aircraft,cars,food101,cub200,imagenetr,objectnet,sun,ucf101", help="Comma-separated dataset names, or 'all'.")
    parser.add_argument("--preset", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--out", default=str(ROOT / "experiment_lab" / "generated_configs"))
    parser.add_argument("--device", default=None, help="Override the single CUDA device id, e.g. 0. Omit to keep base configs.")
    parser.add_argument("--smoke-classes", type=int, default=2)
    parser.add_argument("--smoke-increment", type=int, default=2)
    parser.add_argument("--smoke-memory-per-class", type=int, default=2)
    args = parser.parse_args()

    models = _split_csv(args.models, MODELS)
    datasets = _split_csv(args.datasets, DATASETS)
    out_dir = Path(args.out) / args.preset

    written = []
    for model in models:
        for dataset in datasets:
            config = load_base_config(model)
            config = apply_common_overrides(config, model, dataset, args.device)
            if args.preset == "smoke":
                config = apply_smoke_overrides(
                    config,
                    smoke_classes=args.smoke_classes,
                    smoke_increment=args.smoke_increment,
                    smoke_memory_per_class=args.smoke_memory_per_class,
                )
            written.append(write_config(config, out_dir, model, dataset, args.preset))

    for path in written:
        print(display_path(path))
    print(f"Wrote {len(written)} config files.")


if __name__ == "__main__":
    main()
