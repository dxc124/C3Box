#!/usr/bin/env python3
"""Static readiness checks for running C3Box model x dataset experiments."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    classes: int
    storage: str
    note: str = ""


@dataclass(frozen=True)
class ModelSpec:
    name: str
    config: str
    family: str
    text_templates: bool = False
    engine_descriptions: bool = False
    concepts: bool = False
    local_checkpoint: str | None = None
    note: str = ""
    extra_packages: tuple[str, ...] = field(default_factory=tuple)


DATASETS: dict[str, DatasetSpec] = {
    "cifar224": DatasetSpec("cifar224", 100, "torchvision_cifar", "CIFAR100 resized to 224."),
    "cifar100": DatasetSpec("cifar100", 100, "torchvision_cifar"),
    "imagenetr": DatasetSpec("imagenetr", 200, "image_folder"),
    "imageneta": DatasetSpec("imageneta", 200, "image_folder", "Missing templates.json entry in this checkout."),
    "objectnet": DatasetSpec("objectnet", 200, "image_folder"),
    "cub200": DatasetSpec("cub200", 200, "image_folder"),
    "caltech101": DatasetSpec("caltech101", 100, "image_folder"),
    "food101": DatasetSpec("food101", 100, "image_folder"),
    "flowers": DatasetSpec("flowers", 100, "image_folder"),
    "aircraft": DatasetSpec("aircraft", 100, "image_folder"),
    "ucf101": DatasetSpec("ucf101", 100, "image_folder"),
    "cars": DatasetSpec("cars", 100, "image_folder"),
    "sun": DatasetSpec("sun", 300, "image_folder"),
    "tv100": DatasetSpec("tv100", 100, "image_folder", "Dataset class exists, but labels/templates are missing."),
}


MODELS: dict[str, ModelSpec] = {
    "finetune": ModelSpec("finetune", "finetune.json", "supervised clip baseline"),
    "zs_clip": ModelSpec("zs_clip", "zs_clip.json", "zero-shot clip", text_templates=True),
    "simplecil": ModelSpec("simplecil", "simplecil.json", "prototype baseline"),
    "foster": ModelSpec("foster", "foster.json", "classic cil with exemplars"),
    "memo": ModelSpec("memo", "memo.json", "classic cil with exemplars"),
    "l2p": ModelSpec("l2p", "l2p.json", "prompt tuning"),
    "dualprompt": ModelSpec("dualprompt", "dual.json", "prompt tuning", local_checkpoint="c.pth"),
    "coda": ModelSpec("coda", "coda.json", "prompt tuning", local_checkpoint="c.pth"),
    "ease": ModelSpec("ease", "ease.json", "adapter ensemble", local_checkpoint="c.pth"),
    "tuna": ModelSpec("tuna", "tuna.json", "adapter fusion", local_checkpoint=".p/c.pth"),
    "rapf": ModelSpec("rapf", "rapf.json", "clip adaptation", text_templates=True),
    "proof": ModelSpec("proof", "proof.json", "vision-language cil", text_templates=True),
    "engine": ModelSpec(
        "engine",
        "engine.json",
        "knowledge-injected clip",
        text_templates=True,
        engine_descriptions=True,
    ),
    "bofa": ModelSpec("bofa", "bofa.json", "clip adapter", text_templates=True),
    "mind": ModelSpec("mind", "mind.json", "lora clip", text_templates=True, extra_packages=("clip",)),
    "clg_cbm": ModelSpec(
        "clg_cbm",
        "clg_cbm.json",
        "concept bottleneck",
        text_templates=True,
        concepts=True,
        extra_packages=("clip",),
    ),
    "aper_adapter": ModelSpec("aper_adapter", "aper_adapter.json", "aper", local_checkpoint="c.pth"),
    "aper_ssf": ModelSpec("aper_ssf", "aper_ssf.json", "aper", local_checkpoint="c.pth"),
    "aper_vpt": ModelSpec("aper_vpt", "aper_vpt_deep.json", "aper", local_checkpoint="c.pth"),
    "aper_finetune": ModelSpec("aper_finetune", "aper_finetune.json", "aper", local_checkpoint="c.pth"),
}


PYTHON_PACKAGES = ("torch", "torchvision", "numpy", "scipy", "PIL", "tqdm", "open_clip", "easydict", "timm")


CONCEPT_PATHS = {
    "cifar224": "utils/clg_cbm/concepts/cifar100/concepts.json",
    "cub200": "utils/clg_cbm/concepts/cub200/cub200_4o_simple_cpts.json",
    "food101": "utils/clg_cbm/concepts/food101/food_4o_simple_cpts.json",
    "ucf101": "utils/clg_cbm/concepts/ucf101.json",
    "cars": "utils/clg_cbm/concepts/cars/cars_4o_simple_cpts.json",
    "imagenetr": "utils/clg_cbm/concepts/imagenetr.json",
    "aircraft": "utils/clg_cbm/concepts/aircraft.json",
    "sun": "utils/clg_cbm/concepts/sun.json",
    "objectnet": "utils/clg_cbm/concepts/objectnet.json",
}


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _split_csv(value: str | None, choices: dict[str, object]) -> list[str]:
    if not value or value == "all":
        return list(choices)
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(names) - set(choices))
    if unknown:
        raise SystemExit(f"Unknown names: {', '.join(unknown)}")
    return names


def _env_key(dataset: str) -> str:
    return dataset.upper().replace("-", "_")


def dataset_root(dataset: str) -> Path:
    data_root = Path(os.environ.get("C3BOX_DATA_ROOT", ROOT / "data"))
    return Path(os.environ.get(f"C3BOX_{_env_key(dataset)}_ROOT", data_root / dataset))


def split_dir(dataset: str, split: str) -> Path:
    return Path(os.environ.get(f"C3BOX_{_env_key(dataset)}_{split.upper()}_DIR", dataset_root(dataset) / split))


def checkpoint_exists(relative_path: str) -> bool:
    return (ROOT / relative_path).exists()


def dependency_status(models: Iterable[str]) -> list[tuple[str, bool]]:
    packages = set(PYTHON_PACKAGES)
    for model in models:
        packages.update(MODELS[model].extra_packages)
    return [(pkg, importlib.util.find_spec(pkg) is not None) for pkg in sorted(packages)]


def dataset_status(dataset: str, labels: dict, templates: dict) -> dict:
    spec = DATASETS[dataset]
    labels_ok = dataset in labels
    templates_ok = dataset in templates

    if spec.storage == "torchvision_cifar":
        root = dataset_root(dataset)
        data_ready = root.exists()
        path_note = f"root={root}"
    else:
        train_dir = split_dir(dataset, "train")
        val_dir = split_dir(dataset, "val")
        data_ready = train_dir.is_dir() and val_dir.is_dir()
        path_note = f"train={train_dir}, val={val_dir}"

    return {
        "name": dataset,
        "classes": spec.classes,
        "labels": labels_ok,
        "templates": templates_ok,
        "data_ready": data_ready,
        "path_note": path_note,
        "note": spec.note,
    }


def pair_status(model: str, dataset: str, labels: dict, templates: dict) -> dict:
    spec = MODELS[model]
    blockers: list[str] = []
    warnings: list[str] = []

    if dataset not in labels:
        blockers.append("missing utils/labels.json entry; DataManager loads it for every model")
    if dataset not in templates:
        blockers.append("missing utils/templates.json entry; DataManager loads it for every model")
    if spec.text_templates and dataset not in templates:
        blockers.append("model uses text prompts, but dataset templates are missing")
    if spec.engine_descriptions:
        des_path = ROOT / "utils" / "engine" / "chat" / f"{dataset}_des.json"
        if not des_path.exists():
            blockers.append(f"missing ENGINE description file: {des_path.relative_to(ROOT)}")
    if spec.concepts:
        concept_path = CONCEPT_PATHS.get(dataset)
        if concept_path is None:
            blockers.append("CLG-CBM has no concept path for this dataset")
        elif not (ROOT / concept_path).exists():
            blockers.append(f"missing CLG-CBM concept file expected by code: {concept_path}")
    if spec.local_checkpoint and not checkpoint_exists(spec.local_checkpoint):
        blockers.append(f"missing local checkpoint: {spec.local_checkpoint}")

    ds = dataset_status(dataset, labels, templates)
    if not ds["data_ready"]:
        warnings.append("dataset files not found locally; set C3BOX_DATA_ROOT or dataset-specific env vars")

    return {
        "model": model,
        "dataset": dataset,
        "ready": not blockers,
        "blockers": blockers,
        "warnings": warnings,
    }


def print_report(models: list[str], datasets: list[str], as_json: bool) -> None:
    labels = _load_json(ROOT / "utils" / "labels.json")
    templates = _load_json(ROOT / "utils" / "templates.json")

    report = {
        "root": str(ROOT),
        "dependencies": dependency_status(models),
        "datasets": [dataset_status(name, labels, templates) for name in datasets],
        "pairs": [pair_status(model, dataset, labels, templates) for model in models for dataset in datasets],
    }

    if as_json:
        print(json.dumps(report, indent=2))
        return

    print(f"C3Box root: {ROOT}")
    print("\nPython packages:")
    for package, ok in report["dependencies"]:
        marker = "ok" if ok else "missing"
        print(f"  [{marker}] {package}")

    print("\nDatasets:")
    for item in report["datasets"]:
        markers = []
        markers.append("labels" if item["labels"] else "no-labels")
        markers.append("templates" if item["templates"] else "no-templates")
        markers.append("data" if item["data_ready"] else "no-data")
        print(f"  {item['name']:<12} classes={item['classes']:<3} {'/'.join(markers)}")
        print(f"    {item['path_note']}")
        if item["note"]:
            print(f"    note: {item['note']}")

    print("\nModel x dataset readiness:")
    for item in report["pairs"]:
        marker = "ready" if item["ready"] else "blocked"
        print(f"  [{marker}] {item['model']} x {item['dataset']}")
        for blocker in item["blockers"]:
            print(f"    blocker: {blocker}")
        for warning in item["warnings"]:
            print(f"    warning: {warning}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="all", help="Comma-separated model names, or 'all'.")
    parser.add_argument("--datasets", default="cifar224,aircraft,cars,food101,cub200,imagenetr,objectnet,sun,ucf101", help="Comma-separated dataset names, or 'all'.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    models = _split_csv(args.models, MODELS)
    datasets = _split_csv(args.datasets, DATASETS)
    print_report(models, datasets, args.json)


if __name__ == "__main__":
    main()
