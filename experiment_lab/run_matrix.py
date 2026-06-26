#!/usr/bin/env python3
"""Run generated C3Box experiment configs one by one."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def collect_configs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("*.json"))


def resolve_configs_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.exists() or path.is_absolute():
        return path
    rooted = ROOT / path
    if rooted.exists():
        return rooted
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", default=str(ROOT / "experiment_lab" / "generated_configs" / "smoke"), help="Config file or directory.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use.")
    parser.add_argument("--dry-run", action="store_true", help="Only print commands.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop at the first failed run.")
    args = parser.parse_args()

    configs = collect_configs(resolve_configs_path(args.configs))
    if not configs:
        raise SystemExit(f"No config files found under {args.configs}")

    failures: list[tuple[Path, int]] = []
    for config in configs:
        command = [args.python, "main.py", f"--config={config.resolve()}"]
        printable = " ".join(str(item) for item in command)
        print(f"\n$ cd {ROOT} && {printable}")
        if args.dry_run:
            continue
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode != 0:
            failures.append((config, result.returncode))
            if args.stop_on_error:
                break

    if failures:
        print("\nFailed runs:")
        for config, code in failures:
            print(f"  {config}: exit {code}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
