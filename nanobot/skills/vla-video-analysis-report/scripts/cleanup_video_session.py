#!/usr/bin/env python3
"""Clean temporary cache or entire session for VLA video analysis runs."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


DEFAULT_OUTPUT_ROOT = Path("./nanobot/workspace/output")


def ensure_within_root(path: Path, root: Path) -> Path:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise RuntimeError(f"Path is outside output root: {resolved_path}") from exc
    return resolved_path


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Delete the entire session directory instead of cache only",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        session_dir = ensure_within_root(args.session_dir, args.output_root)
        if not session_dir.exists():
            print(f"Session does not exist: {session_dir}")
            return 0

        target = session_dir if args.all else session_dir / "cache"
        if target.exists():
            shutil.rmtree(target)
            print(f"Deleted: {target}")
        else:
            print(f"Nothing to delete: {target}")
        return 0
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
