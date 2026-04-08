from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
VERSION_RE = re.compile(r'^version = "(\d+)\.(\d+)\.(\d+)"$', re.MULTILINE)
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


def read_pyproject() -> str:
    return PYPROJECT.read_text(encoding="utf-8")


def parse_current_version(text: str) -> tuple[int, int, int]:
    match = VERSION_RE.search(text)
    if not match:
        raise SystemExit("Failed to read version from pyproject.toml")
    return tuple(int(part) for part in match.groups())


def format_version(parts: tuple[int, int, int]) -> str:
    return ".".join(str(part) for part in parts)


def bump(parts: tuple[int, int, int], target: str) -> tuple[int, int, int]:
    major, minor, patch = parts
    if target == "patch":
        return major, minor, patch + 1
    if target == "minor":
        return major, minor + 1, 0
    if target == "major":
        return major + 1, 0, 0
    if SEMVER_RE.fullmatch(target):
        return tuple(int(part) for part in target.split("."))
    raise SystemExit(f"Unsupported bump target: {target}")


def ensure_clean_git(allow_dirty: bool) -> None:
    if allow_dirty:
        return
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    if result.stdout.strip():
        raise SystemExit("Git working tree is not clean; use --allow-dirty to override")


def main() -> int:
    parser = argparse.ArgumentParser(description="Update project version in pyproject.toml")
    parser.add_argument(
        "target",
        nargs="?",
        default="patch",
        help="patch, minor, major, or an exact version like 0.2.0",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the new version without writing")
    parser.add_argument("--allow-dirty", action="store_true", help="Allow local uncommitted changes")
    args = parser.parse_args()

    ensure_clean_git(args.allow_dirty)

    text = read_pyproject()
    current_parts = parse_current_version(text)
    new_parts = bump(current_parts, args.target)

    current_version = format_version(current_parts)
    new_version = format_version(new_parts)

    if current_version == new_version:
        raise SystemExit(f"Version is already {new_version}")

    new_text, count = VERSION_RE.subn(f'version = "{new_version}"', text, count=1)
    if count != 1:
        raise SystemExit("Expected to replace exactly one version field")

    print(f"{current_version} -> {new_version}")

    if args.dry_run:
        return 0

    PYPROJECT.write_text(new_text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())