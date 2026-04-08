from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
VERSION_RE = re.compile(r'^version = "([^"]+)"$', re.MULTILINE)
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


def run(*args: str, capture: bool = False) -> str:
    result = subprocess.run(
        args,
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=capture,
    )
    return result.stdout.strip() if capture else ""


def read_project_version() -> str:
    text = PYPROJECT.read_text(encoding="utf-8")
    match = VERSION_RE.search(text)
    if not match:
        raise SystemExit("Failed to read version from pyproject.toml")
    return match.group(1)


def ensure_clean_git() -> None:
    status = run("git", "status", "--porcelain", capture=True)
    if status:
        raise SystemExit("Git working tree is not clean")


def ensure_tag_does_not_exist(tag: str, remote: str) -> None:
    local_tags = run("git", "tag", "--list", tag, capture=True)
    if local_tags == tag:
        raise SystemExit(f"Tag already exists locally: {tag}")

    remote_tags = run("git", "ls-remote", "--tags", remote, f"refs/tags/{tag}", capture=True)
    if remote_tags:
        raise SystemExit(f"Tag already exists on {remote}: {tag}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create and optionally push a release tag")
    parser.add_argument("version", help="Release version, for example 0.1.2 or v0.1.2")
    parser.add_argument("--remote", default="origin", help="Git remote name")
    parser.add_argument("--push", action="store_true", help="Push the tag after creating it")
    args = parser.parse_args()

    version = args.version.removeprefix("v")
    if not SEMVER_RE.fullmatch(version):
        raise SystemExit(f"Invalid version format: {args.version}")

    project_version = read_project_version()
    if version != project_version:
        raise SystemExit(
            f"Version mismatch: requested {version}, but pyproject.toml contains {project_version}"
        )

    ensure_clean_git()

    tag = f"v{version}"
    ensure_tag_does_not_exist(tag, args.remote)

    run("git", "tag", "-a", tag, "-m", f"bmmpy {version}")
    print(f"Created tag: {tag}")

    if args.push:
        run("git", "push", args.remote, tag)
        print(f"Pushed tag to {args.remote}: {tag}")
        print("GitHub Actions release workflow should start automatically.")
    else:
        print("Tag created locally only. Use --push to trigger the GitHub release workflow.")

    return 0


if __name__ == "__main__":
    sys.exit(main())