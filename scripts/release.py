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


def run(*args: str, capture: bool = False, check: bool = True) -> str:
    result = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        capture_output=capture,
        check=check,
    )
    return result.stdout.strip() if capture else ""


def normalize_version(value: str) -> str:
    version = value.removeprefix("v")
    if not SEMVER_RE.fullmatch(version):
        raise SystemExit(f"Invalid version format: {value}")
    return version


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


def local_tag_exists(tag: str) -> bool:
    return run("git", "tag", "--list", tag, capture=True) == tag


def remote_tag_exists(tag: str, remote: str) -> bool:
    return bool(run("git", "ls-remote", "--tags", remote, f"refs/tags/{tag}", capture=True))


def ensure_tag_does_not_exist(tag: str, remote: str) -> None:
    if local_tag_exists(tag):
        raise SystemExit(f"Tag already exists locally: {tag}")
    if remote_tag_exists(tag, remote):
        raise SystemExit(f"Tag already exists on {remote}: {tag}")


def create_release(args: argparse.Namespace) -> int:
    version = normalize_version(args.version)
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
        print("Tag created locally only. Use --push to trigger the release workflow.")

    return 0


def rollback_release(args: argparse.Namespace) -> int:
    version = normalize_version(args.version)
    tag = f"v{version}"

    if local_tag_exists(tag):
        run("git", "tag", "-d", tag)
        print(f"Deleted local tag: {tag}")
    else:
        print(f"Local tag not found: {tag}")

    if remote_tag_exists(tag, args.remote):
        run("git", "push", args.remote, "--delete", tag)
        print(f"Deleted remote tag from {args.remote}: {tag}")
    else:
        print(f"Remote tag not found on {args.remote}: {tag}")

    if args.delete_release:
        gh = run("which", "gh", capture=True, check=False)
        if not gh:
            raise SystemExit("GitHub CLI 'gh' is required for --delete-release")

        run("gh", "release", "delete", tag, "--yes", check=False)
        print(f"Requested GitHub Release deletion: {tag}")

    if args.cancel_runs:
        gh = run("which", "gh", capture=True, check=False)
        if not gh:
            raise SystemExit("GitHub CLI 'gh' is required for --cancel-runs")

        runs = run(
            "gh",
            "run",
            "list",
            "--json",
            "databaseId,headBranch,headSha,event,status,displayTitle",
            "--limit",
            "50",
            capture=True,
        )
        print("Inspect current runs manually if needed:")
        print(runs)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage bmmpy release tags")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="Create a release tag")
    create.add_argument("version", help="Release version, for example 0.1.2 or v0.1.2")
    create.add_argument("--remote", default="origin", help="Git remote name")
    create.add_argument("--push", action="store_true", help="Push the tag after creating it")
    create.set_defaults(func=create_release)

    rollback = subparsers.add_parser("rollback", help="Delete a release tag")
    rollback.add_argument("version", help="Release version, for example 0.1.2 or v0.1.2")
    rollback.add_argument("--remote", default="origin", help="Git remote name")
    rollback.add_argument(
        "--delete-release",
        action="store_true",
        help="Also delete the GitHub Release with gh CLI",
    )
    rollback.add_argument(
        "--cancel-runs",
        action="store_true",
        help="Inspect or cancel GitHub Actions runs with gh CLI",
    )
    rollback.set_defaults(func=rollback_release)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())