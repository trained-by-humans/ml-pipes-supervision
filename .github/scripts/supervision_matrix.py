"""Resolve CI's Supervision release bounds from pyproject.toml and PyPI.

Run with Python 3.11+ and packaging installed. The target Python versions may
be older than the interpreter used to generate the matrix.
"""

import argparse
import json
import sys
import tomllib
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version


def supervision_requirement(pyproject: dict[str, Any]) -> Requirement:
    requirements = [
        Requirement(dependency)
        for dependency in pyproject["project"]["dependencies"]
    ]
    matches = [
        requirement
        for requirement in requirements
        if canonicalize_name(requirement.name) == "supervision"
    ]
    if len(matches) != 1 or matches[0].url is not None:
        raise ValueError("Expected exactly one index-based Supervision dependency.")
    return matches[0]


def release_bounds(
    requirement: Requirement, releases: dict[str, Any], python_version: str
) -> tuple[str, str]:
    target = Version(python_version)
    environment = default_environment()
    environment.update(
        python_version=f"{target.major}.{target.minor}",
        python_full_version=f"{target.major}.{target.minor}.{target.micro}",
    )
    if requirement.marker is not None and not requirement.marker.evaluate(environment):
        raise ValueError(f"{requirement} does not apply to Python {python_version}.")

    candidates: set[Version] = set()
    for release, files in releases.items():
        try:
            version = Version(release)
        except InvalidVersion:
            continue
        if version.is_prerelease or version.is_devrelease:
            continue
        if not requirement.specifier.contains(version):
            continue
        if any(
            not file.get("yanked", False)
            and file.get("packagetype") in {"bdist_wheel", "sdist"}
            and SpecifierSet(file.get("requires_python") or "").contains(target)
            for file in files
        ):
            candidates.add(version)

    if not candidates:
        raise ValueError(
            f"No non-yanked stable release matches {requirement} "
            f"on Python {python_version}."
        )
    return str(min(candidates)), str(max(candidates))


def build_matrix(
    pyproject: dict[str, Any], releases: dict[str, Any], python_versions: list[str]
) -> dict[str, list[dict[str, str]]]:
    requirement = supervision_requirement(pyproject)
    supported_python = SpecifierSet(pyproject["project"]["requires-python"])
    rows = []
    for python_version in python_versions:
        if not supported_python.contains(python_version):
            raise ValueError(f"Project does not support Python {python_version}.")
        lowest, highest = release_bounds(requirement, releases, python_version)
        profiles = (
            [("lowest/highest", lowest)]
            if lowest == highest
            else [("lowest", lowest), ("highest", highest)]
        )
        rows.extend(
            {
                "python-version": python_version,
                "supervision-profile": profile,
                "supervision-version": version,
            }
            for profile, version in profiles
        )
    return {"include": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    parser.add_argument("--python-versions", nargs="+", required=True)
    args = parser.parse_args()
    try:
        with args.pyproject.open("rb") as stream:
            pyproject = tomllib.load(stream)
        with urlopen("https://pypi.org/pypi/supervision/json", timeout=20) as response:
            releases = json.load(response)["releases"]
        matrix = build_matrix(pyproject, releases, args.python_versions)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))

    # Keep stdout machine-readable for GITHUB_OUTPUT; diagnostics go to stderr.
    print(f"Dependency: {supervision_requirement(pyproject)}", file=sys.stderr)
    for row in matrix["include"]:
        print(
            f"Python {row['python-version']}: {row['supervision-profile']} "
            f"Supervision {row['supervision-version']}",
            file=sys.stderr,
        )
    print(json.dumps(matrix, separators=(",", ":")))


if __name__ == "__main__":
    main()
