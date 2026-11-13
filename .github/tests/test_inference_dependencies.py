"""Regression checks for optional Inference dependency selection."""

import tomllib
from pathlib import Path

import pytest
from packaging.markers import default_environment
from packaging.requirements import Requirement

PROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def active_requirements(extra: str, python_version: str) -> dict[str, Requirement]:
    with PROJECT.open("rb") as stream:
        metadata = tomllib.load(stream)
    environment = default_environment()
    environment.update(
        python_version=python_version,
        python_full_version=f"{python_version}.0",
    )
    requirements = [
        Requirement(value)
        for value in metadata["project"]["optional-dependencies"][extra]
    ]
    return {
        requirement.name: requirement
        for requirement in requirements
        if requirement.marker is None or requirement.marker.evaluate(environment)
    }


@pytest.mark.parametrize(
    ("python_version", "inference_version"),
    [("3.10", "1.3.8"), ("3.12", "1.3.8"), ("3.13", "1.7.4")],
)
def test_inference_extra_selects_supported_release(
    python_version: str, inference_version: str,
) -> None:
    requirement = active_requirements("inference", python_version)["inference"]

    assert str(requirement.specifier) == f"=={inference_version}"


@pytest.mark.parametrize("python_version", ["3.10", "3.11", "3.12", "3.13"])
def test_core_test_extra_does_not_install_inference(python_version: str) -> None:
    requirements = active_requirements("test", python_version)

    assert "pytest" in requirements
    assert "inference" not in requirements


def test_example_only_qwen_runtime_is_not_declared_as_package_dependencies():
    with PROJECT.open("rb") as stream:
        project = tomllib.load(stream)["project"]

    extras = project["optional-dependencies"]
    assert "local-qwen" not in extras
    requirements = project["dependencies"] + [value for values in extras.values() for value in values]
    names = {Requirement(value).name.lower() for value in requirements}
    assert not names.intersection({"transformers", "torch", "pillow"})


def test_bedrock_runtime_is_not_declared_as_package_dependencies():
    with PROJECT.open("rb") as stream:
        project = tomllib.load(stream)["project"]

    extras = project["optional-dependencies"]
    assert "bedrock" not in extras
    requirements = project["dependencies"] + [value for values in extras.values() for value in values]
    names = {Requirement(value).name.lower() for value in requirements}
    assert not names.intersection({"boto3", "botocore"})
