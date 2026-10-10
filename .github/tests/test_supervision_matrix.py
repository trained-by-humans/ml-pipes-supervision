import importlib.util
import json
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import pytest
from packaging.requirements import Requirement

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "supervision_matrix.py"
spec = importlib.util.spec_from_file_location("supervision_matrix", SCRIPT)
matrix_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matrix_module)


def project(requirement: str = "supervision>=0.29,<0.31") -> dict:
    return {
        "project": {
            "requires-python": ">=3.10",
            "dependencies": ["another-package>=1", requirement],
        }
    }


def artifact(
    *,
    requires_python: str | None = ">=3.10",
    yanked: bool = False,
    packagetype: str = "bdist_wheel",
) -> dict:
    return {
        "requires_python": requires_python,
        "yanked": yanked,
        "packagetype": packagetype,
    }


def test_reads_requirement_from_project_metadata() -> None:
    requirement = matrix_module.supervision_requirement(
        project("Supervision>=1.2,!=1.3,<2")
    )

    assert requirement.specifier == Requirement("supervision>=1.2,!=1.3,<2").specifier


@pytest.mark.parametrize(
    "dependencies",
    [
        [],
        ["numpy"],
        ["supervision", "supervision>=1"],
        ["supervision @ https://example.com/supervision.whl"],
    ],
)
def test_requires_one_index_based_supervision_dependency(dependencies: list[str]) -> None:
    metadata = project()
    metadata["project"]["dependencies"] = dependencies

    with pytest.raises(ValueError, match="exactly one index-based"):
        matrix_module.supervision_requirement(metadata)


def test_selects_bounds_using_pep440_order_and_constraint() -> None:
    releases = {
        version: [artifact()]
        for version in ["0.30.2", "0.29.1", "0.28.0", "0.30.10", "0.31.0", "0.29.0"]
    }

    assert matrix_module.release_bounds(
        Requirement("supervision>=0.29,!=0.29.0,<0.31"), releases, "3.12"
    ) == ("0.29.1", "0.30.10")


def test_excludes_prereleases_yanked_and_empty_releases() -> None:
    releases = {
        "invalid-version": [artifact()],
        "0.29.0": [artifact(yanked=True)],
        "0.29.1": [artifact()],
        "0.30.0": [artifact()],
        "0.30.1": [],
        "0.30.2": [artifact(yanked=True)],
        "0.30.3rc1": [artifact()],
        "0.30.4.dev0": [artifact()],
    }

    assert matrix_module.release_bounds(
        Requirement("supervision>=0.29,<0.31"), releases, "3.12"
    ) == ("0.29.1", "0.30.0")


def test_accepts_post_releases_and_unyanked_source_distributions() -> None:
    releases = {
        "0.29.0": [artifact()],
        "0.29.0.post0": [artifact(yanked=True), artifact(packagetype="sdist")],
    }

    assert matrix_module.release_bounds(
        Requirement("supervision>=0.29,<0.31"), releases, "3.12"
    ) == ("0.29.0", "0.29.0.post0")


def test_filters_python_support_per_unyanked_artifact() -> None:
    releases = {
        "0.29.0": [artifact(requires_python=None)],
        "0.30.0": [artifact(requires_python=">=3.12")],
        "0.30.1": [artifact(), artifact(requires_python=">=3.14")],
        "0.30.2": [artifact(yanked=True), artifact(requires_python=">=3.14")],
        "0.30.3": [artifact(packagetype="bdist_egg")],
    }

    assert matrix_module.release_bounds(
        Requirement("supervision>=0.29,<0.31"), releases, "3.10"
    ) == ("0.29.0", "0.30.1")


def test_builds_python_specific_matrix_from_project_constraint() -> None:
    releases = {
        "1.0.0": [artifact()],
        "1.5.0": [artifact()],
        "1.9.0": [artifact(requires_python=">=3.12")],
        "2.0.0": [artifact()],
    }

    matrix = matrix_module.build_matrix(
        project("supervision>=1,<2"), releases, ["3.10", "3.12"]
    )

    assert matrix == {
        "include": [
            {
                "python-version": "3.10",
                "supervision-profile": "lowest",
                "supervision-version": "1.0.0",
            },
            {
                "python-version": "3.10",
                "supervision-profile": "highest",
                "supervision-version": "1.5.0",
            },
            {
                "python-version": "3.12",
                "supervision-profile": "lowest",
                "supervision-version": "1.0.0",
            },
            {
                "python-version": "3.12",
                "supervision-profile": "highest",
                "supervision-version": "1.9.0",
            },
        ]
    }


def test_deduplicates_identical_bounds() -> None:
    matrix = matrix_module.build_matrix(
        project("supervision==0.30.0"), {"0.30.0": [artifact()]}, ["3.12"]
    )

    assert matrix["include"] == [
        {
            "python-version": "3.12",
            "supervision-profile": "lowest/highest",
            "supervision-version": "0.30.0",
        }
    ]


def test_fails_when_no_eligible_releases_exist() -> None:
    with pytest.raises(ValueError, match="No non-yanked stable release.*Python 3.12"):
        matrix_module.release_bounds(
            Requirement("supervision>=0.29,<0.31"),
            {"0.29.0": [artifact(yanked=True)]},
            "3.12",
        )


def test_rejects_python_version_outside_project_support() -> None:
    with pytest.raises(ValueError, match="Project does not support Python 3.9"):
        matrix_module.build_matrix(project(), {"0.29.0": [artifact()]}, ["3.9"])


def test_evaluates_dependency_markers_for_target_python() -> None:
    requirement = Requirement('supervision>=0.29; python_version >= "3.12"')
    releases = {"0.29.0": [artifact()]}

    assert matrix_module.release_bounds(requirement, releases, "3.12") == (
        "0.29.0", "0.29.0"
    )
    with pytest.raises(ValueError, match="does not apply to Python 3.10"):
        matrix_module.release_bounds(requirement, releases, "3.10")


def test_cli_reads_toml_and_emits_only_matrix_json_on_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    metadata = tmp_path / "pyproject.toml"
    metadata.write_text(
        '[project]\nrequires-python = ">=3.10"\n'
        'dependencies = ["supervision>=1,<2"]\n',
        encoding="utf-8",
    )
    releases = {"1.0.0": [artifact()], "1.5.0": [artifact()]}
    fetch = Mock(return_value=BytesIO(json.dumps({"releases": releases}).encode()))
    monkeypatch.setattr(matrix_module, "urlopen", fetch)
    monkeypatch.setattr(
        matrix_module.sys, "argv",
        [str(SCRIPT), "--pyproject", str(metadata), "--python-versions", "3.12"],
    )

    matrix_module.main()

    captured = capsys.readouterr()
    assert json.loads(captured.out) == matrix_module.build_matrix(
        project("supervision>=1,<2"), releases, ["3.12"]
    )
    assert "Dependency: supervision<2,>=1" in captured.err
    fetch.assert_called_once_with("https://pypi.org/pypi/supervision/json", timeout=20)


def test_cli_fails_on_index_error_without_emitting_a_fallback_matrix(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(matrix_module, "urlopen", Mock(side_effect=OSError("offline")))
    monkeypatch.setattr(
        matrix_module.sys, "argv",
        [str(SCRIPT), "--pyproject", str(SCRIPT.parents[2] / "pyproject.toml"),
         "--python-versions", "3.12"],
    )

    with pytest.raises(SystemExit) as error:
        matrix_module.main()

    assert error.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "offline" in captured.err
