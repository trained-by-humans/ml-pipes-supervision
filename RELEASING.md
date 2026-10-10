# Releasing `ml-pipes-supervision`

`ml-pipes-supervision` is an independently versioned, opt-in community
distribution. It is not part of the `ml-pipes` umbrella package's default or
`all` install. This keeps its Supervision, tracker, and optional Inference
runtimes separate from the core framework installation.

This contract is shared with
[`ml-pipes-ultralytics`](https://github.com/trained-by-humans/ml-pipes-ultralytics/blob/main/RELEASING.md).

## Prepare a release

Open a release-preparation pull request that updates, as applicable:

- the package version in `pyproject.toml`;
- framework and upstream dependency bounds;
- release notes or changelog material;
- README and documentation installation snippets; and
- compatibility and package tests.

For release candidates, exact pins to the tested framework candidate are
allowed. Stable releases should use documented, tested bounds rather than
permanent exact framework pins.

CI must pass on Python 3.10 through 3.13. It runs package tests, builds both
the wheel and source distribution, runs `twine check`, and installs the built
wheel in a clean environment for an import and dependency check.

The stable `Compat: Gate` check aggregates matrix resolution and
every compatibility matrix test. It succeeds only when those jobs succeed;
failed, cancelled, or skipped jobs block it. Configure GitHub PR rules to
require this check rather than the versioned matrix job names, and require
`Build and validate distribution` separately. Future standalone jobs are not
included automatically and need their own required checks when applicable.

## Stage on TestPyPI

After merging, create an annotated tag named `v<package-version>`, for example
`v0.1.2rc2`. Pushing that tag starts the `Stage` workflow:

1. It checks that the tag matches `project.version` and runs CI.
2. It builds the wheel and source distribution, validates them, and records
   their SHA-256 checksums.
3. It creates the GitHub Release containing those exact artifacts.
4. It downloads those release assets, verifies their checksums, publishes only
   the wheel and source distribution to TestPyPI using OIDC, and verifies a
   clean TestPyPI installation.

The TestPyPI trusted publisher must be configured with:

```text
Owner:       trained-by-humans
Repository:  ml-pipes-supervision
Workflow:    .github/workflows/stage-release.yml
Environment: test-pypi
```

TestPyPI never permits reusing an uploaded filename, including after deletion.
If a candidate must be rebuilt, increment its prerelease version instead of
retagging it with the same package version.

## Promote to PyPI

After the TestPyPI verification succeeds, manually dispatch the `Release`
workflow *from the staged tag itself*. The workflow rejects branch refs,
downloads the GitHub Release assets, verifies their checksums, and publishes
those same artifacts to PyPI using OIDC. The `pypi` environment supplies the
required deployment review.

The PyPI trusted publisher must be configured with:

```text
Owner:       trained-by-humans
Repository:  ml-pipes-supervision
Workflow:    .github/workflows/publish-release.yml
Environment: pypi
```

This deliberately is not a workflow triggered by a GitHub Release event. The
tag-triggered staging workflow creates the Release using `GITHUB_TOKEN`, and
that token does not reliably trigger a second release-event workflow. The
manual, tag-ref promotion keeps production publication explicit, approved, and
artifact-identical to staging.

## Release authority

TBH organization administrators may approve release preparation, create
release tags, and approve the `pypi` deployment. Repository rules protect
`v*` tags and require review and CI for changes to `pyproject.toml` and
`.github/workflows/`. Community contributors retain normal pull-request access
but do not receive publication authority.
