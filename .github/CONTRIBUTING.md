# Contributing

Thanks for contributing to `ml-pipes-supervision`.
Honestly, just the fact that you opened this page makes me happy! ^^

## Install the repository

Clone the repository, create and activate a Python 3.10+ virtual environment,
then install the package in editable mode:

```bash
git clone https://github.com/trained-by-humans/ml-pipes-supervision.git
cd ml-pipes-supervision
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Tests

Install the test dependencies, then run the full package, example, and CI-helper
suite from the repository root before submitting a change:

```bash
python -m pip install -e '.[test,inference]' packaging
python -m pytest -q tests .github/tests
```

Add or update tests in `tests/` for behavior changes. While developing, run a
single test module with:

```bash
python -m pytest tests/test_core.py -q
```

See [the testing guide](../tests/README.md) for environment setup, targeted test
commands, suite organization, and scope, including the CI-helper tests.
On Python 3.13, optional Inference tests require Supervision >=0.30.6;
use `.[test]` for core-only tests at the package's 0.30.0 floor.
The CI-helper tests need the explicit `.github/tests` path during local runs.

Keep tests independent of network credentials, model downloads, and interactive
windows by mocking model inference and GUI boundaries. Test the behavior this
integration owns rather than duplicating upstream regression suites.

The suite may emit deprecation warnings from third-party dependencies; these
warnings do not currently fail the test run.

## Pull requests

Keep changes focused and include tests for behavior changes. Update the
documentation and `docs/coverage.md` when a change adds or changes a public
Supervision compatibility wrapper.

## Documentation

Use lowercase filenames and directory names for documentation guides in
`docs/` and package-local documentation directories. Keep existing underscores
and conventional filenames such as `README.md`, `AGENTS.md`, and `SKILL.md`.
Visible page titles and navigation labels retain their normal capitalization.
Update Markdown links, package documentation URLs, and cross-site links when
renaming a guide; URLs follow source filename casing.

The `Pages` workflow tests the naming checks and builds the site on pull
requests and `main`, but deploys only from `main`. It checks filenames before
building and generated URLs afterward. To run these checks locally:

```bash
python .github/scripts/check_docs_paths.py
python -m mkdocs build --strict --site-dir _site
python .github/scripts/check_docs_paths.py --site-dir _site
```

Preview the documentation site locally from the package root:

```bash
python -m pip install -e '.[docs]'
python -m mkdocs serve
```

Open `http://127.0.0.1:8000` in a browser. Use the following command to
validate the static site before submitting documentation changes:

```bash
python -m mkdocs build --strict
```

GitHub Actions deploys documentation to GitHub Pages when changes are pushed
to `main`; contributors do not need to deploy it manually.
