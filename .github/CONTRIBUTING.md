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

Add or update focused tests for behavior changes. Keep generic Supervision
tests independent of optional Inference, and test our integration boundaries
without model downloads or interactive windows.

See [the testing guide](../tests/README.md) for environment setup, suite
commands, CI-helper checks, and testing scope.

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
