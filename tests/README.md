# Testing

From the repository root, install the package and test dependencies in your
virtual environment:

```sh
python -m pip install -e '.[test,inference]' packaging
```

`packaging` is used by the CI-helper tests. The tests never download or execute
models. On Python 3.13, the `inference` extra requires Supervision >=0.30.6.
Use only `.[test]` to test the core integration at its 0.30.0 floor; when Inference
is absent, only `test_inference.py` skips. An installed Inference package's
import failures are not skipped.

## Running tests

```sh
# All package, example, and CI-helper tests
python -m pytest -q tests .github/tests

# Package and example tests only (also the default pytest discovery)
python -m pytest -q tests

# A single integration boundary
python -m pytest -q tests/test_trackers.py

# Real-library multi-operator smoke tests
python -m pytest -q tests/test_pipelines.py

# Example-owned behavior
python -m pytest -q tests/examples

# CI matrix selection, without querying PyPI
python -m pytest -q .github/tests
```

CI runs package/example tests against the lowest and highest stable Supervision
releases permitted by `pyproject.toml` for each supported Python version. Matrix
and dependency-selection tests run separately in the setup job. Inference 1.3.8
is tested at both bounds on Python 3.10–3.12. Python 3.13 tests the core integration
at both bounds, and adds Inference 1.7.4 only to the highest-version profile.
`.github` is hidden, so its tests need an explicit path during local runs.

## Organization and scope

| Location | Responsibility |
| --- | --- |
| `test_exports.py` | Public package imports and exports |
| `test_core.py` | Detection/image adapters, filtering, stitching, smoothing |
| `test_annotators.py` | Annotator payload preservation and label callbacks |
| `test_zones.py` | Zone delegation and our tracking-timer logic |
| `test_views.py` | Display adapter, with only the GUI boundary replaced |
| `test_inspection.py` | Registration of the Supervision BGR image convention |
| `test_inference.py` | Model selection, inference options, and image normalization |
| `test_trackers.py` | Tracker construction, updates, frame handling, reset, readers |
| `test_pipelines.py` | Validated multi-operator flows using real Supervision objects |
| `examples/` | Behavior implemented in our runnable examples |
| `../.github/tests/` | CI tooling rather than runtime integrations |

Tests should verify behavior this repository owns: argument mapping, tuple
selection, conversion, validation, result identity, and preservation of inputs.
Use autospecced mocks based on real dependency signatures for forwarding tests,
and a few real-library smoke tests at important integration boundaries. Patch
model factories/inference, tracker updates, and GUI display rather than requiring
network credentials, model downloads, or interactive windows.

Do not reproduce upstream rasterization, association, history-aging, or counting
regression suites. Edge cases belong here when our code implements the behavior
(for example, `TrackingTimer` or example-owned visit analytics).

Keep fixtures in the test module that uses them. Introduce a shared `conftest.py`
only when multiple modules genuinely need the same setup; avoid global mocks that
would hide dependency API incompatibilities. Parametrize supported input forms
and wrapper options instead of copying test bodies.
