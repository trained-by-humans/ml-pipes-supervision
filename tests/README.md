# Testing

The generic Supervision suite and optional `inference-compat` suite use separate
virtual environments so Inference can select a compatible Supervision release
without constraining the generic suite.

## Running tests

From the repository root, install and run the generic suite:

```sh
python -m pip install -e '.[test]' packaging
python -m pytest -q tests --ignore=tests/inference_compat
python -m pytest -q .github/tests
```

`packaging` is used by the CI-helper tests. The tests never download or execute
models. To test a specific supported release, add an exact `supervision` pin
to the install command; CI resolves the lowest and highest permitted releases
automatically.

In a separate virtual environment, install the optional Inference dependency
and run its compatibility suite:

```sh
python -m pip install -e '.[test,inference]'
python -m pip check
python -m pytest -q tests/inference_compat
```

Let pip resolve Supervision for this environment; optional Inference may
require an older supported release. The public install extra is `inference`;
`inference-compat` names the additional CI suite.

Other useful commands:

```sh
# Combined package/example suite, using this environment's resolved versions
python -m pytest -q tests

# A single integration boundary
python -m pytest -q tests/test_trackers.py

# Real-library multi-operator smoke tests
python -m pytest -q tests/test_pipelines.py

# Generic example-owned behavior
python -m pytest -q tests/examples

# CI matrix selection, without querying PyPI
python -m pytest -q .github/tests
```

Default pytest discovery includes both suites. Optional wrapper and dependent
example tests skip when Inference is absent; import failures from an installed
Inference package fail the tests. `.github` is hidden, so its tests need an
explicit path during local runs.

CI runs the generic suite against the lowest and highest stable Supervision
releases permitted by `pyproject.toml` on Python 3.10–3.13. Independently, the
`inference-compat` matrix runs only the optional suite on the same Python
versions, with dependencies resolved from `.[test,inference]`. Matrix and
dependency-selection tests run in the setup job. Both compatibility matrices
and matrix resolution feed `Compat: Gate`.

Supervision 0.31 validates finite coordinates when constructing detections.
Tests for our invalid-row handling corrupt an already constructed object's
coordinates so they still exercise the operator boundary on both series.

## Organization and scope

| Location | Responsibility |
| --- | --- |
| `test_exports.py` | Public package imports and exports |
| `test_core.py` | Detection/image adapters, filtering, stitching, smoothing |
| `test_attach_data.py` | Single-/multi-field data alignment, nested-value preservation, atomic collision policy, non-mutation, and validated pipelines |
| `test_transformations.py` | Detection-aware box geometry, rounding/invalid policies, aligned crops, concatenation, and validated pipelines |
| `test_annotators.py` | Annotator payload preservation and label callbacks |
| `test_zones.py` | Zone delegation and our tracking-timer logic |
| `test_views.py` | Display adapter, with only the GUI boundary replaced |
| `test_inspection.py` | Registration of the Supervision BGR image convention |
| `examples/test_qwen_local.py` | Example-only local Qwen RGB payload validation, lazy loading, device selection, and generation boundaries |
| `test_trackers.py` | Tracker construction, updates, frame handling, reset, readers |
| `test_pipelines.py` | Validated multi-operator flows using real Supervision objects |
| `examples/` | Generic example behavior without optional Inference |
| `inference_compat/` | Optional Inference wrappers and examples that depend on them |
| `inference_compat/test_inference.py` | Model selection, inference options, and image normalization |
| `../.github/tests/` | CI tooling rather than runtime integrations |

Tests should verify behavior this repository owns: argument mapping, tuple
selection, conversion, validation, result identity, and preservation of inputs.
Use autospecced mocks based on real dependency signatures for forwarding tests,
and a few real-library smoke tests at important integration boundaries. Patch
model factories/inference, tracker updates, and GUI display rather than requiring
network credentials, model downloads, or interactive windows.
Local Qwen unit tests mock its optional runtime modules, so they also run without
PyTorch/Transformers installed. Example tests exercise real resize/color-conversion
operators and tracking caches while replacing generation.

Do not reproduce upstream rasterization, association, history-aging, or counting
regression suites. Edge cases belong here when our code implements the behavior
(for example, `TrackingTimer` or example-owned visit analytics).

Keep fixtures in the test module that uses them. Introduce a shared `conftest.py`
only when multiple modules genuinely need the same setup; avoid global mocks that
would hide dependency API incompatibilities. Parametrize supported input forms
and wrapper options instead of copying test bodies.
