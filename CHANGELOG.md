# Changelog

## 0.1.5

### Changed

- Extend the Supervision ceiling from `<0.31` to `<0.32`, retaining the
  `0.30.0` floor for optional Inference compatibility.
- Separate the generic Supervision suite from the additional `inference-compat`
  suite on Python 3.10–3.13. The `test` extra no longer installs Inference;
  optional tests use `.[test,inference]` and independently resolve a compatible
  Supervision release. Both matrices feed `Compat: Gate`.
- Adapt invalid-coordinate fixtures to Supervision 0.31's construction-time
  validation while retaining clipping, dropping, cropping, and alignment
  checks. The default clip policy now follows upstream rejection of
  nonfinite output coordinates on 0.31.

[Full comparison with 0.1.4](https://github.com/trained-by-humans/ml-pipes-supervision/compare/v0.1.4...v0.1.5)
