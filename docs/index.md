# ml-pipes-supervision

`ml-pipes-supervision` provides [Roboflow Supervision](https://github.com/roboflow/supervision)
capabilities as composable operators in [ml-pipes](https://github.com/trained-by-humans/ml-pipes).
Supervision remains the source of detection, annotation, and zone logic.

Install from PyPI in a Python 3.10+ environment:

```bash
python -m pip install ml-pipes-supervision
```

The package installs the required `ml-pipes` core and vision packages, plus
the Supervision and tracker runtime dependencies. To use `RoboflowInference`,
install the optional integration with
`python -m pip install "ml-pipes-supervision[inference]"`.

Start with [Detect and Annotate](tutorials/detect_and_annotate.md) for a
pipeline-oriented port of a Supervision guide. See [Reference](reference.md)
for the public surface and [Coverage](coverage.md) for compatibility status.
