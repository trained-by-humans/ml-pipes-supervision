# Detect and attribute: local Qwen attribute selection

This follows [detect-and-describe](../docs/recipes/detect_and_describe.md), using local
Qwen3-VL to assign an attribute from supplied candidate labels rather than
displaying free-form text. Label names and descriptions guide the prompt;
answers outside the supplied set are marked `unknown`. The pipeline uses only
local Qwen for attributes, with no CLIP or Bedrock calls.

## Run

From the repository root:

```bash
python -m pip install -e '.[inference]'
python -m pip install "transformers>=4.57.0,<6" "torch>=2.6" "Pillow>=10"
python examples/run_detect_and_attribute.py
python examples/run_detect_and_attribute.py --input people.jpg --attribute activity --labels walking standing sitting --output-image annotated.jpg --output results.json
```

The Qwen runtime is example-specific, not a dependency or extra of the package.
Install it explicitly with the second command above. On startup, the CLI checks
the required imports and exits with an install message if they are missing or
incompatible, before downloading inputs or loading models. `--help` still works
without Qwen installed.

By default, the example classifies fashion style using the attribute and
described labels below. Both `--attribute` and `--labels` can be overridden;
each accepts `name: description`:

```bash
python examples/run_detect_and_attribute.py \
  --attribute "fashion style: overall visual style of the outfit" \
  --labels \
    "casual: relaxed everyday clothing" \
    "formal: office attire, such as a suit or dress shirt" \
    "sportswear: athletic clothing, such as shorts or a windbreaker"
```

Each argument is split at its **first** colon, trimming surrounding spaces.
The description is included in Qwen's prompt; the name is retained in results.
Without a colon, the name also serves as its description. Plain and described labels
can be mixed. Quote each complete argument, with no separating commas between
items; commas and additional colons inside descriptions are preserved. Names
must be non-empty, label names must be unique, and a colon must be followed by
a non-empty description.

The default input is Supervision's people-walking image, downloaded to
`examples/.example_assets` if missing. RF-DETR Small detects people locally.
Use `--class-ids`, `--detection-model-id`, `--confidence`, and `--padding` just
as in the describe example. For instance, use `--class-ids 2 --attribute condition
--labels new worn damaged` for bicycles (RF-DETR category IDs).

Qwen defaults to `Qwen/Qwen3-VL-2B-Instruct` on CPU. Use `--qwen-model-id` for
another compatible model or local path, `--device cpu|mps|cuda|cuda:N` for its
device, and `--max-tokens` for the generated-token budget (default: 256).
Weights and the processor load lazily on the first retained crop and are reused
across images. The first use downloads weights from Hugging Face unless cached;
inference is local. See the detect-and-describe guide for memory requirements.
`--device` controls Qwen only, not the detector.

Detection may require a Roboflow key via `ROBOFLOW_API_KEY` or `--api-key`.
This key is used only by detection, not Qwen. No AWS credentials are needed.

## Pipeline and results

The example uses public `Detections.ScaleBoxes`, `Detections.ClipBoxes`, and
`Detections.Crop`. It drops
invalid crop regions and uses the surviving padded, image-clipped boxes for
annotation. Detection, class/confidence filters, NMS, scatter/gather,
annotation, display, and optional JSON/image outputs follow the same structure.
The stages mirror the describe pipeline; the prompt and response attachment differ:

- The prompt includes the attribute and every candidate's name and description,
  with instructions to copy exactly one label name onto the first line and
  explain briefly afterward. Shared grounding instructions discourage guessing
  hidden attributes and allow `unknown` when visible evidence is insufficient.
- `ConvertColorSpace(output_color_space="RGB")` supplies an RGB `ImagePayload`
  directly to `LocalQwen3VL`. Detection and annotation still receive BGR pixels.
  Crops retain their source resolution; the model processor handles image sizing,
  normalization, and patch alignment.
- `AttachAttributeLabels` strips surrounding whitespace from the first line and
  accepts it only if it exactly matches a supplied label name. Unrecognized,
  malformed, or incomplete labels become `unknown`, even if `unknown` was not
  supplied as a candidate. Full responses are preserved for inspection.

The annotated image is always displayed, with **only the selected label name**,
such as `casual`; attribute names, descriptions, and scores are not annotated.
`build_pipeline()` uses the same fashion-style defaults; override them with
`build_pipeline(attribute="...", labels=(...))`. The pipeline
returns `(annotated_image, detections)`. Attribute fields are stored in
`detections.data["attribute"]`, `["label"]`, and `["response"]`. Original
detector classes, confidence, and boxes are preserved.

JSON is printed and optionally saved with `--output`. Each entry contains
`detection_index`, source-image `xyxy`, detector `class_id`/`confidence`, and
the short `attribute` name, selected `label` name, and full `response`.
There is no CLIP `similarity` score or synthesized Qwen confidence score.
`--output-image` saves the annotated image. If no detections survive
filtering/cropping, Qwen is not loaded or called.

The `CLIPClassify` operator and its `AttachClassifications` adapter remain in the
example for separate experiments. Neither is instantiated by `build_pipeline()`.
The CLI uses only Qwen and no longer exposes `--clip-model-id`. CLIP is not a
dependency or extra of this package. Install its runtime explicitly, using the
same Inference release as the detector integration:

```bash
python -m pip install "inference[clip]==1.3.8; python_version < '3.13'" "inference[clip]==1.7.4; python_version >= '3.13'"
```

CLIP experiments can check dependencies before loading inputs or models:

```python
from examples.run_detect_and_attribute import require_clip

require_clip()
```

If the CLIP backend cannot be imported, this prints an install command on stderr
and exits with status 1, without a traceback. The command pins the currently
installed Inference version so enabling CLIP does not upgrade the detector SDK.
`CLIPClassify` also checks on its first crop and raises an `ImportError` with
the same guidance for programmatic callers. Checks do not load model weights.
Neither check runs in the Qwen pipeline; it continues to work without CLIP.
CLIP experiments still need their own preprocessing: native-size letterboxing
and BGR arrays for Roboflow Inference.

## Limitations

Qwen generates text; this is prompt-based selection followed by label validation,
not constrained decoding or a calibrated classifier. Validation prevents novel
labels from leaking into annotations, but does not verify the model's visual
judgment. Wording and the candidate set affect predictions; evaluate with
labeled examples. Abstention is encouraged, not guaranteed to be correct.

Non-empty truncated responses are retained. A complete valid first-line label
still works if its explanation is cut off; a truncated or invalid label becomes
`unknown`. Empty answers and generation failures still raise errors, as in
the describe example. Calls are sequential, and CPU generation can be slow.
Keep descriptions short and visually grounded; longer prompts do not guarantee
better accuracy.

Use visually distinguishable attributes. Actual occupation, tourist/resident
status, and other hidden personal attributes cannot reliably be established
from appearance; neither an attribute name nor a candidate list changes that.
