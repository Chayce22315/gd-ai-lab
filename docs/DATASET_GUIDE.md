# Dataset guide

## Current corpus

The starter generator creates synthetic examples with fixed, inspectable rules. It is intentionally transparent and reproducible. It does not scrape rated Geometry Dash levels or claim to have human-rated quality labels.

- `decoNET`: art-direction metadata, palettes, section plans, semantic decoration roles, abstract anchor points, layer intent, transitions, and readability rules.
- `gdCORE`: platformer room plans, classic section plans, title-screen/shop layouts, and cautious trigger plans.

The current dataset is a scaffold, not enough data for a powerful expert model. Closely related synthetic records can leak across the shuffled train/validation split; use the split as a training smoke test only.

## How to improve it

Add curated JSONL records with fields `prompt`, `response`, optional `task_type`, and optional `source` metadata. Keep source, license, game version, human review, and validation status in a separate metadata field or sidecar. Do not upload private data or copyrighted level exports unless you have the rights to use them.

For decoNET, label why a composition is strong or weak, explain color/contrast decisions, identify the section's musical purpose, and note where the route must remain readable.

For gdCORE, include minimal verified trigger examples, symbolic ID ledgers, successful and failing mechanics, reset cases, platformer reachability, and imported `.gmd` test results. Record the exact GD version and a source for every exact field/ID fact.

## Data quality checks to add next

- strict JSON parse
- schema validation
- numeric bounds and coordinate checks
- no overlap with spawn-safe zone
- semantic catalog terms resolve to verified object entries
- no unverified numeric IDs
- train/validation splits grouped by unique level or scenario rather than random rows
- human visual review and imported-gameplay labels

## Compact output key map

The byte-level starter models learn compact JSON to stay inside a small context window. `gdai.normalize.normalize_output()` expands it before later tools consume it.

- decoNET section fields: `n` = name, `x` = abstract x-range, `m` = motif, `l` = lighting, `t` = transition. Decoration objects use `xy` as an abstract anchor, `color` as a palette role, and `lookup_required: true` to force a verified catalog lookup.
- gdCORE platformer rooms use `x` for abstract room range, `p` for platform triples `[x1, x2, y]`, `h` for hazard triples `[kind, x, y]`, and `optional_coins` for optional collectible anchors. `normalize_gdcore()` expands these into descriptive fields.
- UI rectangles are normalized proportions from 0 to 1, not pixel coordinates.

These are planning coordinates, not yet exact Geometry Dash editor units or serialized object placements.
