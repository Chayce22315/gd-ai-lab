# Architecture

## Two independent networks, initialized from random weights

`decoNET` and `gdCORE` are separate decoder-only Transformer models. Each has a new, randomly initialized parameter set, its own optimizer state, its own generated training/evaluation files, and its own checkpoint. No pretrained base model, hosted inference API, or downloaded language-model weights are used.

The starter tokenizer is a deterministic UTF-8 byte vocabulary (256 byte tokens plus five control IDs). It is deliberately simple and fully local. The cost is longer sequences than a learned BPE tokenizer.

## Pipeline

1. Hand-written domain templates create initial supervised prompt/JSON-completion examples.
2. The trainer predicts only response tokens; prompt tokens are masked out of the loss.
3. Every run records its random-initialization baseline, training loss, validation loss, model dimensions, parameter count, seed, and checkpoint.
4. Generation runs locally from the checkpoint.
5. Lightweight validators check JSON shape and schema rules. They do not prove gameplay feasibility, visual quality, valid object serialization, or feature-worthiness.
6. Future milestones can connect these plans to GD Forge's serializer, a verified 2.2081 catalog, screenshot critique, and real import/play tests.

## Why this model is intentionally small

Default configuration: 192 hidden dimensions, four Transformer blocks, six attention heads, and a 1,536-byte sequence window. Parameter count is around a few million, depending on exact implementation. This is a learning/engineering baseline that should fit in an 8 GB VRAM environment with a small batch. It is not a substitute for a general-purpose language model.

## Honest evaluation

The first corpus is synthetic and generated from hand-authored templates. A lower validation loss shows the network learned the generated distribution; it does not show that it can design compelling levels or correctly implement every GD 2.2 trigger. Before increasing model size, build a reviewed corpus containing valid examples, negative examples, and objective evaluation tasks.


## decoNET SVG preview

`gdai.render_svg` turns a valid semantic decoNET plan into a self-contained SVG art-direction preview. This helps inspect palette and section-composition ideas without a raster renderer or external image service. The SVG remains a visualization only: the Geometry Dash builder must map semantic roles through a versioned object catalog and serializer before producing importable editor data.
