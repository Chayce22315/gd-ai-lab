# GD AI Lab: decoNET + gdCORE

**two new neural networks, trained from randomly initialized weights.** this repository is a practical first training lab for Geometry Dash-specific tasks, built around PyTorch, GitHub, and free GPU notebook sessions.

- **decoNET** learns a structured language for decoration direction: palettes, section identity, motifs, depth, lighting, transitions, object-role planning, and readability.
- **gdCORE** learns structured plans for platformer/classic layouts, title screens, shop UI, and careful trigger plans grounded in a small Geometry Dash 2.2081 reference sheet.

There are no API keys, hosted inference endpoints, downloaded pretrained weights, or paid model services in this training project. Model parameters are initialized randomly.

> **scope warning:** these are small research prototypes trained on synthetic starter examples. they will not instantly act like frontier models, know all of GD 2.2081, or guarantee featured-worthy levels. the first milestone is learning a narrow structured-output task; making them genuinely strong requires a much larger, curated and verified dataset, iterative evaluations, an object serializer, and real in-game testing.

## 1. put the project on GitHub

1. open [create a GitHub repository](https://github.com/new).
2. name it `gd-ai-lab`. choose **Public** if you want cloud notebooks to `git clone` it without authentication. Private is possible, but notebooks need authenticated repository access.
3. do not initialize the repository with a README or license (this folder already has both).
4. install Git for Windows if necessary, extract this project, open Command Prompt in the `gd-ai-lab` folder, then run:

```bat
git init
git branch -M main
git add .
git commit -m "bootstrap decoNET and gdCORE from scratch"
git remote add origin https://github.com/YOUR_USERNAME/gd-ai-lab.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username. Do not commit checkpoints, API keys, private data, or large generated runs. The `.gitignore` excludes common training outputs.

## 2. train on a free notebook GPU

Notebook files are in `notebooks/`:

- `01_train_deconet.ipynb`
- `02_train_gdcore.ipynb`

You can use [Google Colab](https://colab.research.google.com/) or [Kaggle Notebooks](https://www.kaggle.com/code/). GPU access, available GPU type, and weekly/session quotas vary by account and time. A free GPU session can end unexpectedly, so keep exported checkpoints.

1. open a notebook in Colab or Kaggle.
2. in the first code cell, replace `YOUR_USERNAME` with the GitHub owner of your repository.
3. choose a GPU runtime in notebook settings.
4. run the cells from top to bottom.
5. save the generated checkpoint bundle when training finishes. **do not commit `.pt` checkpoints to Git**; keep them in notebook outputs or cloud drive instead.

The notebooks train one model at a time. On an 8 GB GPU, use a small batch with gradient accumulation. If you hit CUDA out-of-memory, lower batch size to 1 or 2, then lower the context length; don't change model size until the baseline works.

## 3. local setup (Windows)

Python 3.10+ is required. A CUDA-enabled PyTorch install is needed to use the RTX GPU; choose the install command recommended by the [official PyTorch selector](https://pytorch.org/get-started/locally/) for your Windows/CUDA configuration. CPU training is supported but much slower.

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[test]"
python scripts/build_dataset.py --samples-per-model 300 --output-dir data/generated
```

Quick smoke-train a tiny model (uses CPU if CUDA isn't available):

```bat
python -m gdai.train --model deconet --train-file data/generated/deconet_train.jsonl --val-file data/generated/deconet_val.jsonl --output-dir runs/deconet-smoke --epochs 1 --batch-size 1 --gradient-accumulation 1 --max-seq-len 512 --d-model 96 --layers 2 --heads 4 --max-train-batches 5
```

For an actual GPU run, start with the notebook settings (`d_model=192`, `layers=4`, `heads=6`, sequence length 1,536) and increase training duration after confirming loss decreases. This is an experiment, not a pre-trained release.

## 4. generate samples

After a checkpoint exists, decoNET can save both its structured plan and a self-contained SVG art-direction preview:

```bat
python -m gdai.generate --model deconet --checkpoint runs/deconet/checkpoint_best.pt --prompt "modern glow, abandoned orbital greenhouse, cyan and amber, restrained opening then explosive drop" --json-output runs/deconet/sample_plan.json --svg-output runs/deconet/decoration_preview.svg
```

The SVG is a **concept preview rendered from decoNET's semantic plan**. It is not Geometry Dash editor data and cannot be imported as a `.gmd`; the plan still needs a verified object-to-GD builder. The decoNET notebook displays this SVG preview after generation. [Open the included example SVG](examples/deconet-concept-preview.svg) and [its matching JSON plan](examples/deconet-plan-example.json); these example files come from the synthetic starter-template generator, not from a trained model.

```bat
python -m gdai.generate --model gdcore --checkpoint runs/gdcore/checkpoint_best.pt --task platformer --prompt "abandoned laboratory escape with vertical rooms, optional coin routes, and a clear exit"
```

The first generated responses may be invalid JSON or repetitive. That's normal for a tiny randomly initialized model with a small corpus. Use the evaluation command to measure it rather than judging one lucky output.

## 5. validate progress

```bat
pytest
python -m gdai.evaluate --model deconet --checkpoint runs/deconet/checkpoint_best.pt --data data/generated/deconet_val.jsonl --limit 20
```

Track random-init baseline vs training/validation losses, JSON-valid rate, schema-valid rate, and human-review ratings. A decreasing loss is necessary but not sufficient.

## files

[view the SVG architecture diagram](assets/architecture.svg)


```text
src/gdai/model.py             decoder Transformer initialized from scratch
src/gdai/train.py             masked prompt-to-completion training loop
src/gdai/tokenizer.py         fixed byte tokenizer; no learned/pretrained vocab
src/gdai/dataset_builder.py   reproducible synthetic training data generator
src/gdai/generate.py          local checkpoint inference + plan/SVG outputs
src/gdai/render_svg.py        safe standalone SVG concept preview renderer
src/gdai/evaluate.py          JSON/schema evaluation
knowledge/                    conservative starter GD 2.2081 notes
notebooks/                    separate GPU notebooks for both specialists
.github/workflows/tests.yml   CPU test suite on GitHub Actions
```

## recommended next milestones

1. prove both networks train and beat the random-init baseline on held-out schema examples.
2. gather high-quality human-authored examples and define objective visual/gameplay tests.
3. expand the verified 2.2081 object/trigger catalog with sources and version fields.
4. connect decoNET's semantic placements and gdCORE's game plans to the `.gmd` builder.
5. import generated levels into Geometry Dash, capture images, and measure playability/readability; use those results to improve training data.

## references

- PyTorch installation selector: https://pytorch.org/get-started/locally/
- Colab: https://colab.research.google.com/
- Kaggle Notebooks: https://www.kaggle.com/code/
- Geometry Dash trigger overview: https://geometrydash.wiki.gg/wiki/Triggers
- GDCreatorSchool ID guide: https://github.com/GDCreatorSchool/gdcs2/blob/main/content/docs/guides/the-editor/using-ids.md
- Item Edit / Compare / Persistence: https://github.com/GDCreatorSchool/gdcs2/blob/main/content/docs/guides/triggers-1/item-edit-comp-pers.md
