# design2code

A small vision-language model that turns an Android app screenshot into **Jetpack Compose** code.

This is a personal learning project. The goal was to go through the whole ML pipeline end to end on one
consumer GPU (RTX 4060, 8 GB): find data, clean it, build training pairs, fine-tune a model and evaluate it.

![Example](reports/compare/007_458632.png)
<sub>Left to right: original screenshot · rule-based converter · fine-tuned model · base Qwen (not fine-tuned)</sub>

## Results

Evaluated on **200 screens from 200 apps** that never appear in training:

| | Compiles | Visual similarity (mean) | Texts recovered |
|---|---|---|---|
| Rule-based converter (upper bound) | 100% | 0.80 | 100% |
| **Fine-tuned model (5,000 samples)** | **92%** | **0.43** | **79%** |
| Base Qwen3.5-4B (not fine-tuned) | 7.5–10% | 0.01 | 37–60% |

- All 16 failures of the fine-tuned model hit the 4,096-token output limit and were cut off.
- The base model often gets stuck repeating `import` lines.
- Every model gets the same leniency at compile time: standard imports are added automatically,
  `@Preview` is stripped, and Material3 "experimental" APIs are opted in.
- 12 random (not cherry-picked) examples: [`reports/compare/`](reports/compare). Full numbers:
  [`reports/eval_compare.json`](reports/eval_compare.json).

**Limitations:**
- The code uses fixed sizes and positions for every element (e.g. `size(338.dp, 37.dp)`). The model learned this
  style from the converter, so layouts break on other screen sizes.
- Images and icons become plain colored boxes.
- The similarity metric can be harsh: a screen that looks close may still get a low score.

## How it works

```
MobileViews (122,301 screens: screenshot + view hierarchy)
  │  scripts/clean.py          cleaning → 81,048 screens from 3,722 apps
  ▼
  │  scripts/vh2compose.py     view hierarchy → Compose code (rule-based converter)
  │  scripts/run_batch.py      Gradle + Paparazzi: compile the code and render it
  ▼
80,726 pairs: (screenshot, Compose code, render, similarity)
  │  scripts/make_sft_split.py 5,000 training + 200 test screens (no shared apps)
  ▼
  │  scripts/train_sft.py      Qwen3.5-4B, QLoRA (4-bit, r=16), 625 steps, 3.8 h on an RTX 4060
  ▼
  │  scripts/eval_model.py     generate code → compile → render → similarity
  ▼
Results table
```

Data: [MobileViews](https://huggingface.co/datasets/mllmTeam/MobileViews) (MIT); only one shard
(400000–522301) is used. No code written by other AI models is used for training: every training
sample comes from the rule-based converter.

## Install

Requirements: Linux (x86_64), an NVIDIA GPU with 8 GB+, Python 3.11+, `curl`, `unzip`, ~25 GB of free disk.

```bash
git clone https://github.com/q0d1r0v/design2code.git
cd design2code
./setup.sh
```

`setup.sh` installs everything without sudo: Python packages, Java 17, the Android SDK (if missing),
the Qwen3.5-4B base model (~9.3 GB) and the fine-tuned adapter
([Releases](https://github.com/q0d1r0v/design2code/releases)). It is safe to re-run: finished parts are skipped.

## Try it

```bash
./try.sh examples/screen1.jpg
```

Results go to `output/`:
- `screen1.kt`: the code written by the model;
- `screen1_compare.png`: the original on the left, the rendered code on the right.

The 5 images in `examples/` were not seen in training. You can also pass your own screenshot
(a phone screen in portrait mode). Each image takes 1–3 minutes.

## Retrain (optional)

```bash
scripts/download_mobileviews.sh                 # ~16 GB
.venv-train/bin/python -I scripts/clean.py
scripts/run_all.sh                              # 80K pairs, several hours
.venv-train/bin/python -I scripts/make_sft_split.py
scripts/train_loop.sh                           # ~4 h on an RTX 4060, resumes after a crash
scripts/eval.sh gen   --name sft5k --adapter runs/sft_5k/final --batch 8
scripts/eval.sh score --name sft5k
```

## License

Code: MIT. Base model: [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (Apache 2.0).
Data: MobileViews (MIT); the screenshots belong to the third-party apps they come from.
