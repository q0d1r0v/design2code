"""Compares evaluation results: a table (reports/eval_compare.json) and sample images.

reports/compare/NNN_<id>.png: original | rule converter | fine-tuned model | base Qwen (vis_sim under each)
The sample is random (seed 0), not cherry-picked.
"""
import io, json, os, random

import pyarrow.parquet as pq
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES = [("reference", "Rule converter"), ("sft5k", "Fine-tuned model"), ("qwen_base_sampled", "Base Qwen")]
ALL = NAMES + [("qwen_base", "Base Qwen (greedy)")]

reps = {n: json.load(open(f"{ROOT}/reports/eval_{n}.json")) for n, _ in ALL}
json.dump(reps, open(f"{ROOT}/reports/eval_compare.json", "w"), indent=1, ensure_ascii=False)

test = {r["id"]: r["image"] for r in pq.read_table(f"{ROOT}/data/sft/test_200.parquet", columns=["id", "image"]).to_pylist()}
sc = {n: {r["id"]: r for r in pq.read_table(f"{ROOT}/data/eval/{n}/scored.parquet",
                                              columns=["id", "render_png", "vis_sim", "error"]).to_pylist()}
      for n, _ in NAMES}
out = f"{ROOT}/reports/compare"; os.makedirs(out, exist_ok=True)
for f in os.listdir(out):
    os.remove(f"{out}/{f}")
W, H = 317, 500
random.seed(0)
for k, sid in enumerate(random.sample(sorted(test), 12)):
    canvas = Image.new("RGB", (W * 4 + 30, H + 40), "white")
    d = ImageDraw.Draw(canvas)
    tiles = [("Original", Image.open(io.BytesIO(test[sid])), None)]
    for n, label in NAMES:
        r = sc[n][sid]
        im = Image.open(io.BytesIO(r["render_png"])) if r["render_png"] else None
        tiles.append((label, im, r["vis_sim"] if im else r["error"] or "error"))
    for j, (label, im, s) in enumerate(tiles):
        x = j * (W + 10)
        if im:
            canvas.paste(im.convert("RGB").resize((W, H)), (x, 20))
        else:
            d.rectangle([x, 20, x + W, 20 + H], outline="red")
            d.text((x + 10, 240), "failed", fill="red")
        d.text((x + 4, 4), label, fill="black")
        if s is not None:
            d.text((x + 4, H + 24), f"vis_sim: {s}" if isinstance(s, float) else str(s)[:40], fill="black")
    canvas.save(f"{out}/{k:03d}_{sid}.png")
print(json.dumps({n: {k: v for k, v in r.items() if k != "name"} for n, r in reps.items()}, indent=1, ensure_ascii=False))
