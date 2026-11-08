"""Imtihon natijalarini solishtirish: jadval (reports/eval_compare.json) va namunaviy rasmlar.

reports/compare/NNN_<id>.png: asl ekran | bizning konverter | o'qitilgan model | asl Qwen (har biri ostida vis_sim)
Tanlov tasodifiy (seed 0), "yaxshilarini tanlash" yo'q.
"""
import io, json, os, random

import pyarrow.parquet as pq
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES = [("reference", "Konverter"), ("sft5k", "O'qitilgan model"), ("qwen_base_sampled", "Asl Qwen")]
ALL = NAMES + [("qwen_base", "Asl Qwen (greedy)")]

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
    tiles = [("Asl ekran", Image.open(io.BytesIO(test[sid])), None)]
    for n, label in NAMES:
        r = sc[n][sid]
        im = Image.open(io.BytesIO(r["render_png"])) if r["render_png"] else None
        tiles.append((label, im, r["vis_sim"] if im else r["error"] or "xato"))
    for j, (label, im, s) in enumerate(tiles):
        x = j * (W + 10)
        if im:
            canvas.paste(im.convert("RGB").resize((W, H)), (x, 20))
        else:
            d.rectangle([x, 20, x + W, 20 + H], outline="red")
            d.text((x + 10, 240), "ishlamadi", fill="red")
        d.text((x + 4, 4), label, fill="black")
        if s is not None:
            d.text((x + 4, H + 24), f"o'xshashlik: {s}" if isinstance(s, float) else str(s)[:40], fill="black")
    canvas.save(f"{out}/{k:03d}_{sid}.png")
print(json.dumps({n: {k: v for k, v in r.items() if k != "nom"} for n, r in reps.items()}, indent=1, ensure_ascii=False))
