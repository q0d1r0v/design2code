"""Final report for step 3: checks every batch and writes reports/pairs_report.json.

Checks:
  * every batch file can be read (not corrupt) and has a skips.json;
  * coverage: every clean screen (81,048) is either a pair or counted as dropped;
  * no duplicate ids;
  * rendered PNGs open (sample) and the code has a Screen() function.
"""
import collections, glob, io, json, os, random, statistics

import pyarrow.parquet as pq
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN = f"{ROOT}/data/clean"
PAIRS = f"{ROOT}/data/pairs"
# skip reasons written by older versions of the scripts -> current names
LEGACY = {"kirdi": "in", "chiqdi": "out", "render_xatolari": "render_errors", "funksiya_juda_katta": "function_too_large",
          "juda_kam_element": "too_few_nodes", "juda_kop_element": "too_many_nodes", "bosh_ekran": "empty_screen",
          "bosh_daraxt": "empty_tree", "rasm_olchami": "bad_image_size", "kompilyatsiya_xato": "compile_error",
          "render_xato": "render_error", "png_yoq": "png_missing", "konverter_xato": "converter_error"}

clean_total = sum(pq.ParquetFile(f"{CLEAN}/{f}").metadata.num_rows for f in os.listdir(CLEAN))
files = sorted(glob.glob(f"{PAIRS}/batch_*.parquet"))
ids, vis, nodes, sync = [], [], [], []
skips = collections.Counter()
n_in = 0
missing_skips, bad_files = [], []
for f in files:
    try:
        t = pq.read_table(f, columns=["id", "vis_sim", "nodes", "sync"])
    except Exception as e:
        bad_files.append(f); continue
    ids += t.column("id").to_pylist(); vis += t.column("vis_sim").to_pylist()
    nodes += t.column("nodes").to_pylist(); sync += t.column("sync").to_pylist()
    sj = f.replace(".parquet", ".skips.json")
    if not os.path.exists(sj):
        missing_skips.append(f); continue
    d = {LEGACY.get(k.split(":")[0], k.split(":")[0]) + (":" + k.split(":", 1)[1] if ":" in k else ""): v
         for k, v in json.load(open(sj)).items()}
    n_in += d.pop("in"); d.pop("out"); d.pop("render_errors", None)
    skips.update({k: v for k, v in d.items() if v})

# sample: check PNG and code of 300 random pairs
random.seed(0)
png_bad = code_bad = 0
for f in random.sample(files, min(30, len(files))):
    t = pq.read_table(f, columns=["code", "render_png"])
    for k in random.sample(range(t.num_rows), min(10, t.num_rows)):
        try:
            im = Image.open(io.BytesIO(t.column("render_png")[k].as_py())); im.verify()
        except Exception:
            png_bad += 1
        c = t.column("code")[k].as_py()
        if "fun Screen()" not in c or c.count("{") != c.count("}"):
            code_bad += 1

dup = len(ids) - len(set(ids))
report = {
    "clean_screens": clean_total,
    "batches": len(files),
    "screens_in": n_in,
    "pairs": len(ids),
    "duplicate_ids": dup,
    "dropped": dict(skips),
    "dropped_total": sum(skips.values()),
    "coverage_ok": n_in == clean_total and n_in == len(ids) + sum(skips.values()),
    "corrupt_files": bad_files,
    "missing_skips_json": missing_skips,
    "sample_png_errors": png_bad,
    "sample_code_errors": code_bad,
    "vis_sim_median": round(statistics.median(vis), 3) if vis else None,
    "vis_sim_at_least_0.5_%": round(sum(v >= 0.5 for v in vis) / len(vis) * 100, 1) if vis else None,
    "vis_sim_at_least_0.7_%": round(sum(v >= 0.7 for v in vis) / len(vis) * 100, 1) if vis else None,
    "nodes_median": statistics.median(nodes) if nodes else None,
    "disk_GB": round(sum(os.path.getsize(f) for f in files) / 1e9, 2),
}
json.dump(report, open(f"{ROOT}/reports/pairs_report.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(report, indent=1, ensure_ascii=False))
