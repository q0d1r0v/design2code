"""3-qadam yakuniy hisoboti: barcha guruhlarni tekshiradi va reports/pairs_report.json yozadi.

Tekshiruvlar:
  * har bir guruh fayli o'qiladi (buzilmaganmi), skips.json bor-yo'qligi;
  * barcha toza ekranlar (81 048) qamrab olinganmi: har bir id yo juftlikda, yo tashlab yuborilganlar hisobida;
  * takroriy id yo'qligi;
  * render PNG'lar ochiladimi (namuna), kodda Screen() funksiyasi bormi.
"""
import collections, glob, io, json, os, random, statistics

import pyarrow.parquet as pq
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN = f"{ROOT}/data/clean"
PAIRS = f"{ROOT}/data/pairs"

clean_total = sum(pq.ParquetFile(f"{CLEAN}/{f}").metadata.num_rows for f in os.listdir(CLEAN))
files = sorted(glob.glob(f"{PAIRS}/batch_*.parquet"))
ids, vis, nodes, sync = [], [], [], []
skips = collections.Counter()
kirdi = 0
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
    d = json.load(open(sj))
    kirdi += d.pop("kirdi"); d.pop("chiqdi"); d.pop("render_xatolari", None)
    skips.update({k: v for k, v in d.items() if v})

# namuna: 300 ta tasodifiy juftlikning PNG va kodini tekshirish
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
    "toza_ekranlar": clean_total,
    "guruhlar": len(files),
    "kirgan_ekranlar": kirdi,
    "juftliklar": len(ids),
    "takroriy_id": dup,
    "tashlab_yuborilgan": dict(skips),
    "tashlab_yuborilgan_jami": sum(skips.values()),
    "qamrov_tekshiruvi": kirdi == clean_total and kirdi == len(ids) + sum(skips.values()),
    "buzilgan_fayllar": bad_files,
    "skips_json_yoq": missing_skips,
    "namuna_png_xato": png_bad,
    "namuna_kod_xato": code_bad,
    "vis_sim_mediana": round(statistics.median(vis), 3) if vis else None,
    "vis_sim_0.5_dan_yuqori_%": round(sum(v >= 0.5 for v in vis) / len(vis) * 100, 1) if vis else None,
    "vis_sim_0.7_dan_yuqori_%": round(sum(v >= 0.7 for v in vis) / len(vis) * 100, 1) if vis else None,
    "element_mediana": statistics.median(nodes) if nodes else None,
    "disk_GB": round(sum(os.path.getsize(f) for f in files) / 1e9, 2),
}
json.dump(report, open(f"{ROOT}/reports/pairs_report.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(report, indent=1, ensure_ascii=False))
