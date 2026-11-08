"""Butun parquet bo'ylab tasodifiy namuna: ilovalar xilma-xilligi, WebView, Compose ulushi, pastki qora qism."""
import collections, json, os, random, sys
import pyarrow.parquet as pq
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQ = os.path.join(ROOT, "data/raw/mobileviews/MobileViews_Screenshots_ViewHierarchies/Parquets/MobileViews_400000-522301.parquet")
OUT = os.path.join(ROOT, "data/sample_random"); os.makedirs(OUT, exist_ok=True)
pf = pq.ParquetFile(PARQ)
G = pf.num_row_groups; print("row group'lar:", G, "| qatorlar:", pf.metadata.num_rows)
random.seed(0)
groups = sorted(random.sample(range(G), min(40, G)))
apps = collections.Counter(); n = web = comp = flutter_like = 0; content_bottom = []
saved = 0
for i, g in enumerate(groups):
    t = pf.read_row_group(g, columns=["image_content", "json_content"])
    imgs, jss = t.column(0).to_pylist(), t.column(1).to_pylist()
    for k in random.sample(range(len(jss)), min(50, len(jss))):
        d = json.loads(jss[k]); views = [v for v in d.get("views", []) if v.get("visible", True)]
        if not views: continue
        n += 1
        apps[(d.get("foreground_activity") or "?").split("/")[0]] += 1
        cls = [v.get("class") or "" for v in views]
        web += any("WebView" in c for c in cls)
        comp += any("Compose" in c for c in cls)
        generic = sum(c == "android.view.View" for c in cls)
        flutter_like += generic / len(cls) > 0.5
        content_bottom.append(max(v["bounds"][1][1] for v in views))
        if saved < 12 and random.random() < 0.15:
            open(f"{OUT}/{saved}.jpg", "wb").write(imgs[k]); saved += 1
    print(f"\r{i+1}/{len(groups)} ({(i+1)/len(groups)*100:.0f}%)", end="", file=sys.stderr)
print(file=sys.stderr)
cb = sorted(content_bottom)
print(f"Namuna: {n} ekran, {len(apps)} xil ilova")
print(f"WebView bor: {web/n*100:.1f}% | Compose: {comp/n*100:.1f}% | elementlarining >50% i nomsiz 'View' (Compose/Flutter/RN bo'lishi mumkin): {flutter_like/n*100:.1f}%")
print(f"Kontent pastki chegarasi (px): mediana {cb[len(cb)//2]}, 90-foiz {cb[int(len(cb)*.9)]}  (rasm balandligi 2340)")
print("Eng ko'p ekranli ilovalar:", apps.most_common(5))
