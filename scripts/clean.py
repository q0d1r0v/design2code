"""2-qadam: MobileViews ekranlarini tozalash.

Olib tashlanadi: bo'sh, gorizontal, tizim oynalari, WebView, asosan nomsiz 'View'dan iborat
(Flutter/RN/Compose), juda kam yoki juda ko'p elementli, matnsiz ekranlar.
Rasm 1080x1704 ga kesiladi (status bar va pastki navigatsiya/qora qism olib tashlanadi),
element koordinatalari ham shunga moslab suriladi.

Natija: data/clean/shard_XXX.parquet (id, package, layout_key, image, views) + reports/clean_report.json
Ishga tushirish: .venv/bin/python -I scripts/clean.py
"""
import collections, io, json, os, sys, time
from multiprocessing import Pool

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQ = f"{ROOT}/data/raw/mobileviews/MobileViews_Screenshots_ViewHierarchies/Parquets/MobileViews_400000-522301.parquet"
OUT = f"{ROOT}/data/clean"
TOP, BOTTOM, W = 72, 1776, 1080  # ilova ko'rinadigan maydon (status bar va nav bar'siz)

SYSTEM_PKGS = ("android", "com.android.", "com.google.android.", "com.sec.android.")
KEEP_FIELDS = ("class", "text", "content_description", "resource_id", "clickable", "editable",
               "checkable", "checked", "scrollable", "is_password", "selected", "enabled")
MIN_VIEWS, MAX_VIEWS = 5, 300
MAX_GENERIC = 0.5


def is_system(pkg):
    return pkg == "android" or any(pkg.startswith(p) for p in SYSTEM_PKGS if p != "android")


def process(group):
    pf = pq.ParquetFile(PARQ)
    t = pf.read_row_group(group, columns=["image_content", "json_content"])
    imgs, jss = t.column(0).to_pylist(), t.column(1).to_pylist()
    base = sum(pf.metadata.row_group(g).num_rows for g in range(group))
    reasons = collections.Counter()
    rows = {"id": [], "package": [], "layout_key": [], "image": [], "views": []}
    for k, (img, js) in enumerate(zip(imgs, jss)):
        try:
            d = json.loads(js)
        except Exception:
            reasons["json_xato"] += 1; continue
        views = [v for v in d.get("views") or [] if v.get("visible", True)]
        if not views:
            reasons["bosh"] += 1; continue
        root_b = views[0]["bounds"]
        if root_b[1][0] > root_b[1][1]:
            reasons["gorizontal"] += 1; continue
        pkg = (d.get("foreground_activity") or "?").split("/")[0]
        if is_system(pkg):
            reasons["tizim_oynasi"] += 1; continue
        cls = [v.get("class") or "" for v in views]
        if any("WebView" in c for c in cls):
            reasons["webview"] += 1; continue
        if sum(c == "android.view.View" for c in cls) / len(cls) > MAX_GENERIC:
            reasons["nomsiz_view"] += 1; continue
        if not MIN_VIEWS <= len(views) <= MAX_VIEWS:
            reasons["element_soni"] += 1; continue
        if not any((v.get("text") or "").strip() for v in views):
            reasons["matnsiz"] += 1; continue
        try:
            im = Image.open(io.BytesIO(img)).convert("RGB")
            if im.size[0] != W or im.size[1] < BOTTOM:
                reasons["rasm_olchami"] += 1; continue
            buf = io.BytesIO(); im.crop((0, TOP, W, BOTTOM)).save(buf, "JPEG", quality=90)
        except Exception:
            reasons["rasm_xato"] += 1; continue
        # koordinatalarni kesilgan rasmga moslash, kerakli maydonlarni qoldirish
        idmap = {v["temp_id"]: i for i, v in enumerate(views)}
        slim = []
        for v in views:
            (x1, y1), (x2, y2) = v["bounds"]
            y1, y2 = max(y1 - TOP, 0), min(y2 - TOP, BOTTOM - TOP)
            s = {f: v.get(f) for f in KEEP_FIELDS if v.get(f) not in (None, False, "")}
            s["bounds"] = [x1, y1, x2, y2]
            s["parent"] = idmap.get(v.get("parent"), -1)
            slim.append(s)
        rows["id"].append(400000 + base + k)
        rows["package"].append(pkg)
        rows["layout_key"].append(f"{pkg}|{d.get('state_str_content_free', '')}")
        rows["image"].append(buf.getvalue())
        rows["views"].append(json.dumps(slim, ensure_ascii=False))
        reasons["qoldi"] += 1
    if rows["id"]:
        pq.write_table(pa.table(rows), f"{OUT}/shard_{group:03d}.parquet")
    return group, len(imgs), reasons


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); os.makedirs(f"{ROOT}/reports", exist_ok=True)
    G = pq.ParquetFile(PARQ).num_row_groups
    total = collections.Counter(); seen = 0; t0 = time.time()
    with Pool(12) as pool:
        for i, (g, n, r) in enumerate(pool.imap_unordered(process, range(G)), 1):
            total.update(r); seen += n
            print(f"\r{i}/{G} guruh ({i / G * 100:.0f}%)  ko'rildi {seen}  qoldi {total['qoldi']}  {time.time() - t0:.0f}s",
                  end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    keys = set()
    for f in sorted(os.listdir(OUT)):
        keys.update(pq.read_table(f"{OUT}/{f}", columns=["layout_key"]).column(0).to_pylist())
    pkgs = set()
    for f in sorted(os.listdir(OUT)):
        pkgs.update(pq.read_table(f"{OUT}/{f}", columns=["package"]).column(0).to_pylist())
    report = {"korildi": seen, **dict(total), "ilovalar": len(pkgs), "noyob_layout": len(keys)}
    json.dump(report, open(f"{ROOT}/reports/clean_report.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(report, indent=1, ensure_ascii=False))
