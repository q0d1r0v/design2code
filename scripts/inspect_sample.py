"""1-qadam: MobileViews'dan namuna olib ko'rish.
Birinchi N ta qatorni o'qiydi, 8 ta namunani (jpg + json) saqlaydi va statistika chiqaradi.
Ishga tushirish: ~/uz-model/.venv/bin/python -I scripts/inspect_sample.py
"""
import collections, io, json, os, sys
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQ = f"{ROOT}/data/raw/mobileviews/MobileViews_Screenshots_ViewHierarchies/Parquets/MobileViews_400000-522301.parquet"
OUT = f"{ROOT}/data/sample"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
os.makedirs(OUT, exist_ok=True)


def walk(node, depth=0):
    """View hierarchy tugunlarini aylanib chiqadi."""
    if not isinstance(node, dict):
        return
    yield node, depth
    for ch in node.get("children") or []:
        yield from walk(ch, depth + 1)


def short(cls):
    return (cls or "?").split(".")[-1]


pf = pq.ParquetFile(PARQ)
cls_count = collections.Counter()
nodes_per_screen, depths, texts_per_screen, sizes = [], [], [], []
compose_screens = webview_screens = empty_screens = 0
keys_seen = collections.Counter()
done = 0
for batch in pf.iter_batches(batch_size=200, columns=["image_content", "json_content"]):
    for img, js in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
        vh = json.loads(js) if isinstance(js, str) else js
        root = vh if isinstance(vh, dict) else {}
        views = root.get("views") or []
        # DroidBot formati: tekis ro'yxat, har bir view'da parent indeksi bor
        depth = {}
        for v in views:
            p = v.get("parent", -1)
            depth[v["temp_id"]] = 0 if p == -1 else depth.get(p, 0) + 1
        nodes = [(v, depth.get(v.get("temp_id"), 0)) for v in views if v.get("visible", True)]
        if done == 0:
            keys_seen.update(views[0].keys() if views else [])
        classes = [short(n.get("class")) for n, _ in nodes]
        cls_count.update(classes)
        nt = sum(1 for n, _ in nodes if (n.get("text") or "").strip())
        nodes_per_screen.append(len(nodes))
        texts_per_screen.append(nt)
        depths.append(max((d for _, d in nodes), default=0))
        if any("Compose" in c for c in classes):
            compose_screens += 1
        if any("WebView" in c for c in classes):
            webview_screens += 1
        if len(nodes) <= 3:
            empty_screens += 1
        if done < 8:
            open(f"{OUT}/{done}.jpg", "wb").write(img)
            json.dump(root, open(f"{OUT}/{done}.json", "w"), ensure_ascii=False, indent=1)
        if done == 0:
            import subprocess
            sizes.append(subprocess.run(["file", f"{OUT}/0.jpg"], capture_output=True, text=True).stdout.split(",")[-2:])
        done += 1
        if done >= N:
            break
    print(f"\r{done}/{N} ({done / N * 100:.0f}%)", end="", file=sys.stderr)
    if done >= N:
        break
print(file=sys.stderr)


def stat(xs):
    xs = sorted(xs)
    return f"o'rtacha {sum(xs) / len(xs):.0f}, mediana {xs[len(xs) // 2]}, max {xs[-1]}"


print(f"Ko'rilgan ekranlar: {done}")
print(f"View kalitlari: {sorted(keys_seen)}")
print(f"Birinchi rasm o'lchami: {sizes[:1]}")
print(f"Ekrandagi elementlar soni: {stat(nodes_per_screen)}")
print(f"Matnli elementlar: {stat(texts_per_screen)}")
print(f"Daraxt chuqurligi: {stat(depths)}")
print(f"Deyarli bo'sh ekranlar (<=3 element): {empty_screens} ({empty_screens / done * 100:.1f}%)")
print(f"Compose ishlatgan ekranlar: {compose_screens} ({compose_screens / done * 100:.1f}%)")
print(f"WebView bor ekranlar: {webview_screens} ({webview_screens / done * 100:.1f}%)")
print("Eng ko'p uchraydigan 20 element turi:")
total = sum(cls_count.values())
for c, k in cls_count.most_common(20):
    print(f"  {c:35s} {k:7d}  {k / total * 100:5.1f}%")
