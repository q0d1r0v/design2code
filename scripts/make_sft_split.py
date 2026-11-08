"""Prepares step 6: the training (5,000) and test (200) sets.

Test screens come from completely different apps (those apps never appear in training),
so the test checks real generalisation, not memorisation.
Training: vis_sim >= 0.7 (code matches the original), code <= 3000 tokens, at most 4 per app,
no repeated layouts.
Output: data/sft/train_5k.parquet (id, package, image, code), data/sft/test_200.parquet (id, package, image, code)
"""
import collections, glob, json, os, random
import pyarrow as pa, pyarrow.parquet as pq
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
random.seed(42)
tok = AutoTokenizer.from_pretrained(f"{ROOT}/models/Qwen3.5-4B")

# layout_key and shard location from the clean data
lk, shard_of = {}, {}
for f in sorted(glob.glob(f"{ROOT}/data/clean/shard_*.parquet")):
    t = pq.read_table(f, columns=["id", "layout_key"])
    for i, k in zip(t.column("id").to_pylist(), t.column("layout_key").to_pylist()):
        lk[i] = k; shard_of[i] = f

rows = []
for f in sorted(glob.glob(f"{ROOT}/data/pairs/batch_*.parquet")):
    t = pq.read_table(f, columns=["id", "package", "vis_sim", "code"])
    for i, p, v, c in zip(*(t.column(x).to_pylist() for x in ("id", "package", "vis_sim", "code"))):
        rows.append((i, p, v, c))
pkgs = sorted({r[1] for r in rows})
random.shuffle(pkgs)
test_pkgs = set(pkgs[:330])           # 330 apps reserved for testing (200 screens are picked from them)

test, train_pool = [], []
for r in rows:
    (test if r[1] in test_pkgs else train_pool).append(r)
# test: 1 per app, vis_sim >= 0.6 (screens whose screenshot matches the hierarchy)
by_p = collections.defaultdict(list)
for r in test:
    if r[2] >= 0.6: by_p[r[1]].append(r)
test_sel = [random.choice(v) for v in by_p.values()]
random.shuffle(test_sel); test_sel = test_sel[:200]

cand = [r for r in train_pool if r[2] >= 0.7]
random.shuffle(cand)
train_sel, per_pkg, seen_lk = [], collections.Counter(), set()
for r in cand:
    if per_pkg[r[1]] >= 4 or lk[r[0]] in seen_lk: continue
    if len(tok(r[3]).input_ids) > 3000: continue
    train_sel.append(r); per_pkg[r[1]] += 1; seen_lk.add(lk[r[0]])
    if len(train_sel) == 5000: break

def images(ids):
    need = collections.defaultdict(set)
    for i in ids: need[shard_of[i]].add(i)
    out = {}
    for f, s in need.items():
        t = pq.read_table(f, columns=["id", "image"])
        for i, im in zip(t.column("id").to_pylist(), t.column("image").to_pylist()):
            if i in s: out[i] = im
    return out

for name, sel in (("train_5k", train_sel), ("test_200", test_sel)):
    im = images([r[0] for r in sel])
    pq.write_table(pa.table({"id": [r[0] for r in sel], "package": [r[1] for r in sel],
                             "image": [im[r[0]] for r in sel], "code": [r[3] for r in sel]}),
                   f"{ROOT}/data/sft/{name}.parquet")
rep = {"train": len(train_sel), "test": len(test_sel), "test_apps": len({r[1] for r in test_sel}),
       "train_apps": len(per_pkg), "overlapping_apps": len({r[1] for r in test_sel} & set(per_pkg))}
json.dump(rep, open(f"{ROOT}/reports/sft_split.json", "w"), indent=1)
print(rep)
