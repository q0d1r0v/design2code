"""3-qadam: ekranlarni Compose kodiga aylantirish, kompilyatsiya qilish va Paparazzi bilan rasmga olish.

Har bir guruh (batch): N ta ekran -> Kotlin fayllar -> gradle test -> PNG'lar.
Kompilyatsiya xatosi bo'lsa, xato bergan ekranlar olib tashlanib, qayta urinadi (3 martagacha).
Natija: data/pairs/batch_XXXX.parquet  (id, package, code, render_png, nodes)
         data/pairs/batch_XXXX.skips.json (tashlab yuborilganlar sababi bilan)

Ishga tushirish:
  .venv/bin/python -I scripts/run_batch.py --start 0 --count 100 --batch-size 100 --workdir render
"""
import argparse, collections, json, os, re, shutil, subprocess, sys, time

import io

import numpy as np
import pyarrow as pa
from PIL import Image, ImageFilter
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vh2compose import IMPORTS, Skip, convert  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN = f"{ROOT}/data/clean"
PAIRS = f"{ROOT}/data/pairs"
PER_FILE = 25


def vis_sim(orig_bytes, render_bytes):
    """Asl rasm va render o'xshashligi (0..1): kichraytirilgan kulrang tasvirlar tuzilishining korrelyatsiyasi."""
    def prep(b):
        im = Image.open(io.BytesIO(b)).convert("L").resize((64, 101), Image.BILINEAR).filter(ImageFilter.GaussianBlur(1))
        a = np.asarray(im, float)
        e = np.abs(np.diff(a, axis=0))[:, :-1] + np.abs(np.diff(a, axis=1))[:-1, :]  # chegaralar xaritasi
        return e.ravel()
    x, y = prep(orig_bytes), prep(render_bytes)
    if x.std() < 1e-6 or y.std() < 1e-6:
        return 0.0
    return float(max(0.0, np.corrcoef(x, y)[0, 1]))


def iter_clean(start, count):
    """Toza shard'lardan ketma-ket (id, package, image, views) qaytaradi."""
    i = 0
    for f in sorted(os.listdir(CLEAN)):
        n = pq.ParquetFile(f"{CLEAN}/{f}").metadata.num_rows  # o'qimasdan qator sonini bilish
        if i + n <= start:
            i += n
            continue
        t = pq.read_table(f"{CLEAN}/{f}")
        ids, pk, im, vs = (t.column(c).to_pylist() for c in ("id", "package", "image", "views"))
        for k in range(n):
            if i >= start + count:
                return
            if i >= start:
                yield ids[k], pk[k], im[k], vs[k]
            i += 1


def write_sources(workdir, items):
    gen = f"{workdir}/screens/src/main/kotlin/uz/d2c/screens/gen"
    shutil.rmtree(gen, ignore_errors=True)
    os.makedirs(gen)
    regs, fmap = [], {}
    for fi in range(0, len(items), PER_FILE):
        chunk = items[fi:fi + PER_FILE]
        fname = f"G{fi // PER_FILE:04d}"
        body = ["package uz.d2c.screens.gen", "", IMPORTS]
        for sid, code in chunk:
            body.append(code)
            body.append("")
            fmap.setdefault(f"{fname}.kt", []).append(sid)
        body.append(f"val SCREENS_{fname}: List<Pair<Long, @Composable () -> Unit>> = listOf(")
        body.append(",\n".join(f"    {sid}L to {{ S{sid}() }}" for sid, _ in chunk))
        body.append(")")
        open(f"{gen}/{fname}.kt", "w").write("\n".join(body) + "\n")
        regs.append(f"SCREENS_{fname}")
    open(f"{gen}/All.kt", "w").write(
        "package uz.d2c.screens.gen\n\nimport androidx.compose.runtime.Composable\n\n"
        "val ALL_SCREENS: List<Pair<Long, @Composable () -> Unit>> = " + " + ".join(regs or ["emptyList()"]) + "\n")
    return gen, fmap


def bad_screens_from_errors(log, gen, fmap):
    """Kompilyator xatolaridan qaysi ekran funksiyasida xato borligini topadi."""
    bad = set()
    # JVM 64 KB chegarasi: "Method too large: .../G0009Kt.S483480 (...)"
    for m in re.finditer(r"Method too large: \S*?/G\d+Kt\.S(\d+)\b", log):
        bad.add(int(m.group(1)))
    for m in re.finditer(r"e: file://\S*/gen/(G\d+\.kt):(\d+):", log):
        fname, line = m.group(1), int(m.group(2))
        lines = open(f"{gen}/{fname}").read().split("\n")[:line]
        for l in reversed(lines):
            mm = re.match(r"fun S(\d+)\(\)", l)
            if mm:
                bad.add(int(mm.group(1)))
                break
    return bad


def gradle(workdir, out):
    env = dict(os.environ)
    jdk = [d for d in os.listdir(f"{ROOT}/tools") if d.startswith("jdk-17")][0]
    env.update(JAVA_HOME=f"{ROOT}/tools/{jdk}",
               ANDROID_HOME=os.environ.get("ANDROID_HOME") or os.path.expanduser("~/Android/Sdk"),
               GRADLE_USER_HOME=os.environ.get("D2C_GRADLE_HOME") or f"{ROOT}/tools/gradle-home")
    cmd = ["./gradlew", "-q", ":screens:testDebugUnitTest", f"-Pd2cOut={out}", "--console=plain"]
    p = subprocess.run(cmd, cwd=workdir, env=env, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def run(batch_no, rows, workdir):
    os.makedirs(PAIRS, exist_ok=True)
    skips = collections.Counter()
    items, meta = [], {}
    origs = {sid: img for sid, _, img, _ in rows}
    for sid, pkg, img, vs in rows:
        try:
            code, nodes, sync = convert(json.loads(vs), img, f"S{sid}")
        except Skip as e:
            skips[str(e)] += 1
            continue
        except Exception as e:  # konverter xatosi -> sababi bilan hisobga olinadi
            skips["konverter_xato:" + type(e).__name__] += 1
            continue
        items.append((sid, code))
        meta[sid] = (pkg, code, nodes, sync)
    out = f"{ROOT}/data/render_tmp/{os.path.basename(workdir)}_b{batch_no:06d}"
    for attempt in range(8):
        shutil.rmtree(out, ignore_errors=True)
        gen, fmap = write_sources(workdir, items)
        for g_try in range(3):  # vaqtinchalik Gradle xatolari (masalan, xotira) uchun qayta urinish
            rc, log = gradle(workdir, out)
            if rc == 0 or "e: file://" in log or "Method too large" in log:
                break
            open(f"{PAIRS}/batch_{batch_no:06d}.gradle_retry.log", "a").write(log[-5000:] + "\n")
            time.sleep(20)
        if "e: file://" not in log and "Method too large" not in log:
            break
        bad = bad_screens_from_errors(log, gen, fmap)
        if not bad:
            raise RuntimeError("Kompilyatsiya xatosi, lekin ekran topilmadi:\n" + log[-3000:])
        skips["kompilyatsiya_xato"] += len(bad)
        open(f"{PAIRS}/batch_{batch_no:06d}.compile_errors.log", "a").write(log[-20000:] + "\n")
        items = [x for x in items if x[0] not in bad]
    else:
        raise RuntimeError("8 urinishda ham kompilyatsiya bo'lmadi")
    if rc != 0 and not os.path.isdir(out):
        raise RuntimeError("Gradle xatosi:\n" + log[-3000:])
    failed = {}
    if os.path.exists(f"{out}/failures.txt"):
        for l in open(f"{out}/failures.txt"):
            sid, msg = l.rstrip("\n").split("\t", 1)
            failed[int(sid)] = msg
    skips["render_xato"] += len(failed)
    rec = {"id": [], "package": [], "code": [], "render_png": [], "nodes": [], "sync": [], "vis_sim": []}
    for sid, _ in items:
        png = f"{out}/{sid}.png"
        if sid in failed:
            continue
        if not os.path.exists(png):
            skips["png_yoq"] += 1
            continue
        pkg, code, nodes, sync = meta[sid]
        rec["id"].append(sid); rec["package"].append(pkg)
        rec["code"].append(IMPORTS + "\n" + re.sub(rf"\bS{sid}(?=\(|Part)", "Screen", code))
        rec["render_png"].append(open(png, "rb").read()); rec["nodes"].append(nodes); rec["sync"].append(round(sync, 3))
        rec["vis_sim"].append(round(vis_sim(origs[sid], rec["render_png"][-1]), 3))
    pq.write_table(pa.table(rec), f"{PAIRS}/batch_{batch_no:06d}.parquet")
    json.dump({"kirdi": len(rows), "chiqdi": len(rec["id"]), **skips, "render_xatolari": failed},
              open(f"{PAIRS}/batch_{batch_no:06d}.skips.json", "w"), indent=1, ensure_ascii=False)
    shutil.rmtree(out, ignore_errors=True)
    return len(rows), len(rec["id"]), skips


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--workdir", default="render")
    ap.add_argument("--reverse", action="store_true", help="guruhlarni oxiridan boshlab ishlash (yordamchi ishchi uchun)")
    a = ap.parse_args()
    workdir = f"{ROOT}/{a.workdir}"
    t0, tot_in, tot_out, tot_sk = time.time(), 0, 0, collections.Counter()
    total = a.count
    order = list(range(a.start, a.start + a.count, a.batch_size))
    if a.reverse:
        order.reverse()
    for bn in order:  # guruh nomi = birinchi ekranning tartib raqami
        n = min(a.batch_size, a.start + a.count - bn)
        if os.path.exists(f"{PAIRS}/batch_{bn:06d}.parquet"):  # oldin tayyor bo'lgan -> o'tkazib yuborish
            tot_in += n
            continue
        rows = list(iter_clean(bn, n))  # faqat shu guruh xotiraga yuklanadi
        if not rows:
            break
        n_in, n_out, sk = run(bn, rows, workdir)
        tot_in += n_in; tot_out += n_out; tot_sk.update(sk)
        el = time.time() - t0
        print(f"[{a.workdir}] {tot_in}/{total} ({tot_in / total * 100:.1f}%)  juftlik {tot_out}  "
              f"{el:.0f}s  {dict(tot_sk)}", flush=True)
