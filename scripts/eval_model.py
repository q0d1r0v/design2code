"""Imtihon: model 200 ta yangi ekran rasmidan kod yozadi -> kompilyatsiya -> Paparazzi render -> asl rasm bilan solishtirish.

Ikki bosqich:
  gen   (GPU)   : model kod yozadi -> data/eval/<nom>/gen.parquet   (to'xtasa davom etadi)
  score (Gradle): kod kompilyatsiya + render + baho -> reports/eval_<nom>.json, data/eval/<nom>/scored.parquet

Misollar:
  scripts/eval.sh gen   --name sft5k --adapter runs/sft_5k/final
  scripts/eval.sh gen   --name qwen_base                 # adaptersiz asl Qwen3.5-4B
  scripts/eval.sh score --name sft5k
  scripts/eval.sh score --name reference                 # bizning konverter kodi (taqqoslash uchun yuqori chegara)

Baholar (har bir ekran):
  compiled  - kod kompilyatsiyadan o'tdimi
  rendered  - ekran chizildimi (ishlash vaqtida xato bermadimi)
  vis_sim   - render va asl rasm tuzilishi o'xshashligi (0..1, run_batch.vis_sim bilan bir xil)
  text_recall - asl ekrandagi matnlarning qanchasi kodda bor (0..1)
Kompilyatsiya/render bo'lmagan ekran vis_sim = 0 oladi (o'rtacha hamma 200 ta ekran bo'yicha).
"""
import argparse, io, json, os, re, shutil, statistics, sys, time

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST = f"{ROOT}/data/sft/test_200.parquet"


# ------------------------------------------------------------------ kodni ajratib olish
def extract_code(text):
    """Model javobidan Kotlin kodini ajratadi (``` bloklari, <think> va h.k.)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    blocks = re.findall(r"```(?:kotlin|kt)?\s*\n(.*?)```", text, flags=re.S)
    if blocks:
        text = max(blocks, key=len)
    elif "```" in text:  # yopilmagan blok (javob kesilgan)
        text = text.split("```", 1)[1].split("\n", 1)[-1]
    return text.strip()


def texts_in(code):
    """Koddagi satr literallari (matnlar) to'plami."""
    return {m.encode().decode("unicode_escape", "ignore") if "\\u" in m else m
            for m in re.findall(r'"((?:[^"\\\n]|\\.){2,})"', code)}


# ------------------------------------------------------------------ gen bosqichi
def gen(a):
    import torch
    from train_sft import load_model, load_image, prompt_text
    from transformers import AutoProcessor
    from train_sft import MODEL
    out_dir = f"{ROOT}/data/eval/{a.name}"; os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/gen.parquet"
    done = {}
    if os.path.exists(path):
        t = pq.read_table(path).to_pylist(); done = {r["id"]: r for r in t}
    t = pq.read_table(TEST, columns=["id", "image"])
    ids, imgs = t.column("id").to_pylist()[:a.limit], t.column("image").to_pylist()[:a.limit]
    processor = AutoProcessor.from_pretrained(MODEL)
    processor.tokenizer.padding_side = "left"
    model = load_model(f"{ROOT}/{a.adapter}" if a.adapter else None, train=False)
    model.eval()
    ptext = prompt_text(processor)
    todo = [k for k in range(len(ids)) if ids[k] not in done]
    t0 = time.time(); n0 = len(done)
    for s in range(0, len(todo), a.batch):
        ks = todo[s:s + a.batch]
        enc = processor(text=[ptext] * len(ks), images=[load_image(imgs[k]) for k in ks],
                        return_tensors="pt", padding=True).to("cuda")
        # --sample: Qwen tavsiya qilgan sozlama (non-thinking): temperature 0.7, top_p 0.8, top_k 20
        gk = (dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05) if a.sample
              else dict(do_sample=False))
        torch.manual_seed(s)
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=a.max_new, **gk)
        for j, k in enumerate(ks):
            seq = o[j, enc.input_ids.shape[1]:]
            raw = processor.tokenizer.decode(seq, skip_special_tokens=True)
            done[ids[k]] = {"id": ids[k], "raw": raw, "new_tokens": int((seq != processor.tokenizer.pad_token_id).sum())}
        pq.write_table(pa.Table.from_pylist(list(done.values())), path)
        el = time.time() - t0; n = len(done) - n0
        print(f"{time.strftime('%H:%M:%S')} [{a.name}] {len(done)}/{len(ids)} ({len(done)/len(ids)*100:.0f}%)  "
              f"{el/n:.1f} s/ekran  qoldi ~{el/n*(len(ids)-len(done))/60:.0f} daq", flush=True)


# ------------------------------------------------------------------ score bosqichi
# IDE'dagi "auto-import" o'rnini bosadi: model unutgan standart importlar (barcha modellarga bir xil qo'shiladi).
AUTO_IMPORTS = [f"import {p}" for p in (
    "androidx.compose.runtime.*", "androidx.compose.foundation.lazy.*", "androidx.compose.foundation.lazy.grid.*",
    "androidx.compose.foundation.text.*", "androidx.compose.ui.draw.*", "androidx.compose.ui.graphics.*",
    "androidx.compose.ui.text.*", "androidx.compose.ui.text.font.*", "androidx.compose.ui.text.input.*",
    "androidx.compose.ui.layout.*", "androidx.compose.ui.unit.*",
    "androidx.compose.material.icons.Icons", "androidx.compose.material.icons.filled.*")]


def kotlin_file(sid, code):
    """Har bir ekran o'z paketida (nomlar to'qnashmasligi uchun). Standart importlar + modelning importlari."""
    from vh2compose import IMPORTS
    # @Preview zararsiz, lekin ui-tooling kutubxonasi loyihada yo'q -> olib tashlanadi (barcha modellarga bir xil)
    code = re.sub(r"@Preview(\([^)]*\))?", "", code)
    lines = code.split("\n")
    imports = [l.strip() for l in lines if l.strip().startswith("import ") and "tooling.preview" not in l]
    body = [l for l in lines if not l.strip().startswith(("import ", "package "))]
    seen, imp = set(), []
    for l in IMPORTS.strip().split("\n") + AUTO_IMPORTS + imports:
        if l not in seen:
            seen.add(l); imp.append(l)
    return f"package uz.d2c.screens.gen.s{sid}\n\n" + "\n".join(imp) + "\n\n" + "\n".join(body) + "\n"


def score(a):
    from run_batch import gradle, vis_sim
    work = f"{ROOT}/{a.workdir}"
    out_dir = f"{ROOT}/data/eval/{a.name}"; os.makedirs(out_dir, exist_ok=True)
    test = {r["id"]: r for r in pq.read_table(TEST).to_pylist()[:a.limit]}
    if a.name == "reference":
        codes = {i: r["code"] for i, r in test.items()}
        raw_len = {i: None for i in test}
    else:
        g = {r["id"]: r for r in pq.read_table(f"{out_dir}/gen.parquet").to_pylist()}
        codes = {i: extract_code(g[i]["raw"]) for i in test if i in g}
        raw_len = {i: g[i]["new_tokens"] for i in codes}
    has_fn = {i: bool(re.search(r"fun\s+Screen\s*\(", c)) for i, c in codes.items()}
    alive = [i for i in codes if has_fn[i]]
    compile_fail = set()
    gen = f"{work}/screens/src/main/kotlin/uz/d2c/screens/gen"
    rtmp = f"{ROOT}/data/render_tmp/eval_{a.name}"
    for attempt in range(12):
        shutil.rmtree(gen, ignore_errors=True); os.makedirs(gen)
        for i in alive:
            open(f"{gen}/s{i}.kt", "w").write(kotlin_file(i, codes[i]))
        open(f"{gen}/All.kt", "w").write(
            "package uz.d2c.screens.gen\n\nimport androidx.compose.runtime.Composable\n"
            + "".join(f"import uz.d2c.screens.gen.s{i}.Screen as S{i}\n" for i in alive)
            + "\nval ALL_SCREENS: List<Pair<Long, @Composable () -> Unit>> = listOf(\n"
            + ",\n".join(f"    {i}L to {{ S{i}() }}" for i in alive) + "\n)\n")
        shutil.rmtree(rtmp, ignore_errors=True)
        rc, log = gradle(work, rtmp)
        bad = {int(m) for m in re.findall(r"e: file://\S*/gen/s(\d+)\.kt:", log)}
        bad |= {int(m) for m in re.findall(r"Method too large: \S*?/s(\d+)/", log)}
        # All.kt dagi xato (masalan, Screen private yoki parametr talab qiladi) -> o'sha qatordagi ekran
        all_lines = open(f"{gen}/All.kt").read().split("\n")
        for ln in re.findall(r"e: file://\S*/gen/All\.kt:(\d+):", log):
            m = re.search(r"\bS(\d+)\b", all_lines[int(ln) - 1])
            if m:
                bad.add(int(m.group(1)))
        if not bad:
            break
        with open(f"{out_dir}/compile_errors.log", "w" if attempt == 0 else "a") as fh:  # tahlil uchun
            fh.write("\n".join(l for l in log.split("\n") if l.startswith("e: ")) + "\n")
        compile_fail |= bad
        alive = [i for i in alive if i not in bad]
        print(f"  urinish {attempt + 1}: {len(bad)} ta ekran kompilyatsiyadan o'tmadi, qoldi {len(alive)}", flush=True)
    else:
        raise RuntimeError("12 urinishda ham kompilyatsiya tugamadi")
    if rc != 0 and not os.path.isdir(rtmp):
        raise RuntimeError("Gradle xatosi:\n" + log[-3000:])
    failed = {}
    if os.path.exists(f"{rtmp}/failures.txt"):
        for l in open(f"{rtmp}/failures.txt"):
            sid, msg = l.rstrip("\n").split("\t", 1); failed[int(sid)] = msg
    ref_texts = {i: texts_in(test[i]["code"]) for i in test}
    rows = []
    for i in test:
        c = codes.get(i, "")
        png = f"{rtmp}/{i}.png"
        rendered = i in alive and i not in failed and os.path.exists(png)
        rb = open(png, "rb").read() if rendered else None
        rt = ref_texts[i]
        rows.append({"id": i, "code": c, "has_screen_fn": has_fn.get(i, False),
                     "compiled": i in alive, "rendered": rendered,
                     "vis_sim": round(vis_sim(test[i]["image"], rb), 3) if rendered else 0.0,
                     "text_recall": round(sum(x in c for x in rt) / len(rt), 3) if rt else 1.0,
                     "new_tokens": raw_len.get(i), "render_png": rb,
                     "error": failed.get(i) or ("kompilyatsiya" if i in compile_fail else
                                                ("Screen() yo'q" if not has_fn.get(i, False) else None))})
    pq.write_table(pa.Table.from_pylist(rows), f"{out_dir}/scored.parquet")
    shutil.rmtree(rtmp, ignore_errors=True)
    n = len(rows)
    vs = [r["vis_sim"] for r in rows]
    rep = {"nom": a.name, "ekranlar": n,
           "kompilyatsiya_%": round(sum(r["compiled"] for r in rows) / n * 100, 1),
           "render_%": round(sum(r["rendered"] for r in rows) / n * 100, 1),
           "vis_sim_ortacha": round(statistics.mean(vs), 3),
           "vis_sim_mediana": round(statistics.median(vs), 3),
           "vis_sim_0.7_dan_yuqori_%": round(sum(v >= 0.7 for v in vs) / n * 100, 1),
           "matn_topildi_ortacha": round(statistics.mean(r["text_recall"] for r in rows), 3),
           "screen_fn_yoq": sum(not r["has_screen_fn"] for r in rows)}
    json.dump(rep, open(f"{ROOT}/reports/eval_{a.name}.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gen", "score"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=4096)
    ap.add_argument("--workdir", default="render")
    ap.add_argument("--sample", action="store_true", help="greedy o'rniga Qwen tavsiya qilgan sampling")
    a = ap.parse_args()
    (gen if a.stage == "gen" else score)(a)
