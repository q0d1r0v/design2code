"""Try the fine-tuned model on one or more screenshots.

Usage:  ./try.sh screenshot.png [screenshot2.jpg ...]
Results go to output/:
  <name>.kt           - Compose code written by the model
  <name>_compare.png  - your screenshot on the left, the rendered code on the right
"""
import argparse, os, re, shutil, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/output"


def generate(paths, adapter, max_new):
    import torch
    from transformers import AutoProcessor
    from train_sft import MODEL, load_model, load_image, prompt_text
    from eval_model import extract_code
    print("Loading the model (~1 min)...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL)
    model = load_model(adapter, train=False).eval()
    ptext = prompt_text(processor)
    codes = {}
    for k, p in enumerate(paths, 1):
        t0 = time.time()
        print(f"[{k}/{len(paths)}] {os.path.basename(p)}: writing code (~1-3 min)...", flush=True)
        enc = processor(text=[ptext], images=[load_image(open(p, "rb").read())], return_tensors="pt").to("cuda")
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
        seq = o[0, enc.input_ids.shape[1]:]
        codes[p] = extract_code(processor.tokenizer.decode(seq, skip_special_tokens=True))
        cut = len(seq) >= max_new
        print(f"      done: {time.time() - t0:.0f} s" + ("  (WARNING: the code hit the length limit and was cut off)" if cut else ""), flush=True)
    del model
    torch.cuda.empty_cache()
    return codes


def render(items):
    """items: {number: code}. Compile + Paparazzi render. Returns {number: png path or error text}."""
    from eval_model import kotlin_file
    from run_batch import gradle
    work = f"{ROOT}/render"
    gen = f"{work}/screens/src/main/kotlin/uz/d2c/screens/gen"
    tmp = f"{ROOT}/data/render_tmp/try"
    alive, errors = dict(items), {}
    for _ in range(len(items) + 1):
        shutil.rmtree(gen, ignore_errors=True); os.makedirs(gen)
        for i, c in alive.items():
            open(f"{gen}/s{i}.kt", "w").write(kotlin_file(i, c))
        open(f"{gen}/All.kt", "w").write(
            "package uz.d2c.screens.gen\n\nimport androidx.compose.runtime.Composable\n"
            + "".join(f"import uz.d2c.screens.gen.s{i}.Screen as S{i}\n" for i in alive)
            + "\nval ALL_SCREENS: List<Pair<Long, @Composable () -> Unit>> = listOf(\n"
            + ",\n".join(f"    {i}L to {{ S{i}() }}" for i in alive) + "\n)\n")
        shutil.rmtree(tmp, ignore_errors=True)
        rc, log = gradle(work, tmp)
        bad = {}
        for m in re.finditer(r"e: file://\S*/gen/s(\d+)\.kt:(\d+):\d+ (.*)", log):
            bad.setdefault(int(m.group(1)), []).append(f"line {m.group(2)}: {m.group(3)}")
        all_lines = open(f"{gen}/All.kt").read().split("\n")
        for m in re.finditer(r"e: file://\S*/gen/All\.kt:(\d+):\d+ (.*)", log):
            s = re.search(r"\bS(\d+)\b", all_lines[int(m.group(1)) - 1])
            if s:
                bad.setdefault(int(s.group(1)), []).append(m.group(2))
        if not bad:
            break
        for i, e in bad.items():
            errors[i] = "The code did not compile:\n  " + "\n  ".join(e[:5])
            alive.pop(i, None)
        if not alive:
            break
    res = dict(errors)
    for i in alive:
        png = f"{tmp}/{i}.png"
        if os.path.exists(png):
            res[i] = png
        else:
            res[i] = "The code compiled, but the screen did not render." + ("" if rc == 0 else "\n" + log[-800:])
    shutil.rmtree(gen, ignore_errors=True)
    return res, tmp


def side_by_side(orig, png, dst):
    from PIL import Image, ImageDraw
    H = 1000
    a = Image.open(orig).convert("RGB"); a = a.resize((round(a.width * H / a.height), H))
    b = Image.open(png).convert("RGB"); b = b.resize((round(b.width * H / b.height), H))
    c = Image.new("RGB", (a.width + b.width + 20, H + 30), "white")
    c.paste(a, (0, 30)); c.paste(b, (a.width + 20, 30))
    d = ImageDraw.Draw(c)
    d.text((5, 8), "Your screenshot", fill="black"); d.text((a.width + 25, 8), "Rendered model code", fill="black")
    c.save(dst)


def main():
    ap = argparse.ArgumentParser(description="Try the fine-tuned model on screenshots")
    ap.add_argument("images", nargs="+", help="screenshot (png/jpg)")
    ap.add_argument("--adapter", default=f"{ROOT}/runs/sft_5k/final")
    ap.add_argument("--max-new", type=int, default=4096)
    a = ap.parse_args()
    paths = [os.path.abspath(p) for p in a.images]
    for p in paths:
        if not os.path.isfile(p):
            sys.exit(f"File not found: {p}")
    os.makedirs(OUT, exist_ok=True)
    codes = generate(paths, a.adapter, a.max_new)
    names = {}
    for k, p in enumerate(paths, 1):
        base = os.path.splitext(os.path.basename(p))[0]
        names[k] = (p, base)
        open(f"{OUT}/{base}.kt", "w").write(codes[p] + "\n")
    print("Compiling and rendering (~1-2 min)...", flush=True)
    res, tmp = render({k: codes[p] for k, (p, _) in names.items()})
    print("\n===== RESULT =====")
    for k, (p, base) in names.items():
        print(f"\n{os.path.basename(p)}:")
        print(f"  Code:  {OUT}/{base}.kt")
        r = res.get(k)
        if r and r.endswith(".png") and os.path.exists(r):
            side_by_side(p, r, f"{OUT}/{base}_compare.png")
            print(f"  Image: {OUT}/{base}_compare.png")
        else:
            print("  " + (r or "unknown error").replace("\n", "\n  "))
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
