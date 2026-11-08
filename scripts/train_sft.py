"""6-qadam: Qwen3.5-4B ni QLoRA bilan o'qitish (ekran rasmi -> Jetpack Compose kodi).

RTX 4060 (8 GB) uchun: til modeli 4-bit, ko'rish qismi bf16 (muzlatilgan), LoRA faqat til qismida.
Har 100 qadamda checkpoint saqlanadi; to'xtab qolsa --resume bilan davom etadi.
Progress (haqiqiy %, tezlik, qolgan vaqt): logs/train.log

Ishga tushirish (C_INCLUDE_PATH tools/pydev uchun kerak - triton/fla kompilyatsiyasi):
  scripts/train.sh               # to'liq o'qitish
  scripts/train.sh --max-steps 5 # tezlik sinovi
"""
import argparse, io, json, math, os, random, time

import pyarrow.parquet as pq
import torch
from PIL import Image
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = f"{ROOT}/models/Qwen3.5-4B"
IMG_W, IMG_H = 576, 896          # 32 px ga karrali -> 18x28 = 504 rasm tokeni
PROMPT = ("Convert this Android app screenshot into Jetpack Compose code. "
          "Write a single file with `@Composable fun Screen()` that reproduces the layout, texts and colors.")
LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj_qkv|in_proj_z|out_proj)"


def load_image(b):
    return Image.open(io.BytesIO(b)).convert("RGB").resize((IMG_W, IMG_H), Image.BICUBIC)


def prompt_text(processor):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    return processor.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)


def encode(processor, img_bytes, code, ptext):
    """input_ids, labels (faqat kod qismi o'qitiladi), rasm tensorlari."""
    p = processor(text=[ptext], images=[load_image(img_bytes)], return_tensors="pt")
    ans = processor.tokenizer(code + "<|im_end|>\n", add_special_tokens=False, return_tensors="pt").input_ids
    ids = torch.cat([p.input_ids, ans], 1)
    labels = torch.cat([torch.full_like(p.input_ids, -100), ans], 1)
    out = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels,
           "pixel_values": p.pixel_values, "image_grid_thw": p.image_grid_thw}
    if "mm_token_type_ids" in p:  # M-RoPE uchun: qaysi token rasm ekanligi (kod qismi = 0)
        out["mm_token_type_ids"] = torch.cat([p.mm_token_type_ids, torch.zeros_like(ans)], 1)
    return out


def load_model(adapter=None, train=True):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=["visual", "lm_head"])
    model = AutoModelForImageTextToText.from_pretrained(MODEL, quantization_config=bnb, dtype=torch.bfloat16,
                                                        device_map={"": 0})
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=train)
    elif train:
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=LORA_TARGETS,
                                                 task_type="CAUSAL_LM"))
    if train:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model.config.use_cache = False
    return model


def chunked_loss(model, b, chunk=512):
    """Lug'at 248K so'z: hamma logit'larni birdan hisoblash 8 GB ga sig'maydi.
    Shuning uchun lm_head + loss bo'lak-bo'lak (checkpoint bilan) hisoblanadi."""
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    labels = b["labels"]
    h = inner.model(**{k: v for k, v in b.items() if k != "labels"}).last_hidden_state
    h, y = h[0, :-1], labels[0, 1:]          # keyingi tokenni bashorat qilish
    keep = (y != -100).nonzero().squeeze(1)  # faqat kod qismi
    h, y = h[keep], y[keep]
    head = inner.lm_head

    def part(hc, yc):
        return torch.nn.functional.cross_entropy(head(hc).float(), yc, reduction="sum")
    tot = sum(torch.utils.checkpoint.checkpoint(part, h[i:i + chunk], y[i:i + chunk], use_reentrant=False)
              for i in range(0, len(y), chunk))
    return tot / len(y)


def log(msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    open(os.environ.get("D2C_TRAIN_LOG", f"{ROOT}/logs/train.log"), "a").write(line + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=f"{ROOT}/data/sft/train_5k.parquet")
    ap.add_argument("--out", default=f"{ROOT}/runs/sft_5k")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-steps", type=int, default=0, help="faqat tezlik sinovi uchun (mikro-qadamlar)")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)

    t = pq.read_table(a.data, columns=["image", "code"])
    imgs, codes = t.column("image").to_pylist(), t.column("code").to_pylist()
    order = []
    for ep in range(a.epochs):
        o = list(range(len(codes))); random.Random(ep).shuffle(o); order += o
    total_micro = len(order) if not a.max_steps else a.max_steps
    total_opt = math.ceil(total_micro / a.accum)

    state = {"micro": 0}
    ck = f"{a.out}/last"
    processor = AutoProcessor.from_pretrained(MODEL)
    model = load_model(ck if (a.resume and os.path.exists(f"{ck}/adapter_config.json")) else None)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    warm = max(1, int(total_opt * 0.05))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total_opt - warm))))
    if a.resume and os.path.exists(f"{ck}/train_state.pt"):
        st = torch.load(f"{ck}/train_state.pt", weights_only=False)
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); state = st["state"]
        log(f"davom: {state['micro']}/{total_micro} mikro-qadamdan")
    ntrain = sum(p.numel() for p in params)
    log(f"boshlandi: {len(codes)} misol, {total_micro} mikro-qadam, {total_opt} qadam, LoRA {ntrain/1e6:.1f}M parametr")

    ptext = prompt_text(processor)
    model.train()
    t0, tok_seen, loss_acc, n_acc = time.time(), 0, 0.0, 0
    start_micro = state["micro"]
    for mi in range(state["micro"], total_micro):
        b = encode(processor, imgs[order[mi]], codes[order[mi]], ptext)
        b = {k: v.to("cuda") for k, v in b.items()}
        loss = chunked_loss(model, b)
        (loss / a.accum).backward()
        loss_acc += loss.item(); n_acc += 1; tok_seen += b["input_ids"].shape[1]
        state["micro"] = mi + 1
        if state["micro"] % a.accum == 0 or state["micro"] == total_micro:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step = math.ceil(state["micro"] / a.accum)
            el = time.time() - t0
            done = state["micro"] - start_micro
            eta = el / done * (total_micro - state["micro"])
            log(f"{state['micro'] / total_micro * 100:5.1f}%  qadam {step}/{total_opt}  loss {loss_acc / n_acc:.4f}  "
                f"{tok_seen / el:.0f} tok/s  {el / done:.1f} s/misol  qoldi ~{eta / 3600:.1f} soat  "
                f"GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
            loss_acc, n_acc = 0.0, 0
            if step % 100 == 0 or state["micro"] == total_micro:
                model.save_pretrained(ck)
                torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "state": state}, f"{ck}/train_state.pt")
    if not a.max_steps:
        model.save_pretrained(f"{a.out}/final")
        json.dump({"misollar": len(codes), "qadamlar": total_opt, "soat": round((time.time() - t0) / 3600, 2)},
                  open(f"{a.out}/final/summary.json", "w"), indent=1)
        log("TUGADI: " + f"{a.out}/final")


if __name__ == "__main__":
    main()
