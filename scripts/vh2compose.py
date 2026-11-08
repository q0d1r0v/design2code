"""View hierarchy + original screenshot -> Jetpack Compose code.

Rule-based converter (no AI). Main ideas:
  * simplify the tree (drop empty or redundant containers),
  * infer each container's layout from geometry: Column / Row / Box,
  * measure colors, font size, text alignment and corner radius from the screenshot pixels.
All sizes are px -> dp (density 420 dpi, 1 dp = 2.625 px).
"""
import io
import json
import os
import re

import numpy as np
from PIL import Image

DENSITY = 420 / 160
SCREEN_W, SCREEN_H = 1080, 1704
MAX_TEXT = 300
DEFAULT_TEXT = (0x1D, 0x1B, 0x20)  # Material3 onSurface

TEXT_CLS = {"TextView", "CheckedTextView", "MultiAutoCompleteTextView", "TSpanView", "Chip"}
BUTTON_CLS = {"Button", "ToggleButton"}
FIELD_CLS = {"EditText", "AutoCompleteTextView", "MultiAutoCompleteTextView"}
CHECK_CLS = {"CheckBox"}
RADIO_CLS = {"RadioButton"}
SWITCH_CLS = {"Switch", "CompoundButton"}
IMAGE_CLS = {"ImageView", "ImageButton", "Image", "VideoView", "PathView", "SvgView", "RectView",
             "CircleView", "EllipseView", "UseView"}
PROGRESS_CLS = {"ProgressBar", "SeekBar", "RatingBar"}
GENERIC_CLS = {"View", "ViewGroup", "FrameLayout", "LinearLayout", "RelativeLayout", "Spinner"}
SCROLL_V = {"ScrollView", "ListView", "RecyclerView", "GridView", "ExpandableListView", "NestedScrollView"}
SCROLL_H = {"HorizontalScrollView", "ViewPager"}

IMPORTS = """import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.painter.ColorPainter
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
"""


class Skip(Exception):
    """The screen cannot be converted (with a reason)."""


# ---------------------------------------------------------------- helpers
def short(cls):
    return (cls or "").split(".")[-1].split("$")[0]


def dp(px):
    return int(round(px / DENSITY))


def kstr(s):
    """Kotlin string literal (safe escaping)."""
    s = s[:MAX_TEXT]
    out = []
    for ch in s:
        o = ord(ch)
        if ch == "\\": out.append("\\\\")
        elif ch == '"': out.append('\\"')
        elif ch == "$": out.append("\\$")
        elif ch == "\n": out.append("\\n")
        elif ch == "\r": continue
        elif ch == "\t": out.append(" ")
        elif o < 32 or 0xD800 <= o <= 0xDFFF or o in (0xFFFE, 0xFFFF, 0xFFFC, 0xFEFF, 0x2060) \
                or 0x200B <= o <= 0x200F or 0x202A <= o <= 0x202E: continue
        else: out.append(ch)
    return '"' + "".join(out) + '"'


def kcolor(c):
    return "Color(0xFF%02X%02X%02X)" % tuple(int(x) for x in c)


def cdist(a, b):
    return float(np.sqrt(((np.asarray(a, float) - np.asarray(b, float)) ** 2).sum()))


_FONTS = {}
FONT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets/fonts/Roboto-Regular.ttf")


def fit_font(text, font, width_px):
    """Shrinks the font until single-line text fits the given width (using Roboto metrics)."""
    from PIL import ImageFont
    t = text.split("\n")[0][:MAX_TEXT]
    while font > 8:
        f = _FONTS.get(font)
        if f is None:
            f = _FONTS[font] = ImageFont.truetype(FONT_PATH, int(round(font * DENSITY)))
        try:
            length = f.getlength(t)
        except Exception:  # raqm fails on some complex glyphs -> basic layout, then an estimate
            try:
                fb = _FONTS.get(("basic", font))
                if fb is None:
                    fb = _FONTS[("basic", font)] = ImageFont.truetype(FONT_PATH, int(round(font * DENSITY)),
                                                                      layout_engine=ImageFont.Layout.BASIC)
                length = fb.getlength(t)
            except Exception:
                length = len(t) * 0.55 * font * DENSITY
        if length <= width_px:
            break
        font -= 1
    return font


class Px:
    """Measures colors and fonts from screenshot pixels."""

    def __init__(self, img):
        self.a = np.asarray(img.convert("RGB"), dtype=np.int16)

    def region(self, b):
        x1, y1, x2, y2 = b
        return self.a[y1:y2, x1:x2]

    def mode(self, b):
        r = self.region(b)
        if r.size == 0:
            return None
        h, w = r.shape[:2]
        step = max(1, int(np.sqrt(h * w / 20000)))  # sparse sampling on large regions (for speed)
        p = r[::step, ::step].reshape(-1, 3)
        q = p // 8
        key = (q[:, 0].astype(np.int32) << 10) | (q[:, 1].astype(np.int32) << 5) | q[:, 2]
        cnt = np.bincount(key, minlength=32768)
        top = cnt.argmax()
        return tuple(int(v) for v in p[key == top].mean(0)), float(cnt[top] / len(key))

    def ring_color(self, b, inset=3):
        """Element background: the color is uniform along the edges and continues further inside
        (otherwise it is a frame or border line, not a background)."""
        c = self._ring(b, inset)
        if c is None:
            return None
        x1, y1, x2, y2 = b
        if x2 - x1 >= 2 * (inset + 12) + 2 and y2 - y1 >= 2 * (inset + 12) + 2:
            c2 = self._ring(b, inset + 12)
            if c2 is None or cdist(c, c2) > 25:
                return None
        return c

    def _ring(self, b, inset):
        x1, y1, x2, y2 = b
        if x2 - x1 < 2 * inset + 2 or y2 - y1 < 2 * inset + 2:
            return None
        a = self.a
        edges = [a[y1 + inset, x1 + inset:x2 - inset], a[y2 - 1 - inset, x1 + inset:x2 - inset],
                 a[y1 + inset:y2 - inset, x1 + inset], a[y1 + inset:y2 - inset, x2 - 1 - inset]]
        # each edge separately: uniform edges; if at least 3 share a color -> background color
        # (a line or shadow on one edge does not break background detection)
        good = []
        for e in edges:
            med = np.median(e, axis=0)
            if (np.abs(e - med).sum(1) < 30).mean() >= 0.85:
                good.append(med)
        for g in good:
            if sum(np.abs(h - g).sum() < 30 for h in good) >= 3:
                return tuple(int(v) for v in g)
        return None

    def mean(self, b):
        r = self.region(b)
        return tuple(int(v) for v in r.reshape(-1, 3).mean(0)) if r.size else (128, 128, 128)

    def text_style(self, b, bg=None, text="", excl=()):
        """(color, font_dp, lines, horizontal alignment, vertical alignment)."""
        r = self.region(b)
        if r.size == 0:
            return None
        bgc = np.asarray(bg if bg is not None else self.mode(b)[0], float)
        d = np.sqrt(((r - bgc) ** 2).sum(2))
        mask = d > 70
        for e in excl:  # neighbouring elements (e.g. an icon above a title) must not count as ink
            ex1, ey1 = max(e[0] - b[0], 0), max(e[1] - b[1], 0)
            ex2, ey2 = min(e[2] - b[0], mask.shape[1]), min(e[3] - b[1], mask.shape[0])
            if ex2 > ex1 and ey2 > ey1:
                mask[ey1:ey2, ex1:ex2] = False
        if mask.sum() < 6:
            return None
        ink = r[mask]
        far = d[mask]
        col = tuple(int(v) for v in np.median(ink[far >= np.percentile(far, 60)], axis=0))
        rows = mask.any(1)
        bands, start = [], None
        for i, v in enumerate(np.append(rows, False)):
            if v and start is None: start = i
            if not v and start is not None:
                if bands and start - bands[-1][1] <= 3: bands[-1] = (bands[-1][0], i)
                else: bands.append((start, i))
                start = None
        bands = [x for x in bands if x[1] - x[0] >= 4]
        if not bands:
            return None
        hb = sorted(x[1] - x[0] for x in bands)
        # Ink height / font size ratio depends on the letters (Roboto metrics):
        # top: capitals/digits/ascenders ~0.73em, only lowercase ~0.53em; bottom: g,j,p,q,y add +0.21em
        top = 0.76 if re.search(r"[A-Z0-9bdfhklt]|[^\x00-\x7f]", text or "A") else 0.55
        low = 0.22 if re.search(r"[gjpqy,;()]", text or "") else 0.0
        font_px = hb[len(hb) // 2] / (top + low)
        font = max(8, min(48, int(round(font_px / DENSITY))))
        cols = np.where(mask.any(0))[0]
        h, w = mask.shape
        left, right = cols[0], w - 1 - cols[-1]
        if w > 40 and abs(left - right) < 0.08 * w and left > 0.12 * w: ha = "Center"
        elif w > 40 and right < 0.06 * w and left > 0.3 * w: ha = "End"
        else: ha = "Start"
        top, bot = bands[0][0], h - bands[-1][1]
        if abs(top - bot) <= max(4, 0.15 * h): va = "Center"
        elif top < bot: va = "Top"
        else: va = "Bottom"
        return col, font, len(bands), ha, va

    def corner_radius(self, b, fill):
        """Diagonal distance from the corner to the fill color -> radius (dp)."""
        x1, y1, x2, y2 = b
        lim = min(x2 - x1, y2 - y1) // 2
        for k in range(0, lim):
            if cdist(self.a[y1 + k, x1 + k], fill) < 30:
                return min(dp(k * 3.41), dp(lim))
        return dp(lim)

    def fill_box(self, b, fill):
        """The real box covered by the fill color inside the element (removes Material insets)."""
        r = self.region(b)
        m = np.abs(r - np.asarray(fill)).sum(2) < 30
        rows, cols = np.where(m.any(1))[0], np.where(m.any(0))[0]
        if len(rows) == 0:
            return b
        x1, y1 = b[0], b[1]
        return [x1 + int(cols[0]), y1 + int(rows[0]), x1 + int(cols[-1]) + 1, y1 + int(rows[-1]) + 1]


# ---------------------------------------------------------------- tree
class Node:
    __slots__ = ("v", "b", "kids", "kind", "text", "bg", "excl")

    def __init__(self, v, b):
        self.v, self.b, self.kids, self.kind, self.text, self.bg, self.excl = v, b, [], None, "", None, []

    @property
    def w(self): return self.b[2] - self.b[0]

    @property
    def h(self): return self.b[3] - self.b[1]


def clip(b, c):
    return [max(b[0], c[0]), max(b[1], c[1]), min(b[2], c[2]), min(b[3], c[3])]


def build_tree(views):
    nodes = [Node(v, list(v["bounds"])) for v in views]
    roots = []
    for i, n in enumerate(nodes):
        p = views[i].get("parent", -1)
        (nodes[p].kids if 0 <= p < len(nodes) and p != i else roots).append(n)
    screen = [0, 0, SCREEN_W, SCREEN_H]

    def fix(n, cb, depth=0):
        if depth > 60:
            n.kids = []
        n.b = clip(n.b, cb)
        n.kids = [k for k in n.kids]
        for k in n.kids:
            fix(k, n.b, depth + 1)
        n.kids = [k for k in n.kids if k.w >= 3 and k.h >= 3]

    for r in roots:
        fix(r, screen)
    roots = [r for r in roots if r.w >= 3 and r.h >= 3]
    if not roots:
        raise Skip("empty_tree")
    if len(roots) == 1:
        return roots[0]
    top = Node({"class": "FrameLayout"}, screen)
    top.kids = roots
    return top


def classify(n):
    v, c = n.v, short(n.v.get("class"))
    txt = (v.get("text") or "").strip()
    cd = (v.get("content_description") or "").strip()
    # Flutter/RN: text often lives in content_description
    if not txt and cd and not n.kids and re.search(r"\w", cd) and \
            c in GENERIC_CLS | TEXT_CLS | BUTTON_CLS | CHECK_CLS | RADIO_CLS | SWITCH_CLS:
        txt = cd
    n.text = txt
    if c in FIELD_CLS or v.get("editable"):
        return "field"
    if c in CHECK_CLS: return "checkbox"
    if c in RADIO_CLS: return "radio"
    if c in SWITCH_CLS: return "switch"
    if c in PROGRESS_CLS: return "progress"
    if c in BUTTON_CLS and not n.kids: return "button"
    if n.kids:
        return "container"
    if txt and (c in TEXT_CLS or c in ("View", "ViewGroup", "FrameLayout", "LinearLayout", "Spinner")):
        return "text"
    if c in IMAGE_CLS or v.get("content_description"):
        return "image"
    if c in ("View", "ViewGroup", "FrameLayout", "LinearLayout", "RelativeLayout"):
        return "leafbox"
    if txt:
        return "text"
    return "image"


def simplify(n, px, parent_bg):
    """Classify nodes, drop empty containers, collapse single-child containers."""
    n.kind = classify(n)
    own = px.ring_color(n.b)
    distinct = own is not None and cdist(own, parent_bg) > 12
    if n.kind == "container":
        if distinct:
            n.bg = own
        cur = n.bg or parent_bg
        n.kids = [k for k in (simplify(k, px, cur) for k in n.kids) if k is not None]
        if not n.kids:
            if n.bg is None:
                return None
            n.kind = "leafbox"
            return n
        if len(n.kids) == 1 and n.bg is None:
            k = n.kids[0]
            if all(abs(a - b) <= 3 for a, b in zip(k.b, n.b)) or k.kind == "container":
                return _lift(k, n)
        return n
    if n.kind == "leafbox":
        if n.h <= 6 or n.w <= 6:
            n.kind = "divider"
            n.bg = px.mean(n.b)
            return n if cdist(n.bg, parent_bg) > 12 else None
        if not distinct:
            return None
        n.bg = own
        return n
    n.bg = parent_bg
    return n


def _lift(k, n):
    # single child: the child replaces its parent, but keep scrolling if the parent was a scroll container
    if short(n.v.get("class")) in SCROLL_V | SCROLL_H and k.kind == "container" and \
            short(k.v.get("class")) not in SCROLL_V | SCROLL_H:
        k.v = dict(k.v, _scroll=short(n.v.get("class")))
    return k


# ---------------------------------------------------------------- code generation
SPLIT_TOTAL = 60      # screens larger than this are split into parts
SPLIT_MIN = 12        # containers with at least this many nodes become a separate function
FUNC_MAX = 90         # node limit per function (stays well below the JVM 64 KB method limit)


def subtree_size(n):
    return 1 + sum(subtree_size(k) for k in n.kids) if n.kind == "container" else 1


class Writer:
    def __init__(self, px, name="", shared=None):
        self.px = px
        self.name = name
        self.lines = []
        self.local = 0       # nodes in this function
        # screen-wide counters (shared with the part functions)
        self.sh = shared if shared is not None else {"count": 0, "texts": 0, "ink": 0, "parts": [], "split": False}

    @property
    def count(self): return self.sh["count"]

    @property
    def texts(self): return self.sh["texts"]

    @property
    def texts_ink(self): return self.sh["ink"]

    def emit(self, depth, s):
        self.lines.append("    " * depth + s)

    @staticmethod
    def size_mod(n, parent=None, fill_w=False):
        w, h = dp(n.b[2]) - dp(n.b[0]), dp(n.b[3]) - dp(n.b[1])
        if fill_w:
            return f".fillMaxWidth().height({h}.dp)"
        return f".size({w}.dp, {h}.dp)" if w != h else f".size({w}.dp)"

    def layout(self, n):
        kids = n.kids
        if len(kids) == 1:
            return "box"
        ys = sorted(kids, key=lambda k: (k.b[1], k.b[0]))
        if all(ys[i + 1].b[1] >= ys[i].b[3] - 4 for i in range(len(ys) - 1)):
            return "column"
        xs = sorted(kids, key=lambda k: (k.b[0], k.b[1]))
        if all(xs[i + 1].b[0] >= xs[i].b[2] - 4 for i in range(len(xs) - 1)):
            return "row"
        return "box"

    def node(self, n, depth, pos_mod):
        """pos_mod: position modifier given by the parent container (padding/offset)."""
        if self.sh["split"] and n.kind == "container" and subtree_size(n) >= SPLIT_MIN:
            # move a large subtree into its own @Composable function
            part = f"{self.name}Part{len(self.sh['parts']) + 1}"
            self.sh["parts"].append(None)
            idx = len(self.sh["parts"]) - 1
            w2 = Writer(self.px, self.name, self.sh)
            w2.emit(0, "@Composable")
            w2.emit(0, f"private fun {part}(modifier: Modifier = Modifier) {{")
            w2.k_container(n, 1, "", base="modifier")
            w2.emit(0, "}")
            if w2.local > FUNC_MAX:
                raise Skip("function_too_large")
            self.sh["parts"][idx] = "\n".join(w2.lines)
            self.emit(depth, f"{part}(Modifier{pos_mod})" if pos_mod else f"{part}()")
            return
        self.sh["count"] += 1
        self.local += 1
        k = n.kind
        getattr(self, "k_" + k)(n, depth, pos_mod)

    # --- container
    def k_container(self, n, depth, pos_mod, root=False, base="Modifier"):
        lay = self.layout(n)
        for c in n.kids:
            c.excl = [o.b for o in n.kids if o is not c and o.b[0] < c.b[2] and c.b[0] < o.b[2]
                      and o.b[1] < c.b[3] and c.b[1] < o.b[3] and o.w * o.h < c.w * c.h]
        scroll = n.v.get("_scroll") or short(n.v.get("class"))
        mod = base + pos_mod + (".fillMaxSize()" if root else self.size_mod(n))
        if n.bg is not None:
            mod += f".background({kcolor(n.bg)})"
        if lay == "column" and scroll in SCROLL_V:
            mod += ".verticalScroll(rememberScrollState())"
        if lay == "row" and (scroll in SCROLL_H or (scroll in SCROLL_V and n.w > n.h * 1.5)):
            mod += ".horizontalScroll(rememberScrollState())"
        cx, cy = dp(n.b[0]), dp(n.b[1])
        if lay == "column":
            self.emit(depth, f"Column({mod}) {{")
            prev = cy
            for c in sorted(n.kids, key=lambda c: (c.b[1], c.b[0])):
                gap = dp(c.b[1]) - prev
                if gap > 0:
                    self.emit(depth + 1, f"Spacer(Modifier.height({gap}.dp))")
                start = dp(c.b[0]) - cx
                self.node(c, depth + 1, f".padding(start = {start}.dp)" if start > 0 else "")
                prev = prev + max(gap, 0) + (dp(c.b[3]) - dp(c.b[1]))
            self.emit(depth, "}")
        elif lay == "row":
            self.emit(depth, f"Row({mod}) {{")
            prev = cx
            for c in sorted(n.kids, key=lambda c: (c.b[0], c.b[1])):
                gap = dp(c.b[0]) - prev
                if gap > 0:
                    self.emit(depth + 1, f"Spacer(Modifier.width({gap}.dp))")
                top = dp(c.b[1]) - cy
                self.node(c, depth + 1, f".padding(top = {top}.dp)" if top > 0 else "")
                prev = prev + max(gap, 0) + (dp(c.b[2]) - dp(c.b[0]))
            self.emit(depth, "}")
        else:
            self.emit(depth, f"Box({mod}) {{")
            back = ("image", "leafbox", "divider")
            for c in sorted(n.kids, key=lambda c: (c.kind not in back, c.b[1], c.b[0])):
                x, y = dp(c.b[0]) - cx, dp(c.b[1]) - cy
                off = f".offset(x = {x}.dp, y = {y}.dp)" if (x or y) else ""
                self.node(c, depth + 1, off)
            self.emit(depth, "}")

    # --- leaves
    def _text_args(self, n, bg, sty, size_known=True):
        col, font, lines, ha, va = sty
        hdp = dp(n.b[3]) - dp(n.b[1])
        font = max(8, min(font, int(hdp / (1.15 if lines <= 1 else 1.25 * lines))))
        if lines <= 1:
            font = fit_font(n.text, font, n.w - 4)
        args = [kstr(n.text)]
        if cdist(col, DEFAULT_TEXT) > 40:
            args.append(f"color = {kcolor(col)}")
        args.append(f"fontSize = {font}.sp")
        if lines > 1:
            args.append(f"lineHeight = {int(round(font * 1.25))}.sp")
        if ha != "Start":
            args.append(f"textAlign = TextAlign.{ha}")
        if lines <= 1:
            args.append("softWrap = false")
        args.append(f"maxLines = {max(1, lines)}")
        args.append("overflow = TextOverflow.Clip")
        return args

    def k_text(self, n, depth, pos_mod):
        own = self.px.ring_color(n.b, inset=1)
        bgmod = ""
        if own is not None and cdist(own, n.bg) > 20:
            n.bg = own
            bgmod = f".background({kcolor(own)})"
        sty = self.px.text_style(n.b, n.bg, n.text, n.excl)
        self.sh["texts"] += 1
        self.sh["ink"] += sty is not None
        if sty is None:
            lum = 0.299 * n.bg[0] + 0.587 * n.bg[1] + 0.114 * n.bg[2] if n.bg else 255
            sty = ((0, 0, 0) if lum > 128 else (255, 255, 255), max(8, min(16, int(dp(n.h) / 1.4))), 1, "Start", "Center")
        args = self._text_args(n, n.bg, sty)
        va, ha = sty[4], sty[3]
        if va == "Top":
            args.insert(1, f"modifier = Modifier{pos_mod}{self.size_mod(n)}{bgmod}")
            self.emit(depth, f"Text({', '.join(args)})")
        else:
            align = {"Center": "Center", "Top": "Top", "Bottom": "Bottom"}[va] + {"Start": "Start", "Center": "", "End": "End"}[ha]
            align = {"Center": "Center", "CenterStart": "CenterStart", "CenterEnd": "CenterEnd", "Bottom": "BottomCenter",
                     "BottomStart": "BottomStart", "BottomEnd": "BottomEnd"}.get(align, "CenterStart")
            self.emit(depth, f"Box(Modifier{pos_mod}{self.size_mod(n)}{bgmod}, contentAlignment = Alignment.{align}) {{")
            self.emit(depth + 1, f"Text({', '.join(args)})")
            self.emit(depth, "}")

    def k_button(self, n, depth, pos_mod):
        m = self.px.mode(n.b)
        fill = m[0] if m and m[1] > 0.35 else None
        if fill is not None and cdist(fill, n.bg) > 20:
            fb = self.px.fill_box(n.b, fill)
            if fb[2] - fb[0] >= 6 and fb[3] - fb[1] >= 6:
                # Material insets: keep the outer gap as padding
                pt, pb = dp(fb[1]) - dp(n.b[1]), dp(n.b[3]) - dp(fb[3])
                pl, pr = dp(fb[0]) - dp(n.b[0]), dp(n.b[2]) - dp(fb[2])
                inner = Node(n.v, fb)
                inner.text = n.text
                inset = ""
                if pt or pb or pl or pr:
                    inset = f".padding(start = {pl}.dp, top = {pt}.dp, end = {pr}.dp, bottom = {pb}.dp)"
                r = self.px.corner_radius(fb, fill)
                rp = int(r * DENSITY) + 2
                tb = [fb[0] + rp, fb[1] + 2, fb[2] - rp, fb[3] - 2] if fb[2] - fb[0] > 2 * rp + 10 else fb
                sty = self.px.text_style(tb, fill, n.text, n.excl) if n.text else None
                self.emit(depth, f"Button(onClick = {{}}, modifier = Modifier{pos_mod}{self.size_mod(n)}{inset}, "
                                 f"shape = RoundedCornerShape({r}.dp), colors = ButtonDefaults.buttonColors(containerColor = {kcolor(fill)}), "
                                 f"contentPadding = PaddingValues(0.dp)) {{")
                if n.text:
                    if sty is None:
                        sty = ((255, 255, 255), 14, 1, "Center", "Center")
                    self.emit(depth + 1, f"Text({', '.join(self._text_args(inner, fill, sty))})")
                self.emit(depth, "}")
                return
        if not n.text:
            self.k_image(n, depth, pos_mod)
            return
        sty = self.px.text_style(n.b, n.bg, n.text, n.excl) if n.text else None
        self.emit(depth, f"TextButton(onClick = {{}}, modifier = Modifier{pos_mod}{self.size_mod(n)}, "
                         f"contentPadding = PaddingValues(0.dp)) {{")
        if n.text:
            if sty is None:
                sty = ((0x67, 0x50, 0xA4), 14, 1, "Center", "Center")
            self.emit(depth + 1, f"Text({', '.join(self._text_args(n, n.bg, sty))})")
        self.emit(depth, "}")

    def k_field(self, n, depth, pos_mod):
        # analyse the inner area (10 px from each edge) so frame lines do not count as ink
        ib = [n.b[0] + 10, n.b[1] + 10, n.b[2] - 10, n.b[3] - 10] if n.w > 40 and n.h > 30 else n.b
        sty = self.px.text_style(ib, n.bg, n.text, n.excl) if n.text else None
        hdp = dp(n.b[3]) - dp(n.b[1])
        font = sty[1] if sty else 16
        font = max(10, min(font, 22, int(hdp * 0.5)))
        col = sty[0] if sty else (0x1D, 0x1B, 0x20)
        pw = ", visualTransformation = PasswordVisualTransformation()" if n.v.get("is_password") else ""
        line = self.px.mean([n.b[0], max(n.b[3] - 6, n.b[1]), n.b[2], n.b[3]])
        self.emit(depth, f"BasicTextField(value = {kstr(n.text)}, onValueChange = {{}}, "
                         f"modifier = Modifier{pos_mod}{self.size_mod(n)}, "
                         f"textStyle = TextStyle(color = {kcolor(col)}, fontSize = {font}.sp), singleLine = true{pw}, "
                         f"decorationBox = {{ inner ->")
        self.emit(depth + 1, "Column(Modifier.fillMaxSize()) {")
        self.emit(depth + 2, "Box(Modifier.weight(1f).fillMaxWidth().padding(horizontal = 8.dp), contentAlignment = Alignment.CenterStart) { inner() }")
        self.emit(depth + 2, f"HorizontalDivider(color = {kcolor(line)})")
        self.emit(depth + 1, "}")
        self.emit(depth, "})")

    def _labeled(self, n, depth, pos_mod, control, control_first=True):
        if not n.text:
            self.emit(depth, f"Box(Modifier{pos_mod}{self.size_mod(n)}, contentAlignment = Alignment.CenterStart) {{ {control} }}")
            return
        sty = self.px.text_style([n.b[0] + min(int(48 * DENSITY), n.w // 2), n.b[1], n.b[2], n.b[3]], n.bg, n.text, n.excl) \
            if control_first else self.px.text_style([n.b[0], n.b[1], max(n.b[0] + 4, n.b[2] - int(60 * DENSITY)), n.b[3]], n.bg, n.text, n.excl)
        if sty is None:
            sty = ((0, 0, 0), 14, 1, "Start", "Center")
        t = f"Text({', '.join(self._text_args(n, n.bg, sty))})"
        self.emit(depth, f"Row(Modifier{pos_mod}{self.size_mod(n)}, verticalAlignment = Alignment.CenterVertically) {{")
        if control_first:
            self.emit(depth + 1, control)
            self.emit(depth + 1, t)
        else:
            self.emit(depth + 1, f"Box(Modifier.weight(1f)) {{ {t} }}")
            self.emit(depth + 1, control)
        self.emit(depth, "}")

    def k_checkbox(self, n, depth, pos_mod):
        self._labeled(n, depth, pos_mod, f"Checkbox(checked = {str(bool(n.v.get('checked'))).lower()}, onCheckedChange = {{}})")

    def k_radio(self, n, depth, pos_mod):
        self._labeled(n, depth, pos_mod, f"RadioButton(selected = {str(bool(n.v.get('checked'))).lower()}, onClick = {{}})")

    def k_switch(self, n, depth, pos_mod):
        self._labeled(n, depth, pos_mod, f"Switch(checked = {str(bool(n.v.get('checked'))).lower()}, onCheckedChange = {{}})",
                      control_first=False)

    def k_progress(self, n, depth, pos_mod):
        if 0.6 <= n.w / max(n.h, 1) <= 1.6:
            s = min(dp(n.w), dp(n.h))
            self.emit(depth, f"Box(Modifier{pos_mod}{self.size_mod(n)}, contentAlignment = Alignment.Center) {{ "
                             f"CircularProgressIndicator(progress = {{ 0.75f }}, modifier = Modifier.size({s}.dp)) }}")
        else:
            self.emit(depth, f"Box(Modifier{pos_mod}{self.size_mod(n)}, contentAlignment = Alignment.Center) {{ "
                             f"LinearProgressIndicator(progress = {{ 0.5f }}, modifier = Modifier.fillMaxWidth()) }}")

    def k_image(self, n, depth, pos_mod):
        cd = (n.v.get("content_description") or "").strip()
        cds = kstr(cd) if cd else "null"
        self.emit(depth, f"Image(painter = ColorPainter({kcolor(self.px.mean(n.b))}), contentDescription = {cds}, "
                         f"modifier = Modifier{pos_mod}{self.size_mod(n)})")

    def k_leafbox(self, n, depth, pos_mod):
        self.emit(depth, f"Box(Modifier{pos_mod}{self.size_mod(n)}.background({kcolor(n.bg)}))")

    def k_divider(self, n, depth, pos_mod):
        if n.w >= n.h:
            self.emit(depth, f"HorizontalDivider(modifier = Modifier{pos_mod}.width({dp(n.b[2]) - dp(n.b[0])}.dp), "
                             f"thickness = {max(1, dp(n.h))}.dp, color = {kcolor(n.bg)})")
        else:
            self.emit(depth, f"VerticalDivider(modifier = Modifier{pos_mod}.height({dp(n.b[3]) - dp(n.b[1])}.dp), "
                             f"thickness = {max(1, dp(n.w))}.dp, color = {kcolor(n.bg)})")


def convert(views, image_bytes, name, max_nodes=250):
    """-> (kotlin_code, node_count, text_sync). Raises Skip when the screen is dropped."""
    img = Image.open(io.BytesIO(image_bytes))
    if img.size != (SCREEN_W, SCREEN_H):
        raise Skip("bad_image_size")
    px = Px(img)
    root = build_tree(views)
    root_bg = px.ring_color([0, 0, SCREEN_W, SCREEN_H], inset=2) or px.mode([0, 0, SCREEN_W, SCREEN_H])[0]
    root.kind = "container"
    root.text = ""
    root.kids = [k for k in (simplify(k, px, root_bg) for k in root.kids) if k is not None]
    if not root.kids:
        raise Skip("empty_screen")
    root.bg = root_bg
    if subtree_size(root) > max_nodes:
        raise Skip("too_many_nodes")
    w = Writer(px, name)
    w.sh["split"] = subtree_size(root) > SPLIT_TOTAL
    w.emit(0, "@Composable")
    w.emit(0, f"fun {name}() {{")
    w.k_container(root, 1, "", root=True)
    w.emit(0, "}")
    if w.local > FUNC_MAX:
        raise Skip("function_too_large")
    if w.count > max_nodes:
        raise Skip("too_many_nodes")
    if w.count < 3:
        raise Skip("too_few_nodes")
    sync = w.texts_ink / w.texts if w.texts else 1.0
    code = "\n\n".join(["\n".join(w.lines)] + [p for p in w.sh["parts"] if p])
    return code, w.count, sync
