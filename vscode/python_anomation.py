# batch_title_paragraph_anim.py
# Create N animated reels in one run (one MP4 per input).
# Each item: <Title> >> <Description> || <Example code...>
#
# INPUTS ARE EMBEDDED BELOW IN ITEM_TEXTS. No interactive prompts.
# Content is hard-clamped to remain INSIDE the white outline border.

from moviepy.editor import *
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np
import os, re, textwrap, sys, argparse, unicodedata, glob

# ========= External assets (given) =========
PY_LOGO_PATH   = r"C:\Users\LENOVO\Downloads\SwitchTech\mandatory files\python_logo.png"
WATERMARK_PATH = r"C:\Users\LENOVO\Downloads\SwitchTech\mandatory files\SwitchTech lite.png"

# ========= EMBEDDED INPUTS (edit this list only) =========
ITEM_TEXTS = [

"""
GCD & LCM >>
Find GCD and LCM of a list of numbers ||
import math

nums = list(map(int, input("Enter numbers: ").split()))

gcd_val = nums[0]
for n in nums[1:]:
gcd_val = math.gcd(gcd_val, n)

lcm_val = nums[0]
for n in nums[1:]:
lcm_val = lcm_val * n // math.gcd(lcm_val, n)

print("GCD:", gcd_val)
print("LCM:", lcm_val)
"""
,
"""
Order-Preserving Unique >>
Remove duplicates from a list while preserving order ||
nums = list(map(int, input("Enter numbers: ").split()))
seen = set()
result = []
for n in nums:
if n not in seen:
seen.add(n)
result.append(n)
print("After removing duplicates:", result)
"""
,
"""
Second Largest >>
Find the second largest element in a list without sorting ||
nums = list(map(int, input("Enter numbers: ").split()))
if len(nums) < 2:
print("Not enough elements")
else:
first = second = None
for n in nums:
if first is None or n > first:
second = first
first = n
elif n != first and (second is None or n > second):
second = n
if second is None:
print("All elements are equal")
else:
print("Second largest element:", second)
"""
,
"""
Balanced Parentheses >>
Check if an expression has balanced brackets using a stack ||
expr = input("Enter expression: ")
stack = []
pairs = {')': '(', ']': '[', '}': '{'}
balanced = True

for ch in expr:
if ch in "([{":
stack.append(ch)
elif ch in ")]}":
if not stack or stack[-1] != pairs[ch]:
balanced = False
break
stack.pop()

if balanced and not stack:
print("Balanced")
else:
print("Not Balanced")
"""
,
"""
Word Frequency >>
Count frequency of words and print in descending order of frequency ||
text = input("Enter a sentence: ")
words = text.split()
freq = {}
for w in words:
w = w.lower()
freq[w] = freq.get(w, 0) + 1

sorted_items = sorted(freq.items(), key=lambda x: x[1], reverse=True)
for word, count in sorted_items:
print(word, ":", count)
"""
,
"""
Nested List Sum >>
Recursively find the sum of all numbers in a nested list ||
def nested_sum(lst):
total = 0
for item in lst:
if isinstance(item, list):
total += nested_sum(item)
else:
total += item
return total

Example input: [1, [2, 3], [4, [5, 6]]]
Enter as: 1 [2,3] [4,[5,6]] is hard to parse, so we use eval for simplicity.

nested = eval(input("Enter a nested list (e.g. [1,[2,3],[4,[5,6]]]): "))
print("Sum of all elements:", nested_sum(nested))
"""
]

# ------------------ CLI (kept for tuning, not for data entry) ------------------
parser = argparse.ArgumentParser(
    description="Batch: Yellow title + white description + VSCode-like code panel. "
                "Input format: <Title> >> <Description> || <Example code...>"
)
parser.add_argument("--hold", type=float, default=3.0, help="Seconds to hold final full text on screen.")
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--wrap_code", type=int, default=0, help="(Optional) Soft wrap width for code lines (chars); 0 = auto-scale panel.")
parser.add_argument("--outdir", type=str, default="outputs", help="Folder to save MP4s and images.")
parser.add_argument("--fast", action="store_true", help="Fast test mode: shorter durations, lower fps, no watermark/logo")
parser.add_argument("--max-items", type=int, default=0, help="(Test) maximum number of items to render; 0 = all")
args, _unk = parser.parse_known_args()

# ------------------ Canvas / Style ------------------
W, H = 1080, 1520
BG = (0, 0, 0)
YELLOW = (247, 204, 69)
WHITE = (245, 245, 245)

# Border + safe padding INSIDE the border
BORDER_COLOR = (255, 255, 255, 255)
BORDER_THICK_PX = 15
INNER_PAD = 24  # inner inset beyond the border
CONTENT_X = BORDER_THICK_PX + INNER_PAD  # left bound for all content
CONTENT_R = W - BORDER_THICK_PX - INNER_PAD
MAX_CONTENT_W = CONTENT_R - CONTENT_X  # max width allowed for any rendered block

TITLE_Y = 140
PARA_GAP = 28
LINE_SP_EXTRA = 10

FPS = args.fps
CLIP_LONG_DUR = 600.0
FINAL_DURATION_PAD = 0.1
TAIL_HOLD = max(0.0, float(args.hold))
# Fast mode: shorten durations and reduce expensive effects (useful for testing)
if getattr(args, "fast", False):
    FPS = min(FPS, 15)
    CLIP_LONG_DUR = 6.0
    TAIL_HOLD = 0.0

# ------------------ VSCode-like code panel theme ------------------
CODE_PANEL_BG   = (30, 34, 39, 235)
CODE_PANEL_EDGE = (50, 54, 61, 255)
CODE_DEFAULT    = (212, 212, 212, 255)

CODE_COLOR_KW   = (197, 134, 192, 255)
CODE_COLOR_BU   = (97, 175, 239, 255)
CODE_COLOR_STR  = (206, 145, 120, 255)
CODE_COLOR_NUM  = (181, 206, 168, 255)
CODE_COLOR_COM  = (106, 153, 85, 255)

CODE_GUTTER_PAD = 24
CODE_TOP_PAD    = 20
CODE_BOTTOM_PAD = 24
CODE_CORNER_R   = 18

# ------------------ Fonts ------------------
def pick_font(bold=False):
    c = []
    if os.name == "nt":
        base = r"C:\Windows\Fonts"
        c += [os.path.join(base, "segoeuib.ttf" if bold else "segoeui.ttf")]
        c += [os.path.join(base, "arialbd.ttf" if bold else "arial.ttf")]
    else:
        c += ["/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf"]
        c += ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    for p in c:
        if os.path.exists(p):
            return p
    return None

def pick_mono_font():
    c = []
    if os.name == "nt":
        base = r"C:\Windows\Fonts"
        c += [os.path.join(base, "consola.ttf")]
        c += [os.path.join(base, "cour.ttf")]
    else:
        c += ["/System/Library/Fonts/Supplemental/Menlo.ttf"]
        c += ["/System/Library/Fonts/Supplemental/Courier New.ttf"]
        c += ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"]
    for p in c:
        if os.path.exists(p):
            return p
    return pick_font(False)

FONT_BOLD = pick_font(True)
FONT_REG  = pick_font(False)
FONT_MONO = pick_mono_font()

TITLE_SIZE = 76
WHITE_SIZE = 46
CODE_SIZE  = 42

def get_font(size, bold=False, mono=False):
    if mono:
        path = FONT_MONO
    else:
        path = FONT_BOLD if bold else FONT_REG
    return ImageFont.truetype(path or "", size=size)

def line_height(size, bold=False, mono=False, extra=LINE_SP_EXTRA):
    font = get_font(size, bold=bold, mono=mono)
    ascent, descent = font.getmetrics()
    return ascent + descent + extra

def white_line_height():
    return line_height(WHITE_SIZE)

# ------------------ Text utils ------------------
def slugify(value, allow_unicode=False):
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    return re.sub(r'[-\s]+', '-', value) or "video"

def normalize_code_block(s: str) -> str:
    s = (s or "").strip()
    if "\n" in s:
        return s
    s = re.sub(r'([}\)])\s+(?=[A-Za-z_#])', r'\1\n', s)
    s = re.sub(r'\s+(?=(print|return|raise|yield|for|while|if|elif|else|try|except|finally|with|def|class)\b)', r'\n', s)
    s = re.sub(r'\s{2,}', '\n', s)
    return s

# NOTE: Delimiters are ">>" and "||"
def parse_title_desc_code(s):
    """
    Expected format:
      <Title> >> <Description> || <Example code...>
    Only the first '>>' splits Title vs rest, and only the first '||' splits
    Description vs Code. Everything after the first '||' belongs to Code.
    """
    if not s:
        return "", "", ""
    parts = s.split(">>", 1)
    if len(parts) == 2:
        title_raw = parts[0].strip()
        rest = parts[1].strip()
    else:
        return s.strip(), "", ""
    desc, code = "", ""
    if "||" in rest:
        desc_part, code_part = rest.split("||", 1)
        desc = desc_part.strip(" -–—\n\t ")
        code = normalize_code_block(code_part.lstrip())
    else:
        desc = rest.strip()
        code = ""
    return title_raw, desc, code

# ------------------ Syntax coloring ------------------
_PY_KW = {"False","None","True","and","as","assert","async","await","break","class","continue","def",
          "del","elif","else","except","finally","for","from","global","if","import","in","is",
          "lambda","nonlocal","not","or","pass","raise","return","try","while","with","yield"}
_PY_BUILTINS = {"print","range","len","enumerate","int","float","str","list","dict","set","tuple",
                "bool","sum","min","max","open","zip","map","filter","any","all","sorted",
                "abs","pow","round","isinstance","type"}

def tokenize_python(line: str):
    import re
    out = []
    triq = re.compile(r"('''.*?'''|\"\"\".*?\"\"\")", re.DOTALL)
    sq   = re.compile(r"(\".*?\"|'.*?')", re.DOTALL)
    num  = re.compile(r"\b\d+(?:\.\d+)?\b")
    com  = re.compile(r"#.*$")
    ident= re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
    m = com.search(line)
    comment_start = m.start() if m else None
    def add(txt, col):
        if txt: out.append((txt, col))
    def paint(seg):
        j=0
        while j<len(seg):
            for pat,col in [(triq,CODE_COLOR_STR),(sq,CODE_COLOR_STR),(num,CODE_COLOR_NUM)]:
                m=pat.match(seg,j)
                if m: add(m.group(),col); j=m.end(); break
            else:
                m=ident.match(seg,j)
                if m:
                    tok=m.group()
                    if tok in _PY_KW: add(tok,CODE_COLOR_KW)
                    elif tok in _PY_BUILTINS: add(tok,CODE_COLOR_BU)
                    else: add(tok,CODE_DEFAULT)
                    j=m.end()
                else: add(seg[j],CODE_DEFAULT); j+=1
    if comment_start is not None:
        paint(line[:comment_start])
        add(line[comment_start:], CODE_COLOR_COM)
    else:
        paint(line)
    return out

# ------------------ Pixel-accurate wrapping ------------------
def wrap_text_to_width(text: str, font: ImageFont.FreeTypeFont, max_px: int):
    """
    Wrap paragraphs by measuring pixel width. Preserves blank lines.
    """
    tmp = Image.new("L", (1, 1)); d = ImageDraw.Draw(tmp)
    wrapped_lines = []
    for para in (text or "").split("\n"):
        if para.strip() == "":
            wrapped_lines.append("")  # preserve blank line
            continue
        words = para.split(" ")
        line = ""
        for w in words:
            cand = (line + " " + w).strip() if line else w
            w_px = d.textbbox((0,0), cand, font=font)[2]
            if w_px <= max_px:
                line = cand
            else:
                if line:
                    wrapped_lines.append(line)
                    line = w  # start new line with current word
                else:
                    # single long token: hard-break by characters
                    token = w
                    chunk = ""
                    for ch in token:
                        cand2 = (chunk + ch)
                        if d.textbbox((0,0), cand2, font=font)[2] <= max_px:
                            chunk = cand2
                        else:
                            if chunk:
                                wrapped_lines.append(chunk)
                            chunk = ch
                    line = chunk
        if line:
            wrapped_lines.append(line)
    return wrapped_lines

# ------------------ Renderers (bounded to MAX_CONTENT_W) ------------------
TITLE_SIZE = 76
WHITE_SIZE = 46
CODE_SIZE  = 42

def render_paragraph_rgba_bounded(text, size, color, bold=False, max_px=MAX_CONTENT_W):
    font = get_font(size, bold=bold)
    lines = wrap_text_to_width(text, font, max_px)
    ascent, descent = font.getmetrics()
    line_h = ascent + descent + LINE_SP_EXTRA
    tmp = Image.new("L", (1, 1)); dtmp = ImageDraw.Draw(tmp)
    max_w = max([dtmp.textbbox((0,0),ln,font=font)[2] if ln else 1 for ln in lines] + [1])
    img = Image.new("RGBA",(max_w+12,line_h*len(lines)+12),(0,0,0,0))
    d=ImageDraw.Draw(img); y=6
    for ln in lines:
        d.text((6,y), ln, font=font, fill=color)
        y+=line_h
    return np.array(img), line_h, lines

def render_code_line_rgba(line_text):
    font=get_font(CODE_SIZE,mono=True)
    ascent,descent=font.getmetrics()
    line_h=ascent+descent+LINE_SP_EXTRA+4
    tokens=tokenize_python(line_text if line_text.strip()!="" else " ")
    tmp=Image.new("L",(1,1));dtmp=ImageDraw.Draw(tmp)
    total_w=sum([max(1,dtmp.textbbox((0,0),t,font=font)[2]) for t,_ in tokens])
    img=Image.new("RGBA",(total_w,line_h+12),(0,0,0,0))
    draw=ImageDraw.Draw(img);x=0;y=6
    for t,col in tokens:
        draw.text((x,y),t,font=font,fill=col)
        x+=max(1,dtmp.textbbox((0,0),t,font=font)[2])
    return np.array(img), line_h

def render_full_code_panel_rgba_bounded(code_text, max_panel_w):
    """
    Renders code panel; if wider than max_panel_w, scales the entire panel down to fit.
    This guarantees it stays inside the white border.
    """
    lines=code_text.splitlines() or [""]
    rendered=[render_code_line_rgba(ln) for ln in lines]
    max_w=max(r[0].shape[1] for r in rendered)
    total_h=sum(r[1] for r in rendered)

    panel_w=max_w+CODE_GUTTER_PAD*2
    panel_h=total_h+CODE_TOP_PAD+CODE_BOTTOM_PAD
    panel_img=Image.new("RGBA",(panel_w,panel_h),(0,0,0,0))
    pd=ImageDraw.Draw(panel_img)
    pd.rounded_rectangle([(0,0),(panel_w-1,panel_h-1)],radius=CODE_CORNER_R,
                         fill=CODE_PANEL_BG,outline=CODE_PANEL_EDGE,width=2)
    cur_y=CODE_TOP_PAD;inner_x=CODE_GUTTER_PAD
    for arr,lh in rendered:
        panel_img.alpha_composite(Image.fromarray(arr),dest=(inner_x,cur_y))
        cur_y+=lh

    # Scale down if needed
    if panel_img.size[0] > max_panel_w:
        scale = max_panel_w / float(panel_img.size[0])
        new_w = int(panel_img.size[0] * scale)
        new_h = int(panel_img.size[1] * scale)
        panel_img = panel_img.resize((new_w, new_h), Image.LANCZOS)

    arr = np.array(panel_img)
    return arr, panel_img.size[0], panel_img.size[1]

# ---- Border clip
def white_border_clip(start=0.0, duration=CLIP_LONG_DUR):
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    t = max(1, int(BORDER_THICK_PX))
    d.rectangle([(0, 0), (W-1, t-1)], fill=BORDER_COLOR)       # top
    d.rectangle([(0, H-t), (W-1, H-1)], fill=BORDER_COLOR)     # bottom
    d.rectangle([(0, 0), (t-1, H-1)], fill=BORDER_COLOR)       # left
    d.rectangle([(W-t, 0), (W-1, H-1)], fill=BORDER_COLOR)     # right
    return ImageClip(np.array(img)).set_start(start).set_position((0, 0)).set_duration(duration)

# ---- Assets (logo + watermark) helpers
def load_rgba(path):
    im = Image.open(path).convert("RGBA")
    return im

def logo_clip_top_right(start=0.0, duration=CLIP_LONG_DUR, pad=36, target_w=220):
    if not os.path.exists(PY_LOGO_PATH):
        return None
    im = load_rgba(PY_LOGO_PATH)
    w, h = im.size
    scale = target_w / float(w)
    new = im.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    arr = np.array(new)
    clip = ImageClip(arr).set_start(start).set_duration(duration)
    # stays inside border by design:
    def pos(_t):
        return (W - BORDER_THICK_PX - pad - arr.shape[1], BORDER_THICK_PX + pad)
    return clip.set_position(pos)

def big_center_watermark(start=0.0, duration=CLIP_LONG_DUR, fill_frac=0.82, opacity=0.12):
    if not os.path.exists(WATERMARK_PATH):
        return None
    im = load_rgba(WATERMARK_PATH)
    iw, ih = im.size
    avail_w = int((W - 2*(BORDER_THICK_PX + INNER_PAD)) * fill_frac)
    avail_h = int((H - 2*(BORDER_THICK_PX + INNER_PAD)) * fill_frac)
    scale = min(avail_w / iw, avail_h / ih)
    new = im.resize((int(iw*scale), int(ih*scale)), Image.LANCZOS)
    arr = np.array(new)
    clip = ImageClip(arr).set_start(start).set_duration(duration).set_opacity(opacity)
    def pos(_t):
        return ((W - arr.shape[1]) // 2, (H - arr.shape[0]) // 2)
    return clip.set_position(pos)

# ------------------ Animation helpers (use bounded renderers) ------------------
def title_clip(text,start,x,y,fade=0.6):
    arr,_,_=render_paragraph_rgba_bounded(text,TITLE_SIZE,YELLOW,True,MAX_CONTENT_W)
    clip=ImageClip(arr).set_start(start).set_position((x,y)).fx(vfx.fadein,fade).set_duration(CLIP_LONG_DUR)
    return clip,arr.shape[0]

def description_line_by_line(text,start,x,y,delay=0.45,fade=0.35):
    _,lh,lines=render_paragraph_rgba_bounded(text,WHITE_SIZE,WHITE,False,MAX_CONTENT_W)
    clips=[];cur_y=y;t=start
    for ln in lines:
        arr,_,_=render_paragraph_rgba_bounded(ln,WHITE_SIZE,WHITE,False,MAX_CONTENT_W)
        ic=ImageClip(arr).set_start(t).set_position((x,cur_y)).fx(vfx.fadein,fade).set_duration(CLIP_LONG_DUR)
        clips.append(ic);cur_y+=arr.shape[0];t+=delay
    return clips,cur_y,t

def example_heading_clip(start,x,y):
    arr,_,_=render_paragraph_rgba_bounded("Example:",WHITE_SIZE,WHITE,True,MAX_CONTENT_W)
    clip=ImageClip(arr).set_start(start).set_position((x,y)).fx(vfx.fadein,0.35).set_duration(CLIP_LONG_DUR)
    return clip,arr.shape[0]

def code_block_static(code_text,start,x,y,fade=0.35):
    rgba,_,_=render_full_code_panel_rgba_bounded(code_text, max_panel_w=MAX_CONTENT_W)
    clip=ImageClip(rgba).set_start(start).set_position((x,y)).fx(vfx.fadein,fade).set_duration(CLIP_LONG_DUR)
    cur_y=y+rgba.shape[0];t=start+fade
    return [clip],cur_y,t

def build_video_clip(title,desc,code):
    clips=[ColorClip((W,H),color=BG,duration=CLIP_LONG_DUR)]
    # Skip heavy watermark/logo in fast/test mode
    wm = None if getattr(args, "fast", False) else big_center_watermark(0.0, CLIP_LONG_DUR)
    if wm: clips.append(wm)
    clips.append(white_border_clip(start=0.0, duration=CLIP_LONG_DUR))
    lg = None if getattr(args, "fast", False) else logo_clip_top_right(0.0, CLIP_LONG_DUR)
    if lg: clips.append(lg)

    t=0.6
    tclip,th=title_clip(title,t,CONTENT_X,TITLE_Y)
    clips.append(tclip)
    cur_y=TITLE_Y+th+white_line_height();t+=0.6

    pclips,cur_y,t=description_line_by_line(desc,t,CONTENT_X,cur_y)
    clips+=pclips;cur_y+=PARA_GAP

    if code.strip():
        cur_y+=white_line_height()
        ex_clip,hh=example_heading_clip(t,CONTENT_X,cur_y)
        clips.append(ex_clip)
        cur_y+=hh+int(white_line_height()*0.5)
        cclips,cur_y,t=code_block_static(code,t,CONTENT_X,cur_y)
        clips+=cclips

    final_d=min(t+FINAL_DURATION_PAD+TAIL_HOLD,CLIP_LONG_DUR)
    video=CompositeVideoClip(clips,size=(W,H)).set_duration(final_d)
    if TAIL_HOLD>0: video=video.fx(vfx.freeze,t=final_d-TAIL_HOLD,freeze_duration=TAIL_HOLD)
    return video

# ------------------ Incremental naming helpers ------------------
def next_incremental_index(prefix, ext, outdir):
    pattern = os.path.join(outdir, f"{prefix}[0-9][0-9][0-9].{ext}")
    nums = []
    for p in glob.glob(pattern):
        m = re.search(rf"{re.escape(prefix)}(\d+)\.{re.escape(ext)}$", os.path.basename(p))
        if m:
            try: nums.append(int(m.group(1)))
            except: pass
    return (max(nums) + 1) if nums else 1

def save_poster_image(video_clip, outdir, prefix="concept_animation_image"):
    os.makedirs(outdir, exist_ok=True)
    idx = next_incremental_index(prefix, "jpg", outdir)
    fname = f"{prefix}{idx:03d}.jpg"
    fpath = os.path.join(outdir, fname)
    frame_t = max(0.0, video_clip.duration - 0.01)
    frame = video_clip.get_frame(frame_t)
    im = Image.fromarray(frame[:, :, :3], mode="RGB")
    im.save(fpath, format="JPEG", quality=95)
    return fpath

# ------------------ Main ------------------
def main():
    os.makedirs(args.outdir, exist_ok=True)
    items=[]
    for raw in ITEM_TEXTS:
        raw = (raw or "").strip()
        if not raw:
            continue
        title,desc,code = parse_title_desc_code(raw)
        items.append((title,desc,code))

    # Honor --max-items for quick tests
    if getattr(args, "max_items", 0) and args.max_items > 0:
        items = items[: args.max_items]

    if not items:
        print("No items in ITEM_TEXTS. Add at least one string with '>>' and '||'.")
        return

    video_start_idx = next_incremental_index("concept_animation", "mp4", args.outdir)

    for k,(title,desc,code) in enumerate(items,1):
        print(f"\nRendering {k}/{len(items)}: {title!r}")
        clip=build_video_clip(title,desc,code)
        vid_idx = video_start_idx + (k - 1)
        out_video = os.path.join(args.outdir,f"python_concept_animation{vid_idx:03d}.mp4")
        clip.write_videofile(out_video,fps=FPS,codec="libx264",audio=False,bitrate="3500k",threads=4)
        print("Saved video:", out_video)
        # Optional poster:
        # img_path = save_poster_image(clip, args.outdir, prefix="concept_animation_image")
        # print("Saved image:", img_path)

    print("\nAll outputs saved in:", os.path.abspath(args.outdir))

if __name__ == "__main__":
    main()
