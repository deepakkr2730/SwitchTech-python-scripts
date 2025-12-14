"""
SwitchTech Code Video Generator - Optimized Version
Generates 30-40 second educational videos explaining Python code snippets.
Memory-optimized for cloud environments.
"""

import os
import re
import textwrap
from dataclasses import dataclass
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    width: int = 1080
    height: int = 1920
    fps: int = 24
    duration_per_line: float = 1.2
    title_duration: float = 3.0
    outro_duration: float = 2.5
    title_font_size: int = 68
    subtitle_font_size: int = 40
    code_font_size: int = 30
    brand_font_size: int = 34
    padding: int = 50
    line_spacing: float = 1.5


@dataclass
class Theme:
    bg_start: Tuple[int, int, int] = (10, 22, 40)
    bg_end: Tuple[int, int, int] = (25, 55, 95)
    code_bg: Tuple[int, int, int, int] = (15, 25, 45, 230)
    title_color: Tuple[int, int, int] = (255, 255, 255)
    subtitle_color: Tuple[int, int, int] = (0, 212, 255)
    text_color: Tuple[int, int, int] = (220, 220, 220)
    accent: Tuple[int, int, int] = (0, 212, 255)
    brand: Tuple[int, int, int] = (255, 107, 53)
    keyword: Tuple[int, int, int] = (255, 121, 198)
    function: Tuple[int, int, int] = (80, 250, 123)
    string: Tuple[int, int, int] = (241, 250, 140)
    number: Tuple[int, int, int] = (189, 147, 249)
    comment: Tuple[int, int, int] = (98, 114, 164)
    operator: Tuple[int, int, int] = (255, 184, 108)
    builtin: Tuple[int, int, int] = (139, 233, 253)


THEMES = {
    "dark_tech": Theme(),
    "purple": Theme(
        bg_start=(20, 10, 35), bg_end=(60, 30, 100),
        code_bg=(30, 20, 50, 230), subtitle_color=(189, 147, 249),
        accent=(189, 147, 249), brand=(255, 121, 198)
    ),
    "teal": Theme(
        bg_start=(13, 59, 62), bg_end=(5, 25, 30),
        code_bg=(10, 40, 45, 230), subtitle_color=(80, 250, 123),
        accent=(80, 250, 123), brand=(255, 184, 108)
    ),
}


# ============================================================================
# SYNTAX HIGHLIGHTER
# ============================================================================

KEYWORDS = {'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await',
    'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except',
    'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda',
    'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield'}

BUILTINS = {'abs', 'all', 'any', 'bin', 'bool', 'bytes', 'callable', 'chr',
    'dict', 'dir', 'divmod', 'enumerate', 'eval', 'exec', 'filter', 'float',
    'format', 'frozenset', 'getattr', 'globals', 'hasattr', 'hash', 'help',
    'hex', 'id', 'input', 'int', 'isinstance', 'issubclass', 'iter', 'len',
    'list', 'locals', 'map', 'max', 'min', 'next', 'object', 'oct', 'open',
    'ord', 'pow', 'print', 'range', 'repr', 'reversed', 'round', 'set',
    'setattr', 'slice', 'sorted', 'str', 'sum', 'super', 'tuple', 'type', 'vars', 'zip'}


def tokenize(code: str) -> List[Tuple[str, str]]:
    tokens = []
    i = 0
    while i < len(code):
        if code[i].isspace():
            j = i
            while j < len(code) and code[j].isspace():
                j += 1
            tokens.append(('ws', code[i:j]))
            i = j
        elif code[i] == '#':
            j = code.find('\n', i)
            j = j if j != -1 else len(code)
            tokens.append(('comment', code[i:j]))
            i = j
        elif code[i] in '"\'':
            q = code[i]
            triple = code[i:i+3] in ('"""', "'''")
            if triple:
                end = code.find(code[i:i+3], i+3)
                j = end + 3 if end != -1 else len(code)
            else:
                j = i + 1
                while j < len(code) and (code[j] != q or code[j-1] == '\\'):
                    if code[j] == '\n': break
                    j += 1
                j = j + 1 if j < len(code) and code[j] == q else j
            tokens.append(('string', code[i:j]))
            i = j
        elif code[i].isdigit() or (code[i] == '.' and i+1 < len(code) and code[i+1].isdigit()):
            j = i
            while j < len(code) and (code[j].isdigit() or code[j] in '.xXoObBeE+-'):
                j += 1
            tokens.append(('number', code[i:j]))
            i = j
        elif code[i].isalpha() or code[i] == '_':
            j = i
            while j < len(code) and (code[j].isalnum() or code[j] == '_'):
                j += 1
            word = code[i:j]
            if word in KEYWORDS:
                tokens.append(('keyword', word))
            elif word in BUILTINS:
                tokens.append(('builtin', word))
            else:
                k = j
                while k < len(code) and code[k].isspace(): k += 1
                tokens.append(('function' if k < len(code) and code[k] == '(' else 'id', word))
            i = j
        elif code[i] in '+-*/%=<>!&|^~@:;,.()[]{}':
            tokens.append(('op', code[i]))
            i += 1
        else:
            tokens.append(('id', code[i]))
            i += 1
    return tokens


# ============================================================================
# FRAME GENERATOR
# ============================================================================

class FrameGen:
    def __init__(self, cfg: Config, theme: Theme):
        self.cfg = cfg
        self.theme = theme
        self.fonts = self._load_fonts()
    
    def _load_fonts(self):
        # Try common Windows and Linux font locations so the script works on both
        paths = [
            # Windows Consolas / Courier
            r"C:\\Windows\\Fonts\\consola.ttf",
            r"C:\\Windows\\Fonts\\consolab.ttf",
            r"C:\\Windows\\Fonts\\Consolas.ttf",
            r"C:\\Windows\\Fonts\\cour.ttf",
            r"C:\\Windows\\Fonts\\courbd.ttf",
            # Windows Arial
            r"C:\\Windows\\Fonts\\arialbd.ttf",
            r"C:\\Windows\\Fonts\\arial.ttf",
            # Linux common fonts (fallback)
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        ]
        fonts = {}
        bold = next((p for p in paths if os.path.exists(p) and 'Bold' in p), None)
        reg = next((p for p in paths if os.path.exists(p) and 'Bold' not in p and 'Mono' not in p), None)
        mono = next((p for p in paths if os.path.exists(p) and 'Mono' in p), None)
        
        for name, path, size in [
            ('title', bold or reg, self.cfg.title_font_size),
            ('subtitle', reg or bold, self.cfg.subtitle_font_size),
            ('code', mono or reg, self.cfg.code_font_size),
            ('brand', bold or reg, self.cfg.brand_font_size),
            ('small', reg, self.cfg.code_font_size - 4),
        ]:
            try:
                fonts[name] = ImageFont.truetype(path, size) if path else ImageFont.load_default()
            except:
                fonts[name] = ImageFont.load_default()
        return fonts
    
    def gradient_bg(self) -> Image.Image:
        img = Image.new('RGB', (self.cfg.width, self.cfg.height))
        px = img.load()
        r1, g1, b1 = self.theme.bg_start
        r2, g2, b2 = self.theme.bg_end
        for y in range(self.cfg.height):
            r = y / self.cfg.height
            for x in range(self.cfg.width):
                px[x, y] = (int(r1+(r2-r1)*r), int(g1+(g2-g1)*r), int(b1+(b2-b1)*r))
        return img
    
    def add_decor(self, img: Image.Image) -> Image.Image:
        draw = ImageDraw.Draw(img, 'RGBA')
        c = (*self.theme.accent, 100)
        s = 80
        # Corners
        draw.line([(20, 20), (20+s, 20)], fill=c, width=3)
        draw.line([(20, 20), (20, 20+s)], fill=c, width=3)
        draw.line([(self.cfg.width-20-s, 20), (self.cfg.width-20, 20)], fill=c, width=3)
        draw.line([(self.cfg.width-20, 20), (self.cfg.width-20, 20+s)], fill=c, width=3)
        draw.line([(20, self.cfg.height-20), (20+s, self.cfg.height-20)], fill=c, width=3)
        draw.line([(20, self.cfg.height-20-s), (20, self.cfg.height-20)], fill=c, width=3)
        draw.line([(self.cfg.width-20-s, self.cfg.height-20), (self.cfg.width-20, self.cfg.height-20)], fill=c, width=3)
        draw.line([(self.cfg.width-20, self.cfg.height-20-s), (self.cfg.width-20, self.cfg.height-20)], fill=c, width=3)
        return img
    
    def add_brand(self, img: Image.Image) -> Image.Image:
        draw = ImageDraw.Draw(img, 'RGBA')
        draw.rectangle([(0, 0), (self.cfg.width, 75)], fill=(*self.theme.code_bg[:3], 200))
        txt = "SwitchTech"
        bbox = draw.textbbox((0, 0), txt, font=self.fonts['brand'])
        x = (self.cfg.width - (bbox[2] - bbox[0])) // 2
        draw.text((x, 18), txt, fill=self.theme.brand, font=self.fonts['brand'])
        draw.rectangle([(0, 72), (self.cfg.width, 75)], fill=self.theme.accent)
        return img
    
    def get_color(self, t: str) -> Tuple[int, int, int]:
        return {'keyword': self.theme.keyword, 'builtin': self.theme.builtin,
                'function': self.theme.function, 'string': self.theme.string,
                'number': self.theme.number, 'comment': self.theme.comment,
                'op': self.theme.operator}.get(t, self.theme.text_color)
    
    def title_frame(self, title: str, desc: str) -> Image.Image:
        img = self.gradient_bg()
        img = self.add_decor(img)
        img = self.add_brand(img)
        draw = ImageDraw.Draw(img, 'RGBA')
        cy = self.cfg.height // 2
        
        # Title
        lines = textwrap.wrap(title, width=18)
        y = cy - 150
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=self.fonts['title'])
            x = (self.cfg.width - (bbox[2]-bbox[0])) // 2
            draw.text((x, y), line, fill=self.theme.title_color, font=self.fonts['title'])
            y += self.cfg.title_font_size + 10
        
        # Line
        draw.rectangle([((self.cfg.width-200)//2, y+20), ((self.cfg.width+200)//2, y+24)], fill=self.theme.accent)
        
        # Description
        if desc:
            lines = textwrap.wrap(desc, width=32)
            y += 60
            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=self.fonts['subtitle'])
                x = (self.cfg.width - (bbox[2]-bbox[0])) // 2
                draw.text((x, y), line, fill=self.theme.subtitle_color, font=self.fonts['subtitle'])
                y += self.cfg.subtitle_font_size + 8
        
        # Python indicator
        py = "🐍 Python"
        bbox = draw.textbbox((0, 0), py, font=self.fonts['subtitle'])
        x = (self.cfg.width - (bbox[2]-bbox[0])) // 2
        draw.text((x, self.cfg.height - 140), py, fill=self.theme.text_color, font=self.fonts['subtitle'])
        
        return img
    
    def code_frame(self, title: str, code_lines: List[str], visible: int) -> Image.Image:
        img = self.gradient_bg()
        img = self.add_decor(img)
        img = self.add_brand(img)
        draw = ImageDraw.Draw(img, 'RGBA')
        
        # Small title
        t = title[:25] + "..." if len(title) > 25 else title
        bbox = draw.textbbox((0, 0), t, font=self.fonts['subtitle'])
        draw.text(((self.cfg.width-(bbox[2]-bbox[0]))//2, 95), t, fill=self.theme.subtitle_color, font=self.fonts['subtitle'])
        
        # Code box
        top, btm = 165, self.cfg.height - 90
        left, right = self.cfg.padding, self.cfg.width - self.cfg.padding
        draw.rectangle([(left, top), (right, btm)], fill=self.theme.code_bg)
        
        # Header
        draw.rectangle([(left, top), (right, top+38)], fill=(30, 35, 45, 255))
        for i, c in enumerate([(255,95,86), (255,189,46), (39,201,63)]):
            draw.ellipse([(left+18+i*22, top+13), (left+30+i*22, top+25)], fill=c)
        bbox = draw.textbbox((0, 0), "code.py", font=self.fonts['small'])
        draw.text(((self.cfg.width-(bbox[2]-bbox[0]))//2, top+10), "code.py", fill=(150,150,150), font=self.fonts['small'])
        
        # Code
        y = top + 55
        lh = int(self.cfg.code_font_size * self.cfg.line_spacing)
        for ln, line in enumerate(code_lines[:visible]):
            # Line number
            draw.text((left+12, y), f"{ln+1:3d}", fill=(100,100,120), font=self.fonts['code'])
            # Syntax
            cx = left + 65
            for typ, txt in tokenize(line):
                if cx > right - 40: break
                draw.text((cx, y), txt, fill=self.get_color(typ), font=self.fonts['code'])
                bbox = draw.textbbox((0, 0), txt, font=self.fonts['code'])
                cx += bbox[2] - bbox[0]
            y += lh
            if y > btm - 35: break
        
        # Cursor
        if visible > 0 and visible <= len(code_lines):
            cy = top + 55 + (visible-1)*lh
            lline = code_lines[visible-1] if visible <= len(code_lines) else ""
            bbox = draw.textbbox((0, 0), lline, font=self.fonts['code'])
            cx = left + 65 + bbox[2] - bbox[0] + 4
            if cx < right - 15:
                draw.rectangle([(cx, cy), (cx+10, cy+self.cfg.code_font_size)], fill=self.theme.accent)
        
        # Progress
        prog = visible / len(code_lines) if code_lines else 0
        pw = int((right-left-30) * prog)
        py = btm - 12
        draw.rectangle([(left+15, py), (right-15, py+3)], fill=(50,55,65))
        if pw > 0:
            draw.rectangle([(left+15, py), (left+15+pw, py+3)], fill=self.theme.accent)
        
        return img
    
    def outro_frame(self, title: str) -> Image.Image:
        img = self.gradient_bg()
        img = self.add_decor(img)
        img = self.add_brand(img)
        draw = ImageDraw.Draw(img, 'RGBA')
        cy = self.cfg.height // 2
        
        # Check
        draw.text(((self.cfg.width-60)//2, cy-180), "✓", fill=self.theme.accent, font=self.fonts['title'])
        
        # Complete
        txt = "Code Complete!"
        bbox = draw.textbbox((0, 0), txt, font=self.fonts['title'])
        draw.text(((self.cfg.width-(bbox[2]-bbox[0]))//2, cy-70), txt, fill=self.theme.title_color, font=self.fonts['title'])
        
        # Topic
        t = title[:28] + "..." if len(title) > 28 else title
        bbox = draw.textbbox((0, 0), t, font=self.fonts['subtitle'])
        draw.text(((self.cfg.width-(bbox[2]-bbox[0]))//2, cy+30), t, fill=self.theme.subtitle_color, font=self.fonts['subtitle'])
        
        # CTA
        cta = ["Follow @SwitchTech", "for more Python tips!", "", "🔗 Link in Bio"]
        y = cy + 130
        for i, line in enumerate(cta):
            if line:
                bbox = draw.textbbox((0, 0), line, font=self.fonts['subtitle'])
                c = self.theme.brand if i == 0 else self.theme.text_color
                draw.text(((self.cfg.width-(bbox[2]-bbox[0]))//2, y), line, fill=c, font=self.fonts['subtitle'])
            y += 48
        
        return img


# ============================================================================
# VIDEO GENERATOR
# ============================================================================

def parse_input(text: str):
    text = text.strip()
    if '>>' in text:
        title, rest = text.split('>>', 1)
        title = title.strip()
    else:
        title, rest = "Python Code", text
    if '||' in rest:
        desc, code = rest.split('||', 1)
        desc, code = desc.strip(), code.strip()
    else:
        desc, code = "", rest.strip()
    lines = [l for l in code.split('\n') if l.strip()]
    return title, desc, lines


def generate_video(input_text: str, output_path: str, theme_name: str = "dark_tech"):
    title, desc, code_lines = parse_input(input_text)
    print(f"📝 Generating: {title}")
    print(f"   Lines: {len(code_lines)}")
    
    cfg = Config()
    theme = THEMES.get(theme_name, THEMES["dark_tech"])
    gen = FrameGen(cfg, theme)
    
    # Calculate frames
    title_frames = int(cfg.title_duration * cfg.fps)
    frames_per_line = int(cfg.duration_per_line * cfg.fps)
    hold_frames = int(0.8 * cfg.fps)
    outro_frames = int(cfg.outro_duration * cfg.fps)
    
    total_frames = title_frames + len(code_lines) * frames_per_line + hold_frames + outro_frames
    duration = total_frames / cfg.fps
    print(f"   Duration: {duration:.1f}s")
    
    # Try moviepy
    try:
        from moviepy.editor import ImageSequenceClip
    except ImportError:
        from moviepy import ImageSequenceClip
    
    # Stream frames directly to a video writer to avoid holding all frames in memory
    print("🎬 Generating and streaming frames...")

    try:
        import imageio
        have_imageio = True
    except Exception:
        have_imageio = False

    def _to_rgb_arr(img):
        arr = np.array(img)
        # If image has alpha channel, drop it
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        # Ensure uint8
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        return arr

    if have_imageio:
        # Use imageio (ffmpeg) to stream frames to disk incrementally
        print("🎥 Writing video via imageio (streaming)")
        writer = imageio.get_writer(output_path, fps=cfg.fps, codec='libx264')

        # Title frames
        tf = gen.title_frame(title, desc)
        tf_arr = _to_rgb_arr(tf)
        for _ in range(title_frames):
            writer.append_data(tf_arr)
        del tf

        # Code reveal
        for v in range(1, len(code_lines) + 1):
            cf = gen.code_frame(title, code_lines, v)
            cf_arr = _to_rgb_arr(cf)
            for _ in range(frames_per_line):
                writer.append_data(cf_arr)
            del cf

        # Hold
        hf = gen.code_frame(title, code_lines, len(code_lines))
        hf_arr = _to_rgb_arr(hf)
        for _ in range(hold_frames):
            writer.append_data(hf_arr)
        del hf

        # Outro
        of = gen.outro_frame(title)
        of_arr = _to_rgb_arr(of)
        for _ in range(outro_frames):
            writer.append_data(of_arr)
        del of

        writer.close()
        print(f"✅ Saved: {output_path}")
        return output_path
    else:
        # Fallback: collect frames in memory and use moviepy (may use lots of RAM)
        print("⚠️ imageio not available — falling back to in-memory write (may be large)")
        frames = []
        tf = gen.title_frame(title, desc)
        tf_arr = np.array(tf)
        for _ in range(title_frames):
            frames.append(tf_arr)
        del tf

        for v in range(1, len(code_lines) + 1):
            cf = gen.code_frame(title, code_lines, v)
            cf_arr = np.array(cf)
            for _ in range(frames_per_line):
                frames.append(cf_arr)
            del cf

        hf = gen.code_frame(title, code_lines, len(code_lines))
        hf_arr = np.array(hf)
        for _ in range(hold_frames):
            frames.append(hf_arr)
        del hf

        of = gen.outro_frame(title)
        of_arr = np.array(of)
        for _ in range(outro_frames):
            frames.append(of_arr)
        del of

        print(f"   Frames: {len(frames)}")
        print("🎥 Creating video (moviepy, in-memory)...")
        clip = ImageSequenceClip(frames, fps=cfg.fps)
        clip.write_videofile(output_path, fps=cfg.fps, codec='libx264', preset='ultrafast',
                             threads=2, logger=None)
        print(f"✅ Saved: {output_path}")
        return output_path


# ============================================================================
# MAIN
# ============================================================================

INPUT_TEXTS = [
"""
GCD & LCM >>
Find GCD and LCM of a list of numbers ||
import math
nums = list(map(int, input().split()))
gcd_val = nums[0]
for n in nums[1:]:
    gcd_val = math.gcd(gcd_val, n)
lcm_val = nums[0]
for n in nums[1:]:
    lcm_val = lcm_val * n // math.gcd(lcm_val, n)
print("GCD:", gcd_val)
print("LCM:", lcm_val)
""",
"""
Order-Preserving Unique >>
Remove duplicates while preserving order ||
nums = list(map(int, input().split()))
seen = set()
result = []
for n in nums:
    if n not in seen:
        seen.add(n)
        result.append(n)
print("Result:", result)
""",
"""
Second Largest >>
Find second largest without sorting ||
nums = list(map(int, input().split()))
first = second = None
for n in nums:
    if first is None or n > first:
        second = first
        first = n
    elif n != first:
        if second is None or n > second:
            second = n
print("Second largest:", second)
""",
"""
Balanced Parentheses >>
Check balanced brackets using stack ||
expr = input("Expression: ")
stack = []
pairs = {')':'(', ']':'[', '}':'{'}
balanced = True
for ch in expr:
    if ch in "([{":
        stack.append(ch)
    elif ch in ")]}":
        if not stack or stack[-1] != pairs[ch]:
            balanced = False
            break
        stack.pop()
print("Balanced" if balanced and not stack else "Not Balanced")
""",
"""
Word Frequency >>
Count word frequency in descending order ||
text = input("Sentence: ")
freq = {}
for w in text.lower().split():
    freq[w] = freq.get(w, 0) + 1
sorted_freq = sorted(freq.items(), key=lambda x: -x[1])
for word, count in sorted_freq:
    print(f"{word}: {count}")
""",
"""
Nested List Sum >>
Recursively sum all numbers in nested list ||
def nested_sum(lst):
    total = 0
    for item in lst:
        if isinstance(item, list):
            total += nested_sum(item)
        else:
            total += item
    return total
data = [1, [2, 3], [4, [5, 6]]]
print("Sum:", nested_sum(data))
"""
]


if __name__ == "__main__":
    import sys
    # Create a local outputs folder next to the script (works on Windows and Linux)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    themes = ["dark_tech", "purple", "teal"]
    
    # Generate all videos
    for i, inp in enumerate(INPUT_TEXTS):
        title, _, _ = parse_input(inp)
        safe = re.sub(r'[^\w\s-]', '', title)[:25].strip().replace(' ', '_')
        out = os.path.join(outputs_dir, f"{i+1:02d}_{safe}.mp4")
        theme = themes[i % len(themes)]
        
        try:
            generate_video(inp, out, theme)
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*50)
    print("✅ All videos generated!")
"""
SwitchTech Code Video Generator - Optimized Version
Generates 30-40 second educational videos explaining Python code snippets.
Memory-optimized for cloud environments.
"""

import os
import re
import textwrap
from dataclasses import dataclass
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    width: int = 1080
    height: int = 1920
    fps: int = 24
    duration_per_line: float = 1.2
    title_duration: float = 3.0
    outro_duration: float = 2.5
    title_font_size: int = 68
    subtitle_font_size: int = 40
    code_font_size: int = 30
    brand_font_size: int = 34
    padding: int = 50
    line_spacing: float = 1.5


@dataclass
class Theme:
    bg_start: Tuple[int, int, int] = (10, 22, 40)
    bg_end: Tuple[int, int, int] = (25, 55, 95)
    code_bg: Tuple[int, int, int, int] = (15, 25, 45, 230)
    title_color: Tuple[int, int, int] = (255, 255, 255)
    subtitle_color: Tuple[int, int, int] = (0, 212, 255)
    text_color: Tuple[int, int, int] = (220, 220, 220)
    accent: Tuple[int, int, int] = (0, 212, 255)
    brand: Tuple[int, int, int] = (255, 107, 53)
    keyword: Tuple[int, int, int] = (255, 121, 198)
    function: Tuple[int, int, int] = (80, 250, 123)
    string: Tuple[int, int, int] = (241, 250, 140)
    number: Tuple[int, int, int] = (189, 147, 249)
    comment: Tuple[int, int, int] = (98, 114, 164)
    operator: Tuple[int, int, int] = (255, 184, 108)
    builtin: Tuple[int, int, int] = (139, 233, 253)


THEMES = {
    "dark_tech": Theme(),
    "purple": Theme(
        bg_start=(20, 10, 35), bg_end=(60, 30, 100),
        code_bg=(30, 20, 50, 230), subtitle_color=(189, 147, 249),
        accent=(189, 147, 249), brand=(255, 121, 198)
    ),
    "teal": Theme(
        bg_start=(13, 59, 62), bg_end=(5, 25, 30),
        code_bg=(10, 40, 45, 230), subtitle_color=(80, 250, 123),
        accent=(80, 250, 123), brand=(255, 184, 108)
    ),
}


# ============================================================================
# SYNTAX HIGHLIGHTER
# ============================================================================

KEYWORDS = {'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await',
    'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except',
    'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda',
    'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield'}

BUILTINS = {'abs', 'all', 'any', 'bin', 'bool', 'bytes', 'callable', 'chr',
    'dict', 'dir', 'divmod', 'enumerate', 'eval', 'exec', 'filter', 'float',
    'format', 'frozenset', 'getattr', 'globals', 'hasattr', 'hash', 'help',
    'hex', 'id', 'input', 'int', 'isinstance', 'issubclass', 'iter', 'len',
    'list', 'locals', 'map', 'max', 'min', 'next', 'object', 'oct', 'open',
    'ord', 'pow', 'print', 'range', 'repr', 'reversed', 'round', 'set',
    'setattr', 'slice', 'sorted', 'str', 'sum', 'super', 'tuple', 'type', 'vars', 'zip'}


def tokenize(code: str) -> List[Tuple[str, str]]:
    tokens = []
    i = 0
    while i < len(code):
        if code[i].isspace():
            j = i
            while j < len(code) and code[j].isspace():
                j += 1
            tokens.append(('ws', code[i:j]))
            i = j
        elif code[i] == '#':
            j = code.find('\n', i)
            j = j if j != -1 else len(code)
            tokens.append(('comment', code[i:j]))
            i = j
        elif code[i] in '"\'':
            q = code[i]
            triple = code[i:i+3] in ('"""', "'''")
            if triple:
                end = code.find(code[i:i+3], i+3)
                j = end + 3 if end != -1 else len(code)
            else:
                j = i + 1
                while j < len(code) and (code[j] != q or code[j-1] == '\\'):
                    if code[j] == '\n': break
                    j += 1
                j = j + 1 if j < len(code) and code[j] == q else j
            tokens.append(('string', code[i:j]))
            i = j
        elif code[i].isdigit() or (code[i] == '.' and i+1 < len(code) and code[i+1].isdigit()):
            j = i
            while j < len(code) and (code[j].isdigit() or code[j] in '.xXoObBeE+-'):
                j += 1
            tokens.append(('number', code[i:j]))
            i = j
        elif code[i].isalpha() or code[i] == '_':
            j = i
            while j < len(code) and (code[j].isalnum() or code[j] == '_'):
                j += 1
            word = code[i:j]
            if word in KEYWORDS:
                tokens.append(('keyword', word))
            elif word in BUILTINS:
                tokens.append(('builtin', word))
            else:
                k = j
                while k < len(code) and code[k].isspace(): k += 1
                tokens.append(('function' if k < len(code) and code[k] == '(' else 'id', word))
            i = j
        elif code[i] in '+-*/%=<>!&|^~@:;,.()[]{}':
            tokens.append(('op', code[i]))
            i += 1
        else:
            tokens.append(('id', code[i]))
            i += 1
    return tokens


# ============================================================================
# FRAME GENERATOR
# ============================================================================

class FrameGen:
    def __init__(self, cfg: Config, theme: Theme):
        self.cfg = cfg
        self.theme = theme
        self.fonts = self._load_fonts()
    
    def _load_fonts(self):
        # Try common Windows and Linux font locations so the script works on both
        paths = [
            # Windows Consolas / Courier
            r"C:\\Windows\\Fonts\\consola.ttf",
            r"C:\\Windows\\Fonts\\consolab.ttf",
            r"C:\\Windows\\Fonts\\Consolas.ttf",
            r"C:\\Windows\\Fonts\\cour.ttf",
            r"C:\\Windows\\Fonts\\courbd.ttf",
            # Windows Arial
            r"C:\\Windows\\Fonts\\arialbd.ttf",
            r"C:\\Windows\\Fonts\\arial.ttf",
            # Linux common fonts (fallback)
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        ]
        fonts = {}
        bold = next((p for p in paths if os.path.exists(p) and 'Bold' in p), None)
        reg = next((p for p in paths if os.path.exists(p) and 'Bold' not in p and 'Mono' not in p), None)
        mono = next((p for p in paths if os.path.exists(p) and 'Mono' in p), None)
        
        for name, path, size in [
            ('title', bold or reg, self.cfg.title_font_size),
            ('subtitle', reg or bold, self.cfg.subtitle_font_size),
            ('code', mono or reg, self.cfg.code_font_size),
            ('brand', bold or reg, self.cfg.brand_font_size),
            ('small', reg, self.cfg.code_font_size - 4),
        ]:
            try:
                fonts[name] = ImageFont.truetype(path, size) if path else ImageFont.load_default()
            except:
                fonts[name] = ImageFont.load_default()
        return fonts
    
    def gradient_bg(self) -> Image.Image:
        img = Image.new('RGB', (self.cfg.width, self.cfg.height))
        px = img.load()
        r1, g1, b1 = self.theme.bg_start
        r2, g2, b2 = self.theme.bg_end
        for y in range(self.cfg.height):
            r = y / self.cfg.height
            for x in range(self.cfg.width):
                px[x, y] = (int(r1+(r2-r1)*r), int(g1+(g2-g1)*r), int(b1+(b2-b1)*r))
        return img
    
    def add_decor(self, img: Image.Image) -> Image.Image:
        draw = ImageDraw.Draw(img, 'RGBA')
        c = (*self.theme.accent, 100)
        s = 80
        # Corners
        draw.line([(20, 20), (20+s, 20)], fill=c, width=3)
        draw.line([(20, 20), (20, 20+s)], fill=c, width=3)
        draw.line([(self.cfg.width-20-s, 20), (self.cfg.width-20, 20)], fill=c, width=3)
        draw.line([(self.cfg.width-20, 20), (self.cfg.width-20, 20+s)], fill=c, width=3)
        draw.line([(20, self.cfg.height-20), (20+s, self.cfg.height-20)], fill=c, width=3)
        draw.line([(20, self.cfg.height-20-s), (20, self.cfg.height-20)], fill=c, width=3)
        draw.line([(self.cfg.width-20-s, self.cfg.height-20), (self.cfg.width-20, self.cfg.height-20)], fill=c, width=3)
        draw.line([(self.cfg.width-20, self.cfg.height-20-s), (self.cfg.width-20, self.cfg.height-20)], fill=c, width=3)
        return img
    
    def add_brand(self, img: Image.Image) -> Image.Image:
        draw = ImageDraw.Draw(img, 'RGBA')
        draw.rectangle([(0, 0), (self.cfg.width, 75)], fill=(*self.theme.code_bg[:3], 200))
        txt = "SwitchTech"
        bbox = draw.textbbox((0, 0), txt, font=self.fonts['brand'])
        x = (self.cfg.width - (bbox[2] - bbox[0])) // 2
        draw.text((x, 18), txt, fill=self.theme.brand, font=self.fonts['brand'])
        draw.rectangle([(0, 72), (self.cfg.width, 75)], fill=self.theme.accent)
        return img
    
    def get_color(self, t: str) -> Tuple[int, int, int]:
        return {'keyword': self.theme.keyword, 'builtin': self.theme.builtin,
                'function': self.theme.function, 'string': self.theme.string,
                'number': self.theme.number, 'comment': self.theme.comment,
                'op': self.theme.operator}.get(t, self.theme.text_color)
    
    def title_frame(self, title: str, desc: str) -> Image.Image:
        img = self.gradient_bg()
        img = self.add_decor(img)
        img = self.add_brand(img)
        draw = ImageDraw.Draw(img, 'RGBA')
        cy = self.cfg.height // 2
        
        # Title
        lines = textwrap.wrap(title, width=18)
        y = cy - 150
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=self.fonts['title'])
            x = (self.cfg.width - (bbox[2]-bbox[0])) // 2
            draw.text((x, y), line, fill=self.theme.title_color, font=self.fonts['title'])
            y += self.cfg.title_font_size + 10
        
        # Line
        draw.rectangle([((self.cfg.width-200)//2, y+20), ((self.cfg.width+200)//2, y+24)], fill=self.theme.accent)
        
        # Description
        if desc:
            lines = textwrap.wrap(desc, width=32)
            y += 60
            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=self.fonts['subtitle'])
                x = (self.cfg.width - (bbox[2]-bbox[0])) // 2
                draw.text((x, y), line, fill=self.theme.subtitle_color, font=self.fonts['subtitle'])
                y += self.cfg.subtitle_font_size + 8
        
        # Python indicator
        py = "🐍 Python"
        bbox = draw.textbbox((0, 0), py, font=self.fonts['subtitle'])
        x = (self.cfg.width - (bbox[2]-bbox[0])) // 2
        draw.text((x, self.cfg.height - 140), py, fill=self.theme.text_color, font=self.fonts['subtitle'])
        
        return img
    
    def code_frame(self, title: str, code_lines: List[str], visible: int) -> Image.Image:
        img = self.gradient_bg()
        img = self.add_decor(img)
        img = self.add_brand(img)
        draw = ImageDraw.Draw(img, 'RGBA')
        
        # Small title
        t = title[:25] + "..." if len(title) > 25 else title
        bbox = draw.textbbox((0, 0), t, font=self.fonts['subtitle'])
        draw.text(((self.cfg.width-(bbox[2]-bbox[0]))//2, 95), t, fill=self.theme.subtitle_color, font=self.fonts['subtitle'])
        
        # Code box
        top, btm = 165, self.cfg.height - 90
        left, right = self.cfg.padding, self.cfg.width - self.cfg.padding
        draw.rectangle([(left, top), (right, btm)], fill=self.theme.code_bg)
        
        # Header
        draw.rectangle([(left, top), (right, top+38)], fill=(30, 35, 45, 255))
        for i, c in enumerate([(255,95,86), (255,189,46), (39,201,63)]):
            draw.ellipse([(left+18+i*22, top+13), (left+30+i*22, top+25)], fill=c)
        bbox = draw.textbbox((0, 0), "code.py", font=self.fonts['small'])
        draw.text(((self.cfg.width-(bbox[2]-bbox[0]))//2, top+10), "code.py", fill=(150,150,150), font=self.fonts['small'])
        
        # Code
        y = top + 55
        lh = int(self.cfg.code_font_size * self.cfg.line_spacing)
        for ln, line in enumerate(code_lines[:visible]):
            # Line number
            draw.text((left+12, y), f"{ln+1:3d}", fill=(100,100,120), font=self.fonts['code'])
            # Syntax
            cx = left + 65
            for typ, txt in tokenize(line):
                if cx > right - 40: break
                draw.text((cx, y), txt, fill=self.get_color(typ), font=self.fonts['code'])
                bbox = draw.textbbox((0, 0), txt, font=self.fonts['code'])
                cx += bbox[2] - bbox[0]
            y += lh
            if y > btm - 35: break
        
        # Cursor
        if visible > 0 and visible <= len(code_lines):
            cy = top + 55 + (visible-1)*lh
            lline = code_lines[visible-1] if visible <= len(code_lines) else ""
            bbox = draw.textbbox((0, 0), lline, font=self.fonts['code'])
            cx = left + 65 + bbox[2] - bbox[0] + 4
            if cx < right - 15:
                draw.rectangle([(cx, cy), (cx+10, cy+self.cfg.code_font_size)], fill=self.theme.accent)
        
        # Progress
        prog = visible / len(code_lines) if code_lines else 0
        pw = int((right-left-30) * prog)
        py = btm - 12
        draw.rectangle([(left+15, py), (right-15, py+3)], fill=(50,55,65))
        if pw > 0:
            draw.rectangle([(left+15, py), (left+15+pw, py+3)], fill=self.theme.accent)
        
        return img
    
    def outro_frame(self, title: str) -> Image.Image:
        img = self.gradient_bg()
        img = self.add_decor(img)
        img = self.add_brand(img)
        draw = ImageDraw.Draw(img, 'RGBA')
        cy = self.cfg.height // 2
        
        # Check
        draw.text(((self.cfg.width-60)//2, cy-180), "✓", fill=self.theme.accent, font=self.fonts['title'])
        
        # Complete
        txt = "Code Complete!"
        bbox = draw.textbbox((0, 0), txt, font=self.fonts['title'])
        draw.text(((self.cfg.width-(bbox[2]-bbox[0]))//2, cy-70), txt, fill=self.theme.title_color, font=self.fonts['title'])
        
        # Topic
        t = title[:28] + "..." if len(title) > 28 else title
        bbox = draw.textbbox((0, 0), t, font=self.fonts['subtitle'])
        draw.text(((self.cfg.width-(bbox[2]-bbox[0]))//2, cy+30), t, fill=self.theme.subtitle_color, font=self.fonts['subtitle'])
        
        # CTA
        cta = ["Follow @SwitchTech", "for more Python tips!", "", "🔗 Link in Bio"]
        y = cy + 130
        for i, line in enumerate(cta):
            if line:
                bbox = draw.textbbox((0, 0), line, font=self.fonts['subtitle'])
                c = self.theme.brand if i == 0 else self.theme.text_color
                draw.text(((self.cfg.width-(bbox[2]-bbox[0]))//2, y), line, fill=c, font=self.fonts['subtitle'])
            y += 48
        
        return img


# ============================================================================
# VIDEO GENERATOR
# ============================================================================

def parse_input(text: str):
    text = text.strip()
    if '>>' in text:
        title, rest = text.split('>>', 1)
        title = title.strip()
    else:
        title, rest = "Python Code", text
    if '||' in rest:
        desc, code = rest.split('||', 1)
        desc, code = desc.strip(), code.strip()
    else:
        desc, code = "", rest.strip()
    lines = [l for l in code.split('\n') if l.strip()]
    return title, desc, lines


def generate_video(input_text: str, output_path: str, theme_name: str = "dark_tech"):
    title, desc, code_lines = parse_input(input_text)
    print(f"📝 Generating: {title}")
    print(f"   Lines: {len(code_lines)}")
    
    cfg = Config()
    theme = THEMES.get(theme_name, THEMES["dark_tech"])
    gen = FrameGen(cfg, theme)
    
    # Calculate frames
    title_frames = int(cfg.title_duration * cfg.fps)
    frames_per_line = int(cfg.duration_per_line * cfg.fps)
    hold_frames = int(0.8 * cfg.fps)
    outro_frames = int(cfg.outro_duration * cfg.fps)
    
    total_frames = title_frames + len(code_lines) * frames_per_line + hold_frames + outro_frames
    duration = total_frames / cfg.fps
    print(f"   Duration: {duration:.1f}s")
    
    # Try moviepy
    try:
        from moviepy.editor import ImageSequenceClip
    except ImportError:
        from moviepy import ImageSequenceClip
    
    # Stream frames directly to a video writer to avoid holding all frames in memory
    print("🎬 Generating and streaming frames...")

    try:
        import imageio
        have_imageio = True
    except Exception:
        have_imageio = False

    def _to_rgb_arr(img):
        arr = np.array(img)
        # If image has alpha channel, drop it
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        # Ensure uint8
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        return arr

    if have_imageio:
        # Use imageio (ffmpeg) to stream frames to disk incrementally
        print("🎥 Writing video via imageio (streaming)")
        writer = imageio.get_writer(output_path, fps=cfg.fps, codec='libx264')

        # Title frames
        tf = gen.title_frame(title, desc)
        tf_arr = _to_rgb_arr(tf)
        for _ in range(title_frames):
            writer.append_data(tf_arr)
        del tf

        # Code reveal
        for v in range(1, len(code_lines) + 1):
            cf = gen.code_frame(title, code_lines, v)
            cf_arr = _to_rgb_arr(cf)
            for _ in range(frames_per_line):
                writer.append_data(cf_arr)
            del cf

        # Hold
        hf = gen.code_frame(title, code_lines, len(code_lines))
        hf_arr = _to_rgb_arr(hf)
        for _ in range(hold_frames):
            writer.append_data(hf_arr)
        del hf

        # Outro
        of = gen.outro_frame(title)
        of_arr = _to_rgb_arr(of)
        for _ in range(outro_frames):
            writer.append_data(of_arr)
        del of

        writer.close()
        print(f"✅ Saved: {output_path}")
        return output_path
    else:
        # Fallback: collect frames in memory and use moviepy (may use lots of RAM)
        print("⚠️ imageio not available — falling back to in-memory write (may be large)")
        frames = []
        tf = gen.title_frame(title, desc)
        tf_arr = np.array(tf)
        for _ in range(title_frames):
            frames.append(tf_arr)
        del tf

        for v in range(1, len(code_lines) + 1):
            cf = gen.code_frame(title, code_lines, v)
            cf_arr = np.array(cf)
            for _ in range(frames_per_line):
                frames.append(cf_arr)
            del cf

        hf = gen.code_frame(title, code_lines, len(code_lines))
        hf_arr = np.array(hf)
        for _ in range(hold_frames):
            frames.append(hf_arr)
        del hf

        of = gen.outro_frame(title)
        of_arr = np.array(of)
        for _ in range(outro_frames):
            frames.append(of_arr)
        del of

        print(f"   Frames: {len(frames)}")
        print("🎥 Creating video (moviepy, in-memory)...")
        clip = ImageSequenceClip(frames, fps=cfg.fps)
        clip.write_videofile(output_path, fps=cfg.fps, codec='libx264', preset='ultrafast',
                             threads=2, logger=None)
        print(f"✅ Saved: {output_path}")
        return output_path


# ============================================================================
# MAIN
# ============================================================================

INPUT_TEXTS = [
"""
GCD & LCM >>
Find GCD and LCM of a list of numbers ||
import math
nums = list(map(int, input().split()))
gcd_val = nums[0]
for n in nums[1:]:
    gcd_val = math.gcd(gcd_val, n)
lcm_val = nums[0]
for n in nums[1:]:
    lcm_val = lcm_val * n // math.gcd(lcm_val, n)
print("GCD:", gcd_val)
print("LCM:", lcm_val)
""",
"""
Order-Preserving Unique >>
Remove duplicates while preserving order ||
nums = list(map(int, input().split()))
seen = set()
result = []
for n in nums:
    if n not in seen:
        seen.add(n)
        result.append(n)
print("Result:", result)
""",
"""
Second Largest >>
Find second largest without sorting ||
nums = list(map(int, input().split()))
first = second = None
for n in nums:
    if first is None or n > first:
        second = first
        first = n
    elif n != first:
        if second is None or n > second:
            second = n
print("Second largest:", second)
""",
"""
Balanced Parentheses >>
Check balanced brackets using stack ||
expr = input("Expression: ")
stack = []
pairs = {')':'(', ']':'[', '}':'{'}
balanced = True
for ch in expr:
    if ch in "([{":
        stack.append(ch)
    elif ch in ")]}":
        if not stack or stack[-1] != pairs[ch]:
            balanced = False
            break
        stack.pop()
print("Balanced" if balanced and not stack else "Not Balanced")
""",
"""
Word Frequency >>
Count word frequency in descending order ||
text = input("Sentence: ")
freq = {}
for w in text.lower().split():
    freq[w] = freq.get(w, 0) + 1
sorted_freq = sorted(freq.items(), key=lambda x: -x[1])
for word, count in sorted_freq:
    print(f"{word}: {count}")
""",
"""
Nested List Sum >>
Recursively sum all numbers in nested list ||
def nested_sum(lst):
    total = 0
    for item in lst:
        if isinstance(item, list):
            total += nested_sum(item)
        else:
            total += item
    return total
data = [1, [2, 3], [4, [5, 6]]]
print("Sum:", nested_sum(data))
"""
]


if __name__ == "__main__":
    import sys
    # Create a local outputs folder next to the script (works on Windows and Linux)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    themes = ["dark_tech", "purple", "teal"]
    
    # Generate all videos
    for i, inp in enumerate(INPUT_TEXTS):
        title, _, _ = parse_input(inp)
        safe = re.sub(r'[^\w\s-]', '', title)[:25].strip().replace(' ', '_')
        out = os.path.join(outputs_dir, f"{i+1:02d}_{safe}.mp4")
        theme = themes[i % len(themes)]
        
        try:
            generate_video(inp, out, theme)
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*50)
    print("✅ All videos generated!")
    print("="*50)