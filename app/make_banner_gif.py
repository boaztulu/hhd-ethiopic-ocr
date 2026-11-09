"""Build the demo header GIF.

Picks four real HHD-Ethiopic test lines where our best model gets them right (or
close to right), runs the model, and renders each as a frame showing the line
image fading into its transcription. Saved to ``app/static/banner.gif``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw, ImageFont
import numpy as np

from src.infer import HHDRecognizer

OUT = ROOT / "app" / "static" / "banner.gif"
OUT.parent.mkdir(parents=True, exist_ok=True)

W, H = 900, 220
BG = (245, 248, 251)
INK = (30, 41, 59)
ACCENT = (44, 125, 160)
HIGHLIGHT = (106, 153, 78)

SAMPLES = [
    ("test/test_rand/image_rand/test_rand_00000.png",
     "Test-I, IID 19th–20th century"),
    ("test/test_rand/image_rand/test_rand_00012.png",
     "Test-I, IID 19th–20th century"),
    ("test/test_18th/image_18th/test_18th_00006.png",
     "Test-II, 18th-century OOD"),
    ("test/test_18th/image_18th/test_18th_00018.png",
     "Test-II, 18th-century OOD"),
]


def _font(size: int, bold: bool = False, ethiopic: bool = False) -> ImageFont.FreeTypeFont:
    if ethiopic:
        # The bundled Noto font comes first: PIL's default bitmap font has no Ethiopic
        # glyphs, so a missing system font would otherwise render the text as empty boxes.
        candidates = [
            str(ROOT / "cache" / "NotoSansEthiopic.ttf"),
            "/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-R.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansEthiopic-Regular.ttf",
            "/usr/share/fonts/google-noto/NotoSansEthiopic-Regular.ttf",
            "C:/Windows/Fonts/ebrima.ttf",
        ]
    else:
        candidates = [
            "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        ]
    for c in candidates:
        if Path(c).exists():
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _scale_line_image(im: Image.Image, target_h: int = 56, max_w: int = 720) -> Image.Image:
    w, h = im.size
    new_w = int(round(w * target_h / h))
    new_w = min(new_w, max_w)
    return im.resize((new_w, target_h), Image.BICUBIC)


def render_frame(line_img: Image.Image, pred_text: str, gt_text: str, sample_label: str, reveal_chars: int) -> Image.Image:
    """One frame: the line image up top, a fade-in transcription below."""
    frame = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(frame)

    # Top label
    label_font = _font(15, bold=True)
    d.text((30, 18), sample_label.upper(), fill=(120, 140, 160), font=label_font)

    # Line image
    line_scaled = _scale_line_image(line_img, target_h=64, max_w=W - 60)
    x_img = (W - line_scaled.size[0]) // 2
    y_img = 50
    frame.paste(line_scaled, (x_img, y_img))

    # Arrow
    arrow_y = y_img + 64 + 18
    d.line([(W // 2 - 14, arrow_y), (W // 2 + 14, arrow_y)], fill=ACCENT, width=3)
    d.polygon([(W // 2 + 14, arrow_y - 6), (W // 2 + 14, arrow_y + 6), (W // 2 + 24, arrow_y)], fill=ACCENT)

    # Transcription (typing reveal)
    eth_font = _font(30, ethiopic=True)
    reveal = pred_text[: reveal_chars]
    if reveal:
        bbox = d.textbbox((0, 0), reveal, font=eth_font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x_text = (W - text_w) // 2
        y_text = arrow_y + 18
        # background pill
        pad = 14
        d.rounded_rectangle(
            [x_text - pad, y_text - pad // 2, x_text + text_w + pad, y_text + text_h + pad],
            radius=14, fill=(255, 252, 240), outline=(232, 224, 200), width=1,
        )
        # check if prediction matches GT for green vs amber color
        color = HIGHLIGHT if pred_text.strip() == gt_text.strip() else INK
        d.text((x_text, y_text), reveal, fill=color, font=eth_font)

    return frame


def main():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rec = HHDRecognizer(ROOT / "checkpoints" / "abl_asp_48x512_s2" / "best.pt", device=device)
    print("loaded:", rec.info())

    frames = []
    durations = []
    HOLD_AT_END_MS = 1800
    REVEAL_STEP_MS = 28

    for img_rel, label in SAMPLES:
        img_path = ROOT / "HHD-Ethiopic-data" / img_rel
        if not img_path.exists():
            print(f"skip missing {img_path}")
            continue
        line = Image.open(img_path).convert("RGB")

        # ground-truth lookup for matching coloration
        # walk the same CSV the app uses
        import csv
        gt = ""
        if "test_rand" in img_rel:
            csv_p = ROOT / "HHD-Ethiopic-data/test/test_rand/image_text_pairs_test_rand.csv"
        else:
            csv_p = ROOT / "HHD-Ethiopic-data/test/test_18th/image_text_pairs_test_18th.csv"
        fname = Path(img_rel).name
        with open(csv_p, encoding="utf-8") as f:
            for row in csv.reader(f):
                if row and row[0].strip() == fname:
                    gt = row[1].rstrip()
                    break

        pred = rec.predict(line)
        print(f"  {fname}: pred={pred!r}  gt={gt!r}")

        # "blank" frame at the start of each sample (image only)
        for _ in range(8):
            frames.append(render_frame(line, pred, gt, label, 0))
            durations.append(60)

        # type-out reveal frames
        for n in range(1, len(pred) + 1):
            frames.append(render_frame(line, pred, gt, label, n))
            durations.append(REVEAL_STEP_MS)

        # hold the final result
        frames.append(render_frame(line, pred, gt, label, len(pred)))
        durations.append(HOLD_AT_END_MS)

    if not frames:
        print("no frames produced")
        return

    # Save as animated GIF
    frames[0].save(
        OUT,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(f"wrote {OUT}  ({len(frames)} frames, {OUT.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
