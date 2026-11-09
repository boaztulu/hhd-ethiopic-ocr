"""Gradio web demo for the HHD-Ethiopic recognizer.

Run on the cluster:
    python -m app.web --port 7860 --host 0.0.0.0 [--share]

The user picks a checkpoint (any best.pt under checkpoints/), either uploads a
line image OR picks one of the bundled HHD-Ethiopic test samples, and gets the
recognised text + character error rate against the ground truth (if known).

The default checkpoint is the strongest seed of the aspect-preserved variant.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

import editdistance
import gradio as gr
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.infer import HHDRecognizer  # noqa: E402

DEFAULT_CKPT = "checkpoints/abl_asp_48x512_s2/best.pt"


# ----------------- ckpt + sample discovery -----------------

PRETTY_NAMES = {
    # key: tuple(label_for_dropdown, short_description_shown_in_info_card)
    "abl_asp_48x512_s2": ("★ Best model, wide input · 21.7 CER on hard split",
                          "CRNN VGG, 8.86M params, aspect-preserved 48×512 input, medium augmentation. Trained 80 epochs, seed 2."),
    "abl_asp_48x512_s1": ("Wide input, seed 1 · 21.9 CER on hard split",
                          "CRNN VGG, 8.86M params, aspect-preserved 48×512 input, medium augmentation. Trained 80 epochs, seed 1."),
    "abl_asp_48x512_s3": ("Wide input, seed 3 · 22.0 CER on hard split",
                          "CRNN VGG, 8.86M params, aspect-preserved 48×512 input, medium augmentation. Trained 80 epochs, seed 3."),
    "crnn_vgg_s1":       ("48×368 stretch, seed 1 · 22.8 CER on hard split",
                          "CRNN VGG with a 48×368 stretch (the published input size), light augmentation, 80 epochs, seed 1."),
    "crnn_vgg_s2":       ("48×368 stretch, seed 2 · 22.8 CER on hard split",
                          "CRNN VGG with a 48×368 stretch (the published input size), light augmentation, 80 epochs, seed 2."),
    "crnn_vgg_s3":       ("48×368 stretch, seed 3 · 23.1 CER on hard split",
                          "CRNN VGG with a 48×368 stretch (the published input size), light augmentation, 80 epochs, seed 3."),
    "abl_aug_heavy_s1":  ("Heavy augmentation ablation · 22.5 CER",
                          "CRNN VGG with a 48×368 stretch and the heaviest augmentation level."),
    "abl_aug_none_s1":   ("No augmentation ablation, run A · 24.0 CER",
                          "CRNN VGG with a 48×368 stretch and no augmentation."),
    "abl_pub_48x368_s1": ("No augmentation ablation, run B · 24.2 CER",
                          "Same settings and seed as run A; the two runs differ only through non-deterministic GPU kernels."),
    "diff_ocr_s1":       ("Diffusion recognizer · 24.6 CER",
                          "DiffOCR: mask-based discrete diffusion. 10M params, 12 iterative refinement steps."),
    "svtr_small_s1":     ("SVTR Small transformer · 25.2 CER",
                          "Single visual transformer encoder with CTC head. 3.91M params."),
    "crnn_convnext_s1":  ("CRNN ConvNeXt (16 CTC frames) · 43.1 CER",
                          "ImageNet-pretrained ConvNeXt-Tiny encoder. Its stride-32 output leaves 16 CTC time steps, so lines longer than 16 characters cannot be transcribed."),
    "parseq_small_s1":   ("PARSeq Small (stopped after 5 epochs) · 56.0 CER",
                          "Left-to-right autoregressive decoder. The run stopped after 5 of 80 epochs, so this is not a converged result."),
}


def _label_for(run: str) -> str:
    item = PRETTY_NAMES.get(run, (run, ""))
    return item[0] if isinstance(item, tuple) else item


def _info_for(run: str) -> str:
    item = PRETTY_NAMES.get(run, ("", ""))
    return item[1] if isinstance(item, tuple) else ""


def list_checkpoints() -> dict[str, Path]:
    """Map display label -> checkpoint path. Best model first."""
    # Preferred order matches PRETTY_NAMES; best model surfaces at the top.
    order = list(PRETTY_NAMES.keys())
    out: dict[str, Path] = {}
    found = {p.parent.name: p for p in (ROOT / "checkpoints").glob("*/best.pt")}
    for run in order:
        if run in found:
            out[_label_for(run)] = found[run]
    # Anything we didn't enumerate (future runs) tacked on
    for run, p in sorted(found.items()):
        if run not in order:
            out[_label_for(run)] = p
    return out


def _info_for_path(path: Path) -> str:
    run = path.parent.name
    return _info_for(run)


def _read_pairs_csv(path: Path) -> dict[str, str]:
    """HHD-Ethiopic image-text-pairs CSV has no header; read into {filename: text}."""
    pairs: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                fname = row[0].strip()
                gt = row[1].rstrip(" ").rstrip()
                pairs[fname] = gt
    return pairs


@lru_cache(maxsize=4)
def sample_index() -> dict[str, dict[str, str]]:
    """Build {pretty_label: {'path': abs path, 'gt': ground truth}} for first
    24 samples from Test-I and 24 from Test-II."""
    out: dict[str, dict[str, str]] = {}
    for split, csv_name, img_dir, label in [
        ("test_I",  "test_rand/image_text_pairs_test_rand.csv",  "test_rand/image_rand", "Test-I (IID)"),
        ("test_II", "test_18th/image_text_pairs_test_18th.csv", "test_18th/image_18th", "Test-II (OOD 18th c.)"),
    ]:
        csv_p = ROOT / "HHD-Ethiopic-data" / "test" / csv_name
        if not csv_p.exists():
            continue
        pairs = _read_pairs_csv(csv_p)
        items = sorted(pairs.items())[:24]
        for i, (fname, gt) in enumerate(items):
            img_p = ROOT / "HHD-Ethiopic-data" / "test" / img_dir / fname
            if not img_p.exists():
                continue
            key = f"{label}  #{i:02d}   {gt[:30]}{'…' if len(gt)>30 else ''}"
            out[key] = {"path": str(img_p), "gt": gt}
    return out


# ----------------- recogniser cache -----------------

@lru_cache(maxsize=4)
def get_recognizer(ckpt_label: str, device: str) -> HHDRecognizer:
    ckpts = list_checkpoints()
    ckpt_path = ckpts.get(ckpt_label) or Path(ckpt_label)
    return HHDRecognizer(ckpt_path, device=device)


# ----------------- helpers -----------------

def _cer_pct(pred: str, gt: str) -> float:
    if not gt:
        return float("nan")
    return 100.0 * editdistance.eval(pred, gt) / max(len(gt), 1)


def _highlight_diff(pred: str, gt: str) -> str:
    """Return an HTML highlighted diff: green chars match, red wrong."""
    out = []
    for i, ch in enumerate(pred):
        g = gt[i] if i < len(gt) else None
        if g is None:
            out.append(f"<span class='hhd-miss'>{ch}</span>")
        elif ch == g:
            out.append(f"<span class='hhd-match'>{ch}</span>")
        else:
            out.append(f"<span class='hhd-miss'>{ch}</span>")
    if len(gt) > len(pred):
        for ch in gt[len(pred) :]:
            out.append(f"<span class='hhd-extra'>{ch}</span>")
    return "".join(out) or "<span class='hhd-note'>(empty)</span>"


# ----------------- core inference handler -----------------

def recognise(ckpt_label: str, sample_label: Optional[str], uploaded_image: Optional[Image.Image],
              device: str = "cpu") -> tuple[Image.Image, str, str, str]:
    """Returns (image_used, prediction_text, html_diff_or_status, model_info_md)."""
    rec = get_recognizer(ckpt_label, device)
    info = rec.info()

    image: Optional[Image.Image] = None
    gt: Optional[str] = None

    if uploaded_image is not None:
        image = uploaded_image.convert("RGB")
    elif sample_label and sample_label != "(none)":
        idx = sample_index()
        entry = idx.get(sample_label)
        if entry:
            image = Image.open(entry["path"]).convert("RGB")
            gt = entry["gt"]
    if image is None:
        return None, "", "Upload an image or pick a sample from the dropdown.", ""

    pred = rec.predict(image)
    if gt is not None:
        cer = _cer_pct(pred, gt)
        cer_color = "#1f8a3d" if cer < 5 else ("#b8860b" if cer < 20 else "#b22a2a")
        diff_html = f"""
        <div class='hhd-result'>
          <div class='hhd-row'>
            <div class='hhd-label'>Ground truth</div>
            <div class='hhd-text'>{gt}</div>
          </div>
          <div class='hhd-row'>
            <div class='hhd-label'>Prediction</div>
            <div class='hhd-text'>{_highlight_diff(pred, gt)}</div>
          </div>
          <div class='hhd-row'>
            <div class='hhd-label'>Character Error Rate</div>
            <div class='hhd-cer' style='color:{cer_color}'>{cer:.2f}%</div>
          </div>
        </div>
        """
    else:
        diff_html = f"""
        <div class='hhd-result'>
          <div class='hhd-row'>
            <div class='hhd-label'>Prediction</div>
            <div class='hhd-text'>{pred}</div>
          </div>
          <div class='hhd-row hhd-note'>No ground truth available for this image.</div>
        </div>
        """

    info_md = (
        f"### Model details\n"
        f"**Architecture** &nbsp; `{info['model']}` &nbsp;&nbsp; **Parameters** &nbsp; {info['params_M']} M  \n"
        f"**Checkpoint** &nbsp; `{info['ckpt']}`  \n"
        f"**Input size** &nbsp; {info['input_hw'][0]} × {info['input_hw'][1]} px"
        f" (aspect preserved: {'yes' if info['preserve_aspect'] else 'no'})  \n"
        f"**Device** &nbsp; {info['device'].upper()}"
    )
    return image, pred, diff_html, info_md


# ----------------- UI -----------------

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Ethiopic:wght@400;700&family=Inter:wght@400;600;700&display=swap');

.gradio-container { max-width: 1280px !important; font-family: 'Inter', sans-serif; }

/* Header banner */
.hhd-banner {
    background: linear-gradient(135deg, #1f4e6b 0%, #2c7da0 50%, #6a994e 100%);
    color: white;
    padding: 28px 32px;
    border-radius: 16px;
    margin-bottom: 18px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    display: flex;
    align-items: center;
    gap: 24px;
    flex-wrap: wrap;
}
.hhd-banner-left { flex: 1 1 360px; }
.hhd-banner h1 { color: white; margin: 0 0 6px 0; font-size: 2.0em; font-weight: 700; }
.hhd-banner p  { color: rgba(255,255,255,0.92); margin: 0; font-size: 1.05em; line-height: 1.5; }

.hhd-banner-gif-wrap {
    flex: 0 1 460px;
    max-width: 100%;
    background: rgba(255,255,255,0.10);
    padding: 10px;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.22);
    box-shadow: 0 4px 18px rgba(0,0,0,0.18);
}
.hhd-banner-gif {
    width: 100%;
    display: block;
    border-radius: 8px;
    background: #f5f8fb;
}

.hhd-tagline {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    padding: 4px 10px;
    border-radius: 6px;
    margin-left: 8px;
    font-size: 0.9em;
}

/* Cards */
.hhd-card {
    background: #ffffff;
    border: 1px solid #e3e8ee;
    border-radius: 14px;
    padding: 20px 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.hhd-card h3 { margin-top: 0; color: #1f4e6b; font-weight: 600; }

/* Large image preview */
.hhd-image-big img {
    width: 100% !important;
    max-height: 480px !important;
    object-fit: contain !important;
    background: #f6f8fa;
    border-radius: 10px;
    padding: 8px;
    image-rendering: -webkit-optimize-contrast;
    image-rendering: crisp-edges;
}
.hhd-image-up { min-height: 180px; }

/* Prediction text */
.hhd-pred textarea {
    font-family: 'Noto Sans Ethiopic', 'Inter', sans-serif !important;
    font-size: 1.8em !important;
    line-height: 1.4 !important;
    background: #fff8e1 !important;
    border-radius: 10px !important;
    padding: 14px !important;
}

/* Result panel rows */
.hhd-result {
    background: #ffffff;
    border-left: 4px solid #2c7da0;
    border-radius: 8px;
    padding: 18px 22px;
    margin-top: 8px;
}
.hhd-row { padding: 8px 0; border-bottom: 1px solid #eef2f5; }
.hhd-row:last-child { border-bottom: none; }
.hhd-label {
    color: #5a6b7a;
    font-size: 0.78em;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
    margin-bottom: 4px;
}
.hhd-text {
    font-family: 'Noto Sans Ethiopic', monospace;
    font-size: 1.7em;
    line-height: 1.45;
    color: #1f2937;
}
.hhd-cer { font-size: 2.0em; font-weight: 700; }
.hhd-note { color: #8896a4; font-style: italic; font-size: 0.95em; }

/* Match / mismatch character spans */
.hhd-match { color: #1f8a3d; }
.hhd-miss  { background: #ffe2e2; color: #b22a2a; border-radius: 3px; padding: 0 2px; }
.hhd-extra { background: #fff3b0; color: #8a6a00; border-radius: 3px; padding: 0 2px; }

/* Big primary button */
.hhd-go button {
    background: linear-gradient(135deg, #2c7da0 0%, #1f4e6b 100%) !important;
    color: white !important;
    font-size: 1.15em !important;
    font-weight: 600 !important;
    padding: 14px 24px !important;
    border-radius: 10px !important;
    border: none !important;
    width: 100% !important;
    margin-top: 8px;
}
.hhd-go button:hover { transform: translateY(-1px); box-shadow: 0 4px 14px rgba(44,125,160,0.35); }
"""

def _banner_html() -> str:
    """Render the header banner. Embeds the auto-generated banner.gif as a
    base64 data URI so it works inside Gradio's iframe regardless of static
    file serving."""
    gif_path = ROOT / "app" / "static" / "banner.gif"
    gif_tag = ""
    if gif_path.exists():
        import base64
        data = base64.b64encode(gif_path.read_bytes()).decode("ascii")
        gif_tag = (
            "<div class='hhd-banner-gif-wrap'>"
            f"<img class='hhd-banner-gif' src='data:image/gif;base64,{data}' "
            "alt='Ethiopic manuscript lines being recognised as text' />"
            "</div>"
        )
    return f"""
<div class='hhd-banner'>
  <div class='hhd-banner-left'>
    <h1>Ethiopic Manuscript Recogniser</h1>
    <p>
      Modern handwritten OCR for historical Ethiopic manuscripts.<br>
      Our best model reaches
      <b>CER&nbsp;12.7</b> on Test&#8209;I and
      <b>CER&nbsp;21.9</b> on Test&#8209;II
      <span class='hhd-tagline'>&minus;6.8 vs previous state of the art</span>
    </p>
  </div>
  {gif_tag}
</div>
"""

HOWTO_MD = """
**Try it.** Either pick a checkpoint and a bundled sample from the dropdowns
below, or upload your own Ethiopic line image (one line, text running left to right).
Click **Recognise** to run the model.
"""


def _update_image_from_sample(sample_label: str):
    """When the user picks a sample, show the image in the big preview straight away."""
    if not sample_label or sample_label == "(none)":
        return None
    entry = sample_index().get(sample_label)
    if not entry:
        return None
    return Image.open(entry["path"]).convert("RGB")


def build_app(default_device: str = "cpu") -> gr.Blocks:
    ckpts = list_checkpoints()
    if not ckpts:
        raise RuntimeError("No checkpoints found under checkpoints/*/best.pt")
    samples = sample_index()
    sample_choices = ["(none)"] + sorted(samples.keys())

    default_label = next((k for k in ckpts if "★" in k), next(iter(ckpts)))

    with gr.Blocks(title="HHD-Ethiopic Recogniser", css=CSS, theme=gr.themes.Soft()) as demo:
        gr.HTML(_banner_html())
        gr.Markdown(HOWTO_MD)

        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                with gr.Group(elem_classes=["hhd-card"]):
                    gr.Markdown("### 1. Choose a model")
                    ckpt_dd = gr.Dropdown(
                        choices=list(ckpts.keys()),
                        value=default_label,
                        label="Checkpoint",
                        interactive=True,
                    )
                    device_dd = gr.Dropdown(
                        choices=["cpu", "cuda"],
                        value=default_device,
                        label="Device",
                    )

                with gr.Group(elem_classes=["hhd-card"]):
                    gr.Markdown("### 2. Pick an image")
                    sample_dd = gr.Dropdown(
                        choices=sample_choices,
                        value="(none)",
                        label="Bundled HHD-Ethiopic sample",
                    )
                    gr.Markdown("…or upload your own line image:")
                    img_in = gr.Image(
                        type="pil",
                        label="Upload",
                        height=180,
                        elem_classes=["hhd-image-up"],
                    )

                go = gr.Button("Recognise", variant="primary", elem_classes=["hhd-go"])

            with gr.Column(scale=2, min_width=560):
                with gr.Group(elem_classes=["hhd-card"]):
                    gr.Markdown("### Image")
                    img_view = gr.Image(
                        label="",
                        show_label=False,
                        height=480,
                        elem_classes=["hhd-image-big"],
                        interactive=False,
                    )

                with gr.Group(elem_classes=["hhd-card"]):
                    gr.Markdown("### Prediction")
                    pred_box = gr.Textbox(
                        label="Predicted Ethiopic text",
                        show_label=True,
                        elem_classes=["hhd-pred"],
                        lines=2,
                    )
                    diff_html = gr.HTML()

                with gr.Group(elem_classes=["hhd-card"]):
                    info_md = gr.Markdown()

        # Live image preview when sample is picked
        sample_dd.change(fn=_update_image_from_sample, inputs=sample_dd, outputs=img_view)
        # Allow upload to also preview immediately
        img_in.change(fn=lambda x: x, inputs=img_in, outputs=img_view)

        go.click(
            fn=recognise,
            inputs=[ckpt_dd, sample_dd, img_in, device_dd],
            outputs=[img_view, pred_box, diff_html, info_md],
        )

        gr.HTML("""
        <div style='margin-top:24px;padding:16px;border-top:1px solid #e3e8ee;color:#6c7a89;font-size:0.9em;text-align:center'>
          Dataset used: <b>HHD&#8209;Ethiopic</b> (Belay et al., ICDAR 2024),
          released under CC&nbsp;BY 4.0. <br>
          Source manuscripts from the Ethiopian National Archive and Library Agency (ENALA).
        </div>
        """)
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", default=7860, type=int)
    ap.add_argument("--share", action="store_true",
                    help="Expose via gradio.live (useful for SSH-only sessions).")
    ap.add_argument("--device", default="cpu",
                    help="cpu / cuda; default cpu so the web demo doesn't claim a GPU.")
    args = ap.parse_args()

    demo = build_app(default_device=args.device)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_api=False,            # avoid gradio_client api_info schema bug on 4.44
        show_error=True,
    )


if __name__ == "__main__":
    main()
