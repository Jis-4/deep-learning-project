"""
Gradio demo for the Invoice Intelligence pipeline (OCR + LayoutLMv3).

This is the entry point for the Hugging Face Space (SDK: gradio — no
Docker required, works on the free CPU Basic / ZeroGPU tiers). See
README.md for the "push this repo to a Space" instructions.

NOTE: this demo runs the *base pretrained* LayoutLMv3 weights, not a
fine-tuned checkpoint, so extracted field values are not yet reliable —
see the in-app note and the README's "Training" section for how to fix
that.
"""

import os

import gradio as gr
from PIL import Image
from transformers import LayoutLMv3Processor

try:
    import spaces  # only available when running on a Hugging Face Space
except ImportError:  # running locally, outside any Space
    class _NoOpSpaces:
        @staticmethod
        def GPU(fn=None, **_kwargs):
            return fn if fn is not None else (lambda f: f)

    spaces = _NoOpSpaces()

from src import config
from src.infer import predict_invoice
from src.model import InvoiceExtractor
from src.utils import load_checkpoint

print("Loading LayoutLMv3 processor + model (first run only, ~500MB download)...")
_processor = LayoutLMv3Processor.from_pretrained(config.MODEL_NAME, apply_ocr=False)
_model = InvoiceExtractor(num_labels=config.NUM_LABELS)

_checkpoint_path = os.environ.get("CHECKPOINT_PATH", "")
_using_checkpoint = False
if _checkpoint_path and os.path.exists(_checkpoint_path):
    _model = load_checkpoint(_model, _checkpoint_path)
    _using_checkpoint = True
_model.eval()

_DEMO_NOTE = (
    "⚠️ Running base pretrained LayoutLMv3 weights (not fine-tuned on invoices yet). "
    "The pipeline runs end-to-end, but the field values below aren't reliable until "
    "the model is trained — see the README's Training section."
)


@spaces.GPU
def extract(image: Image.Image):
    if image is None:
        return {}, "Upload an invoice image first."

    tmp_path = "/tmp/_uploaded_invoice.png"
    image.convert("RGB").save(tmp_path)

    try:
        fields, words, _predictions = predict_invoice(tmp_path, _processor, _model)
    except Exception as exc:  # noqa: BLE001 - show the error in the UI instead of crashing
        return {}, f"Extraction failed: {exc}"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    status = f"OCR found {len(words)} word(s)."
    if not _using_checkpoint:
        status += " " + _DEMO_NOTE
    return fields, status


with gr.Blocks(title="Invoice Intelligence Demo") as demo:
    gr.Markdown("# 🧾 Invoice Intelligence Demo")
    gr.Markdown(
        "OCR ([EasyOCR](https://github.com/JaidedAI/EasyOCR)) + a layout-aware "
        "transformer ([LayoutLMv3](https://huggingface.co/microsoft/layoutlmv3-base)) "
        "extract invoice number, date, vendor name, and total amount from an image."
    )
    if not _using_checkpoint:
        gr.Markdown(f"> {_DEMO_NOTE}")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Invoice image")
        with gr.Column():
            fields_output = gr.JSON(label="Extracted fields")
            status_output = gr.Textbox(label="Status", interactive=False)

    submit_btn = gr.Button("Extract fields", variant="primary")
    submit_btn.click(fn=extract, inputs=image_input, outputs=[fields_output, status_output])

    gr.Markdown(
        "Don't have an invoice handy? Run `python scripts/generate_sample_data.py` "
        "locally to get `data/sample_invoice_1.png`, then upload that — or use any "
        "invoice screenshot."
    )


if __name__ == "__main__":
    demo.launch()
