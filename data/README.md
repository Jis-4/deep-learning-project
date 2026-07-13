# data/

`annotations.csv` is checked in, but `sample_invoice_1.png` is generated
rather than committed as binary. Run this once after cloning:

```bash
python scripts/generate_sample_data.py
```

This deterministically writes `sample_invoice_1.png` (a synthetic invoice
image) and regenerates `annotations.csv` to match it exactly, so
`python -m pytest tests/`, `python train.py`, and the `/extract` API
endpoint all have something real to work with immediately.

To train on your own invoices instead:

1. Add your invoice images here.
2. Run OCR (`src/ocr.run_ocr`) on each to get words + boxes.
3. Label each word with one of the classes in `src/config.py`
   (`O`, `INVOICE_NUMBER`, `TOTAL_AMOUNT`, `DATE`, `VENDOR_NAME`).
4. Append a row per image to `annotations.csv` in the same format.
