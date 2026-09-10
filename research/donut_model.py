"""
Donut (naver-clova-ix/donut-base-finetuned-cord-v2) wrapper.
OCR-free: encodes the image directly (Swin Transformer encoder) and
decodes structured text (BART decoder) -- no separate text-recognition
step. Fine-tuned specifically on CORD, a receipt-parsing dataset.

First run downloads ~800MB of model weights from Hugging Face (one-time,
cached afterwards in ~/.cache/huggingface).
"""
import re
import time

import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

MODEL_NAME = "naver-clova-ix/donut-base-finetuned-cord-v2"

_processor = None
_model = None
_device = None


def _load():
    global _processor, _model, _device
    if _model is None:
        _processor = DonutProcessor.from_pretrained(MODEL_NAME)
        _model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _model.to(_device)
        _model.eval()
    return _processor, _model, _device


def _to_number(value):
    """CORD prices come as strings like '10,000' or '10.000' -- strip
    separators and parse as an int (Rupiah has no meaningful decimals)."""
    if value is None:
        return 0
    digits = re.sub(r"[^\d]", "", str(value))
    return int(digits) if digits else 0


def _normalize(cord_json: dict) -> dict:
    """Converts Donut/CORD's raw schema into this app's common schema:
    {items: [{name, qty, unit_price, total_price}], subtotal,
     additional_charges: [{label, amount}], total}"""
    menu = cord_json.get("menu", [])
    if isinstance(menu, dict):
        menu = [menu]

    items = []
    for m in menu:
        qty = _to_number(m.get("cnt", 1)) or 1
        total_price = _to_number(m.get("price"))
        unit_price = _to_number(m.get("unitprice")) or (total_price // qty if qty else total_price)
        items.append({
            "name": m.get("nm", "Unknown item"),
            "qty": qty,
            "unit_price": unit_price,
            "total_price": total_price,
        })

    sub_total = cord_json.get("sub_total", {})
    subtotal = _to_number(sub_total.get("subtotal_price")) if isinstance(sub_total, dict) else 0

    total_block = cord_json.get("total", {})
    total = _to_number(total_block.get("total_price")) if isinstance(total_block, dict) else 0

    additional_charges = []
    if isinstance(sub_total, dict):
        for key, label in [("tax_price", "Pajak"), ("service_price", "Service Charge")]:
            if key in sub_total:
                additional_charges.append({"label": label, "amount": _to_number(sub_total[key])})

    return {
        "items": items,
        "subtotal": subtotal or sum(i["total_price"] for i in items),
        "additional_charges": additional_charges,
        "total": total or subtotal or sum(i["total_price"] for i in items),
    }


def extract(image_path: str) -> dict:
    """Runs Donut on a receipt image, returns (normalized_result, elapsed_seconds, raw_output)."""
    processor, model, device = _load()

    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    task_prompt = "<s_cord-v2>"
    decoder_input_ids = processor.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=model.decoder.config.max_position_embeddings,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )
    elapsed = time.perf_counter() - start

    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()
    raw_json = processor.token2json(sequence)

    return _normalize(raw_json), elapsed, raw_json
