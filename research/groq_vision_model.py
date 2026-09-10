"""
Groq vision wrapper (qwen/qwen3.6-27b) -- a general-purpose vision-language
model, OCR-free by architecture (reads the image holistically, no discrete
text-recognition step), accessed via API rather than run locally.
"""
import base64
import json
import os
import re
import time

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL_NAME = "qwen/qwen3.6-27b"

PROMPT = """\
Baca gambar struk belanja ini dan ekstrak datanya ke JSON PERSIS format ini, HANYA JSON tanpa spasi/newline berlebih, tanpa teks lain, TANPA markdown code block:

{"items":[{"name":"nama item","qty":1,"unit_price":10000,"total_price":10000}],"subtotal":43500,"additional_charges":[{"label":"Pajak","amount":0}],"total":43500}

Aturan:
- Semua angka adalah integer, tanpa titik/koma pemisah ribuan.
- additional_charges cuma diisi kalau ada biaya tambahan (pajak/service charge) yang eksplisit tertulis di struk, kalau tidak ada, isi list kosong [].
- Nama item boleh disingkat kalau di struk memang singkat, jangan menambah keterangan.
- JSON HARUS lengkap dan valid -- prioritaskan kelengkapan struktur JSON dibanding detail nama item yang panjang.
"""


def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in model response: {text[:200]}")
    return json.loads(match.group(0))


def extract(image_path: str) -> dict:
    """Runs Groq vision on a receipt image, returns (normalized_result, elapsed_seconds, raw_response_text)."""
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    b64_image = _encode_image(image_path)

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
            ],
        }],
        temperature=0,
        max_tokens=950,
        reasoning_effort="none",
    )
    elapsed = time.perf_counter() - start

    raw_text = response.choices[0].message.content
    finish_reason = response.choices[0].finish_reason
    if finish_reason == "length":
        raise ValueError(
            f"Response terpotong (finish_reason=length) -- struk ini butuh lebih banyak token "
            f"daripada max_tokens yang diizinkan tier akun ini. Raw (terpotong): {raw_text[:200]}"
        )
    result = _extract_json(raw_text)

    items = [{
        "name": i.get("name", "Unknown item"),
        "qty": int(i.get("qty", 1) or 1),
        "unit_price": int(i.get("unit_price", 0) or 0),
        "total_price": int(i.get("total_price", 0) or 0),
    } for i in result.get("items", [])]

    normalized = {
        "items": items,
        "subtotal": int(result.get("subtotal", 0) or sum(i["total_price"] for i in items)),
        "additional_charges": [
            {"label": c.get("label", ""), "amount": int(c.get("amount", 0) or 0)}
            for c in result.get("additional_charges", [])
        ],
        "total": int(result.get("total", 0) or 0),
    }

    return normalized, elapsed, raw_text