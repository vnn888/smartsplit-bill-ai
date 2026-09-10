"""
Runs both models against both receipt photos, times inference, and prints
results for qualitative + speed comparison (brief requirement 1d).

Run: python research/compare_models.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

RECEIPTS = [
    ("BreadTalk (simple, 4 items)", "data/receipt_1_breadtalk.png"),
    ("Grocery store (complex, 20 items)", "data/receipt_2_grocery.png"),
]


def run_model(model_name, extract_fn):
    print(f"\n{'='*70}\nMODEL: {model_name}\n{'='*70}")
    for label, path in RECEIPTS:
        print(f"\n--- {label} ---")
        try:
            result, elapsed, _raw = extract_fn(path)
            print(f"Waktu inference: {elapsed:.2f} detik")
            print(f"Jumlah item terbaca: {len(result['items'])}")
            for item in result["items"]:
                print(f"  - {item['name']}: qty={item['qty']}, harga={item['total_price']}")
            print(f"Subtotal: {result['subtotal']}")
            print(f"Biaya tambahan: {result['additional_charges']}")
            print(f"Total: {result['total']}")
        except Exception as e:
            print(f"GAGAL: {type(e).__name__}: {e}")


def main():
    try:
        from research.donut_model import extract as donut_extract
        run_model("Donut (naver-clova-ix/donut-base-finetuned-cord-v2)", donut_extract)
    except Exception as e:
        print(f"\nDonut tidak bisa dijalankan: {e}")

    try:
        from research.groq_vision_model import extract as groq_extract
        run_model("Groq Vision (qwen/qwen3.6-27b)", groq_extract)
    except Exception as e:
        print(f"\nGroq Vision tidak bisa dijalankan: {e}")


if __name__ == "__main__":
    main()
