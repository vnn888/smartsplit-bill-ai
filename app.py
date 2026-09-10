import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

# Bridge Streamlit Cloud secrets into env vars (same pattern as the
# Customer Assistant project, for local .env vs cloud st.secrets).
import os
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

from research.bill_splitter import split_bill

st.set_page_config(page_title="SmartSplit Bill AI", page_icon="🧾", layout="centered")

st.title("🧾 SmartSplit Bill AI")
st.caption(
    "Upload foto struk belanja, sistem baca itemnya otomatis pakai AI, "
    "lalu bagi biayanya ke beberapa orang."
)

# Model choice -- see README for the full research/comparison writeup.
# Groq is the default (see reasoning in README); switch here if your own
# comparison run favors Donut instead.
MODEL_CHOICE = "groq"  # "groq" or "donut"


def run_extraction(image_path):
    if MODEL_CHOICE == "groq":
        from research.groq_vision_model import extract
    else:
        from research.donut_model import extract
    result, elapsed, _raw = extract(image_path)
    return result, elapsed


if "extracted" not in st.session_state:
    st.session_state.extracted = None
if "assignments" not in st.session_state:
    st.session_state.assignments = {}

uploaded_file = st.file_uploader("Upload foto struk", type=["png", "jpg", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Struk yang diupload", width="stretch")

    if st.button("Baca struk ini"):
        temp_path = "temp_receipt.png"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner(f"Membaca struk pakai model ({MODEL_CHOICE})..."):
            try:
                result, elapsed = run_extraction(temp_path)
                st.session_state.extracted = result
                st.session_state.assignments = {}
                st.success(f"Selesai dibaca dalam {elapsed:.1f} detik, {len(result['items'])} item ditemukan.")
            except Exception as e:
                st.error(f"Gagal membaca struk: {e}")

if st.session_state.extracted:
    result = st.session_state.extracted

    st.divider()
    st.subheader("1. Hasil Baca Struk")
    for i, item in enumerate(result["items"]):
        st.write(f"{i+1}. **{item['name']}** — qty {item['qty']} — Rp{item['total_price']:,}")
    st.write(f"**Subtotal:** Rp{result['subtotal']:,}")
    for c in result["additional_charges"]:
        st.write(f"**{c['label']}:** Rp{c['amount']:,}")
    st.write(f"**Total:** Rp{result['total']:,}")

    st.divider()
    st.subheader("2. Siapa Saja yang Ikut Split?")
    names_input = st.text_input("Nama peserta, pisahkan dengan koma", placeholder="Amerta, Budi, Citra")
    people = [n.strip() for n in names_input.split(",") if n.strip()]

    if people:
        st.divider()
        st.subheader("3. Item Ini Punya Siapa?")
        for i, item in enumerate(result["items"]):
            choice = st.selectbox(
                f"{item['name']} (Rp{item['total_price']:,})",
                options=people,
                key=f"assign_{i}",
            )
            st.session_state.assignments[i] = choice

        st.divider()
        st.subheader("4. Hasil Pembagian")
        if st.button("Hitung Split"):
            split_result = split_bill(
                result["items"], st.session_state.assignments,
                result["additional_charges"], result["total"],
            )
            check_sum = sum(p["final_total"] for p in split_result.values())
            for person, data in split_result.items():
                with st.container(border=True):
                    st.markdown(f"**{person}** — Rp{data['final_total']:,}")
                    for item in data["items"]:
                        st.caption(f"  {item['name']}: Rp{item['total_price']:,}")
                    if data["charge_share"]:
                        st.caption(f"  Bagian pajak/service: Rp{data['charge_share']:,}")

            if check_sum == result["total"]:
                st.success(f"Total tervalidasi: Rp{check_sum:,} = Rp{result['total']:,} ✓")
            else:
                st.warning(f"Selisih terdeteksi: split Rp{check_sum:,} vs total struk Rp{result['total']:,}")
