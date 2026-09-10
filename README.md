# SmartSplit Bill AI

Web app (Streamlit) yang membaca struk belanja pakai AI, lalu membagi
biayanya ke beberapa orang berdasarkan siapa pesan/beli apa.

## Cara Install & Menjalankan

1. `pip install -r requirements.txt`
   (baris pertama kali akan lama karena PyTorch + Transformers cukup besar)
2. Copy `.env.example` jadi `.env`, isi `GROQ_API_KEY` (gratis di
   console.groq.com/keys)
3. Jalankan riset perbandingan model dulu (lihat bagian di bawah)
4. `streamlit run app.py`

## 1. Riset & Perbandingan Model AI

Dua model yang dibandingkan untuk membaca struk (keduanya OCR-free --
tidak memakai EasyOCR/PyTesseract):

- **Donut** (`naver-clova-ix/donut-base-finetuned-cord-v2`) -- model lokal
  dari Hugging Face, arsitektur encoder-decoder (Swin Transformer + BART)
  yang membaca gambar langsung tanpa langkah text-recognition terpisah.
  Secara spesifik dilatih di dataset CORD (dataset struk belanja).
- **Groq Vision** (`qwen/qwen3.6-27b`) -- model vision-language
  general-purpose, diakses lewat API. OCR-free by design (memproses
  gambar secara holistik, tidak ada langkah OCR eksplisit terpisah).

Jalankan perbandingannya sendiri (perlu koneksi internet -- kedua model
butuh akses ke server masing-masing, tidak bisa diverifikasi dari
sandbox development ini):
```
python research/compare_models.py
```

### Hasil Pembacaan (real, dari eksperimen)

**Struk 1 -- BreadTalk (4 item, cetakan rapi):**
- **Donut**: berhasil baca 4/4 item dengan benar, total Rp43.500 (cocok persis dengan struk asli).
- **Groq Vision**: berhasil baca 4/4 item dengan benar, waktu inference ~1.3-2.3 detik, total Rp43.500 (cocok persis).

**Struk 2 -- Toko kelontong (20 item, foto agak buram):**
- **Donut**: berhasil membaca dan menampilkan seluruh 20 item, tapi total akhir Rp433.548 -- **tidak cocok** dengan total asli di struk (Rp451.190), selisih Rp17.642. Kemungkinan ada 1-2 item yang salah dibaca kuantitas/harganya, atau ada baris yang terlewat, karena Donut dilatih khusus di dataset CORD yang formatnya beda dari struk kelontong Indonesia ini.
- **Groq Vision**: berhasil membaca seluruh 20 item, total akhir Rp451.190 -- **cocok persis** dengan total asli di struk. Sempat gagal di percobaan awal karena mode "thinking" Qwen menghabiskan seluruh jatah token sebelum sempat menulis JSON (lihat catatan teknis di bawah), setelah `reasoning_effort="none"` diaktifkan, berhasil sempurna.

### Analisis Perbandingan

| Kriteria | Donut | Groq Vision |
|---|---|---|
| Kecepatan inference | Lebih lambat (perlu load model besar, inference lokal CPU) | Cepat (~1-2 detik per struk) |
| Akurasi struk sederhana | Sempurna (4/4 item, total cocok) | Sempurna (4/4 item, total cocok) |
| Akurasi struk kompleks (20 item) | **Tidak akurat** -- total meleset Rp17.642 dari struk asli | **Akurat** -- total cocok persis |
| Kebutuhan resource | Berat (download ~800MB sekali, butuh CPU/GPU lokal tiap run) | Ringan (cukup panggilan API) |
| Generalisasi ke format struk lain | Terbatas -- dilatih khusus format CORD, terbukti kurang akurat di struk kelontong Indonesia dengan 20 item | Lebih fleksibel -- model general-purpose, terbukti akurat di kedua jenis struk |
| Catatan teknis khusus | - | Model preview dengan limit ketat 1000 token output/menit dan mode "thinking" yang aktif default -- perlu `reasoning_effort="none"` dan `max_tokens` dijaga di bawah limit supaya tidak terpotong |

**Model yang dipilih untuk prototype: Groq Vision (`qwen/qwen3.6-27b`)**

Alasan: pada struk sederhana kedua model sama-sama akurat, tapi pada struk
yang lebih kompleks dan realistis (20 item, foto agak buram) Groq Vision
terbukti membaca dengan akurat sementara Donut meleset. Karena tujuan
prototype ini adalah membaca struk belanja nyata milik pengguna (yang
formatnya sangat beragam), kemampuan generalisasi Groq Vision jadi
pertimbangan utama dibanding Donut yang terikat pada satu format latihan
spesifik. Groq Vision juga jauh lebih ringan dijalankan (tidak perlu
download model besar tiap deployment).

## 2. Prototype

Struktur:
- `app.py` -- Streamlit UI (upload struk, assign item, hitung split)
- `research/donut_model.py` -- wrapper model Donut
- `research/groq_vision_model.py` -- wrapper model Groq Vision
- `research/bill_splitter.py` -- logic pembagian biaya
- `research/compare_models.py` -- script perbandingan riset

Model yang dipakai di prototype diatur lewat variabel `MODEL_CHOICE` di
`app.py` (`"groq"` atau `"donut"`) -- default `"groq"`, ganti sesuai hasil
analisis di atas.

### Logic pembagian biaya

Tiap item di-assign ke satu orang. Pajak/service charge dibagi
proporsional berdasarkan porsi belanja masing-masing orang (bukan dibagi
rata) -- orang yang belanja lebih banyak menanggung pajak lebih besar,
lebih adil. Sisa pembulatan Rupiah otomatis dialokasikan supaya total
split PERSIS sama dengan total struk (sudah diverifikasi lewat test,
lihat `research/bill_splitter.py`).

## 3. Evaluasi & Analisis Produk

**Kelemahan model pembaca struk:**
- Donut kurang akurat untuk struk di luar distribusi data latihnya (terbukti meleset Rp17.642 di struk kelontong 20 item)
- Groq Vision (model preview) punya limit ketat 1000 token output/menit -- untuk struk yang jauh lebih panjang dari 20 item, JSON hasil ekstraksi berisiko terpotong sebelum lengkap
- Kedua model kesulitan membedakan quantity vs harga kalau formatnya tidak standar (nama produk yang disingkat di kasir, seperti "PRONAS CLS C/B CH", juga bisa ambigu ditafsirkan)

**Ide improvement model:**
- Untuk Groq Vision: proses struk panjang dalam beberapa potongan (chunking) kalau terdeteksi lebih dari ~20 item, supaya tidak kena limit token
- Tambahkan validasi otomatis: kalau jumlah semua item tidak sama dengan subtotal yang diklaim, tandai sebagai "perlu dicek manual"
- Coba ensemble kedua model dan bandingkan hasilnya otomatis, kasih peringatan ke user kalau hasil keduanya beda jauh

**Kelemahan produk (fitur/bug):**
- Belum ada validasi kalau item hasil AI salah baca -- user tidak bisa edit manual nama/harga item sebelum split
- Belum ada opsi split satu item ke beberapa orang sekaligus (misal makanan yang dimakan berdua) -- saat ini satu item = satu orang saja
- Foto struk miring/gelap kemungkinan menurunkan akurasi kedua model (belum ada preprocessing gambar seperti auto-crop atau perbaikan kontras)

**Ide improvement produk:**
- Tambah fitur edit manual item (nama, qty, harga) setelah dibaca AI, sebelum lanjut ke split
- Tambah opsi "split rata ke beberapa orang" per item, bukan cuma satu orang per item
- Tambah preview/crop gambar sebelum diproses, supaya user bisa pastikan fotonya jelas dulu

## Video Demo

Rekam screen record singkat: upload salah satu struk asli, tunjukkan hasil
bacaan AI-nya, assign item ke beberapa nama, tunjukkan hasil split akhir
(dan bahwa totalnya cocok dengan total struk asli).
