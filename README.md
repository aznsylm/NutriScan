# NUTRISCAN v3

Aplikasi web berbasis Flask untuk scanning dan analisis label gizi makanan menggunakan teknologi OCR (Optical Character Recognition). Sistem ini membantu konsumen memahami kandungan nutrisi produk makanan berdasarkan standar kesehatan Kemenkes/WHO.

## Fitur Utama

### 1. OCR Scanner Label Gizi

- Upload foto label makanan untuk ekstraksi data otomatis
- Mendeteksi 4 nutrisi utama: Energi (kkal), Gula (g), Garam (mg), Lemak (g)
- Multi-method preprocessing untuk akurasi maksimal
- Confidence cascade system (0.5 → 0.35 → 0.2)
- Early termination saat semua data ditemukan
- Timeout protection (max 10 detik)

### 2. Image Processing Canggih

- **Shadow Normalization**: CLAHE untuk perbaikan pencahayaan
- **Auto Skew Correction**: Koreksi rotasi foto miring (±10°)
- **Selective Upscaling**: 2x untuk gambar resolusi rendah (<800px)
- **Quality Detection**: Deteksi blur, gelap, terang, kontras rendah, bayangan tidak merata
- **Spatial Reasoning**: Identifikasi posisi angka berdasarkan layout tabel

### 3. Analisis Nutrisi

Berdasarkan standar Kemenkes/WHO untuk konsumsi harian:

**Batas Maksimal Harian:**

- Energi: 2100 kkal
- Gula: 50 gram
- Lemak: 67 gram
- Garam: 2000 miligram

**Kategori Risiko:**

- 🟢 **RENDAH**: < 34% dari batas harian
- 🟡 **SEDANG**: 34-67% dari batas harian
- 🔴 **TINGGI**: ≥ 67% dari batas harian

**Rekomendasi Otomatis:**

- ≥2 kategori tinggi → "SEBAIKNYA SANGAT DIBATASI"
- 1 kategori tinggi → "DISARANKAN UNTUK MENGURANGI PORSI"
- ≥1 kategori sedang → "DAPAT DIKONSUMSI SECUKUPNYA"
- Semua kategori rendah → "DAPAT DIKONSUMSI DALAM PORSI WAJAR"

### 4. Export Laporan PDF

- Format: `Laporan-NutriScan.pdf`
- Konten lengkap: Header, data input, bar chart berwarna, analisis per nutrisi, rekomendasi
- Format A4 professional dengan margin optimal
- Tidak terpotong, siap print

### 5. User Experience

- Tips foto yang baik sebelum upload
- Loading indicator dengan estimasi waktu
- Quality warnings spesifik (blur, pencahayaan, dll)
- Missing data detection dengan saran perbaikan
- Validasi real-time saat input data
- Angka bulat untuk kemudahan membaca

## Teknologi

### Backend

- **Flask**: Web framework Python
- **EasyOCR**: OCR engine dengan dukungan bahasa Indonesia & Inggris
- **OpenCV**: Image processing dan preprocessing
- **NumPy**: Operasi array dan manipulasi data
- **thefuzz**: Fuzzy string matching untuk pencocokan keyword

### Frontend

- **HTML5/CSS3**: Struktur dan styling
- **JavaScript**: Logika client-side dan interaktivitas
- **html2pdf.js**: Export hasil ke PDF

## Instalasi

### Requirements

```bash
pip install -r requirements.txt
```

### Dependencies

```
flask
easyocr
opencv-python-headless
numpy
thefuzz
```

### Menjalankan Aplikasi

```bash
python app.py
```

Aplikasi akan berjalan di `http://127.0.0.1:5000`

## Cara Penggunaan

### 1. Upload Foto Label Gizi

- Klik tombol "Browse" untuk memilih foto
- Pastikan foto jelas, fokus, dan terang
- Foto dari depan (tidak miring)
- Hindari bayangan dan pantulan cahaya
- Untuk label di botol bulat, usahakan foto saat label rata

### 2. Review Hasil OCR

- Sistem akan otomatis mengisi nilai nutrisi
- Jika ada data yang hilang, isi manual sesuai label
- Perhatikan warning kualitas foto jika ada

### 3. Input Jumlah Sajian

- Masukkan jumlah sajian yang akan dikonsumsi
- Sistem akan menghitung total nutrisi secara otomatis

### 4. Hitung Analisis

- Klik tombol "HITUNG ANALISIS"
- Lihat bar chart berwarna untuk setiap nutrisi
- Baca rekomendasi konsumsi di bagian bawah

### 5. Export PDF (Opsional)

- Klik "SIMPAN PDF" untuk download laporan
- File akan tersimpan sebagai `Laporan-NutriScan.pdf`

## Validasi & Aturan Sistem

### Validasi Input

- **Jumlah Sajian**: 1-100
- **Energi**: 0-2100 kkal
- **Gula**: 0-50 gram
- **Lemak**: 0-67 gram
- **Garam**: 0-2000 mg
- **Cross-validation**: Energi dari lemak (lemak × 9) tidak boleh > total energi

### OCR Processing Rules

- **Character Sanitization**: 21 mapping (S→5, O→0, I→1, dll)
- **Fuzzy Matching**: Threshold 80%
- **kJ to kkal Conversion**: kJ × 0.239
- **Range Validation**: Gula/lemak > 100g per sajian = reject (kemungkinan error)
- **Spatial Reasoning**: Angka di kanan label = nilai nutrisi
- **Dynamic Row Grouping**: Adaptive threshold berdasarkan tinggi teks

### Multiple Preprocessing Methods

1. **Adaptive Threshold**: Best untuk general use
2. **Bilateral + Otsu**: Best untuk noisy images
3. **Unsharp Mask**: Best untuk slightly blur images

## Performa

### Target Metrics

- **Processing Time**: 3-7 detik (average 4-5 detik)
- **Accuracy**: 60-75% untuk foto consumer umum
- **Timeout**: Max 10 detik
- **Memory Usage**: ~200-300MB RAM

### Akurasi OCR Berdasarkan Kondisi Foto

| Kondisi Foto                          | Estimasi Akurasi |
| ------------------------------------- | ---------------- |
| Ideal (frontal, terang, fokus)        | 80-90%           |
| Good (sedikit miring, pencahayaan ok) | 65-80%           |
| Fair (blur ringan, shadow)            | 50-70%           |
| Poor (cylindrical, glare ekstrem)     | 30-50%           |

## Struktur Project

```
Project-Andrian/
├── app.py                 # Flask application (backend)
├── templates/
│   └── index.html         # Frontend UI
├── requirements.txt       # Python dependencies
└── README.md             # Dokumentasi
```

## Troubleshooting

### OCR Tidak Akurat

1. Pastikan foto terang dan fokus
2. Foto dari depan (jangan miring)
3. Hindari refleksi cahaya (glare)
4. Untuk botol bulat, foto saat label tampak flat
5. Resolusi minimal 800px width

### Server Error

1. Pastikan semua dependencies terinstall: `pip install -r requirements.txt`
2. Check Flask running: `python app.py`
3. Port 5000 available (tidak digunakan aplikasi lain)

### Data Tidak Terdeteksi

1. Foto ulang dengan pencahayaan lebih baik
2. Isi manual mengikuti label asli
3. Pastikan format label standar Indonesia/Inggris

### PDF Terpotong

- Refresh halaman browser
- Clear browser cache
- Gunakan browser terbaru (Chrome/Firefox/Edge)

## Standar Kesehatan

Sistem ini menggunakan standar batas konsumsi harian dari:

- **Kementerian Kesehatan RI**
- **World Health Organization (WHO)**

Rekomendasi bersifat umum dan tidak menggantikan konsultasi dengan ahli gizi profesional.

## Batasan Sistem

- OCR mungkin tidak 100% akurat untuk semua jenis label
- Memerlukan koneksi internet untuk library CDN (html2pdf.js)
- Tidak mendukung label dalam bahasa selain Indonesia/Inggris
- Background processing tidak async (sequential per request)
- Shared hosting dengan RAM terbatas mungkin slower

## Pengembangan Selanjutnya

Potensi improvement:

- [ ] Database untuk menyimpan history scan
- [ ] User authentication & profile
- [ ] Mobile app (React Native/Flutter)
- [ ] Barcode scanner integration
- [ ] Multi-language support
- [ ] Cloud deployment (Heroku/AWS/GCP)
- [ ] GPU acceleration untuk OCR lebih cepat
- [ ] Machine learning untuk brand recognition

## Lisensi

Proyek ini dibuat untuk keperluan edukasi dan penelitian.

## Kontributor

- **OCR Engine**: EasyOCR
- **Image Processing**: OpenCV
- **PDF Export**: html2pdf.js

---

**NUTRISCAN v3** - Membantu Anda Memahami Nutrisi Makanan Anda 🍎📊
