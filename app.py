from flask import Flask, render_template, request, jsonify
import easyocr
import re
import numpy as np
import cv2
from thefuzz import fuzz

app = Flask(__name__)

# Inisialisasi OCR (GPU=False untuk kompatibilitas laptop standar)
reader = easyocr.Reader(['id', 'en'], gpu=False)

def sanitize_val(text):
    """Membersihkan karakter pengganggu dan memperbaiki salah baca angka (Poin 1-10, 20, 34)"""
    # Hapus simbol bintang, kurung, garis tegak, atau slash yang sering terbaca angka
    text = re.sub(r'[*()|I/]', '', text)
    # Perbaikan karakter mirip
    mapping = {
        'S': '5', 's': '5', 'O': '0', 'o': '0', 
        'l': '1', 'B': '8', 'A': '4', 'Z': '2', 'E': '3'
    }
    for k, v in mapping.items():
        text = text.replace(k, v)
    # Ambil hanya angka (mendukung desimal titik atau koma)
    match = re.search(r'(\d+[.,]?\d*)', text)
    if match:
        val = match.group(1).replace(',', '.')
        return val
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scan', methods=['POST'])
def scan():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # PREPROCESSING: Adaptive Thresholding untuk menangani cahaya/bayangan (Poin 40)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Deteksi Teks dengan koordinat
    results = reader.readtext(processed)
    
    # LOGIKA ROW GROUPING: Mengelompokkan teks yang sejajar secara horizontal
    rows = []
    results.sort(key=lambda x: x[0][0][1]) # Urutkan berdasarkan koordinat Y (atas ke bawah)
    
    if results:
        current_row = [results[0]]
        for i in range(1, len(results)):
            prev_y = current_row[-1][0][0][1]
            curr_y = results[i][0][0][1]
            # Jika selisih vertikal < 25px, anggap satu baris (Poin 30)
            if abs(curr_y - prev_y) < 25:
                current_row.append(results[i])
            else:
                rows.append(current_row)
                current_row = [results[i]]
        rows.append(current_row)

    extracted = {'gula': 0, 'lemak': 0, 'garam': 0, 'energi': 0}
    target_keys = {
        'gula': ['gula', 'sugar', 'total sugar'],
        'lemak': ['lemak total', 'total fat', 'lemak'],
        'garam': ['garam', 'natrium', 'sodium'],
        'energi': ['energi total', 'total energy', 'kkal', 'calories']
    }

    # Cari kata kunci di setiap baris
    for row in rows:
        row_text_full = " ".join([r[1].lower() for r in row])
        for key, aliases in target_keys.items():
            for alias in aliases:
                # Fuzzy matching untuk menangani typo (Poin 33, 22)
                if fuzz.partial_ratio(alias, row_text_full) > 85:
                    # Cari angka di baris yang sama, abaikan angka persen (%) (Poin 27)
                    for item in row:
                        val = sanitize_val(item[1])
                        if val and '%' not in item[1]:
                            extracted[key] = float(val)
                            break
    
    return jsonify(extracted)

if __name__ == '__main__':
    app.run(debug=True)