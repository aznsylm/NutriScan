from flask import Flask, render_template, request, jsonify
import easyocr
import re
import numpy as np
import cv2
from thefuzz import fuzz
import time

app = Flask(__name__)
reader = easyocr.Reader(['id', 'en'], gpu=False)

def sanitize_ocr_text(text):
    # Enhanced mapping dengan lebih banyak karakter yang sering salah terbaca
    mapping = {
        'S': '5', 's': '5', 'O': '0', 'o': '0', 'l': '1', 'I': '1', 'i': '1',
        'B': '8', 'Z': '2', 'z': '2', 'A': '4', 'E': '3', 'b': '6', 'g': '9',
        'G': '6', 'T': '7', 'D': '0', 'C': '0', 'Q': '0', 'U': '0'
    }
    for k, v in mapping.items():
        text = text.replace(k, v)
    return text


def check_image_quality(img):
    """Cek kualitas gambar sebelum OCR"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    # Cek brightness
    brightness = np.mean(gray)
    
    # Cek blur menggunakan Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Cek contrast
    contrast = gray.std()
    
    # Shadow detection (variance dalam brightness)
    h, w = gray.shape
    regions = []
    for i in range(0, h, h//3):
        for j in range(0, w, w//3):
            region = gray[i:min(i+h//3, h), j:min(j+w//3, w)]
            regions.append(np.mean(region))
    shadow_variance = np.std(regions)
    
    quality_issues = []
    is_good = True
    
    if laplacian_var < 100:
        quality_issues.append("BLUR")
        is_good = False
    if brightness < 50:
        quality_issues.append("TERLALU_GELAP")
        is_good = False
    elif brightness > 220:
        quality_issues.append("TERLALU_TERANG")
        is_good = False
    if contrast < 30:
        quality_issues.append("KONTRAS_RENDAH")
        is_good = False
    if shadow_variance > 40:
        quality_issues.append("SHADOW_TIDAK_MERATA")
    
    quality_score = {
        'brightness': brightness,
        'sharpness': laplacian_var,
        'contrast': contrast,
        'shadow_variance': shadow_variance,
        'is_good': is_good,
        'issues': quality_issues
    }
    
    return quality_score

def enhance_image(img):
    """Tingkatkan kualitas gambar - optimized version"""
    # Shadow normalization using CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced

def detect_and_correct_skew(img):
    """Deteksi dan koreksi rotasi/skew - lightweight version"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect edges
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=100, maxLineGap=10)
    
    if lines is not None and len(lines) > 5:
        angles = []
        for line in lines[:20]:  # Only check first 20 lines for speed
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # Normalize angle
            if angle < -45:
                angle = 90 + angle
            elif angle > 45:
                angle = angle - 90
            angles.append(angle)
        
        if angles:
            median_angle = np.median(angles)
            # Only correct if skew is noticeable but not extreme
            if abs(median_angle) > 0.5 and abs(median_angle) < 10:
                h, w = img.shape[:2]
                center = (w // 2, h // 2)
                matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                rotated = cv2.warpAffine(img, matrix, (w, h), 
                                        flags=cv2.INTER_LINEAR, 
                                        borderMode=cv2.BORDER_REPLICATE)
                return rotated, median_angle
    
    return img, 0

def upscale_image(img, scale=2):
    """Upscale image untuk OCR lebih baik pada teks kecil"""
    h, w = img.shape[:2]
    # Only upscale if image is small
    if w < 1000:
        return cv2.resize(img, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
    return img

def preprocess_adaptive_threshold(gray):
    """Method 1: Adaptive Threshold - Best for general use"""
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, 11, 2)

def preprocess_bilateral_otsu(gray):
    """Method 2: Bilateral + Otsu - Best for noisy images"""
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def preprocess_unsharp_mask(gray):
    """Method 3: Unsharp masking - Best for slightly blurry images"""
    gaussian = cv2.GaussianBlur(gray, (0, 0), 3.0)
    unsharp = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
    return cv2.adaptiveThreshold(unsharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, 11, 2)


def extract_number(text, nutrient_key):
    text = text.lower().replace(',', '.')
    
    # 1. FITUR ADOPSI: Konversi kJ ke kkal (1 kJ = 0.239 kkal)
    is_kj = 'kj' in text and nutrient_key == 'energi'
    
    match = re.search(r'(\d+\.?\d*)', text)
    if match:
        val = float(match.group(1))
        
        # 2. FITUR ADOPSI: Validasi Rentang Logis (Poin 15 & 37)
        # Jika gula/lemak > 100g per sajian, kemungkinan besar itu salah baca (kecuali kalori/garam)
        if nutrient_key in ['gula', 'lemak'] and val > 100:
            return None 
            
        if is_kj:
            val = val * 0.239
            
        return round(val, 3)
    return None

def perform_ocr_with_confidence(processed_img, min_confidence=0.5):
    """OCR dengan confidence filtering"""
    results = reader.readtext(processed_img)
    # Filter hasil berdasarkan confidence score
    filtered_results = [(bbox, text, conf) for bbox, text, conf in results if conf >= min_confidence]
    return filtered_results

def extract_with_spatial_reasoning(results):
    """Extract dengan spatial reasoning - angka di kanan label adalah nilainya"""
    if not results:
        return {'gula': None, 'lemak': None, 'garam': None, 'energi': None}
    
    # Sort by Y coordinate untuk grouping rows
    results.sort(key=lambda x: x[0][0][1])
    
    # Dynamic row grouping
    rows = []
    if results:
        heights = [abs(r[0][2][1] - r[0][0][1]) for r in results]
        avg_height = np.mean(heights) if heights else 20
        row_threshold = max(avg_height * 0.8, 15)
        
        current_row = [results[0]]
        for i in range(1, len(results)):
            y_diff = abs(results[i][0][0][1] - current_row[-1][0][0][1])
            if y_diff < row_threshold:
                current_row.append(results[i])
            else:
                rows.append(current_row)
                current_row = [results[i]]
        rows.append(current_row)

    extracted = {'gula': None, 'lemak': None, 'garam': None, 'energi': None}
    dict_keys = {
        'gula': ['gula total', 'gula', 'sugar', 'sukrosa', 'glukosa', 'fruktosa', 'total sugar'],
        'lemak': ['lemak total', 'total lemak', 'lemak', 'lipid', 'total fat', 'fat'],
        'garam': ['natrium', 'sodium', 'garam', 'salt', 'natriumz'],
        'energi': ['energi total', 'energi', 'kalori', 'kkal', 'kcal', 'calories', 'energy', 'tenaga']
    }

    for row in rows:
        # Sort row items by X coordinate (left to right)
        row_sorted = sorted(row, key=lambda x: x[0][0][0])
        row_texts = [r[1] for r in row_sorted]
        row_full_text = sanitize_ocr_text(" ".join([t.lower() for t in row_texts]))
        
        # Check if this row contains nutrient label
        for key, aliases in dict_keys.items():
            for alias in aliases:
                if fuzz.partial_ratio(alias, row_full_text) > 80:
                    # SPATIAL REASONING: Find number to the RIGHT of this label
                    label_found = False
                    for idx, item in enumerate(row_sorted):
                        item_lower = item[1].lower()
                        if fuzz.partial_ratio(alias, item_lower) > 80:
                            label_found = True
                            # Look for numbers in items to the right
                            for right_item in row_sorted[idx:]:
                                val = extract_number(sanitize_ocr_text(right_item[1]), key)
                                if val is not None and '%' not in right_item[1]:
                                    if extracted[key] is None or 'total' in alias:
                                        extracted[key] = val
                                    break
                            break
                    
                    if label_found:
                        break
    
    return extracted

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scan', methods=['POST'])
def scan():
    start_time = time.time()
    
    if 'image' not in request.files: 
        return jsonify({'error': 'No file'}), 400
    
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Step 1: Check image quality
    quality = check_image_quality(img)
    
    # Step 2: Enhance image (shadow normalization)
    enhanced = enhance_image(img)
    
    # Step 3: Correct skew/rotation
    corrected, skew_angle = detect_and_correct_skew(enhanced)
    
    # Step 4: Selective upscaling for small images
    h, w = corrected.shape[:2]
    if w < 800:
        corrected = upscale_image(corrected, scale=2)
    
    # Step 5: Convert to grayscale
    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
    
    # Step 6: Multi-method preprocessing with confidence cascade
    preprocessing_methods = [
        ('adaptive', preprocess_adaptive_threshold),
        ('bilateral_otsu', preprocess_bilateral_otsu),
        ('unsharp', preprocess_unsharp_mask)
    ]
    
    confidence_levels = [0.5, 0.35, 0.2]  # Cascade: strict → medium → lenient
    
    best_extracted = {'gula': None, 'lemak': None, 'garam': None, 'energi': None}
    best_score = 0
    processing_log = []
    
    for conf_level in confidence_levels:
        for method_name, method_func in preprocessing_methods:
            # Timeout protection
            if time.time() - start_time > 10:
                processing_log.append(f"Timeout reached at {time.time() - start_time:.1f}s")
                break
            
            try:
                processed = method_func(gray)
                results = perform_ocr_with_confidence(processed, min_confidence=conf_level)
                extracted = extract_with_spatial_reasoning(results)
                
                # Score based on how many nutrients found
                score = sum(1 for v in extracted.values() if v is not None)
                
                processing_log.append(f"{method_name} @ conf={conf_level}: {score}/4 found")
                
                if score > best_score:
                    best_score = score
                    best_extracted = extracted
                    
                # Early termination if all 4 found
                if score == 4:
                    processing_log.append(f"✓ All data found! Stopping early.")
                    break
                    
            except Exception as e:
                processing_log.append(f"{method_name} error: {str(e)[:50]}")
                continue
        
        # Break outer loop if all found
        if best_score == 4:
            break
    
    # Identifikasi data hilang
    missing = [k for k, v in best_extracted.items() if v is None]
    
    elapsed_time = time.time() - start_time
    
    return jsonify({
        'data': {k: (v if v is not None else 0) for k, v in best_extracted.items()},
        'missing': missing,
        'quality': {
            'brightness': float(round(quality['brightness'], 2)),
            'sharpness': float(round(quality['sharpness'], 2)),
            'is_good': bool(quality['is_good']),
            'issues': quality['issues']
        },
        'processing': {
            'time': round(elapsed_time, 2),
            'skew_corrected': round(skew_angle, 2) if skew_angle != 0 else 0,
            'score': f"{best_score}/4",
            'log': processing_log[:5]  # Only send first 5 logs
        }
    })

if __name__ == '__main__':
    app.run(debug=True)