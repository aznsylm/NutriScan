# ========================================
# IMPORT LIBRARIES
# ========================================
from flask import Flask, render_template, request, jsonify
import easyocr  # OCR untuk baca teks di gambar
import re  # Regex untuk extract angka
import numpy as np  # Operasi array & matematika
import cv2  # OpenCV untuk image processing
from thefuzz import fuzz  # Fuzzy string matching untuk toleransi typo
import time  # Tracking waktu proses
# ========================================
# SETUP FLASK & EASYOCR
# ========================================
app = Flask(__name__)
reader = easyocr.Reader(['id', 'en'], gpu=False)  # Inisialisasi OCR bahasa Indonesia & Inggris
# ========================================
# FUNGSI SANITASI TEKS UNTUK KEYWORD MATCHING
# Konversi angka yang salah baca OCR jadi huruf (contoh: "3n3rgi" → "energi")
# ========================================
def sanitize_for_keywords(text):
    # Untuk matching keywords seperti "energi", "gula" yang OCR salah baca jadi "en0r91", "9u1a"
    mapping = {
        '0': 'o', '1': 'i', '2': 'z', '3': 'e', '4': 'a',
        '5': 's', '6': 'g', '7': 't', '8': 'b', '9': 'g',
        '@': 'a', '$': 's', '€': 'e', '%': '', '*': ''
    }
    result = text.lower()
    for k, v in mapping.items():
        result = result.replace(k, v)
    result = re.sub(r'\s+', ' ', result).strip()
    return result
# ========================================
# FUNGSI SANITASI TEKS UNTUK EXTRACT ANGKA
# Konversi huruf yang salah baca OCR jadi angka (contoh: "O" → "0", "I" → "1")
# ========================================
def sanitize_ocr_text(text):
    # Untuk extract angka - convert letters to numbers
    mapping = {
        # Angka 0
        'O': '0', 'o': '0', 'D': '0', 'Q': '0', 'C': '0', 'U': '0', 'Ο': '0', 'О': '0',
        # Angka 1
        'I': '1', 'i': '1', 'l': '1', '|': '1', 'İ': '1', 'Ι': '1', 'ı': '1',
        # Angka 2
        'Z': '2', 'z': '2', 'Ƶ': '2',
        # Angka 3
        'E': '3', 'Ε': '3', 'З': '3',
        # Angka 4
        'A': '4', 'Α': '4',
        # Angka 5
        'S': '5', 's': '5', '§': '5', 'ș': '5',
        # Angka 6
        'G': '6', 'b': '6', 'ь': '6',
        # Angka 7
        'T': '7', '†': '7', 'Ţ': '7',
        # Angka 8
        'B': '8', 'ß': '8', 'В': '8',
        # Angka 9
        'g': '9', 'q': '9', 'ց': '9',
        # Cleaning special characters
        '°': '', '•': '', '●': '', '○': '', '□': '', '■': ''
    }
    
    result = text
    for k, v in mapping.items():
        result = result.replace(k, v)
    
    # Hapus multiple spaces
    result = re.sub(r'\s+', ' ', result)
    return result.strip()
# ========================================
# FUNGSI CEK KUALITAS GAMBAR
# Deteksi blur, gelap, terang, kontras rendah, bayangan tidak merata
# ========================================
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
# ========================================
# FUNGSI ENHANCE GAMBAR
# Normalisasi bayangan & sharpening untuk gambar blur
# ========================================
def enhance_image(img, quality_info=None):
    """Tingkatkan kualitas gambar - optimized with blur handling"""
    # Shadow normalization using CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # AGGRESSIVE SHARPENING for blur images
    if quality_info and 'sharpness' in quality_info and quality_info['sharpness'] < 100:
        # Image is blur - apply strong unsharp mask
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 3.0)
        enhanced = cv2.addWeighted(enhanced, 2.0, gaussian, -1.0, 0)  # Stronger sharpening
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    
    return enhanced
# ========================================
# FUNGSI DETEKSI & KOREKSI ROTASI GAMBAR
# Putar gambar jika miring agar teks lurus
# ========================================
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
# ========================================
# FUNGSI UPSCALE GAMBAR
# Perbesar gambar kecil agar OCR lebih akurat
# ========================================
def upscale_image(img, scale=2):
    """Upscale image untuk OCR lebih baik pada teks kecil"""
    h, w = img.shape[:2]
    # Only upscale if image is small
    if w < 1000:
        return cv2.resize(img, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
    return img
# ========================================
# METODE PREPROCESSING #1: ADAPTIVE THRESHOLD
# Untuk gambar dengan pencahayaan tidak merata
# ========================================
def preprocess_adaptive_threshold(gray):
    """Method 1: Adaptive Threshold - Best for general use"""
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, 11, 2)

def preprocess_bilateral_otsu(gray):
    """Method 2: Bilateral + Otsu - Best for noisy images"""
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary
# ========================================
# METODE PREPROCESSING #3: UNSHARP MASKING
# Untuk gambar sedikit blur
# ========================================
def preprocess_unsharp_mask(gray):
    """Method 3: Unsharp masking - Best for slightly blurry images"""
    gaussian = cv2.GaussianBlur(gray, (0, 0), 3.0)
    unsharp = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
    return cv2.adaptiveThreshold(unsharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, 11, 2)
# ========================================
# METODE PREPROCESSING #4: MORPHOLOGICAL CLEANUP
# Untuk gambar blur & noise dengan sharpening agresif
# ========================================
def preprocess_morphological_cleanup(gray):
    """IMPROVEMENT 4: Morphological Operations - BEST for blur/noise + ULTRA sharpening"""
    # ULTRA AGGRESSIVE sharpening for thin text
    gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
    sharpened = cv2.addWeighted(gray, 2.5, gaussian, -1.5, 0)  # Very strong sharpening
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    # Stronger bilateral filter for blur
    denoised = cv2.bilateralFilter(sharpened, 9, 75, 75)
    
    # OTSU threshold (better than adaptive for blur)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations untuk cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))  # Smaller kernel for thin text
    # Remove tiny noise
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    # Close small gaps in text
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return cleaned
# ========================================
# METODE PREPROCESSING #5: THIN TEXT OPTIMIZED
# Dioptimalkan khusus untuk teks tipis/pudar di label
# ========================================
def preprocess_thin_text_optimized(gray):
    """NEW: Optimized specifically for thin/light text on labels"""
    # Step 1: Strong contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Step 2: Aggressive sharpening
    gaussian = cv2.GaussianBlur(enhanced, (0, 0), 1.5)
    sharpened = cv2.addWeighted(enhanced, 3.0, gaussian, -2.0, 0)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    # Step 3: Adaptive threshold with large block (works better for thin text)
    binary = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 5)
    
    # Step 4: Minimal morphology (preserve thin strokes)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return cleaned
# ========================================
# METODE PREPROCESSING #6: FONT ADAPTIVE
# Deteksi ketebalan font & sesuaikan threshold
# ========================================
def preprocess_font_adaptive(gray):
    """IMPROVEMENT 7: Font-Adaptive Thresholding - handle bold/thin fonts"""
    # Calculate local variance untuk detect font thickness
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    variance = cv2.Laplacian(blur, cv2.CV_64F).var()
    
    if variance > 150:  # Bold/thick font
        # Use aggressive threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:  # Thin font or low contrast
        # Use adaptive threshold with larger block
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 21, 10)
    
    return binary
# ========================================
# FUNGSI DETEKSI ROI (REGION OF INTEREST)
# Crop langsung ke area tabel nutrisi, buang area lain yang bikin noise
# ========================================
def detect_nutrition_table_roi(img):
    """NEW: Detect and crop to nutrition table region only - FILTER NOISE!"""
    try:
        # Quick OCR pass to find "INFORMASI NILAI GIZI" or "NUTRITION FACTS"
        h, w = img.shape[:2]
        
        # Upscale if small for better header detection
        if w < 800:
            scale = 800 / w
            img_scaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        else:
            img_scaled = img
            scale = 1.0
        
        # Run quick OCR with high confidence
        results = reader.readtext(img_scaled, detail=1, paragraph=False, width_ths=0.7)
        
        # Find nutrition table header
        table_headers = ['informasi nilai gizi', 'nutrition facts', 'nilai gizi', 'fakta nutrisi', 'informasi gizi']
        roi_bbox = None
        
        for bbox, text, conf in results:
            text_lower = text.lower()
            for header in table_headers:
                if fuzz.partial_ratio(text_lower, header) > 70:
                    # Found header! Create ROI
                    x_min = int(min([p[0] for p in bbox]) / scale)
                    y_min = int(min([p[1] for p in bbox]) / scale)
                    x_max = int(max([p[0] for p in bbox]) / scale)
                    y_max = int(max([p[1] for p in bbox]) / scale)
                    
                    # Expand ROI: wider horizontally, much taller vertically (capture all nutrients below)
                    margin_x = int(w * 0.15)  # 15% horizontal margin (increased)
                    margin_y_top = int(h * 0.02)  # 2% top margin
                    height_multiplier = int(h * 0.65)  # 65% of image height for nutrients (increased from 40%)
                    
                    x1 = max(0, x_min - margin_x)
                    y1 = max(0, y_min - margin_y_top)
                    x2 = min(w, x_max + margin_x)
                    y2 = min(h, y_max + height_multiplier)
                    
                    roi_bbox = (x1, y1, x2, y2)
                    break
            
            if roi_bbox:
                break
        
        # Crop to ROI if found
        if roi_bbox:
            x1, y1, x2, y2 = roi_bbox
            cropped = img[y1:y2, x1:x2]
            
            # UPSCALE cropped region for better small text detection
            h_crop, w_crop = cropped.shape[:2]
            if w_crop < 1500:  # If cropped region is small, upscale 2x
                cropped = cv2.resize(cropped, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            
            return cropped, True
        else:
            return img, False
            
    except Exception as e:
        return img, False
# ========================================
# FUNGSI DETEKSI STRUKTUR TABEL
# Cek apakah ada garis horizontal/vertikal (border tabel)
# ========================================
def detect_table_structure(img):
    """IMPROVEMENT 6: Table Structure Detection"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    # Detect lines using morphological operations
    # Horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    # Vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    
    # Combine
    table_mask = cv2.add(horizontal, vertical)
    
    # Check if table structure exists
    has_table = np.sum(table_mask > 0) > (gray.size * 0.01)  # >1% of image is lines
    
    return has_table, table_mask
# ========================================
# FUNGSI EXTRACT ANGKA DARI TEKS
# Extract angka nutrisi dari hasil OCR dengan validasi pintar
# ========================================
def extract_number(text, nutrient_key):
    """IMPROVEMENT 5: Smart Number Extraction with better validation"""
    original_text = text
    text_lower = text.lower()
    
    # CRITICAL: Skip pure percentage text (e.g., "7%", "25%")
    if '%' in original_text:
        # Check if this is ONLY "number%" pattern
        if re.match(r'^\s*\d+\.?\d*\s*%\s*$', original_text.strip()):
            return None
        # If text has % somewhere, we'll extract all numbers and filter later
    
    # Konversi kJ ke kkal
    is_kj = 'kj' in text_lower and nutrient_key == 'energi'
    
    # Remove common non-numeric characters
    text_clean = text_lower.replace('~', '').replace('±', '').replace('<', '').replace('>', '').replace('%', ' ')
    
    # Handle Indonesian/European format: 1.234,56 atau 1 234,56
    if ',' in text_clean and '.' in text_clean:
        # European: 1.234,56 → 1234.56
        text_clean = text_clean.replace('.', '').replace(',', '.')
    elif ',' in text_clean:
        # Check if comma is decimal separator or thousand
        parts = text_clean.split(',')
        if len(parts) == 2 and len(parts[1]) <= 2:
            # Decimal: 12,5 → 12.5
            text_clean = text_clean.replace(',', '.')
        else:
            # Thousand: 1,234 → 1234
            text_clean = text_clean.replace(',', '')
    
    # Remove spaces between numbers (1 234 → 1234)
    text_clean = re.sub(r'(\d)\s+(\d)', r'\1\2', text_clean)
    
    # Extract ALL numbers and pick the BEST one
    matches = re.findall(r'(\d+\.?\d*)', text_clean)
    if not matches:
        return None
    
    # Convert to floats
    candidates = []
    for m in matches:
        try:
            candidates.append(float(m))
        except:
            continue
            continue
    
    if not candidates:
        return None
    
    # SMART SELECTION: Pick FIRST good number (not max!) based on nutrient type
    val = None
    if nutrient_key == 'energi':
        # Energy: typical 30-600 kkal per serving, avoid very small (%) and very large
        valid = [x for x in candidates if 30 <= x <= 1000]
        val = valid[0] if valid else (candidates[0] if candidates[0] >= 10 else None)
    elif nutrient_key == 'garam':
        # Salt: typical 10-3000 mg, take FIRST valid (avoid % at end)
        valid = [x for x in candidates if 5 <= x <= 5000]
        val = valid[0] if valid else None
    elif nutrient_key == 'gula':
        # Sugar: 0-50g, take FIRST valid (even small like 1g, 2g, 4g)
        valid = [x for x in candidates if 0 <= x <= 100]
        val = valid[0] if valid else None
    elif nutrient_key == 'lemak':
        # Fat: 0-50g, take FIRST valid
        valid = [x for x in candidates if 0 <= x <= 50]
        val = valid[0] if valid else None
    else:
        val = candidates[0] if candidates else None
    
    if val is None:
        return None
        
    # Convert kJ to kkal if needed
    if is_kj and val > 100:
        val = val * 0.239
        
    return round(val, 2)
# ========================================
# FUNGSI OCR DENGAN CONFIDENCE FILTERING
# Jalankan OCR & filter hasil berdasarkan confidence score
# ========================================
def perform_ocr_with_confidence(processed_img, min_confidence=0.5):
    """OCR dengan confidence filtering"""
    results = reader.readtext(processed_img)
    # Filter hasil berdasarkan confidence score
    filtered_results = [(bbox, text, conf) for bbox, text, conf in results if conf >= min_confidence]
    return filtered_results
# ========================================
# FUNGSI EXTRACT DATA DENGAN SPATIAL REASONING
# Analisis posisi teks (horizontal & vertikal) untuk match label-nilai
# ========================================
def extract_with_spatial_reasoning(results):
    """IMPROVEMENT 3: Vertical + Horizontal Spatial Reasoning"""
    if not results:
        return {'gula': None, 'lemak': None, 'garam': None, 'energi': None}
    
    # Sort by Y coordinate
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
        'gula': ['gula', 'guia', 'gulatotal', 'totalgu la', 'sugar', 'sugars'],
        'lemak': ['lemak', 'iemak', 'lemaktotal', 'totallemak', 'fat', 'totalfat'],
        'garam': ['garam', 'qaram', 'gaa ram', 'natrium', 'sodium', 'salt', 'na'],
        'energi': ['energi total', 'totalenergi', 'energitotal', 'energi', 'enerqi', 'energy', 'totalenergy', 'kalori', 'kkal', 'kcal']
    }

    # IMPROVEMENT 2: Lower fuzzy threshold to 55 (very tolerant for OCR errors)
    FUZZY_THRESHOLD = 55

    for row_idx, row in enumerate(rows):
        # Sort row items by X coordinate (left to right)
        row_sorted = sorted(row, key=lambda x: x[0][0][0])
        row_texts = [r[1] for r in row_sorted]
        
        # TWO versions: one for keyword matching, one for number extraction
        row_full_text_keywords = sanitize_for_keywords(" ".join([t for t in row_texts]))
        row_full_text_numbers = sanitize_ocr_text(" ".join([t for t in row_texts]))
        
        # CRITICAL: Skip rows that are serving info (not nutrient values!)
        skip_keywords = ['sajian', 'serving', 'kemasan', 'container', 'takaran', 'size']
        if any(kw in row_full_text_keywords for kw in skip_keywords):
            continue
        
        # Check if this row contains nutrient label
        for key, aliases in dict_keys.items():
            if extracted[key] is not None:  # Skip if already found
                continue
            
            # IMPORTANT: Skip "energi dari lemak" / "energy from fat" rows
            if key == 'energi':
                if any(x in row_full_text_keywords for x in ['dari lemak', 'from fat', 'darifat']):
                    continue
                
            for alias in aliases:
                # Match against keyword-sanitized version (angka→huruf)
                if fuzz.partial_ratio(alias, row_full_text_keywords) > FUZZY_THRESHOLD:
                    # HORIZONTAL REASONING: Find number to the RIGHT
                    label_found = False
                    label_item = None
                    
                    for idx, item in enumerate(row_sorted):
                        item_keywords = sanitize_for_keywords(item[1])
                        if fuzz.partial_ratio(alias, item_keywords) > FUZZY_THRESHOLD:
                            label_found = True
                            label_item = item
                            
                            # Collect ALL numbers to the right of label
                            candidates = []
                            for right_idx, right_item in enumerate(row_sorted[idx:]):
                                # Skip if this text item is part of serving info
                                item_lower = sanitize_for_keywords(right_item[1])
                                if any(x in item_lower for x in ['sajian', 'serving', 'kemasan', 'container']):
                                    continue
                                    
                                val = extract_number(sanitize_ocr_text(right_item[1]), key)
                                if val is not None:
                                    candidates.append(val)
                            
                            # Pick the FIRST valid number (extract_number returns best via validation)
                            if candidates and extracted[key] is None:
                                extracted[key] = candidates[0]
                            break
                    
                    # VERTICAL REASONING: Find number BELOW label (NEW!)
                    if label_found and extracted[key] is None and label_item is not None:
                        # Get label's X position and width
                        label_x = label_item[0][0][0]
                        label_x_end = label_item[0][1][0]
                        label_y_bottom = label_item[0][2][1]
                        
                        # Look in next 3 rows below (expanded from 2)
                        for next_row_idx in range(row_idx + 1, min(row_idx + 4, len(rows))):
                            next_row = rows[next_row_idx]
                            for next_item in next_row:
                                item_x = next_item[0][0][0]
                                item_y = next_item[0][0][1]
                                
                                # Check if item is below label (relax X tolerance to ±100px)
                                if (label_x - 100 <= item_x <= label_x_end + 100 and 
                                    item_y > label_y_bottom and 
                                    item_y - label_y_bottom < avg_height * 4):
                                    
                                    val = extract_number(sanitize_ocr_text(next_item[1]), key)
                                    if val is not None and '%' not in next_item[1]:
                                        extracted[key] = val
                                        break
                            
                            if extracted[key] is not None:
                                break
                    
                    if extracted[key] is not None:
                        break
    
    return extracted
# ========================================
# ROUTE FLASK - HALAMAN UTAMA
# ========================================
@app.route('/')
def index():
    return render_template('index.html')
# ========================================
# ROUTE FLASK - API SCAN FOTO LABEL
# Terima foto, proses dengan OCR, return data nutrisi
# ========================================
@app.route('/scan', methods=['POST'])
def scan():
    start_time = time.time()
    # Validasi ada file atau tidak
    if 'image' not in request.files: 
        return jsonify({'error': 'No file'}), 400
    
    file = request.files['image']
    # Load gambar dari file upload
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    # Log proses untuk debugging
    processing_log = []
    # STEP 1: Cek kualitas gambar (blur, gelap, terang, dll)
    quality = check_image_quality(img)
    # STEP 2: Perbaiki kualitas gambar (normalisasi bayangan & sharpening)
    enhanced = enhance_image(img, quality_info=quality)
    # STEP 3: Koreksi rotasi gambar jika miring
    corrected, skew_angle = detect_and_correct_skew(enhanced)
    # STEP 4: Upscale gambar kecil agar OCR lebih baik
    h, w = corrected.shape[:2]
    if w < 800:
        corrected = upscale_image(corrected, scale=2)
    # STEP 5: Crop ke area tabel nutrisi saja (buang noise)
    roi_img, roi_found = detect_nutrition_table_roi(corrected)
    if roi_found:
        processing_log.append("✓ ROI: Nutrition table cropped")
    else:
        processing_log.append("⚠ ROI: Using full image")
    corrected = roi_img
    # STEP 6: Konversi ke grayscale untuk preprocessing
    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
    # STEP 7: Cek struktur tabel (ada garis atau tidak)
    has_table, table_mask = detect_table_structure(corrected)
    # STEP 8: Multi-method preprocessing - coba 3 metode terbaik
    preprocessing_methods = [
        ('morphological', preprocess_morphological_cleanup),  # Best for blur
        ('bilateral_otsu', preprocess_bilateral_otsu),  # Best for noise
        ('thin_text', preprocess_thin_text_optimized),  # NEW: Best for thin/light text
    ]
    # Level confidence OCR (lebih rendah = lebih toleran)
    confidence_levels = [0.2, 0.15]
    # Inisialisasi hasil terbaik
    best_extracted = {'gula': None, 'lemak': None, 'garam': None, 'energi': None}
    best_score = 0
    # Log jika tabel terdeteksi
    if has_table:
        processing_log.append("✓ Table structure detected")
    # LOOP: Coba semua kombinasi metode preprocessing & confidence level
    for conf_level in confidence_levels:
        for method_name, method_func in preprocessing_methods:
            # Timeout protection - max 70 detik
            if time.time() - start_time > 70:
                processing_log.append(f"Timeout at {time.time() - start_time:.1f}s")
                break
            
            try:
                # Jalankan preprocessing
                processed = method_func(gray)
                # Dilasi untuk expand area deteksi (optional)
                if has_table and conf_level == 0.5:
                    kernel = np.ones((3, 3), np.uint8)
                    dilated = cv2.dilate(processed, kernel, iterations=1)
                    results = perform_ocr_with_confidence(dilated, min_confidence=conf_level)
                else:
                    results = perform_ocr_with_confidence(processed, min_confidence=conf_level)
                # Extract data nutrisi dari hasil OCR
                extracted = extract_with_spatial_reasoning(results)
                # Hitung score: berapa banyak nutrisi yang berhasil ditemukan
                score = sum(1 for v in extracted.values() if v is not None)
                # Log hasil untuk debugging
                detected_count = len(results)
                sample_texts = [r[1][:20] for r in results[:3]] if results else []
                processing_log.append(f"{method_name} @ conf={conf_level}: {score}/4 found | {detected_count} texts | Sample: {sample_texts}")
                
                if score > best_score:
                    best_score = score
                    best_extracted = extracted
                # Early termination jika sudah dapat semua (4/4)
                if score == 4:
                    processing_log.append(f"✓ All data found! Stopping early.")
                    break
                    
            except Exception as e:
                processing_log.append(f"{method_name} error: {str(e)[:50]}")
                continue
        # Break outer loop jika sudah dapat semua
        if best_score == 4:
            break
    # Identifikasi nutrisi mana yang tidak terdeteksi
    missing = [k for k, v in best_extracted.items() if v is None]
    # Hitung total waktu proses
    elapsed_time = time.time() - start_time
    # Return hasil dalam format JSON
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
            'log': processing_log[:10]  # Processing log for user feedback
        }
    })
# ========================================
# JALANKAN APLIKASI FLASK
# ========================================
if __name__ == '__main__':
    app.run(debug=True)  # Set debug=False untuk production