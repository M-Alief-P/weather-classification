import os
import cv2
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy

app = Flask(__name__)

# --- 1. MEMUAT MODEL & SCALER ---
try:
    model = joblib.load('net_cuaca_python_v2.pkl')
    scaler = joblib.load('scaler_cuaca_python_v2.pkl')
    classList = joblib.load('classList_python.pkl')
    print("Model berhasil dimuat!")
except Exception as e:
    print("Error memuat model. Pastikan file .pkl ada di direktori yang sama.")
    print(e)

# --- 2. FUNGSI EKSTRAKSI FITUR (Sama dengan Pelatihan) ---
def extract_features(image_path):
    # Baca gambar
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Gambar tidak dapat dibaca")
        
    # Resize ke 256x256
    img = cv2.resize(img, (256, 256))
    
    # Konversi BGR (OpenCV) ke RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Konversi ke Grayscale untuk GLCM dan Entropi
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Rata-rata RGB
    MeanR = np.mean(rgb_img[:, :, 0])
    MeanG = np.mean(rgb_img[:, :, 1])
    MeanB = np.mean(rgb_img[:, :, 2])
    
    # GLCM Fitur Tekstur
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(gray, distances=[1], angles=angles, symmetric=True, normed=True)
    
    CiriCON = np.mean(graycoprops(glcm, 'contrast'))
    CiriCOR = np.mean(graycoprops(glcm, 'correlation'))
    CiriASM = np.mean(graycoprops(glcm, 'energy')) 
    CiriIDM = np.mean(graycoprops(glcm, 'homogeneity')) 
    
    # Entropy
    CiriENTR = shannon_entropy(gray)
    
    # Gabungkan menjadi 1 array 2D untuk scaler
    features = np.array([[MeanR, MeanG, MeanB, CiriENTR, CiriASM, CiriCON, CiriCOR, CiriIDM]])
    return features

# --- 3. ROUTING WEB ---
@app.route('/')
def index():
    # Menampilkan halaman utama
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'Tidak ada file yang diunggah'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'File tidak valid'})
        
    # UBAH BAGIAN INI: Simpan ke folder /tmp/ yang diizinkan oleh Vercel
    temp_path = 'temp_upload.jpg'
    file.save(temp_path)
    
    try:
        # Ekstraksi fitur
        features = extract_features(temp_path)
        
        # Normalisasi fitur menggunakan parameter data latih
        features_norm = scaler.transform(features)
        
        # Prediksi menggunakan JST (MLPClassifier)
        prediction = model.predict(features_norm)
        predicted_class_index = int(prediction[0])
        
        # Ambil nama kelas (Cloudy, Rain, Shine)
        result_class = classList[predicted_class_index]
        
        # Hapus file sementara
        os.remove(temp_path)
        
        return jsonify({
            'status': 'success', 
            'prediction': result_class
        })
        
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    # Jalankan server
    app.run(debug=True, port=5000)
