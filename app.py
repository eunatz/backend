import datetime
import math
import firebase_admin
from firebase_admin import credentials, firestore
import requests
import numpy as np
import tensorflow.compat.v1 as tf
import cv2
import os
import cloudinary
import cloudinary.api
import cloudinary.uploader
from flask import Flask, request, jsonify
from scipy.spatial.distance import cosine

app = Flask(__name__)

# Inisialisasi Firebase Admin SDK
cred = credentials.Certificate('serviceAccountKey.json')  # Path ke kunci JSON Anda
firebase_admin.initialize_app(cred)

db = firestore.client()  # Inisialisasi Firestore


cloudinary.config(
    cloud_name="dtkvuy3bd",  # Ganti dengan Cloud Name Anda
    api_key="587117121626688",        # Ganti dengan API Key Anda
    api_secret="DnaGQVrjoBI8gWLO-Y5ES8CUmAY"   # Ganti dengan API Secret Anda
)

# Load Haar Cascade Classifiers
frontal_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

def detect_faces_multi_cascade(img, nama_siswa, uid):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    # Deteksi wajah
    frontal_faces = frontal_face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    profile_faces_left = profile_face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    gray_flipped = cv2.flip(gray, 1)
    profile_faces_right = profile_face.detectMultiScale(gray_flipped, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    profile_faces_right = [(gray.shape[1] - (x+w), y, w, h) for (x, y, w, h) in profile_faces_right]

    all_faces = list(frontal_faces) + list(profile_faces_left) + list(profile_faces_right)
    img_with_box = img.copy()
    face_results = []

    # Buat folder dataset untuk siswa di Cloudinary
    dataset_folder = f"dataset/{nama_siswa}"
    cropped_folder = f"{dataset_folder}/cropped_faces"
    boxed_folder = f"{dataset_folder}/boxed_faces"

    # Proses setiap wajah
    for i, (x, y, w, h) in enumerate(all_faces):
        face_crop = img[y:y+h, x:x+w]
        
        # Simpan wajah yang dicrop ke folder dinamis di Cloudinary
        cropped_upload = cloudinary.uploader.upload(
            cv2.imencode('.jpg', face_crop)[1].tobytes(),
            folder=cropped_folder,
            resource_type="image"
        )
        
        if (x, y, w, h) in frontal_faces:
            color, face_type = (0, 255, 0), "Frontal"
        elif (x, y, w, h) in profile_faces_left:
            color, face_type = (255, 0, 0), "Profil Kiri"
        else:
            color, face_type = (0, 0, 255), "Profil Kanan"

        cv2.rectangle(img_with_box, (x, y), (x+w, y+h), color, 2)
        label = f'{face_type} {i+1}: ({x},{y}) {w}x{h}'
        cv2.putText(img_with_box, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        face_results.append({
            'type': face_type,
            'coordinates': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
            'cropped_face_url': cropped_upload['secure_url']
        })

    # Upload gambar dengan bounding box
    boxed_upload = cloudinary.uploader.upload(
        cv2.imencode('.jpg', img_with_box)[1].tobytes(),
        folder=boxed_folder,
        resource_type="image"
    )
    
    # Update dokumen siswa di Firestore dengan URL dataset
    try:
        doc_ref = db.collection('siswa').document(uid)
        doc_ref.update({
            'url_dataset': cropped_folder
        })
    except Exception as e:
        print(f"Error updating Firestore: {e}")

    return {
        'faces': face_results, 
        'boxed_image_url': boxed_upload['secure_url'],
        'dataset_folder': dataset_folder
    }

@app.route('/detect', methods=['POST'])
def detect_faces():
    # Cek apakah UID siswa ada di request
    uid = request.form.get('uid')
    if not uid:
        return jsonify({'error': 'UID siswa tidak disertakan'}), 400

    # Ambil nama siswa dari Firestore berdasarkan UID
    try:
        doc_ref = db.collection('siswa').document(uid)
        doc = doc_ref.get()
        if not doc.exists:
            return jsonify({'error': 'Siswa dengan UID ini tidak ditemukan'}), 404
        
        nama_siswa = doc.to_dict().get('nama')
        if not nama_siswa:
            return jsonify({'error': 'Data nama siswa tidak lengkap'}), 400
    except Exception as e:
        return jsonify({'error': f'Error saat mengambil data Firestore: {str(e)}'}), 500

    # Cek apakah file gambar ada di request
    if 'image' not in request.files:
        return jsonify({'error': 'Tidak ada file gambar'}), 400

    # Ambil file gambar
    file = request.files['image']
    img_array = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Cek apakah gambar valid
    if img is None:
        return jsonify({'error': 'Gagal membaca gambar'}), 400

    try:
        # Panggil fungsi pendukung untuk mendeteksi wajah
        result = detect_faces_multi_cascade(img, nama_siswa, uid)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Disable eager execution
tf.disable_eager_execution()

class FaceNet:
    def __init__(self, model_path):
        # Reset the default graph
        tf.reset_default_graph()
        
        # Create a new graph
        self.graph = tf.Graph()
        
        with self.graph.as_default():
            # Create session configuration
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            
            # Persistent session
            self.sess = tf.Session(config=config)
            
            # Load the graph
            with tf.gfile.GFile(model_path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')

            # Initialize all variables
            self.sess.run(tf.global_variables_initializer())
            
            # Get input and output tensors
            self.input_tensor = self.graph.get_tensor_by_name('input:0')
            self.phase_train_tensor = self.graph.get_tensor_by_name('phase_train:0')
            self.embeddings_tensor = self.graph.get_tensor_by_name('embeddings:0')

    def preprocess_image(self, image):
        # Resize and normalize image
        image = cv2.resize(image, (160, 160))
        image = image.astype(np.float32)
        image = (image - 127.5) / 128.0
        return image

    def get_face_embedding(self, image):
        # Preprocess image
        preprocessed_image = self.preprocess_image(image)
        
        # Add batch dimension
        input_image = np.expand_dims(preprocessed_image, axis=0)

        # Get face embedding using the persistent session
        with self.graph.as_default():
            feed_dict = {
                self.input_tensor: input_image,
                self.phase_train_tensor: False
            }
            embedding = self.sess.run(self.embeddings_tensor, feed_dict=feed_dict)
        
        return embedding[0]

    def __del__(self):
        # Close the session when object is deleted
        if hasattr(self, 'sess'):
            self.sess.close()
            tf.reset_default_graph()

def download_cloudinary_image(image_path):
    """
    Download gambar dari Cloudinary
    """
    try:
        # Mendapatkan URL gambar dari Cloudinary
        result = cloudinary.api.resource(image_path)
        image_url = result['secure_url']

        # Download gambar
        response = requests.get(image_url, stream=True)
        if response.status_code == 200:
            # Membaca gambar langsung dari URL
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            return image
        else:
            print(f"Gagal download gambar: {image_url}")
            return None
    except Exception as e:
        print(f"Error download gambar dari Cloudinary: {e}")
        return None
    
def proses_dataset(uid):
    """
    Memproses seluruh gambar .jpg dalam dataset untuk user tertentu 
    dan menyimpan embedding ke Firestore
    """
    try:
        # Query data siswa berdasarkan uid
        siswa_ref = db.collection("siswa").document(uid)
        doc = siswa_ref.get()

        if doc.exists:
            data = doc.to_dict()
            url_dataset = data.get("url_dataset")
            
            if url_dataset:
                # Inisialisasi FaceNet
                facenet = FaceNet('20180408-102900.pb')
                
                # Dapatkan daftar file di folder dataset Cloudinary
                result = cloudinary.api.resources(
                    type='upload',
                    prefix=url_dataset,
                    max_results=500  # Sesuaikan dengan kebutuhan
                )
                
                embeddings_map = {}  # Dictionary untuk menyimpan embeddings

                # Proses setiap gambar
                for resource in result['resources']:
                    # Pastikan hanya file jpg
                    if resource['format'].lower() == 'jpg':
                        image_path = resource['public_id']
                        
                        # Download gambar
                        image = download_cloudinary_image(image_path)
                        
                        if image is not None:
                            # Generate embedding
                            embedding = facenet.get_face_embedding(image)
                            
                            # Simpan embedding ke dictionary
                            image_name = os.path.basename(image_path)
                            embeddings_map[image_name] = embedding.tolist()
                            print(f"Embedding untuk {image_name} berhasil dihasilkan.")
                        else:
                            print(f"Gagal memproses {image_path}")
                
                # Simpan embeddings ke Firestore sebagai map
                if embeddings_map:
                    siswa_ref.update({
                        'embedding_wajah': embeddings_map
                    })
                    print(f"Semua embeddings berhasil disimpan untuk UID {uid}")
                    return True
                else:
                    print("Tidak ada embeddings yang dihasilkan.")
                    return False
        
        else:
            print(f"Dokumen tidak ditemukan untuk UID: {uid}")
            return False
    
    except Exception as e:
        print(f"Error memproses dataset untuk UID {uid}: {e}")
        return False

# Route untuk memproses dataset
@app.route('/proses_dataset', methods=['POST'])
def proses_dataset_route():
    try:
        # Ambil UID dari request JSON
        data = request.get_json()
        uid = data.get('uid')
        
        if not uid:
            return jsonify({
                'status': 'error',
                'message': 'UID tidak diberikan'
            }), 400
        
        # Proses dataset
        result = proses_dataset(uid)
        
        if result:
            return jsonify({
                'status': 'success',
                'message': f'Dataset berhasil diproses untuk UID {uid}'
            }), 200
        else:
            return jsonify({
                'status': 'error',
                'message': f'Gagal memproses dataset untuk UID {uid}'
            }), 500
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/recognize', methods=['POST'])
def recognize_faces():
    try:
        # Cek apakah file gambar ada di request
        if 'image' not in request.files:
            return jsonify({'error': 'Tidak ada file gambar'}), 400

        # Ambil file gambar
        file = request.files['image']
        img_array = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Cek apakah gambar valid
        if img is None:
            return jsonify({'error': 'Gagal membaca gambar'}), 400

        # Deteksi wajah dalam gambar
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # Perbaiki pencahayaan untuk hasil deteksi lebih baik

        # Deteksi wajah frontal
        frontal_faces = frontal_face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Deteksi wajah profil kiri
        profile_faces_left = profile_face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Deteksi wajah profil kanan (gunakan gambar terbalik)
        gray_flipped = cv2.flip(gray, 1)
        profile_faces_right = profile_face.detectMultiScale(gray_flipped, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        profile_faces_right = [(gray.shape[1] - (x + w), y, w, h) for (x, y, w, h) in profile_faces_right]

        # Gabungkan semua deteksi wajah
        all_faces = list(frontal_faces) + list(profile_faces_left) + list(profile_faces_right)

        if len(all_faces) == 0:
            return jsonify({'error': 'Tidak ada wajah yang terdeteksi'}), 404

        # Inisialisasi FaceNet
        facenet = FaceNet('20180408-102900.pb')

        # Ambil semua *embedding* wajah dari Firestore
        try:
            siswa_ref = db.collection('siswa').stream()
            embeddings_data = {}
            for doc in siswa_ref:
                data = doc.to_dict()
                if 'embedding_wajah' in data:
                    embeddings_data[doc.id] = data['embedding_wajah']
        except Exception as e:
            return jsonify({'error': f'Gagal mengambil data dari Firestore: {str(e)}'}), 500

        recognized_faces = []

        # Proses setiap wajah yang terdeteksi
        for (x, y, w, h) in all_faces:
            face_crop = img[y:y+h, x:x+w]

            # Hasilkan *embedding* untuk wajah yang diunggah
            embedding = facenet.get_face_embedding(face_crop)

            # Bandingkan dengan *embedding* di dataset
            best_match_uid = None
            best_match_name = None
            best_match_score = float('inf')  # Semakin kecil semakin mirip

            for uid, embeddings_map in embeddings_data.items():
                for image_name, dataset_embedding in embeddings_map.items():
                    score = cosine(embedding, dataset_embedding)
                    if score < best_match_score:
                        best_match_uid = uid
                        best_match_name = db.collection('siswa').document(uid).get().to_dict().get('nama', 'Tidak diketahui')
                        best_match_score = score

            # Hasil pencocokan
            if best_match_score < 0.6:  # Threshold pencocokan
                recognized_faces.append({
                    'coordinates': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                    'name': best_match_name,
                    'uid': best_match_uid,
                    'similarity_score': best_match_score
                })

        return jsonify({'recognized_faces': recognized_faces}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Fungsi untuk menghitung jarak antara dua koordinat
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Radius bumi dalam kilometer
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c * 1000  # Jarak dalam meter

# Endpoint untuk validasi lokasi
@app.route('/validate-location', methods=['POST'])
def validate_location():
    try:
        # Ambil data dari request
        data = request.json
        uid_siswa = data.get('uid')
        latitude = data.get('latitude')
        longitude = data.get('longitude')

        if not uid_siswa or latitude is None or longitude is None:
            return jsonify({'status': 'error', 'message': 'Data tidak lengkap'}), 400

        # Ambil data siswa berdasarkan UID
        siswa_ref = db.collection('siswa').document(uid_siswa)
        siswa_doc = siswa_ref.get()

        if not siswa_doc.exists:
            return jsonify({'status': 'error', 'message': 'Siswa tidak ditemukan'}), 404

        siswa_data = siswa_doc.to_dict()
        uid_kelas = siswa_data.get('kelas')

        if not uid_kelas:
            return jsonify({'status': 'error', 'message': 'UID kelas tidak ditemukan untuk siswa ini'}), 404

        # Ambil data kelas berdasarkan UID
        kelas_ref = db.collection('kelas').document(uid_kelas)
        kelas_doc = kelas_ref.get()

        if not kelas_doc.exists:
            return jsonify({'status': 'error', 'message': 'Kelas tidak ditemukan'}), 404

        kelas_data = kelas_doc.to_dict()
        koordinat = kelas_data.get('koordinat')
        radius = kelas_data.get('radius')

        if not koordinat or not radius:
            return jsonify({'status': 'error', 'message': 'Koordinat atau radius kelas tidak tersedia'}), 404

        # Validasi lokasi
        jarak = calculate_distance(latitude, longitude, koordinat['latitude'], koordinat['longitude'])

        if jarak <= radius:
            # Lokasi valid, simpan absensi
            now = datetime.datetime.now()
            formatted_date = now.strftime('%Y-%m-%d')  # Format tanggal (YYYY-MM-DD)
            formatted_time = now.strftime('%H:%M')     # Format waktu (HH:mm)

            # Contoh logika penentuan status berdasarkan waktu
            late_threshold = now.replace(hour=7, minute=0, second=0, microsecond=0)  # Jam 07:00
            status = "Terlambat" if now > late_threshold else "Hadir"

            # Data absensi
            absensi_data = {
                'nama_siswa': siswa_data.get('nama'),
                'uid_siswa': uid_siswa,
                'kelas': kelas_data.get('nama_kelas'),
                'tanggal': formatted_date,
                'waktu_absensi': formatted_time,
                'status': status,
                'lokasi_absensi': {
                    'latitude': latitude,
                    'longitude': longitude
                },
                'jarak': jarak
            }

            # Simpan ke koleksi absensi
            absensi_ref = db.collection('absensi')
            absensi_ref.add(absensi_data)

            return jsonify({'status': 'success', 'message': 'Lokasi valid dan absensi disimpan', 'distance': jarak})

        else:
            return jsonify({'status': 'error', 'message': 'Lokasi tidak valid', 'distance': jarak})

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
    
# Jalankan Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)