import face_recognition
import io
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

# Inisialisasi aplikasi FastAPI
app = FastAPI(title="Face Recognition Internal API")

@app.post("/extract-encoding")
async def extract_encoding(file: UploadFile = File(...)):
    try:
        # Membaca file foto yang dikirim dari Golang nanti
        contents = await file.read()
        
        # Mengubah file menjadi format yang bisa dibaca face_recognition
        image = face_recognition.load_image_file(io.BytesIO(contents))
        
        # Mengekstrak wajah
        encodings = face_recognition.face_encodings(image)
        
        if len(encodings) == 0:
            # Jika tidak ada wajah, kembalikan error
            return JSONResponse(status_code=400, content={"error": "Wajah tidak terdeteksi"})
        
        # Ambil wajah pertama dan ubah numpy array menjadi list Python biasa
        face_encoding = encodings[0].tolist()
        
        # Kembalikan deretan angka dalam format JSON
        return {"encoding": face_encoding}
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Terjadi kesalahan di server Python: {str(e)}"})

# Catatan: Kita tidak butuh fungsi komparasi (compare_faces) di sini.
# Pencocokan jarak euclidean nantinya bisa dilakukan langsung di query database Supabase atau di Golang.