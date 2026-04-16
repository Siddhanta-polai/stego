import hashlib, hmac, struct, uuid, io, os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pymongo import MongoClient
from gridfs import GridFS
from bson import ObjectId
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import numpy as np
from PIL import Image
from scipy.ndimage import convolve
from skimage.filters import threshold_otsu

from evaluator import StegoEvaluator
from existing_models import MSBModel, DCTJStegModel, DWTHaarModel
from ranking import AlgorithmRanker

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_SIZE_MB", "10"))
MAX_DIMENSION = int(os.getenv("MAX_DIMENSION", "4000"))

MAGIC = 0x53544547
HEADER_SIZE = 72
THRESHOLD_FIXED = 50
BLUR = np.array([[1,2,1],[2,4,2],[1,2,1]], dtype=np.float32) / 16

client = MongoClient(MONGODB_URI)
db = client.steganography_db
fs = GridFS(db)
task_results = {}

class StegoSystem:
    def _edge_mask_dynamic(self, img):
        green = img[:,:,1] & 0xFE
        gray = green.astype(np.float32)
        smooth = convolve(gray, BLUR, mode="reflect")
        sx = convolve(smooth, [[-1,0,1],[-2,0,2],[-1,0,1]], mode="reflect")
        sy = convolve(smooth, [[-1,-2,-1],[0,0,0],[1,2,1]], mode="reflect")
        mag = np.sqrt(sx*sx + sy*sy)
        mag_uint8 = np.clip(mag, 0, 255).astype(np.uint8)
        thresh = threshold_otsu(mag_uint8)
        if thresh < 5:
            thresh = THRESHOLD_FIXED
        return mag > thresh

    def _permute(self, password, n):
        seed = int.from_bytes(hashlib.sha256(password.encode()).digest()[:4], "big")
        rng = np.random.RandomState(seed)
        p = list(range(n))
        for i in range(n-1, 0, -1):
            j = rng.randint(0, i+1)
            p[i], p[j] = p[j], p[i]
        return p

    def embed(self, img, encrypted_payload, password):
        edges = np.argwhere(self._edge_mask_dynamic(img))
        if len(edges) == 0:
            h,w,_ = img.shape
            edges = np.array([(i,j) for i in range(h) for j in range(w)])
        header = struct.pack(">II", MAGIC, len(encrypted_payload)) + \
                 hashlib.sha256(encrypted_payload).digest() + \
                 hmac.new(hashlib.sha256(password.encode()).digest(),
                          encrypted_payload + hashlib.sha256(encrypted_payload).digest(),
                          hashlib.sha256).digest()
        bits = []
        for b in (header + encrypted_payload):
            bits.extend((b>>i)&1 for i in range(7,-1,-1))
        if len(bits) > len(edges) * 3:
            raise ValueError("Data exceeds capacity")
        perm = self._permute(password, len(edges))
        stego = img.copy()
        k = 0
        for idx in perm:
            if k >= len(bits): break
            i,j = edges[idx]
            for c in range(3):
                if k >= len(bits): break
                stego[i,j,c] = (stego[i,j,c] & 0xFE) | bits[k]
                k += 1
        return stego

    def extract(self, img, password):
        edges = np.argwhere(self._edge_mask_dynamic(img))
        perm = self._permute(password, len(edges))
        bits = []
        for idx in perm:
            i,j = edges[idx]
            for c in range(3):
                bits.append(img[i,j,c] & 1)
        data = bytes(int(''.join(str(b) for b in bits[i:i+8]), 2) for i in range(0, len(bits), 8))
        magic, length = struct.unpack(">II", data[:8])
        if magic != MAGIC:
            raise ValueError("Invalid header or key")
        return data[HEADER_SIZE:HEADER_SIZE+length]

stego_sys = StegoSystem()

def encrypt_message(message: bytes, password: str) -> bytes:
    key = hashlib.sha256(password.encode()).digest()
    nonce = os.urandom(12)
    cipher = AESGCM(key)
    ciphertext = cipher.encrypt(nonce, message, None)
    return nonce + ciphertext

def decrypt_message(encrypted: bytes, password: str) -> bytes:
    key = hashlib.sha256(password.encode()).digest()
    nonce = encrypted[:12]
    ciphertext = encrypted[12:]
    cipher = AESGCM(key)
    return cipher.decrypt(nonce, ciphertext, None)

def save_image_to_gridfs(image_array, filename):
    img_bytes = io.BytesIO()
    Image.fromarray(image_array).save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return str(fs.put(img_bytes, filename=filename))

def process_embedding(task_id: str, cover_arr: np.ndarray, password: str, plain_message: str):
    try:
        encrypted = encrypt_message(plain_message.encode(), password)
        our_stego = stego_sys.embed(cover_arr, encrypted, password)
        our_metrics = StegoEvaluator.calculate_all_metrics(cover_arr, our_stego, encrypted)

        models = {
            "MSB (Most Significant Bit)": MSBModel,
            "DCT (JSteg)": DCTJStegModel,
            "DWT (Haar)": DWTHaarModel,
        }
        leaderboard = [{"name": "KRISHNA (Adaptive Pixel)", **our_metrics, "is_ours": True}]
        for name, ModelClass in models.items():
            try:
                stego_arr = ModelClass.embed(cover_arr, encrypted, password)
                extracted_enc = ModelClass.extract(stego_arr, len(encrypted), password)
                metrics = StegoEvaluator.calculate_all_metrics(cover_arr, stego_arr, encrypted, extracted_enc)
                metrics["name"] = name
                metrics["is_ours"] = False
                metrics["_stego_arr"] = stego_arr
                leaderboard.append(metrics)
            except Exception as e:
                print(f"{name} failed: {e}")
                leaderboard.append({"name": name, "is_ours": False, "psnr": 0, "ssim": 0, "snr": 0,
                                    "mse": 999, "capacity_bpp": 0, "fdm": 999, "epi": 0,
                                    "entropy_original": 0, "entropy_stego": 0, "sei": 0})
        leaderboard.sort(key=lambda x: x.get("sei", 0), reverse=True)
        for i, entry in enumerate(leaderboard, 1):
            entry["rank"] = i

        stego_file_id = save_image_to_gridfs(our_stego, f"stego_{task_id}.png")
        gallery = []
        for entry in leaderboard:
            if not entry.get("is_ours") and "_stego_arr" in entry:
                gid = save_image_to_gridfs(entry["_stego_arr"], f"gallery_{entry['name']}_{task_id}.png")
                gallery.append({"name": entry["name"], "file_id": gid})
                del entry["_stego_arr"]

        best_algo, recommendation = AlgorithmRanker.get_best_algorithm(leaderboard)

        task_results[task_id] = {
            "success": True,
            "stego_file_id": stego_file_id,
            "metrics": our_metrics,
            "leaderboard": leaderboard,
            "gallery": gallery,
            "recommendation": recommendation
        }
    except Exception as e:
        task_results[task_id] = {"success": False, "error": str(e)}

app = FastAPI(title="Adaptive Pixel Steganography")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ---------- Root route (added) ----------
@app.get("/")
async def root():
    return {"message": "Steganography API is running", "docs": "/docs"}

# ---------- API Endpoints ----------
@app.post("/api/hide")
async def hide(background_tasks: BackgroundTasks, image: UploadFile = File(...), password: str = Form(...), message: str = Form(...)):
    contents = await image.read()
    if len(contents) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
        raise HTTPException(400, f"Image exceeds {MAX_IMAGE_SIZE_MB} MB")
    try:
        cover = Image.open(io.BytesIO(contents)).convert("RGB")
    except:
        raise HTTPException(400, "Invalid image file")
    if cover.width > MAX_DIMENSION or cover.height > MAX_DIMENSION:
        raise HTTPException(400, f"Dimensions exceed {MAX_DIMENSION}px")
    cover_arr = np.array(cover)
    edges = np.argwhere(stego_sys._edge_mask_dynamic(cover_arr))
    capacity_bits = len(edges) * 3
    if len(message.encode()) * 8 > capacity_bits:
        raise HTTPException(400, f"Message too large. Capacity: {capacity_bits//8} bytes")
    task_id = str(uuid.uuid4())
    background_tasks.add_task(process_embedding, task_id, cover_arr, password, message)
    return {"success": True, "task_id": task_id}

@app.get("/api/result/{task_id}")
async def get_result(task_id: str):
    if task_id not in task_results:
        return {"success": False, "pending": True}
    return task_results[task_id]

@app.get("/api/image/{file_id}")
async def get_image(file_id: str):
    try:
        grid_out = fs.get(ObjectId(file_id))
        return StreamingResponse(grid_out, media_type="image/png")
    except:
        raise HTTPException(404, "Image not found")

@app.post("/api/reveal")
async def reveal(image: UploadFile = File(...), password: str = Form(...)):
    contents = await image.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except:
        raise HTTPException(400, "Invalid image file")
    try:
        encrypted_payload = stego_sys.extract(np.array(img), password)
        plain_message = decrypt_message(encrypted_payload, password)
        return {"success": True, "message": plain_message.decode()}
    except Exception as e:
        raise HTTPException(400, f"Extraction failed: {str(e)}")

@app.post("/api/capacity")
async def get_capacity(image: UploadFile = File(...)):
    contents = await image.read()
    cover = Image.open(io.BytesIO(contents)).convert("RGB")
    cover_arr = np.array(cover)
    edges = np.argwhere(stego_sys._edge_mask_dynamic(cover_arr))
    capacity_bits = len(edges) * 3
    return {"capacity_bits": capacity_bits}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8002"))
    uvicorn.run(app, host="0.0.0.0", port=port)
