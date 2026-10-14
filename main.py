import os
import shutil
import secrets
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from engines import image_classifier, analyze_metadata, compute_ela, process_video_frames
from PIL import Image
import cv2

app = FastAPI(title="Media Provenance & Verification API for Journalists")

# Enable cross-laptop communication with frontend UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploaded_media"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def generate_secure_path(original_filename: str) -> str:
    _, ext = os.path.splitext(original_filename.lower())
    secure_name = f"{secrets.token_hex(16)}{ext}"
    return os.path.join(UPLOAD_DIR, secure_name)

@app.post("/verify/image")
async def verify_image(file: UploadFile = File(...)):
    file_path = generate_secure_path(file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        with Image.open(file_path) as img:
            img.verify()
        
        img = Image.open(file_path)
        ai_predictions = image_classifier(img)
        
        # Pull out the top prediction object classification array
        top_prediction = ai_predictions[0] if ai_predictions else {"label": "unknown", "score": 0.0}
        
        metadata_report = analyze_metadata(file_path)
        ela_image_path = compute_ela(file_path)
        
        # Hybrid Verdict Analysis Engine
        if metadata_report["software_found"]:
            final_verdict = "Manipulated/Edited (Software Metadata Detected)"
        elif metadata_report["is_stripped"] and any(x in top_prediction["label"].lower() for x in ["web site", "screen", "monitor"]):
            final_verdict = "Digital Capture Artifact (Verified Screenshot or Flat Web Graphic)"
        elif top_prediction["score"] > 0.85:
            final_verdict = "High Statistical Anomaly (Suspected AI/Synthetic Media Patterns)"
        else:
            final_verdict = "Likely Authentic Photographic Source Media"
            
        return {
            "original_filename": file.filename,
            "structural_classification": ai_predictions,
            "metadata_forensics": metadata_report,
            "ela_heatmap_path": ela_image_path,
            "final_journalism_verdict": final_verdict,
            "status": "Success"
        }
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=400, 
            detail=f"Uploaded file failed processing routines. Error: {str(e)}"
        )

@app.post("/verify/video")
async def verify_video(file: UploadFile = File(...)):
    file_path = generate_secure_path(file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        cap = cv2.VideoCapture(file_path)
        is_opened = cap.isOpened()
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_opened else 0
        cap.release()
        
        if not is_opened or total_frames <= 0:
            raise ValueError("OpenCV engine could not parse video tracks.")
            
        video_report = process_video_frames(file_path)
        
        return {
            "original_filename": file.filename,
            "video_forensics": video_report,
            "status": "Success"
        }
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=400, 
            detail=f"Uploaded file failed processing video arrays. Error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
