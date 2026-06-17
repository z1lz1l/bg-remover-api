"""
BG Remover API - simple, minimal FastAPI service for the customer-facing site.

Provides a single batch endpoint that uploads images to FAL.AI's
pixelcut/background-removal model in parallel and returns the processed PNGs.
"""
import base64
import os
import uuid
import logging
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from remover import process_batch
from config import FAL_KEY as DEFAULT_FAL_KEY


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend", "dist")

ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB per image
MAX_BATCH = 50

os.makedirs(UPLOAD_DIR, exist_ok=True)

logger = logging.getLogger("bg-remover")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
app = FastAPI(title="BG Remover API", version="2.0.0")

# Comma-separated list of allowed origins (e.g. "https://remove.b-magnet.co.il,https://www.remove.b-magnet.co.il").
# Defaults to "*" in development, must be set in production via the ALLOWED_ORIGINS env var.
_allowed = os.environ.get("ALLOWED_ORIGINS", "*").strip()
allow_origins_list = [o.strip() for o in _allowed.split(",") if o.strip()] or ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins_list,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _is_valid_upload(file: UploadFile) -> bool:
    """Validate file extension and content type."""
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext and ext not in ALLOWED_EXTS:
        return False
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        return False
    return True


def _get_fal_key() -> str:
    """Resolve the FAL key from env var or the embedded default."""
    return os.environ.get("FAL_KEY", "").strip() or DEFAULT_FAL_KEY


def _save_upload(file: UploadFile, dest_path: str) -> int:
    """Stream an UploadFile to disk, enforcing a max size. Returns bytes written."""
    bytes_written = 0
    with open(dest_path, "wb") as out:
        while chunk := file.file.read(1024 * 1024):
            bytes_written += len(chunk)
            if bytes_written > MAX_FILE_SIZE:
                out.close()
                try:
                    os.remove(dest_path)
                except OSError:
                    pass
                raise HTTPException(
                    status_code=413,
                    detail=f"File '{file.filename}' exceeds the {MAX_FILE_SIZE // (1024*1024)} MB size limit.",
                )
            out.write(chunk)
    return bytes_written


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health():
    """Simple health check."""
    fal_configured = bool(_get_fal_key())
    return {
        "status": "ok",
        "model": "pixelcut/background-removal",
        "fal_configured": fal_configured,
        "max_file_size_mb": MAX_FILE_SIZE // (1024 * 1024),
        "max_batch": MAX_BATCH,
    }


@app.post("/api/process-batch")
async def process_batch_endpoint(files: List[UploadFile] = File(default=[])):
    """
    Accept multiple images, remove backgrounds via FAL.AI in parallel,
    and return the processed file URLs.

    Each result in the response has the same `id` as the corresponding input
    file (index-based when no original name is available).
    """
    fal_key = _get_fal_key()
    if not fal_key:
        raise HTTPException(
            status_code=500,
            detail="FAL_KEY is not configured on the server. Set the FAL_KEY environment variable.",
        )

    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    if len(files) > MAX_BATCH:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size exceeds the {MAX_BATCH} image limit.",
        )

    # Persist uploads and validate
    saved_paths: list[str] = []
    items: list[dict] = []
    for idx, file in enumerate(files):
        if not _is_valid_upload(file):
            # Cleanup any previously saved files in this batch
            for p in saved_paths:
                try:
                    os.remove(p)
                except OSError:
                    pass
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: '{file.filename}'. Allowed: PNG, JPG, WEBP.",
            )

        original_name = file.filename or f"image_{idx}"
        ext = os.path.splitext(original_name)[1].lower() or ".png"
        file_id = str(uuid.uuid4())
        upload_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
        try:
            _save_upload(file, upload_path)
        except HTTPException:
            for p in saved_paths:
                try:
                    os.remove(p)
                except OSError:
                    pass
            raise

        saved_paths.append(upload_path)
        items.append({
            "id": file_id,
            "original_name": original_name,
            "upload_path": upload_path,
            "processed_filename": f"{file_id}_no_bg.png",
        })

    # Run the FAL.AI batch
    try:
        results = await process_batch(saved_paths, fal_key)
    except Exception as e:
        logger.exception("Batch processing failed")
        # Cleanup uploads
        for p in saved_paths:
            try:
                os.remove(p)
            except OSError:
                pass
        raise HTTPException(
            status_code=502,
            detail=f"Background removal service failed: {e}",
        )

    # Encode processed PNGs as base64 data URLs and build response
    response_items: list[dict] = []
    for item, png_bytes in zip(items, results):
        data_url = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
        response_items.append({
            "id": item["id"],
            "original_name": item["original_name"],
            "data_url": data_url,
            "download_name": os.path.splitext(item["original_name"])[0] + "_no_bg.png",
        })
        # Remove the now-unused upload
        try:
            os.remove(item["upload_path"])
        except OSError:
            pass

    return {"count": len(response_items), "items": response_items}


# --------------------------------------------------------------------------- #
# Static files - the frontend is built into ../frontend/dist and served as
# the SPA root below.
# --------------------------------------------------------------------------- #
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    logger.info("Serving frontend from: %s", FRONTEND_DIR)
else:
    logger.info("Frontend dist not found, running in API-only mode.")


# --------------------------------------------------------------------------- #
# Error handler - always JSON
# --------------------------------------------------------------------------- #
@app.exception_handler(HTTPException)
async def http_exception_handler(_request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=False)
