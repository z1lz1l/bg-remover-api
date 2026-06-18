"""
BG Remover API - FAL.AI processing with SSE streaming.

Each uploaded image is processed independently via FAL.AI.
It supports uploading up to 50 images and streams results back
in real-time to the frontend.
"""
import os
import uuid
import base64
import json
import asyncio
import logging
from typing import List, AsyncGenerator

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

try:
    from config import FAL_KEY as DEFAULT_FAL_KEY
except Exception:
    DEFAULT_FAL_KEY = ""


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend", "dist")

ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
ALLOWED_CONTENT_TYPES = {
    "image/png", "image/jpeg", "image/jpg", "image/webp",
    "application/octet-stream", "binary/octet-stream", "",
}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB per image
MAX_BATCH = 50  # Allow up to 50 images per request

UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/tmp/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

logger = logging.getLogger("bg-remover")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
app = FastAPI(title="BG Remover API", version="6.0.0")

_allowed = os.environ.get("ALLOWED_ORIGINS", "*").strip()
allow_origins_list = [o.strip() for o in _allowed.split(",") if o.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins_list,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _get_fal_key() -> str:
    return os.environ.get("FAL_KEY", "").strip() or DEFAULT_FAL_KEY


def _is_valid_upload(file: UploadFile) -> bool:
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext and ext not in ALLOWED_EXTS:
        return False
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        return False
    return True


async def _process_one_fal(fal_key: str, name: str, data: bytes) -> bytes:
    """Send one image to FAL.AI and return the raw PNG bytes."""
    from remover import process_batch as fal_process_batch

    file_id = str(uuid.uuid4())
    ext = os.path.splitext(name)[1].lower() or ".png"
    path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
    try:
        with open(path, "wb") as out:
            out.write(data)
        results = await fal_process_batch([path], fal_key, max_concurrency=1)
        if not results or not results[0]:
            raise RuntimeError("FAL.AI returned no result")
        return results[0]
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


async def process_one_image(idx: int, name: str, data: bytes, fal_key: str) -> dict:
    """
    Process a single image via FAL.AI.
    Returns a dict that is safe to JSON-serialise directly.
    """
    image_id = str(uuid.uuid4())
    png_bytes: bytes | None = None
    error: str | None = None

    if fal_key:
        try:
            png_bytes = await _process_one_fal(fal_key, name, data)
            logger.info("[%s] processed by FAL.AI", name)
        except Exception as e:
            logger.exception("[%s] FAL.AI failed", name)
            error = f"FAL.AI failed: {e}"
    else:
        error = "FAL_KEY is not configured on the server."

    if png_bytes is None:
        return {
            "id": image_id,
            "index": idx,
            "original_name": name,
            "error": error or "Processing failed.",
            "status": "error",
            "backend": "fal",
        }

    data_url = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
    return {
        "id": image_id,
        "index": idx,
        "original_name": name,
        "data_url": data_url,
        "download_name": os.path.splitext(name)[0] + "_no_bg.png",
        "backend": "fal",
        "status": "ok",
    }


def _sse_pack(payload: dict, event: str = "item") -> str:
    """Pack a dict as a single SSE message."""
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "version": "6.0.0",
        "max_file_size_mb": MAX_FILE_SIZE // (1024 * 1024),
        "max_batch": MAX_BATCH,
    }


@app.post("/api/process-stream")
async def process_stream_endpoint(files: List[UploadFile] = File(default=[])):
    """
    Accept up to MAX_BATCH images and stream back per-image results as
    Server-Sent Events.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")
    if len(files) > MAX_BATCH:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size exceeds the {MAX_BATCH} image limit.",
        )

    # ---- 1. Validate + read every upload up front ------------------------- #
    file_data: list[tuple[str, bytes]] = []
    for idx, file in enumerate(files):
        if not _is_valid_upload(file):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: '{file.filename}'.",
            )
        original_name = file.filename or f"image_{idx}"
        data = await file.read()
        if len(data) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File '{original_name}' exceeds the {MAX_FILE_SIZE // (1024*1024)} MB size limit.",
            )
        file_data.append((original_name, data))

    fal_key = _get_fal_key()
    logger.info("Stream start: %d files, fal=%s", len(file_data), bool(fal_key))

    # ---- 2. Build the per-image queue, streaming as they finish ---------- #
    queue: asyncio.Queue = asyncio.Queue()
    for idx, (name, data) in enumerate(file_data):
        queue.put_nowait((idx, name, data))

    summary = {"total": len(file_data), "ok": 0, "errors": 0}

    async def worker() -> None:
        async def run_one(idx: int, name: str, data: bytes) -> None:
            result = await process_one_image(idx, name, data, fal_key)
            if result.get("status") == "ok":
                summary["ok"] += 1
            else:
                summary["errors"] += 1
            await queue.put(("item", result))

        # Run up to 5 images concurrently to FAL.AI
        sem = asyncio.Semaphore(5)

        async def sem_run(idx: int, name: str, data: bytes) -> None:
            async with sem:
                await run_one(idx, name, data)

        tasks = [
            asyncio.create_task(sem_run(idx, name, data))
            for idx, (name, data) in enumerate(file_data)
        ]
        await asyncio.gather(*tasks)
        await queue.put(("end", None))

    async def event_stream() -> AsyncGenerator[bytes, None]:
        worker_task = asyncio.create_task(worker())

        yield _sse_pack({
            "total": len(file_data),
            "fal_available": bool(fal_key),
        }, event="start").encode("utf-8")

        while True:
            kind, payload = await queue.get()
            if kind == "end":
                yield _sse_pack(summary, event="end").encode("utf-8")
                break
            yield _sse_pack(payload, event="item").encode("utf-8")

        try:
            await worker_task
        except Exception:
            pass

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/api/process-batch")
async def process_batch_endpoint(files: List[UploadFile] = File(default=[])):
    """Non-streaming variant for backwards compatibility."""
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")
    if len(files) > MAX_BATCH:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size exceeds the {MAX_BATCH} image limit.",
        )

    file_data: list[tuple[str, bytes]] = []
    for idx, file in enumerate(files):
        if not _is_valid_upload(file):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: '{file.filename}'.",
            )
        original_name = file.filename or f"image_{idx}"
        data = await file.read()
        if len(data) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File '{original_name}' exceeds the {MAX_FILE_SIZE // (1024*1024)} MB size limit.",
            )
        file_data.append((original_name, data))

    fal_key = _get_fal_key()
    sem = asyncio.Semaphore(5)

    async def run_with_sem(idx: int, name: str, data: bytes) -> dict:
        async with sem:
            return await process_one_image(idx, name, data, fal_key)

    tasks = [run_with_sem(i, n, d) for i, (n, d) in enumerate(file_data)]
    results = await asyncio.gather(*tasks)

    err = sum(1 for r in results if r.get("status") == "error")
    logger.info("Batch complete: %d items, errors=%d", len(results), err)

    return {"count": len(results), "items": results}


# --------------------------------------------------------------------------- #
# Static files
# --------------------------------------------------------------------------- #
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    logger.info("Serving frontend from: %s", FRONTEND_DIR)
else:
    logger.info("Frontend dist not found, running in API-only mode.")


# --------------------------------------------------------------------------- #
# Error handler
# --------------------------------------------------------------------------- #
@app.exception_handler(HTTPException)
async def http_exception_handler(_request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))