"""
Background removal using FAL.AI's pixelcut/background-removal model.

Uses asynchronous batch processing to handle multiple images efficiently.
"""
import os
import re
import asyncio
import base64
import httpx
import fal_client


# FAL.AI model endpoint
FAL_MODEL = "pixelcut/background-removal"

# data:[<mediatype>];base64,<data>  OR  data:[<mediatype>],<data>
_DATA_URL_RE = re.compile(r"^data:([^;,]+)(?:;base64)?,(.*)$", re.DOTALL)


def _decode_data_url(url: str) -> bytes:
    """Decode a data URL into raw bytes."""
    match = _DATA_URL_RE.match(url)
    if not match:
        raise RuntimeError(f"Unrecognized data URL format: {url[:80]}...")
    media_type, payload = match.group(1), match.group(2)
    if ";base64" in url:
        return base64.b64decode(payload)
    # URL-encoded data
    from urllib.parse import unquote
    return unquote(payload).encode("utf-8")


async def _download_or_decode(client: httpx.AsyncClient, url: str) -> bytes:
    """Return the raw bytes of an image, whether given as a data URL or HTTP URL."""
    if url.startswith("data:"):
        return _decode_data_url(url)
    response = await client.get(url, timeout=60.0)
    response.raise_for_status()
    return response.content


async def _process_single(
    client: httpx.AsyncClient,
    file_path: str,
    fal_key: str,
) -> bytes:
    """
    Upload one image to FAL CDN and call the pixelcut/background-removal model.

    Returns the raw PNG bytes of the processed (transparent) image.
    """
    # Configure the FAL key for the underlying async SDK calls
    os.environ["FAL_KEY"] = fal_key

    # Upload the local file to FAL's CDN asynchronously
    cdn_url = await fal_client.upload_file_async(file_path)

    # Call the model asynchronously
    result = await fal_client.subscribe_async(
        FAL_MODEL,
        arguments={"image_url": cdn_url},
    )

    image = (result or {}).get("image") or {}
    processed_url = image.get("url")
    if not processed_url:
        raise RuntimeError(f"FAL.AI response did not contain image URL. Response: {result}")

    # Download the processed image (handles data: URLs and http(s) URLs)
    return await _download_or_decode(client, processed_url)


async def process_batch(
    file_paths: list[str],
    fal_key: str,
    max_concurrency: int = 4,
) -> list[bytes]:
    """
    Process multiple images in parallel using FAL.AI's pixelcut/background-removal.

    Returns a list of raw PNG bytes, one per input file (same order).
    """
    if not fal_key:
        raise ValueError("FAL_KEY is not configured on the server.")

    if not file_paths:
        return []

    semaphore = asyncio.Semaphore(max_concurrency)
    async with httpx.AsyncClient() as client:
        async def _run(path: str) -> bytes:
            async with semaphore:
                return await _process_single(client, path, fal_key)

        tasks = [_run(p) for p in file_paths]
        return await asyncio.gather(*tasks)
