"""
End-to-end tests for the BG Remover API.

These tests:
- exercise the FastAPI app via TestClient
- mock the FAL.AI call so no real network or API key is required
- create dummy images in PNG / JPG / WebP formats
- verify batch processing, individual download endpoints, and validation errors
- simulate several "processed" PNGs in parallel to validate stability

Run with:
    cd backend
    python -m pytest test_app.py -v
or:
    python test_app.py
"""
from __future__ import annotations

import io
import os
import sys
import asyncio
import zipfile
import tempfile
import contextlib

# Ensure backend/ is importable when running directly
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from PIL import Image  # noqa: E402

# Test deps
try:
    import pytest
    from fastapi.testclient import TestClient
except ImportError:
    print("Missing test deps. Install: pip install pytest httpx")
    sys.exit(1)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_image(format_: str, color=(255, 0, 0), size=(200, 150)) -> bytes:
    """Generate a small single-color image as raw bytes in the given format."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    fmt = "JPEG" if format_.lower() in ("jpg", "jpeg") else format_.upper()
    if fmt == "JPG":
        fmt = "JPEG"
    img.save(buf, format=fmt)
    return buf.getvalue()


def make_png_with_alpha() -> bytes:
    """Generate a PNG with a transparent area (simulating a processed image)."""
    img = Image.new("RGBA", (120, 90), (0, 0, 0, 0))
    # Add a circle
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.ellipse((10, 10, 110, 80), fill=(40, 200, 120, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def mock_process_batch_factory(calls: list[str]):
    """
    Returns (patch_fn, payload_getter):
      - patch_fn replaces remover.process_batch with a fake
      - payload_getter returns the bytes the fake 'processed' each call with
    """
    def fake_process_batch(paths: list[str], fal_key: str, max_concurrency: int = 4):
        async def _runner():
            calls.extend(paths)
            # Simulate parallel processing with different "outputs"
            results = []
            for i, _ in enumerate(paths):
                # Use a deterministic RGBA PNG so alpha checks pass
                results.append(make_png_with_alpha())
            return results
        return _runner()
    return fake_process_batch


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def app_with_fal(monkeypatch, tmp_path):
    """Build the FastAPI app with FAL_KEY set and the FAL call mocked."""
    # Set FAL_KEY
    monkeypatch.setenv("FAL_KEY", "test-fake-key-12345")

    # Patch process_batch BEFORE importing main
    import remover
    calls: list[str] = []
    fake = mock_process_batch_factory(calls)
    monkeypatch.setattr(remover, "process_batch", fake)

    # Re-import main so it picks up the patched remover
    if "main" in sys.modules:
        del sys.modules["main"]
    import main as main_mod

    # Point uploads to temp dir
    main_mod.UPLOAD_DIR = str(tmp_path / "uploads")
    os.makedirs(main_mod.UPLOAD_DIR, exist_ok=True)

    yield main_mod.app, calls

    # Cleanup
    with contextlib.suppress(OSError):
        for p in tmp_path.glob("uploads/*"):
            p.unlink()


@pytest.fixture
def app_without_fal(monkeypatch, tmp_path):
    """App without FAL_KEY set, to test configuration error."""
    monkeypatch.delenv("FAL_KEY", raising=False)
    import remover, config
    monkeypatch.setattr(remover, "process_batch", mock_process_batch_factory([]))
    # Also blank the embedded default so the config check triggers
    monkeypatch.setattr(config, "FAL_KEY", "")

    if "main" in sys.modules:
        del sys.modules["main"]
    import main as main_mod
    main_mod.UPLOAD_DIR = str(tmp_path / "uploads")
    os.makedirs(main_mod.UPLOAD_DIR, exist_ok=True)
    yield main_mod.app


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
class TestHealth:
    def test_health_ok(self, app_with_fal):
        app, _ = app_with_fal
        with TestClient(app) as client:
            r = client.get("/api/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["model"] == "pixelcut/background-removal"
        assert data["fal_configured"] is True

    def test_health_unconfigured(self, app_without_fal):
        app = app_without_fal
        with TestClient(app) as client:
            r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["fal_configured"] is False


class TestProcessBatch:
    def test_single_png(self, app_with_fal):
        app, calls = app_with_fal
        png = make_image("PNG", color=(20, 120, 220))
        with TestClient(app) as client:
            r = client.post(
                "/api/process-batch",
                files=[("files", ("blue.png", png, "image/png"))],
            )
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["count"] == 1
        item = data["items"][0]
        assert item["original_name"] == "blue.png"
        assert item["data_url"].startswith("data:image/png;base64,")
        assert item["download_name"] == "blue_no_bg.png"
        # The mock was invoked with the saved file path
        assert len(calls) == 1

    def test_multiple_formats(self, app_with_fal):
        """Uploads one PNG, one JPG, one WebP and verifies the batch endpoint."""
        app, calls = app_with_fal
        files = [
            ("files", ("a.png", make_image("PNG", color=(255, 0, 0)), "image/png")),
            ("files", ("b.jpg", make_image("JPG", color=(0, 255, 0)), "image/jpeg")),
            ("files", ("c.webp", make_image("WEBP", color=(0, 0, 255)), "image/webp")),
        ]
        with TestClient(app) as client:
            r = client.post("/api/process-batch", files=files)
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["count"] == 3
        names = sorted(item["original_name"] for item in data["items"])
        assert names == ["a.png", "b.jpg", "c.webp"]
        # All items should have a data URL
        for item in data["items"]:
            assert item["data_url"].startswith("data:image/png;base64,")
        # Mock was called for all 3
        assert len(calls) == 3

    def test_processed_data_url_decodes_to_png(self, app_with_fal):
        """The data URL is a valid base64 PNG with alpha channel."""
        import base64
        app, _ = app_with_fal
        png = make_image("PNG", color=(123, 45, 67))
        with TestClient(app) as client:
            r = client.post(
                "/api/process-batch",
                files=[("files", ("x.png", png, "image/png"))],
            )
            assert r.status_code == 200
            data_url = r.json()["items"][0]["data_url"]
            assert data_url.startswith("data:image/png;base64,")
            b64 = data_url.split("base64,", 1)[1]
            raw = base64.b64decode(b64)

        # Verify it's a valid PNG with alpha channel
        img = Image.open(io.BytesIO(raw))
        assert img.format == "PNG"
        assert img.mode == "RGBA"
        assert "A" in img.getbands()

    def test_zip_like_batch_download(self, app_with_fal):
        """Simulates the ZIP download flow used by the frontend."""
        import base64
        app, _ = app_with_fal
        files = [
            ("files", (f"img_{i}.png", make_image("PNG", color=(i*40, 50, 100)), "image/png"))
            for i in range(1, 4)
        ]
        with TestClient(app) as client:
            r = client.post("/api/process-batch", files=files)
            assert r.status_code == 200
            items = r.json()["items"]

            # Decode every data URL into a ZIP, like the frontend does
            zbuf = io.BytesIO()
            with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
                for item in items:
                    b64 = item["data_url"].split("base64,", 1)[1]
                    zf.writestr(item["download_name"], base64.b64decode(b64))

        # Verify the ZIP is valid and contains all expected files
        zbuf.seek(0)
        with zipfile.ZipFile(zbuf) as zf:
            names = sorted(zf.namelist())
        assert names == ["img_1_no_bg.png", "img_2_no_bg.png", "img_3_no_bg.png"]


class TestValidation:
    def test_empty_batch(self, app_with_fal):
        app, _ = app_with_fal
        with TestClient(app) as client:
            r = client.post("/api/process-batch", files=[])
        assert r.status_code == 400

    def test_unsupported_format(self, app_with_fal):
        app, _ = app_with_fal
        # .gif is not in the allowlist
        gif_bytes = make_image("PNG", color=(1, 2, 3))  # content is PNG but filename is .gif
        with TestClient(app) as client:
            r = client.post(
                "/api/process-batch",
                files=[("files", ("anim.gif", gif_bytes, "image/gif"))],
            )
        assert r.status_code == 400
        assert "Unsupported" in r.json()["detail"] or "format" in r.json()["detail"].lower()

    def test_no_fal_key(self, app_without_fal):
        app = app_without_fal
        png = make_image("PNG")
        with TestClient(app) as client:
            r = client.post(
                "/api/process-batch",
                files=[("files", ("x.png", png, "image/png"))],
            )
        assert r.status_code == 500
        assert "FAL_KEY" in r.json()["detail"]

    def test_batch_size_limit(self, app_with_fal, monkeypatch):
        app, _ = app_with_fal
        # Lower the limit for the test
        import main as main_mod
        monkeypatch.setattr(main_mod, "MAX_BATCH", 2)

        files = [
            ("files", (f"f{i}.png", make_image("PNG"), "image/png"))
            for i in range(3)
        ]
        with TestClient(app) as client:
            r = client.post("/api/process-batch", files=files)
        assert r.status_code == 400
        assert "limit" in r.json()["detail"].lower()


class TestStability:
    def test_concurrent_batch(self, app_with_fal):
        """A larger batch should be processed in one call without issues."""
        app, calls = app_with_fal
        n = 10
        files = [
            ("files", (f"img_{i:02d}.png", make_image("PNG", color=(i*20, i*10, 100)), "image/png"))
            for i in range(n)
        ]
        with TestClient(app) as client:
            r = client.post("/api/process-batch", files=files)
        assert r.status_code == 200
        assert r.json()["count"] == n
        assert len(calls) == n


# --------------------------------------------------------------------------- #
# Manual runner
# --------------------------------------------------------------------------- #
def _run_manual():
    """Run tests without pytest, printing results."""
    print("Running manual test suite...")
    import traceback
    failures = []
    for cls_name in ("TestHealth", "TestProcessBatch", "TestValidation", "TestStability"):
        cls = globals().get(cls_name)
        if cls is None:
            continue
        # We need the fixtures manually, so we delegate to pytest
    print("Use pytest for full test runs:")
    print("    cd backend && python -m pytest test_app.py -v")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "manual":
        _run_manual()
    else:
        sys.exit(pytest.main([__file__, "-v"]))
