"""
Найцінніший smoke test у проєкті:
Upload -> DB record -> POST /ml/images/{id}/segment/async -> ARQ ->
MobileSAM worker -> GET /ml/jobs/{job_id} until complete -> segments.

SAM_DEVICE defaults to "cuda" in settings.py, and the worker warms up the
whole pipeline (get_pipeline()) on startup, so this needs the real GPU
worker container running — hence @pytest.mark.gpu.
"""
import pytest

from tests.smoke.conftest import wait_for_job

pytestmark = pytest.mark.gpu


async def test_segmentation_smoke(authed_client, sample_image_bytes):
    upload_resp = await authed_client.post(
        "/images/upload",
        files={"file": ("smoke_seg.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert upload_resp.status_code == 201
    image_id = upload_resp.json()["id"]

    enqueue_resp = await authed_client.post(f"/ml/images/{image_id}/segment/async", json={})
    assert enqueue_resp.status_code == 200, enqueue_resp.text
    job_id = enqueue_resp.json()["job_id"]

    job = await wait_for_job(authed_client, job_id)
    assert "error" not in job, job.get("error")
    result = job["result"]

    assert "segments" in result
    assert isinstance(result["segments"], list)