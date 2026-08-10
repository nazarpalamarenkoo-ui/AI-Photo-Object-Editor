import pytest

from tests.smoke.conftest import wait_until_picked_up

pytestmark = pytest.mark.gpu


async def test_worker_picks_up_job(authed_client, sample_image_bytes):
    upload_resp = await authed_client.post(
        "/images/upload",
        files={"file": ("smoke_worker.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert upload_resp.status_code == 201
    image_id = upload_resp.json()["id"]

    enqueue_resp = await authed_client.post(f"/ml/images/{image_id}/segment/async", json={})
    assert enqueue_resp.status_code == 200, enqueue_resp.text
    job_id = enqueue_resp.json()["job_id"]

    job = await wait_until_picked_up(authed_client, job_id)
    assert job["status"] in ("in_progress", "complete")