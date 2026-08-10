import pytest

pytestmark = pytest.mark.smoke


async def test_detection_smoke(authed_client, sample_image_bytes):
    upload_resp = await authed_client.post(
        "/images/upload",
        files={"file": ("smoke_detect.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert upload_resp.status_code == 201
    image_id = upload_resp.json()["id"]

    detect_resp = await authed_client.post(f"/ml/images/{image_id}/detect", json={})
    assert detect_resp.status_code == 200, detect_resp.text

    persisted_resp = await authed_client.get(f"/detections/images/{image_id}")
    assert persisted_resp.status_code == 200, persisted_resp.text
    assert isinstance(persisted_resp.json(), list)


async def test_supported_classes_endpoint(authed_client):
    resp = await authed_client.get("/ml/classes")
    assert resp.status_code == 200
    classes = resp.json()
    assert isinstance(classes, list)
    assert len(classes) > 0