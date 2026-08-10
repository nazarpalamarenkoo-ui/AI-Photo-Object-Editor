import pytest

pytestmark = pytest.mark.smoke


async def test_upload_creates_record(authed_client, sample_image_bytes):
    resp = await authed_client.post(
        "/images/upload",
        files={"file": ("smoke.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert resp.status_code == 201, resp.text

    body = resp.json()
    assert "id" in body
    assert body["filename"] == "smoke.jpg"
    assert "uploaded_at" in body

    get_resp = await authed_client.get(f"/images/{body['id']}")
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == body["id"]


async def test_list_user_images_includes_uploaded(authed_client, sample_image_bytes):
    upload_resp = await authed_client.post(
        "/images/upload",
        files={"file": ("smoke_list.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert upload_resp.status_code == 201
    image_id = upload_resp.json()["id"]

    list_resp = await authed_client.get("/images/")
    assert list_resp.status_code == 200
    ids = [img["id"] for img in list_resp.json()]
    assert image_id in ids