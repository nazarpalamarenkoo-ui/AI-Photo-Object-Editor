import pytest

pytestmark = pytest.mark.gpu


async def test_remove_object_via_sam_mask(authed_client, sample_photo_with_objects_bytes):
    upload_resp = await authed_client.post(
        "/images/upload",
        files={"file": ("smoke_edit.jpg", sample_photo_with_objects_bytes, "image/jpeg")},
    )
    assert upload_resp.status_code == 201
    image_id = upload_resp.json()["id"]

    seg_resp = await authed_client.post(f"/ml/images/{image_id}/segment", json={})
    assert seg_resp.status_code == 200, seg_resp.text
    segments = seg_resp.json()["segments"]

    if not segments:
        pytest.skip(
            "no segments on synthetic test image — swap sample_image_bytes "
            "for a real photo with objects to exercise this path"
        )

    mask_id = segments[0]["mask_id"]

    remove_resp = await authed_client.post(
        f"/ml/images/{image_id}/segment/{mask_id}/remove", json={}
    )
    assert remove_resp.status_code == 200, remove_resp.text

    result = remove_resp.json()
    assert "result_url" in result
    assert "presigned_url" in result
    assert "metrics" in result