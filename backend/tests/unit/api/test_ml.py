import pytest
from unittest.mock import MagicMock, AsyncMock
from fastapi import HTTPException, UploadFile

from app.api.v1.ml import (
    detect_objects,
    get_supported_classes,
    remove_object,
    remove_multiple_objects,
    replace_object,
    reset_current_state,
    save_result,
    undo,
    redo,
    get_history,
    segment_objects,
    segment_with_prompt,
    sam_remove_object,
    sam_replace_object,
    extract_object,
    paste_extracted_object,
    _http_status,
)
from app.db.schemas.ml import (
    BboxSchema,
    DetectRequest,
    RemoveRequest,
    RemoveMultipleRequest,
    ReplaceRequest,
    SegmentRequest,
    SegmentWithPromptRequest,
    SamRemoveRequest,
    SamReplaceRequest,
    ExtractRequest,
    PasteRequest,
)

@pytest.fixture
def mock_user():
    user = MagicMock()
    user.id = 1
    return user


@pytest.fixture
def mock_detector_service():
    service = MagicMock()
    service.detect_objects = AsyncMock()
    service.get_supported_classes = MagicMock()
    return service


@pytest.fixture
def mock_editor_service():
    service = MagicMock()
    service.remove_object = AsyncMock()
    service.remove_multiple_objects = AsyncMock()
    service.replace_object = AsyncMock()
    service._get_image_authorized = AsyncMock()
    service.reset_current_state = AsyncMock()
    service.save_result = AsyncMock()
    service.undo = AsyncMock()
    service.redo = AsyncMock()
    service.get_history = AsyncMock()
    return service


@pytest.fixture
def mock_segmentation_service():
    service = MagicMock()
    service.segment_objects = AsyncMock()
    service.segment_with_prompt = AsyncMock()
    service.sam_remove_object = AsyncMock()
    service.sam_replace_object = AsyncMock()
    return service


@pytest.fixture
def mock_asset_service():
    service = MagicMock()
    service.extract_object = AsyncMock()
    service.paste_extracted_object = AsyncMock()
    service.get_asset_image = AsyncMock()
    return service


@pytest.fixture
def mock_file():
    file = MagicMock(spec=UploadFile)
    file.read = AsyncMock(return_value=b"image-bytes")
    return file


@pytest.mark.unit
class TestHttpStatusHelper:
    def test_not_found_maps_to_404(self):
        assert _http_status(ValueError("Image not found")) == 404

    def test_no_valid_detections_maps_to_404(self):
        assert _http_status(ValueError("no valid detections")) == 404

    def test_unauthorized_maps_to_403(self):
        assert _http_status(ValueError("Unauthorized access")) == 403

    def test_generic_error_maps_to_400(self):
        assert _http_status(ValueError("bad confidence threshold")) == 400

    def test_case_insensitive_matching(self):
        assert _http_status(ValueError("IMAGE NOT FOUND")) == 404
        assert _http_status(ValueError("UNAUTHORIZED")) == 403


@pytest.mark.unit
@pytest.mark.asyncio
class TestDetectObjects:
    async def test_success(self, mock_user, mock_detector_service):
        mock_detector_service.detect_objects.return_value = {"detections": []}
        body = DetectRequest(conf_threshold=0.6, classes=["person"])

        result = await detect_objects(
            image_id=1, body=body, current_user=mock_user, service=mock_detector_service
        )

        mock_detector_service.detect_objects.assert_called_once_with(
            image_id=1, user_id=1, conf_threshold=0.6, classes=["person"]
        )
        assert result == {"detections": []}

    async def test_default_body(self, mock_user, mock_detector_service):
        mock_detector_service.detect_objects.return_value = {"detections": []}

        await detect_objects(
            image_id=1, current_user=mock_user, service=mock_detector_service
        )

        mock_detector_service.detect_objects.assert_called_once_with(
            image_id=1, user_id=1, conf_threshold=0.5, classes=None
        )

    async def test_not_found(self, mock_user, mock_detector_service):
        mock_detector_service.detect_objects.side_effect = ValueError("Image not found")

        with pytest.raises(HTTPException) as exc:
            await detect_objects(image_id=1, current_user=mock_user, service=mock_detector_service)

        assert exc.value.status_code == 404

    async def test_generic_error_returns_400(self, mock_user, mock_detector_service):
        mock_detector_service.detect_objects.side_effect = ValueError("bad confidence threshold")

        with pytest.raises(HTTPException) as exc:
            await detect_objects(image_id=1, current_user=mock_user, service=mock_detector_service)

        assert exc.value.status_code == 400


@pytest.mark.unit
@pytest.mark.asyncio
class TestGetSupportedClasses:
    async def test_success(self, mock_user, mock_detector_service):
        mock_detector_service.get_supported_classes.return_value = ["person", "car"]

        result = await get_supported_classes(current_user=mock_user, service=mock_detector_service)

        assert result == ["person", "car"]
        mock_detector_service.get_supported_classes.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
class TestRemoveObject:
    async def test_success_default_body(self, mock_user, mock_editor_service):
        # RemoveRequest defaults: expand_mask_pixels=5, use_edge_blending=False
        mock_editor_service.remove_object.return_value = {"result_url": "s3://out.jpg"}

        result = await remove_object(
            image_id=1, bbox_id=2, current_user=mock_user, service=mock_editor_service
        )

        mock_editor_service.remove_object.assert_called_once_with(
            image_id=1,
            bbox_id=2,
            user_id=1,
            expand_mask_pixels=5,
            use_edge_blending=False,
            ldm_steps=25,
            ldm_sampler='plms',
            hd_strategy='CROP',
        )
        assert result["result_url"] == "s3://out.jpg"

    async def test_custom_body(self, mock_user, mock_editor_service):
        from app.db.schemas.ml import LdmConfig

        body = RemoveRequest(
            expand_mask_pixels=20,
            use_edge_blending=False,
            ldm=LdmConfig(ldm_steps=40, ldm_sampler="ddim", hd_strategy="RESIZE"),
        )

        await remove_object(image_id=1, bbox_id=2, body=body, current_user=mock_user, service=mock_editor_service)

        mock_editor_service.remove_object.assert_called_once_with(
            image_id=1,
            bbox_id=2,
            user_id=1,
            expand_mask_pixels=20,
            use_edge_blending=False,
            ldm_steps=40,
            ldm_sampler="ddim",
            hd_strategy="RESIZE",
        )

    async def test_not_found(self, mock_user, mock_editor_service):
        mock_editor_service.remove_object.side_effect = ValueError("bbox not found")

        with pytest.raises(HTTPException) as exc:
            await remove_object(image_id=1, bbox_id=2, current_user=mock_user, service=mock_editor_service)

        assert exc.value.status_code == 404

    async def test_unauthorized(self, mock_user, mock_editor_service):
        mock_editor_service.remove_object.side_effect = ValueError("unauthorized")

        with pytest.raises(HTTPException) as exc:
            await remove_object(image_id=1, bbox_id=2, current_user=mock_user, service=mock_editor_service)

        assert exc.value.status_code == 403

    async def test_generic_error(self, mock_user, mock_editor_service):
        mock_editor_service.remove_object.side_effect = ValueError("mask generation failed")

        with pytest.raises(HTTPException) as exc:
            await remove_object(image_id=1, bbox_id=2, current_user=mock_user, service=mock_editor_service)

        assert exc.value.status_code == 400


@pytest.mark.unit
@pytest.mark.asyncio
class TestRemoveMultipleObjects:
    async def test_success(self, mock_user, mock_editor_service):
        # RemoveMultipleRequest defaults: expand_mask_pixels=5, use_edge_blending=False
        mock_editor_service.remove_multiple_objects.return_value = {"result_url": "s3://out.jpg"}
        body = RemoveMultipleRequest(bbox_ids=[1, 2, 3])

        result = await remove_multiple_objects(
            image_id=1, body=body, current_user=mock_user, service=mock_editor_service
        )

        mock_editor_service.remove_multiple_objects.assert_called_once_with(
            image_id=1,
            bbox_ids=[1, 2, 3],
            user_id=1,
            expand_mask_pixels=5,
            use_edge_blending=False,
            ldm_steps=25,
            ldm_sampler='plms',
            hd_strategy='CROP',
        )
        assert result["result_url"] == "s3://out.jpg"

    async def test_unauthorized(self, mock_user, mock_editor_service):
        mock_editor_service.remove_multiple_objects.side_effect = ValueError("unauthorized")
        body = RemoveMultipleRequest(bbox_ids=[1, 2])

        with pytest.raises(HTTPException) as exc:
            await remove_multiple_objects(image_id=1, body=body, current_user=mock_user, service=mock_editor_service)

        assert exc.value.status_code == 403

    async def test_generic_error(self, mock_user, mock_editor_service):
        mock_editor_service.remove_multiple_objects.side_effect = ValueError("no overlapping masks")
        body = RemoveMultipleRequest(bbox_ids=[1, 2])

        with pytest.raises(HTTPException) as exc:
            await remove_multiple_objects(image_id=1, body=body, current_user=mock_user, service=mock_editor_service)

        assert exc.value.status_code == 400


@pytest.mark.unit
@pytest.mark.asyncio
class TestReplaceObject:
    async def test_success_full_params(self, mock_user, mock_editor_service, mock_file):
        mock_editor_service.replace_object.return_value = {"result": "ok"}
        body = ReplaceRequest(
            expand_mask_pixels=15,
            use_color_matching=True,
            use_edge_blending=True,
            color_match_method="histogram",
        )

        result = await replace_object(
            image_id=1,
            bbox_id=2,
            replacement_file=mock_file,
            body=body,
            current_user=mock_user,
            service=mock_editor_service,
        )

        mock_editor_service.replace_object.assert_called_once_with(
            image_id=1,
            bbox_id=2,
            replace_image_bytes=b"image-bytes",
            user_id=1,
            expand_mask_pixels=15,
            use_color_matching=True,
            use_edge_blending=True,
            color_match_method="histogram",
            ldm_steps=25,
            ldm_sampler='plms',
            hd_strategy='CROP',
        )
        assert result["result"] == "ok"

    async def test_unauthorized(self, mock_user, mock_editor_service, mock_file):
        mock_editor_service.replace_object.side_effect = ValueError("unauthorized access")
        body = ReplaceRequest()

        with pytest.raises(HTTPException) as exc:
            await replace_object(
                image_id=1,
                bbox_id=2,
                replacement_file=mock_file,
                body=body,
                current_user=mock_user,
                service=mock_editor_service,
            )

        assert exc.value.status_code == 403

    async def test_generic_value_error_returns_400(self, mock_user, mock_editor_service, mock_file):
        mock_editor_service.replace_object.side_effect = ValueError("invalid color match method")
        body = ReplaceRequest()

        with pytest.raises(HTTPException) as exc:
            await replace_object(
                image_id=1,
                bbox_id=2,
                replacement_file=mock_file,
                body=body,
                current_user=mock_user,
                service=mock_editor_service,
            )

        assert exc.value.status_code == 400


@pytest.mark.unit
@pytest.mark.asyncio
class TestResetCurrentState:
    async def test_success(self, mock_user, mock_editor_service):
        result = await reset_current_state(image_id=1, current_user=mock_user, service=mock_editor_service)

        mock_editor_service._get_image_authorized.assert_awaited_once_with(1, 1)
        mock_editor_service.reset_current_state.assert_awaited_once_with(1)
        assert result == {"detail": "State reset to original image"}

    async def test_not_found(self, mock_user, mock_editor_service):
        mock_editor_service._get_image_authorized.side_effect = ValueError("not found")

        with pytest.raises(HTTPException) as exc:
            await reset_current_state(image_id=1, current_user=mock_user, service=mock_editor_service)

        assert exc.value.status_code == 404

    async def test_generic_error(self, mock_user, mock_editor_service):
        mock_editor_service._get_image_authorized.side_effect = ValueError("corrupted state")

        with pytest.raises(HTTPException) as exc:
            await reset_current_state(image_id=1, current_user=mock_user, service=mock_editor_service)

        assert exc.value.status_code == 400


@pytest.mark.unit
@pytest.mark.asyncio
class TestSaveResult:
    async def test_success(self, mock_user, mock_editor_service):
        mock_editor_service.save_result.return_value = {"id": 42, "filename": "result.jpg"}

        result = await save_result(image_id=1, current_user=mock_user, service=mock_editor_service)

        mock_editor_service.save_result.assert_called_once_with(image_id=1, user_id=1)
        assert result["id"] == 42

    async def test_not_found(self, mock_user, mock_editor_service):
        mock_editor_service.save_result.side_effect = ValueError("image not found")

        with pytest.raises(HTTPException) as exc:
            await save_result(image_id=1, current_user=mock_user, service=mock_editor_service)

        assert exc.value.status_code == 404

    async def test_unauthorized(self, mock_user, mock_editor_service):
        mock_editor_service.save_result.side_effect = ValueError("unauthorized")

        with pytest.raises(HTTPException) as exc:
            await save_result(image_id=1, current_user=mock_user, service=mock_editor_service)

        assert exc.value.status_code == 403

    async def test_generic_error(self, mock_user, mock_editor_service):
        mock_editor_service.save_result.side_effect = ValueError("no processed state to save")

        with pytest.raises(HTTPException) as exc:
            await save_result(image_id=1, current_user=mock_user, service=mock_editor_service)

        assert exc.value.status_code == 400


@pytest.mark.unit
@pytest.mark.asyncio
class TestUndo:
    async def test_success(self, mock_user, mock_editor_service):
        mock_editor_service.undo.return_value = {"detail": "Undone"}

        result = await undo(image_id=1, current_user=mock_user, service=mock_editor_service)

        mock_editor_service.undo.assert_called_once_with(1, 1)
        assert result["detail"] == "Undone"

    async def test_not_found(self, mock_user, mock_editor_service):
        mock_editor_service.undo.side_effect = ValueError("not found")

        with pytest.raises(HTTPException) as exc:
            await undo(image_id=1, current_user=mock_user, service=mock_editor_service)

        assert exc.value.status_code == 404

    async def test_unauthorized(self, mock_user, mock_editor_service):
        mock_editor_service.undo.side_effect = ValueError("unauthorized")

        with pytest.raises(HTTPException) as exc:
            await undo(image_id=1, current_user=mock_user, service=mock_editor_service)

        assert exc.value.status_code == 403

    async def test_nothing_to_undo(self, mock_user, mock_editor_service):
        mock_editor_service.undo.side_effect = ValueError("nothing to undo")

        with pytest.raises(HTTPException) as exc:
            await undo(image_id=1, current_user=mock_user, service=mock_editor_service)

        assert exc.value.status_code == 400


@pytest.mark.unit
@pytest.mark.asyncio
class TestRedo:
    async def test_success(self, mock_user, mock_editor_service):
        mock_editor_service.redo.return_value = {"detail": "Redone"}

        result = await redo(image_id=1, current_user=mock_user, service=mock_editor_service)

        mock_editor_service.redo.assert_called_once_with(1, 1)
        assert result["detail"] == "Redone"

    async def test_not_found(self, mock_user, mock_editor_service):
        mock_editor_service.redo.side_effect = ValueError("not found")

        with pytest.raises(HTTPException) as exc:
            await redo(image_id=1, current_user=mock_user, service=mock_editor_service)

        assert exc.value.status_code == 404

    async def test_unauthorized(self, mock_user, mock_editor_service):
        mock_editor_service.redo.side_effect = ValueError("unauthorized")

        with pytest.raises(HTTPException) as exc:
            await redo(image_id=1, current_user=mock_user, service=mock_editor_service)

        assert exc.value.status_code == 403

    async def test_nothing_to_redo(self, mock_user, mock_editor_service):
        mock_editor_service.redo.side_effect = ValueError("nothing to redo")

        with pytest.raises(HTTPException) as exc:
            await redo(image_id=1, current_user=mock_user, service=mock_editor_service)

        assert exc.value.status_code == 400


@pytest.mark.unit
@pytest.mark.asyncio
class TestGetHistory:
    async def test_success(self, mock_user, mock_editor_service):
        mock_editor_service.get_history.return_value = {
            "current_index": 1,
            "states": ["original.jpg", "edited.jpg"],
        }

        result = await get_history(image_id=1, current_user=mock_user, service=mock_editor_service)

        mock_editor_service.get_history.assert_called_once_with(1, 1)
        assert result["current_index"] == 1
        assert len(result["states"]) == 2

    async def test_not_found(self, mock_user, mock_editor_service):
        mock_editor_service.get_history.side_effect = ValueError("not found")

        with pytest.raises(HTTPException) as exc:
            await get_history(image_id=1, current_user=mock_user, service=mock_editor_service)

        assert exc.value.status_code == 404

    async def test_unauthorized(self, mock_user, mock_editor_service):
        mock_editor_service.get_history.side_effect = ValueError("unauthorized")

        with pytest.raises(HTTPException) as exc:
            await get_history(image_id=1, current_user=mock_user, service=mock_editor_service)

        assert exc.value.status_code == 403


@pytest.mark.unit
@pytest.mark.asyncio
class TestSegmentObjects:
    async def test_success_default_body(self, mock_user, mock_segmentation_service):
        mock_segmentation_service.segment_objects.return_value = {"segments": []}

        result = await segment_objects(
            image_id=1, current_user=mock_user, service=mock_segmentation_service
        )

        mock_segmentation_service.segment_objects.assert_called_once_with(
            image_id=1, user_id=1, min_area=500, max_segments=50
        )
        assert result == {"segments": []}

    async def test_custom_body(self, mock_user, mock_segmentation_service):
        body = SegmentRequest(min_area=100, max_segments=10)

        await segment_objects(image_id=1, body=body, current_user=mock_user, service=mock_segmentation_service)

        mock_segmentation_service.segment_objects.assert_called_once_with(
            image_id=1, user_id=1, min_area=100, max_segments=10
        )

    async def test_not_found(self, mock_user, mock_segmentation_service):
        mock_segmentation_service.segment_objects.side_effect = ValueError("image not found")

        with pytest.raises(HTTPException) as exc:
            await segment_objects(image_id=1, current_user=mock_user, service=mock_segmentation_service)

        assert exc.value.status_code == 404

    async def test_generic_error(self, mock_user, mock_segmentation_service):
        mock_segmentation_service.segment_objects.side_effect = ValueError("SAM model failed")

        with pytest.raises(HTTPException) as exc:
            await segment_objects(image_id=1, current_user=mock_user, service=mock_segmentation_service)

        assert exc.value.status_code == 400


@pytest.mark.unit
@pytest.mark.asyncio
class TestSegmentWithPrompt:
    async def test_success_with_points(self, mock_user, mock_segmentation_service):
        # route always forwards multimask_output (default None) alongside bbox
        mock_segmentation_service.segment_with_prompt.return_value = {"segments": []}
        body = SegmentWithPromptRequest(point_coords=[(10, 20)], point_labels=[1])

        result = await segment_with_prompt(
            image_id=1, body=body, current_user=mock_user, service=mock_segmentation_service
        )

        mock_segmentation_service.segment_with_prompt.assert_called_once_with(
            image_id=1,
            user_id=1,
            point_coords=[(10, 20)],
            point_labels=[1],
            bbox=None,
            multimask_output=None,
        )
        assert result == {"segments": []}

    async def test_success_with_bbox(self, mock_user, mock_segmentation_service):
        mock_segmentation_service.segment_with_prompt.return_value = {"segments": []}
        bbox = BboxSchema(x1=0, y1=0, x2=50, y2=50)
        body = SegmentWithPromptRequest(bbox=bbox)

        await segment_with_prompt(
            image_id=1, body=body, current_user=mock_user, service=mock_segmentation_service
        )

        _, kwargs = mock_segmentation_service.segment_with_prompt.call_args
        assert kwargs["bbox"] == bbox.model_dump()
        assert kwargs["multimask_output"] is None

    async def test_success_with_multimask_output(self, mock_user, mock_segmentation_service):
        mock_segmentation_service.segment_with_prompt.return_value = {"segments": []}
        body = SegmentWithPromptRequest(multimask_output=True)

        await segment_with_prompt(
            image_id=1, body=body, current_user=mock_user, service=mock_segmentation_service
        )

        _, kwargs = mock_segmentation_service.segment_with_prompt.call_args
        assert kwargs["multimask_output"] is True

    async def test_not_found(self, mock_user, mock_segmentation_service):
        mock_segmentation_service.segment_with_prompt.side_effect = ValueError("mask not found")
        body = SegmentWithPromptRequest()

        with pytest.raises(HTTPException) as exc:
            await segment_with_prompt(image_id=1, body=body, current_user=mock_user, service=mock_segmentation_service)

        assert exc.value.status_code == 404


@pytest.mark.unit
@pytest.mark.asyncio
class TestSamRemoveObject:
    async def test_success_default_body(self, mock_user, mock_segmentation_service):
        # SamRemoveRequest defaults: expand_mask_pixels=12, use_edge_blending=False
        mock_segmentation_service.sam_remove_object.return_value = {"result_url": "s3://out.jpg"}

        result = await sam_remove_object(
            image_id=1, mask_id=3, current_user=mock_user, service=mock_segmentation_service
        )

        mock_segmentation_service.sam_remove_object.assert_called_once_with(
            image_id=1,
            mask_id=3,
            user_id=1,
            expand_mask_pixels=12,
            use_edge_blending=False,
            ldm_steps=25,
            ldm_sampler='plms',
            hd_strategy='CROP',
        )
        assert result["result_url"] == "s3://out.jpg"

    async def test_not_found(self, mock_user, mock_segmentation_service):
        mock_segmentation_service.sam_remove_object.side_effect = ValueError("mask not found")

        with pytest.raises(HTTPException) as exc:
            await sam_remove_object(image_id=1, mask_id=3, current_user=mock_user, service=mock_segmentation_service)

        assert exc.value.status_code == 404

    async def test_unauthorized(self, mock_user, mock_segmentation_service):
        mock_segmentation_service.sam_remove_object.side_effect = ValueError("unauthorized")

        with pytest.raises(HTTPException) as exc:
            await sam_remove_object(image_id=1, mask_id=3, current_user=mock_user, service=mock_segmentation_service)

        assert exc.value.status_code == 403


@pytest.mark.unit
@pytest.mark.asyncio
class TestSamReplaceObject:
    async def test_success_default_body_with_file(self, mock_user, mock_segmentation_service, mock_file):
        # SamReplaceRequest defaults: expand_mask_pixels=8, use_color_matching=False,
        # use_edge_blending=False, color_match_method='color_transfer'
        mock_segmentation_service.sam_replace_object.return_value = {"result_url": "s3://out.jpg"}
        body = SamReplaceRequest()

        result = await sam_replace_object(
            image_id=1,
            mask_id=3,
            replacement_file=mock_file,
            asset_id=None,
            body=body,
            current_user=mock_user,
            service=mock_segmentation_service,
            asset_service=MagicMock(),
        )

        mock_segmentation_service.sam_replace_object.assert_called_once_with(
            image_id=1,
            mask_id=3,
            replacement_image_bytes=b"image-bytes",
            user_id=1,
            expand_mask_pixels=8,
            use_color_matching=False,
            use_edge_blending=False,
            color_match_method='color_transfer',
            ldm_steps=25,
            ldm_sampler='plms',
            hd_strategy='CROP',
            replacement_is_cutout=False,
        )
        assert result["result_url"] == "s3://out.jpg"

    async def test_success_with_asset_id(self, mock_user, mock_segmentation_service):
        mock_segmentation_service.sam_replace_object.return_value = {"result_url": "s3://out.jpg"}
        mock_asset_service = MagicMock()
        mock_asset_service.get_asset_image = AsyncMock(return_value=b"asset-bytes")
        body = SamReplaceRequest()

        result = await sam_replace_object(
            image_id=1,
            mask_id=3,
            replacement_file=None,
            asset_id="asset-1",
            body=body,
            current_user=mock_user,
            service=mock_segmentation_service,
            asset_service=mock_asset_service,
        )

        mock_asset_service.get_asset_image.assert_awaited_once_with(1, "asset-1")
        _, kwargs = mock_segmentation_service.sam_replace_object.call_args
        assert kwargs["replacement_image_bytes"] == b"asset-bytes"
        assert kwargs["replacement_is_cutout"] is True
        assert result["result_url"] == "s3://out.jpg"

    async def test_asset_not_found_returns_404(self, mock_user, mock_segmentation_service):
        mock_asset_service = MagicMock()
        mock_asset_service.get_asset_image = AsyncMock(return_value=None)
        body = SamReplaceRequest()

        with pytest.raises(HTTPException) as exc:
            await sam_replace_object(
                image_id=1,
                mask_id=3,
                replacement_file=None,
                asset_id="missing-asset",
                body=body,
                current_user=mock_user,
                service=mock_segmentation_service,
                asset_service=mock_asset_service,
            )

        assert exc.value.status_code == 404

    async def test_missing_file_and_asset_id_returns_400(self, mock_user, mock_segmentation_service):
        body = SamReplaceRequest()

        with pytest.raises(HTTPException) as exc:
            await sam_replace_object(
                image_id=1,
                mask_id=3,
                replacement_file=None,
                asset_id=None,
                body=body,
                current_user=mock_user,
                service=mock_segmentation_service,
                asset_service=MagicMock(),
            )

        assert exc.value.status_code == 400
        assert exc.value.detail == "Provide replacement_file or asset_id"

    async def test_generic_error(self, mock_user, mock_segmentation_service, mock_file):
        mock_segmentation_service.sam_replace_object.side_effect = ValueError("blending failed")
        body = SamReplaceRequest()

        with pytest.raises(HTTPException) as exc:
            await sam_replace_object(
                image_id=1,
                mask_id=3,
                replacement_file=mock_file,
                asset_id=None,
                body=body,
                current_user=mock_user,
                service=mock_segmentation_service,
                asset_service=MagicMock(),
            )

        assert exc.value.status_code == 400


@pytest.mark.unit
@pytest.mark.asyncio
class TestExtractObject:
    async def test_success_default_body(self, mock_user, mock_asset_service):
        # ExtractResponse.asset_id is required, so the service must return it
        mock_asset_service.extract_object.return_value = {
            "asset_id": "asset-1",
            "extracted_url": "s3://obj.png",
            "presigned_url": "https://presigned.url/obj.png",
            "object_size": (50, 60),
            "area_pixels": 3000,
            "cropped_bbox": {"x1": 0, "y1": 0, "x2": 50, "y2": 60},
            "timestamp": "2025-01-01T00:00:00",
        }

        result = await extract_object(
            image_id=1, mask_id=3, current_user=mock_user, service=mock_asset_service
        )

        mock_asset_service.extract_object.assert_called_once_with(
            image_id=1, mask_id=3, user_id=1, padding_pixels=8, label=None, persist_to_s3=False
        )
        assert result["asset_id"] == "asset-1"
        assert result["extracted_url"] == "s3://obj.png"

    async def test_custom_padding(self, mock_user, mock_asset_service):
        mock_asset_service.extract_object.return_value = {"asset_id": "asset-2"}
        body = ExtractRequest(padding_pixels=20)

        await extract_object(image_id=1, mask_id=3, body=body, current_user=mock_user, service=mock_asset_service)

        mock_asset_service.extract_object.assert_called_once_with(
            image_id=1, mask_id=3, user_id=1, padding_pixels=20, label=None, persist_to_s3=False
        )

    async def test_custom_label_and_persist_to_s3(self, mock_user, mock_asset_service):
        mock_asset_service.extract_object.return_value = {"asset_id": "asset-3"}
        body = ExtractRequest(padding_pixels=10, label="my-object", persist_to_s3=True)

        await extract_object(image_id=1, mask_id=3, body=body, current_user=mock_user, service=mock_asset_service)

        mock_asset_service.extract_object.assert_called_once_with(
            image_id=1, mask_id=3, user_id=1, padding_pixels=10, label="my-object", persist_to_s3=True
        )

    async def test_not_found(self, mock_user, mock_asset_service):
        mock_asset_service.extract_object.side_effect = ValueError("mask not found")

        with pytest.raises(HTTPException) as exc:
            await extract_object(image_id=1, mask_id=3, current_user=mock_user, service=mock_asset_service)

        assert exc.value.status_code == 404


@pytest.mark.unit
@pytest.mark.asyncio
class TestPasteExtractedObject:
    async def test_success_with_extracted_url(self, mock_user, mock_asset_service):
        mock_asset_service.paste_extracted_object.return_value = {"result_url": "s3://pasted.jpg"}
        target_bbox = BboxSchema(x1=0, y1=0, x2=50, y2=50)
        body = PasteRequest(
            extracted_url="s3://obj.png",
            target_bbox=target_bbox,
            scale=1.5,
            use_color_matching=False,
            use_edge_blending=True,
            color_match_method="histogram",
        )

        result = await paste_extracted_object(
            image_id=1, body=body, current_user=mock_user, service=mock_asset_service
        )

        mock_asset_service.paste_extracted_object.assert_called_once_with(
            image_id=1,
            user_id=1,
            asset_id=None,
            extracted_url="s3://obj.png",
            target_bbox=target_bbox.model_dump(),
            scale=1.5,
            use_color_matching=False,
            use_edge_blending=True,
            color_match_method="histogram",
        )
        assert result["result_url"] == "s3://pasted.jpg"

    async def test_success_with_asset_id(self, mock_user, mock_asset_service):
        mock_asset_service.paste_extracted_object.return_value = {"result_url": "s3://pasted.jpg"}
        target_bbox = BboxSchema(x1=0, y1=0, x2=10, y2=10)
        body = PasteRequest(asset_id="asset-1", target_bbox=target_bbox)

        await paste_extracted_object(
            image_id=1, body=body, current_user=mock_user, service=mock_asset_service
        )

        _, kwargs = mock_asset_service.paste_extracted_object.call_args
        assert kwargs["asset_id"] == "asset-1"
        assert kwargs["extracted_url"] is None

    def test_missing_source_raises_validation_error(self):
        # PasteRequest requires either asset_id or extracted_url via model_validator
        with pytest.raises(ValueError):
            PasteRequest(target_bbox=BboxSchema(x1=0, y1=0, x2=10, y2=10))

    async def test_not_found(self, mock_user, mock_asset_service):
        mock_asset_service.paste_extracted_object.side_effect = ValueError("extracted object not found")
        body = PasteRequest(
            extracted_url="s3://obj.png",
            target_bbox=BboxSchema(x1=0, y1=0, x2=10, y2=10),
        )

        with pytest.raises(HTTPException) as exc:
            await paste_extracted_object(image_id=1, body=body, current_user=mock_user, service=mock_asset_service)

        assert exc.value.status_code == 404

    async def test_generic_error(self, mock_user, mock_asset_service):
        mock_asset_service.paste_extracted_object.side_effect = ValueError("invalid scale")
        body = PasteRequest(
            extracted_url="s3://obj.png",
            target_bbox=BboxSchema(x1=0, y1=0, x2=10, y2=10),
        )

        with pytest.raises(HTTPException) as exc:
            await paste_extracted_object(image_id=1, body=body, current_user=mock_user, service=mock_asset_service)

        assert exc.value.status_code == 400