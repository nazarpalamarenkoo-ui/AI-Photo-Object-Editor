import pytest
from unittest.mock import AsyncMock, MagicMock

from app.services.ml.assets_service import AssetService
from app.db.models.image import Image


pytestmark = pytest.mark.unit


@pytest.fixture
def mock_db():
    return AsyncMock()


@pytest.fixture
def mock_s3():
    s3 = AsyncMock()
    s3.download = AsyncMock(return_value=b"original-bytes")
    s3.upload_bytes = AsyncMock(return_value="s3://bucket/path.png")
    s3.get_presigned_url = AsyncMock(return_value="https://presigned.example/path.png")
    return s3


@pytest.fixture
def mock_redis_storage():
    redis_storage = AsyncMock()
    redis_storage.get_cache_image = AsyncMock(return_value=None)
    redis_storage.cache_image = AsyncMock(return_value=None)
    redis_storage.get_cached_segments = AsyncMock(return_value=None)
    return redis_storage


@pytest.fixture
def mock_redis_history():
    history = AsyncMock()
    history.push_undo_state = AsyncMock(return_value=None)
    return history


@pytest.fixture
def mock_image_repo():
    return AsyncMock()


@pytest.fixture
def mock_detection_repo():
    return AsyncMock()


@pytest.fixture
def mock_pipeline():
    return AsyncMock()


@pytest.fixture
def sample_image():
    image = MagicMock(spec=Image)
    image.id = 1
    image.user_id = 42
    image.storage_path = "raw/42/1/original.jpg"
    image.filename = "original.jpg"
    return image


@pytest.fixture
def service(
    mock_db, mock_s3, mock_redis_storage, mock_redis_history,
    mock_image_repo, mock_detection_repo, mock_pipeline,
):
    return AssetService(
        db=mock_db,
        s3_storage=mock_s3,
        redis_storage=mock_redis_storage,
        redis_history=mock_redis_history,
        image_repo=mock_image_repo,
        detection_repo=mock_detection_repo,
        pipeline=mock_pipeline,
    )


def make_segment(mask_id=1):
    return {
        "mask_id": mask_id,
        "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
        "mask_bytes": b"mask-bytes",
    }

class TestExtractObject:
    async def test_extract_object_success(
        self, service, mock_image_repo, sample_image, mock_redis_storage,
        mock_pipeline, mock_s3,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_redis_storage.get_cached_segments = AsyncMock(return_value=[make_segment(9)])
        mock_pipeline.sam_extract_object = AsyncMock(return_value={
            "extracted_bytes": b"png-bytes",
            "object_size": (100, 100),
            "area_pixels": 5000,
            "cropped_bbox": {"x1": 0, "y1": 0, "x2": 100, "y2": 100},
            "timestamp": "t",
        })

        result = await service.extract_object(image_id=1, mask_id=9, user_id=42)

        mock_pipeline.sam_extract_object.assert_awaited_once()
        _, kwargs = mock_pipeline.sam_extract_object.call_args
        assert kwargs["mask_bytes"] == b"mask-bytes"
        assert kwargs["padding_pixels"] == 8

        mock_s3.upload_bytes.assert_awaited_once()
        upload_kwargs = mock_s3.upload_bytes.call_args.kwargs
        assert upload_kwargs["content_type"] == "image/png"

        assert result["extracted_url"] == "s3://bucket/path.png"
        assert result["presigned_url"] == "https://presigned.example/path.png"
        assert result["object_size"] == (100, 100)
        assert result["area_pixels"] == 5000
        assert "timestamp" in result and result["timestamp"]

    async def test_extract_object_image_not_found(self, service, mock_image_repo):
        mock_image_repo.get_by_id = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="not found"):
            await service.extract_object(image_id=1, mask_id=1, user_id=42)

    async def test_extract_object_unauthorized(self, service, mock_image_repo, sample_image):
        sample_image.user_id = 999
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)

        with pytest.raises(ValueError, match="Unauthorized"):
            await service.extract_object(image_id=1, mask_id=1, user_id=42)

    async def test_extract_object_no_segments_cached(
        self, service, mock_image_repo, sample_image, mock_redis_storage,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_redis_storage.get_cached_segments = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="No segments found"):
            await service.extract_object(image_id=1, mask_id=1, user_id=42)

    async def test_extract_object_mask_id_not_found(
        self, service, mock_image_repo, sample_image, mock_redis_storage,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_redis_storage.get_cached_segments = AsyncMock(return_value=[make_segment(1)])

        with pytest.raises(ValueError, match="Segment with mask_id=999 not found"):
            await service.extract_object(image_id=1, mask_id=999, user_id=42)

    async def test_extract_object_pipeline_exception(
        self, service, mock_image_repo, sample_image, mock_redis_storage, mock_pipeline,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_redis_storage.get_cached_segments = AsyncMock(return_value=[make_segment(9)])
        mock_pipeline.sam_extract_object = AsyncMock(side_effect=RuntimeError("sam crash"))

        with pytest.raises(RuntimeError, match="sam crash"):
            await service.extract_object(image_id=1, mask_id=9, user_id=42)

    async def test_extract_object_s3_exception(
        self, service, mock_image_repo, sample_image, mock_redis_storage,
        mock_pipeline, mock_s3,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_redis_storage.get_cached_segments = AsyncMock(return_value=[make_segment(9)])
        mock_pipeline.sam_extract_object = AsyncMock(return_value={
            "extracted_bytes": b"png", "object_size": (1, 1), "area_pixels": 1,
            "cropped_bbox": {}, "timestamp": "t",
        })
        mock_s3.upload_bytes = AsyncMock(side_effect=IOError("s3 down"))

        with pytest.raises(IOError, match="s3 down"):
            await service.extract_object(image_id=1, mask_id=9, user_id=42)

    async def test_extract_object_boundary_padding_zero(
        self, service, mock_image_repo, sample_image, mock_redis_storage, mock_pipeline,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_redis_storage.get_cached_segments = AsyncMock(return_value=[make_segment(9)])
        mock_pipeline.sam_extract_object = AsyncMock(return_value={
            "extracted_bytes": b"png", "object_size": (1, 1), "area_pixels": 1,
            "cropped_bbox": {}, "timestamp": "t",
        })

        await service.extract_object(image_id=1, mask_id=9, user_id=42, padding_pixels=0)

        _, kwargs = mock_pipeline.sam_extract_object.call_args
        assert kwargs["padding_pixels"] == 0

class TestPasteExtractedObject:
    async def test_paste_extracted_object_success(
        self, service, mock_image_repo, sample_image, mock_s3, mock_redis_history,
        mock_pipeline, mock_redis_storage,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_s3.download = AsyncMock(return_value=b"extracted-png")
        target_bbox = {"x1": 0, "y1": 0, "x2": 50, "y2": 50}
        mock_pipeline.sam_paste_extracted_object = AsyncMock(return_value={
            "result_bytes": b"composited",
            "paste_bbox": target_bbox,
            "object_size": (50, 50),
            "timestamp": "t",
        })

        result = await service.paste_extracted_object(
            image_id=1, user_id=42, extracted_url="s3://bucket/extracted.png",
            target_bbox=target_bbox,
        )

        # undo pushed before pipeline call
        mock_redis_history.push_undo_state.assert_awaited_once()
        mock_pipeline.sam_paste_extracted_object.assert_awaited_once()
        mock_redis_storage.cache_image.assert_awaited_once()

        assert result["paste_bbox"] == target_bbox
        assert result["object_size"] == (50, 50)
        assert "timestamp" in result and result["timestamp"]

    async def test_paste_extracted_object_image_not_found(self, service, mock_image_repo):
        mock_image_repo.get_by_id = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="not found"):
            await service.paste_extracted_object(
                image_id=1, user_id=42, extracted_url="x", target_bbox={}
            )

    async def test_paste_extracted_object_unauthorized(
        self, service, mock_image_repo, sample_image,
    ):
        sample_image.user_id = 999
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)

        with pytest.raises(ValueError, match="Unauthorized"):
            await service.paste_extracted_object(
                image_id=1, user_id=42, extracted_url="x", target_bbox={}
            )

    async def test_paste_extracted_object_s3_download_failure(
        self,
        service,
        mock_image_repo,
        sample_image,
        mock_s3,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)

        mock_s3.download = AsyncMock(
            side_effect=[
                b"original-image",          
                IOError("missing object"), 
            ]
        )

        with pytest.raises(
            ValueError,
            match="Failed to download extracted object",
        ):
            await service.paste_extracted_object(
                image_id=1,
                user_id=42,
                extracted_url="s3://missing.png",
                target_bbox={},
            )

    async def test_paste_extracted_object_pipeline_exception(
        self, service, mock_image_repo, sample_image, mock_s3, mock_pipeline,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_s3.download = AsyncMock(return_value=b"extracted-png")
        mock_pipeline.sam_paste_extracted_object = AsyncMock(
            side_effect=RuntimeError("composite failed")
        )

        with pytest.raises(RuntimeError, match="composite failed"):
            await service.paste_extracted_object(
                image_id=1, user_id=42, extracted_url="s3://x.png", target_bbox={}
            )

    async def test_paste_extracted_object_boundary_scale_min(
        self, service, mock_image_repo, sample_image, mock_s3, mock_pipeline,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_s3.download = AsyncMock(return_value=b"extracted-png")
        mock_pipeline.sam_paste_extracted_object = AsyncMock(return_value={
            "result_bytes": b"r", "paste_bbox": {}, "object_size": (1, 1), "timestamp": "t",
        })

        await service.paste_extracted_object(
            image_id=1, user_id=42, extracted_url="s3://x.png", target_bbox={}, scale=0.1
        )

        _, kwargs = mock_pipeline.sam_paste_extracted_object.call_args
        assert kwargs["scale"] == 0.1

    async def test_paste_extracted_object_boundary_scale_max(
        self, service, mock_image_repo, sample_image, mock_s3, mock_pipeline,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_s3.download = AsyncMock(return_value=b"extracted-png")
        mock_pipeline.sam_paste_extracted_object = AsyncMock(return_value={
            "result_bytes": b"r", "paste_bbox": {}, "object_size": (1, 1), "timestamp": "t",
        })

        await service.paste_extracted_object(
            image_id=1, user_id=42, extracted_url="s3://x.png", target_bbox={}, scale=3.0
        )

        _, kwargs = mock_pipeline.sam_paste_extracted_object.call_args
        assert kwargs["scale"] == 3.0

    async def test_paste_extracted_object_default_color_matching_true(
        self, service, mock_image_repo, sample_image, mock_s3, mock_pipeline,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_s3.download = AsyncMock(return_value=b"extracted-png")
        mock_pipeline.sam_paste_extracted_object = AsyncMock(return_value={
            "result_bytes": b"r", "paste_bbox": {}, "object_size": (1, 1), "timestamp": "t",
        })

        await service.paste_extracted_object(
            image_id=1, user_id=42, extracted_url="s3://x.png", target_bbox={}
        )

        _, kwargs = mock_pipeline.sam_paste_extracted_object.call_args
        assert kwargs["use_color_matching"] is True
        assert kwargs["color_match_method"] == "color_transfer"