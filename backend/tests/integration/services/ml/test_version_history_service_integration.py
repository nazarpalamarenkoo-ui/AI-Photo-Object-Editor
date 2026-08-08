import io
import pytest
from unittest.mock import AsyncMock
from PIL import Image as PILImage

pytestmark = pytest.mark.integration


def _jpeg_bytes(color=(0, 100, 200)):
    buf = io.BytesIO()
    PILImage.new("RGB", (10, 10), color=color).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def mock_redis_history():
    return AsyncMock()


@pytest.fixture
def mock_redis_assets():
    return AsyncMock()


@pytest.fixture
def mock_pipeline():
    return AsyncMock()


def _make_service(cls, mock_s3_storage, mock_redis_cache, mock_redis_history,
                  mock_redis_assets, image_repo, image_version_repo,
                  image_content_repo, detection_repo, segmentation_repo,
                  edit_history_repo, assets_repo, mock_pipeline):
    return cls(
        s3_storage=mock_s3_storage,
        redis_storage=mock_redis_cache,
        redis_history=mock_redis_history,
        redis_assets=mock_redis_assets,
        image_repo=image_repo,
        image_version_repo=image_version_repo,
        image_content_repo=image_content_repo,
        detection_repo=detection_repo,
        segmentation_repo=segmentation_repo,
        edit_history_repo=edit_history_repo,
        assets_repo=assets_repo,
        pipeline=mock_pipeline,
    )

from app.services.ml.version_history_service import VersionHistoryService


@pytest.fixture
def version_history_service(
    mock_s3_storage, mock_redis_cache, mock_redis_history, mock_redis_assets,
    image_repo, image_version_repo, image_content_repo,
    detection_repo, segmentation_repo, edit_history_repo, assets_repo, mock_pipeline,
):
    return _make_service(
        VersionHistoryService, mock_s3_storage, mock_redis_cache, mock_redis_history,
        mock_redis_assets, image_repo, image_version_repo, image_content_repo,
        detection_repo, segmentation_repo, edit_history_repo, assets_repo, mock_pipeline,
    )


class TestUndo:
    @pytest.mark.asyncio
    async def test_success_restores_previous_state(
        self,
        version_history_service,
        sample_image,
        sample_user,
        mock_redis_cache,
        mock_redis_history,
        mock_s3_storage,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"current-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_redis_history.pop_undo_state = AsyncMock(
            return_value={"bytes": b"prev-bytes", "label": "remove bbox_id=1"}
        )
        mock_redis_history.push_redo_state = AsyncMock()
        mock_redis_history.get_history_labels = AsyncMock(return_value=[])
        mock_s3_storage.upload_bytes = AsyncMock(return_value="s3://t/undo.jpg")
        mock_s3_storage.get_presigned_url = AsyncMock(return_value="https://undo-url")

        result = await version_history_service.undo(sample_image.id, sample_user.id)

        assert result["presigned_url"] == "https://undo-url"
        assert result["label"] == "remove bbox_id=1"
        mock_redis_cache.cache_image.assert_awaited_once_with(
            image_id=sample_image.id,
            image_data=b"prev-bytes",
            suffix="current_state",
            ttl=7200,
        )

    @pytest.mark.asyncio
    async def test_raises_when_nothing_to_undo(
        self,
        version_history_service,
        sample_image,
        sample_user,
        mock_redis_cache,
        mock_redis_history,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"current")
        mock_redis_history.pop_undo_state = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="Nothing to undo"):
            await version_history_service.undo(sample_image.id, sample_user.id)

    @pytest.mark.asyncio
    async def test_pushes_redo_state_when_current_exists(
        self,
        version_history_service,
        sample_image,
        sample_user,
        mock_redis_cache,
        mock_redis_history,
        mock_s3_storage,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"current-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_redis_history.pop_undo_state = AsyncMock(
            return_value={"bytes": b"prev", "label": "x"}
        )
        mock_redis_history.push_redo_state = AsyncMock()
        mock_redis_history.get_history_labels = AsyncMock(return_value=[])
        mock_s3_storage.upload_bytes = AsyncMock(return_value="s3://t/u.jpg")
        mock_s3_storage.get_presigned_url = AsyncMock(return_value="https://u")

        await version_history_service.undo(sample_image.id, sample_user.id)

        mock_redis_history.push_redo_state.assert_awaited_once_with(
            sample_image.id, b"current-bytes", label="redo"
        )

    @pytest.mark.asyncio
    async def test_does_not_push_redo_when_no_current_state(
        self,
        version_history_service,
        sample_image,
        sample_user,
        mock_redis_cache,
        mock_redis_history,
        mock_s3_storage,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=None)
        mock_redis_history.pop_undo_state = AsyncMock(
            return_value={"bytes": b"prev", "label": "x"}
        )
        mock_redis_history.push_redo_state = AsyncMock()
        mock_redis_history.get_history_labels = AsyncMock(return_value=[])
        mock_redis_cache.cache_image = AsyncMock()
        mock_s3_storage.upload_bytes = AsyncMock(return_value="s3://t/u.jpg")
        mock_s3_storage.get_presigned_url = AsyncMock(return_value="https://u")

        await version_history_service.undo(sample_image.id, sample_user.id)

        mock_redis_history.push_redo_state.assert_not_called()

    @pytest.mark.asyncio
    async def test_raises_when_unauthorized(
        self, version_history_service, sample_image
    ):
        with pytest.raises(ValueError, match="Unauthorized"):
            await version_history_service.undo(sample_image.id, sample_image.user_id + 1)


class TestRedo:
    @pytest.mark.asyncio
    async def test_success_restores_next_state(
        self,
        version_history_service,
        sample_image,
        sample_user,
        mock_redis_cache,
        mock_redis_history,
        mock_s3_storage,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"current-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_redis_history.pop_redo_state = AsyncMock(
            return_value={"bytes": b"next-bytes", "label": "redo"}
        )
        mock_redis_history.push_undo_state = AsyncMock()
        mock_redis_history.get_history_labels = AsyncMock(return_value=[])
        mock_s3_storage.upload_bytes = AsyncMock(return_value="s3://t/r.jpg")
        mock_s3_storage.get_presigned_url = AsyncMock(return_value="https://redo-url")

        result = await version_history_service.redo(sample_image.id, sample_user.id)

        assert result["presigned_url"] == "https://redo-url"
        assert result["label"] == "redo"
        mock_redis_cache.cache_image.assert_awaited_once_with(
            image_id=sample_image.id,
            image_data=b"next-bytes",
            suffix="current_state",
            ttl=7200,
        )

    @pytest.mark.asyncio
    async def test_raises_when_nothing_to_redo(
        self,
        version_history_service,
        sample_image,
        sample_user,
        mock_redis_cache,
        mock_redis_history,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"current")
        mock_redis_history.pop_redo_state = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="Nothing to redo"):
            await version_history_service.redo(sample_image.id, sample_user.id)

    @pytest.mark.asyncio
    async def test_pushes_undo_checkpoint_when_current_exists(
        self,
        version_history_service,
        sample_image,
        sample_user,
        mock_redis_cache,
        mock_redis_history,
        mock_s3_storage,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"current-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_redis_history.pop_redo_state = AsyncMock(
            return_value={"bytes": b"next", "label": "redo"}
        )
        mock_redis_history.push_undo_state = AsyncMock()
        mock_redis_history.get_history_labels = AsyncMock(return_value=[])
        mock_s3_storage.upload_bytes = AsyncMock(return_value="s3://t/r.jpg")
        mock_s3_storage.get_presigned_url = AsyncMock(return_value="https://r")

        await version_history_service.redo(sample_image.id, sample_user.id)

        mock_redis_history.push_undo_state.assert_awaited_once_with(
            sample_image.id, b"current-bytes", label="redo_checkpoint"
        )

    @pytest.mark.asyncio
    async def test_raises_when_unauthorized(
        self, version_history_service, sample_image
    ):
        with pytest.raises(ValueError, match="Unauthorized"):
            await version_history_service.redo(sample_image.id, sample_image.user_id + 1)


class TestGetHistory:
    @pytest.mark.asyncio
    async def test_returns_existing_labels(
        self,
        version_history_service,
        sample_image,
        sample_user,
        mock_redis_history,
    ):
        mock_redis_history.get_history_labels = AsyncMock(
            return_value=["remove bbox_id=1", "replace bbox_id=2"]
        )

        result = await version_history_service.get_history(sample_image.id, sample_user.id)

        assert result["history"] == ["remove bbox_id=1", "replace bbox_id=2"]

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_history(
        self,
        version_history_service,
        sample_image,
        sample_user,
        mock_redis_history,
    ):
        mock_redis_history.get_history_labels = AsyncMock(return_value=[])

        result = await version_history_service.get_history(sample_image.id, sample_user.id)

        assert result["history"] == []

    @pytest.mark.asyncio
    async def test_raises_when_unauthorized(
        self, version_history_service, sample_image
    ):
        with pytest.raises(ValueError, match="Unauthorized"):
            await version_history_service.get_history(
                sample_image.id, sample_image.user_id + 1
            )


class TestGetCurrentState:
    @pytest.mark.asyncio
    async def test_returns_redis_url_when_cached(
        self,
        version_history_service,
        sample_image,
        sample_image_version,
        sample_user,
        mock_redis_cache,
        mock_redis_history,
        mock_s3_storage,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"cached-bytes")
        mock_redis_history.get_history_labels = AsyncMock(return_value=["op1"])
        mock_s3_storage.upload_bytes = AsyncMock(return_value="s3://t/cur.jpg")
        mock_s3_storage.get_presigned_url = AsyncMock(return_value="https://redis-url")

        result = await version_history_service.get_current_state(
            sample_image.id, sample_user.id
        )

        assert result["presigned_url"] == "https://redis-url"
        assert result["is_edited"] is True
        assert result["image_version_id"] == sample_image_version.id

    @pytest.mark.asyncio
    async def test_returns_s3_url_on_cache_miss(
        self,
        version_history_service,
        sample_image,
        sample_image_version,
        sample_user,
        mock_redis_cache,
        mock_redis_history,
        mock_s3_storage,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=None)
        mock_redis_history.get_history_labels = AsyncMock(return_value=[])
        mock_s3_storage.get_presigned_url = AsyncMock(return_value="https://s3-url")

        result = await version_history_service.get_current_state(
            sample_image.id, sample_user.id
        )

        assert result["presigned_url"] == "https://s3-url"
        assert result["is_edited"] is False

    @pytest.mark.asyncio
    async def test_raises_when_no_current_version(
        self, version_history_service, sample_image, sample_user
    ):
        with pytest.raises(ValueError, match="no current version"):
            await version_history_service.get_current_state(sample_image.id, sample_user.id)


class TestSaveResult:
    @pytest.mark.asyncio
    async def test_success_creates_new_image_record(
        self,
        version_history_service,
        sample_image,
        sample_user,
        mock_redis_cache,
        mock_s3_storage,
    ):
        result_bytes = _jpeg_bytes()
        mock_redis_cache.get_cache_image = AsyncMock(return_value=result_bytes)
        mock_s3_storage.upload_bytes = AsyncMock(return_value="s3://bucket/saved.jpg")

        saved = await version_history_service.save_result(sample_image.id, sample_user.id)

        assert saved.user_id == sample_user.id
        assert saved.storage_path == "s3://bucket/saved.jpg"
        assert saved.status.value == "ready"
        assert saved.id != sample_image.id

    @pytest.mark.asyncio
    async def test_raises_when_no_current_state(
        self,
        version_history_service,
        sample_image,
        sample_user,
        mock_redis_cache,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="No processed result"):
            await version_history_service.save_result(sample_image.id, sample_user.id)

    @pytest.mark.asyncio
    async def test_raises_when_unauthorized(
        self, version_history_service, sample_image
    ):
        with pytest.raises(ValueError, match="Unauthorized"):
            await version_history_service.save_result(
                sample_image.id, sample_image.user_id + 1
            )


class TestResetCurrentState:
    @pytest.mark.asyncio
    async def test_clears_redis_and_restores_version_0(
        self,
        version_history_service,
        sample_image,
        sample_image_version,
        sample_user,
        mock_redis_cache,
        mock_redis_history,
    ):
        mock_redis_cache.delete = AsyncMock()
        mock_redis_history.clear_history = AsyncMock()

        await version_history_service.reset_current_state(sample_image.id, sample_user.id)

        mock_redis_cache.delete.assert_awaited_once_with(
            f"image:{sample_image.id}:current_state"
        )
        mock_redis_history.clear_history.assert_awaited_once_with(sample_image.id)

        # current version should be reset to version_number=0
        current = await version_history_service.image_version_repo.get_current(sample_image)
        assert current.version_number == 0

    @pytest.mark.asyncio
    async def test_raises_when_unauthorized(
        self, version_history_service, sample_image
    ):
        with pytest.raises(ValueError, match="Unauthorized"):
            await version_history_service.reset_current_state(
                sample_image.id, sample_image.user_id + 1
            )