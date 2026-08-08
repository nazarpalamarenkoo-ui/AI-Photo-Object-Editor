import io
import pytest
from unittest.mock import AsyncMock, MagicMock

from PIL import Image as PILImage

from app.services.ml.version_history_service import VersionHistoryService
from app.db.enums.image_status import ImageStatus

pytestmark = pytest.mark.unit


def _png_bytes(size=(20, 20)):
    buf = io.BytesIO()
    PILImage.new("RGB", size, "black").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def mock_s3():
    s3 = AsyncMock()
    s3.upload_bytes = AsyncMock(return_value="s3://bucket/result.jpg")
    s3.get_presigned_url = AsyncMock(return_value="https://presigned.url/result.jpg")
    return s3


@pytest.fixture
def mock_redis_storage():
    redis = AsyncMock()
    redis.get_cache_image = AsyncMock(return_value=None)
    redis.cache_image = AsyncMock(return_value=None)
    redis.delete = AsyncMock(return_value=None)
    return redis


@pytest.fixture
def mock_redis_history():
    history = AsyncMock()
    history.pop_undo_state = AsyncMock(return_value=None)
    history.pop_redo_state = AsyncMock(return_value=None)
    history.push_undo_state = AsyncMock(return_value=None)
    history.push_redo_state = AsyncMock(return_value=None)
    history.get_history_labels = AsyncMock(return_value=[])
    history.clear_history = AsyncMock(return_value=None)
    return history


@pytest.fixture
def mock_redis_assets():
    return AsyncMock()


@pytest.fixture
def mock_image_repo():
    return AsyncMock()


@pytest.fixture
def mock_image_version_repo():
    return AsyncMock()


@pytest.fixture
def mock_image_content_repo():
    return AsyncMock()


@pytest.fixture
def mock_detection_repo():
    return AsyncMock()


@pytest.fixture
def mock_segmentation_repo():
    return AsyncMock()


@pytest.fixture
def mock_edit_history_repo():
    return AsyncMock()


@pytest.fixture
def mock_assets_repo():
    return AsyncMock()


@pytest.fixture
def mock_pipeline():
    return AsyncMock()


@pytest.fixture
def sample_image():
    image = MagicMock()
    image.id = 1
    image.user_id = 42
    image.filename = "original.png"
    return image


@pytest.fixture
def sample_version():
    version = MagicMock()
    version.id = 10
    version.image_id = 1
    version.content_id = 100
    version.storage_path = "raw/42/1/original.jpg"
    version.version_number = 1
    return version


@pytest.fixture
def original_version():
    version = MagicMock()
    version.id = 1
    version.image_id = 1
    version.content_id = 100
    version.version_number = 0
    return version


@pytest.fixture
def service(
    mock_s3, mock_redis_storage, mock_redis_history, mock_redis_assets,
    mock_image_repo, mock_image_version_repo, mock_image_content_repo,
    mock_detection_repo, mock_segmentation_repo, mock_edit_history_repo,
    mock_assets_repo, mock_pipeline,
):
    return VersionHistoryService(
        s3_storage=mock_s3,
        redis_storage=mock_redis_storage,
        redis_history=mock_redis_history,
        redis_assets=mock_redis_assets,
        image_repo=mock_image_repo,
        image_version_repo=mock_image_version_repo,
        image_content_repo=mock_image_content_repo,
        detection_repo=mock_detection_repo,
        segmentation_repo=mock_segmentation_repo,
        edit_history_repo=mock_edit_history_repo,
        assets_repo=mock_assets_repo,
        pipeline=mock_pipeline,
    )


class TestUndo:

    async def test_success_pushes_redo_and_restores_previous_bytes(
        self, service, mock_image_repo, mock_redis_storage, mock_redis_history,
        mock_s3, sample_image,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_redis_storage.get_cache_image = AsyncMock(return_value=b"current-bytes")
        mock_redis_history.pop_undo_state = AsyncMock(
            return_value={"bytes": b"prev-bytes", "label": "remove_object"}
        )
        mock_redis_history.get_history_labels = AsyncMock(return_value=["a", "b"])

        result = await service.undo(image_id=1, user_id=42)

        mock_redis_history.pop_undo_state.assert_awaited_once_with(1)
        mock_redis_history.push_redo_state.assert_awaited_once_with(
            1, b"current-bytes", label="redo"
        )
        mock_redis_storage.cache_image.assert_awaited_once_with(
            image_id=1, image_data=b"prev-bytes", suffix="current_state", ttl=7200,
        )
        upload_kwargs = mock_s3.upload_bytes.call_args.kwargs
        assert upload_kwargs["data"] == b"prev-bytes"
        assert upload_kwargs["path"].startswith("temp/42/1/undo_")
        assert result["presigned_url"] == "https://presigned.url/result.jpg"
        assert result["label"] == "remove_object"
        assert result["history"] == ["a", "b"]

    async def test_no_current_state_skips_redo_push(
        self, service, mock_image_repo, mock_redis_storage, mock_redis_history, sample_image,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_redis_storage.get_cache_image = AsyncMock(return_value=None)
        mock_redis_history.pop_undo_state = AsyncMock(
            return_value={"bytes": b"prev-bytes", "label": "l"}
        )

        await service.undo(image_id=1, user_id=42)

        mock_redis_history.push_redo_state.assert_not_awaited()

    async def test_nothing_to_undo_raises(
        self, service, mock_image_repo, mock_redis_history, sample_image,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_redis_history.pop_undo_state = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="Nothing to undo"):
            await service.undo(image_id=1, user_id=42)

    async def test_image_not_found(self, service, mock_image_repo):
        mock_image_repo.get_by_id = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="not found"):
            await service.undo(image_id=1, user_id=42)

    async def test_unauthorized(self, service, mock_image_repo, sample_image):
        sample_image.user_id = 999
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)

        with pytest.raises(ValueError, match="Unauthorized"):
            await service.undo(image_id=1, user_id=42)


class TestRedo:

    async def test_success_pushes_undo_checkpoint_and_restores_next_bytes(
        self, service, mock_image_repo, mock_redis_storage, mock_redis_history,
        mock_s3, sample_image,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_redis_storage.get_cache_image = AsyncMock(return_value=b"current-bytes")
        mock_redis_history.pop_redo_state = AsyncMock(
            return_value={"bytes": b"next-bytes", "label": "replace_object"}
        )
        mock_redis_history.get_history_labels = AsyncMock(return_value=["a"])

        result = await service.redo(image_id=1, user_id=42)

        mock_redis_history.pop_redo_state.assert_awaited_once_with(1)
        mock_redis_history.push_undo_state.assert_awaited_once_with(
            1, b"current-bytes", label="redo_checkpoint"
        )
        mock_redis_storage.cache_image.assert_awaited_once_with(
            image_id=1, image_data=b"next-bytes", suffix="current_state", ttl=7200,
        )
        upload_kwargs = mock_s3.upload_bytes.call_args.kwargs
        assert upload_kwargs["path"].startswith("temp/42/1/redo_")
        assert result["label"] == "replace_object"
        assert result["history"] == ["a"]

    async def test_no_current_state_skips_undo_push(
        self, service, mock_image_repo, mock_redis_storage, mock_redis_history, sample_image,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_redis_storage.get_cache_image = AsyncMock(return_value=None)
        mock_redis_history.pop_redo_state = AsyncMock(
            return_value={"bytes": b"next-bytes", "label": "l"}
        )

        await service.redo(image_id=1, user_id=42)

        mock_redis_history.push_undo_state.assert_not_awaited()

    async def test_nothing_to_redo_raises(
        self, service, mock_image_repo, mock_redis_history, sample_image,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_redis_history.pop_redo_state = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="Nothing to redo"):
            await service.redo(image_id=1, user_id=42)

    async def test_unauthorized(self, service, mock_image_repo, sample_image):
        sample_image.user_id = 999
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)

        with pytest.raises(ValueError, match="Unauthorized"):
            await service.redo(image_id=1, user_id=42)


class TestGetHistory:

    async def test_returns_history_labels(
        self, service, mock_image_repo, mock_redis_history, sample_image,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_redis_history.get_history_labels = AsyncMock(return_value=["a", "b", "c"])

        result = await service.get_history(image_id=1, user_id=42)

        assert result == {"history": ["a", "b", "c"]}

    async def test_unauthorized(self, service, mock_image_repo, sample_image):
        sample_image.user_id = 999
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)

        with pytest.raises(ValueError, match="Unauthorized"):
            await service.get_history(image_id=1, user_id=42)


class TestGetCurrentState:
    """get_current_state delegates the presigned-url/is_edited decision to
    the inherited _get_current_state_url helper — that helper's own
    contract is covered elsewhere, so here we only verify the orchestration:
    correct version resolution and correct pass-through of its result."""

    async def test_success(
        self, service, mock_image_repo, mock_image_version_repo, mock_redis_history,
        sample_image, sample_version,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_image_version_repo.get_current = AsyncMock(return_value=sample_version)
        mock_redis_history.get_history_labels = AsyncMock(return_value=["x"])
        service._get_current_state_url = AsyncMock(return_value=("https://url", True))

        result = await service.get_current_state(image_id=1, user_id=42)

        service._get_current_state_url.assert_awaited_once_with(
            1, 42, sample_version.storage_path
        )
        assert result == {
            "presigned_url": "https://url",
            "is_edited": True,
            "history": ["x"],
            "image_version_id": 10,
        }

    async def test_no_current_version_raises(
        self, service, mock_image_repo, mock_image_version_repo, sample_image,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_image_version_repo.get_current = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="has no current version"):
            await service.get_current_state(image_id=1, user_id=42)

    async def test_unauthorized(self, service, mock_image_repo, sample_image):
        sample_image.user_id = 999
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)

        with pytest.raises(ValueError, match="Unauthorized"):
            await service.get_current_state(image_id=1, user_id=42)


class TestSaveResult:

    async def test_success_persists_new_image_as_ready(
        self, service, mock_image_repo, mock_redis_storage, mock_s3, sample_image,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        png_bytes = _png_bytes(size=(30, 15))
        mock_redis_storage.get_cache_image = AsyncMock(return_value=png_bytes)
        mock_s3.upload_bytes = AsyncMock(return_value="s3://bucket/saved.jpg")
        saved_image = MagicMock()
        mock_image_repo.create = AsyncMock(return_value=saved_image)

        result = await service.save_result(image_id=1, user_id=42)

        upload_kwargs = mock_s3.upload_bytes.call_args.kwargs
        assert upload_kwargs["data"] == png_bytes
        assert upload_kwargs["path"].startswith("saved/42/1/result_")
        assert upload_kwargs["content_type"] == "image/jpeg"

        create_kwargs = mock_image_repo.create.call_args.kwargs
        assert create_kwargs["filename"] == "edited_original.png"
        assert create_kwargs["storage_path"] == "s3://bucket/saved.jpg"
        assert create_kwargs["user_id"] == 42
        assert create_kwargs["cache_key"] is None
        assert create_kwargs["mime_type"] == "image/jpeg"
        assert create_kwargs["width"] == 30
        assert create_kwargs["height"] == 15
        assert create_kwargs["file_size"] == len(png_bytes)

        assert saved_image.status == ImageStatus.READY
        mock_image_repo.update.assert_awaited_once_with(saved_image)
        assert result is saved_image

    async def test_no_processed_result_raises(
        self, service, mock_image_repo, mock_redis_storage, sample_image,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_redis_storage.get_cache_image = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="No processed result to save"):
            await service.save_result(image_id=1, user_id=42)

    async def test_unauthorized(self, service, mock_image_repo, sample_image):
        sample_image.user_id = 999
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)

        with pytest.raises(ValueError, match="Unauthorized"):
            await service.save_result(image_id=1, user_id=42)


class TestResetCurrentState:

    async def test_success_moves_pointer_and_clears_session(
        self, service, mock_image_repo, mock_image_version_repo, mock_redis_storage,
        mock_redis_history, sample_image, original_version, sample_version,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_image_version_repo.list_by_image = AsyncMock(
            return_value=[sample_version, original_version]
        )

        await service.reset_current_state(image_id=1, user_id=42)

        mock_image_version_repo.set_current.assert_awaited_once_with(
            sample_image, original_version.id
        )
        mock_redis_storage.delete.assert_awaited_once_with("image:1:current_state")
        mock_redis_history.clear_history.assert_awaited_once_with(1)

    async def test_no_original_version_skips_pointer_move(
        self, service, mock_image_repo, mock_image_version_repo, mock_redis_storage,
        mock_redis_history, sample_image, sample_version,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_image_version_repo.list_by_image = AsyncMock(return_value=[sample_version])

        await service.reset_current_state(image_id=1, user_id=42)

        mock_image_version_repo.set_current.assert_not_awaited()
        mock_redis_storage.delete.assert_awaited_once_with("image:1:current_state")
        mock_redis_history.clear_history.assert_awaited_once_with(1)

    async def test_unauthorized(self, service, mock_image_repo, sample_image):
        sample_image.user_id = 999
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)

        with pytest.raises(ValueError, match="Unauthorized"):
            await service.reset_current_state(image_id=1, user_id=42)