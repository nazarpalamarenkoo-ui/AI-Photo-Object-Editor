import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from sqlalchemy.exc import IntegrityError

from app.repository.base_repo import BaseRepository
from app.repository.user_repo import UserRepository
from app.repository.image_repo import ImageRepository
from app.repository.detection_repo import DetectionRepository
from app.repository.assets_repo import AssetRepository
from app.repository.image_content_repo import ImageContentRepository
from app.repository.image_version_repo import ImageVersionRepository
from app.repository.edit_history_repo import ImageEditHistoryRepository
from app.repository.mljob_repo import MLJobRepository
from app.repository.segmentation_repo import SegmentationRepository

from app.db.models.user import User
from app.db.models.image import Image
from app.db.models.detection import Detection
from app.db.models.assets import Asset
from app.db.models.image_content import ImageContent
from app.db.models.image_version import ImageVersion
from app.db.models.image_edit_history import ImageEditHistory
from app.db.models.mljobs import MLJob
from app.db.models.segmentation import SegmentationMask

from app.db.enums.edit_operation import EditOperation
from app.db.enums.engine_types import EngineType
from app.db.enums.ml_task_status import MLTaskType
from app.db.enums.ml_job_status import JobStatus


class _SessionCtx:
    """Minimal async context manager standing in for `session_factory()`."""

    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.fixture
def db():
    """A fresh mock AsyncSession for each test.

    `add`/`add_all` are plain (sync) methods on AsyncSession, so they're
    forced to MagicMock; everything else defaults to AsyncMock's own
    async children (commit, refresh, execute, merge, delete, get, flush).
    """
    session = AsyncMock()
    session.add = MagicMock()
    session.add_all = MagicMock()
    return session


@pytest.fixture
def session_factory(db):
    return lambda: _SessionCtx(db)


def make_scalar_result(value):
    result = MagicMock()
    result.scalar_one_or_none.return_value = value
    result.scalar_one.return_value = value
    return result


def make_scalars_result(values):
    result = MagicMock()
    result.scalars.return_value.all.return_value = values
    return result


@pytest.mark.unit
class TestBaseRepository:
    def test_stores_session_factory(self, session_factory):
        repo = BaseRepository(session_factory)
        assert repo.session_factory is session_factory


@pytest.mark.unit
class TestUserRepository:
    @pytest.mark.asyncio
    async def test_create(self, db, session_factory):
        repo = UserRepository(session_factory)
        user = await repo.create("john", "john@test.com", "hash")

        db.add.assert_called_once()
        db.commit.assert_awaited_once()
        db.refresh.assert_awaited_once_with(user)
        assert user.username == "john"
        assert user.email == "john@test.com"
        assert user.password_hash == "hash"

    @pytest.mark.asyncio
    async def test_get_by_id_found(self, db, session_factory):
        repo = UserRepository(session_factory)
        mock_user = User(id=1, username="u", email="e", password_hash="h")
        db.execute = AsyncMock(return_value=make_scalar_result(mock_user))

        assert await repo.get_by_id(1) is mock_user

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, db, session_factory):
        repo = UserRepository(session_factory)
        db.execute = AsyncMock(return_value=make_scalar_result(None))

        assert await repo.get_by_id(999) is None

    @pytest.mark.asyncio
    async def test_get_by_email(self, db, session_factory):
        repo = UserRepository(session_factory)
        mock_user = User(id=1, username="u", email="e@test.com", password_hash="h")
        db.execute = AsyncMock(return_value=make_scalar_result(mock_user))

        assert await repo.get_by_email("e@test.com") is mock_user

    @pytest.mark.asyncio
    async def test_get_by_username(self, db, session_factory):
        repo = UserRepository(session_factory)
        mock_user = User(id=1, username="u", email="e", password_hash="h")
        db.execute = AsyncMock(return_value=make_scalar_result(mock_user))

        assert await repo.get_by_username("u") is mock_user

    @pytest.mark.asyncio
    async def test_exists_by_email_true(self, db, session_factory):
        repo = UserRepository(session_factory)
        db.execute = AsyncMock(return_value=make_scalar_result(1))

        assert await repo.exists_by_email("e@test.com") is True

    @pytest.mark.asyncio
    async def test_exists_by_email_false(self, db, session_factory):
        repo = UserRepository(session_factory)
        db.execute = AsyncMock(return_value=make_scalar_result(None))

        assert await repo.exists_by_email("e@test.com") is False

    @pytest.mark.asyncio
    async def test_update_merges_detached_instance(self, db, session_factory):
        repo = UserRepository(session_factory)
        user = User(id=1, username="old", email="e", password_hash="h")
        merged_user = User(id=1, username="new", email="e", password_hash="h")
        db.merge = AsyncMock(return_value=merged_user)

        result = await repo.update(user)

        db.merge.assert_awaited_once_with(user)
        db.commit.assert_awaited_once()
        db.refresh.assert_awaited_once_with(merged_user)
        assert result is merged_user

    @pytest.mark.asyncio
    async def test_update_password(self, db, session_factory):
        repo = UserRepository(session_factory)
        user = User(id=1, username="u", email="e", password_hash="old")
        db.merge = AsyncMock(return_value=user)

        updated = await repo.update_password(user, "new_hash")

        assert updated.password_hash == "new_hash"
        db.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_delete_found(self, db, session_factory):
        repo = UserRepository(session_factory)
        user = User(id=1, username="u", email="e", password_hash="h")
        db.get = AsyncMock(return_value=user)

        result = await repo.delete(1)

        assert result is True
        db.delete.assert_awaited_once_with(user)

    @pytest.mark.asyncio
    async def test_delete_not_found(self, db, session_factory):
        repo = UserRepository(session_factory)
        db.get = AsyncMock(return_value=None)

        assert await repo.delete(999) is False


@pytest.mark.unit
class TestImageRepository:
    @pytest.mark.asyncio
    async def test_create(self, db, session_factory):
        repo = ImageRepository(session_factory)
        image = await repo.create(
            filename="f.jpg", storage_path="s3://f.jpg", user_id=1,
            mime_type="image/jpeg", width=100, height=100, file_size=1024,
        )

        db.add.assert_called_once()
        assert image.filename == "f.jpg"
        assert image.cache_key is None

    @pytest.mark.asyncio
    async def test_get_by_id(self, db, session_factory):
        repo = ImageRepository(session_factory)
        mock_image = Image(id=1, filename="f", storage_path="p", user_id=1)
        db.execute = AsyncMock(return_value=make_scalar_result(mock_image))

        assert await repo.get_by_id(1) is mock_image

    @pytest.mark.asyncio
    async def test_get_user_images(self, db, session_factory):
        repo = ImageRepository(session_factory)
        images = [
            Image(id=1, filename="a", storage_path="p", user_id=1),
            Image(id=2, filename="b", storage_path="p", user_id=1),
        ]
        db.execute = AsyncMock(return_value=make_scalars_result(images))

        assert await repo.get_user_images(1) == images

    @pytest.mark.asyncio
    async def test_update_merges_detached_instance(self, db, session_factory):
        repo = ImageRepository(session_factory)
        image = Image(id=1, filename="f", storage_path="p", user_id=1)
        db.merge = AsyncMock(return_value=image)

        result = await repo.update(image)

        db.merge.assert_awaited_once_with(image)
        db.commit.assert_awaited_once()
        assert result is image

    @pytest.mark.asyncio
    async def test_delete_found(self, db, session_factory):
        repo = ImageRepository(session_factory)
        image = Image(id=1, filename="f", storage_path="p", user_id=1)
        db.get = AsyncMock(return_value=image)

        result = await repo.delete(1)

        assert result is True
        db.delete.assert_awaited_once_with(image)

    @pytest.mark.asyncio
    async def test_delete_not_found(self, db, session_factory):
        repo = ImageRepository(session_factory)
        db.get = AsyncMock(return_value=None)

        assert await repo.delete(1) is False
@pytest.mark.unit
class TestDetectionRepository:
    @pytest.mark.asyncio
    async def test_create_many(self, db, session_factory):
        repo = DetectionRepository(session_factory)
        dets = [Detection(content_id=1, bbox_id=0), Detection(content_id=1, bbox_id=1)]

        result = await repo.create_many(dets)

        db.add_all.assert_called_once_with(dets)
        db.commit.assert_awaited_once()
        assert result == dets

    @pytest.mark.asyncio
    async def test_get_by_content_active_only(self, db, session_factory):
        repo = DetectionRepository(session_factory)
        dets = [Detection(content_id=1, bbox_id=0, is_active=True)]
        db.execute = AsyncMock(return_value=make_scalars_result(dets))

        assert await repo.get_by_content(1) == dets

    @pytest.mark.asyncio
    async def test_get_by_content_include_inactive(self, db, session_factory):
        repo = DetectionRepository(session_factory)
        dets = [Detection(content_id=1, bbox_id=0, is_active=False)]
        db.execute = AsyncMock(return_value=make_scalars_result(dets))

        assert await repo.get_by_content(1, active_only=False) == dets

    @pytest.mark.asyncio
    async def test_get_by_id(self, db, session_factory):
        repo = DetectionRepository(session_factory)
        det = Detection(id=1, content_id=1, bbox_id=0)
        db.execute = AsyncMock(return_value=make_scalar_result(det))

        assert await repo.get_by_id(1) is det

    @pytest.mark.asyncio
    async def test_max_bbox_id_empty(self, session_factory):
        repo = DetectionRepository(session_factory)
        repo.get_by_content = AsyncMock(return_value=[])

        assert await repo.max_bbox_id(1) == -1

    @pytest.mark.asyncio
    async def test_max_bbox_id_with_detections(self, session_factory):
        repo = DetectionRepository(session_factory)
        dets = [Detection(bbox_id=0), Detection(bbox_id=3), Detection(bbox_id=1)]
        repo.get_by_content = AsyncMock(return_value=dets)

        assert await repo.max_bbox_id(1) == 3

    @pytest.mark.asyncio
    async def test_soft_delete(self, db, session_factory):
        repo = DetectionRepository(session_factory)
        det = Detection(id=1, content_id=1, bbox_id=0, is_active=False)
        db.execute = AsyncMock(return_value=make_scalar_result(det))

        result = await repo.soft_delete(1)

        assert result is det
        db.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_delete_by_content(self, db, session_factory):
        repo = DetectionRepository(session_factory)
        dets = [Detection(id=1), Detection(id=2)]
        db.execute = AsyncMock(return_value=make_scalars_result(dets))

        count = await repo.delete_by_content(1)

        assert count == 2
        assert db.delete.await_count == 2


@pytest.mark.unit
class TestAssetRepository:
    @pytest.mark.asyncio
    async def test_create(self, db, session_factory):
        repo = AssetRepository(session_factory)
        asset = Asset(user_id=1, storage_path="s3://a.png", width=10, height=10, area_pixels=100)

        result = await repo.create(asset)

        db.add.assert_called_once_with(asset)
        assert result is asset

    @pytest.mark.asyncio
    async def test_get_by_public_id_scoped_by_user(self, db, session_factory):
        repo = AssetRepository(session_factory)
        asset = Asset(id=1, user_id=1, public_id="pub-1", storage_path="p", width=1, height=1, area_pixels=1)
        db.execute = AsyncMock(return_value=make_scalar_result(asset))

        assert await repo.get_by_public_id(1, "pub-1") is asset

    @pytest.mark.asyncio
    async def test_get_overflow_none_when_under_cap(self, db, session_factory):
        repo = AssetRepository(session_factory)
        db.execute = AsyncMock(return_value=make_scalar_result(5))

        assert await repo.get_overflow(1, max_assets=200) == []

    @pytest.mark.asyncio
    async def test_get_overflow_returns_oldest(self, db, session_factory):
        repo = AssetRepository(session_factory)
        overflow_assets = [Asset(id=1), Asset(id=2)]
        db.execute = AsyncMock(
            side_effect=[make_scalar_result(202), make_scalars_result(overflow_assets)]
        )

        result = await repo.get_overflow(1, max_assets=200)

        assert result == overflow_assets

    @pytest.mark.asyncio
    async def test_list_by_user(self, db, session_factory):
        repo = AssetRepository(session_factory)
        assets = [Asset(id=1), Asset(id=2)]
        db.execute = AsyncMock(return_value=make_scalars_result(assets))

        assert await repo.list_by_user(1) == assets

    @pytest.mark.asyncio
    async def test_rename_merges_detached_instance(self, db, session_factory):
        repo = AssetRepository(session_factory)
        asset = Asset(id=1, label="old")
        db.merge = AsyncMock(return_value=asset)

        result = await repo.rename(asset, "new")

        assert result.label == "new"
        db.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_delete_many(self, db, session_factory):
        repo = AssetRepository(session_factory)
        assets = [Asset(id=1), Asset(id=2)]
        db.merge = AsyncMock(side_effect=assets)

        await repo.delete_many(assets)

        assert db.delete.await_count == 2
        db.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_delete(self, db, session_factory):
        repo = AssetRepository(session_factory)
        asset = Asset(id=1)
        db.merge = AsyncMock(return_value=asset)

        await repo.delete(asset)

        db.delete.assert_awaited_once_with(asset)


@pytest.mark.unit
class TestImageContentRepository:
    @pytest.mark.asyncio
    async def test_get_by_id(self, db, session_factory):
        repo = ImageContentRepository(session_factory)
        content = ImageContent(id=1, content_hash="a" * 64)
        db.execute = AsyncMock(return_value=make_scalar_result(content))

        assert await repo.get_by_id(1) is content

    @pytest.mark.asyncio
    async def test_get_by_hash(self, db, session_factory):
        repo = ImageContentRepository(session_factory)
        content = ImageContent(id=1, content_hash="a" * 64)
        db.execute = AsyncMock(return_value=make_scalar_result(content))

        assert await repo.get_by_hash("a" * 64) is content

    @pytest.mark.asyncio
    async def test_create(self, db, session_factory):
        repo = ImageContentRepository(session_factory)

        content = await repo.create("a" * 64, "s3://c.png", 100, 100, 1024)

        db.add.assert_called_once()
        assert content.content_hash == "a" * 64

    @pytest.mark.asyncio
    async def test_get_or_create_existing_found(self, session_factory):
        repo = ImageContentRepository(session_factory)
        existing = ImageContent(id=1, content_hash="a" * 64)
        repo.get_by_hash = AsyncMock(return_value=existing)
        repo.create = AsyncMock()

        content, created = await repo.get_or_create("a" * 64, "p", 1, 1, 1)

        assert content is existing
        assert created is False
        repo.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_or_create_creates_new(self, session_factory):
        repo = ImageContentRepository(session_factory)
        new_content = ImageContent(id=1, content_hash="b" * 64)
        repo.get_by_hash = AsyncMock(return_value=None)
        repo.create = AsyncMock(return_value=new_content)

        content, created = await repo.get_or_create("b" * 64, "p", 1, 1, 1)

        assert content is new_content
        assert created is True

    @pytest.mark.asyncio
    async def test_get_or_create_race_resolved_by_reread(self, session_factory):
        repo = ImageContentRepository(session_factory)
        winner = ImageContent(id=1, content_hash="c" * 64)
        repo.get_by_hash = AsyncMock(side_effect=[None, winner])
        repo.create = AsyncMock(side_effect=IntegrityError("stmt", {}, Exception("dup")))

        content, created = await repo.get_or_create("c" * 64, "p", 1, 1, 1)

        assert content is winner
        assert created is False

    @pytest.mark.asyncio
    async def test_get_or_create_race_reraises_if_still_missing(self, session_factory):
        repo = ImageContentRepository(session_factory)
        repo.get_by_hash = AsyncMock(side_effect=[None, None])
        repo.create = AsyncMock(side_effect=IntegrityError("stmt", {}, Exception("dup")))

        with pytest.raises(IntegrityError):
            await repo.get_or_create("d" * 64, "p", 1, 1, 1)

@pytest.mark.unit
class TestImageEditHistoryRepository:
    @pytest.mark.asyncio
    async def test_create(self, db, session_factory):
        repo = ImageEditHistoryRepository(session_factory)

        entry = await repo.create(
            image_version_id=1, operation=EditOperation.REMOVE, engine=EngineType.LAMA,
        )

        db.add.assert_called_once()
        assert entry.operation == EditOperation.REMOVE
        assert entry.engine == EngineType.LAMA
        assert entry.parameters is None

    @pytest.mark.asyncio
    async def test_get_by_id(self, db, session_factory):
        repo = ImageEditHistoryRepository(session_factory)
        entry = ImageEditHistory(id=1, image_version_id=1)
        db.execute = AsyncMock(return_value=make_scalar_result(entry))

        assert await repo.get_by_id(1) is entry

    @pytest.mark.asyncio
    async def test_get_by_version(self, db, session_factory):
        repo = ImageEditHistoryRepository(session_factory)
        entries = [ImageEditHistory(id=1), ImageEditHistory(id=2)]
        db.execute = AsyncMock(return_value=make_scalars_result(entries))

        assert await repo.get_by_version(1) == entries


@pytest.mark.unit
class TestImageVersionRepository:
    @pytest.mark.asyncio
    async def test_create_original(self, db, session_factory):
        repo = ImageVersionRepository(session_factory)
        image = Image(id=1, storage_path="s3://orig.png")
        db.merge = AsyncMock(return_value=image)

        added = {}
        db.add = MagicMock(side_effect=lambda obj: added.__setitem__("obj", obj))

        async def fake_flush():
            added["obj"].id = 100

        db.flush = AsyncMock(side_effect=fake_flush)

        version = await repo.create_original(image, content_id=5)

        db.merge.assert_awaited_once_with(image)
        db.commit.assert_awaited_once()
        assert version.version_number == 0
        assert version.parent_version_id is None
        assert version.content_id == 5
        assert version.id == 100
        assert image.current_version_id == 100

    @pytest.mark.asyncio
    async def test_create_next_raises_without_current(self, db, session_factory):
        repo = ImageVersionRepository(session_factory)
        image = Image(id=1, current_version_id=None)
        db.merge = AsyncMock(return_value=image)
        repo._get_current = AsyncMock(return_value=None)

        with pytest.raises(ValueError):
            await repo.create_next(image, "s3://next.png", content_id=5)

    @pytest.mark.asyncio
    async def test_create_next_forks_from_current(self, db, session_factory):
        repo = ImageVersionRepository(session_factory)
        image = Image(id=1, current_version_id=10)
        current = ImageVersion(id=10, image_id=1, version_number=1)
        db.merge = AsyncMock(return_value=image)
        repo._get_current = AsyncMock(return_value=current)
        repo._next_version_number = AsyncMock(return_value=2)

        added = {}
        db.add = MagicMock(side_effect=lambda obj: added.__setitem__("obj", obj))

        async def fake_flush():
            added["obj"].id = 200

        db.flush = AsyncMock(side_effect=fake_flush)
        db.execute = AsyncMock(return_value=MagicMock())  # row-lock select

        version = await repo.create_next(image, "s3://next.png", content_id=5)

        assert version.version_number == 2
        assert version.parent_version_id == 10
        assert image.current_version_id == 200

    @pytest.mark.asyncio
    async def test_next_version_number_no_history(self, db, session_factory):
        repo = ImageVersionRepository(session_factory)
        db.execute = AsyncMock(return_value=make_scalar_result(None))

        assert await repo._next_version_number(db, 1) == 1

    @pytest.mark.asyncio
    async def test_next_version_number_continues_from_max(self, db, session_factory):
        repo = ImageVersionRepository(session_factory)
        db.execute = AsyncMock(return_value=make_scalar_result(5))

        assert await repo._next_version_number(db, 1) == 6

    @pytest.mark.asyncio
    async def test_get_current_internal_none_without_pointer(self, session_factory):
        repo = ImageVersionRepository(session_factory)
        image = Image(id=1, current_version_id=None)

        assert await repo._get_current(AsyncMock(), image) is None

    @pytest.mark.asyncio
    async def test_get_current_public_merges_and_delegates(self, db, session_factory):
        repo = ImageVersionRepository(session_factory)
        image = Image(id=1, current_version_id=5)
        version = ImageVersion(id=5)
        db.merge = AsyncMock(return_value=image)
        db.execute = AsyncMock(return_value=make_scalar_result(version))

        assert await repo.get_current(image) is version

    @pytest.mark.asyncio
    async def test_get_by_id(self, db, session_factory):
        repo = ImageVersionRepository(session_factory)
        version = ImageVersion(id=1)
        db.execute = AsyncMock(return_value=make_scalar_result(version))

        assert await repo.get_by_id(1) is version

    @pytest.mark.asyncio
    async def test_list_by_image(self, db, session_factory):
        repo = ImageVersionRepository(session_factory)
        versions = [ImageVersion(id=1), ImageVersion(id=2)]
        db.execute = AsyncMock(return_value=make_scalars_result(versions))

        assert await repo.list_by_image(1) == versions

    @pytest.mark.asyncio
    async def test_set_current_valid(self, db, session_factory):
        repo = ImageVersionRepository(session_factory)
        image = Image(id=1, current_version_id=1)
        version = ImageVersion(id=2, image_id=1)
        db.merge = AsyncMock(return_value=image)
        db.execute = AsyncMock(return_value=make_scalar_result(version))

        result = await repo.set_current(image, 2)

        assert result is version
        assert image.current_version_id == 2

    @pytest.mark.asyncio
    async def test_set_current_wrong_image_raises(self, db, session_factory):
        repo = ImageVersionRepository(session_factory)
        image = Image(id=1)
        other_version = ImageVersion(id=2, image_id=99)
        db.merge = AsyncMock(return_value=image)
        db.execute = AsyncMock(return_value=make_scalar_result(other_version))

        with pytest.raises(ValueError):
            await repo.set_current(image, 2)

@pytest.mark.unit
class TestMLJobRepository:
    @pytest.mark.asyncio
    async def test_create(self, db, session_factory):
        repo = MLJobRepository(session_factory)

        job = await repo.create(content_id=1, image_version_id=1, task_type=MLTaskType.DETECTION)

        db.add.assert_called_once()
        assert job.status == JobStatus.PENDING

    @pytest.mark.asyncio
    async def test_get_by_id(self, db, session_factory):
        repo = MLJobRepository(session_factory)
        job = MLJob(id=1)
        db.execute = AsyncMock(return_value=make_scalar_result(job))

        assert await repo.get_by_id(1) is job

    @pytest.mark.asyncio
    async def test_get_by_content(self, db, session_factory):
        repo = MLJobRepository(session_factory)
        jobs = [MLJob(id=1), MLJob(id=2)]
        db.execute = AsyncMock(return_value=make_scalars_result(jobs))

        assert await repo.get_by_content(1) == jobs

    @pytest.mark.asyncio
    async def test_get_by_version(self, db, session_factory):
        repo = MLJobRepository(session_factory)
        jobs = [MLJob(id=1)]
        db.execute = AsyncMock(return_value=make_scalars_result(jobs))

        assert await repo.get_by_version(1) == jobs

    @pytest.mark.asyncio
    async def test_get_successful_found(self, db, session_factory):
        repo = MLJobRepository(session_factory)
        job = MLJob(id=1, status=JobStatus.SUCCESS)
        db.execute = AsyncMock(return_value=make_scalar_result(job))

        assert await repo.get_successful(1, MLTaskType.DETECTION) is job

    @pytest.mark.asyncio
    async def test_get_pending_without_task_filter(self, db, session_factory):
        repo = MLJobRepository(session_factory)
        jobs = [MLJob(id=1, status=JobStatus.PENDING)]
        db.execute = AsyncMock(return_value=make_scalars_result(jobs))

        assert await repo.get_pending() == jobs

    @pytest.mark.asyncio
    async def test_get_pending_with_task_filter(self, db, session_factory):
        repo = MLJobRepository(session_factory)
        jobs = [MLJob(id=1, status=JobStatus.PENDING, task_type=MLTaskType.DETECTION)]
        db.execute = AsyncMock(return_value=make_scalars_result(jobs))

        assert await repo.get_pending(task_type=MLTaskType.DETECTION) == jobs

    @pytest.mark.asyncio
    async def test_mark_running(self, db, session_factory):
        repo = MLJobRepository(session_factory)
        job = MLJob(id=1, status=JobStatus.PENDING)
        db.get = AsyncMock(return_value=job)

        result = await repo.mark_running(1)

        assert result.status == JobStatus.RUNNING

    @pytest.mark.asyncio
    async def test_mark_success(self, db, session_factory):
        repo = MLJobRepository(session_factory)
        job = MLJob(id=1, status=JobStatus.RUNNING)
        db.get = AsyncMock(return_value=job)

        result = await repo.mark_success(1, processing_time_ms=500)

        assert result.status == JobStatus.SUCCESS
        assert result.processing_time_ms == 500
        assert result.finished_at is not None

    @pytest.mark.asyncio
    async def test_mark_failed(self, db, session_factory):
        repo = MLJobRepository(session_factory)
        job = MLJob(id=1, status=JobStatus.RUNNING)
        db.get = AsyncMock(return_value=job)

        result = await repo.mark_failed(1, error_message="boom")

        assert result.status == JobStatus.FAILED
        assert result.error_message == "boom"
        assert result.finished_at is not None


@pytest.mark.unit
class TestSegmentationRepository:
    @pytest.mark.asyncio
    async def test_create_many(self, db, session_factory):
        repo = SegmentationRepository(session_factory)
        masks = [SegmentationMask(content_id=1, mask_id=0), SegmentationMask(content_id=1, mask_id=1)]

        result = await repo.create_many(masks)

        db.add_all.assert_called_once_with(masks)
        assert result == masks

    @pytest.mark.asyncio
    async def test_get_by_content_active_only(self, db, session_factory):
        repo = SegmentationRepository(session_factory)
        masks = [SegmentationMask(content_id=1, mask_id=0, is_active=True)]
        db.execute = AsyncMock(return_value=make_scalars_result(masks))

        assert await repo.get_by_content(1) == masks

    @pytest.mark.asyncio
    async def test_get_by_id(self, db, session_factory):
        repo = SegmentationRepository(session_factory)
        mask = SegmentationMask(id=1)
        db.execute = AsyncMock(return_value=make_scalar_result(mask))

        assert await repo.get_by_id(1) is mask

    @pytest.mark.asyncio
    async def test_max_mask_id_empty(self, session_factory):
        repo = SegmentationRepository(session_factory)
        repo.get_by_content = AsyncMock(return_value=[])

        assert await repo.max_mask_id(1) == -1

    @pytest.mark.asyncio
    async def test_max_mask_id_with_masks(self, session_factory):
        repo = SegmentationRepository(session_factory)
        masks = [SegmentationMask(mask_id=0), SegmentationMask(mask_id=4)]
        repo.get_by_content = AsyncMock(return_value=masks)

        assert await repo.max_mask_id(1) == 4

    @pytest.mark.asyncio
    async def test_soft_delete(self, db, session_factory):
        repo = SegmentationRepository(session_factory)
        mask = SegmentationMask(id=1, is_active=False)
        db.execute = AsyncMock(return_value=make_scalar_result(mask))

        assert await repo.soft_delete(1) is mask

    @pytest.mark.asyncio
    async def test_delete_by_content(self, db, session_factory):
        repo = SegmentationRepository(session_factory)
        masks = [SegmentationMask(id=1), SegmentationMask(id=2), SegmentationMask(id=3)]
        db.execute = AsyncMock(return_value=make_scalars_result(masks))

        count = await repo.delete_by_content(1)

        assert count == 3
        assert db.delete.await_count == 3