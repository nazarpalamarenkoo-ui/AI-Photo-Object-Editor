import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.workers.worker import sam_extract_object_task


class _FakeDepsCM:
    def __init__(self, deps):
        self._deps = deps

    async def __aenter__(self):
        return self._deps

    async def __aexit__(self, *exc):
        return False


class _FakeDbCM:
    def __init__(self, db):
        self._db = db

    async def __aenter__(self):
        return self._db

    async def __aexit__(self, *exc):
        return False


@pytest.fixture
def fake_deps():
    return {
        "db": MagicMock(name="db"),
        "s3_storage": MagicMock(name="s3_storage"),
        "redis_storage": MagicMock(name="redis_storage"),
        "redis_history": MagicMock(name="redis_history"),
        "redis_assets": MagicMock(name="redis_assets"),
        "image_repo": MagicMock(name="image_repo"),
        "detection_repo": MagicMock(name="detection_repo"),
        "pipeline": MagicMock(name="pipeline"),
    }


@pytest.fixture(autouse=True)
def patch_db_and_deps(fake_deps):
    with patch(
        "app.workers.worker.get_db_session", return_value=_FakeDbCM(fake_deps["db"])
    ), patch("app.workers.worker._build_ml_deps", return_value=_FakeDepsCM(fake_deps)):
        yield fake_deps


@pytest.mark.asyncio
class TestSamExtractObjectTask:
    async def test_passes_redis_assets_explicitly_and_rest_as_kwargs(self, fake_deps):
        with patch("app.workers.worker.AssetService") as MockService:
            instance = MockService.return_value
            instance.extract_object = AsyncMock(return_value={"ok": True})

            result = await sam_extract_object_task(
                ctx={},
                image_id=1,
                mask_id=3,
                user_id=2,
                padding_pixels=16,
                label="cat",
                persist_to_s3=True,
            )

            _, kwargs = MockService.call_args
            assert kwargs["redis_assets"] is fake_deps["redis_assets"]

            remaining = {k: v for k, v in kwargs.items() if k != "redis_assets"}
            expected_remaining = {
                k: v for k, v in fake_deps.items() if k != "redis_assets"
            }
            assert remaining == expected_remaining

            instance.extract_object.assert_awaited_once_with(
                image_id=1,
                mask_id=3,
                user_id=2,
                padding_pixels=16,
                label="cat",
                persist_to_s3=True,
            )
            assert result == {"ok": True}

    async def test_uses_default_params(self, fake_deps):
        with patch("app.workers.worker.AssetService") as MockService:
            instance = MockService.return_value
            instance.extract_object = AsyncMock(return_value={})

            await sam_extract_object_task(ctx={}, image_id=1, mask_id=3, user_id=2)

            instance.extract_object.assert_awaited_once_with(
                image_id=1,
                mask_id=3,
                user_id=2,
                padding_pixels=8,
                label=None,
                persist_to_s3=False,
            )

    async def test_does_not_mutate_deps_returned_by_build_ml_deps(self, fake_deps):
        original_keys = set(fake_deps.keys())

        with patch("app.workers.worker.AssetService") as MockService:
            instance = MockService.return_value
            instance.extract_object = AsyncMock(return_value={})
            await sam_extract_object_task(ctx={}, image_id=1, mask_id=3, user_id=2)

        assert set(fake_deps.keys()) == original_keys
        assert "redis_assets" in fake_deps