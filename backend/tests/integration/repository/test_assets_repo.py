import pytest
from app.repository.assets_repo import AssetRepository, MAX_ASSETS_PER_USER
from app.db.models.assets import Asset



def _make_asset(user_id: int, public_id: str, label: str = "photo") -> Asset:
    return Asset(
        user_id=user_id,
        public_id=public_id,
        label=label,
        storage_path=f"s3://bucket/{public_id}.png",
        width=100,
        height=100,
        area_pixels=100 * 100,
    )



@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_create_asset(db_session, sample_user):
    """Test creating a new Asset record"""
    repo = AssetRepository(db_session)

    asset = await repo.create(_make_asset(sample_user.id, "pub_001", "My photo"))

    assert asset.id is not None
    assert asset.user_id == sample_user.id
    assert asset.public_id == "pub_001"
    assert asset.label == "My photo"


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_get_by_public_id(db_session, sample_user):
    """Test fetching an asset by (user_id, public_id)"""
    repo = AssetRepository(db_session)

    created = await repo.create(_make_asset(sample_user.id, "pub_002"))
    fetched = await repo.get_by_public_id(sample_user.id, "pub_002")

    assert fetched is not None
    assert fetched.id == created.id


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_get_by_public_id_not_found(db_session, sample_user):
    """Test fetching non-existent public_id returns None"""
    repo = AssetRepository(db_session)

    asset = await repo.get_by_public_id(sample_user.id, "nonexistent_pub")

    assert asset is None


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_get_by_public_id_scoped_to_user(db_session, sample_user, another_user):
    """An asset belonging to another_user must not be visible when scoped to sample_user"""
    repo = AssetRepository(db_session)

    await repo.create(_make_asset(another_user.id, "shared_pub_id"))

    asset = await repo.get_by_public_id(sample_user.id, "shared_pub_id")

    assert asset is None


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_list_by_user(db_session, sample_user):
    """list_by_user must return assets in descending created_at order"""
    repo = AssetRepository(db_session)

    a1 = await repo.create(_make_asset(sample_user.id, "pub_010"))
    a2 = await repo.create(_make_asset(sample_user.id, "pub_011"))
    a3 = await repo.create(_make_asset(sample_user.id, "pub_012"))

    assets = await repo.list_by_user(sample_user.id)

    ids = [a.id for a in assets]
    assert a1.id in ids
    assert a2.id in ids
    assert a3.id in ids
    # Descending order
    assert assets[0].created_at >= assets[-1].created_at


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_list_by_user_pagination(db_session, sample_user):
    """list_by_user must respect limit and offset"""
    repo = AssetRepository(db_session)

    for i in range(5):
        await repo.create(_make_asset(sample_user.id, f"pub_page_{i}"))

    page1 = await repo.list_by_user(sample_user.id, limit=3, offset=0)
    page2 = await repo.list_by_user(sample_user.id, limit=3, offset=3)

    assert len(page1) == 3
    # page2 has at least the remaining 2 (possibly more from other tests, but must not overlap)
    page1_ids = {a.id for a in page1}
    for a in page2:
        assert a.id not in page1_ids


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_rename_asset(db_session, sample_user):
    """rename must update only the label, leaving other fields intact"""
    repo = AssetRepository(db_session)

    asset = await repo.create(_make_asset(sample_user.id, "pub_020", "original label"))
    renamed = await repo.rename(asset, "new label")

    assert renamed.label == "new label"
    assert renamed.public_id == "pub_020"
    assert renamed.id == asset.id


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_delete_asset(db_session, sample_user):
    """delete must hard-delete the asset row"""
    repo = AssetRepository(db_session)

    asset = await repo.create(_make_asset(sample_user.id, "pub_030"))
    await repo.delete(asset)

    gone = await repo.get_by_public_id(sample_user.id, "pub_030")
    assert gone is None


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_delete_many_assets(db_session, sample_user):
    """delete_many must remove all provided assets in one call"""
    repo = AssetRepository(db_session)

    a1 = await repo.create(_make_asset(sample_user.id, "pub_040"))
    a2 = await repo.create(_make_asset(sample_user.id, "pub_041"))

    await repo.delete_many([a1, a2])

    assert await repo.get_by_public_id(sample_user.id, "pub_040") is None
    assert await repo.get_by_public_id(sample_user.id, "pub_041") is None


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_get_overflow_within_cap(db_session, sample_user):
    """get_overflow must return empty list when user is within the cap"""
    repo = AssetRepository(db_session)

    # Create 2 assets — well within MAX_ASSETS_PER_USER
    await repo.create(_make_asset(sample_user.id, "pub_050"))
    await repo.create(_make_asset(sample_user.id, "pub_051"))

    overflow = await repo.get_overflow(sample_user.id, max_assets=MAX_ASSETS_PER_USER)

    assert overflow == []


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_get_overflow_returns_oldest_first(db_session, sample_user):
    """get_overflow must return the oldest assets when the cap is exceeded"""
    repo = AssetRepository(db_session)

    # Create 5 assets with a tiny cap of 3 → expect 2 overflow (the oldest)
    assets = []
    for i in range(5):
        a = await repo.create(_make_asset(sample_user.id, f"pub_cap_{i}"))
        assets.append(a)

    overflow = await repo.get_overflow(sample_user.id, max_assets=3)

    assert len(overflow) == 2
    # Oldest two must be in the result
    overflow_ids = {a.id for a in overflow}
    assert assets[0].id in overflow_ids
    assert assets[1].id in overflow_ids