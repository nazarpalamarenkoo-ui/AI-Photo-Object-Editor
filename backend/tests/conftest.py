import pytest
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy import text
from io import BytesIO
import numpy as np
import sys
import types
from PIL import Image as PILImage
from fastapi import UploadFile
from unittest.mock import AsyncMock, MagicMock
import pytest_asyncio
from unittest.mock import patch, MagicMock
from app.db.db_connect import Base
from app.db.models.user import User
from app.db.models.image import Image
from app.db.models.detection import Detection
from app.db.models.assets import Asset
from app.db.models.image_content import ImageContent
from app.db.models.image_version import ImageVersion
from app.db.models.mljobs import MLJob
from app.db.models.segmentation import SegmentationMask
from app.db.models.image_edit_history import ImageEditHistory
from app.db.enums.ml_task_status import MLTaskType
from app.db.enums.edit_operation import EditOperation
from app.db.enums.engine_types import EngineType
from app.db.enums.segmentation_mode import SegmentationMode

from app.repository.image_repo import ImageRepository
from app.repository.detection_repo import DetectionRepository
from app.repository.image_version_repo import ImageVersionRepository
from app.repository.image_content_repo import ImageContentRepository
from app.repository.segmentation_repo import SegmentationRepository
from app.repository.edit_history_repo import ImageEditHistoryRepository
from app.repository.assets_repo import AssetRepository
from app.repository.mljob_repo import MLJobRepository
from app.repository.user_repo import UserRepository
from app.config.test_settings import test_settings
from contextlib import asynccontextmanager
import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
TEST_DATABASE = test_settings.TEST_DATABASE_URL

    
@pytest_asyncio.fixture(scope='function')
async def db_engine():
    engine = create_async_engine(TEST_DATABASE, echo=False, poolclass=NullPool)

    async with engine.begin() as conn:
        # Guarantee a clean slate even if a previous run left things dirty
        await conn.execute(text("DROP SCHEMA public CASCADE"))
        await conn.execute(text("CREATE SCHEMA public"))
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    async with engine.begin() as conn:
        await conn.execute(text("DROP SCHEMA public CASCADE"))
        await conn.execute(text("CREATE SCHEMA public"))

    await engine.dispose()
    
@pytest_asyncio.fixture(scope='function')
async def db_session(db_engine):
    async_session_maker = async_sessionmaker(
        bind=db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
        autocommit=False
    )

    async with async_session_maker() as session:
        proxy = SessionFactoryProxy(session)
        yield proxy
        await session.rollback()
        
class SessionFactoryProxy:
    def __init__(self, session: AsyncSession):
        self._session = session

    def __call__(self):
        return self._session_cm()

    @asynccontextmanager
    async def _session_cm(self):
        yield self._session

    def __getattr__(self, name):
        return getattr(self._session, name)
        
@pytest.fixture
def mock_upload_file():
    return UploadFile(
        filename="test.jpg",
        file=BytesIO(b"fake image data"),
    )
    
@pytest_asyncio.fixture
async def image_repo(db_session):
    return ImageRepository(db_session)

@pytest_asyncio.fixture
async def detection_repo(db_session):
    return DetectionRepository(db_session)

@pytest_asyncio.fixture
async def image_version_repo(db_session):
    return ImageVersionRepository(db_session)

@pytest_asyncio.fixture
async def image_content_repo(db_session):
    return ImageContentRepository(db_session)

@pytest_asyncio.fixture
async def segmentation_repo(db_session):
    return SegmentationRepository(db_session)

@pytest_asyncio.fixture
async def edit_history_repo(db_session):
    return ImageEditHistoryRepository(db_session)

@pytest_asyncio.fixture
async def assets_repo(db_session):
    return AssetRepository(db_session)

@pytest_asyncio.fixture
async def mljob_repo(db_session):
    return MLJobRepository(db_session)

@pytest_asyncio.fixture
async def user_repo(db_session):
    return UserRepository(db_session)


@pytest.fixture
def mock_redis_history():
    """Mock for RedisHistory (undo/redo byte-stack) — separate storage
    class from RedisStorage (mock_redis_cache), never persisted to DB."""
    return AsyncMock()


@pytest.fixture
def mock_redis_assets():
    return AsyncMock()


@pytest.fixture
def mock_pipeline():
    """Mock MLPipeline — every ML service treats this as the boundary to
    the actual models; integration tests exercise real DB + repo wiring
    around it, not the models themselves."""
    return AsyncMock()


@pytest.fixture
def ml_service_kwargs(
    mock_s3_storage, mock_redis_cache, mock_redis_history, mock_redis_assets,
    image_repo, image_version_repo, image_content_repo, detection_repo,
    segmentation_repo, edit_history_repo, assets_repo, mock_pipeline,
):
    """Common BaseMLService constructor kwargs: real DB-backed repos +
    mocked S3/Redis/pipeline. Every ML service (DetectorService,
    SegmentationService, EditingService, AssetService, BaseMLService
    itself) takes exactly this kwarg set — build with
    `SomeService(**ml_service_kwargs)`."""
    return dict(
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


@pytest.fixture
def image_bytes():
    img = PILImage.new("RGB", (20, 20), "black")

    buf = BytesIO()
    img.save(buf, format="PNG")

    return buf.getvalue()

@pytest.fixture
def mock_s3_storage():
    storage = MagicMock()
    storage.upload = AsyncMock(return_value="s3://test-bucket/uploads/test.jpg")
    storage.upload_bytes = AsyncMock(return_value="s3://test-bucket/uploads/test.jpg")
    storage.download = AsyncMock(return_value=b"fake downloaded data")
    storage.delete = AsyncMock(return_value=True)
    storage.exists = AsyncMock(return_value=True)
    storage.get_presigned_url = AsyncMock(return_value="https://presigned.url/test.jpg")
    return storage


@pytest.fixture
def mock_redis_cache():
    cache = AsyncMock()
    
    _storage = {}
    
    async def mock_set(key, value, ttl=None):
        _storage[key] = value
    cache.set = mock_set
    
    async def mock_get(key):
        return _storage.get(key)
    cache.get = mock_get
    
    async def mock_delete(key):
        _storage.pop(key, None)
    cache.delete = mock_delete
    
    async def mock_exists(key):
        return key in _storage
    cache.exists = mock_exists
    
    async def mock_cache_image(image_id, image_data, suffix="processed", ttl=None):
        key = f"image:{image_id}:{suffix}"
        _storage[key] = image_data
        return key
    cache.cache_image = mock_cache_image
    
    async def mock_get_cached_image(image_id, suffix="processed"):
        key = f"image:{image_id}:{suffix}"
        return _storage.get(key)
    cache.get_cached_image = mock_get_cached_image
    
    async def mock_cache_detections(image_id, detections, ttl=None):
        key = f"detections:{image_id}"
        _storage[key] = detections
        return key
    cache.cache_detections = mock_cache_detections
    
    async def mock_get_cached_detections(image_id):
        key = f"detections:{image_id}"
        return _storage.get(key)
    cache.get_cached_detections = mock_get_cached_detections
    
    async def mock_invalidate_image(image_id):
        keys_to_delete = [k for k in _storage.keys() if f":{image_id}:" in k or k.endswith(f":{image_id}")]
        for key in keys_to_delete:
            _storage.pop(key, None)
    cache.invalidate_image = mock_invalidate_image
    
    return cache


@pytest_asyncio.fixture
async def sample_user(db_session):
    user = User(username="testuser", email="test@example.com", password_hash="hashed")
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user

@pytest.fixture(autouse=True)
def mock_mlflow_connection():
    mock_client = MagicMock()
    mock_client.get_experiment_by_name.return_value = None
    mock_client.create_experiment.return_value = "test-exp-id"
 
    with patch("mlflow.set_tracking_uri"), \
         patch("mlflow.set_experiment"), \
         patch("mlflow.create_experiment", return_value="test-exp-id"), \
         patch("mlflow.start_run"), \
         patch("mlflow.end_run"), \
         patch("mlflow.log_metric"), \
         patch("mlflow.set_tag"), \
         patch("mlflow.MlflowClient", return_value=mock_client):
        yield

@pytest.fixture
def tracker():
    tracker = MagicMock()

    tracker.log_metrics = MagicMock()
    tracker.log_params = MagicMock()
    tracker.start_run = MagicMock()
    tracker.end_run = MagicMock()

    return tracker


@pytest_asyncio.fixture
async def sample_image(db_session, sample_user):
    image = Image(
        filename="test.jpg",
        storage_path="s3://bucket/test.jpg",
        user_id=sample_user.id,
        mime_type="image/jpeg",
        width=100,
        height=100,
        file_size=1000,
        status="uploaded"
    )
    db_session.add(image)
    await db_session.commit()
    await db_session.refresh(image)
    return image


@pytest_asyncio.fixture
async def sample_detection(db_session, sample_image_version):
    """Detection is keyed by content_id (ImageContent), not image_id —
    depends on sample_image_version so the underlying image also gets a
    current_version_id, which detection lookups by image resolve through."""
    detection = Detection(
        content_id=sample_image_version.content_id,
        bbox_id=0,
        x1=10, y1=10, x2=100, y2=100,
        detected_class="person",
        confidence=0.95,
        is_active=True,
        model_name="yolov8",
        model_version="v1",
        inference_time_ms=12.5,
    )
    db_session.add(detection)
    await db_session.commit()
    await db_session.refresh(detection)
    return detection


@pytest_asyncio.fixture
async def multiple_images(db_session, sample_user):
    images = []
    for i in range(3):
        img = Image(
            filename=f"img{i}.jpg",
            storage_path=f"s3://bucket/img{i}.jpg",
            user_id=sample_user.id,
            mime_type="image/jpeg",
            width=100,
            height=100,
            file_size=1000,
        )
        db_session.add(img)
        images.append(img)
    await db_session.commit()
    for img in images:
        await db_session.refresh(img)
    return images


@pytest.fixture
def fake_mobile_sam_env(monkeypatch):
    """
    Fakes the `mobile_sam` package so MobileSAMSegmentor can be built
    without the real ViT-Tiny checkpoint or the mobile_sam dependency.
    Yields handles to the mocked model/predictor/auto-generator so tests
    can assert on how they were called.
    """
    model_instance = MagicMock(name="sam_model_instance")
    model_instance.to.return_value = model_instance
    model_instance.eval.return_value = model_instance

    sam_model_registry = {"vit_t": MagicMock(return_value=model_instance)}

    predictor_instance = MagicMock(name="predictor_instance")
    predictor_instance.set_image = MagicMock()
    predictor_instance.predict = MagicMock(
        return_value=(
            np.array([np.ones((32, 32), dtype=bool)]),
            np.array([0.95]),
            None,
        )
    )
    SamPredictor = MagicMock(return_value=predictor_instance)

    auto_generator_instance = MagicMock(name="auto_generator_instance")
    auto_generator_instance.generate = MagicMock(return_value=[])
    SamAutomaticMaskGenerator = MagicMock(return_value=auto_generator_instance)

    fake_module = types.ModuleType("mobile_sam")
    fake_module.sam_model_registry = sam_model_registry
    fake_module.SamPredictor = SamPredictor
    fake_module.SamAutomaticMaskGenerator = SamAutomaticMaskGenerator

    monkeypatch.setitem(sys.modules, "mobile_sam", fake_module)

    yield {
        "module": fake_module,
        "sam_model_registry": sam_model_registry,
        "model_instance": model_instance,
        "SamPredictor": SamPredictor,
        "predictor_instance": predictor_instance,
        "SamAutomaticMaskGenerator": SamAutomaticMaskGenerator,
        "auto_generator_instance": auto_generator_instance,
    }


@pytest.fixture
def segmentor(fake_mobile_sam_env, tracker):
    from importlib import import_module

    mod = import_module("app.ml.segmentor")

    return mod.MobileSAMSegmentor(
        model_path="fake_weights/mobile_sam.pt",
        model_type="vit_t",
        device="cpu",
        tracker=tracker,
    )
    
@pytest_asyncio.fixture
async def another_user(db_session):
    user = User(username="anotheruser", email="another@example.com", password_hash="hashed")
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest_asyncio.fixture
async def another_image(db_session, sample_user):
    image = Image(
        filename="another.jpg",
        storage_path="s3://bucket/another.jpg",
        user_id=sample_user.id,
        mime_type="image/jpeg",
        width=100,
        height=100,
        file_size=1000,
        status="uploaded"
    )
    db_session.add(image)
    await db_session.commit()
    await db_session.refresh(image)
    return image


@pytest_asyncio.fixture
async def sample_image_content(db_session):
    content = ImageContent(
        content_hash="sample_content_hash",
        storage_path="s3://bucket/content/sample.jpg",
        width=800,
        height=600,
        file_size=51200,
    )
    db_session.add(content)
    await db_session.commit()
    await db_session.refresh(content)
    return content


@pytest_asyncio.fixture
async def another_image_content(db_session):
    content = ImageContent(
        content_hash="another_content_hash",
        storage_path="s3://bucket/content/another.jpg",
        width=800,
        height=600,
        file_size=51200,
    )
    db_session.add(content)
    await db_session.commit()
    await db_session.refresh(content)
    return content


@pytest_asyncio.fixture
async def third_image_content(db_session):
    content = ImageContent(
        content_hash="third_content_hash",
        storage_path="s3://bucket/content/third.jpg",
        width=800,
        height=600,
        file_size=51200,
    )
    db_session.add(content)
    await db_session.commit()
    await db_session.refresh(content)
    return content


@pytest_asyncio.fixture
async def sample_image_version(db_session, sample_image, sample_image_content):
    version = ImageVersion(
        image_id=sample_image.id,
        content_id=sample_image_content.id,
        version_number=0,
        storage_path=sample_image.storage_path,
    )
    db_session.add(version)
    await db_session.commit()
    await db_session.refresh(version)

    sample_image.current_version_id = version.id
    db_session.add(sample_image)
    await db_session.commit()
    await db_session.refresh(version)

    return version


@pytest_asyncio.fixture
async def another_image_version(db_session, another_image, another_image_content):
    version = ImageVersion(
        image_id=another_image.id,
        content_id=another_image_content.id,
        version_number=0,
        storage_path=another_image.storage_path,
    )
    db_session.add(version)
    await db_session.commit()
    await db_session.refresh(version)

    another_image.current_version_id = version.id
    db_session.add(another_image)
    await db_session.commit()
    await db_session.refresh(version)

    return version


@pytest_asyncio.fixture
async def sample_mljob(db_session, sample_image_content, sample_image_version):
    job = MLJob(
        content_id=sample_image_content.id,
        image_version_id=sample_image_version.id,
        task_type=MLTaskType.DETECTION,
    )
    db_session.add(job)
    await db_session.commit()
    await db_session.refresh(job)
    return job


@pytest_asyncio.fixture
async def sample_segmentation_mask(db_session, sample_image_content):
    mask = SegmentationMask(
        content_id=sample_image_content.id,
        mask_id=0,
        mask_storage_path="s3://bucket/masks/sample_mask.png",
        preview_storage_path="s3://bucket/masks/sample_preview.png",
        x1=10, y1=10, x2=100, y2=100,
        area=8100.0,
        score=0.92,
        segmentation_mode=SegmentationMode.SAM,
        model_name="mobile_sam",
        model_version="v1",
        inference_time_ms=120.0,
    )
    db_session.add(mask)
    await db_session.commit()
    await db_session.refresh(mask)
    return mask


@pytest_asyncio.fixture
async def sample_edit_history_entry(db_session, sample_image_version):
    entry = ImageEditHistory(
        image_version_id=sample_image_version.id,
        operation=EditOperation.DETECT,
        engine=EngineType.YOLO,
        parameters={"confidence_threshold": 0.5},
        processing_time_ms=42,
    )
    db_session.add(entry)
    await db_session.commit()
    await db_session.refresh(entry)
    return entry