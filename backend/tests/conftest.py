import pytest
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool
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
from app.repository.image_repo import ImageRepository
from app.repository.detection_repo import DetectionRepository
from app.config.test_settings import test_settings

TEST_DATABASE = test_settings.TEST_DATABASE_URL

@pytest.fixture(scope = 'session')
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
    
@pytest_asyncio.fixture(scope = 'function')
async def db_engine():
    engine = create_async_engine(TEST_DATABASE, echo = False, poolclass = NullPool)
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        
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
        yield session
        await session.rollback()
        
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
        status="uploaded"
    )
    db_session.add(image)
    await db_session.commit()
    await db_session.refresh(image)
    return image


@pytest_asyncio.fixture
async def sample_detection(db_session, sample_image):
    detection = Detection(
        image_id=sample_image.id,
        bbox_id=0,         
        x1=10, y1=10, x2=100, y2=100,
        detected_class="person",
        confidence=0.95
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
            user_id=sample_user.id
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