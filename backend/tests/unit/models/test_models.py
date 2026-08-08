import pytest

from app.db.models.user import User
from app.db.models.image import Image
from app.db.models.detection import Detection
from app.db.models.assets import Asset
from app.db.models.image_content import ImageContent
from app.db.models.image_edit_history import ImageEditHistory
from app.db.models.image_version import ImageVersion
from app.db.models.mljobs import MLJob
from app.db.models.segmentation import SegmentationMask

from app.db.enums.image_status import ImageStatus
from app.db.enums.edit_operation import EditOperation
from app.db.enums.engine_types import EngineType
from app.db.enums.ml_job_status import JobStatus
from app.db.enums.ml_task_status import MLTaskType
from app.db.enums.segmentation_mode import SegmentationMode


@pytest.mark.unit
class TestUserModel:
    def test_user_creation(self):
        user = User(username="test", email="test@test.com", password_hash="hash")
        assert user.username == "test"
        assert user.email == "test@test.com"
        assert user.password_hash == "hash"

    def test_user_tablename(self):
        assert User.__tablename__ == "users"

    def test_user_repr(self):
        user = User(id=1, username="john")
        assert "User" in repr(user)
        assert "john" in repr(user)

    def test_user_repr_contains_id(self):
        user = User(id=42, username="jane")
        assert "42" in repr(user)


@pytest.mark.unit
class TestImageModel:
    def test_image_creation(self):
        image = Image(filename="test.jpg", storage_path="s3://test.jpg", user_id=1)
        assert image.filename == "test.jpg"
        assert image.user_id == 1

    def test_image_explicit_status(self):
        image = Image(
            filename="test.jpg",
            storage_path="s3://test.jpg",
            user_id=1,
            status=ImageStatus.UPLOADED,
        )
        assert image.status == ImageStatus.UPLOADED

    def test_image_status_accepts_string_value(self):
        image = Image(
            filename="test.jpg",
            storage_path="s3://test.jpg",
            user_id=1,
            status="ready",
        )
        assert image.status == "ready"

    def test_image_tablename(self):
        assert Image.__tablename__ == "images"

    def test_image_optional_fields_default_none(self):
        image = Image(filename="test.jpg", storage_path="s3://test.jpg", user_id=1)
        assert image.cache_key is None
        assert image.current_version_id is None

@pytest.mark.unit
class TestDetectionModel:
    def test_detection_creation(self):
        det = Detection(
            content_id=1,
            bbox_id=1,
            x1=10,
            y1=10,
            x2=100,
            y2=100,
            detected_class="person",
            confidence=0.9,
            model_name="yolo",
            model_version="v8",
            inference_time_ms=12.5,
        )
        assert det.x2 > det.x1
        assert det.y2 > det.y1
        assert det.detected_class == "person"
        assert det.confidence == 0.9

    def test_detection_tablename(self):
        assert Detection.__tablename__ == "detections"

    def test_detection_unique_constraint_columns(self):
        constraint = Detection.__table_args__[0]
        assert constraint.name == "uq_detection_content_bbox"
        assert set(constraint.columns.keys()) == {"content_id", "bbox_id"}

@pytest.mark.unit
class TestAssetModel:
    def test_asset_creation(self):
        asset = Asset(
            user_id=1,
            storage_path="s3://asset.png",
            width=100,
            height=200,
            area_pixels=20000,
        )
        assert asset.user_id == 1
        assert asset.width == 100
        assert asset.height == 200
        assert asset.area_pixels == 20000

    def test_asset_tablename(self):
        assert Asset.__tablename__ == "assets"

    def test_asset_optional_source_fields_default_none(self):
        asset = Asset(
            user_id=1,
            storage_path="s3://asset.png",
            width=10,
            height=10,
            area_pixels=100,
        )
        assert asset.source_image_version_id is None
        assert asset.source_segmentation_mask_id is None
        assert asset.thumbnail_path is None
        assert asset.label is None

@pytest.mark.unit
class TestImageContentModel:
    def test_image_content_creation(self):
        content = ImageContent(
            content_hash="a" * 64,
            storage_path="s3://content.png",
            width=100,
            height=100,
            file_size=1024,
        )
        assert content.content_hash == "a" * 64
        assert content.file_size == 1024

    def test_image_content_tablename(self):
        assert ImageContent.__tablename__ == "image_contents"

    def test_hash_bytes_returns_sha256_hex_digest(self):
        digest = ImageContent.hash_bytes(b"hello world")
        assert digest == (
            "b94d27b9934d3e08a52e52d7da7dabfa"
            "c484efe37a5380ee9088f7ace2efcde9"
        )
        assert len(digest) == 64

    def test_hash_bytes_is_deterministic(self):
        data = b"same-bytes"
        assert ImageContent.hash_bytes(data) == ImageContent.hash_bytes(data)

    def test_hash_bytes_differs_for_different_input(self):
        assert ImageContent.hash_bytes(b"a") != ImageContent.hash_bytes(b"b")

@pytest.mark.unit
class TestImageVersionModel:
    def test_image_version_creation(self):
        version = ImageVersion(
            image_id=1,
            content_id=1,
            version_number=1,
            storage_path="s3://v1.png",
        )
        assert version.image_id == 1
        assert version.content_id == 1
        assert version.version_number == 1

    def test_image_version_tablename(self):
        assert ImageVersion.__tablename__ == "image_versions"

    def test_image_version_parent_defaults_none(self):
        version = ImageVersion(
            image_id=1,
            content_id=1,
            version_number=1,
            storage_path="s3://v1.png",
        )
        assert version.parent_version_id is None

    def test_image_version_unique_constraint_columns(self):
        constraint = ImageVersion.__table_args__[0]
        assert constraint.name == "uq_image_version_number"
        assert set(constraint.columns.keys()) == {"image_id", "version_number"}

@pytest.mark.unit
class TestImageEditHistoryModel:
    def test_edit_history_creation(self):
        entry = ImageEditHistory(
            image_version_id=1,
            operation=EditOperation.REMOVE,
            engine=EngineType.LAMA,
        )
        assert entry.operation == EditOperation.REMOVE
        assert entry.engine == EngineType.LAMA

    def test_edit_history_tablename(self):
        assert ImageEditHistory.__tablename__ == "image_edit_history"

    def test_edit_history_optional_fields_default_none(self):
        entry = ImageEditHistory(
            image_version_id=1,
            operation=EditOperation.DETECT,
            engine=EngineType.YOLO,
        )
        assert entry.parameters is None
        assert entry.processing_time_ms is None

@pytest.mark.unit
class TestMLJobModel:
    def test_ml_job_creation(self):
        job = MLJob(
            content_id=1,
            image_version_id=1,
            task_type=MLTaskType.DETECTION,
        )
        assert job.content_id == 1
        assert job.image_version_id == 1
        assert job.task_type == MLTaskType.DETECTION

    def test_ml_job_tablename(self):
        assert MLJob.__tablename__ == "ml_jobs"

    def test_ml_job_explicit_status(self):
        job = MLJob(
            content_id=1,
            image_version_id=1,
            task_type=MLTaskType.SEGMENTATION,
            status=JobStatus.RUNNING,
        )
        assert job.status == JobStatus.RUNNING

    def test_ml_job_optional_fields_default_none(self):
        job = MLJob(
            content_id=1,
            image_version_id=1,
            task_type=MLTaskType.DETECTION,
        )
        assert job.finished_at is None
        assert job.processing_time_ms is None
        assert job.error_message is None

@pytest.mark.unit
class TestSegmentationMaskModel:
    def test_segmentation_mask_creation(self):
        mask = SegmentationMask(
            content_id=1,
            mask_id=1,
            mask_storage_path="s3://mask.png",
            preview_storage_path="s3://preview.png",
            x1=0,
            y1=0,
            x2=50,
            y2=50,
            area=2500.0,
            score=0.95,
            segmentation_mode=SegmentationMode.SAM,
            model_name="sam",
            model_version="v1",
            inference_time_ms=15.0,
        )
        assert mask.x2 > mask.x1
        assert mask.y2 > mask.y1
        assert mask.segmentation_mode == SegmentationMode.SAM
        assert mask.score == 0.95

    def test_segmentation_mask_tablename(self):
        assert SegmentationMask.__tablename__ == "segmentation_masks"

    def test_segmentation_mask_unique_constraint_columns(self):
        constraint = SegmentationMask.__table_args__[0]
        assert constraint.name == "uq_segmentation_content_mask"
        assert set(constraint.columns.keys()) == {"content_id", "mask_id"}

@pytest.mark.unit
class TestImageStatusEnum:
    def test_members(self):
        assert ImageStatus.UPLOADED.value == "uploaded"
        assert ImageStatus.PROCESSING.value == "processing"
        assert ImageStatus.READY.value == "ready"
        assert ImageStatus.FAILED.value == "failed"
        assert ImageStatus.DELETED.value == "deleted"

    def test_is_str_enum(self):
        assert ImageStatus.UPLOADED == "uploaded"


@pytest.mark.unit
class TestEditOperationEnum:
    def test_members(self):
        assert {e.value for e in EditOperation} == {
            "detect",
            "segment",
            "remove",
            "replace",
        }


@pytest.mark.unit
class TestEngineTypeEnum:
    def test_members(self):
        assert {e.value for e in EngineType} == {
            "yolo",
            "sam",
            "lama",
            "diffusion",
        }


@pytest.mark.unit
class TestJobStatusEnum:
    def test_members(self):
        assert {e.value for e in JobStatus} == {
            "pending",
            "running",
            "success",
            "failed",
        }


@pytest.mark.unit
class TestMLTaskTypeEnum:
    def test_members_include_core_tasks(self):
        values = {e.value for e in MLTaskType}
        assert "detection" in values
        assert "segmentation" in values
        assert "remove_object" in values
        assert "replace_object" in values
        assert "extract_object" in values

    def test_no_duplicate_values(self):
        values = [e.value for e in MLTaskType]
        assert len(values) == len(set(values))


@pytest.mark.unit
class TestSegmentationModeEnum:
    def test_members(self):
        assert {e.value for e in SegmentationMode} == {"sam", "hybrid", "polygon"}