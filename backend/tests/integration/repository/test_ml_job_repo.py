import pytest
from app.repository.mljob_repo import MLJobRepository
from app.db.enums.ml_task_status import MLTaskType
from app.db.enums.ml_job_status import JobStatus


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_create_mljob(db_session, sample_image_content, sample_image_version):
    """Test creating a new MLJob with PENDING status"""
    repo = MLJobRepository(db_session)

    job = await repo.create(
        content_id=sample_image_content.id,
        image_version_id=sample_image_version.id,
        task_type=MLTaskType.DETECTION,
    )

    assert job.id is not None
    assert job.content_id == sample_image_content.id
    assert job.image_version_id == sample_image_version.id
    assert job.task_type == MLTaskType.DETECTION
    assert job.status == JobStatus.PENDING


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_get_mljob_by_id(db_session, sample_mljob):
    """Test fetching MLJob by primary key"""
    repo = MLJobRepository(db_session)

    job = await repo.get_by_id(sample_mljob.id)

    assert job is not None
    assert job.id == sample_mljob.id


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_get_mljob_by_id_not_found(db_session):
    """Test fetching non-existent job returns None"""
    repo = MLJobRepository(db_session)

    job = await repo.get_by_id(999999)

    assert job is None


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_get_mljob_by_content(db_session, sample_image_content, sample_image_version):
    """Test fetching all jobs for a given content_id"""
    repo = MLJobRepository(db_session)

    j1 = await repo.create(sample_image_content.id, sample_image_version.id, MLTaskType.DETECTION)
    j2 = await repo.create(sample_image_content.id, sample_image_version.id, MLTaskType.SEGMENTATION)

    jobs = await repo.get_by_content(sample_image_content.id)

    ids = [j.id for j in jobs]
    assert j1.id in ids
    assert j2.id in ids


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_get_mljob_by_version(db_session, sample_image_content, sample_image_version):
    """Test fetching all jobs for a given image_version_id"""
    repo = MLJobRepository(db_session)

    job = await repo.create(sample_image_content.id, sample_image_version.id, MLTaskType.DETECTION)

    jobs = await repo.get_by_version(sample_image_version.id)

    assert any(j.id == job.id for j in jobs)


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_mark_running(db_session, sample_mljob):
    """Test transitioning job status to RUNNING"""
    repo = MLJobRepository(db_session)

    job = await repo.mark_running(sample_mljob.id)

    assert job.status == JobStatus.RUNNING


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_mark_success(db_session, sample_mljob):
    """Test transitioning job status to SUCCESS with processing time"""
    repo = MLJobRepository(db_session)

    job = await repo.mark_success(sample_mljob.id, processing_time_ms=350)

    assert job.status == JobStatus.SUCCESS
    assert job.processing_time_ms == 350
    assert job.finished_at is not None


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_mark_failed(db_session, sample_mljob):
    """Test transitioning job status to FAILED with error message"""
    repo = MLJobRepository(db_session)

    job = await repo.mark_failed(sample_mljob.id, error_message="CUDA out of memory")

    assert job.status == JobStatus.FAILED
    assert job.error_message == "CUDA out of memory"
    assert job.finished_at is not None


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_get_successful_returns_latest(db_session, sample_image_content, sample_image_version):
    """get_successful must return the most recent SUCCESS job for content+task_type"""
    repo = MLJobRepository(db_session)

    j1 = await repo.create(sample_image_content.id, sample_image_version.id, MLTaskType.DETECTION)
    j2 = await repo.create(sample_image_content.id, sample_image_version.id, MLTaskType.DETECTION)

    await repo.mark_success(j1.id, processing_time_ms=100)
    await repo.mark_success(j2.id, processing_time_ms=200)

    result = await repo.get_successful(sample_image_content.id, MLTaskType.DETECTION)

    assert result is not None
    assert result.id == j2.id


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_get_successful_returns_none_when_absent(db_session, sample_image_content, sample_image_version):
    """get_successful must return None when no SUCCESS job exists"""
    repo = MLJobRepository(db_session)

    await repo.create(sample_image_content.id, sample_image_version.id, MLTaskType.DETECTION)
    # left in PENDING — no success

    result = await repo.get_successful(sample_image_content.id, MLTaskType.DETECTION)

    assert result is None


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_get_pending_returns_only_pending(db_session, sample_image_content, sample_image_version):
    """get_pending must not include RUNNING or FAILED jobs"""
    repo = MLJobRepository(db_session)

    pending_job = await repo.create(
        sample_image_content.id, sample_image_version.id, MLTaskType.DETECTION
    )
    running_job = await repo.create(
        sample_image_content.id, sample_image_version.id, MLTaskType.SEGMENTATION
    )
    await repo.mark_running(running_job.id)

    pending = await repo.get_pending()

    ids = [j.id for j in pending]
    assert pending_job.id in ids
    assert running_job.id not in ids


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_get_pending_filtered_by_task_type(db_session, sample_image_content, sample_image_version):
    """get_pending with task_type filter must only return matching jobs"""
    repo = MLJobRepository(db_session)

    det_job = await repo.create(
        sample_image_content.id, sample_image_version.id, MLTaskType.DETECTION
    )
    seg_job = await repo.create(
        sample_image_content.id, sample_image_version.id, MLTaskType.SEGMENTATION
    )

    pending_det = await repo.get_pending(task_type=MLTaskType.DETECTION)

    ids = [j.id for j in pending_det]
    assert det_job.id in ids
    assert seg_job.id not in ids