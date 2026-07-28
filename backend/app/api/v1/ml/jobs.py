from arq.connections import ArqRedis
from arq.jobs import Job, JobStatus
from fastapi import APIRouter, Depends, HTTPException

from app.api.auth.auth import get_current_user
from app.db.models.user import User

from .deps import get_arq_pool

router = APIRouter(tags=["ML - Jobs"])


@router.get("/jobs/{job_id}")
async def get_job_status(
    job_id: str,
    current_user: User = Depends(get_current_user),
    pool: ArqRedis = Depends(get_arq_pool),
):
    """
    Poll status/result of an enqueued job.

    status: "deferred" | "queued" | "in_progress" | "complete" | "not_found"
    result: present only when status == "complete" and the task succeeded
    error:  present only when status == "complete" and the task raised
    """
    job = Job(job_id, pool)
    status = await job.status()

    if status == JobStatus.not_found:
        raise HTTPException(status_code=404, detail="Job not found")

    response = {"job_id": job_id, "status": status.value}

    if status == JobStatus.complete:
        result_info = await job.result_info()
        if result_info is not None:
            if result_info.success:
                response["result"] = result_info.result
            else:
                response["error"] = str(result_info.result)

    return response