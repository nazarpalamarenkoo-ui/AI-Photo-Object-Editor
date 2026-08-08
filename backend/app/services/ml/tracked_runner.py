from typing import Any, Optional, Type

from app.db.enums.ml_task_status import MLTaskType
from app.services.ml_job_service import MLJobService
from app.core.logging import get_logger

logger = get_logger(__name__)


async def run_tracked(
    service_cls: Type[Any],
    deps: dict,
    mljob_service: MLJobService,
    method_name: str,
    image_id: int,
    user_id: int,
    task_type: MLTaskType,
    extra_ctor_kwargs: Optional[dict] = None,
    **method_kwargs,
) -> dict:
    """
    Single choke point for "instantiate an ML service and run one of its
    methods with an MLJob row wrapped around it". Used by BOTH:

      - worker.py arq tasks (deps come from _build_ml_deps, which also
        opens/closes the Redis connections for the task's lifetime)
      - sync FastAPI routes (deps come from deps._base_deps via
        get_base_deps, request-scoped like everything else in that layer)
    """
    ctor_kwargs = dict(deps)
    if extra_ctor_kwargs:
        ctor_kwargs.update(extra_ctor_kwargs)

    service = service_cls(**ctor_kwargs)

    async with mljob_service.track(image_id, user_id, task_type) as job:
        logger.info("ml_job_tracked", job_id=job.id, task_type=task_type.value)
        return await getattr(service, method_name)(
            image_id=image_id, user_id=user_id, **method_kwargs
        )