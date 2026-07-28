from fastapi import APIRouter

from . import detect, editing, segmentation, sam_ops, assets, jobs

router = APIRouter(prefix="/ml", tags=["ML"])

router.include_router(detect.router)
router.include_router(editing.router)
router.include_router(segmentation.router)
router.include_router(sam_ops.router)
router.include_router(assets.router)
router.include_router(jobs.router)