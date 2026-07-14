from typing import Dict
from io import BytesIO
from PIL import Image

from app.core.logging import get_logger

logger = get_logger(__name__)


class Validator:
    @staticmethod
    def validate_image_bytes(image_bytes: bytes) -> None:
        if not image_bytes:
            logger.warning("validation_failed", reason="empty_image_bytes")
            raise ValueError("image_bytes cannot be empty")
        if not isinstance(image_bytes, bytes):
            logger.warning("validation_failed", reason="image_bytes_wrong_type")
            raise ValueError("image_bytes must be bytes")
        try:
            img = Image.open(BytesIO(image_bytes))
            img.verify()
        except Exception as e:
            logger.warning("validation_failed", reason="invalid_image_bytes", error=str(e))
            raise ValueError(f"Invalid image bytes: {e}")

    @staticmethod
    def validate_mask_bytes(mask_bytes: bytes) -> None:
        if not mask_bytes:
            logger.warning("validation_failed", reason="empty_mask_bytes")
            raise ValueError("mask_bytes cannot be empty")
        if not isinstance(mask_bytes, bytes):
            logger.warning("validation_failed", reason="mask_bytes_wrong_type")
            raise ValueError("mask_bytes must be bytes")
        try:
            img = Image.open(BytesIO(mask_bytes))
            img.verify()
        except Exception as e:
            logger.warning("validation_failed", reason="invalid_mask_bytes", error=str(e))
            raise ValueError(f"Invalid mask bytes: {e}")

    @staticmethod
    def validate_bbox(bbox: Dict[str, int]) -> None:
        required_keys = ["x1", "y1", "x2", "y2"]
        if not isinstance(bbox, dict):
            logger.warning("validation_failed", reason="bbox_wrong_type")
            raise ValueError("bbox must be a dict")
        for key in required_keys:
            if key not in bbox:
                logger.warning("validation_failed", reason="bbox_missing_key", missing_key=key)
                raise ValueError(f"bbox missing required key: {key}")
        if bbox["x1"] >= bbox["x2"]:
            logger.warning("validation_failed", reason="bbox_x1_not_less_than_x2", bbox=bbox)
            raise ValueError("bbox x1 must be < x2")
        if bbox["y1"] >= bbox["y2"]:
            logger.warning("validation_failed", reason="bbox_y1_not_less_than_y2", bbox=bbox)
            raise ValueError("bbox y1 must be < y2")
        if any(bbox[key] < 0 for key in required_keys):
            logger.warning("validation_failed", reason="bbox_negative_coordinate", bbox=bbox)
            raise ValueError("bbox coordinates must be >= 0")


_validator_instance = None

def get_validator() -> Validator:
    """
    Singleton getter for Validator.

    Returns:
        Validator instance
    """
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = Validator()
    return _validator_instance