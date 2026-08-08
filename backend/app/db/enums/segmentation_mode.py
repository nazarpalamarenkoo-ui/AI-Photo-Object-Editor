from enum import Enum

class SegmentationMode(str, Enum):
    SAM = "sam"
    HYBRID = "hybrid"
    POLYGON = "polygon" 