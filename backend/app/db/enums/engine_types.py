from enum import Enum


class EngineType(str, Enum):
    YOLO = "yolo"
    SAM = "sam"
    LAMA = "lama"
    DIFFUSION = "diffusion"