from enum import Enum


class EditOperation(str, Enum):
    DETECT = "detect"
    SEGMENT = "segment"
    REMOVE = "remove"
    REPLACE = "replace"