from pydantic import BaseModel


class BboxSchema(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int