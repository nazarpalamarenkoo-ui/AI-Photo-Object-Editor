from pydantic import BaseModel


class ModelMeta(BaseModel):
    model_name: str
    model_version: str
    inference_time_ms: float