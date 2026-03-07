from pydantic import BaseModel

class ImageBase(BaseModel):
    filename: str
    file_path: str

class ImageCreate(ImageBase):
    user_id:int

class ImageResponse(ImageBase):
    id: int
    class Config:
        from_attributes = True
