from pydantic import BaseModel, EmailStr
from typing import Optional

class TokenData(BaseModel):
    username: Optional[str] = None

class Token(BaseModel):
    access_token: str
    token_type: str

class SignUpArgs(BaseModel):
    email:EmailStr
    password: str
    username:str

class SignInArgs(BaseModel):
    email:EmailStr
    password: str