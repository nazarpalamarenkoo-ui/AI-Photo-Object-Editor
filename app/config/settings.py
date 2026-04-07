from pickletools import int4

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from pathlib import Path

class Settings(BaseSettings):
    
    # Database
    DATABASE_URL: str
    ALEMBIC_DATABASE_URL: str
    
    # S3/MinIO
    R2_ENDPOINT: str 
    S3_BUCKET: str 
    ACCESS_KEY: str 
    SECRET_KEY: str  
    
    # Cache
    CACHE_TYPE: str = "pickle"
    
    # Redis
    REDIS_URL: str
    REDIS_HOST: str
    REDIS_PORT: int
    
    # AUTH
    SECRET_KEY_AUTH: str
    MAIL_USERNAME: str
    MAIL_PASSWORD: str
    MAIL_FROM: str
    MAIL_PORT: int
    MAIL_SERVER: str
    MAIL_STARTTLS: bool
    MAIL_SSL_TLS: bool
    USE_CREDENTIALS: bool
    
    # App
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
        
settings = Settings()