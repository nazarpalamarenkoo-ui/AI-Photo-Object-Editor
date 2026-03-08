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
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    
    # App
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        
settings = Settings()