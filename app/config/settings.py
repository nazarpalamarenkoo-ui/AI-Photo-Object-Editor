from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from pathlib import Path

class Settings(BaseSettings):
    
    # Database
    DATABASE_URL: str
    
    # Cache
    cache_type: str = "pickle"  # "pickle" or "redis" (use pickle for dev)
    #cache_dir: Path = Path("data_pipeline/outputs/cache")
    redis_url: str = "redis://localhost:6379/0"
    redis_host: str = "localhost"
    redis_port: int = 6379
    
    # Logging
    debug: bool = False
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        
settings = Settings()