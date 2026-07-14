from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str
    ALEMBIC_DATABASE_URL: str

    # S3
    R2_ENDPOINT: str
    S3_BUCKET: str
    ACCESS_KEY: str
    SECRET_KEY: str
    R2_PUBLIC_URL: str

    # Cache
    CACHE_TYPE: str = "pickle"

    # Redis
    REDIS_URL: str
    REDIS_HOST: str
    REDIS_PORT: int

    # Auth
    SECRET_KEY_AUTH: str
    MAIL_USERNAME: str
    MAIL_PASSWORD: str
    MAIL_FROM: str
    MAIL_PORT: int
    MAIL_SERVER: str
    MAIL_STARTTLS: bool
    MAIL_SSL_TLS: bool
    USE_CREDENTIALS: bool

    # Devices
    DEFAULT_DEVICE: str = "cpu"
    YOLO_DEVICE: str = "cpu"
    SAM_DEVICE: str = "cuda"
    LAMA_DEVICE: str = "cuda"

    # App
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()