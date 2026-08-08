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
    DIFFUSION_DEVICE: str = 'cuda'

    # DB — connection pool
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_POOL_TIMEOUT: int = 30
    DB_POOL_RECYCLE: int = 1800
    DB_POOL_PRE_PING: bool = True
    DB_USE_NULLPOOL: bool = False

    DIFFUSION_INPAINT_MODEL_ID: str = "stable-diffusion-v1-5/stable-diffusion-inpainting"
    DIFFUSION_STEPS: int = 30
    DIFFUSION_GUIDANCE_SCALE: float = 5.5
    DIFFUSION_STRENGTH: float = 0.7
    DIFFUSION_WORK_RESOLUTION: int = 640
    DIFFUSION_CROP_PADDING_RATIO: float = 0.35
    DIFFUSION_MIN_CROP_SIZE: int = 256
    DIFFUSION_MASK_BLUR_RADIUS: int = 6
    DIFFUSION_ENABLE_CPU_OFFLOAD: bool = False
    DIFFUSION_NEGATIVE_PROMPT: str = "blurry, distorted, low quality, deformed, artifacts, extra limbs, extra fingers, missing fingers, fused fingers, mutated hands, bad anatomy, disfigured face, asymmetrical face, cloned face, long neck, malformed limbs, floating limbs, disconnected body parts, duplicate"
    DIFFUSION_PROMPT_FALLBACK: str = "high quality, detailed, photorealistic"
    IP_ADAPTER_REPO: str = "h94/IP-Adapter"
    IP_ADAPTER_SUBFOLDER: str = "models"
    IP_ADAPTER_IMAGE_ENCODER_SUBFOLDER: str = "models/image_encoder"
    IP_ADAPTER_VARIANT: str = "plus"
    IP_ADAPTER_SCALE: float = 0.8

    # App
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()