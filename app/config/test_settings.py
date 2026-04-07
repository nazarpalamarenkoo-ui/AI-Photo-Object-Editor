from pydantic_settings import BaseSettings

class TestSettings(BaseSettings):
    TEST_DATABASE_URL: str

    class Config:
        env_file = ".env.test"
        env_file_encoding = "utf-8"
        extra = "ignore"

test_settings = TestSettings()