import aioboto3
from fastapi import UploadFile
from typing import Optional
from io import BytesIO
from app.config.settings import settings

class S3Storage:
    
    def __init__(self):
        
        self.bucket = settings.S3_BUCKET
        self.endpoint = settings.R2_ENDPOINT
        self.access_key = settings.ACCESS_KEY
        self.secret_key = settings.SECRET_KEY
        self.public_url = settings.R2_PUBLIC_URL
        self.session = aioboto3.Session()

    def _parse_key(self, path: str) -> str:

        if path.startswith('s3://'):
            parts = path[5:].split('/', 1)
            return parts[1] if len(parts) > 1 else parts[0]
        return path

    async def upload(self, file: UploadFile, path: str) -> str:
        async with self.session.client( # type: ignore
            's3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key
        ) as s3:
            await s3.upload_fileobj(
                file.file,
                self.bucket,
                path,
                ExtraArgs={'ContentType': file.content_type}
            )
        return f"s3://{self.bucket}/{path}"
    
    async def upload_bytes(self, data: bytes, path: str, content_type: str = 'image/jpeg') -> str:
        async with self.session.client( # type: ignore
            's3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key
        ) as s3:
            await s3.upload_fileobj(
                BytesIO(data),
                self.bucket,
                path,
                ExtraArgs={'ContentType': content_type}
            )
        return f"s3://{self.bucket}/{path}"
    
    async def download(self, path: str) -> bytes:
        key = self._parse_key(path)
        async with self.session.client( # type: ignore
            's3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key
        ) as s3:
            response = await s3.get_object(Bucket=self.bucket, Key=key) # type: ignore
            return await response['Body'].read()
        
    async def delete(self, path: str) -> bool:
        key = self._parse_key(path)
        async with self.session.client( # type: ignore
            's3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key
        ) as s3:
            await s3.delete_object(Bucket=self.bucket, Key=key)
        return True
    
    async def exists(self, path: str) -> bool:
        key = self._parse_key(path)
        try:
            async with self.session.client( # type: ignore
                's3',
                endpoint_url=self.endpoint,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key
            ) as s3:
                await s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except:
            return False
        
    async def get_presigned_url(self, path: str, expiration: int = 3600) -> str:
        key = self._parse_key(path)
        return f"{self.public_url.rstrip('/')}/{key}"