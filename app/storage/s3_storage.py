import aioboto3
from fastapi import UploadFile
from typing import Optional
from io import BytesIO
from app.config.settings import settings

class S3Storage:
    
    def init(self):
        
        self.bucket = settings.S3_BUCKET
        self.endpoint = settings.R2_ENDPOINT
        self.access_key = settings.ACCESS_KEY
        self.secret_key = settings.SECRET_KEY
        self.session = aioboto3.Session()
        
    async def upload(
        self,
        file: UploadFile,
        path: str
    ) -> str:
        
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
    
    async def upload_bytes(
        self,
        data: bytes,
        path: str,
        content_type: str = 'image/jpeg'
    ) -> str:
        
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
                ExtraArgs = {'ContentType': content_type}
            )
        
        return f"s3://{self.bucket}/{path}"
    
    async def download(self, path: str) -> bytes:
        
        if path.startswith('s3://'):
            path = path.replace(f's3://{self.bucket}/', '')
            
        async with self.session.client( # type: ignore
            's3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key
        ) as s3:
            response = await s3.get_object( # type: ignore
                Bucket = self.bucket,
                Key = path
            )
            
            data = await response['Body'].read()
            return data
        
        
    async def delete(self, path: str) -> bool:
        
        if path.startswith('s3://'):
            path = path.replace(f's3://{self.bucket}/', '')
            
        async with self.session.client( # type: ignore
            's3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key
        ) as s3:
            await s3.delete_object(
                Bucket = self.bucket,
                Key = path
            )
        
        return True
    
    async def exsists(self, path: str) -> bool:
        
        if path.startswith('s3://'):
            path = path.replace(f's3://{self.bucket}/', '')
        
        try:
            async with self.session.client( # type: ignore
                's3',
                endpoint_url=self.endpoint,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key
            ) as s3:
                await s3.head_object(
                    Bucket=self.bucket,
                    Key=path
                )
            return True
        except:
            return False
        
        
    async def get_presigned_url(
        self,
        path: str,
        expiration: int = 3600
    ) -> str:
        
        if path.startswith('s3://'):
            path = path.replace(f's3://{self.bucket}/', '')
        
        async with self.session.client( # type: ignore
            's3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key
        ) as s3:
            url = await s3.generate_presigned_url(
                'get_object',
                Params = {
                    'Bucket':self.bucket,
                    'Key':path
                },
            )
        
        return url