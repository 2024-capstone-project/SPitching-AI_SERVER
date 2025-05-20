import boto3
import os
from dotenv import load_dotenv
from botocore.exceptions import NoCredentialsError

load_dotenv() # .env 파일을 로드

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION_NAME = os.getenv("AWS_REGION", "ap-northeast-2")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")


def upload_file_to_s3(file_bytes: bytes, s3_key: str) -> str:
    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION_NAME
        )

        s3.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
            Body=file_bytes,
            ContentType="video/mp4"
        )

        return f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION_NAME}.amazonaws.com/{s3_key}"

    except NoCredentialsError:
        raise Exception("AWS 자격 증명 오류")