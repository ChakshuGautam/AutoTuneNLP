from django.conf import settings
from minio import Minio

print("\nMinIO Client Initialization:")
print(f"Endpoint utils: {settings.MINIO_BASE_URL}")
print(f"Access Key utils: {settings.MINIO_ACCESS_KEY}")
print(f"Secure utils: {settings.MINIO_SECURE_CONN}")

minio_client = Minio(
    settings.MINIO_BASE_URL,
    access_key=settings.MINIO_ACCESS_KEY,
    secret_key=settings.MINIO_SECRET_KEY,
    secure=settings.MINIO_SECURE_CONN == "True"
)