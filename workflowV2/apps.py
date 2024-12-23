from django.apps import AppConfig
from django.conf import settings
from .utils import minio_client

class Workflowv2Config(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "workflowV2"

    def ready(self):
        import workflow.signals

        # Debug prints
        print("\nMinIO Configuration:")
        print(f"Base URL: {settings.MINIO_BASE_URL}")
        print(f"Access Key: {settings.MINIO_ACCESS_KEY}")
        print(f"Bucket Name: {settings.MINIO_BUCKET_NAME}")
        print(f"Secure Connection: {settings.MINIO_SECURE_CONN}")

        bucket_name = settings.MINIO_BUCKET_NAME
        try:
            if not minio_client.bucket_exists(bucket_name):
                minio_client.make_bucket(bucket_name)
                print(f"Created bucket {bucket_name}")
            else:
                print(f"Bucket {bucket_name} already exists")
        except Exception as e:
            print(f"MinIO Error: {str(e)}")
            print(f"Error Type: {type(e)}")