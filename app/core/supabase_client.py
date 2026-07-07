"""
Supabase client for cloud storage.
Handles file upload to Supabase Storage Buckets.
"""
import logging
from supabase import create_client, Client
from app.core.config import get_settings

_supabase_client = None
logger = logging.getLogger(__name__)

def get_supabase_client() -> Client:
    global _supabase_client
    if _supabase_client is None:
        settings = get_settings()
        if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
            raise ValueError("SUPABASE_URL or SUPABASE_KEY is missing")
            
        _supabase_client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
        
        # Ensure bucket exists
        try:
            buckets = _supabase_client.storage.list_buckets()
            bucket_names = [b.name for b in buckets]
            if settings.SUPABASE_BUCKET not in bucket_names:
                _supabase_client.storage.create_bucket(settings.SUPABASE_BUCKET, options={"public": False})
                logger.info(f"Created missing Supabase bucket: {settings.SUPABASE_BUCKET}")
        except Exception as e:
            logger.warning(f"Could not check/create bucket: {e}")
            
    return _supabase_client


def upload_file(file_data: bytes, object_name: str, content_type: str = "application/pdf") -> str:
    """Upload a file to Supabase Storage and return the object path."""
    settings = get_settings()
    client = get_supabase_client()
    client.storage.from_(settings.SUPABASE_BUCKET).upload(
        path=object_name,
        file=file_data,
        file_options={"content-type": content_type}
    )
    return f"{settings.SUPABASE_BUCKET}/{object_name}"


def download_file(object_name: str) -> bytes:
    """Download a file from Supabase Storage."""
    settings = get_settings()
    client = get_supabase_client()
    return client.storage.from_(settings.SUPABASE_BUCKET).download(object_name)
