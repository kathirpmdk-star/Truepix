"""
Storage Manager - Supabase Object Storage Integration
Handles image uploads to Supabase Storage bucket
"""

import os
import io
from typing import Optional
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()


class StorageManager:
    """Manages image uploads to Supabase Storage"""
    
    def __init__(self):
        """Initialize Supabase client and storage bucket"""
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        self.bucket_name = os.getenv("SUPABASE_BUCKET", "truepix-images")
        
        if not self.supabase_url or not self.supabase_key:
            print("⚠️  Warning: Supabase credentials not configured")
            print("   Using mock storage mode for demo")
            self.mock_mode = True
            self.client = None
        else:
            try:
                self.client: Client = create_client(self.supabase_url, self.supabase_key)
                self.mock_mode = False
                self._ensure_bucket_exists()
            except Exception as e:
                print(f"⚠️  Supabase connection failed: {e}")
                print("   Using mock storage mode for demo")
                self.mock_mode = True
                self.client = None
    
    def _ensure_bucket_exists(self):
        """Create storage bucket if it doesn't exist"""
        try:
            buckets = self.client.storage.list_buckets()
            bucket_names = [b['name'] for b in buckets]
            
            if self.bucket_name not in bucket_names:
                self.client.storage.create_bucket(
                    self.bucket_name,
                    options={"public": True}
                )
                print(f"✅ Created storage bucket: {self.bucket_name}")
        except Exception as e:
            print(f"⚠️  Bucket check failed: {e}")
    
    def upload_image(self, image_data: bytes, filename: str) -> str:
        """
        Upload image to Supabase Storage
        
        Args:
            image_data: Image file bytes
            filename: Unique filename for the image
        
        Returns:
            Public URL to access the uploaded image
        """
        if self.mock_mode:
            # Return mock URL for demo purposes
            return f"https://demo.truepix.app/images/{filename}"
        
        try:
            # Upload to Supabase Storage
            path = f"uploads/{filename}"
            
            response = self.client.storage.from_(self.bucket_name).upload(
                path=path,
                file=image_data,
                file_options={"content-type": "image/jpeg"}
            )
            
            # Get public URL
            public_url = self.client.storage.from_(self.bucket_name).get_public_url(path)
            
            return public_url
        
        except Exception as e:
            raise Exception(f"Upload to storage failed: {str(e)}")
    
    def delete_image(self, filename: str) -> bool:
        """Delete image from storage"""
        if self.mock_mode:
            return True
        
        try:
            path = f"uploads/{filename}"
            self.client.storage.from_(self.bucket_name).remove([path])
            return True
        except Exception as e:
            print(f"Delete failed: {e}")
            return False
    
    def is_connected(self) -> bool:
        """Check if storage is connected"""
        if self.mock_mode:
            return True
        
        try:
            self.client.storage.list_buckets()
            return True
        except:
            return False
