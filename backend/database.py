"""
Database Manager - SQLite Database for Image Storage and Analysis Results
Handles storing uploaded images and their analysis results
"""

import sqlite3
import os
from datetime import datetime
from typing import Optional, Dict, Any
import json


class DatabaseManager:
    """Manages SQLite database for image storage and analysis results"""
    
    def __init__(self, db_path: str = "truepix.db"):
        """
        Initialize database connection and create tables if they don't exist
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Create database tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create images table for storing uploaded images
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                image_data BLOB NOT NULL,
                file_size INTEGER,
                image_format TEXT,
                width INTEGER,
                height INTEGER
            )
        """)
        
        # Create analysis_results table for storing detection results
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id TEXT NOT NULL,
                analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                cnn_score REAL,
                fft_score REAL,
                noise_score REAL,
                edge_score REAL,
                final_score REAL,
                prediction TEXT,
                explanation TEXT,
                processing_time REAL,
                FOREIGN KEY (image_id) REFERENCES images (id)
            )
        """)
        
        # Create index for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_image_id 
            ON analysis_results(image_id)
        """)
        
        conn.commit()
        conn.close()
        print(f"✅ Database initialized at {self.db_path}")
    
    def store_image(self, image_id: str, filename: str, image_bytes: bytes, 
                    width: int, height: int, image_format: str) -> bool:
        """
        Store uploaded image in database
        
        Args:
            image_id: Unique identifier for the image
            filename: Original filename
            image_bytes: Raw image data
            width: Image width in pixels
            height: Image height in pixels
            image_format: Image format (JPEG, PNG, etc.)
            
        Returns:
            True if stored successfully, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO images (id, filename, image_data, file_size, 
                                   image_format, width, height)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (image_id, filename, image_bytes, len(image_bytes), 
                  image_format, width, height))
            
            conn.commit()
            conn.close()
            
            print(f"✅ Stored image {image_id} ({filename}) in database")
            return True
        except Exception as e:
            print(f"❌ Error storing image: {e}")
            return False
    
    def retrieve_image(self, image_id: str) -> Optional[bytes]:
        """
        Retrieve image data from database
        
        Args:
            image_id: Unique identifier for the image
            
        Returns:
            Image bytes if found, None otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT image_data FROM images WHERE id = ?
            """, (image_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                print(f"✅ Retrieved image {image_id} from database")
                return result[0]
            else:
                print(f"❌ Image {image_id} not found in database")
                return None
        except Exception as e:
            print(f"❌ Error retrieving image: {e}")
            return None
    
    def get_image_info(self, image_id: str) -> Optional[Dict[str, Any]]:
        """
        Get image metadata from database
        
        Args:
            image_id: Unique identifier for the image
            
        Returns:
            Dictionary with image metadata if found, None otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, filename, upload_timestamp, file_size, 
                       image_format, width, height
                FROM images WHERE id = ?
            """, (image_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    "id": result[0],
                    "filename": result[1],
                    "upload_timestamp": result[2],
                    "file_size": result[3],
                    "image_format": result[4],
                    "width": result[5],
                    "height": result[6]
                }
            return None
        except Exception as e:
            print(f"❌ Error getting image info: {e}")
            return None
    
    def store_analysis_result(self, image_id: str, cnn_score: float, 
                             fft_score: float, noise_score: float,
                             edge_score: Optional[float], final_score: float,
                             prediction: str, explanation: str,
                             processing_time: float) -> bool:
        """
        Store analysis results in database
        
        Args:
            image_id: Unique identifier for the image
            cnn_score: CNN analysis score (0-1)
            fft_score: FFT analysis score (0-1)
            noise_score: Noise analysis score (0-1)
            edge_score: Edge detection score (0-1) or None
            final_score: Final combined score (0-1)
            prediction: Human-readable prediction
            explanation: Detailed explanation of the result
            processing_time: Time taken for analysis in seconds
            
        Returns:
            True if stored successfully, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO analysis_results 
                (image_id, cnn_score, fft_score, noise_score, edge_score,
                 final_score, prediction, explanation, processing_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (image_id, cnn_score, fft_score, noise_score, edge_score,
                  final_score, prediction, explanation, processing_time))
            
            conn.commit()
            conn.close()
            
            print(f"✅ Stored analysis results for image {image_id}")
            return True
        except Exception as e:
            print(f"❌ Error storing analysis results: {e}")
            return False
    
    def get_analysis_result(self, image_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve analysis results for an image
        
        Args:
            image_id: Unique identifier for the image
            
        Returns:
            Dictionary with analysis results if found, None otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT cnn_score, fft_score, noise_score, edge_score,
                       final_score, prediction, explanation, 
                       analysis_timestamp, processing_time
                FROM analysis_results 
                WHERE image_id = ?
                ORDER BY analysis_timestamp DESC
                LIMIT 1
            """, (image_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    "cnn_score": result[0],
                    "fft_score": result[1],
                    "noise_score": result[2],
                    "edge_score": result[3],
                    "final_score": result[4],
                    "prediction": result[5],
                    "explanation": result[6],
                    "analysis_timestamp": result[7],
                    "processing_time": result[8]
                }
            return None
        except Exception as e:
            print(f"❌ Error retrieving analysis results: {e}")
            return None
    
    def close(self):
        """Close database connection (cleanup)"""
        # SQLite connections are closed after each operation
        pass
