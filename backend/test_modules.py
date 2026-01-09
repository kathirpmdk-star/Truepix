#!/usr/bin/env python3
"""
Test script to verify backend image analysis works end-to-end
Creates a simple test image and sends it through the analysis pipeline
"""

import sys
import os
sys.path.insert(0, '/Users/kathir/Documents/GitHub/Truepix/backend')

from PIL import Image
import numpy as np
import io

# Import our modules
from database import DatabaseManager
from preprocessing import ImagePreprocessor
from cnn_detector import CNNDetector
from fft_analyzer import FFTAnalyzer
from noise_analyzer import NoiseAnalyzer
from score_fusion import ScoreFusion

print("=" * 60)
print("🧪 TESTING TRUEPIX BACKEND MODULES")
print("=" * 60)

# Create a simple test image (random noise image)
print("\n1️⃣ Creating test image...")
test_image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
pil_image = Image.fromarray(test_image, 'RGB')

# Convert to bytes
buffer = io.BytesIO()
pil_image.save(buffer, format='JPEG')
image_bytes = buffer.getvalue()
print(f"✅ Created test image: {len(image_bytes)} bytes")

# Test Database
print("\n2️⃣ Testing Database...")
db = DatabaseManager()
success = db.store_image(
    image_id="test-123",
    filename="test.jpg",
    image_bytes=image_bytes,
    width=512,
    height=512,
    image_format="JPEG"
)
print(f"✅ Database store: {'Success' if success else 'Failed'}")

retrieved = db.retrieve_image("test-123")
print(f"✅ Database retrieve: {len(retrieved)} bytes" if retrieved else "❌ Failed to retrieve")

# Test Preprocessing
print("\n3️⃣ Testing Preprocessing...")
preprocessor = ImagePreprocessor()
image_pil, image_array = preprocessor.preprocess(image_bytes)
print(f"✅ Preprocessed: {image_array.shape}, dtype={image_array.dtype}")
print(f"✅ Value range: [{image_array.min():.3f}, {image_array.max():.3f}]")

# Test CNN Detector
print("\n4️⃣ Testing CNN Detector...")
try:
    cnn = CNNDetector(use_gpu=False)
    cnn_result = cnn.analyze(image_array)
    print(f"✅ CNN Score: {cnn_result['score']:.3f}")
    print(f"   Confidence: {cnn_result['confidence']:.3f}")
    print(f"   Explanation: {cnn_result['explanation'][:80]}...")
except Exception as e:
    print(f"❌ CNN Error: {e}")
    import traceback
    traceback.print_exc()

# Test FFT Analyzer
print("\n5️⃣ Testing FFT Analyzer...")
try:
    fft = FFTAnalyzer()
    fft_result = fft.analyze(image_array)
    print(f"✅ FFT Score: {fft_result['score']:.3f}")
    print(f"   Frequency Anomaly: {fft_result['frequency_anomaly']:.3f}")
    print(f"   Periodic Artifacts: {fft_result['periodic_artifacts']:.3f}")
    print(f"   Explanation: {fft_result['explanation'][:80]}...")
except Exception as e:
    print(f"❌ FFT Error: {e}")
    import traceback
    traceback.print_exc()

# Test Noise Analyzer
print("\n6️⃣ Testing Noise Analyzer...")
try:
    noise = NoiseAnalyzer()
    noise_result = noise.analyze(image_array)
    print(f"✅ Noise Score: {noise_result['score']:.3f}")
    print(f"   Variance Score: {noise_result['variance_score']:.3f}")
    print(f"   Entropy Score: {noise_result['entropy_score']:.3f}")
    print(f"   Explanation: {noise_result['explanation'][:80]}...")
except Exception as e:
    print(f"❌ Noise Error: {e}")
    import traceback
    traceback.print_exc()

# Test Score Fusion
print("\n7️⃣ Testing Score Fusion...")
try:
    fusion = ScoreFusion()
    fusion_result = fusion.fuse_scores(
        cnn_result=cnn_result,
        fft_result=fft_result,
        noise_result=noise_result
    )
    print(f"✅ Final Score: {fusion_result['final_score']:.3f}")
    print(f"   Prediction: {fusion_result['prediction']}")
    print(f"   Confidence: {fusion_result['confidence']:.3f}")
    print(f"   Explanation: {fusion_result['explanation'][:150]}...")
except Exception as e:
    print(f"❌ Fusion Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("✅ ALL TESTS COMPLETED")
print("=" * 60)
