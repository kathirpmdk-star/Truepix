"""
Test Script for TruePix Backend API

Run this script to test all API endpoints
"""

import requests
import os
from pathlib import Path


API_URL = "http://localhost:8000"


def test_health_check():
    """Test health check endpoint"""
    print("\n1️⃣ Testing Health Check...")
    try:
        response = requests.get(f"{API_URL}/")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        assert response.status_code == 200
        print("   ✅ Health check passed!")
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")


def test_upload_image(image_path):
    """Test image upload endpoint"""
    print("\n2️⃣ Testing Image Upload...")
    try:
        if not os.path.exists(image_path):
            print(f"   ⚠️  Test image not found: {image_path}")
            return None
        
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{API_URL}/api/upload", files=files)
        
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        assert response.status_code == 200
        print("   ✅ Image upload passed!")
        return response.json()
    except Exception as e:
        print(f"   ❌ Image upload failed: {e}")
        return None


def test_analyze_image(image_path):
    """Test image analysis endpoint"""
    print("\n3️⃣ Testing Image Analysis...")
    try:
        if not os.path.exists(image_path):
            print(f"   ⚠️  Test image not found: {image_path}")
            return
        
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{API_URL}/api/analyze", files=files)
        
        print(f"   Status: {response.status_code}")
        result = response.json()
        
        print(f"\n   📊 Analysis Results:")
        print(f"   Prediction: {result['prediction']}")
        print(f"   Confidence: {result['confidence']}%")
        print(f"   Risk Level: {result['risk_level']}")
        print(f"   Explanation: {result['explanation'][:100]}...")
        
        assert response.status_code == 200
        print("\n   ✅ Image analysis passed!")
    except Exception as e:
        print(f"   ❌ Image analysis failed: {e}")


def test_platform_simulation(image_path):
    """Test platform simulation endpoint"""
    print("\n4️⃣ Testing Platform Simulation...")
    try:
        if not os.path.exists(image_path):
            print(f"   ⚠️  Test image not found: {image_path}")
            return
        
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{API_URL}/api/simulate-platforms", files=files)
        
        print(f"   Status: {response.status_code}")
        result = response.json()
        
        print(f"\n   📱 Platform Simulation Results:")
        print(f"   Stability Score: {result['stability_score']}%")
        
        print(f"\n   Original:")
        print(f"     Prediction: {result['original_result']['prediction']}")
        print(f"     Confidence: {result['original_result']['confidence']}%")
        
        for platform, data in result['platform_results'].items():
            print(f"\n   {platform.title()}:")
            print(f"     Prediction: {data['prediction']}")
            print(f"     Confidence: {data['confidence']}%")
        
        assert response.status_code == 200
        print("\n   ✅ Platform simulation passed!")
    except Exception as e:
        print(f"   ❌ Platform simulation failed: {e}")


def main():
    """Run all tests"""
    print("=" * 60)
    print("🧪 TruePix Backend API Tests")
    print("=" * 60)
    
    # Test with a sample image (you'll need to provide one)
    test_image = "test_image.jpg"  # Replace with actual test image path
    
    # Run tests
    test_health_check()
    test_upload_image(test_image)
    test_analyze_image(test_image)
    test_platform_simulation(test_image)
    
    print("\n" + "=" * 60)
    print("🎉 All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    print("\n⚠️  Make sure the backend server is running on http://localhost:8000")
    print("   Start it with: cd backend && python main.py\n")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test suite failed: {e}")
