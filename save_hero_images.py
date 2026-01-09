"""
Script to save hero images for the landing page
Run this script to save the robot and human images
"""
import os
import shutil

# Create images directory if it doesn't exist
images_dir = r"d:\Truepix\frontend\public\images"
os.makedirs(images_dir, exist_ok=True)

print(f"Images directory created/verified at: {images_dir}")
print("\nPlease manually save the following images:")
print("1. Save the robot image as: robot.png")
print("2. Save the human image as: human.png")
print(f"\nSave them to: {images_dir}")
print("\nThe images should be the ones provided in the chat (robot and human thinking poses)")
