import base64
import os

# Create images directory if it doesn't exist
images_dir = r"d:\Truepix\frontend\public\images"
os.makedirs(images_dir, exist_ok=True)

print("=" * 60)
print("SAVE YOUR IMAGES MANUALLY")
print("=" * 60)
print(f"\nImages directory: {images_dir}")
print("\nYou have 2 images in the chat attachments:")
print("\n1. ROBOT IMAGE (first attachment - white/blue robot thinking)")
print(f"   → Right-click and save as: {os.path.join(images_dir, 'robot.png')}")
print("\n2. HUMAN IMAGE (second attachment - man in suit thinking)")
print(f"   → Right-click and save as: {os.path.join(images_dir, 'human.png')}")
print("\n" + "=" * 60)
print("After saving both images, refresh your browser!")
print("=" * 60)
