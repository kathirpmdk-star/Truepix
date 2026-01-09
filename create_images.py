from PIL import Image
import os

# Check if man and robot images exist
man_file = "man and ai.png"
if os.path.exists(man_file):
    print(f"Found: {man_file}")
else:
    print(f"Not found: {man_file}")

# List all image files
import glob
images = glob.glob("*.png") + glob.glob("*.jpg") + glob.glob("*.jpeg")
print("\nAll images found:")
for img in images:
    print(f"  - {img}")
