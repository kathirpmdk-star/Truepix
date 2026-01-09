from PIL import Image, ImageDraw, ImageFont
import numpy as np

# For now, we'll create a placeholder that matches the style
# In production, you would save the actual uploaded image
width, height = 1920, 1080

# Create gradient background (blue to cyan)
img_array = np.zeros((height, width, 3), dtype=np.uint8)

for y in range(height):
    for x in range(width):
        t = x / width
        r = int(66 + (0 - 66) * t)
        g = int(51 + (242 - 51) * t)
        b = int(153 + (254 - 153) * t)
        img_array[y, x] = [r, g, b]

img = Image.fromarray(img_array)
img.save('hero-image.jpg', quality=95)
print("Hero image updated!")
