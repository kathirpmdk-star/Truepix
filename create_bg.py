from PIL import Image
import numpy as np

# Create a gradient image similar to the attached one (blue to cyan gradient)
width, height = 1920, 1080

# Create gradient from deep blue/purple to bright cyan
img_array = np.zeros((height, width, 3), dtype=np.uint8)

for y in range(height):
    for x in range(width):
        # Diagonal gradient from bottom-left (deep blue/purple) to top-right (cyan)
        t = (x + y) / (width + height)
        
        # Color interpolation
        r = int(66 + (0 - 66) * t)      # 66 -> 0
        g = int(51 + (242 - 51) * t)    # 51 -> 242  
        b = int(153 + (254 - 153) * t)  # 153 -> 254
        
        img_array[y, x] = [r, g, b]

# Save the image
img = Image.fromarray(img_array)
img.save('background.jpg', quality=95)
print("Background image created!")
