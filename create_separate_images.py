from PIL import Image, ImageDraw
import numpy as np

# Create Robot image (white robot on transparent background)
print("Creating robot.png...")
robot = Image.new('RGBA', (500, 800), (0, 0, 0, 0))
# Robot will be shown from attachments when they load
# For now create a simple placeholder
robot.save("robot.png")
print("Created robot.png")

# Create Man image  
print("Creating man.png...")
man = Image.new('RGBA', (500, 800), (0, 0, 0, 0))
man.save("man.png")
print("Created man.png")

print("\nPlaceholder images created. These will be replaced when browser loads the actual attachments.")
