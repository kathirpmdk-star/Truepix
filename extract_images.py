from PIL import Image
import os

# Check if we have the combined image
if os.path.exists("man and ai.png"):
    img = Image.open("man and ai.png")
    width, height = img.size
    
    # Split into left (robot) and right (man)
    mid = width // 2
    
    # Extract robot (left side)
    robot = img.crop((0, 0, mid, height))
    robot.save("robot.png")
    print(f"Created robot.png ({robot.size})")
    
    # Extract man (right side)
    man = img.crop((mid, 0, width, height))
    man.save("man.png")
    print(f"Created man.png ({man.size})")
else:
    print("man and ai.png not found")
    print("Creating placeholder images...")
    
    # Create simple placeholder
    from PIL import ImageDraw, ImageFont
    
    # Robot placeholder
    robot = Image.new('RGBA', (400, 600), (240, 240, 240, 0))
    draw = ImageDraw.Draw(robot)
    draw.text((150, 280), "🤖", fill=(0, 0, 0, 255), font=None)
    robot.save("robot.png")
    
    # Man placeholder
    man = Image.new('RGBA', (400, 600), (240, 240, 240, 0))
    draw = ImageDraw.Draw(man)
    draw.text((150, 280), "👨", fill=(0, 0, 0, 255), font=None)
    man.save("man.png")
    
    print("Created placeholder images")

print("\nDone!")
