# create_better_icon.py
import os
from PIL import Image, ImageDraw, ImageFont

print("Creating professional icon for GRACE Processor...")

# Create 64x64 icon (QGIS can resize it)
size = 64
img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# Colors (professional blue theme)
colors = {
    'dark_blue': (41, 128, 185),    # #2980b9
    'medium_blue': (52, 152, 219),  # #3498db
    'light_blue': (93, 173, 226),   # #5dade2
    'white': (255, 255, 255),
    'green': (26, 188, 156)         # #1abc9c
}

center = size // 2

# Draw Earth (gradient circle)
for r in range(size//2 - 8, 0, -1):
    alpha = int(255 * (r / (size//2 - 8)))
    if r > (size//2 - 8) * 0.7:
        color = (*colors['dark_blue'], alpha)
    elif r > (size//2 - 8) * 0.4:
        color = (*colors['medium_blue'], alpha)
    else:
        color = (*colors['green'], alpha)
    
    draw.ellipse(
        [center - r, center - r, center + r, center + r],
        fill=color
    )

# Draw satellite orbit
orbit_radius = size//2 - 4
draw.ellipse(
    [center - orbit_radius, center - orbit_radius,
     center + orbit_radius, center + orbit_radius],
    outline=colors['white'],
    width=2
)

# Draw satellite
sat_size = 6
sat_x = center + orbit_radius - 8
sat_y = center

# Satellite body
draw.ellipse(
    [sat_x - sat_size, sat_y - sat_size,
     sat_x + sat_size, sat_y + sat_size],
    fill=colors['white'],
    outline=colors['dark_blue'],
    width=2
)

# Solar panels
draw.rectangle(
    [sat_x - sat_size - 6, sat_y - 2,
     sat_x - sat_size - 2, sat_y + 2],
    fill=colors['white']
)
draw.rectangle(
    [sat_x + sat_size + 2, sat_y - 2,
     sat_x + sat_size + 6, sat_y + 2],
    fill=colors['white']
)

# Add text "GRACE"
try:
    # Try to use Arial Bold
    font = ImageFont.truetype("arialbd.ttf", 14)
except:
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()

draw.text(
    (center - 20, center - 30),
    "GRACE",
    fill=colors['white'],
    font=font
)

# Save multiple sizes
sizes = [16, 24, 32, 48, 64]
for s in sizes:
    resized = img.resize((s, s), Image.Resampling.LANCZOS)
    resized.save(f"icon_{s}.png")
    print(f"✅ Created icon_{s}.png")

# Save as main icon (32x32)
main_icon = img.resize((32, 32), Image.Resampling.LANCZOS)
main_icon.save("icon.png")
print("\n✅ Main icon saved as 'icon.png'")

# Create preview
preview = Image.new('RGB', (200, 80), (240, 240, 240))
for i, s in enumerate([16, 24, 32, 48]):
    icon_img = Image.open(f"icon_{s}.png")
    preview.paste(icon_img, (10 + i*45, 10))
    
preview.save("icon_preview.png")
print("✅ Preview saved as 'icon_preview.png'")
print("\nIcons created successfully! The plugin should now show icons.")