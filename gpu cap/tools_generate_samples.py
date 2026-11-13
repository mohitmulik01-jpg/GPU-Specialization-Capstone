#!/usr/bin/env python3
"""Create deterministic synthetic RGB P6 PPM inputs for the capstone."""
from pathlib import Path

OUT = Path("data")
OUT.mkdir(exist_ok=True)

def write_image(path, width, height, mode):
    pixels = bytearray()
    for y in range(height):
        for x in range(width):
            if mode == "gradient":
                r = 255 * x // max(1, width - 1)
                g = 255 * y // max(1, height - 1)
                b = (r + g) // 2
            elif mode == "shapes":
                dx, dy = x - width // 2, y - height // 2
                if dx * dx + dy * dy < min(width, height) ** 2 // 9:
                    r, g, b = 240, 220, 40
                else:
                    r, g, b = ((30, 90, 190) if (x // 32 + y // 32) % 2
                               else (20, 25, 35))
            else:
                v = (x * 17 + y * 29 + (x * y) % 97) % 256
                r, g, b = v, (v * 3) % 256, 255 - v
            pixels.extend((r, g, b))
    with path.open("wb") as stream:
        stream.write(f"P6\\n{width} {height}\\n255\\n".encode("ascii"))
        stream.write(pixels)

if __name__ == "__main__":
    write_image(OUT / "sample_256.ppm", 256, 256, "gradient")
    write_image(OUT / "sample_512.ppm", 512, 512, "shapes")
    write_image(OUT / "sample_1024.ppm", 1024, 768, "pattern")
    print("Generated three deterministic PPM test images in data/.")
