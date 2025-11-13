"""
generate_synthetic_tags.py
-----------------------------------
Creates a dataset of synthetic garment-tag text images for TrOCR fine-tuning.
Dependencies: Pillow, OpenCV, tqdm

Output:
    synthetic_tags/
        train/
            tag_00001.jpg
            tag_00001.txt
            ...
        val/
            ...
"""

import os, random, cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# ------------------------------------------------------------------
# 1️⃣  CONFIGURATION
# ------------------------------------------------------------------
OUT_DIR = "synthetic_tags"
TRAIN_N, VAL_N = 4000, 500
IMG_W, IMG_H = 1000, 220      # a bit taller to fit multiple lines
MAX_LINES = 5

FONTS = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Courier New.ttf",
]

TEXT_CORPUS = [
    # --- Materials ---
    "100% COTTON / COTON / ALGODÓN",
    "65% POLYESTER 35% COTTON",
    "80% NYLON 20% SPANDEX",
    "95% COTTON 5% ELASTANE",
    "SHELL: 100% SILK",
    "LINING: 100% POLYESTER",
    "EXTERIOR: 60% COTTON 40% POLYESTER",
    "MAIN FABRIC: 100% TENCEL™ LYOCELL",
    # --- Origin ---
    "MADE IN GUATEMALA",
    "MADE IN CHINA",
    "HECHO EN MÉXICO",
    "FABRIQUÉ EN VIÊTNAM",
    "HERGESTELLT IN DEUTSCHLAND",
    "ASSEMBLED IN BANGLADESH",
    # --- Care ---
    "MACHINE WASH COLD WITH LIKE COLORS",
    "HAND WASH SEPARATELY",
    "USE ONLY NON-CHLORINE BLEACH",
    "DO NOT BLEACH",
    "TUMBLE DRY LOW",
    "LINE DRY IN SHADE",
    "DO NOT IRON PRINT",
    "COOL IRON IF NEEDED",
    "DRY CLEAN ONLY",
    "DO NOT DRY CLEAN",
    "WASH BEFORE WEAR",
    "WASH DARK COLORS SEPARATELY",
]

os.makedirs(f"{OUT_DIR}/train", exist_ok=True)
os.makedirs(f"{OUT_DIR}/val", exist_ok=True)

# ------------------------------------------------------------------
# 2️⃣  HELPERS
# ------------------------------------------------------------------
def random_font():
    path = random.choice(FONTS)
    size = random.randint(26, 40)
    return ImageFont.truetype(path, size)

def make_background(w, h):
    base = np.full((h, w, 3), 240 + random.randint(-5, 5), np.uint8)
    noise = np.random.randint(0, 10, (h, w, 1), np.uint8)
    bg = cv2.merge([base[:,:,0]-noise[:,:,0],
                    base[:,:,1]-noise[:,:,0],
                    base[:,:,2]-noise[:,:,0]])
    return bg

def render_multiline_text(lines, font):
    """Render 1–3 lines centered vertically on a light background."""
    img = Image.new("L", (IMG_W, IMG_H), 255)
    draw = ImageDraw.Draw(img)

    # Pillow ≥10: use textbbox instead of textsize
    heights = []
    widths = []
    for l in lines:
        bbox = draw.textbbox((0, 0), l, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        widths.append(w)
        heights.append(h)

    line_h = sum(heights)
    y_offset = (IMG_H - line_h) // 2

    for i, line in enumerate(lines):
        w, h = widths[i], heights[i]
        x = (IMG_W - w) // 2 + random.randint(-10, 10)
        draw.text((x, y_offset), line, font=font, fill=random.randint(0, 40))
        y_offset += h + random.randint(0, 5)

    return img

def augment(img_pil):
    img_pil = img_pil.rotate(random.uniform(-3,3), expand=1, fillcolor=255)
    img = np.array(img_pil)
    if random.random() < 0.5:
        img = cv2.GaussianBlur(img, (3,3), random.uniform(0.3,0.8))
    else:
        noise = np.random.normal(0,5,img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16)+noise,0,255).astype(np.uint8)
    bg = make_background(*img.shape[::-1])
    img_col = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(bg,0.85,img_col,0.15,0)
    alpha = random.uniform(0.9,1.2)
    beta = random.randint(-10,15)
    blended = cv2.convertScaleAbs(blended, alpha=alpha, beta=beta)
    return blended

# ------------------------------------------------------------------
# 3️⃣  MAIN GENERATOR
# ------------------------------------------------------------------
def generate_split(split, n):
    for i in tqdm(range(n), desc=f"Generating {split}"):
        # pick 1–3 random lines
        k = random.randint(1, MAX_LINES)
        lines = random.sample(TEXT_CORPUS, k)
        text_gt = " / ".join(lines)
        font = random_font()
        gray = render_multiline_text(lines, font)
        final = augment(gray)
        base = os.path.join(OUT_DIR, split, f"tag_{i:05d}")
        cv2.imwrite(base + ".jpg", final)
        with open(base + ".txt","w") as f:
            f.write(text_gt)

if __name__ == "__main__":
    generate_split("train", TRAIN_N)
    generate_split("val", VAL_N)
    print(f"✅ Synthetic dataset written to '{OUT_DIR}' with up to {MAX_LINES} lines per image.")
