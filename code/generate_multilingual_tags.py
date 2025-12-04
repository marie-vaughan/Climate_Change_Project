"""
generate_multilingual_tags_cached.py
Creates multilingual garment-tag images (English + French + Spanish) with translation caching.
"""

import os, random, json, logging, cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from deep_translator import GoogleTranslator
from codecarbon import EmissionsTracker

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------
OUT_DIR = "synthetic_tags"
TRAIN_N, VAL_N = 4000, 500
IMG_W, IMG_H = 1000, 240
MAX_LANGS = 3
CACHE_PATH = "translations_cache.json"

FONTS = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Courier New.ttf",
]

TEXT_CORPUS_EN = [
    "100% COTTON",
    "65% POLYESTER 35% COTTON",
    "80% NYLON 20% SPANDEX",
    "95% COTTON 5% ELASTANE",
    "SHELL: 100% SILK",
    "LINING: 100% POLYESTER",
    "MAIN FABRIC: 100% TENCEL LYOCELL",
    "MADE IN GUATEMALA",
    "MADE IN CHINA",
    "MADE IN BANGLADESH",
    "ASSEMBLED IN VIETNAM",
    "MACHINE WASH COLD WITH LIKE COLORS",
    "HAND WASH SEPARATELY",
    "DO NOT BLEACH",
    "TUMBLE DRY LOW",
    "LINE DRY IN SHADE",
    "DO NOT IRON PRINT",
    "DRY CLEAN ONLY",
    "WASH BEFORE WEAR",
    "DO NOT WRING OR TWIST",
]

# ----------------------------------------------------------
# SETUP: Logging + emissions tracking
# ----------------------------------------------------------
LOG_DIR = "./emissions"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "generation.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

tracker = EmissionsTracker(
    project_name="TagData_Generation_Multilingual_Cached",
    output_dir=LOG_DIR,
    output_file="emissions.csv",
    save_to_file=True,
    log_level="info"
)
tracker.start()

# ----------------------------------------------------------
# HELPERS
# ----------------------------------------------------------
def load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

cache = load_cache()

def translate_phrase(phrase):
    """Return (en, fr, es) tuple with local caching."""
    if phrase in cache:
        return cache[phrase]

    try:
        fr = GoogleTranslator(source="en", target="fr").translate(phrase)
        es = GoogleTranslator(source="en", target="es").translate(phrase)
    except Exception as e:
        logger.warning(f"Translation failed for '{phrase}': {e}")
        fr, es = phrase, phrase

    cache[phrase] = (phrase, fr, es)
    save_cache(cache)
    return cache[phrase]

def random_font():
    path = random.choice(FONTS)
    size = random.randint(24, 40)
    return ImageFont.truetype(path, size)

def make_background(w, h):
    base = np.full((h, w, 3), 240 + random.randint(-5, 5), np.uint8)
    noise = np.random.randint(0, 10, (h, w, 1), np.uint8)
    bg = cv2.merge([base[:,:,0]-noise[:,:,0]]*3)
    return bg

def render_text(lines, font):
    img = Image.new("L", (IMG_W, IMG_H), 255)
    draw = ImageDraw.Draw(img)
    y = 40
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        x = (IMG_W - w)//2
        draw.text((x, y), line, font=font, fill=random.randint(0,40))
        y += bbox[3] - bbox[1] + 10
    return img

def augment(img_pil):
    img_pil = img_pil.rotate(random.uniform(-3,3), expand=1, fillcolor=255)
    img = np.array(img_pil)
    if random.random() < 0.5:
        img = cv2.GaussianBlur(img, (3,3), random.uniform(0.3,0.8))
    bg = make_background(*img.shape[::-1])
    img_col = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(bg,0.85,img_col,0.15,0)
    alpha = random.uniform(0.9,1.2)
    beta = random.randint(-10,15)
    return cv2.convertScaleAbs(blended, alpha=alpha, beta=beta)

# ----------------------------------------------------------
# MAIN GENERATOR
# ----------------------------------------------------------
def generate_split(split, n):
    os.makedirs(f"{OUT_DIR}/{split}", exist_ok=True)
    for i in tqdm(range(n), desc=f"Generating {split}"):
        en = random.choice(TEXT_CORPUS_EN)
        en, fr, es = translate_phrase(en)
        langs = random.sample([en, fr, es], k=random.randint(1, MAX_LANGS))
        text_gt = " / ".join(langs)
        font = random_font()
        gray = render_text(langs, font)
        final = augment(gray)
        base = os.path.join(OUT_DIR, split, f"tag_{i:05d}")
        cv2.imwrite(base + ".jpg", final)
        with open(base + ".txt", "w", encoding="utf-8") as f:
            f.write(text_gt)

if __name__ == "__main__":
    generate_split("train", TRAIN_N)
    generate_split("val", VAL_N)
    emissions = tracker.stop()
    logger.info(f"Data generation complete. Emissions: {emissions:.4f} kg CO2eq")
    print(f" Multilingual dataset created | Emissions: {emissions:.4f} kg CO2eq")
