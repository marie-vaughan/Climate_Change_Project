import re
from schemas import TagRecord, MaterialComponent, CareProfile

# --- Updated regexes ---

# Matches: 100%Cotton, 100% Cotton, 100 %/Cotton, 100%-Coton, etc.
MATERIAL_RE = re.compile(
    r'(\d{1,3})\s*%?\s*[-/ ]*\s*(cotton|coton|algodon|baumwolle|katoen|綿|algod[oó]n|polyester|viscose|rayon|wool|nylon|linen|silk|modal|lyocell|spandex|elastane|acrylic)',
    re.I
)

# Matches "Made in Guatemala", "Hecho en Guatemala", etc.
ORIGIN_RE = re.compile(
    r'(made\s+in|hecho\s+en|fabriqu[eé]\s+en|hergestellt\s+in|vervaardigd\s+in)\s+([A-Za-z]+)',
    re.I
)

# Fabric print hints
PRINT_RE = re.compile(r'(print|printed|graphic)', re.I)
DYE_RE = re.compile(r'(reactive\s*dye|garment\s*dyed|piece\s*dyed|vat\s*dye)', re.I)

# Wash & dry instructions
WASH_RE = re.compile(r'(machine\s*w(ash)?\s*(cold|warm|hot)|wash\s*(cold|warm|hot))', re.I)
DRY_RE = re.compile(r'(tumble\s*dry[^.;,]*)|(line\s*dry)', re.I)
DRY_CLEAN_RE = re.compile(r'(dry\s*-?\s*clean)(?:\s*(green|eco|perc|conventional))?', re.I)

def parse_from_text(text: str, garment_type="tshirt", default_weight_g=None, washes_per_month=2.0) -> TagRecord:
    # Normalize whitespace & accents
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    materials = []
    for pct, fiber in MATERIAL_RE.findall(text):
        try:
            materials.append(MaterialComponent(fiber=fiber.lower(), pct=float(pct)))
        except ValueError:
            continue

    # normalize totals to 100
    if materials:
        s = sum(m.pct for m in materials)
        if s and abs(s - 100) > 1:
            for m in materials:
                m.pct = round(m.pct * 100.0 / s, 1)

    origin_country = None
    m = ORIGIN_RE.search(text)
    if m:
        origin_country = m.group(2).strip().lower()

    printed = bool(PRINT_RE.search(text))
    dye_hint = DYE_RE.search(text).group(1).lower() if DYE_RE.search(text) else None

    # Washing
    wash = "cold"
    m3 = WASH_RE.search(text)
    if m3:
        # pick whichever group captured temperature
        if "warm" in m3.group(0).lower():
            wash = "warm"
        elif "hot" in m3.group(0).lower():
            wash = "hot"
        else:
            wash = "cold"

    # Drying
    dry = "line"
    m4 = DRY_RE.search(text)
    if m4:
        if m4.group(1):  # tumble dry pattern
            dry = "tumble"
        elif m4.group(2):
            dry = "line"

    dry_clean = "none"
    m5 = DRY_CLEAN_RE.search(text)
    if m5:
        qual = (m5.group(2) or "").lower()
        if "green" in qual or "eco" in qual:
            dry_clean = "green"
        elif "perc" in qual or "conventional" in qual or qual == "":
            dry_clean = "conventional"

    care = CareProfile(
        wash=wash,
        dry=dry,
        dry_clean=dry_clean,
        washes_per_month=washes_per_month
    )

    return TagRecord(
        materials=materials,
        origin_country=origin_country,
        garment_type=garment_type,
        weight_g=default_weight_g,
        dye_hint=dye_hint,
        printed=printed,
        care=care
    )
