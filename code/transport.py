from utils import to_tonnes

DIST_TO_US = {"china": 10000, "turkey": 9000, "usa": 1500, "_default": 8000}

def distance_to_us(origin_country: str) -> float:
    if not origin_country:
        return DIST_TO_US["_default"]
    return DIST_TO_US.get(origin_country.lower(), DIST_TO_US["_default"])

def shipment_kgco2(material_weight_kg, distance_km, kgco2_per_tkm):
    return to_tonnes(material_weight_kg) * distance_km * kgco2_per_tkm
