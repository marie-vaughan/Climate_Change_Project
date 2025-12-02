from schemas import TagRecord, ScenarioResult
from co2factors import FactorRegistry
from utils import to_kg

def estimate(record: TagRecord, preferred_mode="ship") -> ScenarioResult:
    fx = FactorRegistry()
    b = {}
    weight_g = record.weight_g or 1000  # Default 1kg
    weight_kg = to_kg(weight_g)

    # Materials (Fabric)
    mat = 0
    for m in record.materials or []:
        f = fx.materials.get(m.fiber, fx.materials["cotton"])
        mat += (m.pct/100)*f["kgco2_per_kg"]
    b["materials"] = weight_kg * (mat or fx.materials["cotton"]["kgco2_per_kg"])

    # Manufacturing
    # Use China as default if country not found (common manufacturing location)
    country_key = (record.origin_country or "china").lower()
    mfg_f = fx.mfg.get(country_key, fx.mfg.get("china", {"kgco2_per_kg": 0.50}))
    b["manufacturing"] = weight_kg * mfg_f["kgco2_per_kg"]

    # Washing (Use phase)
    washes = int(24 * record.care.washes_per_month)
    
    # Determine wash type
    if record.care.wash == "cold":
        wash_key = "machine_wash_cold"
    elif record.care.wash == "warm":
        wash_key = "machine_wash_warm"
    elif record.care.wash == "hot":
        wash_key = "machine_wash_hot"
    else:
        wash_key = "machine_wash_cold"  # Default to cold
    
    washing = washes * fx.washing.get(wash_key, {"kgco2_per_use": 0.10})["kgco2_per_use"]
    
    # Add drying emissions
    if record.care.dry == "tumble":
        washing += washes * fx.washing.get("tumble_dry_medium", {"kgco2_per_use": 2.60})["kgco2_per_use"]
    
    # Add dry cleaning emissions if applicable
    if record.care.dry_clean:
        washing += washes * fx.washing.get("dry_clean", {"kgco2_per_use": 0.40})["kgco2_per_use"]
    
    b["washing"] = washing

    total=sum(b.values())
    return ScenarioResult(total,b,{"weight_g":str(weight_g),"origin":record.origin_country or "unknown",
        "washes_lifetime":str(washes)})
