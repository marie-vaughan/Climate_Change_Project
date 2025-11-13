from schemas import TagRecord, ScenarioResult
from co2factors import FactorRegistry
from utils import to_kg
from transport import distance_to_us, shipment_kgco2

WEIGHT_PRIORS = {"tshirt":180,"jeans":700,"dress":350,"sweater":400,"_default":300}

def estimate(record: TagRecord, preferred_mode="ship") -> ScenarioResult:
    fx = FactorRegistry()
    b = {}
    weight_g = record.weight_g or WEIGHT_PRIORS.get(record.garment_type,300)
    weight_kg = to_kg(weight_g)

    # Fabric
    mat = 0
    for m in record.materials or []:
        f = fx.materials.get(m.fiber, fx.materials["cotton"])
        mat += (m.pct/100)*f["kgco2_per_kg"]
    b["fabric"] = weight_kg * (mat or fx.materials["cotton"]["kgco2_per_kg"])

    # Dye/print
    d = fx.dyeing["reactive_dye"]["kgco2_per_kg"]
    if record.printed: d += fx.dyeing["print"]["kgco2_per_kg"]
    b["dye_print"] = weight_kg*d

    # Manufacturing
    mfg_f = fx.mfg.get(record.origin_country or "_default", fx.mfg["_default"])
    b["manufacturing"] = mfg_f["kgco2_per_unit"]

    # Transport
    dist = distance_to_us(record.origin_country)
    mode = preferred_mode if preferred_mode in fx.transport else "ship"
    b["transport"] = shipment_kgco2(weight_kg, dist, fx.transport[mode]["kgco2_per_tkm"])

    # Packaging
    b["packaging"] = fx.packaging["polybag_mailer"]["kgco2_per_unit"]

    # Use phase (washing)
    washes = int(24*record.care.washes_per_month)
    per = fx.washing["wash_cold"]["kgco2_per_wash"] if record.care.wash=="cold" else fx.washing["wash_warm"]["kgco2_per_wash"]
    washing = washes*per
    if record.care.dry=="tumble": washing+=washes*fx.washing["tumble_dry"]["kgco2_per_cycle"]
    if record.care.dry_clean=="green": washing+=washes*fx.washing["dry_clean_green"]["kgco2_per_cycle"]
    elif record.care.dry_clean=="conventional": washing+=washes*fx.washing["dry_clean_conventional"]["kgco2_per_cycle"]
    b["washing"]=washing

    # EOL
    b["eol"]=fx.eol["landfill"]["kgco2_per_unit"]

    total=sum(b.values())
    return ScenarioResult(total,b,{"weight_g":str(weight_g),"origin":record.origin_country or "unknown",
        "transport_mode":mode,"washes_lifetime":str(washes)})
