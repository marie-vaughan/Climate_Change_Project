from dataclasses import dataclass, field
from typing import List, Optional, Dict

@dataclass
class MaterialComponent:
    fiber: str
    pct: float

@dataclass
class CareProfile:
    wash: str = "cold"
    dry: str = "line"
    dry_clean: str = "none"
    washes_per_month: float = 2.0

@dataclass
class TagRecord:
    materials: List[MaterialComponent] = field(default_factory=list)
    origin_country: Optional[str] = None
    garment_type: str = "tshirt"
    weight_g: Optional[float] = None
    dye_hint: Optional[str] = None
    printed: bool = False
    care: CareProfile = field(default_factory=CareProfile)

@dataclass
class ScenarioResult:
    total_kgco2e: float
    breakdown: Dict[str, float]
    assumptions: Dict[str, str]
