import json, os

def load_json(name: str):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    with open(os.path.join(data_dir, name), "r") as f:
        return json.load(f)

def lowercase_clean(s: str) -> str:
    return " ".join(s.lower().strip().split())

def to_kg(weight_g: float) -> float:
    return max(0.0, float(weight_g)) / 1000.0

def to_tonnes(weight_kg: float) -> float:
    return max(0.0, float(weight_kg)) / 1000.0

