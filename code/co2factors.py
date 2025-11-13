from utils import load_json

class FactorRegistry:
    def __init__(self):
        self.materials = load_json("materials.json")
        self.dyeing = load_json("dyeing.json")
        self.mfg = load_json("manufacturing.json")
        self.transport = load_json("transport.json")
        self.washing = load_json("washing.json")
        self.packaging = load_json("packaging.json")
        self.eol = load_json("eol.json")
