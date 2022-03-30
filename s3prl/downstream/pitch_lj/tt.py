import json


info = {}
info["train"] = [f"LJ{i+1:03d}" for i in range(40)]
info["dev"] = [f"LJ{i+1:03d}" for i in range(40, 45)]
info["test"] = [f"LJ{i+1:03d}" for i in range(45, 50)]
with open("split.json", 'w', encoding='utf-8') as f:
    json.dump(info, f, indent=4)
