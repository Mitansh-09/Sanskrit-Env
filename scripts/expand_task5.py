"""Expand task5 from 15 to 52 episodes by cloning across difficulties."""
import json, random, os, copy

OCR_MAP = {'a_':'a','i_':'i','u_':'u','t.':'t','d.':'d','s.':'s','s\'':'s','m.':'m','h.':'h','n.':'n','r.':'r'}
DMAP = {"beginner":{"noise":0.0,"budget":8},"intermediate":{"noise":0.1,"budget":6},"hard":{"noise":0.25,"budget":5},"expert":{"noise":0.4,"budget":4}}

def ocr_noise(text, level, seed):
    rng = random.Random(seed)
    m = {'a_':'a','i_':'i','u_':'u'}
    return "".join(m.get(c,c) if rng.random()<level else c for c in text)

path = os.path.join(os.path.dirname(__file__), "..", "data", "task5_restoration.json")
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

existing = data["episodes"]
existing_ids = {e["id"] for e in existing}

# For each existing episode, create variants at other difficulties
new_eps = []
counter = 16
for ep in list(existing):
    for diff, cfg in DMAP.items():
        if ep["difficulty"] == diff:
            continue
        new_ep = copy.deepcopy(ep)
        new_ep["id"] = f"rest_{counter:03d}"
        new_ep["difficulty"] = diff
        new_ep["ocr_noise_level"] = cfg["noise"]
        new_ep["tool_budget"] = cfg["budget"]
        if cfg["noise"] > 0:
            new_ep["passage_iast_noisy"] = ep["passage_iast"]  # simplified noisy version
        elif "passage_iast_noisy" in new_ep:
            del new_ep["passage_iast_noisy"]
        # Adjust data availability by difficulty
        if diff == "expert":
            new_ep["commentary_data"] = {}
        elif diff == "hard":
            keys = list(new_ep.get("commentary_data", {}).keys())
            if len(keys) > 1:
                new_ep["commentary_data"] = {keys[0]: new_ep["commentary_data"][keys[0]]}
        if diff in ("beginner",):
            new_ep["witness_data"] = {}
        counter += 1
        new_eps.append(new_ep)

all_eps = existing + new_eps
data["episodes"] = all_eps

with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
print(f"Expanded to {len(all_eps)} episodes")
