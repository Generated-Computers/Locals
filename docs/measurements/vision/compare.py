import json, sys
from pathlib import Path
D = Path(sys.argv[1])
L = lambda f: json.loads((D / f).read_text())
pairs = [("Qwen3.5-4B: ours vs mlx-vlm", "loc_q35.json", "vlm_q35.json"),
         ("Qwen3-VL-8B: ours vs mlx-vlm", "loc_vl8.json", "vlm_vl8.json"),
         ("Qwen3-VL-8B: ours, no deepstack vs mlx-vlm", "loc_vl8_nodeep.json", "vlm_vl8.json"),
         ("Qwen3-VL-8B: ours, no deepstack vs ours", "loc_vl8_nodeep.json", "loc_vl8.json")]
for f in ("loc_q35.json", "vlm_q35.json", "loc_vl8.json", "vlm_vl8.json", "loc_vl8_nodeep.json"):
    d = L(f); bad = [k for k, v in d.items() if not k.startswith("_") and v["answers"][0] != v["answers"][1]]
    print(f"{f}: repeats differ on {len(bad)} {bad}")
for title, a, b in pairs:
    A, B = L(a), L(b); names = [k for k in A if not k.startswith("_")]
    same = [k for k in names if A[k]["answers"][0] == B[k]["answers"][0]]
    print(f"\n{title}: {len(same)} of {len(names)} the same")
    for k in names:
        if k not in same:
            print(f"  {k}\n    {a}: {A[k]['answers'][0][:150]!r}\n    {b}: {B[k]['answers'][0][:150]!r}")
for f in ("loc_vl8.json",):
    print("\nQwen3-VL CAT 42:", {k: L(f)[k]["answers"][0] for k in L(f) if k.startswith("cat42")})
print("Qwen3.5 CAT 42 ours:", {k: L("loc_q35.json")[k]["answers"][0] for k in L("loc_q35.json") if k.startswith("cat42")})
