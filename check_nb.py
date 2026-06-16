import json
nb = json.load(open("covariates_long_short.ipynb", encoding="utf-8"))
errors = [
    (i, "".join(o.get("evalue", ""))[:120])
    for i, c in enumerate(nb["cells"])
    if c["cell_type"] == "code"
    for o in c.get("outputs", [])
    if o.get("output_type") == "error"
]
if errors:
    for idx, msg in errors:
        print(f"Cell {idx}: {msg}")
else:
    print("No errors in any cell.")
print(f"Total cells: {len(nb['cells'])}")
