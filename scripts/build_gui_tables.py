"""Build the small lookup tables used by the GUI (run once; output is committed in models/gui_tables.json).

    python scripts/build_gui_tables.py

* distance: median real trip distance (miles) for every (pickup, drop-off) pair seen in the data
* fare:     per product, a straight-line fit  price ~ base + per_mile * distance  on NON-surged rides.
            The GUI shows  estimated fare = normal fare x predicted surge.  `price` is never a model feature.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "processed" / "uber_cleaned.csv",
                 usecols=["source", "destination", "name", "distance", "price", "surge_multiplier"])

pairs = df.groupby(["source", "destination"])["distance"].median().round(2)
distance = {f"{s}|{d}": float(v) for (s, d), v in pairs.items()}

normal = df[df["surge_multiplier"] == 1.0]
fare = {}
for product, g in normal.groupby("name"):
    per_mile, base = np.polyfit(g["distance"], g["price"], 1)
    fare[product] = {"base": round(float(base), 2), "per_mile": round(float(per_mile), 2), "rides": int(len(g))}

tables = {"distance": distance, "fare": fare, "max_distance": float(df["distance"].max()),
          "source": "uber_cleaned.csv (real Lyft quotes, Boston, Nov-Dec 2018)"}
(ROOT / "models" / "gui_tables.json").write_text(json.dumps(tables, indent=1))
print(f"{len(distance)} distance pairs, fares for {sorted(fare)}")
