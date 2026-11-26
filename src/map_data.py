"""Map + lookup data for the GUI (approximate area centres, product colours, distance and fare tables)."""
import json
import math
from functools import lru_cache
from pathlib import Path

TABLES_PATH = Path(__file__).resolve().parents[1] / "models" / "gui_tables.json"

# Approximate centres of the 12 pickup/drop-off areas in the dataset (lat, lng) - for drawing only.
AREA_COORDS = {
    "Back Bay": (42.3503, -71.0810),
    "Beacon Hill": (42.3588, -71.0707),
    "Boston University": (42.3505, -71.1054),
    "Fenway": (42.3467, -71.0972),
    "Financial District": (42.3559, -71.0550),
    "Haymarket Square": (42.3638, -71.0583),
    "North End": (42.3647, -71.0542),
    "North Station": (42.3664, -71.0622),
    "Northeastern University": (42.3398, -71.0892),
    "South Station": (42.3519, -71.0552),
    "Theatre District": (42.3519, -71.0643),
    "West End": (42.3663, -71.0644),
}
PRODUCT_COLORS = {"Shared": "#8B5CF6", "Lyft": "#FF00BF", "Lyft XL": "#F97316",
                  "Lux": "#EAB308", "Lux Black": "#E5E7EB", "Lux Black XL": "#38BDF8"}
PRODUCT_ICONS = {"Shared": "👥", "Lyft": "🚗", "Lyft XL": "🚐", "Lux": "✨", "Lux Black": "🖤", "Lux Black XL": "🚙"}


@lru_cache(maxsize=1)
def load_tables() -> dict:
    return json.loads(TABLES_PATH.read_text())


def haversine_miles(a, b) -> float:
    (la1, lo1), (la2, lo2) = a, b
    p1, p2 = math.radians(la1), math.radians(la2)
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lo2 - lo1) / 2) ** 2
    return 3958.8 * 2 * math.asin(math.sqrt(h))


def trip_distance(source: str, destination: str) -> float:
    """Median real distance for this pair (miles). Pairs not seen in the data fall back to straight line x 1.3."""
    key = f"{source}|{destination}"
    table = load_tables()["distance"]
    if key in table:
        return table[key]
    return round(1.3 * haversine_miles(AREA_COORDS[source], AREA_COORDS[destination]), 2)


def estimate_fare(product: str, miles: float, surge: float = 1.0) -> float:
    """Typical non-surged fare for this distance (linear fit on real rides) x surge. An estimate, not a quote."""
    f = load_tables()["fare"][product]
    return round((f["base"] + f["per_mile"] * miles) * surge, 2)
