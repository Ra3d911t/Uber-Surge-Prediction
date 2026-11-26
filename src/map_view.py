"""Interactive Boston map with OSRM road routing and illustrative animated vehicles.

Leaflet and map tiles load in the browser. OSRM supplies road-following route geometry;
a straight fallback is used only if the public routing service is unavailable.
Animated cars are illustrative, not live GPS positions.
"""
import json

LEAFLET_VERSION = "1.9.4"

_TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet@__LV__/dist/leaflet.css"/>
<style>
  html, body, #map { height: 100%; margin: 0; background: #f4f4f4; font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif; }
  .chip { position: absolute; z-index: 1000; color: #fff; background: rgba(0,0,0,.92); border: 1px solid #fff; border-radius: 10px; padding: 9px 14px; font-size: 13px; line-height: 1.35; }
  #route { top: 12px; left: 12px; max-width: 62%; }
  #surge { top: 12px; right: 12px; font-weight: 700; font-size: 15px; }
  #status { bottom: 16px; left: 12px; }
  #msg { top: 50%; left: 50%; transform: translate(-50%,-50%); text-align: center; display: none; }
  .car { width: 22px; height: 22px; }
  .sq { width: 16px; height: 16px; background: #000; border: 3px solid #fff; box-shadow: 0 0 0 2px #000; }
  .leaflet-container { background: #f4f4f4; }
  /* Keep the familiar street map, but render its tiles in Uber-style grayscale. */
  .leaflet-tile { filter: grayscale(1) contrast(1.08); }
  .leaflet-tooltip { background: #000; color: #fff; border: 1px solid #fff; box-shadow: none; font-weight: 600; }
  .leaflet-tooltip-top:before { border-top-color: #000; }
</style></head>
<body>
<div id="map"></div>
<div class="chip" id="route"></div>
<div class="chip" id="surge"></div>
<div class="chip" id="status">Looking for a driver…</div>
<div class="chip" id="msg">The map needs an internet connection<br/>(Leaflet / map tiles could not be loaded).</div>
<script src="https://unpkg.com/leaflet@__LV__/dist/leaflet.js"></script>
<script>
const D = __DATA__;
const $ = (id) => document.getElementById(id);

function surgeColor(v) { return v < 1.1 ? "#2ecc71" : (v < 1.3 ? "#f5a623" : "#e74c3c"); }

(function main() {
  if (typeof L === "undefined") { $("msg").style.display = "block"; return; }

  const names = Object.keys(D.areas);
  const P = D.areas[D.pickup];
  const Q = D.dropoff ? D.areas[D.dropoff] : null;
  const rnd = (a, b) => a + Math.random() * (b - a);

  const map = L.map("map", { zoomControl: true, attributionControl: true }).setView([42.355, -71.075], 13);
  // free tiles, no API key: OpenStreetMap first, Esri street map as a fallback
  const PROVIDERS = [
    { url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png", attr: "&copy; OpenStreetMap contributors", maxZoom: 19 },
    { url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
      attr: "Tiles &copy; Esri", maxZoom: 19 }
  ];
  let provider = 0, tiles = null, loaded = 0, failed = 0;
  function addTiles() {
    loaded = 0; failed = 0;
    tiles = L.tileLayer(PROVIDERS[provider].url, { maxZoom: PROVIDERS[provider].maxZoom, attribution: PROVIDERS[provider].attr });
    tiles.on("tileload", () => { loaded += 1; });
    tiles.on("tileerror", () => {
      failed += 1;
      if (loaded === 0 && failed >= 4) {            // nothing loads from this provider -> try the next one
        map.removeLayer(tiles);
        provider += 1;
        if (provider < PROVIDERS.length) addTiles();
        else { $("msg").innerHTML = "Map tiles could not be loaded.<br/>Check the internet connection."; $("msg").style.display = "block"; }
      }
    });
    tiles.addTo(map);
  }
  addTiles();

  // every area as a small dot
  names.forEach((n) => {
    if (n === D.pickup || n === D.dropoff) return;
    L.circleMarker(D.areas[n], { radius: 5, color: "#fff", fillColor: "#555", fillOpacity: 0.95, weight: 2 })
      .addTo(map).bindTooltip(n);
  });

  // Monochrome pickup/drop-off markers in an Uber-inspired black-and-white palette.
  L.circleMarker(P, { radius: 9, color: "#fff", fillColor: "#000", fillOpacity: 1, weight: 3 })
    .addTo(map).bindTooltip("Pickup: " + D.pickup, { permanent: true, direction: "top", offset: [0, -8] });
  if (Q) {
    L.marker(Q, { icon: L.divIcon({ className: "", html: '<div class="sq"></div>', iconSize: [16, 16], iconAnchor: [8, 8] }) })
      .addTo(map).bindTooltip("Drop-off: " + D.dropoff, { permanent: true, direction: "top", offset: [0, -8] });
  }

  // OSRM road-following route; GeoJSON coordinates are converted to Leaflet [lat, lon].
  // If the public routing service is unavailable, use a clearly labelled straight fallback.
  let R = null;
  let routeReady = !Q;
  let routeLayer = null;
  function drawRoute(points) {
    if (routeLayer) map.removeLayer(routeLayer);
    routeLayer = L.layerGroup([
      L.polyline(points, { color: "#fff", weight: 10, opacity: 1, lineCap: "round", lineJoin: "round" }),
      L.polyline(points, { color: "#000", weight: 5, opacity: 1, lineCap: "round", lineJoin: "round" })
    ]).addTo(map);
    map.fitBounds(L.latLngBounds(points), { padding: [65, 65] });
  }
  if (Q) {
    const routeUrl = "https://router.project-osrm.org/route/v1/driving/" +
      P[1] + "," + P[0] + ";" + Q[1] + "," + Q[0] +
      "?overview=full&geometries=geojson&steps=false";
    fetch(routeUrl)
      .then((response) => { if (!response.ok) throw new Error("Routing service returned " + response.status); return response.json(); })
      .then((result) => {
        if (!result.routes || !result.routes.length) throw new Error("No drivable route found");
        R = result.routes[0].geometry.coordinates.map((xy) => [xy[1], xy[0]]);
        drawRoute(R);
        $("route").textContent = D.pickup + "  →  " + D.dropoff + "  ·  " +
          (result.routes[0].distance / 1609.344).toFixed(1) + " mi";
        routeReady = true;
      })
      .catch((error) => {
        console.warn("OSRM road routing unavailable; using illustrative fallback:", error);
        R = [P, Q];
        drawRoute(R);
        $("status").textContent = "Road routing unavailable · illustrative fallback";
        routeReady = true;
      });
  } else {
    map.setView(P, 14);
  }

  // ---- cars -----------------------------------------------------------------------------------
  function carSvg(color) {
    return '<svg viewBox="0 0 24 24" width="22" height="22"><rect x="6" y="2" width="12" height="20" rx="4" fill="' + color +
      '" stroke="#fff" stroke-width="1.6"/><rect x="8" y="6" width="8" height="5" rx="1.5" fill="#cfe8ff"/>' +
      '<rect x="8" y="15" width="8" height="3" rx="1" fill="#222"/></svg>';
  }
  function makeCar(color, pos, size) {
    const icon = L.divIcon({ className: "", html: '<div class="car">' + carSvg(color) + "</div>", iconSize: [size, size], iconAnchor: [size / 2, size / 2] });
    const m = L.marker(pos, { icon: icon, interactive: false, keyboard: false }).addTo(map);
    return { m: m, pos: [pos[0], pos[1]] };
  }
  function setHeading(car, deg) {
    const el = car.m.getElement && car.m.getElement();
    if (el && el.firstChild) el.firstChild.style.transform = "rotate(" + deg + "deg)";
  }
  const bearing = (a, b) => Math.atan2((b[1] - a[1]) * Math.cos(a[0] * Math.PI / 180), b[0] - a[0]) * 180 / Math.PI;
  const dist = (a, b) => Math.hypot((b[1] - a[1]) * Math.cos(a[0] * Math.PI / 180), b[0] - a[0]);
  function step(car, target, speed, dt) {          // speed in degrees / second; true when target reached
    const d = dist(car.pos, target), mv = speed * dt;
    if (d < 1e-9) return true;
    if (mv >= d) { car.pos = [target[0], target[1]]; car.m.setLatLng(car.pos); return true; }
    const f = mv / d;
    setHeading(car, bearing(car.pos, target));
    car.pos = [car.pos[0] + (target[0] - car.pos[0]) * f, car.pos[1] + (target[1] - car.pos[1]) * f];
    car.m.setLatLng(car.pos);
    return false;
  }
  function randPoint() {
    const c = D.areas[names[Math.floor(Math.random() * names.length)]];
    return [c[0] + rnd(-0.004, 0.004), c[1] + rnd(-0.004, 0.004)];
  }

  const ambient = [];
  for (let i = 0; i < 16; i++) {
    const c = makeCar("#777777", randPoint(), 22);
    c.target = randPoint(); c.speed = rnd(0.0003, 0.0006);
    ambient.push(c);
  }

  // "your" driver, in the colour of the chosen ride type
  const start = [P[0] + 0.007, P[1] - 0.007];
  const car = makeCar("#000000", start, 22);
  const driver = { phase: "approach", t: 0, i: 0 };
  const setStatus = (s) => { $("status").textContent = s; };

  function driverStep(dt) {
    if (driver.phase === "approach") {
      if (step(car, P, 0.0010, dt)) { driver.phase = "wait1"; driver.t = 1.5; setStatus("Your driver has arrived at the pickup"); }
    } else if (driver.phase === "wait1") {
      driver.t -= dt;
      if (driver.t <= 0) {
        if (R && routeReady) { driver.phase = "trip"; driver.i = 0; setStatus("Trip in progress"); }
        else { driver.t = 0.5; setStatus("Waiting for road route…"); }
      }
    } else if (driver.phase === "trip") {
      if (step(car, R[driver.i], 0.0008, dt)) {
        driver.i += 1;
        if (driver.i >= R.length) { driver.phase = "wait2"; driver.t = 2.5; setStatus("You have arrived"); }
      }
    } else {
      driver.t -= dt;
      if (driver.t <= 0) { car.pos = [start[0], start[1]]; car.m.setLatLng(car.pos); driver.phase = "approach"; setStatus("Your driver is on the way"); }
    }
  }

  // ---- overlays ------------------------------------------------------------------------------
  $("route").textContent = D.dropoff ? (D.pickup + "  →  " + D.dropoff + "  ·  " + D.miles.toFixed(1) + " mi") : D.pickup;
  $("surge").textContent = D.product + " · surge " + D.surge.toFixed(2) + "x";
  $("surge").style.borderLeft = "5px solid #fff";
  setStatus("Your driver is on the way");

  let last = null;
  function frame(ts) {
    const dt = last === null ? 0 : Math.min((ts - last) / 1000, 0.1);
    last = ts;
    ambient.forEach((c) => { if (step(c, c.target, c.speed, dt)) c.target = randPoint(); });
    driverStep(dt);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
</script></body></html>"""


def build_map_html(areas: dict, pickup: str, dropoff, miles: float, surge: float, product: str, color: str) -> str:
    """Return a self-contained HTML page for st.components.v1.html. `dropoff` may be None."""
    data = {"areas": {k: list(v) for k, v in areas.items()}, "pickup": pickup, "dropoff": dropoff,
            "miles": float(miles), "surge": float(surge), "product": product, "color": color}
    payload = json.dumps(data).replace("</", "<\\/")
    return _TEMPLATE.replace("__LV__", LEAFLET_VERSION).replace("__DATA__", payload)
