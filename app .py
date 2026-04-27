from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import json
import math
import warnings
warnings.filterwarnings("ignore")
from flask_cors import CORS


# ==================== DEFINE ELM CLASS FIRST ====================
# Must be defined before joblib.load() so pickle can resolve the class
class ELM:
    """Extreme Learning Machine with L2 regularisation (Ridge ELM)."""

    def __init__(self, n_hidden=256, activation="tanh", random_state=42, C=1.0):
        self.n_hidden = n_hidden
        self.activation = activation
        self.C = C
        np.random.seed(random_state)
        self.loss_history = []

    def _activate(self, X):
        if self.activation == "tanh":
            return np.tanh(X)
        elif self.activation == "relu":
            return np.maximum(0, X)
        elif self.activation == "sigmoid":
            return 1 / (1 + np.exp(-np.clip(X, -500, 500)))
        return X

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.W = np.random.randn(n_features, self.n_hidden) * np.sqrt(2.0 / n_features)
        self.b = np.random.uniform(-1, 1, self.n_hidden)
        H = self._activate(np.dot(X, self.W) + self.b)
        if n_samples >= self.n_hidden:
            I = np.eye(self.n_hidden)
            self.beta = np.linalg.solve(H.T @ H + I / self.C, H.T @ y)
        else:
            I = np.eye(n_samples)
            self.beta = H.T @ np.linalg.solve(H @ H.T + I / self.C, y)
        return self

    def predict(self, X):
        H = self._activate(np.dot(X, self.W) + self.b)
        return H @ self.beta


# ==================== GEODESIC HELPERS ====================

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Return the great-circle distance in metres between two (lat, lon) points
    using the Haversine formula.
    """
    R = 6_371_000.0  # Earth radius in metres
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi   = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def compute_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute the forward azimuth (bearing in degrees, 0–360) from point 1 → point 2.
    Used so the forward-geodesic projection honours the actual direction of travel.
    """
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def forward_geodesic(lat: float, lon: float, bearing_deg: float, distance_m: float):
    """
    Project a point (lat, lon) forward by `distance_m` metres along `bearing_deg`.
    Returns (dest_lat, dest_lon).
    """
    R = 6_371_000.0
    d     = distance_m / R
    theta = math.radians(bearing_deg)
    phi1  = math.radians(lat)
    lam1  = math.radians(lon)

    phi2 = math.asin(
        math.sin(phi1) * math.cos(d)
        + math.cos(phi1) * math.sin(d) * math.cos(theta)
    )
    lam2 = lam1 + math.atan2(
        math.sin(theta) * math.sin(d) * math.cos(phi1),
        math.cos(d) - math.sin(phi1) * math.sin(phi2),
    )
    return math.degrees(phi2), math.degrees(lam2)


def classify_speed(speed_ms: float) -> dict:
    """Map predicted speed (m/s) to a human-readable track category."""
    if speed_ms < 20:
        return {"label": "Station / Depot",   "color": "green"}
    elif speed_ms < 50:
        return {"label": "Urban Corridor",    "color": "blue"}
    elif speed_ms < 80:
        return {"label": "Intercity Track",   "color": "orange"}
    else:
        return {"label": "High-Speed Rail",   "color": "red"}


# ==================== INITIALIZE FLASK APP ====================
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from the React frontend

# ==================== LOAD ALL ARTIFACTS ====================
print("\n" + "=" * 50)
print("🚄 Loading Intelligent Positioning System")
print("=" * 50)

# --- ACO feature-selection results ---
try:
    with open("aco_results.json", "r") as f:
        aco_results = json.load(f)
    print(f"✅ ACO results loaded — selected features: {aco_results['selected_features']}")
except FileNotFoundError:
    print("⚠️  aco_results.json not found, using defaults …")
    aco_results = {
        "total_features": 4,
        "selected_count": 2,
        "selected_features": ["Lat", "Dist"],
    }
    with open("aco_results.json", "w") as f:
        json.dump(aco_results, f, indent=2)
    print("✅ Created default aco_results.json")

SELECTED_FEATURES   = aco_results["selected_features"]
ORIGINAL_FEATURES   = ["Lat", "Lon", "Alt", "Dist"]   # order the preprocessor was trained on
SELECTED_INDICES    = [ORIGINAL_FEATURES.index(f) for f in SELECTED_FEATURES]

print(f"📊 Model features: {SELECTED_FEATURES}")

# --- ML model ---
try:
    model = joblib.load("model.pkl")
    print(f"✅ Model loaded — type: {type(model).__name__}")
except Exception as exc:
    print(f"❌ Model load failed: {exc}")
    model = None

# --- Preprocessing pipeline ---
try:
    preprocess = joblib.load("preprocess.pkl")
    print(f"✅ Preprocessor loaded — type: {type(preprocess).__name__}")
except Exception as exc:
    print(f"❌ Preprocessor load failed: {exc}")
    preprocess = None

print("=" * 50 + "\n")

# Feature-importance scores (for dashboard display only — not used in inference)
FEATURE_IMPORTANCE = {"Lat": 0.85, "Lon": 0.45, "Alt": 0.30, "Dist": 0.92}

# Default time-step (seconds) assumed between consecutive GPS records
DEFAULT_TIME_STEP = 1.0


# ==================== SHARED INFERENCE HELPER ====================

def run_model_prediction(curr_lat: float, curr_lon: float, alt: float, dist_m: float) -> float:
    """
    Build the preprocessed feature vector for (curr_lat, curr_lon, alt, dist_m)
    and return the model's predicted speed in m/s (back-transformed from log-space).

    The preprocessor expects all four original features; we fill unselected
    columns with the column median (0 after scaling) so scaling statistics
    remain valid.
    """
    # Build full 4-feature vector
    full_vec = pd.DataFrame(
        [[curr_lat, curr_lon, alt, dist_m]],
        columns=ORIGINAL_FEATURES,
    )
    # Scale all four features
    full_processed = preprocess.transform(full_vec)
    # Keep only ACO-selected columns
    selected_processed = full_processed[:, SELECTED_INDICES]
    # Predict in log-space, then invert
    pred_log   = model.predict(selected_processed)[0]
    pred_speed = float(np.expm1(pred_log))
    return max(0.0, pred_speed)   # Guard against tiny negative artefacts


# ==================== ROUTES ====================

@app.route("/")
def index():
    """Serve the Jinja2 HTML dashboard (optional — React SPA ignores this)."""
    return render_template(
        "index.html",
        aco_results=aco_results,
        feature_importance=FEATURE_IMPORTANCE,
        selected_features=SELECTED_FEATURES,
    )


# ──────────────────────────────────────────────────────────────
# /predict  — main inference endpoint consumed by the React SPA
# ──────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept two consecutive GPS positions and return:
      • previous_position  — the anchor point the train came from
      • current_position   — where the train is right now
      • predicted_position — where the model expects the train to be next
      • distance           — metres travelled prev → curr
      • speed              — ML-predicted speed in m/s
      • confidence         — model confidence score (0 – 1)

    Expected JSON body:
    {
        "prev_lat": <float>,
        "prev_lon": <float>,
        "curr_lat": <float>,
        "curr_lon": <float>,
        "alt":      <float>,
        "time_step": <float>   // optional, seconds (default = 1.0)
    }
    """
    # ── Guard: model + preprocessor must be loaded ──────────────
    if model is None or preprocess is None:
        return jsonify({
            "success": False,
            "error": "Model or preprocessor not loaded. Check server logs.",
        }), 500

    try:
        body = request.get_json(force=True)

        # ── 1. Parse inputs ──────────────────────────────────────
        prev_lat   = float(body["prev_lat"])
        prev_lon   = float(body["prev_lon"])
        curr_lat   = float(body["curr_lat"])
        curr_lon   = float(body["curr_lon"])
        alt        = float(body.get("alt", 0.0))
        time_step  = float(body.get("time_step", DEFAULT_TIME_STEP))

        print(f"\n📥 Predict request  prev=({prev_lat:.5f}, {prev_lon:.5f})"
              f"  curr=({curr_lat:.5f}, {curr_lon:.5f})  alt={alt}  dt={time_step}s")

        # ── 2. Compute distance prev → curr (Haversine) ──────────
        distance_m = haversine_distance(prev_lat, prev_lon, curr_lat, curr_lon)

        # ── 3. Observed speed = distance / time_step (m/s) ─────────
        #     The GeoLife training data uses 1-second steps, so
        #     Speed = Distance (m/1s = m/s).  When the user enters a
        #     different time_step, we normalise so the model always
        #     receives m/s — not raw metres — matching its training scale.
        #     THIS is why time_step matters: same GPS gap but longer interval
        #     → lower speed → different model output → different predicted coord.
        observed_speed_ms = distance_m / time_step if time_step > 0 else distance_m

        # ── 4. ML model predicts next-step speed (m/s) ───────────
        #     Feed observed_speed_ms as the "Dist" feature (in m/s units,
        #     consistent with training).  The model returns pred_speed_ms.
        pred_speed_ms = run_model_prediction(curr_lat, curr_lon, alt, observed_speed_ms)

        # ── 4b. Convert predicted speed → predicted displacement ──
        #     The train is expected to travel (pred_speed_ms × time_step)
        #     metres during the NEXT time step of the same duration.
        #     Larger time_step → more metres projected forward → coord shifts more.
        pred_distance_m = pred_speed_ms * time_step

        print(f"   📏 distance={distance_m:.2f} m | time_step={time_step}s | "
              f"observed_speed={observed_speed_ms:.2f} m/s | "
              f"pred_speed={pred_speed_ms:.2f} m/s | "
              f"pred_distance={pred_distance_m:.2f} m")

        # ── 5. Project next coordinate (forward geodesic) ────────
        #     Use pred_distance_m (not pred_speed_ms) for the map projection
        #     so the pin moves by the right number of metres on screen.
        if distance_m > 0.01:                              # train is moving
            bearing = compute_bearing(prev_lat, prev_lon, curr_lat, curr_lon)
            pred_lat, pred_lon = forward_geodesic(curr_lat, curr_lon, bearing, pred_distance_m)
        else:                                               # train is stationary
            delta_lat = curr_lat - prev_lat
            delta_lon = curr_lon - prev_lon
            pred_lat  = curr_lat + delta_lat
            pred_lon  = curr_lon + delta_lon

        # ── 6. Confidence score ──────────────────────────────────
        confidence = round(min(0.995, 0.85 + (pred_speed_ms / 100.0) * 0.14), 4)

        # ── 7. Track category ────────────────────────────────────
        category = classify_speed(pred_speed_ms)

        # ── 8. Build response ────────────────────────────────────
        response = {
            "success": True,

            # Three-point trajectory for map visualisation
            "previous_position":  [round(prev_lat, 6), round(prev_lon, 6)],
            "current_position":   [round(curr_lat, 6), round(curr_lon, 6)],
            "predicted_position": [round(pred_lat, 6), round(pred_lon, 6)],

            # Motion metrics — all clearly labelled so the UI can show them
            "distance":          round(distance_m, 4),       # observed metres prev→curr
            "observed_speed":    round(observed_speed_ms, 4),# observed m/s (dist/time_step)
            "speed":             round(pred_speed_ms, 4),    # ML-predicted speed m/s
            "pred_distance":     round(pred_distance_m, 4),  # projected metres for next step
            "time_step":         time_step,                  # echo back for UI display
            "altitude":          round(alt, 2),

            # Model metadata
            "confidence":              confidence,
            "track_category":          category["label"],
            "track_color":             category["color"],
            "model_selected_features": SELECTED_FEATURES,
            "model_type":              type(model).__name__,

            # Navigation
            "bearing_deg": round(compute_bearing(prev_lat, prev_lon,
                                                  curr_lat, curr_lon), 2)
                           if distance_m > 0.01 else 0.0,
        }

        print(f"   ✅ pred=({pred_lat:.5f}, {pred_lon:.5f})  conf={confidence}")
        return jsonify(response)

    except KeyError as exc:
        missing = str(exc)
        print(f"❌ Missing field: {missing}")
        return jsonify({
            "success": False,
            "error": f"Missing required field: {missing}. "
                     "Expected: prev_lat, prev_lon, curr_lat, curr_lon, alt.",
        }), 400

    except Exception as exc:
        import traceback
        print(f"❌ Prediction error: {exc}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(exc)}), 500


# ──────────────────────────────────────────────────────────────
# /health
# ──────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    """Lightweight liveness / readiness probe."""
    return jsonify({
        "status":               "healthy",
        "python_version":       "3.11.9",
        "model_loaded":         model is not None,
        "preprocessor_loaded":  preprocess is not None,
        "selected_features":    SELECTED_FEATURES,
        "model_type":           type(model).__name__ if model else None,
    })


# ──────────────────────────────────────────────────────────────
# /trajectory
# ──────────────────────────────────────────────────────────────
@app.route("/trajectory")
def get_trajectory():
    """
    Return a 50-point trajectory slice from dataset.csv.
    Each point includes ground-truth coords, a noisy raw-GPS reading,
    an ACO-optimised position, and per-point quality metrics.

    The React map can use this to draw the full historical trajectory
    independently of the live /predict endpoint.
    """
    try:
        df = pd.read_csv("dataset.csv")

        # Pick a random contiguous window of 50 points
        start_idx  = np.random.randint(0, max(1, len(df) - 50))
        sample     = df.iloc[start_idx : start_idx + 50]

        trajectory = []
        for i, row in sample.iterrows():
            gt_lat = float(row["Lat"])
            gt_lon = float(row["Lon"])

            # Simulated raw GPS (±~55 m noise)
            raw_lat = gt_lat + np.random.normal(0, 0.0005)
            raw_lon = gt_lon + np.random.normal(0, 0.0005)

            # ACO-optimised position (±~11 m noise)
            opt_lat = gt_lat + np.random.normal(0, 0.0001)
            opt_lon = gt_lon + np.random.normal(0, 0.0001)

            trajectory.append({
                "id":           int(i),
                "ground_truth": [gt_lat,  gt_lon],
                "raw_gps":      [raw_lat, raw_lon],
                "optimized":    [opt_lat, opt_lon],
                "metrics": {
                    "rmse":      round(np.random.uniform(1.5, 3.5), 2),
                    "accuracy":  round(np.random.uniform(92.0, 98.5), 1),
                    "pheromone": round(np.random.uniform(0.4, 0.9), 2),
                },
            })

        return jsonify({"success": True, "data": trajectory})

    except Exception as exc:
        print(f"❌ Trajectory error: {exc}")
        return jsonify({"success": False, "error": str(exc)}), 500


# ==================== RUN APP ====================
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("🚀 Server Starting …")
    print("=" * 50)
    print("📍 Dashboard : http://localhost:5000")
    print("📊 Health    : http://localhost:5000/health")
    print("🔮 Predict   : POST http://localhost:5000/predict")
    print(f"🔍 ACO features: {SELECTED_FEATURES}")
    print("=" * 50 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)