"""
F1 Analytics 2026 — FastAPI Backend
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
import joblib
import json
from pathlib import Path
import uvicorn

app = FastAPI(title="F1 Analytics 2026 API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR    = Path(__file__).parent.parent
MODELS_DIR  = BASE_DIR / "models"
EXPORTS_DIR = BASE_DIR / "exports"
PLOTS_DIR   = BASE_DIR / "plots"

win_model      = None
pos_model      = None
feature_list   = []
model_metadata = {}

@app.on_event("startup")
def load_models():
    global win_model, pos_model, feature_list, model_metadata
    try:
        win_model = joblib.load(MODELS_DIR / "winner_lgbm.joblib")
        pos_model = joblib.load(MODELS_DIR / "position_lgbm.joblib")
        print("✅ Models loaded")
    except Exception as e:
        print(f"⚠️  Models not loaded: {e}")
    try:
        with open(MODELS_DIR / "feature_list.json") as f:
            feature_list = json.load(f)
        with open(MODELS_DIR / "model_metadata.json") as f:
            model_metadata = json.load(f)
    except Exception as e:
        print(f"⚠️  Metadata not loaded: {e}")


# ══════════════════════════════════════════════════════════════
# BASIC
# ══════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {"status": "ok", "message": "F1 Analytics 2026 API", "version": "1.0.0"}

@app.get("/health")
def health():
    return {
        "models_loaded":  win_model is not None,
        "features_count": len(feature_list),
        "model_metadata": model_metadata,
    }


# ══════════════════════════════════════════════════════════════
# PREDICTIONS
# ══════════════════════════════════════════════════════════════

@app.get("/predictions/latest")
def get_latest_predictions():
    try:
        with open(EXPORTS_DIR / "latest_predictions.json") as f:
            return {"status": "ok", "predictions": json.load(f)}
    except FileNotFoundError:
        raise HTTPException(404, "No predictions available yet")

@app.get("/predictions/race")
def get_race_predictions(year: int, round: int):
    try:
        with open(EXPORTS_DIR / "latest_predictions.json") as f:
            return {"status": "ok", "year": year, "round": round, "predictions": json.load(f)}
    except FileNotFoundError:
        raise HTTPException(404, "No predictions available")


# ══════════════════════════════════════════════════════════════
# STANDINGS + ELO
# ══════════════════════════════════════════════════════════════

@app.get("/standings/elo")
def get_elo_standings():
    try:
        with open(EXPORTS_DIR / "elo_standings.json") as f:
            return {"status": "ok", "standings": json.load(f)}
    except FileNotFoundError:
        raise HTTPException(404, "Elo standings not available")

@app.get("/standings/championship")
def get_championship():
    try:
        with open(EXPORTS_DIR / "standings_2025.json") as f:
            return {"status": "ok", "standings": json.load(f)}
    except FileNotFoundError:
        raise HTTPException(404, "Championship standings not available")


# ══════════════════════════════════════════════════════════════
# RACES
# ══════════════════════════════════════════════════════════════

@app.get("/races")
def get_races(year: int = 2026):
    races = [
        {"RoundNumber": 1,  "EventName": "Australian Grand Prix",     "EventDate": "2026-03-15", "Country": "Australia",   "Location": "Melbourne"},
        {"RoundNumber": 2,  "EventName": "Chinese Grand Prix",        "EventDate": "2026-03-22", "Country": "China",        "Location": "Shanghai"},
        {"RoundNumber": 3,  "EventName": "Japanese Grand Prix",       "EventDate": "2026-04-05", "Country": "Japan",        "Location": "Suzuka"},
        {"RoundNumber": 4,  "EventName": "Bahrain Grand Prix",        "EventDate": "2026-04-19", "Country": "Bahrain",      "Location": "Sakhir"},
        {"RoundNumber": 5,  "EventName": "Saudi Arabian Grand Prix",  "EventDate": "2026-04-26", "Country": "Saudi Arabia", "Location": "Jeddah"},
        {"RoundNumber": 6,  "EventName": "Miami Grand Prix",          "EventDate": "2026-05-10", "Country": "USA",          "Location": "Miami"},
        {"RoundNumber": 7,  "EventName": "Emilia Romagna Grand Prix", "EventDate": "2026-05-24", "Country": "Italy",        "Location": "Imola"},
        {"RoundNumber": 8,  "EventName": "Monaco Grand Prix",         "EventDate": "2026-05-31", "Country": "Monaco",       "Location": "Monaco"},
        {"RoundNumber": 9,  "EventName": "Spanish Grand Prix",        "EventDate": "2026-06-14", "Country": "Spain",        "Location": "Barcelona"},
        {"RoundNumber": 10, "EventName": "Canadian Grand Prix",       "EventDate": "2026-06-21", "Country": "Canada",       "Location": "Montreal"},
        {"RoundNumber": 11, "EventName": "Austrian Grand Prix",       "EventDate": "2026-07-05", "Country": "Austria",      "Location": "Spielberg"},
        {"RoundNumber": 12, "EventName": "British Grand Prix",        "EventDate": "2026-07-19", "Country": "UK",           "Location": "Silverstone"},
        {"RoundNumber": 13, "EventName": "Hungarian Grand Prix",      "EventDate": "2026-08-02", "Country": "Hungary",      "Location": "Budapest"},
        {"RoundNumber": 14, "EventName": "Belgian Grand Prix",        "EventDate": "2026-08-30", "Country": "Belgium",      "Location": "Spa"},
        {"RoundNumber": 15, "EventName": "Dutch Grand Prix",          "EventDate": "2026-09-06", "Country": "Netherlands",  "Location": "Zandvoort"},
        {"RoundNumber": 16, "EventName": "Italian Grand Prix",        "EventDate": "2026-09-13", "Country": "Italy",        "Location": "Monza"},
        {"RoundNumber": 17, "EventName": "Azerbaijan Grand Prix",     "EventDate": "2026-09-27", "Country": "Azerbaijan",   "Location": "Baku"},
        {"RoundNumber": 18, "EventName": "Singapore Grand Prix",      "EventDate": "2026-10-04", "Country": "Singapore",    "Location": "Singapore"},
        {"RoundNumber": 19, "EventName": "United States Grand Prix",  "EventDate": "2026-10-18", "Country": "USA",          "Location": "Austin"},
        {"RoundNumber": 20, "EventName": "Mexico City Grand Prix",    "EventDate": "2026-10-25", "Country": "Mexico",       "Location": "Mexico City"},
        {"RoundNumber": 21, "EventName": "São Paulo Grand Prix",      "EventDate": "2026-11-08", "Country": "Brazil",       "Location": "São Paulo"},
        {"RoundNumber": 22, "EventName": "Las Vegas Grand Prix",      "EventDate": "2026-11-21", "Country": "USA",          "Location": "Las Vegas"},
        {"RoundNumber": 23, "EventName": "Qatar Grand Prix",          "EventDate": "2026-11-29", "Country": "Qatar",        "Location": "Lusail"},
        {"RoundNumber": 24, "EventName": "Abu Dhabi Grand Prix",      "EventDate": "2026-12-06", "Country": "UAE",          "Location": "Abu Dhabi"},
    ]
    return {"status": "ok", "year": year, "races": races}


# ══════════════════════════════════════════════════════════════
# RACE RESULTS + LAP TIMES
# ══════════════════════════════════════════════════════════════

@app.get("/race/{year}/{round}/results")
def get_race_results(year: int, round: int):
    try:
        with open(EXPORTS_DIR / "race_results.json") as f:
            data = json.load(f)
        filtered = [r for r in data if r.get("year") == year and r.get("round") == round]
        if not filtered:
            raise HTTPException(404, f"No results for {year} Round {round}")
        return {"status": "ok", "results": filtered}
    except FileNotFoundError:
        raise HTTPException(404, "Race results not available")

@app.get("/race/{year}/{round}/lap-times")
def get_lap_times(year: int, round: int, driver: Optional[str] = None):
    try:
        with open(EXPORTS_DIR / "race_results.json") as f:
            data = json.load(f)
        filtered = [r for r in data if r.get("year") == year and r.get("round") == round]
        if driver:
            filtered = [r for r in filtered if r.get("code") == driver]
        if not filtered:
            raise HTTPException(404, "No lap data found")
        return {"status": "ok", "laps": filtered}
    except FileNotFoundError:
        raise HTTPException(404, "Data not available")


# ══════════════════════════════════════════════════════════════
# MODEL INFO
# ══════════════════════════════════════════════════════════════

@app.get("/model/feature-importance")
def get_feature_importance():
    try:
        with open(EXPORTS_DIR / "feature_importance.json") as f:
            return {"status": "ok", "feature_importance": json.load(f)}
    except FileNotFoundError:
        if win_model and feature_list:
            fi = dict(zip(feature_list, win_model.feature_importances_.tolist()))
            fi = dict(sorted(fi.items(), key=lambda x: -x[1])[:30])
            return {"status": "ok", "feature_importance": fi}
        raise HTTPException(404, "Feature importance not available")


# ══════════════════════════════════════════════════════════════
# STRATEGY
# ══════════════════════════════════════════════════════════════

class StrategyRequest(BaseModel):
    driver_name:      str            = "Driver"
    base_pace:        float          = 91.0
    grid_position:    int            = 1
    current_compound: str            = "MEDIUM"
    current_tyre_age: int            = 0
    num_laps:         int            = 57
    pit_lane_loss:    float          = 22.0
    track_temp:       float          = 35.0
    safety_car_prob:  float          = 0.04
    n_simulations:    int            = 300
    strategies:       Optional[List[dict]] = None


@app.post("/strategy/simulate")
def simulate_strategy(req: StrategyRequest):
    deg = {
        "SOFT":   lambda age, t: 0.082 * age * (1 + (t - 35) * 0.005),
        "MEDIUM": lambda age, t: 0.048 * age * (1 + (t - 35) * 0.005),
        "HARD":   lambda age, t: 0.028 * age * (1 + (t - 35) * 0.005),
        "INTER":  lambda age, t: 0.040 * age * (1 + (t - 35) * 0.003),
        "WET":    lambda age, t: 0.025 * age * (1 + (t - 35) * 0.002),
    }

    strategies = req.strategies or [
        {"name": "1-stop S→H",   "stops": [int(req.num_laps * 0.38)],
         "compounds": ["SOFT", "HARD"]},
        {"name": "1-stop M→H",   "stops": [int(req.num_laps * 0.42)],
         "compounds": ["MEDIUM", "HARD"]},
        {"name": "2-stop S→M→H", "stops": [int(req.num_laps * 0.28), int(req.num_laps * 0.58)],
         "compounds": ["SOFT", "MEDIUM", "HARD"]},
        {"name": "2-stop S→S→H", "stops": [int(req.num_laps * 0.23), int(req.num_laps * 0.50)],
         "compounds": ["SOFT", "SOFT", "HARD"]},
    ]

    n_sims  = min(req.n_simulations, 500)
    results = []

    for strat in strategies:
        times = []
        for sim in range(n_sims):
            np.random.seed(sim)
            total = 0.0
            comp_idx = 0
            tyre_age = 0
            sc = 0
            for lap in range(1, req.num_laps + 1):
                sc = max(0, sc - 1)
                if sc == 0 and np.random.random() < req.safety_car_prob:
                    sc = 4
                if lap in strat["stops"]:
                    comp_idx += 1
                    tyre_age  = 0
                    total    += req.pit_lane_loss + np.random.normal(0, 0.4)
                tyre_age += 1
                comp   = strat["compounds"][min(comp_idx, len(strat["compounds"]) - 1)]
                deg_fn = deg.get(comp, deg["MEDIUM"])
                lt     = req.base_pace + deg_fn(tyre_age, req.track_temp) + np.random.normal(0, 0.15)
                if sc > 0:
                    lt += 14
                total += lt
            times.append(total)

        arr = np.array(times)
        results.append({
            "strategy":     strat["name"],
            "compounds":    strat["compounds"],
            "stops":        strat["stops"],
            "mean_time_s":  round(float(arr.mean()), 2),
            "std_time_s":   round(float(arr.std()),  2),
            "best_time_s":  round(float(arr.min()),  2),
            "win_rate":     0.0,
            "podium_rate":  0.0,
            "dnf_rate":     0.01,
            "expected_pos": float(req.grid_position),
        })

    return {
        "status": "ok",
        "driver": req.driver_name,
        "results": sorted(results, key=lambda x: x["mean_time_s"]),
    }


@app.get("/strategy/undercut")
def predict_undercut(
    gap_to_car_ahead: float = Query(...),
    our_tyre_age:     int   = Query(...),
    their_tyre_age:   int   = Query(...),
    our_compound:     str   = Query("MEDIUM"),
    their_compound:   str   = Query("MEDIUM"),
    pit_lane_loss:    float = Query(22.0),
    laps_remaining:   int   = Query(20),
):
    deg_rates  = {"SOFT": 0.082, "MEDIUM": 0.048, "HARD": 0.028}
    our_rate   = deg_rates.get(our_compound,   0.048)
    their_rate = deg_rates.get(their_compound, 0.048)
    delta      = their_rate * their_tyre_age - their_rate
    laps_to    = pit_lane_loss / max(delta, 0.05)

    prob = 1 / (1 + np.exp(-(
        -0.30 * gap_to_car_ahead
        + 0.20 * our_tyre_age
        - 0.10 * their_tyre_age
        + 5.00 * (our_rate - their_rate)
        + 0.05 * laps_remaining
        - 0.05 * pit_lane_loss
    )))

    return {
        "status":                "ok",
        "undercut_success_prob": round(float(prob), 3),
        "laps_to_recoup":        round(float(laps_to), 1),
        "recommended":           bool(prob > 0.5),
        "gap_to_car_ahead":      gap_to_car_ahead,
        "our_tyre_age":          our_tyre_age,
    }


@app.get("/strategy/tyre-models")
def get_tyre_models():
    try:
        with open(EXPORTS_DIR / "tyre_deg_models.json") as f:
            return {"status": "ok", "models": json.load(f)}
    except FileNotFoundError:
        raise HTTPException(404, "Tyre models not available")


# ══════════════════════════════════════════════════════════════
# TELEMETRY — graceful 503
# ══════════════════════════════════════════════════════════════

@app.get("/telemetry/{year}/{round}/{driver}")
def get_telemetry(year: int, round: int, driver: str):
    raise HTTPException(503, "Telemetry requires FastF1 — not available on this deployment")


# ══════════════════════════════════════════════════════════════
# STATIC FILES
# ══════════════════════════════════════════════════════════════

if PLOTS_DIR.exists():
    app.mount("/plots",   StaticFiles(directory=str(PLOTS_DIR)),   name="plots")
if EXPORTS_DIR.exists():
    app.mount("/exports", StaticFiles(directory=str(EXPORTS_DIR)), name="exports")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
