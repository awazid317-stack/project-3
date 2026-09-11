"""
PolarGrid AI Backend
Single-file FastAPI backend for the PolarGrid AI frontend.

Dataset expected:
polargrid_sensor_dataset.csv

Run:
    pip install -r requirements.txt
    uvicorn app:app --reload
"""

from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import math
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset.csv"

# These are engineering assumptions used only where the dataset does
# not provide a direct physical capacity/rating.
BATTERY_CAPACITY_KWH = 120.0
BATTERY_MAX_CHARGE_KW = 35.0
BATTERY_MAX_DISCHARGE_KW = 35.0
DIESEL_MAX_KW = 70.0
DIESEL_FUEL_PER_HOUR_AT_FULL_LOAD_PCT = 4.0

FORECAST_TARGETS = [
    "ambient_temp_C",
    "wind_speed_ms",
    "solar_irradiance_wm2",
    "solar_pv_power_kw",
    "wind_turbine_power_kw",
    "load_power_kw",
]

LAG_STEPS = [1, 2, 3, 6, 12, 24]

app = FastAPI(
    title="PolarGrid AI Backend",
    description="AI/ML backend for the Aurora Ridge Station microgrid dashboard.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For SIH prototype/development. Restrict in production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Global state
# -------------------------------------------------------------------

df = pd.DataFrame()
models = {}
model_metrics = {}
last_row = None
history = []


# -------------------------------------------------------------------
# Data + ML utilities
# -------------------------------------------------------------------


def load_dataset():
    global df, last_row, history

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. Put polargrid_sensor_dataset.csv "
            f"in the backend folder and rename it to dataset.csv."
        )

    data = pd.read_csv(DATA_PATH)

    required = [
        "timestamp",
        "ambient_temp_C",
        "wind_speed_ms",
        "humidity_pct",
        "solar_irradiance_wm2",
        "solar_pv_power_kw",
        "wind_turbine_power_kw",
        "diesel_generator_power_kw",
        "diesel_cumulative_runtime_hr",
        "diesel_fuel_level_pct",
        "battery_soc_pct",
        "battery_voltage_v",
        "battery_temp_C",
        "load_power_kw",
        "renewable_fraction_pct",
    ]

    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values("timestamp").reset_index(drop=True)

    numeric_cols = [c for c in required if c not in ["timestamp", "station_id"]]
    for c in numeric_cols:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    data[numeric_cols] = data[numeric_cols].interpolate().ffill().bfill()
    df = data
    last_row = df.iloc[-1].copy()
    history = df.tail(168).copy().to_dict("records")


def time_features(ts):
    hour = ts.hour
    dayofweek = ts.dayofweek
    dayofyear = ts.dayofyear
    return [
        hour,
        math.sin(2 * math.pi * hour / 24),
        math.cos(2 * math.pi * hour / 24),
        dayofweek,
        math.sin(2 * math.pi * dayofyear / 365),
        math.cos(2 * math.pi * dayofyear / 365),
    ]


def make_training_data(target):
    """Create lag/time features for one-step-ahead regression."""
    data = df.copy()

    feature_names = []
    X_rows = []
    y_rows = []

    for idx in range(max(LAG_STEPS), len(data)):
        ts = data.loc[idx, "timestamp"]
        features = time_features(ts)

        # Recent history of all forecast variables.
        for col in FORECAST_TARGETS:
            for lag in LAG_STEPS:
                features.append(float(data.loc[idx - lag, col]))
                feature_names.append(f"{col}_lag_{lag}")

        X_rows.append(features)
        y_rows.append(float(data.loc[idx, target]))

    return np.asarray(X_rows), np.asarray(y_rows)


def train_models():
    global models, model_metrics

    models = {}
    model_metrics = {}

    for target in FORECAST_TARGETS:
        X, y = make_training_data(target)

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = RandomForestRegressor(
            n_estimators=120,
            max_depth=16,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        model_metrics[target] = {
            "mae": round(float(mean_absolute_error(y_test, pred)), 3),
            "r2": round(float(r2_score(y_test, pred)), 3),
        }
        models[target] = model


def build_feature(ts, recent_values):
    features = time_features(ts)

    for col in FORECAST_TARGETS:
        values = recent_values[col]
        for lag in LAG_STEPS:
            if len(values) >= lag:
                features.append(float(values[-lag]))
            else:
                features.append(float(values[0]))

    return np.asarray(features).reshape(1, -1)


def clamp_prediction(target, value):
    limits = {
        "ambient_temp_C": (-60, 20),
        "wind_speed_ms": (0, 40),
        "solar_irradiance_wm2": (0, 1400),
        "solar_pv_power_kw": (0, 80),
        "wind_turbine_power_kw": (0, 100),
        "load_power_kw": (5, 100),
    }
    low, high = limits[target]
    return float(np.clip(value, low, high))


def forecast(hours=48, start_time=None):
    """Recursive hourly ML forecast."""
    if start_time is None:
        start_time = pd.Timestamp(last_row["timestamp"]) + pd.Timedelta(hours=1)

    recent = {
        col: list(df[col].tail(max(LAG_STEPS) + 5).astype(float).values)
        for col in FORECAST_TARGETS
    }

    results = []

    for step in range(hours):
        ts = pd.Timestamp(start_time) + pd.Timedelta(hours=step)
        row = {"timestamp": ts}

        # Predict each variable using the same recent simulated history.
        for target in FORECAST_TARGETS:
            x = build_feature(ts, recent)
            value = models[target].predict(x)[0]
            value = clamp_prediction(target, value)
            row[target] = value

        # Generation can never exceed an obviously positive load-independent range.
        row["solar_pv_power_kw"] = max(0.0, row["solar_pv_power_kw"])
        row["wind_turbine_power_kw"] = max(0.0, row["wind_turbine_power_kw"])

        # Keep history synchronized for recursive forecasting.
        for col in FORECAST_TARGETS:
            recent[col].append(row[col])

        results.append(row)

    return results


# -------------------------------------------------------------------
# Microgrid / dashboard calculations
# -------------------------------------------------------------------


def safe_float(value, default=0.0):
    try:
        if value is None or not np.isfinite(float(value)):
            return default
        return float(value)
    except Exception:
        return default


def current_state():
    r = last_row
    renewable_kw = safe_float(r["solar_pv_power_kw"]) + safe_float(
        r["wind_turbine_power_kw"]
    )
    load_kw = safe_float(r["load_power_kw"])
    renewable_fraction = (renewable_kw / load_kw * 100.0) if load_kw > 0 else 0.0

    return {
        "timestamp": str(pd.Timestamp(r["timestamp"])),
        "station_id": str(r.get("station_id", "PolarGrid-AuroraRidge-01")),
        "temperature_C": safe_float(r["ambient_temp_C"]),
        "wind_speed_ms": safe_float(r["wind_speed_ms"]),
        "humidity_pct": safe_float(r["humidity_pct"]),
        "solar_irradiance_wm2": safe_float(r["solar_irradiance_wm2"]),
        "solar_power_kw": safe_float(r["solar_pv_power_kw"]),
        "wind_power_kw": safe_float(r["wind_turbine_power_kw"]),
        "renewable_power_kw": round(renewable_kw, 2),
        "load_kw": round(load_kw, 2),
        "diesel_power_kw": safe_float(r["diesel_generator_power_kw"]),
        "diesel_fuel_pct": safe_float(r["diesel_fuel_level_pct"]),
        "diesel_runtime_hr": safe_float(r["diesel_cumulative_runtime_hr"]),
        "battery_soc_pct": safe_float(r["battery_soc_pct"]),
        "battery_voltage_v": safe_float(r["battery_voltage_v"]),
        "battery_temp_C": safe_float(r["battery_temp_C"]),
        "renewable_fraction_pct": round(renewable_fraction, 2),
    }


def dispatch_schedule(forecasts):
    """
    Hybrid dispatch controller:
    1. Renewable generation serves the load first.
    2. Excess renewable charges the battery.
    3. Battery discharges for deficits.
    4. Diesel covers the remaining deficit.

    This is intentionally a transparent SIH prototype controller rather
    than a claim of a production RL/MPC implementation.
    """
    soc = safe_float(last_row["battery_soc_pct"])
    diesel_fuel = safe_float(last_row["diesel_fuel_level_pct"])

    output = []
    for item in forecasts:
        solar = item["solar_pv_power_kw"]
        wind = item["wind_turbine_power_kw"]
        load = item["load_power_kw"]
        renewable = solar + wind

        battery_charge = 0.0
        battery_discharge = 0.0
        diesel = 0.0

        net = renewable - load

        if net >= 0:
            # Charge only until SOC reaches 95%.
            available_charge = min(net, BATTERY_MAX_CHARGE_KW)
            room_kw = max(0.0, (95.0 - soc) * BATTERY_CAPACITY_KWH / 100.0)
            battery_charge = min(available_charge, room_kw)
            soc += battery_charge / BATTERY_CAPACITY_KWH * 100.0
        else:
            deficit = -net

            # Preserve a 20% emergency reserve.
            available_battery = max(0.0, (soc - 20.0) * BATTERY_CAPACITY_KWH / 100.0)
            battery_discharge = min(
                deficit, BATTERY_MAX_DISCHARGE_KW, available_battery
            )
            soc -= battery_discharge / BATTERY_CAPACITY_KWH * 100.0

            remaining = deficit - battery_discharge
            diesel = min(remaining, DIESEL_MAX_KW)

            # Rough fuel-use estimate for simulation/visualization.
            diesel_fuel -= (
                diesel / DIESEL_MAX_KW
            ) * DIESEL_FUEL_PER_HOUR_AT_FULL_LOAD_PCT
            diesel_fuel = max(0.0, diesel_fuel)

        output.append(
            {
                "timestamp": str(item["timestamp"]),
                "solar_kw": round(solar, 2),
                "wind_kw": round(wind, 2),
                "battery_charge_kw": round(battery_charge, 2),
                "battery_discharge_kw": round(battery_discharge, 2),
                "diesel_kw": round(diesel, 2),
                "load_kw": round(load, 2),
                "soc_pct": round(float(np.clip(soc, 0, 100)), 2),
                "fuel_pct": round(diesel_fuel, 2),
            }
        )

    return output


def source_status():
    r = current_state()

    def status_from_power(power, threshold=0.5):
        return "ONLINE" if power > threshold else "STANDBY"

    solar_status = status_from_power(r["solar_power_kw"])
    wind_status = status_from_power(r["wind_power_kw"])
    diesel_status = "ONLINE" if r["diesel_power_kw"] > 0.5 else "STANDBY"
    battery_status = "CHARGING" if r["battery_soc_pct"] < 80 else "AVAILABLE"

    return [
        {
            "name": "Solar PV",
            "type": "Renewable",
            "status": solar_status,
            "power_kw": round(r["solar_power_kw"], 2),
            "capacity_kw": 80,
        },
        {
            "name": "Wind Turbine",
            "type": "Renewable",
            "status": wind_status,
            "power_kw": round(r["wind_power_kw"], 2),
            "capacity_kw": 100,
        },
        {
            "name": "Battery Storage",
            "type": "Storage",
            "status": battery_status,
            "soc_pct": round(r["battery_soc_pct"], 2),
            "capacity_kwh": BATTERY_CAPACITY_KWH,
        },
        {
            "name": "Diesel Generator",
            "type": "Backup",
            "status": diesel_status,
            "power_kw": round(r["diesel_power_kw"], 2),
            "fuel_pct": round(r["diesel_fuel_pct"], 2),
            "capacity_kw": DIESEL_MAX_KW,
        },
    ]


def equipment_health():
    r = current_state()

    # No health/failure label exists in the supplied dataset, so these
    # are transparent rule-based condition scores rather than trained
    # failure predictions.
    battery_score = 100.0
    if r["battery_temp_C"] < -20 or r["battery_temp_C"] > 35:
        battery_score -= 20
    if r["battery_voltage_v"] < 44 or r["battery_voltage_v"] > 54:
        battery_score -= 20
    if r["battery_soc_pct"] < 15:
        battery_score -= 15

    diesel_score = 100.0
    if r["diesel_fuel_pct"] < 20:
        diesel_score -= 30
    if r["diesel_runtime_hr"] > 7000:
        diesel_score -= 10

    wind_score = 100.0 if r["wind_speed_ms"] >= 2 else 85.0
    solar_score = 100.0 if r["solar_irradiance_wm2"] >= 20 else 95.0

    return [
        {
            "equipment": "Battery Bank",
            "health_pct": round(max(0, battery_score), 1),
            "status": "Healthy" if battery_score >= 80 else "Attention",
        },
        {
            "equipment": "Diesel Generator",
            "health_pct": round(max(0, diesel_score), 1),
            "status": "Healthy" if diesel_score >= 80 else "Attention",
        },
        {
            "equipment": "Wind Turbine",
            "health_pct": round(wind_score, 1),
            "status": "Healthy",
        },
        {
            "equipment": "Solar PV Array",
            "health_pct": round(solar_score, 1),
            "status": "Healthy",
        },
    ]


# -------------------------------------------------------------------
# Scenario simulator
# -------------------------------------------------------------------


class ScenarioRequest(BaseModel):
    storm_severity: float = Field(0, ge=0, le=100)
    wind_availability: float = Field(100, ge=0, le=100)
    days_without_resupply: float = Field(0, ge=0, le=240)
    generator_enabled: bool = True


def simulate_scenario(req: ScenarioRequest):
    base = forecast(72)

    soc = safe_float(last_row["battery_soc_pct"])
    fuel = safe_float(last_row["diesel_fuel_level_pct"])

    results = []
    diesel_runtime = 0.0

    # Storm reduces solar and load is increased slightly because harsh
    # conditions imply additional station demand.
    solar_factor = max(0.05, 1.0 - req.storm_severity / 125.0)
    load_factor = 1.0 + req.storm_severity / 500.0

    # Resupply delay increases conservation pressure gradually.
    resupply_factor = 1.0 + min(req.days_without_resupply, 240) / 1200.0

    for item in base:
        solar = item["solar_pv_power_kw"] * solar_factor
        wind = item["wind_turbine_power_kw"] * (req.wind_availability / 100.0)
        load = item["load_power_kw"] * load_factor * resupply_factor

        renewable = solar + wind
        deficit = max(0.0, load - renewable)
        surplus = max(0.0, renewable - load)

        battery_discharge = 0.0
        battery_charge = 0.0
        diesel = 0.0

        if surplus > 0:
            room = max(0.0, (95 - soc) * BATTERY_CAPACITY_KWH / 100)
            battery_charge = min(surplus, BATTERY_MAX_CHARGE_KW, room)
            soc += battery_charge / BATTERY_CAPACITY_KWH * 100
        else:
            available = max(0.0, (soc - 15) * BATTERY_CAPACITY_KWH / 100)
            battery_discharge = min(deficit, BATTERY_MAX_DISCHARGE_KW, available)
            soc -= battery_discharge / BATTERY_CAPACITY_KWH * 100

            remaining = deficit - battery_discharge
            if req.generator_enabled and remaining > 0 and fuel > 0:
                diesel = min(remaining, DIESEL_MAX_KW)
                diesel_runtime += 1.0
                fuel -= (diesel / DIESEL_MAX_KW) * DIESEL_FUEL_PER_HOUR_AT_FULL_LOAD_PCT
                fuel = max(0.0, fuel)

        results.append(
            {
                "timestamp": str(item["timestamp"]),
                "soc_pct": round(float(np.clip(soc, 0, 100)), 2),
                "solar_kw": round(solar, 2),
                "wind_kw": round(wind, 2),
                "load_kw": round(load, 2),
                "diesel_kw": round(diesel, 2),
                "fuel_pct": round(fuel, 2),
            }
        )

    return {
        "parameters": req.model_dump(),
        "summary": {
            "initial_soc_pct": round(safe_float(last_row["battery_soc_pct"]), 2),
            "final_soc_pct": round(results[-1]["soc_pct"], 2),
            "initial_fuel_pct": round(safe_float(last_row["diesel_fuel_level_pct"]), 2),
            "final_fuel_pct": round(results[-1]["fuel_pct"], 2),
            "diesel_runtime_hours": round(diesel_runtime, 2),
            "minimum_soc_pct": round(min(x["soc_pct"] for x in results), 2),
        },
        "hours": results,
    }


# -------------------------------------------------------------------
# API endpoints
# -------------------------------------------------------------------


@app.get("/")
def root():
    return {
        "message": "PolarGrid AI Backend is running",
        "docs": "/docs",
        "dataset_rows": len(df),
        "station": str(last_row.get("station_id", "PolarGrid-AuroraRidge-01")),
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "dataset_loaded": not df.empty,
        "models_loaded": len(models),
        "rows": len(df),
    }


@app.get("/api/model-info")
def model_info():
    return {
        "algorithm": "RandomForestRegressor",
        "forecast_targets": FORECAST_TARGETS,
        "forecast_horizon_hours": [24, 48, 168],
        "metrics": model_metrics,
        "note": "Equipment health is rule-based because the supplied dataset has no equipment-failure/health label.",
    }


@app.get("/api/overview")
def overview():
    state = current_state()
    next_24 = forecast(24)

    renewable_forecast = [
        {
            "timestamp": str(x["timestamp"]),
            "wind_kw": round(x["wind_turbine_power_kw"], 2),
            "solar_kw": round(x["solar_pv_power_kw"], 2),
            "renewable_kw": round(
                x["wind_turbine_power_kw"] + x["solar_pv_power_kw"], 2
            ),
            "load_kw": round(x["load_power_kw"], 2),
        }
        for x in next_24
    ]

    return {
        "current": state,
        "forecast_24h": renewable_forecast,
        "model": {
            "name": "Random Forest Time-Series Forecast",
            "status": "trained",
        },
    }


@app.get("/api/forecast")
def get_forecast(horizon: int = 48):
    if horizon not in [24, 48, 168]:
        raise HTTPException(status_code=400, detail="horizon must be 24, 48, or 168")

    predictions = forecast(horizon)

    return {
        "horizon_hours": horizon,
        "generated_at": datetime.now().isoformat(),
        "predictions": [
            {
                "timestamp": str(x["timestamp"]),
                "temperature_C": round(x["ambient_temp_C"], 2),
                "wind_speed_ms": round(x["wind_speed_ms"], 2),
                "solar_irradiance_wm2": round(x["solar_irradiance_wm2"], 2),
                "solar_power_kw": round(x["solar_pv_power_kw"], 2),
                "wind_power_kw": round(x["wind_turbine_power_kw"], 2),
                "load_kw": round(x["load_power_kw"], 2),
                "renewable_kw": round(
                    x["solar_pv_power_kw"] + x["wind_turbine_power_kw"], 2
                ),
            }
            for x in predictions
        ],
    }


@app.get("/api/dispatch")
def get_dispatch(horizon: int = 24):
    if horizon not in [24, 48, 168]:
        raise HTTPException(status_code=400, detail="horizon must be 24, 48, or 168")

    predictions = forecast(horizon)
    schedule = dispatch_schedule(predictions)

    return {
        "strategy": "Hybrid MPC-style rule-based dispatch",
        "tabs": ["MPC", "RL Agent", "Hybrid"],
        "schedule": schedule,
        "note": "The current dataset does not contain an RL reward/action history, so the backend uses a transparent hybrid optimization heuristic instead of pretending an RL policy was trained.",
    }


@app.get("/api/sources")
def get_sources():
    return {"sources": source_status()}


@app.get("/api/kpis")
def get_kpis():
    state = current_state()

    recent = df.tail(24)
    avg_load = float(recent["load_power_kw"].mean())
    avg_renewable = float(
        (recent["solar_pv_power_kw"] + recent["wind_turbine_power_kw"]).mean()
    )
    renewable_fraction = avg_renewable / avg_load * 100 if avg_load > 0 else 0

    diesel_energy = float(recent["diesel_generator_power_kw"].sum())
    total_load_energy = float(recent["load_power_kw"].sum())

    return {
        "renewable_fraction_pct": round(renewable_fraction, 2),
        "battery_soc_pct": round(state["battery_soc_pct"], 2),
        "station_load_kw": round(state["load_kw"], 2),
        "diesel_generator_kw": round(state["diesel_power_kw"], 2),
        "diesel_fuel_pct": round(state["diesel_fuel_pct"], 2),
        "avg_load_24h_kw": round(avg_load, 2),
        "avg_renewable_24h_kw": round(avg_renewable, 2),
        "diesel_energy_24h_kwh": round(diesel_energy, 2),
        "load_energy_24h_kwh": round(total_load_energy, 2),
        "estimated_co2_kg_24h": round(diesel_energy * 0.74, 2),
    }


@app.get("/api/equipment")
def get_equipment():
    return {"equipment": equipment_health()}


@app.get("/api/communications")
def get_communications():
    return {
        "station": "Aurora Ridge Station",
        "status": "Operational",
        "network": "Satellite Link",
        "latency_ms": 820,
        "uplink": "Available",
        "downlink": "Available",
        "last_sync": str(pd.Timestamp(last_row["timestamp"])),
        "message": "Microgrid telemetry synchronized with local station data.",
    }


@app.post("/api/simulate")
def run_simulation(req: ScenarioRequest):
    return simulate_scenario(req)


# -------------------------------------------------------------------
# Startup
# -------------------------------------------------------------------


@app.on_event("startup")
def startup_event():
    load_dataset()
    train_models()
    print("=" * 60)
    print("PolarGrid AI backend started")
    print(f"Dataset: {DATA_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Models trained: {len(models)}")
    print("Open: http://127.0.0.1:8000/docs")
    print("=" * 60)
