"""
app/streamlit_app.py — Enhanced Monsoon Farm RL Dashboard

New features:
  ✅ Live Bengaluru weather via Open-Meteo API (no key needed)
  ✅ Real mandi prices via data.gov.in / agmarknet with smart fallback
  ✅ 2D farm map with crop icons + health/pest bars
  ✅ Profit/loss pie chart per crop
  ✅ Yield heatmap across slots
  ✅ Pest spread heatmap + line chart
  ✅ 3-day weather forecast panel (real + simulation)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time
from datetime import datetime, timedelta
try:
    import pytz
    _IST = pytz.timezone("Asia/Kolkata")
    def _now_ist():
        return datetime.now(_IST)
    def _ist_hour():
        return _now_ist().hour
    def _ist_time_str():
        return _now_ist().strftime("%d %b %Y  %I:%M:%S %p IST")
    def _ist_short():
        return _now_ist().strftime("%H:%M IST")
except ImportError:
    def _now_ist():
        return datetime.utcnow() + timedelta(hours=5, minutes=30)
    def _ist_hour():
        return _now_ist().hour
    def _ist_time_str():
        return _now_ist().strftime("%d %b %Y  %I:%M:%S %p IST")
    def _ist_short():
        return _now_ist().strftime("%H:%M IST")

def _is_daytime():
    """Return True if it is currently daytime in Bengaluru (6 AM – 7 PM IST)."""
    h = _ist_hour()
    return 6 <= h < 19

def _apply_night(weather_type: str) -> str:
    """If it's night and sky is clear, swap SUNNY to CLEAR_NIGHT."""
    if not _is_daytime() and weather_type == "SUNNY":
        return "CLEAR_NIGHT"
    return weather_type

from env.environment import MonsoonFarmEnv
from env.models import CropStage, WeatherType, FarmAction
from baseline.agent import HeuristicAgent
from grader.grader import get_grader

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Monsoon Farm RL",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
BENGALURU_LAT = 12.9716
BENGALURU_LON = 77.5946

CROP_ICONS = {"spinach": "🥬", "lettuce": "🥗", "tomato": "🍅", "herbs": "🌿", "none": "⬜"}
WEATHER_ICONS = {
    "SUNNY": "☀️", "CLEAR_NIGHT": "🌙", "CLOUDY": "☁️", "LIGHT_RAIN": "🌧️",
    "HEAVY_RAIN": "⛈️", "HEATWAVE": "🔥", "DRY_SPELL": "🏜️"
}
BASE_PRICES = {"spinach": 35.0, "lettuce": 80.0, "tomato": 25.0, "herbs": 120.0}

WMO_TO_TYPE = {
    0: "SUNNY", 1: "SUNNY", 2: "CLOUDY", 3: "CLOUDY",
    45: "CLOUDY", 48: "CLOUDY",
    51: "LIGHT_RAIN", 53: "LIGHT_RAIN", 55: "LIGHT_RAIN",
    61: "LIGHT_RAIN", 63: "LIGHT_RAIN", 65: "HEAVY_RAIN",
    80: "LIGHT_RAIN", 81: "HEAVY_RAIN", 82: "HEAVY_RAIN",
    95: "HEAVY_RAIN", 96: "HEAVY_RAIN", 99: "HEAVY_RAIN",
}

# ─────────────────────────────────────────────
# Real Data Fetching
# ─────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_bengaluru_weather():
    """Fetch live weather + 3-day forecast from Open-Meteo (no API key needed). Refreshes every 5 min."""
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": BENGALURU_LAT,
            "longitude": BENGALURU_LON,
            "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
            "timezone": "Asia/Kolkata",
            "forecast_days": 4,
        }
        r = requests.get(url, params=params, timeout=6)
        r.raise_for_status()
        data = r.json()
        cur = data["current"]
        daily = data["daily"]
        current = {
            "temperature_c": cur["temperature_2m"],
            "humidity_pct": cur["relative_humidity_2m"],
            "rainfall_mm": cur["precipitation"],
            "wind_kmh": cur.get("wind_speed_10m", 0),
            "weather_type": _apply_night(WMO_TO_TYPE.get(cur["weather_code"], "SUNNY")),
            "fetched_at": _ist_short(),
        }
        forecast = []
        for i in range(1, 4):
            forecast.append({
                "date": daily["time"][i],
                "temp_max": daily["temperature_2m_max"][i],
                "temp_min": daily["temperature_2m_min"][i],
                "rain_mm": daily["precipitation_sum"][i],
                "weather_type": WMO_TO_TYPE.get(daily["weather_code"][i], "SUNNY"),
            })
        return {"current": current, "forecast": forecast, "source": "Open-Meteo (live · updates every 5 min)"}
    except Exception as e:
        return {"error": str(e)}


def _scrape_agmarknet_commodity(commodity_code: int, commodity_name: str,
                                date_str: str, state_code: str = "KK") -> list:
    """
    Scrape Agmarknet for one commodity using the per-commodity URL pattern.
    This is the ONLY method that works — the site is ASP.NET WebForms and
    'all commodities' (Tx_Commodity=0) requires ViewState POST; individual
    commodity GETs return a static HTML table that pandas.read_html parses cleanly.

    Returns list of modal prices in ₹/kg (converted from ₹/quintal ÷ 100).
    date_str format: "08-Apr-2026"
    """
    req_headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-IN,en;q=0.9",
    }
    url = (
        f"https://agmarknet.gov.in/SearchCmmMkt.aspx"
        f"?Tx_Commodity={commodity_code}"
        f"&Tx_State={state_code}"
        f"&Tx_District=0&Tx_Market=0"
        f"&DateFrom={date_str}&DateTo={date_str}"
        f"&Fr_Date={date_str}&To_Date={date_str}"
        f"&Tx_Trend=0"
        f"&Tx_CommodityHead={commodity_name}"
        f"&Tx_StateHead={'Karnataka' if state_code == 'KK' else '--Select--'}"
        f"&Tx_DistrictHead=--Select--&Tx_MarketHead=--Select--"
    )
    r = requests.get(url, headers=req_headers, timeout=12)
    r.raise_for_status()

    # pandas.read_html is the robust way — handles ASP.NET GridView encoding
    tables = pd.read_html(r.text)
    modal_prices = []
    for df in tables:
        cols_lower = [str(c).lower() for c in df.columns]
        # Look for the price table — must have "modal" column
        modal_cols = [i for i, c in enumerate(cols_lower) if "modal" in c]
        if not modal_cols:
            continue
        modal_col = df.columns[modal_cols[0]]
        for val in df[modal_col]:
            try:
                p = float(str(val).replace(",", "").strip())
                if p > 0:
                    # Agmarknet prices are ₹/quintal — divide by 100 for ₹/kg
                    modal_prices.append(round(p / 100, 2))
            except (ValueError, TypeError):
                pass
    return modal_prices


@st.cache_data(ttl=1800)
def fetch_mandi_prices():
    """
    Fetch Bengaluru mandi prices by scraping Agmarknet per-commodity URLs.
    Uses pandas.read_html on per-commodity GETs — the only reliable method
    for this ASP.NET portal without Selenium or ViewState handling.
    TTL=1800s (mandi data updates once daily, usually by 10 AM IST).
    """
    import random

    # Agmarknet commodity codes (verified from portal dropdown):
    # Tomato=78, Coriander=23, Spinach=100, Amaranthus(similar to spinach)=3
    # Lettuce is rare in Indian mandis — no standard code; estimated.
    COMMODITY_QUERIES = [
        # (app_crop_key,  agmarknet_code, agmarknet_name)
        ("tomato",  78,  "Tomato"),
        ("spinach", 100, "Spinach"),
        ("spinach", 3,   "Amaranthus"),   # fallback for spinach
        ("herbs",   23,  "Coriander"),
        ("herbs",   157, "Mint"),         # fallback for herbs
    ]

    now_ist = _now_ist()
    errors  = []
    all_prices = {k: [] for k in BASE_PRICES}

    # Try today, yesterday, day-before (data uploads by ~10 AM IST each morning)
    for days_back in range(3):
        target_date = now_ist - timedelta(days=days_back)
        date_str = target_date.strftime("%d-%b-%Y")   # e.g. "08-Apr-2026"

        for app_crop, code, name in COMMODITY_QUERIES:
            # Try Karnataka first, then All-India fallback
            for state_code in ("KK", "0"):
                try:
                    prices = _scrape_agmarknet_commodity(code, name, date_str, state_code)
                    if prices:
                        all_prices[app_crop].extend(prices)
                        break   # Got data for this commodity — skip All-India fallback
                except Exception as e:
                    errors.append(f"{name} {date_str} ({state_code}): {e}")

        if any(len(v) > 0 for v in all_prices.values()):
            break   # Got at least one commodity — stop going further back

    # ── Build result ──
    result    = {c: (round(sum(v) / len(v), 2) if v else None) for c, v in all_prices.items()}
    live_crops = [c for c, v in result.items() if v is not None]

    # Seasonal adjustment factors used for any missing crops
    month = now_ist.month
    if month in (4, 5, 6):
        season_adj = {"spinach": 0.90, "lettuce": 0.95, "tomato": 1.30, "herbs": 1.10}
    elif month in (7, 8, 9):
        season_adj = {"spinach": 1.15, "lettuce": 1.10, "tomato": 0.80, "herbs": 1.05}
    else:
        season_adj = {"spinach": 1.0, "lettuce": 1.0, "tomato": 1.0, "herbs": 1.0}

    rng = random.Random(now_ist.toordinal())

    if live_crops:
        estimated = []
        for c in result:
            if result[c] is None:
                result[c] = round(BASE_PRICES[c] * season_adj[c] * rng.uniform(0.88, 1.15), 2)
                estimated.append(c)
        label = f"✅ Agmarknet Karnataka · {_ist_short()}"
        if estimated:
            label += f"  (estimated: {', '.join(estimated)})"
        return {"prices": result, "source": label, "fetched_at": _ist_short(), "live": True}

    # ── All scrape attempts failed — seasonal fallback ──
    fallback = {
        c: round(BASE_PRICES[c] * season_adj[c] * rng.uniform(0.88, 1.22), 2)
        for c in BASE_PRICES
    }
    # Show only the most recent unique error per commodity to keep the message readable
    unique_errors = list(dict.fromkeys(
        e.split(":")[0] for e in errors   # just the "Name date (state)" prefix
    ))[:3]
    err_summary = " | ".join(unique_errors) if unique_errors else "unknown"
    return {
        "prices": fallback,
        "source": f"🟡 Seasonal estimate · Agmarknet unavailable ({err_summary})",
        "fetched_at": _ist_short(),
        "live": False,
    }


# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
def init_session():
    defaults = {
        "env": None, "agent": HeuristicAgent(eco_mode=True),
        "history": [], "done": False, "running": False,
        "crop_revenue": {c: 0.0 for c in BASE_PRICES},
        "slot_yield_history": {},
        "pest_history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("Farm Controls")
    st.divider()
    task = st.selectbox("Task Difficulty", ["easy", "medium", "hard"], index=1)
    seed = st.number_input("Random Seed", value=42, min_value=0, max_value=9999)
    agent_type = st.radio("Agent Type", ["Heuristic (Expert Rules)", "Random"])
    eco_mode = st.checkbox("Eco Mode", value=True)
    render_frequency = st.slider("Auto-play speed (ms/step)", 50, 2000, 300)
    st.divider()

    task_config = {
        "easy":   {"slots": 8,  "days": 60,  "budget": 75000, "month": 12},
        "medium": {"slots": 12, "days": 90,  "budget": 50000, "month": 6},
        "hard":   {"slots": 16, "days": 120, "budget": 30000, "month": 5},
    }
    cfg = task_config[task]
    st.info(f"**{task.upper()}**: {cfg['slots']} slots · {cfg['days']} days · ₹{cfg['budget']:,}")

    if st.button("🔄 Start / Reset", use_container_width=True):
        env = MonsoonFarmEnv(
            num_slots=cfg["slots"], episode_length=cfg["days"],
            start_month=cfg["month"], seed=int(seed),
            initial_budget_inr=cfg["budget"],
        )
        env.reset(seed=int(seed))
        st.session_state.env = env
        st.session_state.agent = HeuristicAgent(eco_mode=eco_mode)
        st.session_state.history = []
        st.session_state.done = False
        st.session_state.running = False
        st.session_state.crop_revenue = {c: 0.0 for c in BASE_PRICES}
        st.session_state.slot_yield_history = {i: [] for i in range(cfg["slots"])}
        st.session_state.pest_history = []

    col1, col2 = st.columns(2)
    with col1:
        step_btn = st.button("⏭ Step", use_container_width=True)
    with col2:
        if st.session_state.running:
            if st.button("⏹ Stop", use_container_width=True, type="primary"):
                st.session_state.running = False
                st.rerun()
        else:
            if st.button("▶ Auto-play", use_container_width=True):
                st.session_state.running = True
                st.rerun()

    # Live data in sidebar
    st.divider()
    st.subheader("Live Data")

    # ── IST system clock (always current) ──
    st.markdown(
        f"<div style='font-size:12px;color:#888;margin-bottom:4px;'>🕐 System Clock (IST)</div>"
        f"<div style='font-size:15px;font-weight:600;letter-spacing:0.5px;'>{_ist_time_str()}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='margin-bottom:6px'></div>", unsafe_allow_html=True)

    if st.button("🔄 Refresh Live Data", use_container_width=True):
        fetch_bengaluru_weather.clear()
        fetch_mandi_prices.clear()

    wd = fetch_bengaluru_weather()
    md = fetch_mandi_prices()

    if "error" not in wd:
        cur = wd["current"]
        wicon = WEATHER_ICONS.get(cur["weather_type"], "🌡️")
        st.markdown(f"**Bengaluru Now** {wicon}")
        st.caption(f"Last fetched: {cur['fetched_at']} · updates every 5 min")
        ca, cb = st.columns(2)
        ca.metric("🌡️ Temp", f"{cur['temperature_c']}°C")
        cb.metric("💧 Humidity", f"{cur['humidity_pct']}%")
        cc, cd = st.columns(2)
        cc.metric("🌧️ Rain", f"{cur['rainfall_mm']} mm")
        cd.metric("💨 Wind", f"{cur.get('wind_kmh', 0)} km/h")
    else:
        st.warning(f"Weather offline: {wd['error']}")

    st.divider()
    live_badge = "🟢 Live" if md.get("live") else "🟡 Estimated"
    st.markdown(f"**Mandi ₹/kg** {live_badge}")
    if md.get("live"):
        st.caption(md["source"])
    prices = md["prices"]
    pa, pb = st.columns(2)
    with pa:
        st.metric("🥬 Spinach", f"₹{prices['spinach']}")
        st.metric("🍅 Tomato", f"₹{prices['tomato']}")
    with pb:
        st.metric("🥗 Lettuce", f"₹{prices['lettuce']}")
        st.metric("🌿 Herbs", f"₹{prices['herbs']}")


# ─────────────────────────────────────────────
# Step helper
# ─────────────────────────────────────────────
def run_step():
    env = st.session_state.env
    if env is None or st.session_state.done:
        return
    state = env.state()
    action = (
        st.session_state.agent.act(state)
        if agent_type == "Heuristic (Expert Rules)"
        else env.action_space_sample()
    )
    obs, reward, done, info = env.step(action)
    st.session_state.done = done
    state_after = env.state()

    # Track slot yields
    for slot in state_after.crop_slots:
        sid = slot.slot_id
        if sid not in st.session_state.slot_yield_history:
            st.session_state.slot_yield_history[sid] = []
        st.session_state.slot_yield_history[sid].append(slot.expected_yield_kg)

    # Track pest
    for slot in state_after.crop_slots:
        if slot.pest_pressure > 0:
            st.session_state.pest_history.append({
                "day": info["day"],
                "slot_id": slot.slot_id,
                "pest_pressure": slot.pest_pressure,
                "crop": slot.crop_type,
            })

    # Crop revenue attribution
    if info["harvested_kg"] > 0:
        active = [s.crop_type for s in state.crop_slots
                  if s.crop_type != "none" and s.stage in (
                      int(CropStage.HARVEST), int(CropStage.MATURE))]
        if active:
            per = info["step_revenue_inr"] / len(active)
            for c in active:
                if c in st.session_state.crop_revenue:
                    st.session_state.crop_revenue[c] += per

    st.session_state.history.append({
        "day": info["day"],
        "reward": reward,
        "revenue": info["step_revenue_inr"],
        "cost": info["step_cost_inr"],
        "profit_cumulative": state_after.total_profit_inr,
        "yield_cumulative": state_after.total_yield_kg,
        "eco_score": info["eco_score"],
        "budget": info["budget_inr"],
        "temperature": state_after.weather.temperature_c,
        "rainfall": state_after.weather.rainfall_mm,
        "water_tank": state_after.resources.water_tank_liters,
        "weather_type": WeatherType(state_after.weather.weather_type).name,
        "harvested_kg": info["harvested_kg"],
    })


# ─────────────────────────────────────────────
# Guard: not started
# ─────────────────────────────────────────────
if st.session_state.env is None:
    st.title("Smart Monsoon-Resilient Hydroponic Farm")
    st.markdown("**Bengaluru Rooftop Farm · RL Simulation**")
    st.info("👈 Click **🔄 Start / Reset** in the sidebar to begin.")

    wd = fetch_bengaluru_weather()
    md = fetch_mandi_prices()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🌤️ Live Bengaluru Weather")
        st.caption(f"🕐 {_ist_time_str()}")
        if "error" not in wd:
            cur = wd["current"]
            wtype = cur["weather_type"]
            st.markdown(f"## {WEATHER_ICONS.get(wtype,'🌡️')} {wtype.replace('_',' ')}")
            m1, m2, m3 = st.columns(3)
            m1.metric("Temp", f"{cur['temperature_c']}°C")
            m2.metric("Humidity", f"{cur['humidity_pct']}%")
            m3.metric("Rain", f"{cur['rainfall_mm']} mm")
            st.metric("💨 Wind", f"{cur.get('wind_kmh', 0)} km/h")
            st.markdown("**3-Day Forecast**")
            fc = wd.get("forecast", [])
            fcols = st.columns(len(fc)) if fc else []
            for i, (fcol, day) in enumerate(zip(fcols, fc)):
                icon = WEATHER_ICONS.get(day["weather_type"], "🌡️")
                date_label = (_now_ist() + timedelta(days=i+1)).strftime("%a")
                fcol.markdown(
                    f"<div style='text-align:center;background:#f0f2f6;border-radius:8px;padding:8px;color:#1a1a1a;'>"
                    f"<b style='color:#1a1a1a;'>{date_label}</b><br><span style='font-size:22px'>{icon}</span><br>"
                    f"<span style='color:#c0392b;'>🔺{day['temp_max']}°</span> <span style='color:#2980b9;'>🔻{day['temp_min']}°</span><br>"
                    f"<span style='color:#1a1a1a;'>🌧️{day['rain_mm']}mm</span></div>", unsafe_allow_html=True
                )
        else:
            st.warning("Could not fetch live weather.")

    with c2:
        st.subheader("Live Bengaluru Mandi Prices")
        live_badge = "🟢 Live from Agmarknet" if md.get("live") else "🟡 Seasonal estimate"
        st.caption(live_badge)
        prices = md["prices"]
        chart_title = f"Agmarknet Karnataka · {md['fetched_at']}" if md.get("live") else "Bengaluru Market Prices (Seasonal Estimate)"
        fig = go.Figure(go.Bar(
            x=[f"{CROP_ICONS[c]} {c.capitalize()}" for c in prices],
            y=list(prices.values()),
            marker_color=["#4CAF50", "#8BC34A", "#FF5722", "#009688"],
            text=[f"₹{v}" for v in prices.values()],
            textposition="outside",
        ))
        fig.update_layout(
            title=chart_title, yaxis_title="₹/kg", height=300,
            margin=dict(t=60, b=20), template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    st.stop()

# ─────────────────────────────────────────────
# Step / Auto-play
# ─────────────────────────────────────────────
if step_btn and not st.session_state.done:
    run_step()

if st.session_state.running:
    if st.session_state.done:
        st.session_state.running = False
    else:
        run_step()
        time.sleep(render_frequency / 1000.0)
        st.rerun()

# ─────────────────────────────────────────────
# Dashboard
# ─────────────────────────────────────────────
env = st.session_state.env
state = env.state()
history = st.session_state.history

st.title("Monsoon Farm RL Dashboard")
if st.session_state.running:
    st.info("▶ Auto-playing… click **⏹ Stop** in sidebar to pause.")
elif st.session_state.done:
    st.success("✅ Episode Complete!")

# Top metrics
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Day", f"{state.day}/{state.episode_length}")
m2.metric("💰 Profit", f"₹{state.total_profit_inr:,.0f}")
m3.metric("🌾 Yield", f"{state.total_yield_kg:.1f} kg")
m4.metric("🌿 Eco", f"{state.eco.eco_score:.3f}")
m5.metric("💧 Water", f"{state.resources.water_tank_liters:.0f} L")
m6.metric("💵 Budget", f"₹{state.resources.budget_inr:,.0f}")

st.divider()

# ─────────────────────────────────────────────
# Real-World Data Row
# ─────────────────────────────────────────────
st.subheader("Live Real-World Data")
rw1, rw2, rw3 = st.columns([1, 1.3, 1])

with rw1:
    wd = fetch_bengaluru_weather()
    st.markdown("**☁️ Bengaluru Live Weather**")
    st.caption(f"🕐 {_ist_time_str()}")
    if "error" not in wd:
        cur = wd["current"]
        wtype = cur["weather_type"]
        st.markdown(f"### {WEATHER_ICONS.get(wtype,'🌡️')} {wtype.replace('_',' ')}")
        st.caption(f"*{wd['source']} · fetched {cur['fetched_at']}*")
        ca, cb = st.columns(2)
        ca.metric("🌡️ Temp", f"{cur['temperature_c']}°C")
        cb.metric("💧 Humidity", f"{cur['humidity_pct']}%")
        cc, cd = st.columns(2)
        cc.metric("🌧️ Rain", f"{cur['rainfall_mm']} mm")
        cd.metric("💨 Wind", f"{cur.get('wind_kmh', 0)} km/h")
    else:
        st.warning("Weather API offline")

with rw2:
    st.markdown("**Real 3-Day Forecast**")
    if "error" not in wd and wd.get("forecast"):
        fc = wd["forecast"]
        fcols = st.columns(3)
        for i, (col, day) in enumerate(zip(fcols, fc)):
            icon = WEATHER_ICONS.get(day["weather_type"], "🌡️")
            dlabel = (_now_ist() + timedelta(days=i+1)).strftime("%a %d")
            col.markdown(
                f"<div style='text-align:center;background:#f0f8ff;border-radius:10px;padding:10px;color:#1a1a1a;'>"
                f"<b style='color:#1a1a1a;'>{dlabel}</b><br><span style='font-size:26px'>{icon}</span><br>"
                f"<span style='color:#c0392b;'>🔺{day['temp_max']}°C</span><br><span style='color:#2980b9;'>🔻{day['temp_min']}°C</span><br>"
                f"<span style='color:#1a1a1a;'>🌧️{day['rain_mm']}mm</span></div>", unsafe_allow_html=True
            )
    else:
        st.info("Forecast unavailable")

with rw3:
    md = fetch_mandi_prices()
    prices = md["prices"]
    live_badge = "🟢 Live" if md.get("live") else "🟡 Estimated"
    st.markdown(f"**Mandi Prices (₹/kg)** {live_badge}")
    if md.get("live"):
        st.caption(f"*{md['fetched_at']}*")
    for crop, price in prices.items():
        delta = round(price - BASE_PRICES[crop], 1)
        st.metric(
            f"{CROP_ICONS[crop]} {crop.capitalize()}",
            f"₹{price}",
            f"{delta:+.1f} vs base"
        )

st.divider()

# ─────────────────────────────────────────────
# Simulation Weather + Forecast
# ─────────────────────────────────────────────
st.subheader("🌦️ Simulation Weather")
sw1, sw2 = st.columns([1, 2])

with sw1:
    wtype_name = WeatherType(state.weather.weather_type).name
    st.markdown(f"### {WEATHER_ICONS.get(wtype_name,'🌡️')} {wtype_name.replace('_',' ')}")
    c1, c2 = st.columns(2)
    c1.metric("🌡️ Temp", f"{state.weather.temperature_c:.1f}°C")
    c2.metric("💧 Hum", f"{state.weather.humidity_pct:.0f}%")
    st.metric("🌧️ Rainfall", f"{state.weather.rainfall_mm:.1f} mm")
    st.metric("Monsoon", "Yes 🌧️" if state.weather.is_monsoon_season else "No ☀️")
    if state.active_pest_alert:
        st.error("🐛 PEST ALERT!")

with sw2:
    st.markdown("**Simulation 3-Day Forecast (what the agent sees)**")
    type_names = ["SUNNY", "CLOUDY", "LIGHT_RAIN", "HEAVY_RAIN", "HEATWAVE", "DRY_SPELL"]
    fc_vals = state.weather.forecast_next_3_days
    fcs = st.columns(3)
    for i, (col, fval) in enumerate(zip(fcs, fc_vals)):
        fname = type_names[fval] if fval < len(type_names) else "SUNNY"
        ficon = WEATHER_ICONS.get(fname, "🌡️")
        col.markdown(
            f"<div style='text-align:center;background:#e8f5e9;border-radius:10px;padding:12px;color:#1a1a1a;'>"
            f"<b style='color:#1a1a1a;'>Day +{i+1}</b><br><span style='font-size:26px'>{ficon}</span><br>"
            f"<span style='color:#1a1a1a;'>{fname.replace('_',' ')}</span></div>", unsafe_allow_html=True
        )

st.divider()

# ─────────────────────────────────────────────
# 2D Farm Map
# ─────────────────────────────────────────────
st.subheader("🗺️ Farm Map")

stage_colors = {
    0: "#f5f5f5", 1: "#c8e6c9", 2: "#66bb6a",
    3: "#2e7d32", 4: "#fdd835", 5: "#e53935",
}
stage_names = {0:"Empty", 1:"Seeding ", 2:"Juvenile 🌿", 3:"Mature 🌳", 4:"Ready! 🎉", 5:"Dead 💀"}
COLS_PER_ROW = 4

num_slots = len(state.crop_slots)
for row_i in range((num_slots + COLS_PER_ROW - 1) // COLS_PER_ROW):
    row_slots = state.crop_slots[row_i * COLS_PER_ROW:(row_i + 1) * COLS_PER_ROW]
    cols = st.columns(COLS_PER_ROW)
    for col, slot in zip(cols, row_slots):
        color = stage_colors.get(slot.stage, "#f5f5f5")
        crop_icon = CROP_ICONS.get(slot.crop_type, "⬜")
        crop_name = slot.crop_type.capitalize() if slot.crop_type != "none" else "Empty"
        stage_label = stage_names.get(slot.stage, "?")
        hp = int(slot.health * 100)
        pp = int(slot.pest_pressure * 100)
        hc = "#4caf50" if hp > 60 else "#ff9800" if hp > 30 else "#f44336"
        pc = "#4caf50" if pp < 20 else "#ff9800" if pp < 50 else "#f44336"
        with col:
            st.markdown(
                f"""<div style="background:{color};border-radius:12px;padding:10px 8px;
                    text-align:center;font-size:13px;border:2px solid rgba(0,0,0,0.08);
                    min-height:145px;box-shadow:0 2px 6px rgba(0,0,0,0.08);">
                    <span style="font-size:26px">{crop_icon}</span><br>
                    <b>#{slot.slot_id} {crop_name}</b><br>
                    <span style="font-size:11px;color:#555">{stage_label}</span><br>
                    <div style="background:#e0e0e0;border-radius:4px;height:7px;margin:4px 2px 1px;">
                      <div style="background:{hc};width:{hp}%;height:7px;border-radius:4px;"></div>
                    </div>
                    <span style="font-size:10px;color:#666">❤️ {hp}% health</span><br>
                    <div style="background:#e0e0e0;border-radius:4px;height:7px;margin:3px 2px 1px;">
                      <div style="background:{pc};width:{pp}%;height:7px;border-radius:4px;"></div>
                    </div>
                    <span style="font-size:10px;color:#666">🐛 {pp}% pest</span>
                </div>""",
                unsafe_allow_html=True
            )
    st.markdown("<div style='margin-bottom:6px'></div>", unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────
# Charts (5 tabs)
# ─────────────────────────────────────────────
if len(history) > 1:
    df = pd.DataFrame(history)
    st.subheader("📊 Analytics")
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Performance", "🌦️ Weather",
        "💰 Economics", "🥧 Crop Breakdown", "🐛 Pest Spread"
    ])

    with tab1:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=["Cumulative Profit (₹)", "Cumulative Yield (kg)",
                            "Eco Score", "Water Tank (L)"],
        )
        fig.add_trace(go.Scatter(x=df["day"], y=df["profit_cumulative"],
                                 line=dict(color="#2196F3", width=2),
                                 fill="tozeroy", fillcolor="rgba(33,150,243,0.1)",
                                 name="Profit"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["day"], y=df["yield_cumulative"],
                                 line=dict(color="#4CAF50", width=2),
                                 fill="tozeroy", fillcolor="rgba(76,175,80,0.1)",
                                 name="Yield"), row=1, col=2)
        fig.add_trace(go.Scatter(x=df["day"], y=df["eco_score"],
                                 line=dict(color="#8BC34A", width=2), name="Eco"), row=2, col=1)
        fig.add_trace(go.Scatter(x=df["day"], y=df["water_tank"],
                                 line=dict(color="#03A9F4", width=2),
                                 fill="tozeroy", fillcolor="rgba(3,169,244,0.1)",
                                 name="Water"), row=2, col=2)
        fig.update_layout(height=460, showlegend=False, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig2 = make_subplots(rows=2, cols=1,
                             subplot_titles=["Temperature (°C)", "Rainfall (mm)"],
                             shared_xaxes=True)
        fig2.add_trace(go.Scatter(x=df["day"], y=df["temperature"],
                                  fill="tozeroy", line=dict(color="#FF5722", width=1.5),
                                  fillcolor="rgba(255,87,34,0.12)"), row=1, col=1)
        fig2.add_trace(go.Bar(x=df["day"], y=df["rainfall"],
                              marker_color="#2196F3", opacity=0.8), row=2, col=1)
        fig2.update_layout(height=380, showlegend=False, template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)

        wt_counts = df["weather_type"].value_counts()
        fig_wt = go.Figure(go.Pie(
            labels=[f"{WEATHER_ICONS.get(w,'🌡️')} {w.replace('_',' ')}" for w in wt_counts.index],
            values=wt_counts.values, hole=0.4,
        ))
        fig_wt.update_layout(title="Weather Type Distribution", height=300)
        st.plotly_chart(fig_wt, use_container_width=True)

    with tab3:
        fig3 = make_subplots(rows=1, cols=2,
                             subplot_titles=["Daily Revenue vs Cost (₹)", "Budget Trend (₹)"])
        fig3.add_trace(go.Bar(x=df["day"], y=df["revenue"], name="Revenue",
                              marker_color="#4CAF50", opacity=0.8), row=1, col=1)
        fig3.add_trace(go.Bar(x=df["day"], y=df["cost"], name="Cost",
                              marker_color="#F44336", opacity=0.8), row=1, col=1)
        fig3.add_trace(go.Scatter(x=df["day"], y=df["budget"],
                                  line=dict(color="#FF9800", width=2),
                                  fill="tozeroy", fillcolor="rgba(255,152,0,0.1)",
                                  name="Budget"), row=1, col=2)
        fig3.update_layout(height=360, barmode="overlay", template="plotly_white")
        st.plotly_chart(fig3, use_container_width=True)

        df["net"] = df["revenue"] - df["cost"]
        fig_pnl = go.Figure(go.Bar(
            x=df["day"], y=df["net"],
            marker_color=["#4CAF50" if v >= 0 else "#F44336" for v in df["net"]],
        ))
        fig_pnl.update_layout(title="Daily Net P&L (₹)", height=260,
                               yaxis_title="₹", template="plotly_white")
        st.plotly_chart(fig_pnl, use_container_width=True)

    with tab4:
        st.markdown("#### 🥧 Revenue Breakdown by Crop")
        crop_rev = st.session_state.crop_revenue
        total_rev = sum(crop_rev.values())
        if total_rev > 0:
            labels = [f"{CROP_ICONS[c]} {c.capitalize()}" for c in crop_rev]
            values = list(crop_rev.values())
            fig_pie = go.Figure(go.Pie(
                labels=labels, values=values, hole=0.45,
                marker=dict(colors=["#4CAF50", "#8BC34A", "#FF5722", "#009688"]),
                textinfo="label+percent+value",
            ))
            fig_pie.update_layout(
                title="Revenue Share by Crop", height=380,
                annotations=[dict(text=f"₹{total_rev:,.0f}", x=0.5, y=0.5,
                                  font_size=16, showarrow=False)]
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No harvests yet — run more steps.")

        st.markdown("#### 🗺️ Yield Heatmap (Expected kg per Slot per Day)")
        slot_hist = st.session_state.slot_yield_history
        if slot_hist and any(len(v) > 0 for v in slot_hist.values()):
            max_days = max(len(v) for v in slot_hist.values())
            if max_days > 0:
                slot_ids = sorted(slot_hist.keys())
                matrix = []
                for sid in slot_ids:
                    row_vals = slot_hist[sid]
                    padded = row_vals + [0.0] * (max_days - len(row_vals))
                    matrix.append(padded)
                fig_hm = go.Figure(go.Heatmap(
                    z=matrix,
                    x=list(range(1, max_days + 1)),
                    y=[f"Slot {sid}" for sid in slot_ids],
                    colorscale="YlGn",
                    colorbar=dict(title="kg"),
                ))
                fig_hm.update_layout(
                    title="Expected Yield per Slot over Episode",
                    xaxis_title="Day", yaxis_title="Slot",
                    height=max(300, 28 * len(slot_ids)),
                    template="plotly_white",
                )
                st.plotly_chart(fig_hm, use_container_width=True)
        else:
            st.info("Run more steps to see heatmap.")

    with tab5:
        st.markdown("#### 🐛 Pest Pressure Spread")
        pest_hist = st.session_state.pest_history
        if pest_hist:
            pest_df = pd.DataFrame(pest_hist)

            # Line chart
            fig_pest = go.Figure()
            for sid in sorted(pest_df["slot_id"].unique()):
                sd = pest_df[pest_df["slot_id"] == sid]
                crop = sd["crop"].iloc[-1] if len(sd) > 0 else "none"
                fig_pest.add_trace(go.Scatter(
                    x=sd["day"], y=sd["pest_pressure"],
                    name=f"{CROP_ICONS.get(crop,'⬜')} Slot {sid}",
                    mode="lines", line=dict(width=1.5),
                ))
            fig_pest.add_hline(y=0.5, line_dash="dash", line_color="red",
                               annotation_text="⚠️ Alert threshold (0.5)")
            fig_pest.update_layout(
                title="Pest Pressure per Slot over Time",
                xaxis_title="Day", yaxis_title="Pest Pressure (0–1)",
                height=380, template="plotly_white",
            )
            st.plotly_chart(fig_pest, use_container_width=True)

            # Heatmap
            st.markdown("**Pest Spread Heatmap**")
            days = sorted(pest_df["day"].unique())
            slots = sorted(pest_df["slot_id"].unique())
            hm = []
            for sid in slots:
                row = []
                for day in days:
                    val = pest_df[(pest_df["slot_id"] == sid) & (pest_df["day"] == day)]["pest_pressure"]
                    row.append(float(val.iloc[0]) if len(val) > 0 else 0.0)
                hm.append(row)
            fig_phm = go.Figure(go.Heatmap(
                z=hm, x=days,
                y=[f"Slot {s}" for s in slots],
                colorscale=[[0,"#ffffff"],[0.3,"#ffe082"],[0.6,"#ff7043"],[1.0,"#b71c1c"]],
                colorbar=dict(title="Pest Level"),
                zmin=0, zmax=1,
            ))
            fig_phm.update_layout(
                title="Pest Spread Heatmap",
                xaxis_title="Day", yaxis_title="Slot",
                height=max(300, 28 * len(slots)),
                template="plotly_white",
            )
            st.plotly_chart(fig_phm, use_container_width=True)
        else:
            st.info("No pest activity yet.")

# ─────────────────────────────────────────────
# Episode Complete
# ─────────────────────────────────────────────
if st.session_state.done:
    grader = get_grader(task)
    grade = grader.grade(env.state())
    st.divider()
    st.subheader("🏆 Final Grade")
    gc1, gc2, gc3, gc4, gc5 = st.columns(5)
    gc1.metric("Overall", f"{grade.composite_score:.3f} / 1.0")
    gc2.metric("Profit", f"{grade.profit_score:.3f}")
    gc3.metric("Yield", f"{grade.yield_score:.3f}")
    gc4.metric("Eco", f"{grade.eco_score:.3f}")
    gc5.metric("Efficiency", f"{grade.efficiency_score:.3f}")

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=grade.composite_score,
        delta={"reference": 0.5, "valueformat": ".3f"},
        number={"suffix": " / 1.0", "valueformat": ".3f"},
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": "#4CAF50"},
            "steps": [
                {"range": [0, 0.4], "color": "#ffcdd2"},
                {"range": [0.4, 0.7], "color": "#fff9c4"},
                {"range": [0.7, 1.0], "color": "#c8e6c9"},
            ],
            "threshold": {"line": {"color": "red", "width": 3}, "value": 0.5},
        },
        title={"text": "Composite Score"},
    ))
    fig_gauge.update_layout(height=280)
    st.plotly_chart(fig_gauge, use_container_width=True)

    with st.expander("📊 Full Grade Details"):
        st.json(grade.to_dict())
