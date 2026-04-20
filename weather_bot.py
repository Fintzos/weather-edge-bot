"""
Polymarket Weather Trading Bot  v2.4
=====================================
Νέο σε v2.4:
  ✓ Live dashboard export → scan_results.json  (διαβάζεται από dashboard.html)
  ✓ Κάθε scan γράφει forecasts + markets + signals + positions στο JSON
  ✓ dashboard.html auto-refresh κάθε 15 δευτερόλεπτα

Setup:
  pip install -r requirements.txt
  cp .env.example .env
  python weather_bot.py
"""

import os, re, time, json, logging
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
from colorama import Fore, Style, init

load_dotenv()
init(autoreset=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-5s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),                          # console: INFO+
        logging.FileHandler("bot.log"),                   # file: DEBUG+
    ],
)
# Console shows INFO and above only — debug spam goes to bot.log
logging.getLogger().handlers[0].setLevel(logging.INFO)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
POLYMARKET_PRIVATE_KEY = os.getenv("POLYMARKET_PRIVATE_KEY", "")
POLYMARKET_FUNDER_ADDR = os.getenv("POLYMARKET_FUNDER_ADDRESS", "")
ANTHROPIC_API_KEY      = os.getenv("ANTHROPIC_API_KEY", "")

MIN_EDGE           = float(os.getenv("MIN_EDGE", "0.08"))
MIN_EV             = float(os.getenv("MIN_EV", "0.03"))
BANKROLL           = float(os.getenv("BANKROLL", "100"))
KELLY_FRACTION     = float(os.getenv("KELLY_FRACTION", "0.15"))
MAX_BET_USDC       = float(os.getenv("MAX_BET_USDC", "10"))
MIN_VOLUME_USDC    = float(os.getenv("MIN_VOLUME", "500"))
DRY_RUN            = os.getenv("DRY_RUN", "true").lower() == "true"
SCAN_INTERVAL_SECS = int(os.getenv("SCAN_INTERVAL", "300"))
AI_GATEKEEPER      = os.getenv("AI_GATEKEEPER", "true").lower() == "true"
MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "5"))
POSITIONS_FILE     = "positions.json"

CLOB_BASE  = "https://clob.polymarket.com"
GAMMA_BASE = "https://gamma-api.polymarket.com"

# ── Shared HTTP session with retries ─────────────────────────────────────────
def _build_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "PolyWeatherBot/2.1"})
    return session

SESSION = _build_session()

# ── Airport station coords ────────────────────────────────────────────────────
# CRITICAL: Polymarket weather markets resolve on airport stations, NOT city centers.
# Using city-center coords can be 3-8 degrees F off on temp bucket markets.
CITIES = {
    # name         : (nws_station, lat,      lon,       region)
    "New York"     : ("KLGA",      40.7773,  -73.8761,  "us"),   # LaGuardia
    "Chicago"      : ("KORD",      41.9742,  -87.9073,  "us"),   # O'Hare
    "Miami"        : ("KMIA",      25.7959,  -80.2870,  "us"),   # Miami Intl
    "Dallas"       : ("KDAL",      32.8481,  -96.8511,  "us"),   # Love Field (NOT DFW)
    "Seattle"      : ("KSEA",      47.4502,  -122.3088, "us"),   # Sea-Tac
    "Atlanta"      : ("KATL",      33.6407,  -84.4277,  "us"),   # Hartsfield
    "Los Angeles"  : ("KLAX",      33.9425,  -118.4081, "us"),   # LAX
    "London"       : (None,        51.4775,  -0.4614,   "intl"), # Heathrow area
    "Tokyo"        : (None,        35.5494,  139.7798,  "intl"), # Haneda area
    "Seoul"        : (None,        37.5509,  126.8050,  "intl"), # Gimpo area
    "Athens"       : (None,        37.9364,  23.9445,   "intl"), # Eleftherios Venizelos
}

# ── Data classes ──────────────────────────────────────────────────────────────
@dataclass
class WeatherForecast:
    city: str
    station: str
    temp_max_f: float
    temp_max_c: float
    temp_min_c: float
    precip_prob: float
    condition: str

@dataclass
class Market:
    market_id: str
    question: str
    outcome_yes: str
    outcome_no: str
    yes_price: float
    no_price: float
    yes_token_id: str    # correct token id for live YES orders
    no_token_id: str     # correct token id for live NO orders
    volume_usdc: float
    end_date: str

@dataclass
class TradeSignal:
    market: Market
    forecast: WeatherForecast
    side: str
    market_prob: float
    model_prob: float
    edge: float
    ev: float
    kelly_size: float
    reasoning: str


# ── Weather: NWS (US cities) ──────────────────────────────────────────────────
def get_nws_forecast(city: str) -> Optional[WeatherForecast]:
    station, lat, lon, _ = CITIES[city]
    try:
        meta = SESSION.get(
            f"https://api.weather.gov/points/{lat},{lon}",
            timeout=15
        )
        meta.raise_for_status()
        forecast_url = meta.json()["properties"]["forecast"]

        fc = SESSION.get(forecast_url, timeout=15)
        fc.raise_for_status()
        periods = fc.json()["properties"]["periods"]

        temp_max_f  = None
        precip_prob = 0
        condition   = "unknown"

        # Use timezone-aware "today" to avoid midnight edge cases
        today_date = datetime.now(timezone.utc).date()

        # First try: today's daytime period
        for p in periods[:6]:
            start = p.get("startTime", "")
            if p.get("isDaytime", False) and start:
                try:
                    period_date = datetime.fromisoformat(start).date()
                except ValueError:
                    period_date = None
                if period_date == today_date:
                    temp_max_f  = float(p["temperature"])
                    precip_prob = float(p.get("probabilityOfPrecipitation", {}).get("value") or 0)
                    condition   = _parse_condition(p.get("shortForecast", ""))
                    break

        # Fallback: first daytime period
        if temp_max_f is None:
            for p in periods[:4]:
                if p.get("isDaytime", False):
                    temp_max_f  = float(p["temperature"])
                    precip_prob = float(p.get("probabilityOfPrecipitation", {}).get("value") or 0)
                    condition   = _parse_condition(p.get("shortForecast", ""))
                    break

        if temp_max_f is None:
            return None

        temp_max_c = (temp_max_f - 32) * 5 / 9
        return WeatherForecast(
            city=city, station=station,
            temp_max_f=temp_max_f, temp_max_c=temp_max_c,
            temp_min_c=temp_max_c - 5.5,
            precip_prob=precip_prob, condition=condition,
        )
    except Exception as e:
        log.warning(f"NWS failed for {city} ({station}): {e}")
        return None


# ── Weather: OpenMeteo (international / NWS fallback) ─────────────────────────
def get_openmeteo_forecast(city: str) -> Optional[WeatherForecast]:
    _, lat, lon, _ = CITIES[city]
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_max,temperature_2m_min,"
        f"precipitation_probability_max,weathercode"
        f"&forecast_days=2&timezone=auto"
    )
    try:
        r = SESSION.get(url, timeout=15)
        r.raise_for_status()
        d = r.json()["daily"]
        max_c = d["temperature_2m_max"][0]
        return WeatherForecast(
            city=city, station="OpenMeteo",
            temp_max_f=max_c * 9/5 + 32,
            temp_max_c=max_c,
            temp_min_c=d["temperature_2m_min"][0],
            precip_prob=d["precipitation_probability_max"][0],
            condition=_parse_wmo_code(d["weathercode"][0]),
        )
    except Exception as e:
        log.warning(f"OpenMeteo failed for {city}: {e}")
        return None


def get_forecast(city: str) -> Optional[WeatherForecast]:
    _, _, _, region = CITIES[city]
    if region == "us":
        fc = get_nws_forecast(city)
        if fc:
            return fc
        log.info(f"  NWS failed for {city}, trying OpenMeteo fallback...")
    return get_openmeteo_forecast(city)


def _parse_condition(s: str) -> str:
    s = s.lower()
    if "snow" in s:                     return "snow"
    if "thunder" in s:                  return "storm"
    if "rain" in s or "shower" in s:    return "rain"
    if "cloud" in s or "overcast" in s: return "cloudy"
    return "sunny"

def _parse_wmo_code(code: int) -> str:
    if code <= 1:  return "sunny"
    if code <= 3:  return "partly cloudy"
    if code <= 49: return "cloudy"
    if code <= 67: return "rain"
    if code <= 77: return "snow"
    return "storm"


# ── Polymarket API ─────────────────────────────────────────────────────────────
def get_weather_markets() -> Optional[list[Market]]:
    """Returns list of markets, or None if ALL fetches failed."""
    keywords = ["temperature", "rain", "weather", "snow", "heat",
                "celsius", "fahrenheit", "precipitation", "high temp",
                "degrees", "new york", "chicago", "miami", "dallas",
                "seattle", "atlanta", "los angeles", "london", "tokyo", "seoul"]

    search_attempts = [
        {"active": "true", "closed": "false", "limit": 500, "tag_slug": "weather"},
        {"active": "true", "closed": "false", "limit": 500, "search": "temperature"},
        {"active": "true", "closed": "false", "limit": 500, "search": "weather"},
        {"active": "true", "closed": "false", "limit": 500, "search": "rain"},
    ]

    seen_ids: set = set()
    raw = []
    any_success = False

    for params in search_attempts:
        try:
            r = SESSION.get(f"{GAMMA_BASE}/markets", params=params, timeout=20)
            r.raise_for_status()
            data  = r.json()
            batch = data if isinstance(data, list) else data.get("markets", [])
            for m in batch:
                mid = m.get("id", m.get("condition_id", ""))
                if mid and mid not in seen_ids:
                    seen_ids.add(mid)
                    raw.append(m)
            any_success = True
            time.sleep(0.3)
        except Exception as e:
            log.warning(f"Gamma API error ({params.get('search', 'tag')}): {e}")

    if not any_success:
        return None   # All requests failed — caller should skip scan

    log.info(f"Fetched {len(raw)} total markets before filtering")
    markets: list[Market] = []

    for m in raw:
        q = m.get("question", "") or m.get("title", "")
        if not any(kw in q.lower() for kw in keywords):
            continue

        vol = float(m.get("volume", 0))
        if vol < MIN_VOLUME_USDC:
            continue

        outcomes       = m.get("outcomes", [])
        outcome_prices = m.get("outcomePrices", [])
        tokens         = m.get("tokens", [])

        if isinstance(outcomes, str):
            try: outcomes = json.loads(outcomes)
            except Exception: outcomes = []
        if isinstance(outcome_prices, str):
            try: outcome_prices = json.loads(outcome_prices)
            except Exception: outcome_prices = []

        yes_token_id = ""
        no_token_id  = ""

        if len(tokens) >= 2 and isinstance(tokens[0], dict) and isinstance(tokens[1], dict):
            yes_price    = float(tokens[0].get("price", 0.5))
            no_price     = float(tokens[1].get("price", 0.5))
            outcome_yes  = tokens[0].get("outcome", "Yes")
            outcome_no   = tokens[1].get("outcome", "No")
            yes_token_id = tokens[0].get("token_id", tokens[0].get("id", ""))
            no_token_id  = tokens[1].get("token_id", tokens[1].get("id", ""))
        elif len(outcomes) >= 2 and len(outcome_prices) >= 2:
            outcome_yes = outcomes[0]
            outcome_no  = outcomes[1]
            yes_price   = float(outcome_prices[0])
            no_price    = float(outcome_prices[1])
            clob_ids    = m.get("clobTokenIds", [])
            if isinstance(clob_ids, str):
                try: clob_ids = json.loads(clob_ids)
                except Exception: clob_ids = []
            if len(clob_ids) >= 2:
                yes_token_id = clob_ids[0]
                no_token_id  = clob_ids[1]
        else:
            continue

        markets.append(Market(
            market_id=m.get("id", m.get("condition_id", "")),
            question=q,
            outcome_yes=outcome_yes,
            outcome_no=outcome_no,
            yes_price=yes_price,
            no_price=no_price,
            yes_token_id=yes_token_id,
            no_token_id=no_token_id,
            volume_usdc=vol,
            end_date=m.get("end_date_iso", m.get("endDate", "unknown")),
        ))

    log.info(f"Found {len(markets)} liquid weather markets")
    return markets


# ── Kelly Criterion ───────────────────────────────────────────────────────────
def kelly_size(model_prob: float, market_price: float) -> float:
    if market_price <= 0 or market_price >= 1:
        return 0.0
    b = (1.0 / market_price) - 1.0
    full_kelly = (model_prob * (b + 1) - 1) / b
    if full_kelly <= 0:
        return 0.0
    return min(full_kelly * KELLY_FRACTION * BANKROLL, MAX_BET_USDC)


# ── Real EV calculation ───────────────────────────────────────────────────────
def expected_value(model_prob: float, market_price: float) -> float:
    """Real EV per $1 staked. Positive = profitable."""
    if market_price <= 0 or market_price >= 1:
        return -1.0
    return model_prob * (1.0 / market_price) - 1.0


# ── Logistic probability helper ──────────────────────────────────────────────
def _logistic(x: float, k: float = 1.0) -> float:
    """Smooth S-curve: maps any real number to (0,1).
    k controls steepness — higher k = sharper transition."""
    import math
    try:
        return 1.0 / (1.0 + math.exp(-k * x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


# ── Signal Calculation ────────────────────────────────────────────────────────
def calc_signal(market: Market, forecast: WeatherForecast) -> Optional[TradeSignal]:
    q          = market.question.lower()
    city_words = forecast.city.lower().split()

    if not any(w in q for w in city_words):
        return None

    # Skip near-resolved markets — almost no EV left
    if market.yes_price < 0.05 or market.yes_price > 0.95:
        return None

    model_prob = None
    reasoning  = ""

    # Temp bucket: "between 72-73F", "between 72 to 73 F", "between 72–73°F"
    bucket_match = re.search(
        r"between\s+([\d.]+)\s*(?:-|–|—|to)\s*([\d.]+)\s*°?\s*([fc])", q
    )
    # Temp threshold: "above 80°F", "exceed 30°C", "below 32"
    threshold_match = re.search(
        r"(above|exceed|over|at least|below|under)\s+([\d.]+)\s*°?\s*([fc]?)", q
    )

    if bucket_match:
        lo   = float(bucket_match.group(1))
        hi   = float(bucket_match.group(2))
        unit = bucket_match.group(3)
        temp = forecast.temp_max_f if unit == "f" else forecast.temp_max_c
        mid    = (lo + hi) / 2
        spread = max(hi - lo, 0.5)          # avoid div-by-zero on tiny buckets
        # Normalised distance from bucket centre, in units of half-spread
        # 0 = dead centre, 1 = edge of bucket, 2 = one full spread outside
        norm_dist = abs(temp - mid) / (spread / 2)
        # Logistic: at centre (0) → ~0.88; at edge (1) → ~0.50; far outside → ~0.10
        model_prob = _logistic(-2.5 * (norm_dist - 0.3), k=1.0)
        model_prob = max(0.04, min(0.94, model_prob))
        reasoning = f"Forecast={temp:.1f} bucket=[{lo}-{hi}] dist={norm_dist:.2f}σ"

    elif threshold_match:
        direction = threshold_match.group(1)
        threshold = float(threshold_match.group(2))
        unit      = threshold_match.group(3) or "f"
        temp  = forecast.temp_max_f if unit == "f" else forecast.temp_max_c
        delta = temp - threshold
        above = direction in ("above", "exceed", "over", "at least")
        # k=0.4 gives a gradual curve: ±5° ≈ 85%/15%, ±2° ≈ 68%/32%, 0° ≈ 50%
        signed_delta = delta if above else -delta
        model_prob   = _logistic(signed_delta, k=0.4)
        model_prob   = max(0.04, min(0.94, model_prob))
        reasoning = f"Forecast={temp:.1f} vs {threshold} ({direction}) Δ={delta:+.1f}"

    elif any(w in q for w in ["rain", "precipitation", "wet", "precip"]):
        # Direct from forecast — already a probability 0-100
        raw = forecast.precip_prob / 100.0
        # Apply mild logistic shaping to avoid extreme 0/100 values from raw data
        model_prob = _logistic((raw - 0.5) * 6, k=1.0)
        model_prob = max(0.05, min(0.93, model_prob))
        reasoning  = f"{forecast.station} precip={forecast.precip_prob:.0f}%"

    elif "snow" in q:
        model_prob = 0.85 if "snow" in forecast.condition else 0.05
        reasoning  = f"Condition: {forecast.condition}"

    elif any(w in q for w in ["storm", "thunder"]):
        model_prob = 0.75 if "storm" in forecast.condition else 0.07
        reasoning  = f"Condition: {forecast.condition}"

    if model_prob is None:
        return None

    yes_edge = model_prob - market.yes_price
    no_edge  = (1 - model_prob) - market.no_price
    yes_ev   = expected_value(model_prob, market.yes_price)
    no_ev    = expected_value(1 - model_prob, market.no_price)

    # Debug log — always emitted so you can see calibration in bot.log
    log.debug(
        f"[EVAL] {forecast.city:<13} | "
        f"model={model_prob:.2f}  "
        f"yes={market.yes_price:.2f} no={market.no_price:.2f}  "
        f"edge_yes={yes_edge:+.2f} ev_yes={yes_ev:+.2f}  "
        f"| {market.question[:55]}"
    )

    if yes_edge >= no_edge and yes_edge >= MIN_EDGE and yes_ev >= MIN_EV:
        side = "YES"
        edge, ev, market_prob = yes_edge, yes_ev, market.yes_price
        prob_for_kelly = model_prob
    elif no_edge >= MIN_EDGE and no_ev >= MIN_EV:
        side = "NO"
        edge, ev, market_prob = no_edge, no_ev, market.no_price
        prob_for_kelly = 1 - model_prob
    else:
        return None

    size = kelly_size(prob_for_kelly, market_prob)
    if size < 0.50:
        return None

    return TradeSignal(
        market=market, forecast=forecast,
        side=side, market_prob=market_prob,
        model_prob=model_prob, edge=edge, ev=ev,
        kelly_size=round(size, 2), reasoning=reasoning,
    )


# ── Position Tracking ─────────────────────────────────────────────────────────
def load_positions() -> dict:
    try:
        with open(POSITIONS_FILE) as f:
            return json.load(f)
    except Exception:
        return {"open": [], "closed": [], "total_pnl": 0.0, "trades": 0}


def has_open_position(market_id: str, side: str) -> bool:
    pos = load_positions()
    return any(
        p["market_id"] == market_id and p["side"] == side
        for p in pos["open"]
    )


def save_position(signal: TradeSignal):
    pos = load_positions()
    token = signal.market.yes_token_id if signal.side == "YES" else signal.market.no_token_id
    pos["open"].append({
        "ts":          datetime.now(timezone.utc).isoformat(),
        "market":      signal.market.question[:80],
        "market_id":   signal.market.market_id,
        "side":        signal.side,
        "token_id":    token,
        "size":        signal.kelly_size,
        "entry_price": signal.market_prob,
        "model_prob":  signal.model_prob,
        "edge":        round(signal.edge, 4),
        "ev":          round(signal.ev, 4),
        "expires":     signal.market.end_date[:10],
    })
    pos["trades"] += 1
    with open(POSITIONS_FILE, "w") as f:
        json.dump(pos, f, indent=2)


def print_positions():
    pos = load_positions()
    print(f"\n{Fore.CYAN}Open positions: {len(pos['open'])}  |  Total trades: {pos['trades']}{Style.RESET_ALL}")
    for p in pos["open"][-5:]:
        print(f"  {p['side']:<4} ${p['size']:.2f}  @{p['entry_price']:.0%}  edge={p['edge']:+.0%}  {p['market'][:55]}")


SCAN_RESULTS_FILE = "scan_results.json"

def save_scan_results(
    cycle: int,
    forecasts: "dict[str, WeatherForecast]",
    markets:   "list[Market] | None",
    signals:   "list[TradeSignal]",
    trades_executed: int,
):
    """Write full scan snapshot to scan_results.json for the live dashboard."""
    pos = load_positions()

    # Build per-city rows the same way the dashboard expects them
    city_rows = []
    market_list = markets or []
    for city, fc in forecasts.items():
        # Find best signal for this city (highest EV)
        city_sigs = [s for s in signals if s.forecast.city == city]
        best = max(city_sigs, key=lambda s: s.ev) if city_sigs else None

        # Total 24h volume for this city across all matching markets
        city_vol = sum(
            m.volume_usdc for m in market_list
            if any(w in m.question.lower() for w in city.lower().split())
        )

        city_rows.append({
            "city":       city,
            "vol_24h":    round(city_vol),
            "forecast_c": round(fc.temp_max_c, 1),
            "forecast_f": round(fc.temp_max_f, 1),
            "precip":     round(fc.precip_prob),
            "condition":  fc.condition,
            "station":    fc.station,
            "has_signal": best is not None,
            "signal": {
                "side":        best.side,
                "edge":        round(best.edge, 3),
                "ev":          round(best.ev, 3),
                "model_prob":  round(best.model_prob, 3),
                "market_prob": round(best.market_prob, 3),
                "kelly":       best.kelly_size,
                "question":    best.market.question[:80],
                "bucket":      best.market.outcome_yes,
                "mkt_vol":     round(best.market.volume_usdc),
                "expires":     best.market.end_date[:10],
                "reasoning":   best.reasoning,
            } if best else None,
        })

    # Sort by 24h vol descending (matches MoonDev layout)
    city_rows.sort(key=lambda r: r["vol_24h"], reverse=True)

    snapshot = {
        "ts":              datetime.now(timezone.utc).isoformat(),
        "cycle":           cycle,
        "dry_run":         DRY_RUN,
        "gamma_ok":        markets is not None,
        "markets_found":   len(market_list),
        "signals_found":   len(signals),
        "trades_executed": trades_executed,
        "open_positions":  len(pos["open"]),
        "total_trades":    pos["trades"],
        "cities":          city_rows,
        "open_positions_detail": pos["open"][-10:],
    }
    with open(SCAN_RESULTS_FILE, "w") as f:
        json.dump(snapshot, f, indent=2)
    log.info(f"[DASHBOARD] scan_results.json updated (cycle #{cycle})")


# ── AI Confirmation (optional) ────────────────────────────────────────────────
def ai_confirm(signal: TradeSignal) -> bool:
    if not ANTHROPIC_API_KEY:
        return True

    prompt = (
        f'Polymarket weather market: "{signal.market.question}"\n'
        f"Market price (YES): {signal.market.yes_price:.0%}\n"
        f"Model probability: {signal.model_prob:.0%}  |  Station: {signal.forecast.station}\n"
        f"Proposed: BUY {signal.side}  edge={signal.edge:+.0%}  EV={signal.ev:+.2f}  kelly=${signal.kelly_size:.2f}\n"
        f"Reasoning: {signal.reasoning}\n\n"
        f'Respond with JSON only: {{"take_trade": true/false, "confidence": 0-100, "reason": "..."}}'
    )
    try:
        r = SESSION.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_API_KEY,
                     "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": "claude-sonnet-4-20250514", "max_tokens": 150,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=25,
        )
        text   = r.json()["content"][0]["text"]
        result = json.loads(text.replace("```json","").replace("```","").strip())
        log.info(f"  AI ({result.get('confidence')}%): {result.get('reason','?')}")
        return result.get("take_trade", False) and result.get("confidence", 0) >= 65
    except Exception as e:
        log.warning(f"AI confirm failed: {e}")
        return not AI_GATEKEEPER  # False if gatekeeper mode, True if optional


# ── Trade Execution ───────────────────────────────────────────────────────────
def execute_trade(signal: TradeSignal) -> bool:
    token_id = signal.market.yes_token_id if signal.side == "YES" else signal.market.no_token_id

    if DRY_RUN:
        log.info(
            f"{Fore.CYAN}[DRY RUN]{Style.RESET_ALL} BUY {signal.side} "
            f"${signal.kelly_size:.2f}  edge={signal.edge:+.0%}  EV={signal.ev:+.2f}\n"
            f"         {signal.market.question[:65]}\n"
            f"         token_id: {token_id or '(not found in API)'}"
        )
        save_position(signal)
        return True

    # ── Live trading: uncomment + pip install py-clob-client ──────────────────
    # from py_clob_client.client import ClobClient
    # from py_clob_client.clob_types import OrderType
    # from py_clob_client.order_builder.constants import BUY
    #
    # if not token_id:
    #     log.warning("No token_id — cannot place live order. Skipping.")
    #     return False
    #
    # client = ClobClient(
    #     host=CLOB_BASE, chain_id=137,
    #     key=POLYMARKET_PRIVATE_KEY, signature_type=1,
    #     funder=POLYMARKET_FUNDER_ADDR,
    # )
    # client.set_api_creds(client.create_or_derive_api_creds())
    # order = client.create_market_order(
    #     token_id=token_id,        # correct YES or NO token id
    #     side=BUY,
    #     amount=signal.kelly_size,
    # )
    # resp = client.post_order(order, OrderType.FOK)
    # if resp.get("success"):
    #     save_position(signal)
    #     return True
    # log.warning(f"Order rejected: {resp}")
    # return False

    log.warning("Live trading not configured. Set DRY_RUN=false and uncomment py-clob-client block.")
    return False


# ── Display ───────────────────────────────────────────────────────────────────
def print_banner():
    mode = f"{Fore.YELLOW}DRY RUN (paper){Style.RESET_ALL}" if DRY_RUN else f"{Fore.RED}LIVE TRADING{Style.RESET_ALL}"
    print(f"""
{Fore.BLUE}========================================================
   Polymarket Weather Bot  v2.3
   Full logging | Logistic model | Position limits
========================================================{Style.RESET_ALL}
  Mode     : {mode}
  Bankroll : ${BANKROLL:.0f}  |  Kelly: {KELLY_FRACTION:.0%}  |  Max bet: ${MAX_BET_USDC:.0f}
  Min edge : {MIN_EDGE:.0%}  |  Min EV: {MIN_EV:.0%}  |  Min vol: ${MIN_VOLUME_USDC:.0f}
  Max open : {MAX_OPEN_POSITIONS} positions
  AI gate  : {"ON" if AI_GATEKEEPER else "OFF (AI optional)"}
  Debug    : EVAL lines → bot.log (not console)
""")


def print_forecast(f: WeatherForecast):
    icon = {"sunny": "sunny", "rain": "rain", "snow": "snow", "storm": "storm"}.get(f.condition, "cloudy")
    print(
        f"  [{icon}] {f.city:<13} "
        f"{f.temp_max_f:.0f}F / {f.temp_max_c:.1f}C  "
        f"rain={f.precip_prob:.0f}%  "
        f"{f.condition:<14}  [{f.station}]"
    )


def print_signal(s: TradeSignal):
    clr  = Fore.GREEN if s.edge > 0.15 else (Fore.YELLOW if s.edge > 0 else Fore.RED)
    tkid = s.market.yes_token_id if s.side == "YES" else s.market.no_token_id
    prob_for_display = s.model_prob if s.side == "YES" else 1 - s.model_prob
    print(
        f"\n{clr}  SIGNAL  BUY {s.side}  "
        f"edge={s.edge:+.0%}  EV={s.ev:+.2f}  Kelly=${s.kelly_size:.2f}{Style.RESET_ALL}\n"
        f"    Market   : {s.market.question[:68]}\n"
        f"    Prices   : market={s.market_prob:.0%}  model={prob_for_display:.0%}\n"
        f"    Reason   : {s.reasoning}\n"
        f"    Volume   : ${s.market.volume_usdc:,.0f}   expires {s.market.end_date[:10]}\n"
        f"    Token ID : {tkid or '(not found)'}"
    )


# ── Main loop ─────────────────────────────────────────────────────────────────
def run():
    print_banner()
    print_positions()

    cycle = 0
    while True:
        cycle += 1
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        print(f"\n{Fore.YELLOW}-- Scan #{cycle}  {now} --{Style.RESET_ALL}")
        log.info(f"=== SCAN #{cycle} START  {now} ===")

        # 1. Forecasts
        print(f"\n{Fore.WHITE}[1/3] Fetching forecasts (airport stations):{Style.RESET_ALL}")
        forecasts: dict[str, WeatherForecast] = {}
        for city in CITIES:
            f = get_forecast(city)
            if f:
                forecasts[city] = f
                print_forecast(f)
                log.info(
                    f"[FORECAST] {f.city:<13} "
                    f"max={f.temp_max_f:.0f}F/{f.temp_max_c:.1f}C  "
                    f"precip={f.precip_prob:.0f}%  {f.condition}  [{f.station}]"
                )
            time.sleep(0.4)
        log.info(f"[FORECAST] fetched {len(forecasts)}/{len(CITIES)} cities")

        # 2. Markets
        print(f"\n{Fore.WHITE}[2/3] Fetching Polymarket weather markets...{Style.RESET_ALL}")
        markets = get_weather_markets()

        if markets is None:
            msg = "All Gamma API requests failed — skipping scan."
            print(f"  {Fore.RED}{msg}{Style.RESET_ALL}")
            log.error(f"[MARKETS] {msg}")
            log.info(f"=== SCAN #{cycle} END  (gamma failure) ===")
            print(f"  Next scan in {SCAN_INTERVAL_SECS // 60} min.  Ctrl+C to stop.")
            time.sleep(SCAN_INTERVAL_SECS)
            continue

        log.info(f"[MARKETS] fetched {len(markets)} liquid weather markets")

        # 3. Signals
        print(f"\n{Fore.WHITE}[3/3] Calculating edge + EV + Kelly...{Style.RESET_ALL}")
        signals: list[TradeSignal] = []
        for market in markets:
            for city, forecast in forecasts.items():
                sig = calc_signal(market, forecast)
                if sig:
                    signals.append(sig)

        # Keep best signal per market by EV
        seen: dict[str, TradeSignal] = {}
        for sig in signals:
            mid = sig.market.market_id
            if mid not in seen or sig.ev > seen[mid].ev:
                seen[mid] = sig
        signals = sorted(seen.values(), key=lambda s: s.ev, reverse=True)

        trades_this_scan = 0

        if not signals:
            msg = f"No signals above edge={MIN_EDGE:.0%} / EV={MIN_EV:.0%} this scan."
            print(f"  {msg}")
            log.info(f"[SIGNALS] {msg}")
        else:
            print(f"  {Fore.GREEN}Found {len(signals)} signal(s):{Style.RESET_ALL}")
            for sig in signals:
                print_signal(sig)
                log.info(
                    f"[SIGNAL] {sig.side} {sig.market.question[:60]}  "
                    f"edge={sig.edge:+.2f}  ev={sig.ev:+.2f}  "
                    f"market={sig.market_prob:.2f}  model={sig.model_prob:.2f}  "
                    f"kelly=${sig.kelly_size}"
                )

                # Position limit
                open_count = len(load_positions()["open"])
                if open_count >= MAX_OPEN_POSITIONS:
                    msg = f"Max open positions ({MAX_OPEN_POSITIONS}) reached — skipping."
                    print(f"  {Fore.YELLOW}  {msg}{Style.RESET_ALL}")
                    log.info(f"[SKIP] {msg}")
                    continue

                if has_open_position(sig.market.market_id, sig.side):
                    msg = "Already open position — skipping."
                    print(f"  {Fore.YELLOW}  {msg}{Style.RESET_ALL}")
                    log.info(f"[SKIP] {msg}")
                    continue

                confirmed = ai_confirm(sig)
                if confirmed:
                    execute_trade(sig)
                    trades_this_scan += 1
                    log.info(f"[TRADE] EXECUTED  {sig.side}  ${sig.kelly_size}  {sig.market.question[:55]}")
                else:
                    print(f"  {Fore.YELLOW}  AI rejected — skipping.{Style.RESET_ALL}")
                    log.info(f"[SKIP] AI rejected  {sig.market.question[:55]}")

            # Scan summary — always logged
            avg_ev   = sum(s.ev   for s in signals) / len(signals)
            avg_edge = sum(s.edge for s in signals) / len(signals)
            summary = (
                f"signals={len(signals)}  "
                f"avg_edge={avg_edge:+.0%}  avg_EV={avg_ev:+.2f}  "
                f"trades_executed={trades_this_scan}"
            )
            print(f"\n  {Fore.CYAN}Scan summary:{Style.RESET_ALL} {summary}")
            log.info(f"[SUMMARY] {summary}")

        log.info(f"=== SCAN #{cycle} END ===")
        save_scan_results(cycle, forecasts, markets, signals, trades_this_scan)
        print(f"\n  Next scan in {SCAN_INTERVAL_SECS // 60} min.  Ctrl+C to stop.")
        time.sleep(SCAN_INTERVAL_SECS)


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Bot stopped.{Style.RESET_ALL}")
        print_positions()
