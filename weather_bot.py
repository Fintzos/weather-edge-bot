"""
Polymarket Weather Trading Bot  v2
====================================
Upgrades από v1:
  ✓ Airport station coordinates  (ΚΡΙΣΙΜΟ — τα markets κλείνουν με αεροδρόμιο, όχι κέντρο πόλης)
  ✓ NWS API για US cities        (ακριβέστερο από OpenMeteo για US)
  ✓ OpenMeteo fallback           (για non-US πόλεις)
  ✓ Kelly Criterion position sizing
  ✓ Expected Value (EV) filtering — δεν ανοίγει trade αν EV < 0
  ✓ Position tracker             (log + JSON για open/closed trades)
  ✓ Daily P&L summary

Setup:
  pip install -r requirements.txt
  cp .env.example .env
  python weather_bot.py
"""

import os, re, time, json, logging
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Optional

import requests
from dotenv import load_dotenv
from colorama import Fore, Style, init

load_dotenv()
init(autoreset=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(), logging.FileHandler("bot.log")],
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
POLYMARKET_PRIVATE_KEY = os.getenv("POLYMARKET_PRIVATE_KEY", "")
POLYMARKET_FUNDER_ADDR = os.getenv("POLYMARKET_FUNDER_ADDRESS", "")
ANTHROPIC_API_KEY      = os.getenv("ANTHROPIC_API_KEY", "")

MIN_EDGE           = float(os.getenv("MIN_EDGE", "0.08"))     # 8% minimum edge
MIN_EV             = float(os.getenv("MIN_EV", "0.03"))       # 3% min expected value
BANKROLL           = float(os.getenv("BANKROLL", "100"))      # total USDC
KELLY_FRACTION     = float(os.getenv("KELLY_FRACTION", "0.15"))  # fractional Kelly (conservative)
MAX_BET_USDC       = float(os.getenv("MAX_BET_USDC", "10"))   # hard cap per trade
MIN_VOLUME_USDC    = float(os.getenv("MIN_VOLUME", "500"))    # skip illiquid markets
DRY_RUN            = os.getenv("DRY_RUN", "true").lower() == "true"
SCAN_INTERVAL_SECS = int(os.getenv("SCAN_INTERVAL", "300"))
POSITIONS_FILE     = "positions.json"

CLOB_BASE  = "https://clob.polymarket.com"
GAMMA_BASE = "https://gamma-api.polymarket.com"

# ── CRITICAL: Airport station coords ─────────────────────────────────────────
# Polymarket weather markets resolve on airport weather stations, NOT city centers.
# Using city-center coords can be 3-8°F off on temp bucket markets → guaranteed losses.
CITIES = {
    # name           : (nws_station, lat,      lon,       region)
    "New York"       : ("KLGA",      40.7773,  -73.8761,  "us"),   # LaGuardia
    "Chicago"        : ("KORD",      41.9742,  -87.9073,  "us"),   # O'Hare
    "Miami"          : ("KMIA",      25.7959,  -80.2870,  "us"),   # Miami Intl
    "Dallas"         : ("KDAL",      32.8481,  -96.8511,  "us"),   # Love Field (NOT DFW)
    "Seattle"        : ("KSEA",      47.4502,  -122.3088, "us"),   # Sea-Tac
    "Atlanta"        : ("KATL",      33.6407,  -84.4277,  "us"),   # Hartsfield
    "Los Angeles"    : ("KLAX",      33.9425,  -118.4081, "us"),   # LAX
    "London"         : (None,        51.4775,  -0.4614,   "intl"), # Heathrow area
    "Tokyo"          : (None,        35.5494,  139.7798,  "intl"), # Haneda area
    "Seoul"          : (None,        37.5509,  126.8050,  "intl"), # Gimpo area
    "Athens"         : (None,        37.9364,  23.9445,   "intl"), # Eleftherios Venizelos
}

# ── Data classes ──────────────────────────────────────────────────────────────
@dataclass
class WeatherForecast:
    city: str
    station: str          # NWS station or "OpenMeteo"
    temp_max_f: float     # Fahrenheit (Polymarket US markets use °F)
    temp_max_c: float
    temp_min_c: float
    precip_prob: float    # 0-100
    condition: str

@dataclass
class Market:
    market_id: str
    question: str
    outcome_yes: str
    outcome_no: str
    yes_price: float
    no_price: float
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
    ev: float             # expected value
    kelly_size: float     # recommended $ size from Kelly
    reasoning: str


# ── Weather: NWS (US cities) ──────────────────────────────────────────────────
def get_nws_forecast(city: str) -> Optional[WeatherForecast]:
    """
    Fetch forecast from NWS api.weather.gov using airport station coordinates.
    Returns temps in both °F and °C.
    """
    station, lat, lon, _ = CITIES[city]
    try:
        # Step 1: get gridpoint from coordinates
        meta = requests.get(
            f"https://api.weather.gov/points/{lat},{lon}",
            headers={"User-Agent": "PolyWeatherBot/2.0"},
            timeout=10
        )
        meta.raise_for_status()
        props = meta.json()["properties"]
        forecast_url = props["forecast"]

        # Step 2: get forecast
        fc = requests.get(
            forecast_url,
            headers={"User-Agent": "PolyWeatherBot/2.0"},
            timeout=10
        )
        fc.raise_for_status()
        periods = fc.json()["properties"]["periods"]

        # Find today's daytime high
        temp_max_f = None
        precip_prob = 0
        condition = "unknown"
        for p in periods[:4]:
            if p.get("isDaytime", False):
                temp_max_f = float(p["temperature"])
                precip_prob = float(p.get("probabilityOfPrecipitation", {}).get("value") or 0)
                condition = _parse_condition(p.get("shortForecast", ""))
                break

        if temp_max_f is None:
            return None

        temp_max_c = (temp_max_f - 32) * 5 / 9
        # Estimate min as ~10°F lower for daytime-only data
        temp_min_c = temp_max_c - 5.5

        return WeatherForecast(
            city=city, station=station,
            temp_max_f=temp_max_f, temp_max_c=temp_max_c,
            temp_min_c=temp_min_c,
            precip_prob=precip_prob, condition=condition,
        )
    except Exception as e:
        log.warning(f"NWS failed for {city} ({station}): {e}")
        return None


# ── Weather: OpenMeteo (international / NWS fallback) ─────────────────────────
def get_openmeteo_forecast(city: str) -> Optional[WeatherForecast]:
    """Fetch from Open-Meteo API. Free, no key. Used for non-US cities."""
    _, lat, lon, _ = CITIES[city]
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_max,temperature_2m_min,"
        f"precipitation_probability_max,weathercode"
        f"&forecast_days=2&timezone=auto"
    )
    try:
        r = requests.get(url, timeout=10)
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
    """Try NWS first for US cities, fallback to OpenMeteo."""
    _, _, _, region = CITIES[city]
    if region == "us":
        fc = get_nws_forecast(city)
        if fc:
            return fc
        log.info(f"  NWS failed for {city}, trying OpenMeteo fallback...")
    return get_openmeteo_forecast(city)


def _parse_condition(short_forecast: str) -> str:
    s = short_forecast.lower()
    if "snow" in s:        return "snow"
    if "thunder" in s:     return "storm"
    if "rain" in s or "shower" in s: return "rain"
    if "cloud" in s or "overcast" in s: return "cloudy"
    return "sunny"


def _parse_wmo_code(code: int) -> str:
    if code <= 1:   return "sunny"
    if code <= 3:   return "partly cloudy"
    if code <= 49:  return "cloudy"
    if code <= 67:  return "rain"
    if code <= 77:  return "snow"
    return "storm"


# ── Polymarket API ────────────────────────────────────────────────────────────
def get_weather_markets() -> list[Market]:
    keywords = ["temperature", "rain", "weather", "snow", "heat",
                "celsius", "fahrenheit", "precipitation", "high temp"]
    markets: list[Market] = []
    try:
        r = requests.get(
            f"{GAMMA_BASE}/markets",
            params={"active": "true", "closed": "false", "limit": 300, "tag_slug": "weather"},
            timeout=15
        )
        r.raise_for_status()
        data = r.json()
        raw = data if isinstance(data, list) else data.get("markets", [])
    except Exception as e:
        log.warning(f"Gamma API error: {e}")
        raw = []

    for m in raw:
        q = m.get("question", "") or m.get("title", "")
        if not any(kw in q.lower() for kw in keywords):
            continue

        vol = float(m.get("volume", 0))
        if vol < MIN_VOLUME_USDC:
            continue

        # Gamma API sometimes returns these as JSON-encoded strings, sometimes as lists
        outcomes       = m.get("outcomes", [])
        outcome_prices = m.get("outcomePrices", [])
        tokens         = m.get("tokens", [])

        if isinstance(outcomes, str):
            try: outcomes = json.loads(outcomes)
            except Exception: outcomes = []
        if isinstance(outcome_prices, str):
            try: outcome_prices = json.loads(outcome_prices)
            except Exception: outcome_prices = []

        if tokens and isinstance(tokens[0], dict):
            yes_price   = float(tokens[0].get("price", 0.5))
            no_price    = float(tokens[1].get("price", 0.5))
            outcome_yes = tokens[0].get("outcome", "Yes")
            outcome_no  = tokens[1].get("outcome", "No")
        elif len(outcomes) >= 2 and len(outcome_prices) >= 2:
            outcome_yes = outcomes[0]
            outcome_no  = outcomes[1]
            yes_price   = float(outcome_prices[0])
            no_price    = float(outcome_prices[1])
        else:
            continue

        markets.append(Market(
            market_id=m.get("id", m.get("condition_id", "")),
            question=q,
            outcome_yes=outcome_yes,
            outcome_no=outcome_no,
            yes_price=yes_price,
            no_price=no_price,
            volume_usdc=vol,
            end_date=m.get("end_date_iso", m.get("endDate", "unknown")),
        ))
    log.info(f"Found {len(markets)} liquid weather markets")
    return markets


# ── Kelly Criterion ───────────────────────────────────────────────────────────
def kelly_size(model_prob: float, market_price: float, bankroll: float) -> float:
    """
    Full Kelly: f* = (p*(b+1) - 1) / b  where b = (1/price - 1)
    We use fractional Kelly (KELLY_FRACTION) to be conservative.
    """
    if market_price <= 0 or market_price >= 1:
        return 0.0
    b = (1.0 / market_price) - 1.0   # net odds
    p = model_prob
    q = 1.0 - p
    full_kelly = (p * (b + 1) - 1) / b
    if full_kelly <= 0:
        return 0.0
    size = full_kelly * KELLY_FRACTION * bankroll
    return min(size, MAX_BET_USDC)


# ── Edge & EV Calculation ─────────────────────────────────────────────────────
def calc_signal(market: Market, forecast: WeatherForecast) -> Optional[TradeSignal]:
    """
    Compare model probability vs market price.
    Returns TradeSignal if edge > MIN_EDGE and EV > MIN_EV.
    """
    q = market.question.lower()
    city_lower = forecast.city.lower()
    city_words = city_lower.split()

    # Match city in question
    if not any(w in q for w in city_words):
        return None

    model_prob = None
    reasoning  = ""

    # ── Temperature bucket (e.g. "Will the high be between 72–73°F?") ─────────
    # Polymarket US markets use °F; we compare against forecast.temp_max_f
    bucket_match = re.search(
        r"between\s+([\d.]+)[–\-–]([\d.]+)\s*°?([fc])", q
    )
    threshold_match = re.search(
        r"(above|exceed|over|at least|below|under)\s+([\d.]+)\s*°?([fc]?)", q
    )

    if bucket_match:
        lo  = float(bucket_match.group(1))
        hi  = float(bucket_match.group(2))
        unit = bucket_match.group(3)
        temp = forecast.temp_max_f if unit == "f" else forecast.temp_max_c
        mid  = (lo + hi) / 2
        spread = hi - lo
        # Gaussian-like probability centered on forecast temp
        diff = abs(temp - mid)
        if diff < spread * 0.5:     model_prob = 0.82
        elif diff < spread * 1.5:   model_prob = 0.45
        elif diff < spread * 3.0:   model_prob = 0.15
        else:                        model_prob = 0.04
        reasoning = (
            f"Forecast {'max_f' if unit=='f' else 'max_c'}="
            f"{temp:.1f}°{'F' if unit=='f' else 'C'}  bucket=[{lo}–{hi}]"
        )

    elif threshold_match:
        direction = threshold_match.group(1)
        threshold = float(threshold_match.group(2))
        unit      = threshold_match.group(3) or "f"
        temp = forecast.temp_max_f if unit == "f" else forecast.temp_max_c
        delta = temp - threshold
        above = direction in ("above", "exceed", "over", "at least")
        if above:
            if delta > 5:    model_prob = 0.90
            elif delta > 2:  model_prob = 0.72
            elif delta > 0:  model_prob = 0.55
            elif delta > -2: model_prob = 0.35
            else:            model_prob = 0.12
        else:  # below / under
            model_prob = 1.0 - (0.90 if delta < -5 else
                                 0.72 if delta < -2 else
                                 0.55 if delta < 0  else
                                 0.35 if delta < 2  else 0.12)
        reasoning = f"Forecast {temp:.1f}° vs threshold {threshold}° ({direction})"

    elif any(w in q for w in ["rain", "precipitation", "wet", "precip"]):
        model_prob = forecast.precip_prob / 100.0
        reasoning  = f"{forecast.station} precip prob: {forecast.precip_prob:.0f}%"

    elif "snow" in q:
        model_prob = 0.82 if "snow" in forecast.condition else 0.04
        reasoning  = f"Condition: {forecast.condition}"

    elif any(w in q for w in ["storm", "thunder"]):
        model_prob = 0.72 if "storm" in forecast.condition else 0.08
        reasoning  = f"Condition: {forecast.condition}"

    if model_prob is None:
        return None

    # ── Determine best side ───────────────────────────────────────────────────
    yes_edge = model_prob - market.yes_price
    no_edge  = (1 - model_prob) - market.no_price
    yes_ev   = yes_edge  # simplified; full EV = edge / price
    no_ev    = no_edge

    if yes_edge >= no_edge and yes_edge >= MIN_EDGE and yes_ev >= MIN_EV:
        side = "YES"
        edge = yes_edge
        ev   = yes_ev
        market_prob = market.yes_price
    elif no_edge >= MIN_EDGE and no_ev >= MIN_EV:
        side = "NO"
        edge = no_edge
        ev   = no_ev
        market_prob = market.no_price
    else:
        return None

    size = kelly_size(model_prob if side == "YES" else 1 - model_prob, market_prob, BANKROLL)
    if size < 0.50:   # skip if Kelly says bet less than 50 cents
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


def save_position(signal: TradeSignal):
    pos = load_positions()
    entry = {
        "ts":         datetime.now(timezone.utc).isoformat(),
        "market":     signal.market.question[:80],
        "market_id":  signal.market.market_id,
        "side":       signal.side,
        "size":       signal.kelly_size,
        "entry_price": signal.market_prob,
        "model_prob": signal.model_prob,
        "edge":       round(signal.edge, 4),
        "ev":         round(signal.ev, 4),
        "expires":    signal.market.end_date[:10],
    }
    pos["open"].append(entry)
    pos["trades"] += 1
    with open(POSITIONS_FILE, "w") as f:
        json.dump(pos, f, indent=2)


def print_positions():
    pos = load_positions()
    print(f"\n{Fore.CYAN}Open positions: {len(pos['open'])}  |  Total trades: {pos['trades']}{Style.RESET_ALL}")
    for p in pos["open"][-5:]:
        print(f"  {p['side']:<4} ${p['size']:.2f}  @{p['entry_price']:.0%}  edge={p['edge']:+.0%}  {p['market'][:55]}")


# ── AI Confirmation (optional) ────────────────────────────────────────────────
def ai_confirm(signal: TradeSignal) -> bool:
    if not ANTHROPIC_API_KEY:
        return True
    prompt = (
        f'Polymarket weather market: "{signal.market.question}"\n'
        f"Market price (YES): {signal.market.yes_price:.0%}\n"
        f"Model probability: {signal.model_prob:.0%}  |  Station: {signal.forecast.station}\n"
        f"Proposed: BUY {signal.side}  edge={signal.edge:+.0%}  kelly=${signal.kelly_size:.2f}\n"
        f"Reasoning: {signal.reasoning}\n\n"
        f'Respond with JSON only: {{"take_trade": true/false, "confidence": 0-100, "reason": "..."}}'
    )
    try:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": "claude-sonnet-4-20250514", "max_tokens": 150,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=20,
        )
        text = r.json()["content"][0]["text"]
        result = json.loads(text.replace("```json","").replace("```","").strip())
        log.info(f"  AI ({result.get('confidence')}%): {result.get('reason','?')}")
        return result.get("take_trade", False) and result.get("confidence", 0) >= 65
    except Exception as e:
        log.warning(f"AI confirm failed: {e} — proceeding")
        return True


# ── Trade Execution ───────────────────────────────────────────────────────────
def execute_trade(signal: TradeSignal) -> bool:
    if DRY_RUN:
        log.info(
            f"{Fore.CYAN}[DRY RUN]{Style.RESET_ALL} BUY {signal.side} "
            f"${signal.kelly_size:.2f}  edge={signal.edge:+.0%}  EV={signal.ev:+.0%}\n"
            f"         {signal.market.question[:65]}"
        )
        save_position(signal)
        return True

    # ── Uncomment for live trading (needs py-clob-client) ─────────────────────
    # from py_clob_client.client import ClobClient
    # from py_clob_client.clob_types import OrderType
    # from py_clob_client.order_builder.constants import BUY
    # client = ClobClient(
    #     host=CLOB_BASE, chain_id=137,
    #     key=POLYMARKET_PRIVATE_KEY, signature_type=1,
    #     funder=POLYMARKET_FUNDER_ADDR,
    # )
    # client.set_api_creds(client.create_or_derive_api_creds())
    # order = client.create_market_order(
    #     token_id=signal.market.market_id, side=BUY, amount=signal.kelly_size
    # )
    # resp = client.post_order(order, OrderType.FOK)
    # if resp.get("success"):
    #     save_position(signal)
    #     return True
    # return False

    log.warning("Live trading not configured. Uncomment py-clob-client block and set DRY_RUN=false.")
    return False


# ── Display ───────────────────────────────────────────────────────────────────
def print_banner():
    mode = f"{Fore.YELLOW}DRY RUN (paper){Style.RESET_ALL}" if DRY_RUN else f"{Fore.RED}⚡ LIVE TRADING{Style.RESET_ALL}"
    print(f"""
{Fore.BLUE}╔══════════════════════════════════════════════════════════╗
║    🌤  Polymarket Weather Bot  v2  🌤                    ║
║    Airport coords · NWS · Kelly Criterion · EV filter    ║
╚══════════════════════════════════════════════════════════╝{Style.RESET_ALL}
  Mode     : {mode}
  Bankroll : ${BANKROLL:.0f}  |  Kelly fraction: {KELLY_FRACTION:.0%}
  Min edge : {MIN_EDGE:.0%}  |  Min EV: {MIN_EV:.0%}  |  Min volume: ${MIN_VOLUME_USDC:.0f}
""")


def print_forecast(f: WeatherForecast):
    icon = {"sunny":"☀","rain":"🌧","snow":"❄","storm":"⛈"}.get(f.condition,"🌥")
    print(
        f"  {icon} {f.city:<13} "
        f"{f.temp_max_f:.0f}°F / {f.temp_max_c:.1f}°C  "
        f"rain={f.precip_prob:.0f}%  "
        f"{f.condition:<12}  [{f.station}]"
    )


def print_signal(s: TradeSignal):
    clr = Fore.GREEN if s.edge > 0.15 else (Fore.YELLOW if s.edge > 0 else Fore.RED)
    print(
        f"\n{clr}  ★ SIGNAL  BUY {s.side}  edge={s.edge:+.0%}  EV={s.ev:+.0%}  "
        f"Kelly=${s.kelly_size:.2f}{Style.RESET_ALL}\n"
        f"    Market  : {s.market.question[:68]}\n"
        f"    Prices  : market={s.market_prob:.0%}  model={s.model_prob:.0%}\n"
        f"    Reason  : {s.reasoning}\n"
        f"    Volume  : ${s.market.volume_usdc:,.0f}   expires {s.market.end_date[:10]}"
    )


# ── Main loop ─────────────────────────────────────────────────────────────────
def run():
    print_banner()
    print_positions()

    cycle = 0
    while True:
        cycle += 1
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        print(f"\n{Fore.YELLOW}── Scan #{cycle}  {now} {'─'*35}{Style.RESET_ALL}")

        # 1. Forecasts
        print(f"\n{Fore.WHITE}[1/3] Fetching forecasts (airport stations):{Style.RESET_ALL}")
        forecasts: dict[str, WeatherForecast] = {}
        for city in CITIES:
            f = get_forecast(city)
            if f:
                forecasts[city] = f
                print_forecast(f)
            time.sleep(0.4)

        # 2. Markets
        print(f"\n{Fore.WHITE}[2/3] Fetching Polymarket weather markets...{Style.RESET_ALL}")
        markets = get_weather_markets()

        # 3. Signals
        print(f"\n{Fore.WHITE}[3/3] Calculating edge + EV + Kelly...{Style.RESET_ALL}")
        signals: list[TradeSignal] = []
        for market in markets:
            for city, forecast in forecasts.items():
                sig = calc_signal(market, forecast)
                if sig:
                    signals.append(sig)

        # deduplicate: keep best signal per market
        seen: dict[str, TradeSignal] = {}
        for sig in signals:
            mid = sig.market.market_id
            if mid not in seen or sig.ev > seen[mid].ev:
                seen[mid] = sig
        signals = sorted(seen.values(), key=lambda s: s.ev, reverse=True)

        if not signals:
            print(f"  No signals above edge={MIN_EDGE:.0%} / EV={MIN_EV:.0%} this scan.")
        else:
            print(f"  {Fore.GREEN}Found {len(signals)} signal(s):{Style.RESET_ALL}")
            for sig in signals:
                print_signal(sig)
                confirmed = ai_confirm(sig)
                if confirmed:
                    execute_trade(sig)
                else:
                    print(f"  {Fore.YELLOW}  AI rejected — skipping.{Style.RESET_ALL}")

        print(f"\n  Next scan in {SCAN_INTERVAL_SECS // 60} min.  Ctrl+C to stop.")
        time.sleep(SCAN_INTERVAL_SECS)


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Bot stopped.{Style.RESET_ALL}")
        print_positions()
