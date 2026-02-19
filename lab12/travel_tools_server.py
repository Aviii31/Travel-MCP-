"""
TravelTools MCP Server (FastMCP)

Tools:
1) get_weather(city, date): Calls OpenWeatherMap and returns temperature + conditions.
2) search_travel_options(origin, destination, depart_date, return_date): Mock flights/hotels for Tokyo & Udaipur.

Run:
  export OPENWEATHER_API_KEY="..."
  python travel_tools_server.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date as Date
from datetime import datetime, timedelta, timezone
from typing import Any

import requests
from fastmcp import FastMCP


mcp = FastMCP("TravelTools")


OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"


def _parse_yyyy_mm_dd(value: str) -> Date | None:
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d").date()
    except Exception:
        return None


def _fmt_temp_c(temp_c: float) -> str:
    # Avoid overly noisy decimals for UI.
    return f"{temp_c:.1f}°C"


def _safe_get(d: dict[str, Any], path: list[str]) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _owm_request(endpoint: str, params: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    try:
        resp = requests.get(
            f"{OPENWEATHER_BASE_URL}/{endpoint.lstrip('/')}",
            params=params,
            timeout=15,
        )
        if resp.status_code >= 400:
            # OpenWeatherMap errors are often JSON, but not guaranteed.
            return None, f"OpenWeatherMap error {resp.status_code}: {resp.text}"
        return resp.json(), None
    except Exception as e:
        return None, f"Request failed: {e}"


def _weather_from_current(city: str, api_key: str) -> tuple[float | None, str | None, str | None]:
    data, err = _owm_request(
        "weather",
        {"q": city, "appid": api_key, "units": "metric"},
    )
    if err:
        return None, None, err

    temp = _safe_get(data, ["main", "temp"])
    desc = None
    try:
        weather_list = data.get("weather", [])
        if weather_list and isinstance(weather_list[0], dict):
            desc = weather_list[0].get("description")
    except Exception:
        desc = None

    if not isinstance(temp, (int, float)):
        return None, None, "Unexpected OpenWeatherMap response: missing temperature"
    if not isinstance(desc, str) or not desc.strip():
        desc = "Unknown conditions"

    return float(temp), desc, None


def _weather_from_forecast(city: str, api_key: str, target_date: Date) -> tuple[float | None, str | None, str | None]:
    data, err = _owm_request(
        "forecast",
        {"q": city, "appid": api_key, "units": "metric"},
    )
    if err:
        return None, None, err

    tz_offset_seconds = _safe_get(data, ["city", "timezone"])
    if not isinstance(tz_offset_seconds, int):
        tz_offset_seconds = 0
    tz = timezone(timedelta(seconds=tz_offset_seconds))

    entries = data.get("list", [])
    if not isinstance(entries, list) or not entries:
        return None, None, "Unexpected OpenWeatherMap response: missing forecast list"

    # Pick the forecast point that falls on target_date and is closest to 12:00 local time.
    noon = datetime(target_date.year, target_date.month, target_date.day, 12, 0, tzinfo=tz)
    best: dict[str, Any] | None = None
    best_delta: float | None = None

    for item in entries:
        if not isinstance(item, dict):
            continue
        dt_unix = item.get("dt")
        if not isinstance(dt_unix, (int, float)):
            continue
        local_dt = datetime.fromtimestamp(float(dt_unix), tz=timezone.utc).astimezone(tz)
        if local_dt.date() != target_date:
            continue
        delta = abs((local_dt - noon).total_seconds())
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best = item

    if best is None:
        # Fall back to the first available forecast if the requested date isn't in range.
        best = entries[0] if isinstance(entries[0], dict) else None
        if best is None:
            return None, None, "Unexpected OpenWeatherMap response: invalid forecast entry"

    temp = _safe_get(best, ["main", "temp"])
    desc = None
    try:
        weather_list = best.get("weather", [])
        if weather_list and isinstance(weather_list[0], dict):
            desc = weather_list[0].get("description")
    except Exception:
        desc = None

    if not isinstance(temp, (int, float)):
        return None, None, "Unexpected OpenWeatherMap response: missing forecast temperature"
    if not isinstance(desc, str) or not desc.strip():
        desc = "Unknown conditions"

    return float(temp), desc, None


def get_weather_impl(city: str, date: str) -> str:
    """
    Get weather for a city on a given date (YYYY-MM-DD).

    Uses OpenWeatherMap current weather, and uses 5-day/3-hour forecast when the date
    is within the next 5 days.
    
    This is the implementation function that can be called directly.
    """
    api_key = os.environ.get("OPENWEATHER_API_KEY", "").strip()
    if not api_key:
        return (
            "Weather lookup unavailable: missing OPENWEATHER_API_KEY environment variable.\n"
            f"Input: city={city!r}, date={date!r}"
        )

    target = _parse_yyyy_mm_dd(date)
    if target is None:
        return "Invalid date format. Please use YYYY-MM-DD."

    today = datetime.now(timezone.utc).date()
    in_forecast_window = today <= target <= (today + timedelta(days=5))

    if in_forecast_window:
        temp_c, conditions, err = _weather_from_forecast(city, api_key, target)
        source = "forecast"
    else:
        temp_c, conditions, err = _weather_from_current(city, api_key)
        source = "current"

    if err:
        return f"Weather lookup failed for {city} on {target.isoformat()} ({source}): {err}"
    if temp_c is None or conditions is None:
        return f"Weather lookup failed for {city} on {target.isoformat()} ({source}): unexpected response"

    return f"Weather for {city} on {target.isoformat()} ({source}): {_fmt_temp_c(temp_c)}, {conditions}."


@mcp.tool()
def get_weather(city: str, date: str) -> str:
    """
    Get weather for a city on a given date (YYYY-MM-DD).

    Uses OpenWeatherMap current weather, and uses 5-day/3-hour forecast when the date
    is within the next 5 days.
    """
    return get_weather_impl(city, date)


@dataclass(frozen=True)
class _MockFlight:
    airline: str
    price: str  # keep formatted for UI testing
    notes: str | None = None


@dataclass(frozen=True)
class _MockHotel:
    name: str
    rate_per_night: str  # keep formatted for UI testing
    neighborhood: str | None = None


_MOCK_DATA: dict[str, dict[str, Any]] = {
    "tokyo": {
        "flights": [
            _MockFlight(airline="Japan Airlines (JAL)", price="USD 1,120", notes="Round-trip estimate"),
            _MockFlight(airline="All Nippon Airways (ANA)", price="USD 1,080", notes="Round-trip estimate"),
            _MockFlight(airline="Singapore Airlines", price="USD 980", notes="1 stop, round-trip estimate"),
        ],
        "hotels": [
            _MockHotel(name="Shinjuku City Hotel", rate_per_night="USD 165/night", neighborhood="Shinjuku"),
            _MockHotel(name="Asakusa Riverside Inn", rate_per_night="USD 120/night", neighborhood="Asakusa"),
            _MockHotel(name="Ginza Central Stay", rate_per_night="USD 210/night", neighborhood="Ginza"),
        ],
    },
    "udaipur": {
        "flights": [
            _MockFlight(airline="IndiGo", price="INR 9,800", notes="Round-trip estimate"),
            _MockFlight(airline="Air India", price="INR 11,200", notes="Round-trip estimate"),
            _MockFlight(airline="Vistara", price="INR 12,750", notes="Round-trip estimate"),
        ],
        "hotels": [
            _MockHotel(name="Lakeview Heritage Hotel", rate_per_night="INR 5,500/night", neighborhood="Lake Pichola"),
            _MockHotel(name="Old City Haveli Stay", rate_per_night="INR 3,200/night", neighborhood="Old City"),
            _MockHotel(name="Aravalli Resort & Spa", rate_per_night="INR 8,900/night", neighborhood="Outskirts"),
        ],
    },
}


def search_travel_options_impl(origin: str, destination: str, depart_date: str, return_date: str) -> str:
    """
    Flight + hotel search (mock for now).

    Returns mock data for destination in {Tokyo, Udaipur} so you can test UI immediately.
    
    This is the implementation function that can be called directly.
    """
    dest_key = destination.strip().lower()
    data = _MOCK_DATA.get(dest_key)
    if not data:
        return (
            "Mock search only supports destination 'Tokyo' or 'Udaipur' right now.\n"
            f"Input: origin={origin!r}, destination={destination!r}, depart_date={depart_date!r}, return_date={return_date!r}"
        )

    flights: list[_MockFlight] = data["flights"]
    hotels: list[_MockHotel] = data["hotels"]

    lines: list[str] = []
    lines.append("Travel options (MOCK DATA)")
    lines.append(f"Route: {origin} → {destination}")
    lines.append(f"Dates: {depart_date} to {return_date}")
    lines.append("")
    lines.append("Flights:")
    for f in flights:
        extra = f" ({f.notes})" if f.notes else ""
        lines.append(f"- {f.airline}: {f.price}{extra}")
    lines.append("")
    lines.append("Hotels:")
    for h in hotels:
        extra = f" — {h.neighborhood}" if h.neighborhood else ""
        lines.append(f"- {h.name}: {h.rate_per_night}{extra}")

    return "\n".join(lines)


@mcp.tool()
def search_travel_options(origin: str, destination: str, depart_date: str, return_date: str) -> str:
    """
    Flight + hotel search (mock for now).

    Returns mock data for destination in {Tokyo, Udaipur} so you can test UI immediately.
    """
    return search_travel_options_impl(origin, destination, depart_date, return_date)


if __name__ == "__main__":
    # STDIO is the default transport, but we set it explicitly for clarity.
    mcp.run(transport="stdio")

