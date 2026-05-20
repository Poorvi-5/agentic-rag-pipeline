# src/tools/api_tool.py
"""
Tool 4: External REST API Tool
Fetches live data from external APIs.
We use Open-Meteo (free weather API, no key needed)
and wttr.in as backup.
"""

import requests
import json
from langchain_core.tools import Tool


# ── City coordinates lookup ───────────────────────────────────
CITY_COORDS = {
    "mumbai":    (19.0760,  72.8777),
    "delhi":     (28.6139,  77.2090),
    "bangalore": (12.9716,  77.5946),
    "hyderabad": (17.3850,  78.4867),
    "chennai":   (13.0827,  80.2707),
    "pune":      (18.5204,  73.8567),
    "kolkata":   (22.5726,  88.3639),
    "bhopal":    (23.2599,  77.4126),
    "london":    (51.5074,  -0.1278),
    "new york":  (40.7128, -74.0060),
    "tokyo":     (35.6762, 139.6503),
    "paris":     (48.8566,   2.3522),
    "dubai":     (25.2048,  55.2708),
}

WMO_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy",
    3: "Overcast", 45: "Foggy", 48: "Icy fog",
    51: "Light drizzle", 61: "Slight rain", 63: "Moderate rain",
    65: "Heavy rain", 71: "Slight snow", 80: "Rain showers",
    95: "Thunderstorm"
}


def get_weather(query: str) -> str:
    """
    Fetch real-time weather for a city using Open-Meteo API.
    Free, no API key required.

    Args:
        query: City name or weather question like
               "weather in Mumbai" or "Mumbai"
    Returns:
        Formatted weather report string
    """
    # Extract city name from query
    query_lower = query.lower()
    city_name   = None
    coords      = None

    # Try to match a known city
    for city, coord in CITY_COORDS.items():
        if city in query_lower:
            city_name = city.title()
            coords    = coord
            break

    # Default to Bhopal if no city found
    if not coords:
        city_name = "Bhopal"
        coords    = CITY_COORDS["bhopal"]
        note      = f" (City not recognized, defaulting to {city_name})"
    else:
        note = ""

    lat, lon = coords

    try:
        # Open-Meteo free weather API
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,"
            f"wind_speed_10m,weathercode,apparent_temperature"
            f"&daily=temperature_2m_max,temperature_2m_min"
            f"&timezone=auto&forecast_days=3"
        )

        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data     = response.json()

        current  = data["current"]
        daily    = data["daily"]

        temp     = current["temperature_2m"]
        feels    = current["apparent_temperature"]
        humidity = current["relative_humidity_2m"]
        wind     = current["wind_speed_10m"]
        wcode    = current["weathercode"]
        condition = WMO_CODES.get(wcode, "Unknown")

        # 3-day forecast
        forecast_lines = []
        for i in range(min(3, len(daily["time"]))):
            forecast_lines.append(
                f"  {daily['time'][i]}: "
                f"High {daily['temperature_2m_max'][i]}°C / "
                f"Low {daily['temperature_2m_min'][i]}°C"
            )

        forecast = "\n".join(forecast_lines)

        return (
            f"Weather Report for {city_name}{note}\n"
            f"{'─' * 35}\n"
            f"Condition    : {condition}\n"
            f"Temperature  : {temp}°C (feels like {feels}°C)\n"
            f"Humidity     : {humidity}%\n"
            f"Wind Speed   : {wind} km/h\n\n"
            f"3-Day Forecast:\n{forecast}"
        )

    except requests.exceptions.Timeout:
        return "Weather API timed out. Please try again."
    except Exception as e:
        return f"Weather fetch failed: {str(e)}"


def get_joke() -> str:
    """Fetch a random programming joke from a free API."""
    try:
        resp = requests.get(
            "https://official-joke-api.appspot.com/jokes/programming/random",
            timeout=5
        )
        jokes = resp.json()
        if jokes:
            j = jokes[0]
            return f"Q: {j['setup']}\nA: {j['punchline']}"
        return "Could not fetch a joke right now."
    except Exception as e:
        return f"Joke API failed: {str(e)}"


def call_api(query: str) -> str:
    """
    Smart API dispatcher — routes to the right API
    based on keywords in the query.

    Args:
        query: Natural language request
    Returns:
        API response as formatted string
    """
    query_lower = query.lower()

    if any(kw in query_lower for kw in
           ["weather", "temperature", "forecast",
            "rain", "humidity", "wind"]):
        return get_weather(query)

    elif any(kw in query_lower for kw in
             ["joke", "funny", "humor"]):
        return get_joke()

    else:
        # Default: try weather for any city-like query
        return get_weather(query)


# ── Wrap as LangChain Tool ────────────────────────────────────
api_tool = Tool(
    name="external_api",
    func=call_api,
    description=(
        "Call external REST APIs to get live data. "
        "Use this tool for: real-time weather information for any city, "
        "temperature, humidity, wind speed, or 3-day forecasts. "
        "Also use for fun facts or jokes. "
        "Input should describe what you need, e.g. "
        "'weather in Mumbai' or 'temperature in Delhi today'."
    )
)


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n--- Weather Test ---")
    print(get_weather("weather in Mumbai"))

    print("\n--- Joke Test ---")
    print(get_joke())