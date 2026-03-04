"""Open-Meteo API client for downloading hourly weather data."""

from datetime import date

import pandas as pd
import requests


class OpenMeteoClient:
    """Client for Open-Meteo weather API.

    Provides geocoding, hourly weather data retrieval, and design weather
    statistics for HVAC planning calculations.

    API documentation: https://open-meteo.com/en/docs
    """

    GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
    ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

    HOURLY_PARAMS = "temperature_2m,relative_humidity_2m"

    def geocode(self, location: str) -> tuple[float, float, str]:
        """Convert location name or coordinate string to coordinates.

        Args:
            location: City name (e.g. "Wien") or "lat,lon" string
                      (e.g. "48.2,16.3").

        Returns:
            Tuple of (latitude, longitude, resolved_name).

        Raises:
            ValueError: If the location cannot be resolved.
            requests.HTTPError: On API errors.
        """
        # Check if location is already "lat,lon"
        if "," in location:
            parts = location.split(",", maxsplit=1)
            try:
                lat = float(parts[0].strip())
                lon = float(parts[1].strip())
                return lat, lon, f"{lat},{lon}"
            except ValueError:
                pass  # Not numeric -- fall through to geocoding

        resp = requests.get(
            self.GEOCODING_URL,
            params={"name": location, "count": 1, "language": "en"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results")
        if not results:
            raise ValueError(f"Location not found: {location!r}")

        hit = results[0]
        return hit["latitude"], hit["longitude"], hit.get("name", location)

    def get_hourly_weather(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch hourly weather data from Open-Meteo.

        Uses the archive API for past dates and the forecast API for
        recent / future dates.  If the requested range spans both, two
        requests are made and the results concatenated.

        Args:
            latitude: Location latitude in degrees.
            longitude: Location longitude in degrees.
            start_date: Start date in "YYYY-MM-DD" format.
            end_date: End date in "YYYY-MM-DD" format.

        Returns:
            DataFrame with a UTC DatetimeIndex and columns:
                - T: air temperature [degC]
                - phi: relative humidity [fraction 0..1]

        Raises:
            requests.HTTPError: On API errors.
            ValueError: On unexpected response format.
        """
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)

        # The archive API covers dates up to roughly 5 days before today.
        # The forecast API covers the recent past (~2 weeks) and up to 16
        # days ahead.  We use a conservative cutoff of 5 days ago.
        archive_cutoff = date.today()

        frames: list[pd.DataFrame] = []

        if start < archive_cutoff:
            archive_end = min(end, archive_cutoff)
            df_archive = self._fetch(
                self.ARCHIVE_URL, latitude, longitude,
                start.isoformat(), archive_end.isoformat(),
            )
            frames.append(df_archive)

        if end >= archive_cutoff:
            forecast_start = max(start, archive_cutoff)
            df_forecast = self._fetch(
                self.FORECAST_URL, latitude, longitude,
                forecast_start.isoformat(), end.isoformat(),
            )
            frames.append(df_forecast)

        if not frames:
            raise ValueError("No data retrieved for the given date range.")

        df = pd.concat(frames)
        df = df[~df.index.duplicated(keep="first")]
        df.sort_index(inplace=True)
        return df

    def get_design_weather(
        self,
        latitude: float,
        longitude: float,
        year: int | None = None,
    ) -> dict:
        """Derive design weather data for a location.

        Downloads a full year of hourly data and computes summer/winter
        design conditions as well as annual statistics.

        Args:
            latitude: Location latitude in degrees.
            longitude: Location longitude in degrees.
            year: Calendar year to analyse.  Defaults to the previous year
                  so that a full dataset is available.

        Returns:
            Dictionary with keys:
                - summer_design: {T, phi} -- 99th-percentile summer conditions
                - winter_design: {T, phi} -- 1st-percentile winter conditions
                - annual_stats:  {T_mean, T_max, T_min, phi_mean,
                                  hours_below_0, hours_above_30}
        """
        if year is None:
            year = date.today().year - 1

        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"

        df = self.get_hourly_weather(latitude, longitude, start_date, end_date)

        # Summer design: 99th percentile temperature hour
        summer_idx = df["T"].quantile(0.99)
        summer_rows = df[df["T"] >= summer_idx]
        summer_design = {
            "T": round(float(summer_rows["T"].mean()), 1),
            "phi": round(float(summer_rows["phi"].mean()), 3),
        }

        # Winter design: 1st percentile temperature hour
        winter_idx = df["T"].quantile(0.01)
        winter_rows = df[df["T"] <= winter_idx]
        winter_design = {
            "T": round(float(winter_rows["T"].mean()), 1),
            "phi": round(float(winter_rows["phi"].mean()), 3),
        }

        annual_stats = {
            "T_mean": round(float(df["T"].mean()), 1),
            "T_max": round(float(df["T"].max()), 1),
            "T_min": round(float(df["T"].min()), 1),
            "phi_mean": round(float(df["phi"].mean()), 3),
            "hours_below_0": int((df["T"] < 0).sum()),
            "hours_above_30": int((df["T"] > 30).sum()),
        }

        return {
            "summer_design": summer_design,
            "winter_design": winter_design,
            "annual_stats": annual_stats,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch(
        self,
        base_url: str,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Execute a single API request and return a normalised DataFrame.

        Args:
            base_url: Either ARCHIVE_URL or FORECAST_URL.
            latitude, longitude: Location coordinates.
            start_date, end_date: ISO date strings.

        Returns:
            DataFrame with DatetimeIndex (UTC) and columns T, phi.
        """
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": self.HOURLY_PARAMS,
            "timezone": "UTC",
        }

        resp = requests.get(base_url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        hourly = data.get("hourly")
        if hourly is None:
            raise ValueError(
                f"Unexpected API response (no 'hourly' key): {data}"
            )

        timestamps = pd.to_datetime(hourly["time"], utc=True)
        df = pd.DataFrame(
            {
                "T": hourly["temperature_2m"],
                "phi": [
                    rh / 100.0 if rh is not None else None
                    for rh in hourly["relative_humidity_2m"]
                ],
            },
            index=timestamps,
        )
        df.index.name = "time"

        # Drop rows where either value is missing
        df.dropna(inplace=True)

        return df
