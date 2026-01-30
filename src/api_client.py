"""Setlist.fm API client with rate limiting and error handling."""
import time
import logging
from typing import Optional, Dict, Any, List
import requests
from requests.exceptions import RequestException

from .config import (
    SETLISTFM_API_KEY,
    SETLISTFM_BASE_URL,
    REQUEST_DELAY_SECONDS,
    ITEMS_PER_PAGE
)

logger = logging.getLogger(__name__)


class SetlistFMAPIError(Exception):
    """Custom exception for Setlist.fm API errors."""
    def __init__(self, message: str, status_code: Optional[int] = None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class SetlistFMClient:
    """Client for interacting with the Setlist.fm API."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the client."""
        self.api_key = api_key or SETLISTFM_API_KEY
        if not self.api_key or self.api_key == "your_api_key_here":
            raise ValueError(
                "API key required. Set SETLISTFM_API_KEY in your .env file. "
                "Get your key at: https://www.setlist.fm/settings/api"
            )

        self.base_url = SETLISTFM_BASE_URL
        self.session = requests.Session()
        self.session.headers.update({
            "x-api-key": self.api_key,
            "Accept": "application/json"
        })
        self._last_request_time = 0.0

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY_SECONDS:
            sleep_time = REQUEST_DELAY_SECONDS - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self._last_request_time = time.time()

    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """Make a GET request to the API with retry logic."""
        url = f"{self.base_url}{endpoint}"

        for attempt in range(max_retries):
            self._rate_limit()

            try:
                logger.debug(f"Request: GET {url} params={params}")
                response = self.session.get(url, params=params, timeout=30)

                if response.status_code == 200:
                    return response.json()

                elif response.status_code == 404:
                    logger.warning(f"Resource not found: {url}")
                    return {"setlist": [], "total": 0, "page": 1, "itemsPerPage": ITEMS_PER_PAGE}

                elif response.status_code == 429:
                    wait_time = (2 ** attempt) * 10
                    logger.warning(f"Rate limited. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue

                else:
                    error_msg = f"API error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    if attempt == max_retries - 1:
                        raise SetlistFMAPIError(error_msg, response.status_code)

            except RequestException as e:
                logger.error(f"Request failed: {e}")
                if attempt == max_retries - 1:
                    raise SetlistFMAPIError(f"Request failed after {max_retries} attempts: {e}")
                time.sleep(5)

        raise SetlistFMAPIError("Max retries exceeded")

    def get_artist_setlists(self, mbid: str, page: int = 1) -> Dict[str, Any]:
        """Get setlists for an artist by MusicBrainz ID."""
        endpoint = f"/artist/{mbid}/setlists"
        return self._make_request(endpoint, params={"p": page})

    def get_all_artist_setlists(self, mbid: str) -> List[Dict[str, Any]]:
        """Get ALL setlists for an artist, handling pagination."""
        all_setlists = []
        page = 1

        response = self.get_artist_setlists(mbid, page=1)
        total = response.get("total", 0)
        all_setlists.extend(response.get("setlist", []))

        logger.info(f"Found {total} total setlists for artist")

        total_pages = (total + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE

        for page in range(2, total_pages + 1):
            logger.info(f"Fetching page {page}/{total_pages}")
            response = self.get_artist_setlists(mbid, page=page)
            all_setlists.extend(response.get("setlist", []))

        return all_setlists

    def search_setlists(
        self,
        artist_name: Optional[str] = None,
        artist_mbid: Optional[str] = None,
        venue_name: Optional[str] = None,
        city_name: Optional[str] = None,
        date: Optional[str] = None,
        year: Optional[int] = None,
        page: int = 1
    ) -> Dict[str, Any]:
        """Search for setlists with various filters."""
        params: Dict[str, Any] = {"p": page}

        if artist_name:
            params["artistName"] = artist_name
        if artist_mbid:
            params["artistMbid"] = artist_mbid
        if venue_name:
            params["venueName"] = venue_name
        if city_name:
            params["cityName"] = city_name
        if date:
            params["date"] = date
        if year:
            params["year"] = year

        return self._make_request("/search/setlists", params=params)
