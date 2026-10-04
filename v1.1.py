#!/usr/bin/env python3
"""
Import owned games from Backlogia and itch.io into VideoGameTrackarr.

Examples:

    python library_importer.py run \
        --backlogia-url http://localhost:8000 \
        --trackarr-url http://localhost:9000 \
        --backlogia-username alice \
        --backlogia-password secret \
        --trackarr-username alice \
        --trackarr-password secret \
        --itch-api-key "$ITCH_API_KEY"

    python library_importer.py run --config importer.json --dry-run

    python library_importer.py test

The API paths are configurable because installations may expose slightly
different route names. The defaults are listed in DEFAULT_ENDPOINTS below.
"""

from __future__ import annotations

import argparse
import http.cookiejar
import json
import logging
import os
import re
import sys
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
import uuid

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_ENDPOINTS = {
    # Backlogia
    "backlogia_login": "/auth/login",
    "backlogia_steam_sync": "/sync/steam",
    "backlogia_epic_sync": "/sync/epic",
    "backlogia_job": "/sync/jobs/{job_id}",
    "backlogia_games": "/games",

    # VideoGameTrackarr
    "trackarr_login": "/auth/login",
    "trackarr_games": "/games",
    "trackarr_library": "/library",

    # itch.io
    "itch_profile": "/profile",
    "itch_owned_keys": "/profile/owned-keys",
    "itch_bundles": "/profile/bundles",
    "itch_bundle_games": "/bundles/{bundle_id}/games",
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ImporterError(RuntimeError):
    """Base exception for importer failures."""


class APIError(ImporterError):
    """Raised when an API request cannot be completed."""


class AuthenticationError(APIError):
    """Raised when authentication fails."""


class JobFailed(ImporterError):
    """Raised when an asynchronous synchronization job fails."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class GameRecord:
    """
    Common internal representation used by all sources.

    A single source game can produce multiple records when it supports
    multiple platforms.
    """

    store: str
    source_game_id: str
    title: str
    platform: str
    digital: bool = True
    ownership: str = "owned"

    igdb_id: int | None = None
    source_url: str | None = None
    developer: str | None = None
    publisher: str | None = None

    from_normal_itch_library: bool = False
    bundles: list[dict[str, Any]] = field(default_factory=list)

    trackarr_game_id: str | None = None
    match_confidence: str | None = None

    def deduplication_key(self) -> tuple[str, str, str]:
        """
        Store is intentionally included.

        A Steam game and an itch.io game may have the same source ID in
        unusual cases, but they must never be merged.
        """
        return self.store, self.source_game_id, self.platform

    def library_key(self) -> tuple[str, str, str, str]:
        """
        Key used to avoid duplicate Trackarr library entries.
        """
        return (
            self.store,
            self.source_game_id,
            self.platform,
            str(self.trackarr_game_id),
        )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def first_value(data: dict[str, Any], *names: str, default: Any = None) -> Any:
    """Return the first non-null field found in a dictionary."""
    for name in names:
        if data.get(name) is not None:
            return data[name]
    return default


def as_list(value: Any) -> list[Any]:
    """Normalize common API list wrappers."""
    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, dict):
        for key in ("items", "results", "games", "data", "owned_keys", "bundles"):
            if isinstance(value.get(key), list):
                return value[key]

    return []


def normalize_title(title: str) -> str:
    """Normalize titles for conservative matching."""
    title = title.casefold()
    title = re.sub(r"[^\w\s]", " ", title)
    title = re.sub(r"\s+", " ", title)
    return title.strip()


def extract_platforms(game: dict[str, Any]) -> list[str]:
    """
    Extract platform names from several common API representations.

    Unknown platforms are retained as lowercase strings rather than silently
    discarded. This makes the importer forward-compatible.
    """
    raw = first_value(
        game,
        "platforms",
        "supported_platforms",
        "builds",
        "available_platforms",
        default=[],
    )

    if isinstance(raw, dict):
        raw = list(raw.keys())

    if not isinstance(raw, list):
        raw = [raw]

    result = []

    for item in raw:
        if isinstance(item, dict):
            value = first_value(item, "platform", "name", "slug", "os")
        else:
            value = item

        if value is None:
            continue

        value = str(value).casefold()

        if value in {"win", "windows", "pc", "microsoft windows"}:
            value = "Windows"
        elif value in {"mac", "macos", "osx", "apple"}:
            value = "macOS"
        elif value in {"linux", "gnu/linux"}:
            value = "Linux"
        elif value in {"android"}:
            value = "Android"
        elif value in {"ios", "iphone", "ipad"}:
            value = "iOS"
        elif value in {"web", "browser", "html5"}:
            value = "Web"
        else:
            value = value.title()

        if value not in result:
            result.append(value)

    return result or ["Unknown"]


def load_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json_file(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)

    temporary.replace(path)


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------

class HTTPClient:
    """
    Small standard-library HTTP client.

    It waits at least min_interval between requests and waits 60 seconds
    after rate-limit or server failures, as requested.
    """

    def __init__(
        self,
        base_url: str,
        *,
        opener: Any = None,
        min_interval: float = 1.0,
        error_wait: float = 60.0,
        timeout: float = 60.0,
        logger: logging.Logger | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.min_interval = min_interval
        self.error_wait = error_wait
        self.timeout = timeout
        self.last_request = 0.0
        self.logger = logger or logging.getLogger(__name__)

        if opener is None:
            cookie_jar = http.cookiejar.CookieJar()
            opener = urllib.request.build_opener(
                urllib.request.HTTPCookieProcessor(cookie_jar)
            )

        self.opener = opener

    def request_json(
        self,
        method: str,
        path: str,
        *,
        headers: dict[str, str] | None = None,
        query: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        retries: int = 3,
        expected: tuple[int, ...] = (200, 201, 202, 204),
    ) -> Any:
        """Perform a JSON request with rate limiting and retries."""

        url = self.base_url + "/" + path.lstrip("/")

        if query:
            encoded_query = urllib.parse.urlencode(
                [(key, value) for key, value in query.items() if value is not None],
                doseq=True,
            )
            url += "?" + encoded_query

        request_headers = {
            "Accept": "application/json",
            "User-Agent": "library-importer/1.0",
        }

        if headers:
            request_headers.update(headers)

        encoded_body = None

        if body is not None:
            encoded_body = json.dumps(body).encode("utf-8")
            request_headers.setdefault("Content-Type", "application/json")

        for attempt in range(retries + 1):
            elapsed = time.monotonic() - self.last_request

            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)

            request = urllib.request.Request(
                url,
                data=encoded_body,
                headers=request_headers,
                method=method.upper(),
            )

            try:
                self.logger.debug("%s %s", method.upper(), url)
                response = self.opener.open(request, timeout=self.timeout)
                self.last_request = time.monotonic()

                status = getattr(response, "status", 200)
                raw = response.read()

                if status not in expected:
                    raise APIError(f"{method} {url} returned HTTP {status}")

                if not raw:
                    return {}

                return json.loads(raw.decode("utf-8"))

            except urllib.error.HTTPError as exc:
                self.last_request = time.monotonic()
                body_text = exc.read().decode("utf-8", errors="replace")

                retryable = exc.code == 429 or exc.code >= 500

                self.logger.error(
                    "HTTP error %s for %s %s: %s",
                    exc.code,
                    method.upper(),
                    url,
                    body_text[:500],
                )

                if not retryable or attempt >= retries:
                    raise APIError(
                        f"{method} {url} failed with HTTP {exc.code}: "
                        f"{body_text[:500]}"
                    ) from exc

                retry_after = exc.headers.get("Retry-After")

                if retry_after:
                    try:
                        time.sleep(float(retry_after))
                        continue
                    except ValueError:
                        pass

                time.sleep(self.error_wait)

            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                self.last_request = time.monotonic()

                self.logger.error(
                    "Request error for %s %s: %s",
                    method.upper(),
                    url,
                    exc,
                )

                if attempt >= retries:
                    raise APIError(f"{method} {url} failed: {exc}") from exc

                time.sleep(self.error_wait)

        raise APIError(f"Request failed: {method} {url}")


# ---------------------------------------------------------------------------
# Main importer
# ---------------------------------------------------------------------------

class LibraryImporter:
    """
    Sequential importer.

    The methods intentionally remain self-contained so each source can be
    tested independently while run() controls the overall data flow.
    """

    def __init__(
        self,
        backlogia_url: str,
        trackarr_url: str,
        *,
        itch_api_key: str | None = None,
        endpoints: dict[str, str] | None = None,
        state_dir: str = ".library-importer",
        min_interval: float = 1.0,
        error_wait: float = 60.0,
        poll_interval: float = 5.0,
        poll_timeout: float = 3600.0,
        dry_run: bool = False,
        logger: logging.Logger | None = None,
    ):
        self.logger = logger or logging.getLogger("library-importer")
        self.endpoints = dict(DEFAULT_ENDPOINTS)
        self.endpoints.update(endpoints or {})

        self.backlogia = HTTPClient(
            backlogia_url,
            min_interval=min_interval,
            error_wait=error_wait,
            logger=self.logger,
        )

        self.trackarr = HTTPClient(
            trackarr_url,
            min_interval=min_interval,
            error_wait=error_wait,
            logger=self.logger,
        )

        self.itch = HTTPClient(
            "https://api.itch.io",
            min_interval=min_interval,
            error_wait=error_wait,
            logger=self.logger,
        )

        self.itch_api_key = itch_api_key
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout
        self.dry_run = dry_run

        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.backlogia_games: list[dict[str, Any]] = []
        self.itch_games: list[dict[str, Any]] = []
        self.normalized_games: list[GameRecord] = []
        self.resolved_games: list[GameRecord] = []

        self.trackarr_token: str | None = None
        self.igdb_cache: dict[str, int] = {}
        self.trackarr_game_cache: dict[str, str] = {}

    # -----------------------------------------------------------------------
    # Persistence and logging
    # -----------------------------------------------------------------------

    def log_event(self, event: str, **data: Any) -> None:
        record = {
            "timestamp": time.time(),
            "event": event,
            **data,
        }

        with (self.state_dir / "events.jsonl").open(
            "a",
            encoding="utf-8",
        ) as file:
            file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def save_checkpoint(self, name: str = "checkpoint.json") -> None:
        state = {
            "backlogia_games": self.backlogia_games,
            "itch_games": self.itch_games,
            "normalized_games": [asdict(game) for game in self.normalized_games],
            "resolved_games": [asdict(game) for game in self.resolved_games],
            "igdb_cache": self.igdb_cache,
            "trackarr_game_cache": self.trackarr_game_cache,
        }

        path = self.state_dir / name
        save_json_file(path, state)

        print("saved checkpoint:", path.resolve())
        print("saved normalized games:", len(state["normalized_games"]))


    def load_checkpoint(self, name: str = "checkpoint.json") -> None:
        path = self.state_dir / name

        print("loading checkpoint:", path.resolve())

        if not path.exists():
            print("checkpoint does not exist")
            return

        state = load_json_file(path)

        print("checkpoint normalized games:",
            len(state.get("normalized_games", [])))

        self.backlogia_games = state.get("backlogia_games", [])
        self.itch_games = state.get("itch_games", [])

        self.normalized_games = [
            GameRecord(**item)
            for item in state.get("normalized_games", [])
        ]

        self.resolved_games = [
            GameRecord(**item)
            for item in state.get("resolved_games", [])
        ]

        self.igdb_cache = state.get("igdb_cache", {})
        self.trackarr_game_cache = state.get("trackarr_game_cache", {})



    # -----------------------------------------------------------------------
    # Backlogia authentication and synchronization
    # -----------------------------------------------------------------------

    def login_to_backlogia(self, username: str, password: str) -> None:
        response = self.backlogia.request_json(
            "POST",
            self.endpoints["backlogia_login"],
            body={"username": username, "password": password},
        )

        if isinstance(response, dict) and response.get("token"):
            self.backlogia_auth_headers = {
                "Authorization": f"Bearer {response['token']}"
            }

        self.log_event("backlogia_login_success")

    def _backlogia_headers(self) -> dict[str, str]:
        return getattr(self, "backlogia_auth_headers", {})

    def _run_backlogia_sync(self, endpoint_key: str) -> dict[str, Any]:
        response = self.backlogia.request_json(
            "POST",
            self.endpoints[endpoint_key],
            headers=self._backlogia_headers(),
        )

        job_id = first_value(response, "job_id", "jobId", "id")

        if not job_id:
            return response

        started = time.monotonic()

        while True:
            if time.monotonic() - started > self.poll_timeout:
                raise JobFailed(f"Backlogia job timed out: {job_id}")

            status = self.backlogia.request_json(
                "GET",
                self.endpoints["backlogia_job"].format(job_id=job_id),
                headers=self._backlogia_headers(),
            )

            state = str(
                first_value(status, "status", "state", default="")
            ).casefold()

            if state in {"complete", "completed", "success", "succeeded", "done"}:
                self.log_event("backlogia_sync_success", job_id=job_id)
                return status

            if state in {"failed", "failure", "cancelled", "canceled", "error"}:
                raise JobFailed(f"Backlogia job failed: {job_id}: {status}")

            time.sleep(self.poll_interval)

    def sync_backlogia_steam(self) -> dict[str, Any]:
        return self._run_backlogia_sync("backlogia_steam_sync")

    def sync_backlogia_epic(self) -> dict[str, Any]:
        return self._run_backlogia_sync("backlogia_epic_sync")

    def fetch_backlogia_games(self) -> list[dict[str, Any]]:
        response = self.backlogia.request_json(
            "GET",
            self.endpoints["backlogia_games"],
            headers=self._backlogia_headers(),
        )

        games = as_list(response)
        self.backlogia_games.extend(games)

        self.log_event("backlogia_games_fetched", count=len(games))
        return games

    # -----------------------------------------------------------------------
    # itch.io
    # -----------------------------------------------------------------------

    def _itch_headers(self) -> dict[str, str]:
        if not self.itch_api_key:
            raise AuthenticationError(
                "An itch.io API key is required. Use --itch-api-key."
            )

        return {
            "Authorization": f"Bearer {self.itch_api_key}",
        }

    def login_to_itch(self, api_key: str | None = None) -> None:
        if api_key:
            self.itch_api_key = api_key

        self.itch.request_json(
            "GET",
            self.endpoints["itch_profile"],
            headers=self._itch_headers(),
        )

        self.log_event("itch_login_success")

    def _fetch_paginated(
        self,
        client: HTTPClient,
        endpoint: str,
        *,
        headers: dict[str, str] | None = None,
        item_keys: tuple[str, ...] = ("items", "results", "data"),
        extra_query: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        results = []
        page = 1

        while True:
            query = {
                "page": page,
                "per_page": 500,
            }

            if extra_query:
                query.update(extra_query)

            response = client.request_json(
                "GET",
                endpoint,
                headers=headers,
                query=query,
            )

            items: list[Any] = []

            if isinstance(response, list):
                items = response
            elif isinstance(response, dict):
                for key in item_keys:
                    if isinstance(response.get(key), list):
                        items = response[key]
                        break

            if not items:
                break

            results.extend(
                item for item in items if isinstance(item, dict)
            )
            page += 1

        return results

    def fetch_itch_owned_games(self) -> list[dict[str, Any]]:
        games = self._fetch_paginated(
            self.itch,
            self.endpoints["itch_owned_keys"],
            headers=self._itch_headers(),
            item_keys=("owned_keys", "games", "items", "results"),
        )

        for game in games:
            game["_from_normal_library"] = True

        self.itch_games.extend(games)
        self.log_event("itch_owned_games_fetched", count=len(games))
        return games

    def fetch_itch_owned_bundles(self) -> list[dict[str, Any]]:
        return self._fetch_paginated(
            self.itch,
            self.endpoints["itch_bundles"],
            headers=self._itch_headers(),
            item_keys=("bundles", "items", "results", "data"),
        )

    def fetch_games_from_itch_bundle(
        self,
        bundle: dict[str, Any],
    ) -> list[dict[str, Any]]:
        bundle_id = first_value(bundle, "id", "bundle_id", "bundleId")

        if bundle_id is None:
            return []

        endpoint = self.endpoints["itch_bundle_games"].format(
            bundle_id=urllib.parse.quote(str(bundle_id), safe="")
        )

        games = self._fetch_paginated(
            self.itch,
            endpoint,
            headers=self._itch_headers(),
            item_keys=("games", "items", "results", "data"),
        )

        for game in games:
            game["_bundle_id"] = bundle_id
            game["_bundle_name"] = first_value(
                bundle,
                "name",
                "title",
                default=str(bundle_id),
            )

        return games

    def fetch_all_itch_bundle_games(self) -> list[dict[str, Any]]:
        all_games = []

        for bundle in self.fetch_itch_owned_bundles():
            games = self.fetch_games_from_itch_bundle(bundle)
            all_games.extend(games)

        self.itch_games.extend(all_games)
        self.log_event("itch_bundle_games_fetched", count=len(all_games))
        return all_games

    # -----------------------------------------------------------------------
    # Normalization
    # -----------------------------------------------------------------------

    def normalize_steam_game(
        self,
        game: dict[str, Any],
    ) -> list[GameRecord]:
        source_id = first_value(game, "steam_id", "steamId", "appid", "id")
        title = first_value(game, "title", "name", default=str(source_id))

        records = []

        for platform in extract_platforms(game):
            records.append(
                GameRecord(
                    store="Steam",
                    source_game_id=str(source_id),
                    title=str(title),
                    platform=platform,
                    igdb_id=first_value(game, "igdb_id", "igdbId"),
                    source_url=game.get("url"),
                    developer=game.get("developer"),
                    publisher=game.get("publisher"),
                )
            )

        return records

    def normalize_epic_game(
        self,
        game: dict[str, Any],
    ) -> list[GameRecord]:
        source_id = first_value(game, "epic_id", "epicId", "id")
        title = first_value(game, "title", "name", default=str(source_id))

        return [
            GameRecord(
                store="Epic Games Store",
                source_game_id=str(source_id),
                title=str(title),
                platform="Windows",
                igdb_id=first_value(game, "igdb_id", "igdbId"),
                source_url=game.get("url"),
                developer=game.get("developer"),
                publisher=game.get("publisher"),
            )
        ]

    def normalize_itch_owned_game(
        self,
        game: dict[str, Any],
    ) -> list[GameRecord]:
        source_id = first_value(
            game,
            "itch_id",
            "itchId",
            "game_id",
            "gameId",
            "id",
        )
        title = first_value(game, "title", "name", default=str(source_id))

        return [
            GameRecord(
                store="itch.io",
                source_game_id=str(source_id),
                title=str(title),
                platform=platform,
                igdb_id=first_value(game, "igdb_id", "igdbId"),
                source_url=game.get("url"),
                developer=game.get("developer"),
                publisher=game.get("publisher"),
                from_normal_itch_library=True,
            )
            for platform in extract_platforms(game)
        ]

    def normalize_itch_bundle_game(
        self,
        game: dict[str, Any],
    ) -> list[GameRecord]:
        records = self.normalize_itch_owned_game(game)
        bundle_id = game.get("_bundle_id")
        bundle_name = game.get("_bundle_name")

        for record in records:
            record.from_normal_itch_library = False
            record.bundles.append({
                "id": bundle_id,
                "name": bundle_name,
            })

        return records

    def deduplicate_itch_games(
        self,
        records: list[GameRecord] | None = None,
    ) -> list[GameRecord]:
        if records is None:
            records = []

            for game in self.itch_games:
                if game.get("_bundle_id") is not None:
                    records.extend(self.normalize_itch_bundle_game(game))
                else:
                    records.extend(self.normalize_itch_owned_game(game))

        merged: dict[tuple[str, str, str], GameRecord] = {}

        for record in records:
            key = record.deduplication_key()

            if key not in merged:
                merged[key] = record
                continue

            existing = merged[key]
            existing.from_normal_itch_library |= (
                record.from_normal_itch_library
            )

            if not existing.igdb_id:
                existing.igdb_id = record.igdb_id

            if not existing.source_url:
                existing.source_url = record.source_url

            known_bundle_ids = {
                bundle.get("id")
                for bundle in existing.bundles
            }

            for bundle in record.bundles:
                if bundle.get("id") not in known_bundle_ids:
                    existing.bundles.append(bundle)

        return list(merged.values())

    def normalize_all_games(self) -> list[GameRecord]:
        normalized = []

        for game in self.backlogia_games:
            store = str(
                first_value(game, "store", "source", "platform_store", default="")
            ).casefold()

            if store in {"steam", "steamworks"}:
                normalized.extend(self.normalize_steam_game(game))
            elif store in {"epic", "epic games", "epic games store"}:
                normalized.extend(self.normalize_epic_game(game))

        itch_records = self.deduplicate_itch_games()

        normalized.extend(itch_records)
        self.normalized_games = normalized

        self.log_event(
            "games_normalized",
            count=len(self.normalized_games),
        )

        return normalized

    # -----------------------------------------------------------------------
    # Matching
    # -----------------------------------------------------------------------

    def resolve_igdb_game(
        self,
        game: GameRecord,
    ) -> GameRecord:
        if game.igdb_id:
            game.match_confidence = "existing-id"
            return game

        cache_key = normalize_title(game.title)

        if cache_key in self.igdb_cache:
            game.igdb_id = self.igdb_cache[cache_key]
            game.match_confidence = "cached-title"
            return game

        query = urllib.parse.urlencode({"title": game.title})
        endpoint = self.endpoints["trackarr_games"] + "?" + query

        response = self.trackarr.request_json(
            "GET",
            endpoint,
            headers=self._trackarr_headers(),
        )

        candidates = as_list(response)

        exact = [
            candidate
            for candidate in candidates
            if normalize_title(
                str(first_value(candidate, "title", "name", default=""))
            ) == normalize_title(game.title)
        ]

        if len(exact) == 1:
            game.igdb_id = first_value(exact[0], "igdb_id", "igdbId")
            game.match_confidence = "exact-title"

        elif len(candidates) == 1:
            game.igdb_id = first_value(
                candidates[0],
                "igdb_id",
                "igdbId",
            )
            game.match_confidence = "single-candidate"

        else:
            game.match_confidence = "manual-review"
            self._write_review_record(game, candidates)

        if game.igdb_id:
            self.igdb_cache[cache_key] = int(game.igdb_id)

        return game

    def _write_review_record(
        self,
        game: GameRecord,
        candidates: list[dict[str, Any]],
    ) -> None:
        record = {
            "game": asdict(game),
            "candidates": candidates,
        }

        with (self.state_dir / "review.jsonl").open(
            "a",
            encoding="utf-8",
        ) as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def resolve_all_games(self) -> list[GameRecord]:
        resolved = []

        for game in self.normalized_games:
            try:
                game = self.resolve_igdb_game(game)

                if game.igdb_id:
                    resolved.append(game)
                else:
                    self.log_event(
                        "game_needs_review",
                        store=game.store,
                        source_game_id=game.source_game_id,
                        title=game.title,
                    )

            except Exception as exc:
                self.log_event(
                    "game_resolution_failed",
                    title=game.title,
                    error=str(exc),
                )

        self.resolved_games = resolved
        return resolved

    # -----------------------------------------------------------------------
    # Trackarr
    # -----------------------------------------------------------------------

    def login_to_trackarr(self, username: str, password: str) -> None:
        response = self.trackarr.request_json(
            "POST",
            self.endpoints["trackarr_login"],
            body={"username": username, "password": password},
        )

        token = first_value(
            response,
            "access_token",
            "accessToken",
            "token",
        )

        if not token:
            raise AuthenticationError(
                "Trackarr login response did not contain an access token"
            )

        self.trackarr_token = str(token)
        self.log_event("trackarr_login_success")

    def _trackarr_headers(self) -> dict[str, str]:
        if not self.trackarr_token:
            raise AuthenticationError("Trackarr login has not been completed")

        return {
            "Authorization": f"Bearer {self.trackarr_token}",
        }

    def find_or_create_trackarr_game(
        self,
        game: GameRecord,
    ) -> str:
        if not game.igdb_id:
            raise ImporterError(f"Game has no IGDB ID: {game.title}")

        cache_key = str(game.igdb_id)

        if cache_key in self.trackarr_game_cache:
            return self.trackarr_game_cache[cache_key]

        response = self.trackarr.request_json(
            "GET",
            self.endpoints["trackarr_games"],
            headers=self._trackarr_headers(),
            query={"igdb_id": game.igdb_id},
        )

        candidates = as_list(response)
        trackarr_id = None

        if candidates:
            trackarr_id = first_value(
                candidates[0],
                "id",
                "game_id",
                "gameId",
            )

        if trackarr_id is None and not self.dry_run:
            created = self.trackarr.request_json(
                "POST",
                self.endpoints["trackarr_games"],
                headers=self._trackarr_headers(),
                body={
                    "igdb_id": game.igdb_id,
                    "title": game.title,
                },
            )

            trackarr_id = first_value(
                created,
                "id",
                "game_id",
                "gameId",
            )

        if trackarr_id is None:
            trackarr_id = f"dry-run-{game.igdb_id}"

        trackarr_id = str(trackarr_id)
        self.trackarr_game_cache[cache_key] = trackarr_id
        return trackarr_id

    def add_trackarr_library_entry(
        self,
        game: GameRecord,
    ) -> bool:
        if not game.trackarr_game_id:
            raise ImporterError("Trackarr game ID has not been assigned")

        response = self.trackarr.request_json(
            "GET",
            self.endpoints["trackarr_library"],
            headers=self._trackarr_headers(),
            query={
                "game_id": game.trackarr_game_id,
                "store": game.store,
                "source_game_id": game.source_game_id,
                "platform": game.platform,
            },
        )

        if as_list(response) or (
            isinstance(response, dict)
            and response.get("exists") is True
        ):
            self.log_event(
                "library_entry_skipped_duplicate",
                title=game.title,
                store=game.store,
                platform=game.platform,
            )
            return False

        if self.dry_run:
            self.log_event(
                "library_entry_dry_run",
                title=game.title,
                store=game.store,
                platform=game.platform,
            )
            return True

        self.trackarr.request_json(
            "POST",
            self.endpoints["trackarr_library"],
            headers=self._trackarr_headers(),
            body={
                "game_id": game.trackarr_game_id,
                "source": game.source_game_id,
                "store": game.store,
                "platform": game.platform,
                "format": "digital",
                "ownership": "owned",
            },
        )

        self.log_event(
            "library_entry_added",
            title=game.title,
            store=game.store,
            platform=game.platform,
        )
        return True

    def import_all_to_trackarr(self) -> None:
        for game in self.resolved_games:
            try:
                game.trackarr_game_id = (
                    self.find_or_create_trackarr_game(game)
                )
                self.add_trackarr_library_entry(game)
                self.save_checkpoint()

            except Exception as exc:
                self.log_event(
                    "trackarr_import_failed",
                    title=game.title,
                    store=game.store,
                    platform=game.platform,
                    error=str(exc),
                )

    # -----------------------------------------------------------------------
    # Complete sequential workflow
    # -----------------------------------------------------------------------

    def run(
        self,
        *,
        backlogia_username: str,
        backlogia_password: str,
        trackarr_username: str,
        trackarr_password: str,
    ) -> None:
        """
        Run the complete workflow in dependency order.
        """

        self.load_checkpoint()

        self.login_to_backlogia(
            backlogia_username,
            backlogia_password,
        )

        self.sync_backlogia_steam()
        self.sync_backlogia_epic()
        self.fetch_backlogia_games()
        self.save_checkpoint()

        self.login_to_itch()
        self.fetch_itch_owned_games()

        # This is optional for installations where the owned-keys endpoint
        # already contains sufficient ownership information.
        try:
            self.fetch_all_itch_bundle_games()
        except APIError as exc:
            self.logger.warning("Bundle endpoint unavailable: %s", exc)

        self.save_checkpoint()

        self.normalize_all_games()

        self.login_to_trackarr(
            trackarr_username,
            trackarr_password,
        )

        self.resolve_all_games()
        self.save_checkpoint()

        self.import_all_to_trackarr()
        self.save_checkpoint()

        self.log_event(
            "import_complete",
            normalized=len(self.normalized_games),
            resolved=len(self.resolved_games),
        )


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import game libraries into VideoGameTrackarr."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config")
    run_parser.add_argument("--backlogia-url")
    run_parser.add_argument("--trackarr-url")
    run_parser.add_argument("--backlogia-username")
    run_parser.add_argument("--backlogia-password")
    run_parser.add_argument("--trackarr-username")
    run_parser.add_argument("--trackarr-password")
    run_parser.add_argument("--itch-api-key")
    run_parser.add_argument(
        "--state-dir",
        default=".library-importer",
    )
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--verbose", action="store_true")

    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("--verbose", action="store_true")

    return parser


def get_value(
    args: argparse.Namespace,
    config: dict[str, Any],
    section: str,
    name: str,
    *,
    required: bool = True,
) -> str | None:
    value = getattr(args, name, None)

    if value is None:
        value = config.get(section, {}).get(name)

    if value is None:
        env_name = f"{section.upper()}_{name.upper()}"
        value = os.environ.get(env_name)

    if required and not value:
        raise SystemExit(
            f"Missing {section}.{name}. Use an argument, config file, "
            f"or {section.upper()}_{name.upper()}."
        )

    return value


def run_from_cli(args: argparse.Namespace) -> None:
    config = load_json_file(Path(args.config)) if args.config else {}

    backlogia_url = get_value(
        args,
        config,
        "backlogia",
        "url",
    )
    trackarr_url = get_value(
        args,
        config,
        "trackarr",
        "url",
    )

    importer = LibraryImporter(
        backlogia_url,
        trackarr_url,
        itch_api_key=get_value(
            args,
            config,
            "itch",
            "api_key",
        ),
        state_dir=args.state_dir,
        dry_run=args.dry_run,
    )

    importer.run(
        backlogia_username=get_value(
            args,
            config,
            "backlogia",
            "username",
        ),
        backlogia_password=get_value(
            args,
            config,
            "backlogia",
            "password",
        ),
        trackarr_username=get_value(
            args,
            config,
            "trackarr",
            "username",
        ),
        trackarr_password=get_value(
            args,
            config,
            "trackarr",
            "password",
        ),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, payload: Any, status: int = 200):
        self.payload = payload
        self.status = status

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class FakeOpener:
    def __init__(self, responses: list[Any]):
        self.responses = list(responses)
        self.requests: list[urllib.request.Request] = []

    def open(self, request: urllib.request.Request, timeout: float):
        self.requests.append(request)

        if not self.responses:
            raise AssertionError("No fake response remaining")

        response = self.responses.pop(0)

        if isinstance(response, Exception):
            raise response

        return response


class ImporterTests(unittest.TestCase):
    def make_importer(self, state_dir: str | None = None) -> LibraryImporter:
        if state_dir is None:
            state_dir = os.path.join(
                os.getcwd(),
                f".test-state-{uuid.uuid4()}",
            )

        importer = LibraryImporter(
            "https://backlogia.test",
            "https://trackarr.test",
            itch_api_key="itch-test-key",
            state_dir=state_dir,
            min_interval=0,
            error_wait=0,
            poll_interval=0,
        )
        importer.trackarr_token = "trackarr-token"
        return importer


    def test_title_normalization(self):
        self.assertEqual(
            normalize_title("The Witcher 3: Wild Hunt"),
            "the witcher 3 wild hunt",
        )

    def test_platform_extraction(self):
        self.assertEqual(
            extract_platforms({"platforms": ["windows", "linux"]}),
            ["Windows", "Linux"],
        )

    def test_itch_deduplication_merges_provenance(self):
        importer = self.make_importer()

        records = [
            GameRecord(
                store="itch.io",
                source_game_id="42",
                title="Game",
                platform="Windows",
                from_normal_itch_library=True,
            ),
            GameRecord(
                store="itch.io",
                source_game_id="42",
                title="Game",
                platform="Windows",
                bundles=[{"id": "bundle-1", "name": "Bundle"}],
            ),
            GameRecord(
                store="itch.io",
                source_game_id="42",
                title="Game",
                platform="Windows",
                bundles=[{"id": "bundle-2", "name": "Bundle 2"}],
            ),
        ]

        result = importer.deduplicate_itch_games(records)

        self.assertEqual(len(result), 1)
        self.assertTrue(result[0].from_normal_itch_library)
        self.assertEqual(len(result[0].bundles), 2)

    def test_steam_normalization(self):
        importer = self.make_importer()

        result = importer.normalize_steam_game({
            "steam_id": 10,
            "title": "Example",
            "platforms": ["windows", "linux"],
        })

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].store, "Steam")
        self.assertEqual(result[0].source_game_id, "10")

    def test_epic_normalization(self):
        importer = self.make_importer()

        result = importer.normalize_epic_game({
            "epic_id": "epic-1",
            "title": "Example",
        })

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].store, "Epic Games Store")
        self.assertEqual(result[0].platform, "Windows")

    def test_itch_pagination(self):
        importer = self.make_importer()

        fake_opener = FakeOpener([
            FakeResponse({"owned_keys": [{"id": 1}]}),
            FakeResponse({"owned_keys": [{"id": 2}]}),
            FakeResponse({"owned_keys": []}),
        ])

        importer.itch.opener = fake_opener
        result = importer.fetch_itch_owned_games()

        self.assertEqual(len(result), 2)
        self.assertEqual(len(fake_opener.requests), 3)

    def test_checkpoint_round_trip(self):
        importer = self.make_importer()
        importer.normalized_games = [
            GameRecord(
                store="Steam",
                source_game_id="1",
                title="Game",
                platform="Windows",
            )
        ]
        importer.save_checkpoint()

        restored = self.make_importer()
        restored.load_checkpoint()

        self.assertEqual(len(restored.normalized_games), 1)
        self.assertEqual(restored.normalized_games[0].title, "Game")

    def test_existing_igdb_id_is_preserved(self):
        importer = self.make_importer()

        game = GameRecord(
            store="Steam",
            source_game_id="1",
            title="Game",
            platform="Windows",
            igdb_id=999,
        )

        result = importer.resolve_igdb_game(game)

        self.assertEqual(result.igdb_id, 999)
        self.assertEqual(result.match_confidence, "existing-id")

    def test_trackarr_game_cache(self):
        importer = self.make_importer()

        fake_opener = FakeOpener([
            FakeResponse({"games": [{"id": "trackarr-1"}]}),
        ])
        importer.trackarr.opener = fake_opener

        game = GameRecord(
            store="Steam",
            source_game_id="1",
            title="Game",
            platform="Windows",
            igdb_id=999,
        )

        result = importer.find_or_create_trackarr_game(game)

        self.assertEqual(result, "trackarr-1")
        self.assertEqual(importer.trackarr_game_cache["999"], "trackarr-1")

    def test_http_client_returns_json(self):
        opener = FakeOpener([
            FakeResponse({"ok": True}),
        ])

        client = HTTPClient(
            "https://example.test",
            opener=opener,
            min_interval=0,
            error_wait=0,
        )

        result = client.request_json("GET", "/test")

        self.assertEqual(result, {"ok": True})
        self.assertEqual(len(opener.requests), 1)

    def test_different_stores_are_not_merged(self):
        importer = self.make_importer()

        records = [
            GameRecord(
                store="Steam",
                source_game_id="1",
                title="Game",
                platform="Windows",
            ),
            GameRecord(
                store="Epic Games Store",
                source_game_id="1",
                title="Game",
                platform="Windows",
            ),
        ]

        result = importer.deduplicate_itch_games(records)

        self.assertEqual(len(result), 2)


def run_tests(verbose: bool = False) -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        ImporterTests
    )
    runner = unittest.TextTestRunner(
        verbosity=2 if verbose else 1
    )
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


# ---------------------------------------------------------------------------
# Single main entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "test":
        configure_logging(args.verbose)
        return run_tests(args.verbose)

    if args.command == "run":
        configure_logging(args.verbose)

        try:
            run_from_cli(args)
            return 0
        except KeyboardInterrupt:
            logging.getLogger("library-importer").error(
                "Interrupted; progress is available in the checkpoint"
            )
            return 130
        except Exception as exc:
            logging.getLogger("library-importer").exception(
                "Import failed: %s",
                exc,
            )
            return 1

    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
