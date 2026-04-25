import datetime as dt
import hashlib
import html
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

MLB_URL = "https://www.mlb.com/probable-pitchers/"
MLB_STATS_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
MLB_PLAYER_STATS_URL = "https://statsapi.mlb.com/api/v1/people/{player_id}/stats"
TRACKER_FILE = "f5_tracker.csv"
CARD_TRACKER_FILE = "f5_card_tracker.csv"
BACKTEST_CACHE_FILE = "f5_backtest_cache.json"
MATCHUPS_CACHE_FILE = "f5_matchups_cache.json"
ADMIN_BACKTEST_KEY = "thatsbaseball"
# Pinned board on Command Center (must match a `SYSTEMS[].name`).
FEATURED_SYSTEM_NAME = "Tempo Control"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


@dataclass
class Pitcher:
    name: str
    era: Optional[float]
    strikeouts: Optional[int]
    whip: Optional[float]
    games_started: Optional[int]
    innings_pitched: Optional[float]


@dataclass
class Matchup:
    away_team: str
    home_team: str
    away_pitcher: Pitcher
    home_pitcher: Pitcher


@dataclass
class SystemDefinition:
    key: str
    name: str
    rules: List[str]
    evaluator: Callable[[Matchup], Dict[str, Any]]


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text or text in {"-", "--", "N/A", "na"}:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(m.group()) if m else None


def to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = str(value).strip().replace(",", "")
    if not text or text in {"-", "--", "N/A", "na"}:
        return None
    m = re.search(r"-?\d+", text)
    return int(m.group()) if m else None


def parse_ip_to_outs(ip: Any) -> Optional[int]:
    """
    Baseball innings format:
    - 5.0 = 5 innings
    - 5.1 = 5 innings + 1 out
    - 5.2 = 5 innings + 2 outs
    """
    if ip is None:
        return None
    text = str(ip).strip()
    if not text:
        return None
    if "." in text:
        whole, frac = text.split(".", 1)
        inn = to_int(whole)
        if inn is None:
            return None
        frac_digit = to_int(frac[:1]) if frac else 0
        frac_digit = frac_digit if frac_digit in {0, 1, 2} else 0
        return inn * 3 + frac_digit
    inn = to_int(text)
    return inn * 3 if inn is not None else None


def pitcher_sample_score(p: Pitcher) -> float:
    gs = p.games_started or 0
    ip_outs = parse_ip_to_outs(p.innings_pitched) or 0
    ip = ip_outs / 3.0
    gs_factor = min(1.0, gs / 8.0)
    ip_factor = min(1.0, ip / 35.0)
    score = 0.5 * gs_factor + 0.5 * ip_factor
    return max(0.15, min(1.0, score))


def reliability_multiplier(m: Matchup, better_side_key: str) -> float:
    better = side_pitcher(m, better_side_key)
    worse = side_pitcher(m, "home" if better_side_key == "away" else "away")
    b = pitcher_sample_score(better)
    w = pitcher_sample_score(worse)
    # Put more weight on the "better" pitcher sample being legit.
    return max(0.55, min(1.0, 0.70 * b + 0.30 * w))


def norm_name(text: Optional[str], fallback: str = "TBD") -> str:
    if not text:
        return fallback
    if isinstance(text, dict):
        text = text.get("name") or text.get("teamName") or text.get("fullName") or ""
    cleaned = re.sub(r"\s+", " ", str(text)).strip()
    return cleaned if cleaned else fallback


def norm_key(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def canonical_team_key(text: Optional[str]) -> str:
    k = norm_key(text)
    aliases = {
        "as": "athletics",
        "oaklandathletics": "athletics",
        "athletics": "athletics",
        "dbacks": "arizonadiamondbacks",
        "diamondbacks": "arizonadiamondbacks",
        "arizonadiamondbacks": "arizonadiamondbacks",
        "chicagowhitesox": "cws",
        "whitesox": "cws",
        "cws": "cws",
        "chicagocubs": "chc",
        "cubs": "chc",
        "chc": "chc",
    }
    return aliases.get(k, k)


def dig(data: Any, keys: List[str]) -> Optional[Any]:
    if data is None:
        return None
    if isinstance(data, dict):
        lowered = {str(k).lower(): k for k in data.keys()}
        for key in keys:
            match = lowered.get(key.lower())
            if match is not None:
                return data[match]
        for v in data.values():
            found = dig(v, keys)
            if found is not None:
                return found
    elif isinstance(data, list):
        for item in data:
            found = dig(item, keys)
            if found is not None:
                return found
    return None


def parse_from_next_data(html: str) -> List[Matchup]:
    soup = BeautifulSoup(html, "html.parser")
    script = soup.find("script", id="__NEXT_DATA__", type="application/json")
    if not script or not script.string:
        return []

    import json

    try:
        data = json.loads(script.string)
    except Exception:
        return []

    game_nodes: List[Dict[str, Any]] = []
    stack = [data]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            keys = {str(k).lower() for k in node.keys()}
            if (
                ("away" in keys or "home" in keys or "teams" in keys)
                and ("probablepitchers" in keys or "awaypitcher" in keys or "homepitcher" in keys)
            ):
                game_nodes.append(node)
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)

    out: List[Matchup] = []
    for game in game_nodes:
        away_team = dig(game, ["awayTeamName", "awayName", "away"]) or "Away"
        home_team = dig(game, ["homeTeamName", "homeName", "home"]) or "Home"

        if isinstance(away_team, dict):
            away_team = away_team.get("name") or away_team.get("clubName") or away_team.get("abbreviation") or "Away"
        if isinstance(home_team, dict):
            home_team = home_team.get("name") or home_team.get("clubName") or home_team.get("abbreviation") or "Home"

        pp = game.get("probablePitchers") if isinstance(game, dict) else None
        away_node = pp.get("away") if isinstance(pp, dict) else None
        home_node = pp.get("home") if isinstance(pp, dict) else None
        if away_node is None:
            away_node = dig(game, ["awayPitcher", "awayProbablePitcher", "awayStarter"])
        if home_node is None:
            home_node = dig(game, ["homePitcher", "homeProbablePitcher", "homeStarter"])

        away_pitcher = Pitcher(
            name=norm_name(dig(away_node, ["fullName", "name", "displayName"]) if away_node else None),
            era=to_float(dig(away_node, ["era", "seasonEra"]) if away_node else None),
            strikeouts=to_int(dig(away_node, ["strikeouts", "so", "k"]) if away_node else None),
            whip=to_float(dig(away_node, ["whip"]) if away_node else None),
            games_started=to_int(dig(away_node, ["gamesStarted", "gs"]) if away_node else None),
            innings_pitched=to_float(dig(away_node, ["inningsPitched", "ip"]) if away_node else None),
        )
        home_pitcher = Pitcher(
            name=norm_name(dig(home_node, ["fullName", "name", "displayName"]) if home_node else None),
            era=to_float(dig(home_node, ["era", "seasonEra"]) if home_node else None),
            strikeouts=to_int(dig(home_node, ["strikeouts", "so", "k"]) if home_node else None),
            whip=to_float(dig(home_node, ["whip"]) if home_node else None),
            games_started=to_int(dig(home_node, ["gamesStarted", "gs"]) if home_node else None),
            innings_pitched=to_float(dig(home_node, ["inningsPitched", "ip"]) if home_node else None),
        )

        out.append(
            Matchup(
                away_team=norm_name(str(away_team), "Away"),
                home_team=norm_name(str(home_team), "Home"),
                away_pitcher=away_pitcher,
                home_pitcher=home_pitcher,
            )
        )

    seen = set()
    deduped = []
    for m in out:
        key = (m.away_team.lower(), m.home_team.lower(), m.away_pitcher.name.lower(), m.home_pitcher.name.lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(m)
    return deduped


def parse_from_html_fallback(html: str) -> List[Matchup]:
    soup = BeautifulSoup(html, "html.parser")
    cards = soup.select('[class*="probable-pitchers"] [class*="card"], [data-testid*="game"], article')
    out: List[Matchup] = []
    for card in cards:
        txt = " ".join(card.stripped_strings)
        if not txt:
            continue
        teams = [norm_name(t.get_text(" ", strip=True)) for t in card.select('[class*="team"], [data-testid*="team"]') if t.get_text(strip=True)]
        pitchers = [norm_name(p.get_text(" ", strip=True)) for p in card.select('[class*="pitcher"], [data-testid*="pitcher"]') if p.get_text(strip=True)]
        eras = [to_float(x) for x in re.findall(r"ERA\s*[:]?\s*(\d+(?:\.\d+)?)", txt, flags=re.I)]
        ks = [to_int(x) for x in re.findall(r"(?:SO|K|Strikeouts)\s*[:]?\s*(\d+)", txt, flags=re.I)]
        if len(teams) < 2:
            continue
        out.append(
            Matchup(
                away_team=teams[0],
                home_team=teams[1],
                away_pitcher=Pitcher(
                    name=pitchers[0] if len(pitchers) >= 1 else "TBD",
                    era=eras[0] if len(eras) >= 1 else None,
                    strikeouts=ks[0] if len(ks) >= 1 else None,
                    whip=None,
                    games_started=None,
                    innings_pitched=None,
                ),
                home_pitcher=Pitcher(
                    name=pitchers[1] if len(pitchers) >= 2 else "TBD",
                    era=eras[1] if len(eras) >= 2 else None,
                    strikeouts=ks[1] if len(ks) >= 2 else None,
                    whip=None,
                    games_started=None,
                    innings_pitched=None,
                ),
            )
        )
    return out


@st.cache_data(ttl=60 * 60 * 6)
def fetch_pitcher_stats(player_id: int, season: int) -> Dict[str, Any]:
    params = {"stats": "season", "group": "pitching", "season": season}
    r = requests.get(
        MLB_PLAYER_STATS_URL.format(player_id=player_id),
        params=params,
        headers=HEADERS,
        timeout=20,
    )
    r.raise_for_status()
    data = r.json()
    stats = data.get("stats", [])
    if not stats:
        return {}
    splits = stats[0].get("splits", [])
    if not splits:
        return {}
    return splits[0].get("stat", {}) or {}


def resolve_pitcher_stats(probable_pitcher_node: Dict[str, Any], season: int) -> Dict[str, Any]:
    node_stats = probable_pitcher_node.get("stats") or []
    if node_stats:
        stat_dict = node_stats[0].get("stats", {}) or {}
        if stat_dict.get("era") is not None or stat_dict.get("strikeOuts") is not None or stat_dict.get("whip") is not None:
            return stat_dict
    player_id = probable_pitcher_node.get("id")
    if player_id is None:
        return {}
    try:
        return fetch_pitcher_stats(int(player_id), season)
    except Exception:
        return {}


def parse_from_stats_api(game_date: dt.date) -> List[Matchup]:
    season = game_date.year
    params = {
        "sportId": 1,
        "date": game_date.strftime("%Y-%m-%d"),
        "hydrate": f"probablePitcher(note,stats(group=[pitching],type=[season],season={season}),person)",
    }
    r = requests.get(MLB_STATS_SCHEDULE_URL, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    data = r.json()
    out: List[Matchup] = []
    for date_node in data.get("dates", []):
        for game in date_node.get("games", []):
            away_node = game.get("teams", {}).get("away", {})
            home_node = game.get("teams", {}).get("home", {})
            away_team = norm_name(away_node.get("team", {}).get("name"), "Away")
            home_team = norm_name(home_node.get("team", {}).get("name"), "Home")
            away_pp = away_node.get("probablePitcher") or {}
            home_pp = home_node.get("probablePitcher") or {}
            away_stats = resolve_pitcher_stats(away_pp, season)
            home_stats = resolve_pitcher_stats(home_pp, season)
            away_pitcher = Pitcher(
                name=norm_name(away_pp.get("fullName"), "TBD"),
                era=to_float(away_stats.get("era")),
                strikeouts=to_int(away_stats.get("strikeOuts") or away_stats.get("strikeouts")),
                whip=to_float(away_stats.get("whip")),
                games_started=to_int(away_stats.get("gamesStarted")),
                innings_pitched=to_float(away_stats.get("inningsPitched")),
            )
            home_pitcher = Pitcher(
                name=norm_name(home_pp.get("fullName"), "TBD"),
                era=to_float(home_stats.get("era")),
                strikeouts=to_int(home_stats.get("strikeOuts") or home_stats.get("strikeouts")),
                whip=to_float(home_stats.get("whip")),
                games_started=to_int(home_stats.get("gamesStarted")),
                innings_pitched=to_float(home_stats.get("inningsPitched")),
            )
            out.append(
                Matchup(
                    away_team=away_team,
                    home_team=home_team,
                    away_pitcher=away_pitcher,
                    home_pitcher=home_pitcher,
                )
            )
    return out


@st.cache_data(ttl=60 * 20)
def fetch_matchups(_refresh_nonce: int) -> Tuple[List[Matchup], str]:
    try:
        r = requests.get(MLB_URL, headers=HEADERS, timeout=20)
        r.raise_for_status()
        html = r.text
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch MLB probable pitchers: {exc}")

    parsed = parse_from_next_data(html)
    if parsed:
        return parsed, "__NEXT_DATA__ JSON"
    parsed = parse_from_html_fallback(html)
    if parsed:
        return parsed, "HTML fallback"
    try:
        parsed = parse_from_stats_api(dt.date.today())
    except Exception as exc:
        raise RuntimeError(f"Could not parse probable pitchers from MLB page or Stats API: {exc}")
    return parsed, "MLB Stats API fallback"


def matchup_to_dict(m: Matchup) -> Dict[str, Any]:
    return {
        "away_team": m.away_team,
        "home_team": m.home_team,
        "away_pitcher": {
            "name": m.away_pitcher.name,
            "era": m.away_pitcher.era,
            "strikeouts": m.away_pitcher.strikeouts,
            "whip": m.away_pitcher.whip,
            "games_started": m.away_pitcher.games_started,
            "innings_pitched": m.away_pitcher.innings_pitched,
        },
        "home_pitcher": {
            "name": m.home_pitcher.name,
            "era": m.home_pitcher.era,
            "strikeouts": m.home_pitcher.strikeouts,
            "whip": m.home_pitcher.whip,
            "games_started": m.home_pitcher.games_started,
            "innings_pitched": m.home_pitcher.innings_pitched,
        },
    }


def matchup_from_dict(d: Dict[str, Any]) -> Matchup:
    ap = d.get("away_pitcher", {}) if isinstance(d, dict) else {}
    hp = d.get("home_pitcher", {}) if isinstance(d, dict) else {}
    return Matchup(
        away_team=str(d.get("away_team", "")),
        home_team=str(d.get("home_team", "")),
        away_pitcher=Pitcher(
            name=str(ap.get("name", "TBD")),
            era=to_float(ap.get("era")),
            strikeouts=to_int(ap.get("strikeouts")),
            whip=to_float(ap.get("whip")),
            games_started=to_int(ap.get("games_started")),
            innings_pitched=to_float(ap.get("innings_pitched")),
        ),
        home_pitcher=Pitcher(
            name=str(hp.get("name", "TBD")),
            era=to_float(hp.get("era")),
            strikeouts=to_int(hp.get("strikeouts")),
            whip=to_float(hp.get("whip")),
            games_started=to_int(hp.get("games_started")),
            innings_pitched=to_float(hp.get("innings_pitched")),
        ),
    )


def matchups_cache_file_path() -> str:
    return os.path.join(os.path.dirname(__file__), MATCHUPS_CACHE_FILE)


def save_matchups_cache(matchups: List[Matchup], parse_mode: str) -> None:
    payload = {
        "saved_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "parse_mode": parse_mode,
        "matchups": [matchup_to_dict(m) for m in matchups],
    }
    with open(matchups_cache_file_path(), "w", encoding="utf-8") as f:
        json.dump(payload, f)


def load_matchups_cache() -> Tuple[List[Matchup], str, Optional[str]]:
    path = matchups_cache_file_path()
    if not os.path.exists(path):
        return [], "", None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        rows = payload.get("matchups", [])
        parse_mode = str(payload.get("parse_mode", "cached"))
        saved_at = payload.get("saved_at")
        matchups = [matchup_from_dict(r) for r in rows if isinstance(r, dict)]
        return matchups, parse_mode, str(saved_at) if saved_at else None
    except Exception:
        return [], "", None


def matchups_cache_is_stale(saved_at: Optional[str]) -> bool:
    """Refresh slate when missing, wrong calendar day, or older than a few hours."""
    if not saved_at:
        return True
    try:
        saved = dt.datetime.strptime(saved_at[:19], "%Y-%m-%d %H:%M:%S")
    except Exception:
        return True
    if saved.date() != dt.date.today():
        return True
    return (dt.datetime.now() - saved).total_seconds() > 3 * 3600


def better_side(matchup: Matchup) -> Optional[str]:
    a = matchup.away_pitcher
    h = matchup.home_pitcher
    if a.era is not None and h.era is not None:
        if a.era < h.era:
            return "away"
        if h.era < a.era:
            return "home"
    if a.whip is not None and h.whip is not None:
        if a.whip < h.whip:
            return "away"
        if h.whip < a.whip:
            return "home"
    if a.strikeouts is not None and h.strikeouts is not None:
        if a.strikeouts > h.strikeouts:
            return "away"
        if h.strikeouts > a.strikeouts:
            return "home"
    return None


def side_pitcher(m: Matchup, side: str) -> Pitcher:
    return m.away_pitcher if side == "away" else m.home_pitcher


def side_team(m: Matchup, side: str) -> str:
    return m.away_team if side == "away" else m.home_team


def stat_diff(m: Matchup, better: str, stat: str, invert: bool = False) -> float:
    b = getattr(side_pitcher(m, better), stat)
    w = getattr(side_pitcher(m, "home" if better == "away" else "away"), stat)
    if b is None or w is None:
        return 0.0
    val = (w - b) if not invert else (b - w)
    return float(val)


def eval_atlas_ace_gap(m: Matchup) -> Dict[str, Any]:
    b = better_side(m)
    if b is None:
        return {"qualifies": False, "pick": "No Play", "edge_score": 0.0, "confidence": 0.0}
    better = side_pitcher(m, b)
    worse = side_pitcher(m, "home" if b == "away" else "away")
    era_gap = stat_diff(m, b, "era")
    k_diff = (better.strikeouts or 0) - (worse.strikeouts or 0)
    rules = (
        better.era is not None
        and worse.era is not None
        and better.era <= 2.75
        and worse.era >= 4.50
        and era_gap >= 2.0
        and k_diff > 0
    )
    if pitcher_sample_score(better) < 0.45 and (better.era or 99) <= 2.20:
        rules = False
    score = max(0.0, (era_gap - 1.5) * 24.0) + (10.0 if k_diff > 0 else 0.0) + (8.0 if era_gap >= 2.5 else 0.0)
    score *= reliability_multiplier(m, b)
    conf = min(100.0, max(0.0, 25.0 + score))
    if not rules:
        conf = max(0.0, conf - 35.0)
    return {
        "qualifies": rules,
        "pick": f"F5 {side_team(m, b)}" if rules else "No Play",
        "edge_score": round(score, 2),
        "confidence": round(conf, 1),
    }


def eval_iron_whip_clamp(m: Matchup) -> Dict[str, Any]:
    b = better_side(m)
    if b is None:
        return {"qualifies": False, "pick": "No Play", "edge_score": 0.0, "confidence": 0.0}
    better = side_pitcher(m, b)
    worse = side_pitcher(m, "home" if b == "away" else "away")
    whip_gap = stat_diff(m, b, "whip")
    era_gap = stat_diff(m, b, "era")
    rules = (
        better.whip is not None
        and worse.whip is not None
        and better.whip <= 1.12
        and worse.whip >= 1.35
        and whip_gap >= 0.18
        and era_gap >= 1.0
    )
    score = max(0.0, (whip_gap - 0.10) * 160.0) + max(0.0, era_gap * 8.0)
    score *= reliability_multiplier(m, b)
    conf = min(100.0, 22.0 + score)
    if not rules:
        conf = max(0.0, conf - 28.0)
    return {
        "qualifies": rules,
        "pick": f"F5 {side_team(m, b)}" if rules else "No Play",
        "edge_score": round(score, 2),
        "confidence": round(conf, 1),
    }


def eval_k_pressure_index(m: Matchup) -> Dict[str, Any]:
    b = better_side(m)
    if b is None:
        return {"qualifies": False, "pick": "No Play", "edge_score": 0.0, "confidence": 0.0}
    better = side_pitcher(m, b)
    worse = side_pitcher(m, "home" if b == "away" else "away")
    era_gap = stat_diff(m, b, "era")
    k_diff = (better.strikeouts or 0) - (worse.strikeouts or 0)
    rules = (
        better.era is not None
        and worse.era is not None
        and better.era <= 3.40
        and worse.era >= 4.20
        and k_diff >= 18
    )
    score = max(0.0, k_diff * 1.5) + max(0.0, era_gap * 10.0)
    score *= reliability_multiplier(m, b)
    conf = min(100.0, 15.0 + score)
    if not rules:
        conf = max(0.0, conf - 26.0)
    return {
        "qualifies": rules,
        "pick": f"F5 {side_team(m, b)}" if rules else "No Play",
        "edge_score": round(score, 2),
        "confidence": round(conf, 1),
    }


def eval_trinity_starter_blend(m: Matchup) -> Dict[str, Any]:
    b = better_side(m)
    if b is None:
        return {"qualifies": False, "pick": "No Play", "edge_score": 0.0, "confidence": 0.0}
    era_gap = max(0.0, stat_diff(m, b, "era"))
    whip_gap = max(0.0, stat_diff(m, b, "whip"))
    better = side_pitcher(m, b)
    worse = side_pitcher(m, "home" if b == "away" else "away")
    k_diff = max(0.0, (better.strikeouts or 0) - (worse.strikeouts or 0))
    blend = era_gap * 0.52 + whip_gap * 2.5 + (k_diff / 30.0)
    score = blend * 35.0
    score *= reliability_multiplier(m, b)
    rules = score >= 55.0 and era_gap >= 1.25 and k_diff >= 8
    conf = min(100.0, max(0.0, 18.0 + score))
    if not rules:
        conf = max(0.0, conf - 24.0)
    return {
        "qualifies": rules,
        "pick": f"F5 {side_team(m, b)}" if rules else "No Play",
        "edge_score": round(score, 2),
        "confidence": round(conf, 1),
    }


def eval_volatility_fade(m: Matchup) -> Dict[str, Any]:
    b = better_side(m)
    if b is None:
        return {"qualifies": False, "pick": "No Play", "edge_score": 0.0, "confidence": 0.0}
    better = side_pitcher(m, b)
    worse = side_pitcher(m, "home" if b == "away" else "away")
    era_gap = stat_diff(m, b, "era")
    k_diff = (better.strikeouts or 0) - (worse.strikeouts or 0)
    red_flag = (
        (worse.era is not None and worse.era >= 5.20)
        or (worse.whip is not None and worse.whip >= 1.45)
    )
    rules = (
        red_flag
        and better.era is not None
        and better.era <= 3.85
        and era_gap >= 1.30
        and k_diff >= 0
    )
    score = 0.0
    score += max(0.0, era_gap * 14.0)
    score += 12.0 if red_flag else 0.0
    score += max(0.0, k_diff / 2.5)
    score *= reliability_multiplier(m, b)
    conf = min(100.0, 20.0 + score)
    if not rules:
        conf = max(0.0, conf - 22.0)
    return {
        "qualifies": rules,
        "pick": f"F5 {side_team(m, b)}" if rules else "No Play",
        "edge_score": round(score, 2),
        "confidence": round(conf, 1),
    }


def eval_quality_anchor(m: Matchup) -> Dict[str, Any]:
    b = better_side(m)
    if b is None:
        return {"qualifies": False, "pick": "No Play", "edge_score": 0.0, "confidence": 0.0}
    better = side_pitcher(m, b)
    worse = side_pitcher(m, "home" if b == "away" else "away")
    era_gap = stat_diff(m, b, "era")
    whip_gap = stat_diff(m, b, "whip")
    rules = (
        better.era is not None
        and better.whip is not None
        and worse.era is not None
        and worse.whip is not None
        and better.era <= 3.25
        and better.whip <= 1.20
        and worse.era >= 4.10
        and worse.whip >= 1.30
        and era_gap >= 1.2
        and whip_gap >= 0.12
    )
    score = max(0.0, era_gap * 16.0) + max(0.0, whip_gap * 120.0)
    score *= reliability_multiplier(m, b)
    conf = min(100.0, 18.0 + score)
    if not rules:
        conf = max(0.0, conf - 24.0)
    return {
        "qualifies": rules,
        "pick": f"F5 {side_team(m, b)}" if rules else "No Play",
        "edge_score": round(score, 2),
        "confidence": round(conf, 1),
    }


def eval_tempo_control(m: Matchup) -> Dict[str, Any]:
    b = better_side(m)
    if b is None:
        return {"qualifies": False, "pick": "No Play", "edge_score": 0.0, "confidence": 0.0}
    better = side_pitcher(m, b)
    worse = side_pitcher(m, "home" if b == "away" else "away")
    era_gap = stat_diff(m, b, "era")
    whip_gap = stat_diff(m, b, "whip")
    rules = (
        better.era is not None
        and worse.era is not None
        and better.era <= 3.90
        and worse.era >= 4.40
        and era_gap >= 0.90
        and (whip_gap >= 0.08 or (better.whip is not None and better.whip <= 1.22))
    )
    score = max(0.0, era_gap * 13.0) + max(0.0, whip_gap * 85.0)
    score *= reliability_multiplier(m, b)
    conf = min(100.0, 20.0 + score)
    if not rules:
        conf = max(0.0, conf - 20.0)
    return {
        "qualifies": rules,
        "pick": f"F5 {side_team(m, b)}" if rules else "No Play",
        "edge_score": round(score, 2),
        "confidence": round(conf, 1),
    }


def eval_contact_suppression(m: Matchup) -> Dict[str, Any]:
    b = better_side(m)
    if b is None:
        return {"qualifies": False, "pick": "No Play", "edge_score": 0.0, "confidence": 0.0}
    better = side_pitcher(m, b)
    worse = side_pitcher(m, "home" if b == "away" else "away")
    era_gap = stat_diff(m, b, "era")
    whip_gap = stat_diff(m, b, "whip")
    k_diff = (better.strikeouts or 0) - (worse.strikeouts or 0)
    rules = (
        better.whip is not None
        and worse.whip is not None
        and better.whip <= 1.18
        and worse.whip >= 1.28
        and whip_gap >= 0.10
        and k_diff >= 6
        and era_gap >= 0.75
    )
    score = max(0.0, whip_gap * 120.0) + max(0.0, k_diff * 1.2) + max(0.0, era_gap * 8.0)
    score *= reliability_multiplier(m, b)
    conf = min(100.0, 18.0 + score)
    if not rules:
        conf = max(0.0, conf - 22.0)
    return {
        "qualifies": rules,
        "pick": f"F5 {side_team(m, b)}" if rules else "No Play",
        "edge_score": round(score, 2),
        "confidence": round(conf, 1),
    }


def eval_mismatch_momentum(m: Matchup) -> Dict[str, Any]:
    b = better_side(m)
    if b is None:
        return {"qualifies": False, "pick": "No Play", "edge_score": 0.0, "confidence": 0.0}
    better = side_pitcher(m, b)
    worse = side_pitcher(m, "home" if b == "away" else "away")
    era_gap = stat_diff(m, b, "era")
    whip_gap = stat_diff(m, b, "whip")
    k_diff = (better.strikeouts or 0) - (worse.strikeouts or 0)
    red_flag = (
        (worse.era is not None and worse.era >= 4.85)
        or (worse.whip is not None and worse.whip >= 1.38)
    )
    rules = (
        red_flag
        and better.era is not None
        and better.era <= 3.60
        and era_gap >= 1.4
        and k_diff >= 4
    )
    score = max(0.0, era_gap * 16.0) + max(0.0, whip_gap * 60.0) + max(0.0, k_diff)
    score += 10.0 if red_flag else 0.0
    score *= reliability_multiplier(m, b)
    conf = min(100.0, 20.0 + score)
    if not rules:
        conf = max(0.0, conf - 24.0)
    return {
        "qualifies": rules,
        "pick": f"F5 {side_team(m, b)}" if rules else "No Play",
        "edge_score": round(score, 2),
        "confidence": round(conf, 1),
    }


SYSTEMS: List[SystemDefinition] = [
    SystemDefinition(
        key="atlas_ace_gap",
        name="Atlas Ace Gap",
        rules=[
            "Better ERA <= 2.75",
            "Opposing ERA >= 4.50",
            "ERA gap >= 2.0 (2.5+ boost)",
            "Better pitcher strikeouts > opponent",
        ],
        evaluator=eval_atlas_ace_gap,
    ),
    SystemDefinition(
        key="iron_whip_clamp",
        name="Iron WHIP Clamp",
        rules=[
            "Better WHIP <= 1.12",
            "Opposing WHIP >= 1.35",
            "WHIP gap >= 0.18",
            "ERA gap >= 1.0",
        ],
        evaluator=eval_iron_whip_clamp,
    ),
    SystemDefinition(
        key="k_pressure_index",
        name="K-Pressure Index",
        rules=[
            "Better ERA <= 3.40",
            "Opposing ERA >= 4.20",
            "Strikeout edge >= 18",
            "Fades low-miss starters early",
        ],
        evaluator=eval_k_pressure_index,
    ),
    SystemDefinition(
        key="trinity_starter_blend",
        name="Trinity Starter Blend",
        rules=[
            "Weighted blend: ERA gap + WHIP gap + K gap",
            "Requires blended score threshold",
            "Filters to multi-factor edges",
        ],
        evaluator=eval_trinity_starter_blend,
    ),
    SystemDefinition(
        key="volatility_fade",
        name="Volatility Fade",
        rules=[
            "Target opposing pitcher red flags (ERA/WHIP)",
            "Better starter must still be stable",
            "Useful vs blow-up profile arms",
        ],
        evaluator=eval_volatility_fade,
    ),
    SystemDefinition(
        key="quality_anchor",
        name="Quality Anchor",
        rules=[
            "Better starter quality (ERA + WHIP)",
            "Opponent weak on both ERA and WHIP",
            "Requires minimum dual-stat gaps",
        ],
        evaluator=eval_quality_anchor,
    ),
    SystemDefinition(
        key="tempo_control",
        name="Tempo Control",
        rules=[
            "High-frequency edge system",
            "Better ERA <= 3.90 vs Opp ERA >= 4.40",
            "Needs ERA gap + WHIP support",
        ],
        evaluator=eval_tempo_control,
    ),
    SystemDefinition(
        key="contact_suppression",
        name="Contact Suppression",
        rules=[
            "WHIP-driven edge core",
            "Needs better WHIP + K edge + ERA support",
            "Built for stable starter profiles",
        ],
        evaluator=eval_contact_suppression,
    ),
    SystemDefinition(
        key="mismatch_momentum",
        name="Mismatch Momentum",
        rules=[
            "Aggressive fade of volatile opposing starter",
            "Requires red-flag pitcher on other side",
            "Higher edge, slightly lower frequency",
        ],
        evaluator=eval_mismatch_momentum,
    ),
]


def band_from_conf(conf: float) -> str:
    if conf >= 75:
        return "Strong Edge"
    if conf >= 55:
        return "Medium Edge"
    return "No Play"


def build_system_table(matchups: List[Matchup], system: SystemDefinition) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for m in matchups:
        d = system.evaluator(m)
        away_era = m.away_pitcher.era
        home_era = m.home_pitcher.era
        away_k = m.away_pitcher.strikeouts
        home_k = m.home_pitcher.strikeouts
        away_whip = m.away_pitcher.whip
        home_whip = m.home_pitcher.whip
        away_gs = m.away_pitcher.games_started
        home_gs = m.home_pitcher.games_started
        away_ip = m.away_pitcher.innings_pitched
        home_ip = m.home_pitcher.innings_pitched
        era_diff = None
        if away_era is not None and home_era is not None:
            era_diff = round(abs(home_era - away_era), 2)

        bside = better_side(m)
        better_sample = pitcher_sample_score(side_pitcher(m, bside)) if bside else 0.0
        rows.append(
            {
                "System": system.name,
                "Matchup": f"{m.away_team} @ {m.home_team}",
                "Away Team": m.away_team,
                "Home Team": m.home_team,
                "Away Pitcher": m.away_pitcher.name,
                "Home Pitcher": m.home_pitcher.name,
                "Away ERA": away_era,
                "Home ERA": home_era,
                "Away WHIP": away_whip,
                "Home WHIP": home_whip,
                "Away K": away_k,
                "Home K": home_k,
                "Away GS": away_gs,
                "Home GS": home_gs,
                "Away IP": away_ip,
                "Home IP": home_ip,
                "Sample Reliability": round(better_sample, 3),
                "ERA Diff": era_diff,
                "Edge Score": d["edge_score"],
                "Confidence": d["confidence"],
                "Suggested F5 Pick": d["pick"],
                "Qualifies Strategy": "Yes" if d["qualifies"] else "No",
                "Edge Band": band_from_conf(d["confidence"]),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    band_rank = {"Strong Edge": 2, "Medium Edge": 1, "No Play": 0}
    df["_band_rank"] = df["Edge Band"].map(band_rank).fillna(0)
    df = (
        df.sort_values(by=["_band_rank", "Confidence", "Edge Score"], ascending=[False, False, False])
        .drop(columns=["_band_rank"])
        .reset_index(drop=True)
    )
    return df


def tracker_file_path() -> str:
    return os.path.join(os.path.dirname(__file__), TRACKER_FILE)


def card_tracker_file_path() -> str:
    return os.path.join(os.path.dirname(__file__), CARD_TRACKER_FILE)


def ensure_tracker_file() -> None:
    path = tracker_file_path()
    if os.path.exists(path):
        return
    cols = [
        "bet_date",
        "system_name",
        "matchup",
        "away_team",
        "home_team",
        "pick_team",
        "suggested_pick",
        "confidence",
        "edge_score",
        "status",
        "result_note",
        "logged_at",
        "resolved_at",
    ]
    pd.DataFrame(columns=cols).to_csv(path, index=False)


def ensure_card_tracker_file() -> None:
    path = card_tracker_file_path()
    if os.path.exists(path):
        return
    cols = [
        "bet_date",
        "system_name",
        "matchup",
        "away_team",
        "home_team",
        "pick_team",
        "suggested_pick",
        "confidence",
        "edge_score",
        "status",
        "result_note",
        "logged_at",
        "resolved_at",
    ]
    pd.DataFrame(columns=cols).to_csv(path, index=False)


def load_tracker() -> pd.DataFrame:
    ensure_tracker_file()
    path = tracker_file_path()
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.DataFrame()
    if df.empty:
        return pd.DataFrame(
            columns=[
                "bet_date",
                "system_name",
                "matchup",
                "away_team",
                "home_team",
                "pick_team",
                "suggested_pick",
                "confidence",
                "edge_score",
                "status",
                "result_note",
                "logged_at",
                "resolved_at",
            ]
        )
    text_cols = [
        "bet_date",
        "system_name",
        "matchup",
        "away_team",
        "home_team",
        "pick_team",
        "suggested_pick",
        "status",
        "result_note",
        "logged_at",
        "resolved_at",
    ]
    for c in text_cols:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("").astype(str)
    for c in ["confidence", "edge_score"]:
        if c not in df.columns:
            df[c] = pd.NA
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_card_tracker() -> pd.DataFrame:
    ensure_card_tracker_file()
    path = card_tracker_file_path()
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.DataFrame()
    if df.empty:
        return pd.DataFrame(
            columns=[
                "bet_date",
                "system_name",
                "matchup",
                "away_team",
                "home_team",
                "pick_team",
                "suggested_pick",
                "confidence",
                "edge_score",
                "status",
                "result_note",
                "logged_at",
                "resolved_at",
            ]
        )
    text_cols = [
        "bet_date",
        "system_name",
        "matchup",
        "away_team",
        "home_team",
        "pick_team",
        "suggested_pick",
        "status",
        "result_note",
        "logged_at",
        "resolved_at",
    ]
    for c in text_cols:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("").astype(str)
    for c in ["confidence", "edge_score"]:
        if c not in df.columns:
            df[c] = pd.NA
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def save_tracker(df: pd.DataFrame) -> None:
    df.to_csv(tracker_file_path(), index=False)


def save_card_tracker(df: pd.DataFrame) -> None:
    df.to_csv(card_tracker_file_path(), index=False)


@st.cache_data(ttl=60 * 5)
def fetch_scores_for_date(game_date: dt.date) -> Dict[str, Dict[str, Any]]:
    params = {"sportId": 1, "date": game_date.strftime("%Y-%m-%d"), "hydrate": "linescore,team"}
    r = requests.get(MLB_STATS_SCHEDULE_URL, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    data = r.json()

    out: Dict[str, Dict[str, Any]] = {}
    for date_node in data.get("dates", []):
        for game in date_node.get("games", []):
            away_team = norm_name(game.get("teams", {}).get("away", {}).get("team", {}).get("name"), "Away")
            home_team = norm_name(game.get("teams", {}).get("home", {}).get("team", {}).get("name"), "Home")
            key = f"{canonical_team_key(away_team)}|{canonical_team_key(home_team)}"
            status = game.get("status", {}).get("detailedState") or "Unknown"
            linescore = game.get("linescore", {}) or {}
            innings = linescore.get("innings") or []
            if not isinstance(innings, list):
                innings = []
            current_inning = to_int(linescore.get("currentInning"))
            inning_state = str(linescore.get("inningState") or "").lower()
            away_total = to_int(linescore.get("teams", {}).get("away", {}).get("runs"))
            home_total = to_int(linescore.get("teams", {}).get("home", {}).get("runs"))

            away_f5 = 0
            home_f5 = 0
            innings_count = min(5, len(innings))
            first5_complete = True
            for i in range(innings_count):
                inn = innings[i] if isinstance(innings[i], dict) else {}
                ar = to_int(inn.get("away", {}).get("runs"))
                hr = to_int(inn.get("home", {}).get("runs"))
                if ar is None or hr is None:
                    first5_complete = False
                away_f5 += int(ar or 0)
                home_f5 += int(hr or 0)

            final_status = status.lower() in {"final", "game over", "completed early", "completed early: rain"}
            can_grade_live = (
                len(innings) >= 5
                and first5_complete
                and (
                    (current_inning is not None and current_inning > 5)
                    or (current_inning == 5 and inning_state == "end")
                )
            )
            can_grade = (len(innings) >= 5 and first5_complete and final_status) or can_grade_live
            out[key] = {
                "away_team": away_team,
                "home_team": home_team,
                "status": status,
                "away_f5": away_f5,
                "home_f5": home_f5,
                "away_total": int(away_total or 0),
                "home_total": int(home_total or 0),
                "current_inning": current_inning,
                "inning_state": inning_state,
                "can_grade": can_grade,
                "is_final": final_status,
            }
    return out


def grade_f5_bets(tracker_df: pd.DataFrame) -> pd.DataFrame:
    if tracker_df.empty:
        return tracker_df
    out = tracker_df.copy()
    for c in ["status", "result_note", "resolved_at", "away_team", "home_team", "pick_team", "bet_date"]:
        if c not in out.columns:
            out[c] = ""
        out[c] = out[c].fillna("").astype(str)

    now_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pending_mask = out["status"].str.lower().eq("open")
    graded_recent_mask = out["status"].str.lower().isin({"win", "loss", "push"})
    recent_dates = set((dt.date.today() - dt.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(2))
    recent_graded_idxs = out.index[graded_recent_mask & out["bet_date"].isin(recent_dates)]
    candidate_dates = set(out.loc[pending_mask, "bet_date"])
    candidate_dates.update(set(out.loc[recent_graded_idxs, "bet_date"]))
    if not candidate_dates:
        return out

    for bet_date in sorted(candidate_dates):
        try:
            date_obj = dt.datetime.strptime(bet_date, "%Y-%m-%d").date()
        except Exception:
            continue
        try:
            score_map = fetch_scores_for_date(date_obj)
        except Exception as exc:
            idxs = out.index[(out["bet_date"] == bet_date) & out["status"].str.lower().eq("open")]
            for idx in idxs:
                out.at[idx, "result_note"] = f"Score check failed: {str(exc)[:120]}"
            continue

        idxs = out.index[out["bet_date"] == bet_date]
        for idx in idxs:
            current_status = str(out.at[idx, "status"]).lower()
            away_team = str(out.at[idx, "away_team"])
            home_team = str(out.at[idx, "home_team"])
            pick_team = str(out.at[idx, "pick_team"])
            key = f"{canonical_team_key(away_team)}|{canonical_team_key(home_team)}"
            game = score_map.get(key)
            if not game:
                if current_status == "open":
                    out.at[idx, "result_note"] = "Awaiting game match in score feed."
                continue

            status_text = str(game.get("status", "")).lower()
            if "postponed" in status_text or "suspended" in status_text or "cancelled" in status_text:
                out.at[idx, "status"] = "void"
                out.at[idx, "result_note"] = f"Game status: {game.get('status')}"
                out.at[idx, "resolved_at"] = now_str
                continue

            if not game.get("can_grade"):
                if current_status in {"win", "loss", "push"} and not game.get("is_final", False):
                    out.at[idx, "status"] = "open"
                    out.at[idx, "result_note"] = "Reopened: game still in progress (F5 not complete)."
                    out.at[idx, "resolved_at"] = ""
                continue

            away_f5 = int(game.get("away_f5", 0))
            home_f5 = int(game.get("home_f5", 0))
            pick_is_away = canonical_team_key(pick_team) == canonical_team_key(game.get("away_team", ""))
            pick_runs = away_f5 if pick_is_away else home_f5
            opp_runs = home_f5 if pick_is_away else away_f5
            result = "win" if pick_runs > opp_runs else ("loss" if pick_runs < opp_runs else "push")
            out.at[idx, "status"] = result
            out.at[idx, "result_note"] = f"F5 score {game.get('away_team')} {away_f5} - {game.get('home_team')} {home_f5}"
            out.at[idx, "resolved_at"] = now_str
    return out


def add_bets_to_tracker_for_date(
    system_table: pd.DataFrame,
    system_name: str,
    tracker_df: pd.DataFrame,
    bet_date: dt.date,
) -> Tuple[pd.DataFrame, int]:
    if system_table.empty:
        return tracker_df, 0
    bet_date_str = bet_date.strftime("%Y-%m-%d")
    bets = system_table[system_table["Qualifies Strategy"] == "Yes"].copy()
    if bets.empty:
        return tracker_df, 0
    out = tracker_df.copy()
    added = 0
    now_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for _, row in bets.iterrows():
        matchup = str(row["Matchup"])
        suggested = str(row["Suggested F5 Pick"])
        pick_team = suggested.replace("F5 ", "").strip()
        away_team = matchup.split(" @ ")[0].strip() if " @ " in matchup else ""
        home_team = matchup.split(" @ ")[1].strip() if " @ " in matchup else ""
        dup = (
            (out["bet_date"].astype(str) == bet_date_str)
            & (out["system_name"].astype(str) == system_name)
            & (out["matchup"].astype(str) == matchup)
            & (out["suggested_pick"].astype(str) == suggested)
        )
        if dup.any():
            continue
        out = pd.concat(
            [
                out,
                pd.DataFrame(
                    [
                        {
                            "bet_date": bet_date_str,
                            "system_name": system_name,
                            "matchup": matchup,
                            "away_team": away_team,
                            "home_team": home_team,
                            "pick_team": pick_team,
                            "suggested_pick": suggested,
                            "confidence": float(row["Confidence"]) if pd.notna(row["Confidence"]) else None,
                            "edge_score": float(row["Edge Score"]) if pd.notna(row["Edge Score"]) else None,
                            "status": "open",
                            "result_note": "",
                            "logged_at": now_str,
                            "resolved_at": "",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        added += 1
    return out, added


def add_today_bets_to_tracker(system_table: pd.DataFrame, system_name: str, tracker_df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    return add_bets_to_tracker_for_date(system_table, system_name, tracker_df, dt.date.today())


def add_card_bets_to_tracker_for_date(
    card_df: pd.DataFrame,
    card_name: str,
    card_tracker_df: pd.DataFrame,
    bet_date: dt.date,
) -> Tuple[pd.DataFrame, int]:
    if card_df.empty:
        return card_tracker_df, 0
    bet_date_str = bet_date.strftime("%Y-%m-%d")
    out = card_tracker_df.copy()
    added = 0
    now_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for _, row in card_df.iterrows():
        matchup = str(row.get("Matchup", ""))
        suggested = str(row.get("Suggested F5 Pick", ""))
        if not matchup or not suggested or suggested == "No Play":
            continue
        pick_team = suggested.replace("F5 ", "").strip()
        away_team = matchup.split(" @ ")[0].strip() if " @ " in matchup else ""
        home_team = matchup.split(" @ ")[1].strip() if " @ " in matchup else ""
        conf = row.get("Avg Confidence", row.get("Confidence", None))
        edge = row.get("Consensus Score", row.get("Backtest Pick Score", row.get("Priority Score", row.get("Edge Score", None))))
        dup = (
            (out["bet_date"].astype(str) == bet_date_str)
            & (out["system_name"].astype(str) == f"CARD | {card_name}")
            & (out["matchup"].astype(str) == matchup)
            & (out["suggested_pick"].astype(str) == suggested)
        )
        if dup.any():
            continue
        out = pd.concat(
            [
                out,
                pd.DataFrame(
                    [
                        {
                            "bet_date": bet_date_str,
                            "system_name": f"CARD | {card_name}",
                            "matchup": matchup,
                            "away_team": away_team,
                            "home_team": home_team,
                            "pick_team": pick_team,
                            "suggested_pick": suggested,
                            "confidence": float(conf) if pd.notna(conf) else None,
                            "edge_score": float(edge) if pd.notna(edge) else None,
                            "status": "open",
                            "result_note": "",
                            "logged_at": now_str,
                            "resolved_at": "",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        added += 1
    return out, added


def build_missing_tracking_dates(tracker_df: pd.DataFrame, max_backfill_days: int = 30) -> List[dt.date]:
    today = dt.date.today()
    if tracker_df.empty or "bet_date" not in tracker_df.columns:
        return [today]

    parsed_dates = pd.to_datetime(tracker_df["bet_date"], errors="coerce").dt.date.dropna()
    if parsed_dates.empty:
        return [today]

    last_date = max(parsed_dates)
    start_date = last_date + dt.timedelta(days=1)
    if start_date > today:
        return [today]

    days = (today - start_date).days + 1
    if days <= 0:
        return [today]

    if days > max_backfill_days:
        start_date = today - dt.timedelta(days=max_backfill_days - 1)

    return [start_date + dt.timedelta(days=i) for i in range((today - start_date).days + 1)]


MATCHUP_ROW_RGB: List[Tuple[int, int, int]] = [
    (56, 189, 248),
    (167, 139, 250),
    (251, 191, 36),
    (251, 113, 133),
    (52, 211, 153),
    (45, 212, 191),
    (251, 146, 60),
    (163, 230, 53),
    (232, 121, 249),
    (129, 140, 248),
    (252, 211, 77),
    (192, 132, 252),
]


def matchup_row_tone_index(matchup: str) -> int:
    """Same bucket as Command Center cards (must match `matchup_tone_key` logic)."""
    m = " ".join(str(matchup).split()).strip()
    if " @ " not in m:
        k = norm_key(m)
    else:
        away, home = m.split(" @ ", 1)
        k = f"{canonical_team_key(away.strip())}|{canonical_team_key(home.strip())}"
    if not k:
        return 0
    return int(hashlib.md5(k.encode("utf-8")).hexdigest()[:8], 16) % 12


def row_color(row: pd.Series) -> List[str]:
    """Full system board table: matchup tint across the whole row (same hues as slip cards)."""
    m = str(row.get("Matchup", "") or "")
    idx = matchup_row_tone_index(m)
    R, G, B = MATCHUP_ROW_RGB[idx]
    has_era = "Away ERA" in row.index and "Home ERA" in row.index
    no_pitch_stats = (
        has_era and pd.isna(row.get("Away ERA")) and pd.isna(row.get("Home ERA"))
    )
    band = row.get("Edge Band", "No Play") if "Edge Band" in row.index else "No Play"
    if no_pitch_stats:
        alpha = 0.14
    elif band == "Strong Edge":
        alpha = 0.34
    elif band == "Medium Edge":
        alpha = 0.28
    else:
        alpha = 0.24
    style = (
        f"background-color: rgba({R},{G},{B},{alpha}); color: #0f172a; "
        f"border-left: 4px solid rgb({R},{G},{B});"
    )
    return [style] * len(row)


def row_color_matchup_only(row: pd.Series) -> List[str]:
    """Qualifier subset (may omit ERA cols): full matchup wash only."""
    m = str(row.get("Matchup", "") or "")
    idx = matchup_row_tone_index(m)
    R, G, B = MATCHUP_ROW_RGB[idx]
    style = (
        f"background-color: rgba({R},{G},{B},0.3); color: #0f172a; "
        f"border-left: 4px solid rgb({R},{G},{B});"
    )
    return [style] * len(row)


def summary_from_tracker(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {"wins": 0, "losses": 0, "pushes": 0, "open": 0, "graded": 0, "win_rate": 0.0, "record": "0-0-0"}
    status = df["status"].astype(str).str.lower()
    wins = int((status == "win").sum())
    losses = int((status == "loss").sum())
    pushes = int((status == "push").sum())
    open_bets = int((status == "open").sum())
    graded = wins + losses
    win_rate = (wins / graded * 100.0) if graded > 0 else 0.0
    return {
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "open": open_bets,
        "graded": graded,
        "win_rate": win_rate,
        "record": f"{wins}-{losses}-{pushes}",
    }


def build_leaderboard(tracker_df: pd.DataFrame) -> pd.DataFrame:
    if tracker_df.empty:
        return pd.DataFrame(columns=["System", "Record", "Win Rate %", "Graded", "Open", "Total Bets"])
    rows: List[Dict[str, Any]] = []
    for system_name, g in tracker_df.groupby("system_name"):
        s = summary_from_tracker(g)
        rows.append(
            {
                "System": system_name,
                "Record": s["record"],
                "Win Rate %": round(s["win_rate"], 1),
                "Graded": s["graded"],
                "Open": s["open"],
                "Total Bets": len(g),
            }
        )
    lb = pd.DataFrame(rows)
    if lb.empty:
        return lb
    return lb.sort_values(by=["Win Rate %", "Graded", "Total Bets"], ascending=[False, False, False]).reset_index(drop=True)


def scale_status(graded: int, win_rate: float) -> Tuple[str, str, str]:
    if graded < 30:
        return "Sandbox", "Collect sample", "0.25u"
    if graded < 50:
        if win_rate >= 56.0:
            return "Pilot", "Early green light", "0.50u"
        return "Sandbox", "Needs stronger edge", "0.25u"
    if graded < 100:
        if win_rate >= 55.0:
            return "Scale Candidate", "Stable enough to test scale", "0.75u"
        if win_rate >= 53.0:
            return "Pilot", "Hold and monitor", "0.50u"
        return "Sandbox", "Under threshold", "0.25u"
    if win_rate >= 54.0:
        return "Scale", "Large sample + edge", "1.00u"
    if win_rate >= 52.5:
        return "Pilot", "Near threshold", "0.50u"
    return "Sandbox", "Long sample but weak edge", "0.25u"


def build_scale_table(tracker_df: pd.DataFrame) -> pd.DataFrame:
    lb = build_leaderboard(tracker_df)
    if lb.empty:
        return pd.DataFrame(columns=["System", "Status", "Action", "Stake", "Record", "Win Rate %", "Graded", "Open", "Total Bets"])

    rows: List[Dict[str, Any]] = []
    for _, row in lb.iterrows():
        graded = int(row["Graded"])
        win_rate = float(row["Win Rate %"])
        status, action, stake = scale_status(graded, win_rate)
        rows.append(
            {
                "System": row["System"],
                "Status": status,
                "Action": action,
                "Stake": stake,
                "Record": row["Record"],
                "Win Rate %": win_rate,
                "Graded": graded,
                "Open": int(row["Open"]),
                "Total Bets": int(row["Total Bets"]),
            }
        )

    out = pd.DataFrame(rows)
    status_rank = {"Scale": 3, "Scale Candidate": 2, "Pilot": 1, "Sandbox": 0}
    out["_rank"] = out["Status"].map(status_rank).fillna(0)
    out = out.sort_values(by=["_rank", "Win Rate %", "Graded"], ascending=[False, False, False]).drop(columns=["_rank"])
    return out.reset_index(drop=True)


def build_confidence_performance_table(tracker_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "Bucket",
        "Wins",
        "Losses",
        "Pushes",
        "Graded",
        "Win Rate %",
        "Net Units (1u flat)",
        "ROI %",
    ]
    if tracker_df.empty:
        return pd.DataFrame(columns=cols)

    df = tracker_df.copy()
    if "confidence" not in df.columns or "status" not in df.columns:
        return pd.DataFrame(columns=cols)

    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["status"] = df["status"].astype(str).str.lower()
    graded_df = df[df["status"].isin(["win", "loss", "push"])].copy()
    if graded_df.empty:
        return pd.DataFrame(columns=cols)

    # 1 unit risked per graded bet; win returns +0.909u (approx -110 pricing), loss -1u, push 0.
    def row_for_bucket(label: str, min_conf: float) -> Dict[str, Any]:
        b = graded_df[graded_df["confidence"] >= min_conf]
        wins = int((b["status"] == "win").sum())
        losses = int((b["status"] == "loss").sum())
        pushes = int((b["status"] == "push").sum())
        graded = wins + losses
        win_rate = (wins / graded * 100.0) if graded > 0 else 0.0
        net_units = wins * 0.909 - losses
        total_risked = len(b)
        roi = (net_units / total_risked * 100.0) if total_risked > 0 else 0.0
        return {
            "Bucket": label,
            "Wins": wins,
            "Losses": losses,
            "Pushes": pushes,
            "Graded": graded,
            "Win Rate %": round(win_rate, 1),
            "Net Units (1u flat)": round(net_units, 2),
            "ROI %": round(roi, 2),
        }

    out = pd.DataFrame(
        [
            row_for_bucket("All Tracked", 0.0),
            row_for_bucket("80%+", 80.0),
            row_for_bucket("90%+", 90.0),
            row_for_bucket("100%", 100.0),
        ]
    )
    return out


def grade_pick_from_score_map(matchup: str, suggested_pick: str, score_map: Dict[str, Dict[str, Any]]) -> Optional[str]:
    away_team = matchup.split(" @ ")[0].strip() if " @ " in matchup else ""
    home_team = matchup.split(" @ ")[1].strip() if " @ " in matchup else ""
    key = f"{canonical_team_key(away_team)}|{canonical_team_key(home_team)}"
    game = score_map.get(key)
    if not game:
        return None
    if not game.get("can_grade"):
        return None
    away_f5 = int(game.get("away_f5", 0))
    home_f5 = int(game.get("home_f5", 0))
    pick_team = suggested_pick.replace("F5 ", "").strip()
    pick_is_away = canonical_team_key(pick_team) == canonical_team_key(game.get("away_team", ""))
    pick_runs = away_f5 if pick_is_away else home_f5
    opp_runs = home_f5 if pick_is_away else away_f5
    if pick_runs > opp_runs:
        return "win"
    if pick_runs < opp_runs:
        return "loss"
    return "push"


@st.cache_data(ttl=60 * 60 * 8, show_spinner=False)
def summarize_backtest_stats(stats: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for system_name, s in stats.items():
        wins = int(s["wins"])
        losses = int(s["losses"])
        pushes = int(s["pushes"])
        graded = int(s["graded"])
        qualified = int(s["qualified"])
        win_rate = (wins / (wins + losses) * 100.0) if (wins + losses) > 0 else 0.0
        net_units = wins * 0.909 - losses
        roi = (net_units / (wins + losses) * 100.0) if (wins + losses) > 0 else 0.0
        rows.append(
            {
                "System": system_name,
                "Wins": wins,
                "Losses": losses,
                "Pushes": pushes,
                "Graded": graded,
                "Qualified": qualified,
                "Win Rate %": round(win_rate, 1),
                "Net Units (1u flat)": round(net_units, 2),
                "ROI %": round(roi, 2),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        by=["ROI %", "Net Units (1u flat)", "Win Rate %", "Graded"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


@st.cache_data(ttl=60 * 60 * 8, show_spinner=False)
def run_system_backtest_multi() -> Dict[str, pd.DataFrame]:
    max_years = 3
    end_date = dt.date.today() - dt.timedelta(days=1)
    start_date = end_date - dt.timedelta(days=(365 * max_years) - 1)
    system_names = [s.name for s in SYSTEMS]
    windows = {"1Y": 365, "2Y": 730, "3Y": 1095}
    cutoff_dates = {k: end_date - dt.timedelta(days=v - 1) for k, v in windows.items()}
    stats_by_window: Dict[str, Dict[str, Dict[str, float]]] = {
        label: {
            name: {"wins": 0, "losses": 0, "pushes": 0, "qualified": 0, "graded": 0}
            for name in system_names
        }
        for label in windows
    }

    day_count = (end_date - start_date).days + 1
    for day_offset in range(day_count):
        game_date = start_date + dt.timedelta(days=day_offset)
        try:
            matchups = parse_from_stats_api(game_date)
        except Exception:
            continue
        if not matchups:
            continue
        try:
            score_map = fetch_scores_for_date(game_date)
        except Exception:
            continue
        active_windows = [label for label, cutoff in cutoff_dates.items() if game_date >= cutoff]
        if not active_windows:
            continue
        for s in SYSTEMS:
            sys_table = build_system_table(matchups, s)
            if sys_table.empty:
                continue
            bets = sys_table[sys_table["Qualifies Strategy"] == "Yes"]
            if bets.empty:
                continue
            for _, row in bets.iterrows():
                result = grade_pick_from_score_map(str(row["Matchup"]), str(row["Suggested F5 Pick"]), score_map)
                for label in active_windows:
                    stats = stats_by_window[label][s.name]
                    stats["qualified"] += 1
                    if result is None:
                        continue
                    stats["graded"] += 1
                    if result == "win":
                        stats["wins"] += 1
                    elif result == "loss":
                        stats["losses"] += 1
                    else:
                        stats["pushes"] += 1

    return {label: summarize_backtest_stats(stats_by_window[label]) for label in windows}


def load_backtest_cache() -> Tuple[Dict[str, pd.DataFrame], Optional[str]]:
    if not os.path.exists(BACKTEST_CACHE_FILE):
        return {}, None
    try:
        with open(BACKTEST_CACHE_FILE, "r", encoding="utf-8") as f:
            payload = json.load(f)
        windows = payload.get("windows", {})
        generated_at = payload.get("generated_at")
        out: Dict[str, pd.DataFrame] = {}
        for label in ["1Y", "2Y", "3Y"]:
            rows = windows.get(label, [])
            out[label] = pd.DataFrame(rows) if isinstance(rows, list) else pd.DataFrame()
        return out, generated_at
    except Exception:
        return {}, None


def save_backtest_cache(backtest_multi: Dict[str, pd.DataFrame]) -> None:
    payload = {
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "windows": {
            label: df.to_dict(orient="records") if isinstance(df, pd.DataFrame) else []
            for label, df in backtest_multi.items()
        },
    }
    with open(BACKTEST_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def system_quality_weights(tracker_df: pd.DataFrame) -> Dict[str, float]:
    lb = build_leaderboard(tracker_df)
    if lb.empty:
        return {}
    weights: Dict[str, float] = {}
    for _, row in lb.iterrows():
        system = str(row["System"])
        graded = int(row["Graded"])
        win_rate = float(row["Win Rate %"])
        # Conservative quality weight: starts near 1.0 and scales with sample + win rate.
        sample_factor = min(1.0, graded / 120.0)
        edge_factor = max(0.75, min(1.35, 1.0 + (win_rate - 52.0) / 40.0))
        weights[system] = round((0.75 + 0.45 * sample_factor) * edge_factor, 4)
    return weights


def build_consensus_bets(system_tables: Dict[str, pd.DataFrame], tracker_df: pd.DataFrame) -> pd.DataFrame:
    system_map = {s.key: s.name for s in SYSTEMS}
    quality = system_quality_weights(tracker_df)
    rows: List[Dict[str, Any]] = []

    for sys_key, df in system_tables.items():
        if df.empty:
            continue
        sys_name = system_map.get(sys_key, sys_key)
        q_df = df[(df["Qualifies Strategy"] == "Yes") & (df["Suggested F5 Pick"] != "No Play")]
        for _, r in q_df.iterrows():
            matchup = str(r["Matchup"])
            pick = str(r["Suggested F5 Pick"])
            edge = float(r["Edge Score"]) if pd.notna(r["Edge Score"]) else 0.0
            conf = float(r["Confidence"]) if pd.notna(r["Confidence"]) else 0.0
            sample_rel = float(r["Sample Reliability"]) if pd.notna(r.get("Sample Reliability")) else 0.0
            weight = quality.get(sys_name, 1.0)
            rows.append(
                {
                    "Matchup": matchup,
                    "Suggested F5 Pick": pick,
                    "System": sys_name,
                    "Edge": edge,
                    "Confidence": conf,
                    "SampleReliability": sample_rel,
                    "Weight": weight,
                    "WeightedSignal": max(0.0, conf * weight * (0.75 + 0.5 * sample_rel)),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "Matchup",
                "Suggested F5 Pick",
                "Models On Bet",
                "Avg Confidence",
                "Avg Edge",
                "Avg Sample",
                "Consensus Score",
                "Systems",
            ]
        )

    d = pd.DataFrame(rows)
    grouped = (
        d.groupby(["Matchup", "Suggested F5 Pick"], as_index=False)
        .agg(
            models_on_bet=("System", "nunique"),
            avg_confidence=("Confidence", "mean"),
            avg_edge=("Edge", "mean"),
            avg_sample=("SampleReliability", "mean"),
            consensus_score=("WeightedSignal", "sum"),
            systems=("System", lambda s: ", ".join(sorted(set(s)))),
        )
        .rename(
            columns={
                "models_on_bet": "Models On Bet",
                "avg_confidence": "Avg Confidence",
                "avg_edge": "Avg Edge",
                "avg_sample": "Avg Sample",
                "consensus_score": "Consensus Score",
                "systems": "Systems",
            }
        )
    )
    return grouped.sort_values(by=["Models On Bet", "Consensus Score", "Avg Confidence"], ascending=[False, False, False]).reset_index(drop=True)


def build_official_card(consensus_df: pd.DataFrame, max_bets: int) -> pd.DataFrame:
    if consensus_df.empty:
        return consensus_df
    card = consensus_df.copy()
    card = card[
        (card["Models On Bet"] >= 2)
        & (card["Avg Confidence"] >= 68)
        & (card["Avg Sample"] >= 0.45)
    ].copy()
    if card.empty:
        return card
    card = card.sort_values(
        by=["Models On Bet", "Consensus Score", "Avg Confidence", "Avg Sample"],
        ascending=[False, False, False, False],
    ).head(max_bets)
    return card.reset_index(drop=True)


def build_confidence_cards(consensus_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if consensus_df.empty:
        empty = pd.DataFrame(columns=consensus_df.columns)
        return empty, empty, empty

    sort_cols = ["Models On Bet", "Consensus Score", "Avg Confidence", "Avg Sample"]
    c80 = consensus_df[consensus_df["Avg Confidence"] >= 80].copy()
    c90 = consensus_df[consensus_df["Avg Confidence"] >= 90].copy()
    c100 = build_100_confidence_card(consensus_df)

    c80 = c80.sort_values(by=sort_cols, ascending=[False, False, False, False]).reset_index(drop=True)
    c90 = c90.sort_values(by=sort_cols, ascending=[False, False, False, False]).reset_index(drop=True)
    c100 = c100.sort_values(by=sort_cols, ascending=[False, False, False, False]).reset_index(drop=True)
    return c80, c90, c100


def build_top_pick_per_model(system_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    system_name_map = {s.key: s.name for s in SYSTEMS}
    for sys_key, df in system_tables.items():
        if df.empty:
            continue
        bets = df[df["Qualifies Strategy"] == "Yes"].copy()
        if bets.empty:
            continue
        top_bet = bets.sort_values(by=["Confidence", "Edge Score"], ascending=[False, False]).head(1)
        if top_bet.empty:
            continue
        r = top_bet.iloc[0]
        rows.append(
            {
                "System": system_name_map.get(sys_key, sys_key),
                "Matchup": r["Matchup"],
                "Suggested F5 Pick": r["Suggested F5 Pick"],
                "Confidence": r["Confidence"],
                "Edge Score": r["Edge Score"],
                "Sample Reliability": r["Sample Reliability"],
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "System",
                "Matchup",
                "Suggested F5 Pick",
                "Confidence",
                "Edge Score",
                "Sample Reliability",
            ]
        )
    out = pd.DataFrame(rows)
    return out.sort_values(by=["Confidence", "Edge Score"], ascending=[False, False]).reset_index(drop=True)


def build_sharp_model_list(tracker_df: pd.DataFrame, max_models: int = 5) -> List[str]:
    lb = build_leaderboard(tracker_df)
    if lb.empty:
        return []
    sharp_lb = lb[(lb["Graded"] >= 20) & (lb["Win Rate %"] >= 55.0)].copy()
    if sharp_lb.empty:
        sharp_lb = lb.head(max_models).copy()
    return sharp_lb.head(max_models)["System"].astype(str).tolist()


def build_sharp_consensus_card(consensus_df: pd.DataFrame, sharp_models: List[str], max_bets: int = 8) -> pd.DataFrame:
    if consensus_df.empty:
        return consensus_df
    card = consensus_df.copy()
    card["Sharp Model Count"] = (
        card["Systems"]
        .astype(str)
        .apply(lambda s: sum(1 for model in sharp_models if model in s))
    )
    card = card[
        ((card["Models On Bet"] >= 2) & (card["Sharp Model Count"] >= 1))
        | (card["Sharp Model Count"] >= 2)
    ]
    card = card[
        (card["Avg Confidence"] >= 70)
        & (card["Avg Sample"] >= 0.45)
    ].copy()
    if card.empty:
        return card
    card = card.sort_values(
        by=["Sharp Model Count", "Models On Bet", "Consensus Score", "Avg Confidence", "Avg Sample"],
        ascending=[False, False, False, False, False],
    ).head(max_bets)
    return card.reset_index(drop=True)


def build_sharp_model_picks(system_tables: Dict[str, pd.DataFrame], sharp_models: List[str], max_rows: int = 8) -> pd.DataFrame:
    all_top = build_top_pick_per_model(system_tables)
    if all_top.empty:
        return all_top
    if not sharp_models:
        return all_top.head(max_rows).reset_index(drop=True)
    sharp_only = all_top[all_top["System"].isin(sharp_models)].copy()
    if sharp_only.empty:
        return all_top.head(max_rows).reset_index(drop=True)
    return sharp_only.head(max_rows).reset_index(drop=True)


def build_bet_this_first_card(
    sharp_consensus_card: pd.DataFrame,
    backtest_lookup: Dict[str, Dict[str, Any]],
    max_bets: Optional[int] = None,
) -> pd.DataFrame:
    if sharp_consensus_card.empty:
        return sharp_consensus_card

    card = sharp_consensus_card.copy()
    rows: List[Dict[str, Any]] = []
    has_backtest = len(backtest_lookup) > 0
    for _, r in card.iterrows():
        systems = [s.strip() for s in str(r.get("Systems", "")).split(",") if s.strip()]
        rois = []
        for sys_name in systems:
            bt = backtest_lookup.get(sys_name)
            if bt is None:
                continue
            try:
                rois.append(float(bt.get("ROI %", 0.0)))
            except Exception:
                continue
        bt_agree = float(sum(rois) / len(rois)) if rois else 0.0
        bt_models = len(rois)
        live_signal = (
            float(r.get("Consensus Score", 0.0)) * 0.18
            + float(r.get("Avg Confidence", 0.0)) * 2.2
            + float(r.get("Sharp Model Count", 0.0)) * 28.0
            + float(r.get("Models On Bet", 0.0)) * 8.0
            + float(r.get("Avg Sample", 0.0)) * 40.0
        )
        priority_score = live_signal + (bt_agree * 10.0 if has_backtest else 0.0)
        rows.append(
            {
                **r.to_dict(),
                "Backtest ROI Agree %": round(bt_agree, 2),
                "Backtest Models": bt_models,
                "Priority Score": round(priority_score, 1),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(
        by=["Priority Score", "Backtest ROI Agree %", "Sharp Model Count", "Avg Confidence"],
        ascending=[False, False, False, False],
    )
    if max_bets is not None:
        out = out.head(max_bets)
    return out.reset_index(drop=True)


def build_backtest_only_bets(
    system_tables: Dict[str, pd.DataFrame],
    backtest_1y: pd.DataFrame,
    backtest_2y: pd.DataFrame,
    backtest_3y: pd.DataFrame,
) -> pd.DataFrame:
    if backtest_1y.empty:
        return pd.DataFrame()

    model_scores = backtest_1y[["System", "ROI %", "Graded"]].rename(
        columns={"ROI %": "ROI1", "Graded": "Graded1"}
    )
    if not backtest_2y.empty:
        model_scores = model_scores.merge(
            backtest_2y[["System", "ROI %"]].rename(columns={"ROI %": "ROI2"}),
            on="System",
            how="left",
        )
    else:
        model_scores["ROI2"] = 0.0
    if not backtest_3y.empty:
        model_scores = model_scores.merge(
            backtest_3y[["System", "ROI %"]].rename(columns={"ROI %": "ROI3"}),
            on="System",
            how="left",
        )
    else:
        model_scores["ROI3"] = 0.0

    model_scores["ROI2"] = pd.to_numeric(model_scores["ROI2"], errors="coerce").fillna(0.0)
    model_scores["ROI3"] = pd.to_numeric(model_scores["ROI3"], errors="coerce").fillna(0.0)
    model_scores["ROI1"] = pd.to_numeric(model_scores["ROI1"], errors="coerce").fillna(0.0)
    model_scores["Graded1"] = pd.to_numeric(model_scores["Graded1"], errors="coerce").fillna(0.0)
    model_scores["BacktestStrength"] = model_scores["ROI1"] * 0.6 + model_scores["ROI2"] * 0.3 + model_scores["ROI3"] * 0.1

    # Backtest-only gate: positive recent ROI and decent sample.
    qualified_models = model_scores[
        (model_scores["ROI1"] > 0)
        & (model_scores["BacktestStrength"] > 0)
        & (model_scores["Graded1"] >= 40)
    ].copy()
    if qualified_models.empty:
        return pd.DataFrame()

    allowed = set(qualified_models["System"].astype(str).tolist())
    model_name_map = {s.key: s.name for s in SYSTEMS}
    strength_map = {
        str(r["System"]): float(r["BacktestStrength"])
        for _, r in qualified_models.iterrows()
    }

    rows: List[Dict[str, Any]] = []
    for sys_key, df in system_tables.items():
        sys_name = model_name_map.get(sys_key, sys_key)
        if sys_name not in allowed or df.empty:
            continue
        bets = df[(df["Qualifies Strategy"] == "Yes") & (df["Suggested F5 Pick"] != "No Play")].copy()
        if bets.empty:
            continue
        for _, r in bets.iterrows():
            rows.append(
                {
                    "Matchup": r["Matchup"],
                    "Suggested F5 Pick": r["Suggested F5 Pick"],
                    "Model": sys_name,
                    "Confidence": float(r["Confidence"]) if pd.notna(r["Confidence"]) else 0.0,
                    "Edge Score": float(r["Edge Score"]) if pd.notna(r["Edge Score"]) else 0.0,
                    "Sample Reliability": float(r["Sample Reliability"]) if pd.notna(r["Sample Reliability"]) else 0.0,
                    "Model Backtest Strength": strength_map.get(sys_name, 0.0),
                }
            )
    if not rows:
        return pd.DataFrame()

    d = pd.DataFrame(rows)
    out = (
        d.groupby(["Matchup", "Suggested F5 Pick"], as_index=False)
        .agg(
            models_on_bet=("Model", "nunique"),
            avg_confidence=("Confidence", "mean"),
            avg_edge=("Edge Score", "mean"),
            avg_sample=("Sample Reliability", "mean"),
            bt_strength=("Model Backtest Strength", "mean"),
            models=("Model", lambda s: ", ".join(sorted(set(s)))),
        )
        .rename(
            columns={
                "models_on_bet": "Models On Bet",
                "avg_confidence": "Avg Confidence",
                "avg_edge": "Avg Edge",
                "avg_sample": "Avg Sample",
                "bt_strength": "Backtest Strength",
                "models": "Backtested Models",
            }
        )
    )
    out["Backtest Pick Score"] = (
        out["Backtest Strength"] * 8.0
        + out["Avg Confidence"] * 1.2
        + out["Models On Bet"] * 6.0
        + out["Avg Sample"] * 25.0
    )
    return out.sort_values(
        by=["Backtest Pick Score", "Backtest Strength", "Models On Bet", "Avg Confidence"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def build_100_confidence_card(consensus_df: pd.DataFrame) -> pd.DataFrame:
    if consensus_df.empty:
        return consensus_df
    # Shared strict definition for "100% confidence" across app:
    # require true consensus + reliability, not a single-model spike.
    d = consensus_df[
        (consensus_df["Avg Confidence"] >= 100)
        & (consensus_df["Models On Bet"] >= 2)
        & (consensus_df["Avg Sample"] >= 0.55)
    ].copy()
    if d.empty:
        return d
    d["100C Score"] = (
        d["Consensus Score"] * 0.35
        + d["Models On Bet"] * 12.0
        + d["Avg Sample"] * 25.0
    )
    return d.sort_values(
        by=["100C Score", "Consensus Score", "Models On Bet", "Avg Sample"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def render_app_theme() -> None:
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Lexend:wght@500;600;700;800;900&display=swap" rel="stylesheet">
        <style>
        .sa-score-ui {
            font-family: "Lexend", ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
            font-feature-settings: "tnum" 1;
        }
        .main-title {
            font-size: 2.1rem;
            font-weight: 800;
            letter-spacing: 0.3px;
            margin-bottom: 0.2rem;
        }
        .main-title.sa-app-title {
            font-size: 1.35rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin-bottom: 0.15rem;
            color: #e2e8f0;
        }
        .sa-app-sub {
            color: #64748b;
            font-size: 0.82rem;
            margin-bottom: 0.65rem;
            font-weight: 500;
        }
        .subtitle {
            color: #9ca3af;
            margin-bottom: 1rem;
        }
        .sa-sidebar-note {
            font-size: 0.72rem;
            color: #64748b;
            padding: 8px 0 12px 0;
            line-height: 1.4;
            border-bottom: 1px solid rgba(51,65,85,0.5);
            margin-bottom: 8px;
        }
        .bet-card {
            border-radius: 14px;
            padding: 14px 16px;
            background: linear-gradient(145deg, #101623 0%, #1c2436 100%);
            border: 1px solid #283247;
            margin-bottom: 10px;
        }
        .bet-card-title {
            color: #93c5fd;
            font-weight: 700;
            font-size: 0.95rem;
            margin-bottom: 6px;
        }
        .bet-card-pick {
            color: #f8fafc;
            font-size: 1.1rem;
            font-weight: 800;
            margin-bottom: 4px;
        }
        .bet-card-meta {
            color: #cbd5e1;
            font-size: 0.88rem;
        }
        .section-kicker {
            color: #34d399;
            font-weight: 800;
            font-size: 0.68rem;
            letter-spacing: 0.1em;
            margin-bottom: 2px;
            text-transform: uppercase;
        }
        .section-title {
            font-size: 0.92rem;
            font-weight: 700;
            color: #e5e7eb;
            margin-top: 0.35rem;
            margin-bottom: 0.12rem;
        }
        .section-subtitle {
            color: #94a3b8;
            font-size: 0.78rem;
            margin-bottom: 0.45rem;
            line-height: 1.35;
        }
        .panel {
            border: 1px solid #233049;
            border-radius: 12px;
            padding: 10px 12px;
            background: linear-gradient(180deg, rgba(16,22,35,0.55) 0%, rgba(16,22,35,0.2) 100%);
            margin-bottom: 0.65rem;
        }
        .sa-cc-hero {
            background: linear-gradient(135deg, #070d18 0%, #101d32 48%, #0a1424 100%);
            border: 1px solid rgba(56,189,248,0.22);
            border-radius: 14px;
            padding: 14px 16px 12px;
            margin-bottom: 14px;
            box-shadow: 0 8px 28px rgba(0,0,0,0.38), inset 0 1px 0 rgba(255,255,255,0.04);
            position: relative;
            overflow: hidden;
        }
        .sa-cc-hero::before {
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, #38bdf8, #a78bfa, #34d399);
            opacity: 0.9;
        }
        .sa-cc-hero-plain { max-width: 44rem; }
        .sa-cc-track-kv {
            display: flex;
            flex-wrap: wrap;
            align-items: baseline;
            gap: 6px 10px;
            margin-bottom: 10px;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(148,163,184,0.12);
        }
        .sa-cc-track-kv .sa-cc-k {
            font-size: 0.58rem;
            font-weight: 900;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #64748b;
            flex: 0 0 auto;
        }
        .sa-cc-track-kv .sa-cc-v {
            font-size: 0.72rem;
            font-weight: 600;
            color: #94a3b8;
            line-height: 1.35;
            flex: 1 1 200px;
        }
        .sa-cc-record-split {
            font-size: 0.68rem;
            font-weight: 700;
            color: #64748b;
            margin-top: 4px;
            font-variant-numeric: tabular-nums;
            letter-spacing: 0.04em;
        }
        .sa-cc-record-line {
            display: flex;
            flex-wrap: wrap;
            align-items: baseline;
            gap: 8px 12px;
            margin-bottom: 2px;
        }
        .sa-cc-record {
            font-size: clamp(1.45rem, 4vw, 1.85rem);
            font-weight: 900;
            letter-spacing: -1px;
            color: #f8fafc;
            line-height: 1;
            font-variant-numeric: tabular-nums;
        }
        .sa-cc-record-sub {
            font-size: 0.65rem;
            font-weight: 700;
            color: #64748b;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }
        .sa-cc-winrate-inline {
            margin-top: 6px;
            font-size: 0.72rem;
            font-weight: 700;
            color: #a5b4fc;
            font-variant-numeric: tabular-nums;
        }
        .sa-cc-hero-metrics {
            display: flex;
            flex-wrap: wrap;
            gap: 10px 18px;
            margin-top: 12px;
            padding-top: 10px;
            border-top: 1px solid rgba(148,163,184,0.14);
        }
        .sa-cc-metric {
            min-width: 72px;
        }
        .sa-cc-metric .lbl {
            color: #64748b;
            font-size: 0.62rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 2px;
            font-weight: 800;
        }
        .sa-cc-metric .val {
            color: #cbd5e1;
            font-weight: 800;
            font-size: 0.92rem;
            font-variant-numeric: tabular-nums;
        }
        .sa-cc-slip-hint {
            display: flex;
            flex-wrap: wrap;
            align-items: baseline;
            gap: 6px 10px;
            margin: 0 0 12px 0;
            padding: 8px 12px;
            border-radius: 10px;
            border: 1px solid rgba(51,65,85,0.85);
            background: rgba(15,23,42,0.65);
        }
        .sa-cc-slip-hint-k {
            font-size: 0.58rem;
            font-weight: 900;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #64748b;
        }
        .sa-cc-slip-hint-v {
            font-size: 0.72rem;
            font-weight: 600;
            color: #94a3b8;
            line-height: 1.35;
        }
        .sa-cc-duo {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin: 12px 0 14px 0;
        }
        @media (max-width: 720px) {
            .sa-cc-duo { grid-template-columns: 1fr; }
        }
        .sa-cc-panel {
            border-radius: 12px;
            border: 1px solid rgba(51,65,85,0.65);
            padding: 12px 14px;
            background: rgba(15,23,42,0.5);
            min-width: 0;
        }
        .sa-cc-panel-k {
            font-size: 0.58rem;
            font-weight: 900;
            letter-spacing: 0.11em;
            text-transform: uppercase;
            color: #64748b;
            margin-bottom: 6px;
        }
        .sa-cc-panel-v {
            font-size: clamp(1.35rem, 3.5vw, 1.75rem);
            font-weight: 900;
            color: #f8fafc;
            letter-spacing: -0.5px;
            font-variant-numeric: tabular-nums;
            line-height: 1.1;
        }
        .sa-cc-panel-sub {
            margin-top: 6px;
            font-size: 0.72rem;
            font-weight: 600;
            color: #94a3b8;
            font-variant-numeric: tabular-nums;
        }
        .sa-cc-panel-hint {
            margin-top: 8px;
            font-size: 0.62rem;
            line-height: 1.35;
            color: #64748b;
            font-weight: 500;
        }
        .sa-cc-hero-foot {
            display: flex;
            flex-wrap: wrap;
            gap: 10px 18px;
            font-size: 0.68rem;
            color: #64748b;
            padding-top: 10px;
            border-top: 1px solid rgba(148,163,184,0.12);
        }
        .sa-cc-hero-foot span b {
            color: #94a3b8;
            font-weight: 700;
        }
        .sa-score-strip-wrap {
            margin-bottom: 12px;
            overflow-x: auto;
            padding-bottom: 2px;
        }
        .sa-score-strip {
            display: flex;
            flex-wrap: nowrap;
            gap: 10px;
            min-width: min-content;
        }
        .sa-mini-board {
            flex: 0 0 auto;
            width: 200px;
            border-radius: 10px;
            border: 1px solid #2a3f63;
            background: linear-gradient(180deg, #121c2f 0%, #0d1524 100%);
            padding: 8px 10px;
            position: relative;
            overflow: hidden;
        }
        .sa-mini-board > * {
            position: relative;
            z-index: 1;
        }
        .sa-mini-ticker-title {
            font-size: 0.58rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: #64748b;
            text-align: center;
            margin-bottom: 6px;
            line-height: 1.25;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .sa-mini-fg-inline {
            font-size: 0.58rem;
            color: #94a3b8;
            text-align: center;
            margin-top: 6px;
            font-weight: 600;
            line-height: 1.3;
        }
        .sa-mini-pair-row {
            display: flex;
            justify-content: center;
            align-items: flex-end;
            gap: 10px;
            font-variant-numeric: tabular-nums;
            margin-bottom: 4px;
        }
        .sa-mini-side {
            flex: 1;
            min-width: 0;
            text-align: center;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 2px;
        }
        .sa-mini-over {
            font-size: 0.58rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            color: #94a3b8;
            text-transform: uppercase;
            line-height: 1.15;
            max-width: 100%;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .sa-mini-names {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.78rem;
            font-weight: 800;
            color: #cbd5e1;
            margin-top: 8px;
            padding-top: 8px;
            border-top: 1px solid rgba(148,163,184,0.18);
            gap: 8px;
        }
        .sa-mini-names span.tm {
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 42%;
        }
        .sa-mini-names span.at {
            color: #64748b;
            font-weight: 600;
            font-size: 0.7rem;
        }
        .sa-mini-scores {
            display: flex;
            justify-content: center;
            align-items: baseline;
            gap: 16px;
            font-variant-numeric: tabular-nums;
            margin-bottom: 6px;
        }
        .sa-mini-pair-row .n,
        .sa-mini-scores .n {
            font-size: 1.65rem;
            font-weight: 900;
            color: #ffffff;
            min-width: 2ch;
            text-align: center;
            text-shadow: 0 2px 10px rgba(0,0,0,0.5);
            letter-spacing: -0.02em;
            line-height: 1;
        }
        .sa-mini-pair-row .dash,
        .sa-mini-scores .dash {
            color: #e2e8f0;
            font-weight: 800;
            font-size: 1.1rem;
            align-self: center;
            padding-bottom: 4px;
            opacity: 0.85;
        }
        .sa-mini-phase {
            text-align: center;
            font-size: 0.76rem;
            font-weight: 700;
            color: #94a3b8;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        .sa-mini-phase.live {
            color: #fbbf24;
        }
        .sa-mini-phase.final {
            color: #34d399;
        }
        .sa-f5-banner {
            text-align: center;
            padding: 8px 10px;
            border-radius: 10px;
            margin-bottom: 10px;
            border: 1px solid rgba(255,255,255,0.22);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
        }
        .sa-f5-banner-compact {
            padding: 6px 8px;
            margin-bottom: 8px;
            border-radius: 8px;
        }
        .sa-f5-banner-compact .sa-f5-banner-title {
            font-size: 0.58rem;
            letter-spacing: 0.14em;
        }
        .sa-f5-banner-final {
            background: linear-gradient(180deg, rgba(34,197,94,0.45) 0%, rgba(15,23,42,0.92) 55%);
            border-color: rgba(74,222,128,0.45);
        }
        .sa-f5-banner-live {
            background: linear-gradient(180deg, rgba(251,191,36,0.42) 0%, rgba(15,23,42,0.92) 55%);
            border-color: rgba(253,224,71,0.35);
        }
        .sa-f5-banner-title {
            font-size: 0.72rem;
            font-weight: 900;
            letter-spacing: 0.14em;
            color: #ffffff;
            text-transform: uppercase;
            display: block;
            line-height: 1.3;
            text-shadow: 0 1px 3px rgba(0,0,0,0.55);
        }
        .sa-game-position {
            text-align: center;
            font-size: 1.08rem;
            font-weight: 800;
            color: #f8fafc;
            margin-top: 10px;
            letter-spacing: 0.02em;
            line-height: 1.4;
        }
        .sa-mini-game-position {
            text-align: center;
            font-size: 0.88rem;
            font-weight: 800;
            color: #f1f5f9;
            margin-top: 8px;
            line-height: 1.4;
            letter-spacing: 0.02em;
        }
        .sa-mini-full {
            text-align: center;
            font-size: 0.78rem;
            color: #cbd5e1;
            margin-top: 10px;
            font-weight: 600;
            letter-spacing: 0.02em;
        }
        .sa-slip-full-game {
            text-align: center;
            font-size: 0.92rem;
            color: #cbd5e1;
            margin-top: 14px;
            padding-top: 14px;
            border-top: 1px solid rgba(148,163,184,0.35);
            font-weight: 600;
        }
        .sa-fullgame-secondary {
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid rgba(148,163,184,0.32);
            text-align: center;
        }
        .sa-fullgame-secondary-compact {
            margin-top: 8px;
            padding-top: 8px;
        }
        .sa-fullgame-label {
            font-size: 0.62rem;
            font-weight: 800;
            letter-spacing: 0.16em;
            color: #94a3b8;
            margin-bottom: 8px;
        }
        .sa-fg-row {
            display: flex;
            justify-content: center;
            align-items: center;
            flex-wrap: wrap;
            gap: 8px 16px;
            margin-bottom: 6px;
        }
        .sa-fg-chip {
            display: inline-block;
            padding: 5px 14px;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 900;
            letter-spacing: 0.12em;
        }
        .sa-fg-chip-live {
            background: rgba(251,191,36,0.18);
            color: #fde68a;
            border: 1px solid rgba(251,191,36,0.45);
        }
        .sa-fg-chip-final {
            background: rgba(34,197,94,0.16);
            color: #bbf7d0;
            border: 1px solid rgba(74,222,128,0.42);
        }
        .sa-fg-inning {
            font-size: 1rem;
            font-weight: 800;
            color: #f8fafc;
        }
        .sa-fg-total {
            font-size: 1rem;
            font-weight: 800;
            color: #e2e8f0;
            letter-spacing: 0.04em;
        }
        .sa-fullgame-secondary-compact .sa-fullgame-label {
            font-size: 0.58rem;
            margin-bottom: 6px;
        }
        .sa-fullgame-secondary-compact .sa-fg-inning {
            font-size: 0.82rem;
        }
        .sa-fullgame-secondary-compact .sa-fg-total {
            font-size: 0.92rem;
        }
        .sa-fullgame-secondary-compact .sa-fg-chip {
            font-size: 0.62rem;
            padding: 4px 10px;
        }
        .sa-fg-muted {
            color: #94a3b8;
            font-weight: 600;
            font-size: 0.88rem;
        }
        .sa-slip-teams-block {
            text-align: center;
            font-size: 0.82rem;
            font-weight: 700;
            color: #94a3b8;
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid rgba(148,163,184,0.18);
            line-height: 1.35;
            letter-spacing: 0.01em;
        }
        .sa-slip-matchup-micro {
            text-align: center;
            font-size: 0.68rem;
            font-weight: 600;
            color: #64748b;
            margin-top: 8px;
            letter-spacing: 0.02em;
        }
        .sa-slip-scorehdr-min {
            font-size: 0.55rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #64748b;
            margin-bottom: 6px;
            text-align: center;
        }
        .sa-slip-fg-min {
            font-size: 0.7rem;
            color: #94a3b8;
            text-align: center;
            margin-top: 8px;
            padding-top: 8px;
            border-top: 1px solid rgba(148,163,184,0.14);
            line-height: 1.35;
        }
        .sa-slip-fg-k {
            color: #64748b;
            font-weight: 700;
            margin-right: 4px;
        }
        .sa-slip-fg-v {
            color: #e2e8f0;
            font-weight: 800;
            font-variant-numeric: tabular-nums;
        }
        .sa-slip-fg-tag {
            font-size: 0.58rem;
            color: #6ee7b7;
            font-weight: 800;
            margin-left: 6px;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        .sa-slip-fg-meta {
            color: #94a3b8;
            font-size: 0.65rem;
        }
        .sa-slip-fg-muted {
            color: #64748b;
            font-weight: 600;
        }
        .sa-slip-pick-lbl {
            font-size: 0.55rem;
            font-weight: 800;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: #64748b;
            margin-bottom: 3px;
        }
        .sa-slip-pair-row {
            display: flex;
            justify-content: center;
            align-items: flex-end;
            gap: 12px;
            font-variant-numeric: tabular-nums;
            margin-top: 4px;
        }
        .sa-slip-side {
            flex: 1;
            min-width: 0;
            text-align: center;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 3px;
        }
        .sa-slip-over {
            font-size: 0.62rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            color: #94a3b8;
            text-transform: uppercase;
            max-width: 100%;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            line-height: 1.2;
        }
        .sa-slip-stack {
            display: flex;
            flex-direction: column;
            gap: 12px;
            margin-bottom: 8px;
        }
        .sa-slip-card {
            border-radius: 12px;
            border: 1px solid #2c3d5c;
            border-left: 3px solid #475569;
            background: linear-gradient(145deg, #111a2e 0%, #162238 100%);
            padding: 11px 13px;
            box-shadow: 0 6px 18px rgba(0,0,0,0.2);
            margin-bottom: 9px;
            position: relative;
            overflow: hidden;
        }
        .sa-slip-card > * {
            position: relative;
            z-index: 1;
        }
        .sa-slip-card-neutral {
            border-left-color: #64748b;
            background: linear-gradient(145deg, #111a2e 0%, #162238 100%);
        }
        .sa-slip-card-c100 {
            border-left-color: rgba(167, 139, 250, 0.85);
            background: linear-gradient(145deg, #141528 0%, #182238 100%);
            box-shadow: 0 8px 24px rgba(0,0,0,0.22), inset 0 0 48px rgba(139, 92, 246, 0.06);
        }
        .sa-slip-card-core {
            border-left-color: rgba(56, 189, 248, 0.85);
            background: linear-gradient(145deg, #101d2a 0%, #152238 100%);
            box-shadow: 0 8px 24px rgba(0,0,0,0.22), inset 0 0 48px rgba(34, 211, 238, 0.05);
        }
        .sa-slip-card-value {
            border-left-color: rgba(251, 191, 36, 0.75);
            background: linear-gradient(145deg, #121a24 0%, #1a2230 100%);
            box-shadow: 0 8px 24px rgba(0,0,0,0.22), inset 0 0 48px rgba(251, 191, 36, 0.05);
        }
        .sa-slip-card-btonly {
            border-left-color: rgba(244, 114, 182, 0.75);
            background: linear-gradient(145deg, #121821 0%, #1a1f2e 100%);
            box-shadow: 0 8px 24px rgba(0,0,0,0.22), inset 0 0 48px rgba(244, 114, 182, 0.05);
        }
        .sa-slip-card-consensus {
            border-left-color: rgba(52, 211, 153, 0.8);
            background: linear-gradient(145deg, #101f22 0%, #15232e 100%);
            box-shadow: 0 8px 24px rgba(0,0,0,0.22), inset 0 0 48px rgba(52, 211, 153, 0.05);
        }
        .sa-slip-card-multi {
            border-left-color: rgba(148, 163, 184, 0.95);
            background: linear-gradient(145deg, #121826 0%, #181f2e 100%);
            box-shadow: 0 8px 24px rgba(0,0,0,0.22), inset 0 0 40px rgba(148, 163, 184, 0.06);
        }
        .sa-feature-pick-card {
            border-radius: 12px;
            border: 1px solid #2c3d5c;
            border-left: 3px solid #475569;
            background: linear-gradient(145deg, #111a2e 0%, #162238 100%);
            padding: 11px 13px;
            box-shadow: 0 6px 18px rgba(0,0,0,0.2);
            margin-bottom: 9px;
            position: relative;
            overflow: hidden;
        }
        .sa-feature-pick-card > * {
            position: relative;
            z-index: 1;
        }
        .sa-slip-card.sa-game-tone-0, .sa-mini-board.sa-game-tone-0, .sa-feature-pick-card.sa-game-tone-0 { --mt: #38bdf8; --mt-bg: rgba(56,189,248,0.11); --mt-soft: rgba(56,189,248,0.22); }
        .sa-slip-card.sa-game-tone-1, .sa-mini-board.sa-game-tone-1, .sa-feature-pick-card.sa-game-tone-1 { --mt: #a78bfa; --mt-bg: rgba(167,139,250,0.11); --mt-soft: rgba(167,139,250,0.22); }
        .sa-slip-card.sa-game-tone-2, .sa-mini-board.sa-game-tone-2, .sa-feature-pick-card.sa-game-tone-2 { --mt: #fbbf24; --mt-bg: rgba(251,191,36,0.1); --mt-soft: rgba(251,191,36,0.2); }
        .sa-slip-card.sa-game-tone-3, .sa-mini-board.sa-game-tone-3, .sa-feature-pick-card.sa-game-tone-3 { --mt: #fb7185; --mt-bg: rgba(251,113,133,0.1); --mt-soft: rgba(251,113,133,0.2); }
        .sa-slip-card.sa-game-tone-4, .sa-mini-board.sa-game-tone-4, .sa-feature-pick-card.sa-game-tone-4 { --mt: #34d399; --mt-bg: rgba(52,211,153,0.1); --mt-soft: rgba(52,211,153,0.2); }
        .sa-slip-card.sa-game-tone-5, .sa-mini-board.sa-game-tone-5, .sa-feature-pick-card.sa-game-tone-5 { --mt: #2dd4bf; --mt-bg: rgba(45,212,191,0.1); --mt-soft: rgba(45,212,191,0.2); }
        .sa-slip-card.sa-game-tone-6, .sa-mini-board.sa-game-tone-6, .sa-feature-pick-card.sa-game-tone-6 { --mt: #fb923c; --mt-bg: rgba(251,146,60,0.1); --mt-soft: rgba(251,146,60,0.2); }
        .sa-slip-card.sa-game-tone-7, .sa-mini-board.sa-game-tone-7, .sa-feature-pick-card.sa-game-tone-7 { --mt: #a3e635; --mt-bg: rgba(163,230,53,0.1); --mt-soft: rgba(163,230,53,0.2); }
        .sa-slip-card.sa-game-tone-8, .sa-mini-board.sa-game-tone-8, .sa-feature-pick-card.sa-game-tone-8 { --mt: #e879f9; --mt-bg: rgba(232,121,249,0.1); --mt-soft: rgba(232,121,249,0.2); }
        .sa-slip-card.sa-game-tone-9, .sa-mini-board.sa-game-tone-9, .sa-feature-pick-card.sa-game-tone-9 { --mt: #818cf8; --mt-bg: rgba(129,140,248,0.11); --mt-soft: rgba(129,140,248,0.22); }
        .sa-slip-card.sa-game-tone-10, .sa-mini-board.sa-game-tone-10, .sa-feature-pick-card.sa-game-tone-10 { --mt: #fcd34d; --mt-bg: rgba(252,211,77,0.12); --mt-soft: rgba(252,211,77,0.22); }
        .sa-slip-card.sa-game-tone-11, .sa-mini-board.sa-game-tone-11, .sa-feature-pick-card.sa-game-tone-11 { --mt: #c084fc; --mt-bg: rgba(192,132,252,0.11); --mt-soft: rgba(192,132,252,0.22); }
        .sa-slip-card.sa-game-tone-0,
        .sa-slip-card.sa-game-tone-1,
        .sa-slip-card.sa-game-tone-2,
        .sa-slip-card.sa-game-tone-3,
        .sa-slip-card.sa-game-tone-4,
        .sa-slip-card.sa-game-tone-5,
        .sa-slip-card.sa-game-tone-6,
        .sa-slip-card.sa-game-tone-7,
        .sa-slip-card.sa-game-tone-8,
        .sa-slip-card.sa-game-tone-9,
        .sa-slip-card.sa-game-tone-10,
        .sa-slip-card.sa-game-tone-11,
        .sa-feature-pick-card.sa-game-tone-0,
        .sa-feature-pick-card.sa-game-tone-1,
        .sa-feature-pick-card.sa-game-tone-2,
        .sa-feature-pick-card.sa-game-tone-3,
        .sa-feature-pick-card.sa-game-tone-4,
        .sa-feature-pick-card.sa-game-tone-5,
        .sa-feature-pick-card.sa-game-tone-6,
        .sa-feature-pick-card.sa-game-tone-7,
        .sa-feature-pick-card.sa-game-tone-8,
        .sa-feature-pick-card.sa-game-tone-9,
        .sa-feature-pick-card.sa-game-tone-10,
        .sa-feature-pick-card.sa-game-tone-11 {
            border-left: 5px solid var(--mt) !important;
            border-right-color: rgba(44,61,92,0.85) !important;
            border-top-color: rgba(44,61,92,0.85) !important;
            border-bottom-color: rgba(44,61,92,0.85) !important;
            background-image: linear-gradient(110deg, var(--mt-bg) 0%, transparent 42%), linear-gradient(145deg, #111a2e 0%, #162238 100%) !important;
            box-shadow: 0 4px 14px rgba(0,0,0,0.22), inset 0 0 0 1px rgba(255,255,255,0.03), 0 0 20px var(--mt-soft) !important;
        }
        .sa-mini-board.sa-game-tone-0,
        .sa-mini-board.sa-game-tone-1,
        .sa-mini-board.sa-game-tone-2,
        .sa-mini-board.sa-game-tone-3,
        .sa-mini-board.sa-game-tone-4,
        .sa-mini-board.sa-game-tone-5,
        .sa-mini-board.sa-game-tone-6,
        .sa-mini-board.sa-game-tone-7,
        .sa-mini-board.sa-game-tone-8,
        .sa-mini-board.sa-game-tone-9,
        .sa-mini-board.sa-game-tone-10,
        .sa-mini-board.sa-game-tone-11 {
            border-left: 5px solid var(--mt) !important;
            border-right-color: rgba(42,63,99,0.85) !important;
            border-top-color: rgba(42,63,99,0.85) !important;
            border-bottom-color: rgba(42,63,99,0.85) !important;
            background-image: linear-gradient(115deg, var(--mt-bg) 0%, transparent 45%), linear-gradient(180deg, #121c2f 0%, #0d1524 100%) !important;
            box-shadow: 0 3px 12px rgba(0,0,0,0.28), 0 0 18px var(--mt-soft) !important;
        }
        .sa-slip-card-top {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 10px;
            margin-bottom: 6px;
        }
        .sa-slip-pick {
            font-size: 0.98rem;
            font-weight: 900;
            color: #f8fafc;
            line-height: 1.2;
            letter-spacing: -0.02em;
        }
        .sa-slip-matchup {
            font-size: 0.88rem;
            color: #94a3b8;
            margin-bottom: 12px;
            font-weight: 600;
        }
        .sa-slip-scorebox {
            border-radius: 12px;
            background: linear-gradient(180deg, rgba(30,41,59,0.92) 0%, rgba(15,23,42,0.98) 100%);
            border: 1px solid rgba(148,163,184,0.28);
            padding: 10px 12px;
            margin-bottom: 8px;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,0.04);
        }
        .sa-slip-score-line {
            display: flex;
            justify-content: center;
            align-items: baseline;
            gap: 18px;
            font-variant-numeric: tabular-nums;
            margin-top: 6px;
        }
        .sa-slip-pair-row .big,
        .sa-slip-score-line .big {
            font-size: 1.45rem;
            font-weight: 900;
            color: #ffffff;
            text-shadow: 0 2px 12px rgba(0,0,0,0.5);
            letter-spacing: -0.03em;
            line-height: 1;
        }
        .sa-slip-pair-row .sep,
        .sa-slip-score-line .sep {
            color: #f1f5f9;
            font-weight: 800;
            font-size: 1rem;
            opacity: 0.9;
            align-self: center;
            padding-bottom: 2px;
        }
        .sa-slip-phase {
            text-align: center;
            margin-top: 8px;
            font-size: 0.78rem;
            font-weight: 700;
            color: #cbd5e1;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }
        .sa-slip-phase.live { color: #fcd34d; }
        .sa-slip-phase.final { color: #6ee7b7; }
        .sa-slip-phase.pre { color: #93c5fd; }
        .sa-slip-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            align-items: center;
            font-size: 0.68rem;
            color: #a8b9d4;
        }
        .sa-pill {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 999px;
            font-size: 0.62rem;
            font-weight: 700;
            background: rgba(51,65,85,0.65);
            border: 1px solid #3d4f6f;
            color: #cbd5e1;
        }
        .sa-ticket {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 36px;
            padding: 4px 9px;
            border-radius: 8px;
            font-weight: 900;
            font-size: 0.72rem;
            letter-spacing: 0.06em;
        }
        .sa-ticket.W { background: #14532d; color: #bbf7d0; border: 1px solid #22c55e; }
        .sa-ticket.L { background: #450a0a; color: #fecaca; border: 1px solid #ef4444; }
        .sa-ticket.P { background: #422006; color: #fde68a; border: 1px solid #f59e0b; }
        .sa-ticket.OPEN { background: #422006; color: #fde68a; border: 1px solid #fbbf24; }
        .sa-ticket.VOID { background: #1e293b; color: #94a3b8; border: 1px solid #475569; }
        .sa-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin: 12px 0 18px 0;
        }
        .sa-chip {
            border-radius: 12px;
            border: 1px solid #334155;
            background: rgba(30,41,59,0.55);
            padding: 10px 14px;
            min-width: 140px;
        }
        .sa-chip .t {
            font-size: 0.68rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #64748b;
            font-weight: 800;
            margin-bottom: 4px;
        }
        .sa-chip .v {
            font-size: 0.95rem;
            font-weight: 800;
            color: #e2e8f0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str, subtitle: str = "") -> None:
    if subtitle:
        st.markdown(
            f'<div class="panel"><div class="section-title">{title}</div><div class="section-subtitle">{subtitle}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="panel"><div class="section-title">{title}</div></div>',
            unsafe_allow_html=True,
        )


def short_team_label(full_name: str) -> str:
    """Short scoreboard label (nickname, not truncated mid-word when avoidable)."""
    full_name = str(full_name).strip()
    if not full_name:
        return ""
    parts = full_name.split()
    nick = parts[-1] if len(parts) >= 2 else full_name
    if len(nick) <= 8:
        return nick.upper()
    return nick[:8].upper()


def ordinal_inning(n: int) -> str:
    if n <= 0:
        return str(max(n, 0))
    if 11 <= (n % 100) <= 13:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def format_game_position_line(
    inning: Optional[int],
    inning_state: str,
    is_final: bool,
    status_text: str,
) -> str:
    """Human-readable game position so the viewer always knows the inning (or final)."""
    if is_final:
        return "Full game · Final"
    if inning is None:
        return status_text or "Live"
    st = str(inning_state or "").lower()
    ord_i = ordinal_inning(int(inning))
    if st == "end":
        return f"Game · End of {ord_i} inning"
    ip = {"top": "Top", "middle": "Middle", "bottom": "Bottom"}.get(st, "")
    if ip:
        return f"Game · {ip} {ord_i} inning"
    return f"Game · {ord_i} inning"


def format_inning_short(inning: Optional[int], inning_state: str) -> str:
    """Full-game inning only (for secondary row under F5 score)."""
    if inning is None:
        return ""
    st = str(inning_state or "").lower()
    ord_i = ordinal_inning(int(inning))
    if st == "end":
        return f"End of {ord_i} inning"
    ip = {"top": "Top", "middle": "Middle", "bottom": "Bottom"}.get(st, "")
    if ip:
        return f"{ip} {ord_i} inning"
    return f"{ord_i} inning"


def f5_primary_banner_title(game: Dict[str, Any]) -> str:
    """Top banner only: F5 final vs F5 still in play through inn. 5."""
    if bool(game.get("can_grade")):
        return "F5 FINAL SCORE"
    ci = to_int(game.get("current_inning"))
    ist = str(game.get("inning_state", "")).lower()
    if ci is None:
        return "F5 LIVE"
    if ci <= 5:
        ox = ordinal_inning(ci)
        if ist == "end":
            return f"F5 · End {ox}"
        ip = {"top": "Top", "middle": "Mid", "bottom": "Bot", "end": "End"}.get(ist, "")
        if ip:
            return f"F5 · {ip} {ox}"
        return f"F5 · Inn {ox}"
    return "F5 FINAL SCORE"


def full_game_secondary_fields(game: Dict[str, Any]) -> Dict[str, Any]:
    """Secondary block under F5: full-game LIVE/FINAL, inning, and total runs (updates after F5 locks)."""
    status_text = str(game.get("status", ""))
    sl = status_text.lower()
    if any(x in sl for x in ["scheduled", "pre-game", "pregame", "warmup"]):
        return {"fg_mode": "pre", "fg_chip": "", "fg_inning_text": "", "fg_score": ""}

    at = game.get("away_total")
    ht = game.get("home_total")
    ati = int(at) if at is not None else None
    hti = int(ht) if ht is not None else None
    is_final = bool(game.get("is_final")) or "final" in sl
    ci = to_int(game.get("current_inning"))
    ist = str(game.get("inning_state", "")).lower()

    sc = "—"
    if ati is not None and hti is not None:
        sc = f"{ati}–{hti}"

    if is_final:
        return {"fg_mode": "final", "fg_chip": "FINAL", "fg_inning_text": "", "fg_score": sc}

    inn = format_inning_short(ci, ist)
    return {"fg_mode": "live", "fg_chip": "LIVE", "fg_inning_text": inn, "fg_score": sc}


def render_full_game_minimal_html(pl: Dict[str, Any]) -> str:
    """Single calm line for full-game score (cards) — avoids a second giant score block."""
    mode = str(pl.get("fg_mode") or "live")
    if mode == "pre":
        return '<div class="sa-slip-fg-min"><span class="sa-slip-fg-k">Full game</span><span class="sa-slip-fg-muted"> · not started</span></div>'
    tot = html.escape(str(pl.get("fg_score") or "—"))
    if mode == "final":
        return (
            f'<div class="sa-slip-fg-min">'
            f'<span class="sa-slip-fg-k">Full game</span> '
            f'<span class="sa-slip-fg-v">{tot}</span> '
            f'<span class="sa-slip-fg-tag">final</span>'
            f"</div>"
        )
    inn = html.escape(str(pl.get("fg_inning_text") or "").strip())
    inn_bit = f'<span class="sa-slip-fg-meta"> · {inn}</span>' if inn else ""
    return (
        f'<div class="sa-slip-fg-min">'
        f'<span class="sa-slip-fg-k">Full game</span> '
        f'<span class="sa-slip-fg-v">{tot}</span>'
        f"{inn_bit}"
        f'<span class="sa-slip-fg-meta"> · live</span>'
        f"</div>"
    )


def full_game_strip_caption(pl: Dict[str, Any]) -> str:
    """One short line for the horizontal ticker (no extra banners)."""
    mode = str(pl.get("fg_mode") or "live")
    if mode == "pre":
        return "FG —"
    tot = str(pl.get("fg_score") or "—")
    if mode == "final":
        return f"FG {tot} · game final"
    inn = str(pl.get("fg_inning_text") or "").strip()
    if inn:
        return f"FG {tot} · {inn}"
    return f"FG {tot} · live"


def render_full_game_secondary_html(pl: Dict[str, Any], *, compact: bool) -> str:
    """HTML for the full-game strip under the big F5 numbers (same pattern on strip + cards)."""
    mode = str(pl.get("fg_mode") or "live")
    cm = " sa-fullgame-secondary-compact" if compact else ""
    if mode == "pre":
        return f'<div class="sa-fullgame-secondary{cm}"><span class="sa-fg-muted">Not started</span></div>'

    chip = html.escape(str(pl.get("fg_chip") or ""))
    inn = html.escape(str(pl.get("fg_inning_text") or ""))
    tot = html.escape(str(pl.get("fg_score") or "—"))

    # Single-line HTML: indented multiline strings break Streamlit markdown (code-block rules).
    if mode == "final":
        return (
            f'<div class="sa-fullgame-secondary{cm}">'
            f'<div class="sa-fullgame-label">FULL GAME</div>'
            f'<div class="sa-fg-row"><span class="sa-fg-chip sa-fg-chip-final">{chip}</span></div>'
            f'<div class="sa-fg-total">{tot}</div>'
            f"</div>"
        )

    inn_part = f'<span class="sa-fg-inning">{inn}</span>' if inn else ""
    return (
        f'<div class="sa-fullgame-secondary{cm}">'
        f'<div class="sa-fullgame-label">FULL GAME</div>'
        f'<div class="sa-fg-row"><span class="sa-fg-chip sa-fg-chip-live">{chip}</span>{inn_part}</div>'
        f'<div class="sa-fg-total">{tot}</div>'
        f"</div>"
    )


def scoreboard_payload(matchup: str, score_map: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Build display payload: primary score is always F5 (innings 1–5); full game is secondary."""
    away_team = matchup.split(" @ ")[0].strip() if " @ " in matchup else ""
    home_team = matchup.split(" @ ")[1].strip() if " @ " in matchup else ""
    if not away_team or not home_team:
        return None
    game_key = f"{canonical_team_key(away_team)}|{canonical_team_key(home_team)}"
    game = score_map.get(game_key)
    if not game:
        return None
    status_text = str(game.get("status", ""))
    status_lower = status_text.lower()
    away_total = game.get("away_total")
    home_total = game.get("home_total")
    away_f5 = int(game.get("away_f5", 0))
    home_f5 = int(game.get("home_f5", 0))
    ci = to_int(game.get("current_inning"))
    inning_state = str(game.get("inning_state", "")).lower()
    is_final = bool(game.get("is_final")) or "final" in status_lower

    if any(x in status_lower for x in ["scheduled", "pre-game", "pregame", "warmup"]):
        return {
            "away": away_team,
            "home": home_team,
            "away_s": None,
            "home_s": None,
            "phase": "pre",
            "phase_label": "Not started",
            "f5_headline": "",
            "f5_locked": False,
            "banner_kind": "",
            "full_away": None,
            "full_home": None,
            "fg_mode": "pre",
            "fg_chip": "",
            "fg_inning_text": "",
            "fg_score": "",
        }

    at = int(away_total) if away_total is not None else None
    ht = int(home_total) if home_total is not None else None

    f5_locked = bool(game.get("can_grade"))
    f5_headline = f5_primary_banner_title(game)
    banner_kind = "final" if f5_locked else "live"
    fg = full_game_secondary_fields(game)

    inning_prefix = {"top": "Top", "middle": "Mid", "bottom": "Bot", "end": "End"}.get(inning_state, "")
    if is_final:
        phase_label = "Final"
        phase_kind = "final"
    else:
        phase_label = f"{inning_prefix} {ci}".strip() if ci is not None else (status_text or "Live")
        phase_kind = "live"

    base_return = {
        "away": away_team,
        "home": home_team,
        "away_s": away_f5,
        "home_s": home_f5,
        "phase": phase_kind,
        "phase_label": phase_label,
        "f5_headline": f5_headline,
        "f5_locked": f5_locked,
        "banner_kind": banner_kind,
        "full_away": at,
        "full_home": ht,
        "fg_mode": fg["fg_mode"],
        "fg_chip": fg["fg_chip"],
        "fg_inning_text": fg["fg_inning_text"],
        "fg_score": fg["fg_score"],
    }

    if at is not None and ht is not None:
        return base_return

    return {
        **base_return,
        "phase": "unknown",
        "phase_label": status_text or "In progress",
    }


def game_phase_rank(game_str: str) -> int:
    s = str(game_str).lower()
    if "not started" in s:
        return 1
    if "status unavailable" in s:
        return 2
    if "final" in s or "(final)" in s:
        return 3
    return 0


def prepare_betslip_display_order(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "Game" not in out.columns:
        return out
    out["_phase"] = out["Game"].astype(str).map(game_phase_rank)
    out["_ps"] = pd.to_numeric(out.get("Priority Score"), errors="coerce").fillna(0.0)
    out = out.sort_values(by=["_phase", "_ps"], ascending=[True, False])
    return out.drop(columns=["_phase", "_ps"])


def ticket_css_class(ticket: str) -> str:
    t = str(ticket).strip().upper()
    if t in ("W", "L", "P", "OPEN", "VOID"):
        return t
    return "OPEN"


def slip_card_accent_class(source_tags: str) -> str:
    """Board tag accent (optional); primary card color is matchup_tone_class (same game = same color)."""
    raw = str(source_tags).strip()
    if not raw:
        return "sa-slip-card-neutral"
    parts = sorted({p.strip() for p in raw.split(",") if p.strip()})
    if len(parts) > 1:
        return "sa-slip-card-multi"
    p = parts[0]
    if p == "100% Confidence":
        return "sa-slip-card-c100"
    if p == "Core":
        return "sa-slip-card-core"
    if p == "Value":
        return "sa-slip-card-value"
    if p == "Backtest-Only":
        return "sa-slip-card-btonly"
    return "sa-slip-card-neutral"


def matchup_tone_key(matchup: str) -> str:
    """Canonical key so strip + cards share a tone even if matchup wording differs slightly."""
    m = " ".join(str(matchup).split()).strip()
    if " @ " not in m:
        return norm_key(m)
    away, home = m.split(" @ ", 1)
    return f"{canonical_team_key(away.strip())}|{canonical_team_key(home.strip())}"


def matchup_tone_class(matchup: str) -> str:
    """Stable color bucket per matchup so strip bubbles and bet cards match."""
    key = matchup_tone_key(matchup)
    if not key:
        return "sa-game-tone-0"
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return f"sa-game-tone-{int(h[:8], 16) % 12}"


def esc_html_one_line(val: Any) -> str:
    """Escape for HTML inside Streamlit markdown: no newlines (indented continuations become code blocks)."""
    return html.escape(" ".join(str(val if val is not None else "").split()))


def render_command_center_hero(
    today_slip_summary: Dict[str, Any],
    full_tracker_summary: Dict[str, Any],
    last_pull_label: str,
    slip_count: int,
    not_started: int,
    open_tickets: int,
) -> None:
    t_rec = html.escape(str(today_slip_summary.get("record", "—")))
    t_wr = float(today_slip_summary.get("win_rate", 0.0))
    t_graded = int(today_slip_summary.get("graded", 0))
    f_rec = html.escape(str(full_tracker_summary.get("record", "—")))
    f_wr = float(full_tracker_summary.get("win_rate", 0.0))
    f_graded = int(full_tracker_summary.get("graded", 0))
    pull = html.escape(str(last_pull_label or "—"))
    hero = (
        '<div class="sa-cc-hero">'
        '<div class="sa-cc-duo">'
        '<div class="sa-cc-panel">'
        '<div class="sa-cc-panel-k">Today · merged slip vs tracker</div>'
        f'<div class="sa-cc-panel-v">{t_rec}</div>'
        f'<div class="sa-cc-panel-sub">{t_wr:.1f}% WR · {t_graded} graded</div>'
        "<div class=\"sa-cc-panel-hint\">Where this comes from: every pick on <b>today&rsquo;s slip</b> "
        "(100% + Core + Value + Backtest-only), merged by <b>matchup + pick</b>. "
        "The W-L-P counts only <b>tracker tickets dated today</b> that match one of those slip rows.</div>"
        "</div>"
        '<div class="sa-cc-panel">'
        '<div class="sa-cc-panel-k">All time · full tracker</div>'
        f'<div class="sa-cc-panel-v">{f_rec}</div>'
        f'<div class="sa-cc-panel-sub">{f_wr:.1f}% WR · {f_graded} graded</div>'
        "<div class=\"sa-cc-panel-hint\">Separate scope: <b>all boards</b>, <b>all dates</b> in the master tracker file &mdash; "
        "not limited to what is on today&rsquo;s slip.</div>"
        "</div>"
        "</div>"
        '<div class="sa-cc-hero-foot">'
        f"<span><b>Slip rows</b> {slip_count}</span>"
        f"<span><b>Not started</b> {not_started}</span>"
        f"<span><b>Open</b> {open_tickets}</span>"
        f"<span><b>Sync</b> {pull}</span>"
        "</div>"
        "</div>"
    )
    st.markdown(hero, unsafe_allow_html=True)


def render_scoreboard_strip_from_slip(
    today_betslip_df: pd.DataFrame,
    score_map: Dict[str, Dict[str, Any]],
) -> None:
    """Horizontal F5 score bubbles for each unique matchup on the slip (same tones as pick cards)."""
    if today_betslip_df.empty or "Matchup" not in today_betslip_df.columns:
        return
    seen: set = set()
    parts: List[str] = []
    for m in today_betslip_df["Matchup"].astype(str):
        if m in seen:
            continue
        seen.add(m)
        sub = today_betslip_df[today_betslip_df["Matchup"].astype(str) == m]
        game_txt = str(sub.iloc[0]["Game"]) if "Game" in sub.columns else ""
        pl = scoreboard_payload(m, score_map)
        tone = matchup_tone_class(m)
        away_nm = html.escape(short_team_label(m.split(" @ ")[0].strip())) if " @ " in m else ""
        home_nm = html.escape(short_team_label(m.split(" @ ")[1].strip())) if " @ " in m else ""
        if pl:
            away_nm = html.escape(short_team_label(pl["away"]))
            home_nm = html.escape(short_team_label(pl["home"]))
            if pl.get("phase") == "pre":
                mshort = html.escape(f"{short_team_label(pl['away'])} @ {short_team_label(pl['home'])}")
                inner = (
                    f'<div class="sa-mini-board sa-score-ui {tone}">'
                    f'<div class="sa-mini-ticker-title">{mshort}</div>'
                    f'<div class="sa-mini-phase" style="margin-bottom:4px;">NOT STARTED</div>'
                    f'<div class="sa-mini-pair-row">'
                    f'<div class="sa-mini-side"><span class="sa-mini-over">{away_nm}</span>'
                    f'<span class="n" style="font-size:1.05rem;color:#64748b;">—</span></div>'
                    f'<span class="dash">·</span>'
                    f'<div class="sa-mini-side"><span class="sa-mini-over">{home_nm}</span>'
                    f'<span class="n" style="font-size:1.05rem;color:#64748b;">—</span></div>'
                    f"</div></div>"
                )
            elif pl.get("away_s") is not None and pl.get("home_s") is not None:
                a_s = pl["away_s"]
                h_s = pl["home_s"]
                mshort = html.escape(f"{short_team_label(pl['away'])} @ {short_team_label(pl['home'])}")
                fg_one = html.escape(full_game_strip_caption(pl))
                inner = (
                    f'<div class="sa-mini-board sa-score-ui {tone}">'
                    f'<div class="sa-mini-ticker-title">{mshort}</div>'
                    f'<div class="sa-mini-pair-row">'
                    f'<div class="sa-mini-side"><span class="sa-mini-over">{away_nm}</span><span class="n">{a_s}</span></div>'
                    f'<span class="dash">—</span>'
                    f'<div class="sa-mini-side"><span class="sa-mini-over">{home_nm}</span><span class="n">{h_s}</span></div>'
                    f"</div>"
                    f'<div class="sa-mini-fg-inline">{fg_one}</div>'
                    f"</div>"
                )
            else:
                phase_txt = esc_html_one_line(str(pl.get("phase_label", game_txt))[:52])
                inner = (
                    f'<div class="sa-mini-board sa-score-ui {tone}">'
                    f'<div class="sa-mini-phase">{phase_txt}</div>'
                    f'<div class="sa-mini-names"><span class="tm">{away_nm}</span><span class="at">@</span><span class="tm">{home_nm}</span></div>'
                    f"</div>"
                )
        else:
            gt = esc_html_one_line(game_txt[:96])
            inner = (
                f'<div class="sa-mini-board sa-score-ui {tone}">'
                f'<div style="text-align:center;font-size:0.85rem;color:#cbd5e1;padding-bottom:8px;font-weight:600;">{gt}</div>'
                f'<div class="sa-mini-names"><span class="tm">{away_nm}</span><span class="at">@</span><span class="tm">{home_nm}</span></div>'
                f"</div>"
            )
        parts.append(inner)
    if not parts:
        return
    strip_html = (
        '<div class="sa-score-strip-wrap"><div class="sa-score-strip">'
        + "".join(parts).replace("\n", " ")
        + "</div></div>"
    )
    st.markdown(strip_html, unsafe_allow_html=True)


def render_betslip_cards_html(df: pd.DataFrame, score_map: Dict[str, Dict[str, Any]]) -> None:
    if df.empty:
        return
    disp = prepare_betslip_display_order(df)
    blocks: List[str] = []
    for _, row in disp.iterrows():
        pick = esc_html_one_line(row.get("Suggested F5 Pick", ""))
        matchup = esc_html_one_line(row.get("Matchup", ""))
        ticket = ticket_css_class(str(row.get("Ticket", "OPEN")))
        tags = esc_html_one_line(row.get("Source Tags", ""))
        conf = row.get("Avg Confidence")
        samp = row.get("Avg Sample")
        conf_s = f"{float(conf):.1f}%" if pd.notna(conf) else "—"
        samp_s = f"{float(samp):.2f}" if pd.notna(samp) else "—"
        m_raw = str(row.get("Matchup", ""))
        pl = scoreboard_payload(m_raw, score_map)
        game_fallback = esc_html_one_line(row.get("Game", ""))

        if pl and pl.get("away_s") is not None and pl.get("home_s") is not None:
            a_s = pl["away_s"]
            h_s = pl["home_s"]
            fg_min = render_full_game_minimal_html(pl)
            away_lab = esc_html_one_line(short_team_label(pl["away"]))
            home_lab = esc_html_one_line(short_team_label(pl["home"]))
            score_inner = (
                f'<div class="sa-slip-scorebox sa-score-ui">'
                f'<div class="sa-slip-scorehdr-min">F5 · innings 1–5</div>'
                f'<div class="sa-slip-pair-row">'
                f'<div class="sa-slip-side"><span class="sa-slip-over">{away_lab}</span>'
                f'<span class="big">{a_s}</span></div>'
                f'<span class="sep">—</span>'
                f'<div class="sa-slip-side"><span class="sa-slip-over">{home_lab}</span>'
                f'<span class="big">{h_s}</span></div>'
                f"</div>"
                f"{fg_min}"
                f'<div class="sa-slip-matchup-micro">{matchup}</div></div>'
            )
        elif pl and pl.get("phase") == "pre":
            score_inner = (
                f'<div class="sa-slip-scorebox sa-score-ui">'
                f'<div class="sa-slip-phase pre" style="margin-top:0;">NOT STARTED</div>'
                f'<div class="sa-slip-teams-block">{matchup}</div></div>'
            )
        else:
            score_inner = (
                f'<div class="sa-slip-scorebox sa-score-ui">'
                f'<div class="sa-slip-phase" style="margin-top:0;text-transform:none;font-weight:600;">{game_fallback}</div>'
                f'<div class="sa-slip-teams-block">{matchup}</div></div>'
            )

        tone = matchup_tone_class(m_raw)
        blocks.append(
            f'<div class="sa-slip-card {tone}">'
            f'<div class="sa-slip-card-top">'
            f'<div><div class="sa-slip-pick-lbl">Pick</div>'
            f'<div class="sa-slip-pick">{pick}</div></div>'
            f'<span class="sa-ticket {ticket}">{ticket}</span></div>'
            f"{score_inner}"
            f'<div class="sa-slip-meta">'
            f'<span class="sa-pill">Conf {conf_s}</span>'
            f'<span class="sa-pill">Sample {samp_s}</span>'
            f'<span class="sa-pill">{tags}</span></div></div>'
        )

    for card_html in blocks:
        st.markdown(card_html.replace("\n", " ").strip(), unsafe_allow_html=True)


def render_featured_pick_cards_html(
    df: pd.DataFrame,
    score_map: Dict[str, Dict[str, Any]],
    board_display_name: str,
) -> None:
    """Same F5 score styling + matchup tone as slip cards so shared games share the same accent."""
    if df.empty or "Matchup" not in df.columns or "Suggested F5 Pick" not in df.columns:
        return
    board_esc = esc_html_one_line(board_display_name)
    for _, row in df.iterrows():
        pick = esc_html_one_line(row.get("Suggested F5 Pick", ""))
        matchup = esc_html_one_line(row.get("Matchup", ""))
        ticket_raw = str(row.get("Ticket", "OPEN")) if "Ticket" in df.columns else "OPEN"
        ticket = ticket_css_class(ticket_raw)
        conf = row.get("Avg Confidence") if "Avg Confidence" in df.columns else None
        samp = row.get("Avg Sample") if "Avg Sample" in df.columns else None
        conf_s = f"{float(conf):.1f}%" if conf is not None and pd.notna(conf) else "—"
        samp_s = f"{float(samp):.2f}" if samp is not None and pd.notna(samp) else "—"
        m_raw = str(row.get("Matchup", ""))
        pl = scoreboard_payload(m_raw, score_map)
        game_fallback = esc_html_one_line(row.get("Game", ""))

        if pl and pl.get("away_s") is not None and pl.get("home_s") is not None:
            a_s = pl["away_s"]
            h_s = pl["home_s"]
            fg_min = render_full_game_minimal_html(pl)
            away_lab = esc_html_one_line(short_team_label(pl["away"]))
            home_lab = esc_html_one_line(short_team_label(pl["home"]))
            score_inner = (
                f'<div class="sa-slip-scorebox sa-score-ui">'
                f'<div class="sa-slip-scorehdr-min">F5 · innings 1–5</div>'
                f'<div class="sa-slip-pair-row">'
                f'<div class="sa-slip-side"><span class="sa-slip-over">{away_lab}</span>'
                f'<span class="big">{a_s}</span></div>'
                f'<span class="sep">—</span>'
                f'<div class="sa-slip-side"><span class="sa-slip-over">{home_lab}</span>'
                f'<span class="big">{h_s}</span></div>'
                f"</div>"
                f"{fg_min}"
                f'<div class="sa-slip-matchup-micro">{matchup}</div></div>'
            )
        elif pl and pl.get("phase") == "pre":
            score_inner = (
                f'<div class="sa-slip-scorebox sa-score-ui">'
                f'<div class="sa-slip-phase pre" style="margin-top:0;">NOT STARTED</div>'
                f'<div class="sa-slip-teams-block">{matchup}</div></div>'
            )
        else:
            score_inner = (
                f'<div class="sa-slip-scorebox sa-score-ui">'
                f'<div class="sa-slip-phase" style="margin-top:0;text-transform:none;font-weight:600;">{game_fallback}</div>'
                f'<div class="sa-slip-teams-block">{matchup}</div></div>'
            )

        tone = matchup_tone_class(m_raw)
        st.markdown(
            (
                f'<div class="sa-feature-pick-card {tone}">'
                f'<div class="sa-slip-card-top">'
                f'<div><div class="sa-slip-pick-lbl">Pick</div>'
                f'<div class="sa-slip-pick">{pick}</div></div>'
                f'<span class="sa-ticket {ticket}">{esc_html_one_line(ticket_raw)}</span></div>'
                f"{score_inner}"
                f'<div class="sa-slip-meta">'
                f'<span class="sa-pill">{board_esc}</span>'
                f'<span class="sa-pill">Conf {conf_s}</span>'
                f'<span class="sa-pill">Sample {samp_s}</span>'
                f'<span class="sa-pill">Same game tone as slip</span></div></div>'
            )
            .replace("\n", " ")
            .strip(),
            unsafe_allow_html=True,
        )


def render_hero_bet_cards(sharp_consensus_card: pd.DataFrame) -> None:
    st.markdown('<div class="section-kicker">TODAY\'S FEATURED BOARD</div>', unsafe_allow_html=True)
    if sharp_consensus_card.empty:
        st.info("No featured plays yet. Check again after data refresh.")
        return

    top_cards = sharp_consensus_card.head(3).copy()
    cols = st.columns(3)
    for i, (_, row) in enumerate(top_cards.iterrows()):
        with cols[i]:
            st.markdown(
                f"""
                <div class="bet-card">
                    <div class="bet-card-title">#{i+1} Best Bet</div>
                    <div class="bet-card-pick">{row["Suggested F5 Pick"]}</div>
                    <div class="bet-card-meta">{row["Matchup"]}</div>
                    <div class="bet-card-meta">Confidence: {float(row["Avg Confidence"]):.1f}% | Models Aligned: {int(row["Sharp Model Count"])}</div>
                    <div class="bet-card-meta">Consensus: {float(row["Consensus Score"]):.1f} | Sample: {float(row["Avg Sample"]):.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_priority_bet_cards(priority_bets: pd.DataFrame) -> None:
    st.markdown('<div class="section-kicker">BET THIS FIRST</div>', unsafe_allow_html=True)
    if priority_bets.empty:
        st.info("No priority bets available yet.")
        return
    top_cards = priority_bets.head(3).copy()
    cols = st.columns(3)
    for i, (_, row) in enumerate(top_cards.iterrows()):
        with cols[i]:
            st.markdown(
                f"""
                <div class="bet-card">
                    <div class="bet-card-title">Priority #{i+1}</div>
                    <div class="bet-card-pick">{row["Suggested F5 Pick"]}</div>
                    <div class="bet-card-meta">{row["Matchup"]}</div>
                    <div class="bet-card-meta">Priority Score: {float(row["Priority Score"]):.1f} | Confidence: {float(row["Avg Confidence"]):.1f}%</div>
                    <div class="bet-card-meta">Backtest Agree ROI: {float(row["Backtest ROI Agree %"]):.2f}% ({int(row["Backtest Models"])} models)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def prettify_tracker_view(df: pd.DataFrame, limit: Optional[int] = None) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["bet_date"] = pd.to_datetime(out["bet_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "confidence" in out.columns:
        out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce").round(1)
    if "edge_score" in out.columns:
        out["edge_score"] = pd.to_numeric(out["edge_score"], errors="coerce").round(2)
    if "status" in out.columns:
        status_map = {
            "win": "W",
            "loss": "L",
            "push": "P",
            "open": "OPEN",
            "void": "VOID",
        }
        out["status"] = out["status"].astype(str).str.lower().map(status_map).fillna(out["status"])
    out = out.sort_values(by=["bet_date", "logged_at"], ascending=[False, False])
    if limit is not None:
        out = out.head(limit)
    return out


def build_condensed_tracker_view(df: pd.DataFrame, limit: Optional[int] = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["bet_date"] = pd.to_datetime(out["bet_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["confidence"] = pd.to_numeric(out.get("confidence"), errors="coerce")
    out["edge_score"] = pd.to_numeric(out.get("edge_score"), errors="coerce")
    out["logged_at_dt"] = pd.to_datetime(out.get("logged_at"), errors="coerce")
    out["resolved_at_dt"] = pd.to_datetime(out.get("resolved_at"), errors="coerce")
    out["status_norm"] = out.get("status", "").astype(str).str.lower()

    grouped = (
        out.groupby(["bet_date", "matchup", "suggested_pick"], as_index=False)
        .agg(
            models_on_pick=("system_name", "nunique"),
            systems=("system_name", lambda s: ", ".join(sorted(set([str(x) for x in s if str(x).strip()])))),
            avg_confidence=("confidence", "mean"),
            avg_edge=("edge_score", "mean"),
            latest_logged=("logged_at_dt", "max"),
            latest_resolved=("resolved_at_dt", "max"),
            status=("status_norm", lambda s: pd.Series(s).iloc[-1] if len(s) else ""),
            result_note=("result_note", lambda s: pd.Series(s).dropna().astype(str).iloc[-1] if len(pd.Series(s).dropna()) else ""),
        )
        .rename(
            columns={
                "models_on_pick": "Models On Pick",
                "systems": "Systems",
                "avg_confidence": "Avg Confidence",
                "avg_edge": "Avg Edge",
                "latest_logged": "Logged At",
                "latest_resolved": "Resolved At",
                "status": "Status",
                "result_note": "Result",
            }
        )
    )
    grouped["Avg Confidence"] = pd.to_numeric(grouped["Avg Confidence"], errors="coerce").round(1)
    grouped["Avg Edge"] = pd.to_numeric(grouped["Avg Edge"], errors="coerce").round(2)
    grouped["Status"] = grouped["Status"].map(
        {"win": "W", "loss": "L", "push": "P", "open": "OPEN", "void": "VOID"}
    ).fillna(grouped["Status"])
    grouped["Logged At"] = pd.to_datetime(grouped["Logged At"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
    grouped["Resolved At"] = pd.to_datetime(grouped["Resolved At"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
    grouped = grouped.sort_values(by=["bet_date", "Logged At"], ascending=[False, False])
    if limit is not None:
        grouped = grouped.head(limit)
    return grouped.reset_index(drop=True)


def dedupe_tracker_tickets(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["logged_at_dt"] = pd.to_datetime(out.get("logged_at"), errors="coerce")
    out = out.sort_values(by=["logged_at_dt"], ascending=False).drop_duplicates(
        subset=["bet_date", "matchup", "suggested_pick"], keep="first"
    )
    return out


def summary_with_units(df: pd.DataFrame) -> Dict[str, Any]:
    s = summary_from_tracker(df)
    status = df["status"].astype(str).str.lower() if not df.empty and "status" in df.columns else pd.Series(dtype=str)
    wins = int((status == "win").sum())
    losses = int((status == "loss").sum())
    pushes = int((status == "push").sum())
    total_risked = wins + losses + pushes
    net_units = wins * 0.909 - losses
    roi = (net_units / total_risked * 100.0) if total_risked > 0 else 0.0
    s["net_units"] = round(net_units, 2)
    s["roi"] = round(roi, 2)
    return s


def split_priority_tiers(priority_bets_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if priority_bets_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    d = priority_bets_df.copy()
    core_mask = (
        (d["Sharp Model Count"] >= 2)
        & (d["Models On Bet"] >= 3)
        & (d["Avg Confidence"] >= 75)
    )
    core = d[core_mask].copy()
    if core.empty and not d.empty:
        core = d.head(min(2, len(d))).copy()
    value_adds = d.loc[~d.index.isin(core.index)].copy()
    # Keep value adds useful but still quality.
    value_adds = value_adds[
        (value_adds["Avg Confidence"] >= 66)
        & (value_adds["Avg Sample"] >= 0.42)
    ].copy()
    return core.reset_index(drop=True), value_adds.reset_index(drop=True)


def tracker_subset_for_picks(tracker_df: pd.DataFrame, picks_df: pd.DataFrame) -> pd.DataFrame:
    if tracker_df.empty or picks_df.empty:
        return pd.DataFrame()
    required_cols = {"Matchup", "Suggested F5 Pick"}
    if not required_cols.issubset(set(picks_df.columns)):
        return pd.DataFrame()
    join_keys = picks_df[["Matchup", "Suggested F5 Pick"]].drop_duplicates().rename(
        columns={"Matchup": "matchup", "Suggested F5 Pick": "suggested_pick"}
    )
    return tracker_df.merge(join_keys, on=["matchup", "suggested_pick"], how="inner")


def _empty_slip_summary() -> Dict[str, Any]:
    return {
        "record": "0-0-0",
        "win_rate": 0.0,
        "graded": 0,
        "open": 0,
        "wins": 0,
        "losses": 0,
        "pushes": 0,
    }


def today_summary_for_picks(tracker_df: pd.DataFrame, picks_df: pd.DataFrame) -> Dict[str, Any]:
    sub = tracker_subset_for_picks(tracker_df, picks_df)
    if sub.empty:
        return _empty_slip_summary()
    today_str = dt.date.today().strftime("%Y-%m-%d")
    sub = sub[sub["bet_date"].astype(str) == today_str].copy()
    if sub.empty:
        return _empty_slip_summary()
    sub = dedupe_tracker_tickets(sub)
    return summary_from_tracker(sub)


def build_today_system_standings(tracker_df: pd.DataFrame, system_names: List[str]) -> pd.DataFrame:
    """Per-board tracker rows for today's date only (named systems), deduped per matchup+pick."""
    if tracker_df.empty or not system_names:
        return pd.DataFrame(
            columns=["System", "Record", "Win Rate %", "Graded", "Wins", "Open", "Pushes"]
        )
    today_str = dt.date.today().strftime("%Y-%m-%d")
    t = tracker_df[
        (tracker_df["bet_date"].astype(str) == today_str)
        & (tracker_df["system_name"].astype(str).isin(system_names))
    ].copy()
    if t.empty:
        return pd.DataFrame(
            columns=["System", "Record", "Win Rate %", "Graded", "Wins", "Open", "Pushes"]
        )
    rows: List[Dict[str, Any]] = []
    for name in system_names:
        g = t[t["system_name"].astype(str) == name].copy()
        if g.empty:
            continue
        g = dedupe_tracker_tickets(g)
        s = summary_from_tracker(g)
        rows.append(
            {
                "System": name,
                "Record": s["record"],
                "Win Rate %": round(float(s["win_rate"]), 1),
                "Graded": int(s["graded"]),
                "Wins": int(s["wins"]),
                "Open": int(s["open"]),
                "Pushes": int(s["pushes"]),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["System", "Record", "Win Rate %", "Graded", "Wins", "Open", "Pushes"]
        )
    out = pd.DataFrame(rows)
    return out.sort_values(
        by=["Wins", "Graded", "Win Rate %"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def build_system_standings_for_window(
    tracker_df: pd.DataFrame,
    system_names: List[str],
    days: int,
    min_graded: int = 3,
) -> pd.DataFrame:
    """Windowed board standings for named systems, deduped tickets by day/matchup/pick."""
    cols = ["System", "Record", "Win Rate %", "Graded", "Wins", "Open", "Pushes"]
    if tracker_df.empty or not system_names:
        return pd.DataFrame(columns=cols)
    out = tracker_df.copy()
    out = out[out["system_name"].astype(str).isin(system_names)].copy()
    if out.empty:
        return pd.DataFrame(columns=cols)
    out["bet_date_dt"] = pd.to_datetime(out["bet_date"], errors="coerce")
    cutoff = pd.Timestamp(dt.date.today() - dt.timedelta(days=max(1, int(days)) - 1))
    out = out[out["bet_date_dt"] >= cutoff].copy()
    if out.empty:
        return pd.DataFrame(columns=cols)
    rows: List[Dict[str, Any]] = []
    for name in system_names:
        g = out[out["system_name"].astype(str) == name].copy()
        if g.empty:
            continue
        g = dedupe_tracker_tickets(g)
        s = summary_from_tracker(g)
        if int(s["graded"]) < int(min_graded):
            continue
        rows.append(
            {
                "System": name,
                "Record": s["record"],
                "Win Rate %": round(float(s["win_rate"]), 1),
                "Graded": int(s["graded"]),
                "Wins": int(s["wins"]),
                "Open": int(s["open"]),
                "Pushes": int(s["pushes"]),
            }
        )
    if not rows:
        return pd.DataFrame(columns=cols)
    d = pd.DataFrame(rows)
    return d.sort_values(
        by=["Win Rate %", "Wins", "Graded"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def backtest_record_for_models(backtest_df: pd.DataFrame, model_names: List[str]) -> Dict[str, Any]:
    if backtest_df.empty or not model_names:
        return {"record": "0-0-0", "win_rate": 0.0, "graded": 0}
    sub = backtest_df[backtest_df["System"].isin(model_names)].copy()
    if sub.empty:
        return {"record": "0-0-0", "win_rate": 0.0, "graded": 0}
    wins = int(pd.to_numeric(sub["Wins"], errors="coerce").fillna(0).sum())
    losses = int(pd.to_numeric(sub["Losses"], errors="coerce").fillna(0).sum())
    pushes = int(pd.to_numeric(sub["Pushes"], errors="coerce").fillna(0).sum())
    graded = wins + losses
    win_rate = (wins / graded * 100.0) if graded > 0 else 0.0
    return {"record": f"{wins}-{losses}-{pushes}", "win_rate": win_rate, "graded": graded}


def live_record_for_models(
    tracker_df: pd.DataFrame,
    model_names: List[str],
    lookback_days: int = 60,
) -> Dict[str, Any]:
    if tracker_df.empty or not model_names:
        return {"record": "0-0-0", "win_rate": 0.0, "graded": 0}
    out = tracker_df.copy()
    out = out[out["system_name"].astype(str).isin(model_names)].copy()
    if out.empty:
        return {"record": "0-0-0", "win_rate": 0.0, "graded": 0}
    out["bet_date_dt"] = pd.to_datetime(out["bet_date"], errors="coerce")
    cutoff = pd.Timestamp(dt.date.today() - dt.timedelta(days=lookback_days))
    out = out[out["bet_date_dt"] >= cutoff].copy()
    if out.empty:
        return {"record": "0-0-0", "win_rate": 0.0, "graded": 0}
    # De-dupe to one ticket per game/pick/day to avoid overstating results when many models overlap.
    out["logged_at_dt"] = pd.to_datetime(out.get("logged_at"), errors="coerce")
    out = out.sort_values(by=["logged_at_dt"], ascending=False).drop_duplicates(
        subset=["bet_date", "matchup", "suggested_pick"], keep="first"
    )
    return summary_from_tracker(out)


def featured_system_snapshot(
    featured_system_name: str,
    system_tables: Dict[str, pd.DataFrame],
    tracker_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    system_key = None
    for s in SYSTEMS:
        if s.name == featured_system_name:
            system_key = s.key
            break
    if system_key is None:
        return pd.DataFrame(), {"record": "0-0-0", "win_rate": 0.0, "graded": 0, "open": 0}
    df = system_tables.get(system_key, pd.DataFrame())
    if df.empty:
        return pd.DataFrame(), {"record": "0-0-0", "win_rate": 0.0, "graded": 0, "open": 0}
    featured_bets = df[df["Qualifies Strategy"] == "Yes"].copy()
    featured_tracker = (
        tracker_df[tracker_df["system_name"].astype(str) == featured_system_name].copy()
        if not tracker_df.empty
        else pd.DataFrame()
    )
    return featured_bets, summary_from_tracker(featured_tracker)


def build_live_status_board(
    picks_df: pd.DataFrame,
    tracker_df: pd.DataFrame,
    bet_date: dt.date,
) -> pd.DataFrame:
    if picks_df.empty:
        return pd.DataFrame()

    try:
        score_map = fetch_scores_for_date(bet_date)
    except Exception:
        score_map = {}

    latest_ticket: Dict[Tuple[str, str], str] = {}
    if not tracker_df.empty:
        t = tracker_df.copy()
        t["logged_at_dt"] = pd.to_datetime(t.get("logged_at"), errors="coerce")
        t = t.sort_values(by=["logged_at_dt"], ascending=False)
        for _, r in t.iterrows():
            key = (str(r.get("matchup", "")), str(r.get("suggested_pick", "")))
            if key in latest_ticket:
                continue
            st = str(r.get("status", "")).lower()
            latest_ticket[key] = {"win": "W", "loss": "L", "push": "P", "open": "OPEN", "void": "VOID"}.get(st, st.upper())

    rows: List[Dict[str, Any]] = []
    seen = set()
    for _, r in picks_df.iterrows():
        matchup = str(r.get("Matchup", ""))
        pick = str(r.get("Suggested F5 Pick", ""))
        if not matchup or not pick:
            continue
        dedupe_key = (matchup, pick)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        away_team = matchup.split(" @ ")[0].strip() if " @ " in matchup else ""
        home_team = matchup.split(" @ ")[1].strip() if " @ " in matchup else ""
        game_key = f"{canonical_team_key(away_team)}|{canonical_team_key(home_team)}"
        game = score_map.get(game_key, {})

        status_text = str(game.get("status", "No Feed"))
        inning = game.get("current_inning")
        inning_state = str(game.get("inning_state", "")).lower()
        if inning is None:
            inning_label = "-"
        else:
            inning_prefix = {"top": "Top", "middle": "Mid", "bottom": "Bot", "end": "End"}.get(inning_state, "")
            inning_label = f"{inning_prefix} {inning}".strip()

        away_total = game.get("away_total", None)
        home_total = game.get("home_total", None)
        live_score = "-"
        if away_total is not None and home_total is not None and game:
            live_score = f"{away_team} {away_total} - {home_team} {home_total}"

        st_lower = status_text.lower()
        can_bet = "Yes" if any(x in st_lower for x in ["scheduled", "pre-game", "pregame", "warmup"]) else "No"

        ticket_status = latest_ticket.get(dedupe_key, "OPEN")
        if ticket_status == "OPEN" and game.get("can_grade"):
            temp_map = {game_key: game}
            graded = grade_pick_from_score_map(matchup, pick, temp_map)
            if graded == "win":
                ticket_status = "W"
            elif graded == "loss":
                ticket_status = "L"
            elif graded == "push":
                ticket_status = "P"

        rows.append(
            {
                "Matchup": matchup,
                "Pick": pick,
                "Can Bet?": can_bet,
                "Game Status": status_text,
                "Inning": inning_label,
                "Live Score": live_score,
                "Ticket Status": ticket_status,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def game_state_label_for_matchup(matchup: str, score_map: Dict[str, Dict[str, Any]]) -> str:
    away_team = matchup.split(" @ ")[0].strip() if " @ " in matchup else ""
    home_team = matchup.split(" @ ")[1].strip() if " @ " in matchup else ""
    game_key = f"{canonical_team_key(away_team)}|{canonical_team_key(home_team)}"
    game = score_map.get(game_key)
    if not game:
        return "Status Unavailable"
    status_text = str(game.get("status", ""))
    status_lower = status_text.lower()
    if any(x in status_lower for x in ["scheduled", "pre-game", "pregame", "warmup"]):
        return "Not Started"
    away_f5 = int(game.get("away_f5", 0))
    home_f5 = int(game.get("home_f5", 0))
    fg = full_game_secondary_fields(game)
    f5_tag = "F5 FINAL" if game.get("can_grade") else "F5 live"
    base = f"F5 {away_f5}-{home_f5} ({f5_tag})"
    if fg["fg_mode"] == "final":
        return f"{base} · Full {fg['fg_score']} · FINAL"
    if fg["fg_mode"] == "live":
        inn = fg.get("fg_inning_text") or ""
        tail = f" · LIVE · {inn}" if inn else " · LIVE"
        return f"{base} · Full {fg['fg_score']}{tail}"
    return status_text or "Status Unavailable"


def annotate_with_game_state(df: pd.DataFrame, score_map: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    if df.empty or "Matchup" not in df.columns:
        return df
    out = df.copy()
    out["Game"] = out["Matchup"].astype(str).apply(lambda m: game_state_label_for_matchup(m, score_map))
    return out


def annotate_with_ticket_status(df: pd.DataFrame, tracker_df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Matchup" not in df.columns or "Suggested F5 Pick" not in df.columns:
        return df
    out = df.copy()
    out["Ticket"] = "OPEN"
    if tracker_df.empty:
        return out
    t = tracker_df.copy()
    t["logged_at_dt"] = pd.to_datetime(t.get("logged_at"), errors="coerce")
    t = t.sort_values(by=["logged_at_dt"], ascending=False)
    latest: Dict[Tuple[str, str], str] = {}
    status_map = {"win": "W", "loss": "L", "push": "P", "open": "OPEN", "void": "VOID"}
    for _, r in t.iterrows():
        key = (str(r.get("matchup", "")), str(r.get("suggested_pick", "")))
        if key in latest:
            continue
        latest[key] = status_map.get(str(r.get("status", "")).lower(), str(r.get("status", "")).upper())
    out["Ticket"] = out.apply(
        lambda r: latest.get((str(r.get("Matchup", "")), str(r.get("Suggested F5 Pick", ""))), "OPEN"),
        axis=1,
    )
    return out


def annotate_for_command_center(df: pd.DataFrame, score_map: Dict[str, Dict[str, Any]], tracker_df: pd.DataFrame) -> pd.DataFrame:
    out = annotate_with_game_state(df, score_map)
    out = annotate_with_ticket_status(out, tracker_df)
    return out


def build_today_betslip(
    boards: Dict[str, pd.DataFrame],
    score_map: Dict[str, Dict[str, Any]],
    tracker_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for label, df in boards.items():
        if df is None or df.empty:
            continue
        if not {"Matchup", "Suggested F5 Pick"}.issubset(set(df.columns)):
            continue
        for _, r in df.iterrows():
            rows.append(
                {
                    "Matchup": r.get("Matchup"),
                    "Suggested F5 Pick": r.get("Suggested F5 Pick"),
                    "Avg Confidence": r.get("Avg Confidence", r.get("Confidence")),
                    "Avg Sample": r.get("Avg Sample", r.get("Sample Reliability")),
                    "Priority Score": r.get("Priority Score", r.get("100C Score", r.get("Backtest Pick Score", r.get("Consensus Score")))),
                    "Source": label,
                }
            )
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    d["Avg Confidence"] = pd.to_numeric(d["Avg Confidence"], errors="coerce")
    d["Avg Sample"] = pd.to_numeric(d["Avg Sample"], errors="coerce")
    d["Priority Score"] = pd.to_numeric(d["Priority Score"], errors="coerce")
    out = (
        d.groupby(["Matchup", "Suggested F5 Pick"], as_index=False)
        .agg(
            avg_confidence=("Avg Confidence", "mean"),
            avg_sample=("Avg Sample", "mean"),
            priority_score=("Priority Score", "max"),
            source_tags=("Source", lambda s: ", ".join(sorted(set([str(x) for x in s if str(x).strip()])))),
        )
        .rename(
            columns={
                "avg_confidence": "Avg Confidence",
                "avg_sample": "Avg Sample",
                "priority_score": "Priority Score",
                "source_tags": "Source Tags",
            }
        )
    )
    out = out.sort_values(by=["Priority Score", "Avg Confidence", "Avg Sample"], ascending=[False, False, False]).reset_index(drop=True)
    out = annotate_for_command_center(out, score_map, tracker_df)
    return out


def main() -> None:
    st.set_page_config(page_title="MLB F5 Systems Lab", layout="wide")
    if not st.session_state.get("_sa_scores_sess_clear"):
        fetch_scores_for_date.clear()
        st.session_state._sa_scores_sess_clear = True

    render_app_theme()
    st.markdown(
        '<div class="main-title sa-app-title">F5 Systems Lab</div>'
        '<div class="sa-app-sub">Today’s plays · tracker · boards</div>',
        unsafe_allow_html=True,
    )

    admin_rebuild_requested = False
    with st.expander("Controls", expanded=False):
        c1, c2 = st.columns([1.3, 1.7])
        with c1:
            st.caption("MLB slate auto-refreshes when stale.")
            if st.button("Refresh MLB slate now", use_container_width=True, key="sa_refresh_slate"):
                st.session_state.refresh_nonce = int(st.session_state.get("refresh_nonce", 0)) + 1
        with c2:
            st.caption("Backtest cache (admin)")
            admin_key_input = st.text_input("Admin key", type="password", key="admin_backtest_key_input")
            admin_ok = admin_key_input.strip() == ADMIN_BACKTEST_KEY
            if admin_ok:
                admin_rebuild_requested = st.button("Rebuild backtest cache", use_container_width=True, key="sa_admin_bt")
            else:
                st.caption("Locked")

    if "refresh_nonce" not in st.session_state:
        st.session_state.refresh_nonce = 0
    if "applied_fetch_nonce" not in st.session_state:
        st.session_state.applied_fetch_nonce = -1

    cache_matchups, cache_parse_mode, cache_saved_at = load_matchups_cache()
    should_fetch_matchups = (
        (not cache_matchups)
        or matchups_cache_is_stale(cache_saved_at)
        or (st.session_state.refresh_nonce != st.session_state.applied_fetch_nonce)
    )
    matchups: List[Matchup] = []
    parse_mode = ""
    last_pull_label = ""
    if should_fetch_matchups:
        with st.spinner("Syncing MLB slate…"):
            try:
                matchups, parse_mode = fetch_matchups(st.session_state.refresh_nonce)
                save_matchups_cache(matchups, parse_mode)
                last_pull_label = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.applied_fetch_nonce = st.session_state.refresh_nonce
            except Exception as exc:
                if cache_matchups:
                    matchups = cache_matchups
                    parse_mode = f"{cache_parse_mode} (cached fallback)"
                    last_pull_label = cache_saved_at or dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.warning(f"Live pull failed, using cached matchup data. ({str(exc)[:120]})")
                    st.session_state.applied_fetch_nonce = st.session_state.refresh_nonce
                else:
                    st.error(str(exc))
                    st.stop()
    else:
        matchups = cache_matchups
        parse_mode = f"{cache_parse_mode} (cached)"
        last_pull_label = cache_saved_at or dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    system_tables: Dict[str, pd.DataFrame] = {s.key: build_system_table(matchups, s) for s in SYSTEMS}

    tracker_df = load_tracker()
    tracker_df = grade_f5_bets(tracker_df)
    total_added = 0
    processed_dates = 0
    skipped_dates = 0
    backfill_dates = build_missing_tracking_dates(tracker_df, max_backfill_days=30)
    if len(backfill_dates) > 1:
        st.caption(f"Tracker catch-up: {len(backfill_dates)} day(s).")
    for bet_day in backfill_dates:
        if bet_day == dt.date.today():
            day_tables = system_tables
        else:
            try:
                historical_matchups = parse_from_stats_api(bet_day)
                day_tables = {s.key: build_system_table(historical_matchups, s) for s in SYSTEMS}
            except Exception:
                skipped_dates += 1
                continue
        for s in SYSTEMS:
            tracker_df, added = add_bets_to_tracker_for_date(day_tables[s.key], s.name, tracker_df, bet_day)
            total_added += added
        processed_dates += 1
    tracker_df = grade_f5_bets(tracker_df)
    save_tracker(tracker_df)
    if total_added > 0:
        st.success(f"Logged {total_added} new tracker row(s) across {processed_dates} day(s).")
    if skipped_dates > 0:
        st.info(f"Skipped {skipped_dates} day(s) (MLB fetch).")

    lb = build_leaderboard(tracker_df)
    scale_df = build_scale_table(tracker_df)
    top_n = lb.head(5)
    top3 = lb.head(3)
    consensus_df = build_consensus_bets(system_tables, tracker_df)
    sharp_models = build_sharp_model_list(tracker_df, max_models=5)
    sharp_consensus_card = build_sharp_consensus_card(consensus_df, sharp_models, max_bets=8)
    sharp_model_picks = build_sharp_model_picks(system_tables, sharp_models, max_rows=8)
    card_80, card_90, card_100 = build_confidence_cards(consensus_df)
    backtest_generated_at = st.session_state.get("backtest_generated_at")
    backtest_multi = st.session_state.get("backtest_multi", {})
    if not backtest_multi:
        cached_multi, cached_generated_at = load_backtest_cache()
        if cached_multi:
            backtest_multi = cached_multi
            st.session_state["backtest_multi"] = backtest_multi
            st.session_state["backtest_generated_at"] = cached_generated_at
            backtest_generated_at = cached_generated_at

    should_run_backtest = admin_rebuild_requested
    if should_run_backtest:
        with st.spinner("Running 1Y, 2Y, and 3Y backtests in background for this session..."):
            backtest_multi = run_system_backtest_multi()
            st.session_state["backtest_multi"] = backtest_multi
            save_backtest_cache(backtest_multi)
            backtest_generated_at = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state["backtest_generated_at"] = backtest_generated_at
            if admin_rebuild_requested:
                st.success("Admin rebuild complete. Backtest cache refreshed.")
    backtest_1y = backtest_multi.get("1Y", pd.DataFrame()) if isinstance(backtest_multi, dict) else pd.DataFrame()
    backtest_2y = backtest_multi.get("2Y", pd.DataFrame()) if isinstance(backtest_multi, dict) else pd.DataFrame()
    backtest_3y = backtest_multi.get("3Y", pd.DataFrame()) if isinstance(backtest_multi, dict) else pd.DataFrame()
    backtest_df = backtest_1y
    backtest_lookup: Dict[str, Dict[str, Any]] = {}
    if not backtest_1y.empty and "System" in backtest_1y.columns:
        backtest_lookup = {
            str(r["System"]): r
            for _, r in backtest_1y.iterrows()
        }
    priority_bets_df = build_bet_this_first_card(sharp_consensus_card, backtest_lookup, max_bets=None)
    core_plays_df, value_adds_df = split_priority_tiers(priority_bets_df)
    backtest_only_bets_df = build_backtest_only_bets(system_tables, backtest_1y, backtest_2y, backtest_3y)
    confidence_100_bets_df = build_100_confidence_card(consensus_df)
    card_tracker_df = load_card_tracker()
    card_tables = {
        "100% Confidence": confidence_100_bets_df,
        "Core Plays": core_plays_df,
        "Value Adds": value_adds_df,
        "Live+Backtest": priority_bets_df,
        "Backtest-Only": backtest_only_bets_df,
        "Sharp Consensus": sharp_consensus_card,
    }
    card_added = 0
    for card_name, card_df in card_tables.items():
        card_tracker_df, added = add_card_bets_to_tracker_for_date(card_df, card_name, card_tracker_df, dt.date.today())
        card_added += added
    card_tracker_df = grade_f5_bets(card_tracker_df)
    save_card_tracker(card_tracker_df)
    live_bt_models: List[str] = []
    if not priority_bets_df.empty and "Systems" in priority_bets_df.columns:
        model_set = set()
        for s in priority_bets_df["Systems"].astype(str):
            model_set.update([p.strip() for p in s.split(",") if p.strip()])
        live_bt_models = sorted(model_set)
    backtest_only_models: List[str] = []
    if not backtest_only_bets_df.empty and "Backtested Models" in backtest_only_bets_df.columns:
        model_set = set()
        for s in backtest_only_bets_df["Backtested Models"].astype(str):
            model_set.update([p.strip() for p in s.split(",") if p.strip()])
        backtest_only_models = sorted(model_set)
    sharp_consensus_models = sorted(set(sharp_models))
    live_bt_summary = live_record_for_models(tracker_df, live_bt_models, lookback_days=60)
    bt_only_summary = live_record_for_models(tracker_df, backtest_only_models, lookback_days=60)
    sharp_live_summary = live_record_for_models(tracker_df, sharp_consensus_models, lookback_days=60)
    live_bt_backtest = backtest_record_for_models(backtest_1y, live_bt_models)
    bt_only_backtest = backtest_record_for_models(backtest_1y, backtest_only_models)
    confidence_100_tracker = dedupe_tracker_tickets(tracker_subset_for_picks(tracker_df, confidence_100_bets_df))
    confidence_100_summary = summary_with_units(confidence_100_tracker)
    try:
        today_score_map = fetch_scores_for_date(dt.date.today())
    except Exception:
        today_score_map = {}
    confidence_100_show = annotate_for_command_center(confidence_100_bets_df, today_score_map, tracker_df)
    featured_bets_df, featured_summary = featured_system_snapshot(
        FEATURED_SYSTEM_NAME, system_tables, tracker_df
    )
    featured_show = annotate_for_command_center(featured_bets_df, today_score_map, tracker_df)
    system_names = [s.name for s in SYSTEMS]
    system_name_to_key = {s.name: s.key for s in SYSTEMS}
    today_board_standings = build_today_system_standings(tracker_df, system_names)
    week_board_standings = build_system_standings_for_window(tracker_df, system_names, days=7, min_graded=3)
    month_board_standings = build_system_standings_for_window(tracker_df, system_names, days=30, min_graded=6)
    week_model_name = str(week_board_standings.iloc[0]["System"]) if not week_board_standings.empty else ""
    month_model_name = str(month_board_standings.iloc[0]["System"]) if not month_board_standings.empty else ""
    week_model_df = (
        system_tables.get(system_name_to_key.get(week_model_name, ""), pd.DataFrame()).copy()
        if week_model_name
        else pd.DataFrame()
    )
    if not week_model_df.empty and "Qualifies Strategy" in week_model_df.columns:
        week_model_df = week_model_df[week_model_df["Qualifies Strategy"] == "Yes"].copy()
    week_model_show = annotate_for_command_center(week_model_df, today_score_map, tracker_df)
    month_model_df = (
        system_tables.get(system_name_to_key.get(month_model_name, ""), pd.DataFrame()).copy()
        if month_model_name
        else pd.DataFrame()
    )
    if not month_model_df.empty and "Qualifies Strategy" in month_model_df.columns:
        month_model_df = month_model_df[month_model_df["Qualifies Strategy"] == "Yes"].copy()
    month_model_show = annotate_for_command_center(month_model_df, today_score_map, tracker_df)
    today_featured_summary = today_summary_for_picks(tracker_df, featured_bets_df)
    today_betslip_df = build_today_betslip(
        {
            "100% Confidence": confidence_100_bets_df,
            "Core": core_plays_df,
            "Value": value_adds_df,
            "Backtest-Only": backtest_only_bets_df,
        },
        today_score_map,
        tracker_df,
    )
    today_slip_summary = today_summary_for_picks(tracker_df, today_betslip_df)
    today_best_tracker = dedupe_tracker_tickets(tracker_subset_for_picks(tracker_df, today_betslip_df))
    today_best_tracker_summary = summary_from_tracker(today_best_tracker)
    full_tracker_summary = summary_from_tracker(dedupe_tracker_tickets(tracker_df))
    today_100_summary = today_summary_for_picks(tracker_df, confidence_100_bets_df)

    total_qualified = sum(int((df["Qualifies Strategy"] == "Yes").sum()) for df in system_tables.values() if not df.empty)
    with st.expander("Slate diagnostics", expanded=False):
        d1, d2, d3 = st.columns(3)
        with d1:
            st.metric("Games parsed", len(matchups))
        with d2:
            st.metric("Qualified signals", total_qualified)
        with d3:
            st.metric("Boards", len(SYSTEMS))
        st.caption(f"Source: {MLB_URL} · Parse: {parse_mode} · Last slate sync: {last_pull_label}")

    tabs = st.tabs(
        [
            "Command Center",
            "Confidence 80%+",
            "Confidence 90%+",
            "Confidence 100%",
            "Scale Lab",
            "Leaderboard",
            "Backtest",
        ]
        + [s.name for s in SYSTEMS]
    )

    with tabs[0]:
        slip_ns = (
            int((today_betslip_df["Game"] == "Not Started").sum())
            if not today_betslip_df.empty and "Game" in today_betslip_df.columns
            else 0
        )
        slip_open = (
            int((today_betslip_df["Ticket"] == "OPEN").sum())
            if not today_betslip_df.empty and "Ticket" in today_betslip_df.columns
            else 0
        )

        render_section_header(
            "Live F5 scores · slip games",
            "One mini-board per matchup on **today’s merged slip**. **Border / glow color** is the same key as **Top picks** "
            f"and **{FEATURED_SYSTEM_NAME}** when the matchup matches (e.g. Braves game = same accent everywhere).",
        )
        if today_betslip_df.empty:
            st.info("No slip rows yet — refresh the slate after models qualify plays to see live F5 totals here.")
        else:
            render_scoreboard_strip_from_slip(today_betslip_df, today_score_map)
            st.caption("Horizontal scroll on small screens · F5 = innings 1–5 from MLB.")

        render_section_header(
            "Top picks today",
            "Merged slip (100% + Core + Value + Backtest-only). Each card repeats the scorebox so pick, ticket, and "
            "**source tags** stay next to the game.",
        )
        if today_betslip_df.empty:
            st.caption("Nothing to rank until at least one board logs a qualified pick on the slip.")
        else:
            render_betslip_cards_html(today_betslip_df, today_score_map)

        render_section_header(
            f"Featured · {FEATURED_SYSTEM_NAME}",
            "Today’s qualified rows for this board only. **Daily** numbers are tracker tickets dated today for these picks; "
            "**all-time** is every logged ticket for this board in the tracker.",
        )
        hf1, hf2, hf3, hf4 = st.columns(4)
        with hf1:
            st.metric("Daily W-L (tracker)", today_featured_summary["record"])
        with hf2:
            st.metric("Daily WR", f"{today_featured_summary['win_rate']:.1f}%")
        with hf3:
            st.metric("All-time W-L", str(featured_summary.get("record", "0-0-0")))
        with hf4:
            st.metric(
                "All-time WR / open",
                f"{float(featured_summary.get('win_rate', 0.0)):.1f}% · {int(featured_summary.get('open', 0))} open",
            )
        if featured_show.empty:
            st.caption("No qualified strategy rows for this board today.")
        else:
            render_featured_pick_cards_html(featured_show, today_score_map, FEATURED_SYSTEM_NAME)

        with st.expander(f"Win–loss history · {FEATURED_SYSTEM_NAME} (tracker)", expanded=False):
            ft_hist = (
                tracker_df[tracker_df["system_name"].astype(str) == FEATURED_SYSTEM_NAME].copy()
                if not tracker_df.empty
                else pd.DataFrame()
            )
            if ft_hist.empty:
                st.caption("No tracker rows for this board yet.")
            else:
                sort_cols = [c for c in ("bet_date", "logged_at") if c in ft_hist.columns]
                if sort_cols:
                    ft_hist = ft_hist.sort_values(by=sort_cols, ascending=[False] * len(sort_cols))
                st.dataframe(prettify_tracker_view(ft_hist, 25), use_container_width=True, height=280)

        st.divider()
        render_section_header(
            "Best model of the week / month",
            "Ranked from tracker history (named systems only, deduped by date + matchup + pick).",
        )
        if week_board_standings.empty:
            st.metric("Best this week", "No qualified model yet")
            st.caption("Need at least 3 graded bets in the last 7 days.")
        else:
            wk = week_board_standings.iloc[0]
            st.metric(
                "Best this week",
                f"{wk['System']} · {wk['Record']}",
                f"{float(wk['Win Rate %']):.1f}% WR · {int(wk['Graded'])} graded",
            )
            st.caption(f"Today's qualified bets in {wk['System']}:")
            if week_model_show.empty:
                st.caption("No qualified plays today for this model.")
            else:
                render_featured_pick_cards_html(week_model_show, today_score_map, str(wk["System"]))

        st.divider()
        if month_board_standings.empty:
            st.metric("Best this month", "No qualified model yet")
            st.caption("Need at least 6 graded bets in the last 30 days.")
        else:
            mo = month_board_standings.iloc[0]
            st.metric(
                "Best this month",
                f"{mo['System']} · {mo['Record']}",
                f"{float(mo['Win Rate %']):.1f}% WR · {int(mo['Graded'])} graded",
            )
            st.caption(f"Today's qualified bets in {mo['System']}:")
            if month_model_show.empty:
                st.caption("No qualified plays today for this model.")
            else:
                render_featured_pick_cards_html(month_model_show, today_score_map, str(mo["System"]))

        sweep_today = (
            today_board_standings[
                (pd.to_numeric(today_board_standings["Graded"], errors="coerce") >= 3)
                & (pd.to_numeric(today_board_standings["Win Rate %"], errors="coerce") >= 99.9)
            ].copy()
            if not today_board_standings.empty
            else pd.DataFrame()
        )
        if not sweep_today.empty:
            sw = sweep_today.iloc[0]
            st.success(
                f"Today sweep: {sw['System']} went {sw['Record']} ({float(sw['Win Rate %']):.1f}% WR)."
            )
        else:
            st.caption("No 3-0+ system sweep detected today.")

        with st.expander("More detail: best bets tracker + 100% card + day summary", expanded=False):
            render_command_center_hero(
                today_slip_summary,
                full_tracker_summary,
                last_pull_label,
                len(today_betslip_df),
                slip_ns,
                slip_open,
            )
            st.subheader("Best bets tracker (all models, deduped)")
            if today_best_tracker.empty:
                st.caption("No tracker rows matched to today’s merged best-bets list yet.")
            else:
                cts1, cts2, cts3 = st.columns(3)
                cts1.metric("Record", today_best_tracker_summary["record"])
                cts2.metric("Win rate", f"{today_best_tracker_summary['win_rate']:.1f}%")
                cts3.metric("Graded", today_best_tracker_summary["graded"])
                tracker_show = build_condensed_tracker_view(today_best_tracker, limit=30)
                st.dataframe(
                    tracker_show[
                        [
                            "bet_date",
                            "Models On Pick",
                            "matchup",
                            "suggested_pick",
                            "Avg Confidence",
                            "Status",
                            "Result",
                            "Systems",
                        ]
                    ],
                    use_container_width=True,
                    height=260,
                )
            st.caption(
                "Tracker is built from all qualifying best bets across boards (duplicates removed by matchup + pick). "
                "Models On Pick shows how many systems aligned on each bet."
            )

            st.subheader("100% consensus")
            st.caption(
                f"Today’s 100% ticket: {today_100_summary['record']} ({today_100_summary['win_rate']:.1f}% WR) · "
                f"Tracked (these picks): {confidence_100_summary['record']} · ROI {confidence_100_summary['roi']:.1f}% · "
                f"Units {confidence_100_summary['net_units']:.2f}"
            )
            if confidence_100_bets_df.empty:
                st.info("No 100% consensus-qualified bets today.")
            else:
                st.dataframe(
                    confidence_100_show,
                    use_container_width=True,
                    height=min(340, 120 + 36 * len(confidence_100_bets_df)),
                )

    with tabs[1]:
        st.subheader("Confidence 80%+ System")
        if card_80.empty:
            st.info("No consensus bets at 80%+ confidence today.")
        else:
            st.dataframe(card_80, use_container_width=True, height=320)
            tracked_80 = tracker_df[tracker_df["matchup"].isin(card_80["Matchup"])].copy() if not tracker_df.empty else pd.DataFrame()
            s80 = summary_from_tracker(tracked_80)
            st.caption(f"Tracked record: {s80['record']} | Win Rate: {s80['win_rate']:.1f}% | Graded: {s80['graded']}")

    with tabs[2]:
        st.subheader("Confidence 90%+ System")
        if card_90.empty:
            st.info("No consensus bets at 90%+ confidence today.")
        else:
            st.dataframe(card_90, use_container_width=True, height=320)
            tracked_90 = tracker_df[tracker_df["matchup"].isin(card_90["Matchup"])].copy() if not tracker_df.empty else pd.DataFrame()
            s90 = summary_from_tracker(tracked_90)
            st.caption(f"Tracked record: {s90['record']} | Win Rate: {s90['win_rate']:.1f}% | Graded: {s90['graded']}")

    with tabs[3]:
        st.subheader("Confidence 100% System")
        if card_100.empty:
            st.info("No consensus bets at 100% confidence today.")
        else:
            st.dataframe(card_100, use_container_width=True, height=320)
            tracked_100 = tracker_df[tracker_df["matchup"].isin(card_100["Matchup"])].copy() if not tracker_df.empty else pd.DataFrame()
            s100 = summary_from_tracker(tracked_100)
            st.caption(f"Tracked record: {s100['record']} | Win Rate: {s100['win_rate']:.1f}% | Graded: {s100['graded']}")

    with tabs[4]:
        st.subheader("Scale Lab")
        st.caption("Guidance based on graded sample size and win rate. Use this to size systems up responsibly.")
        if scale_df.empty:
            st.info("No scale data yet. Let systems accumulate graded bets.")
        else:
            st.dataframe(scale_df, use_container_width=True, height=340)
        st.markdown("- `Sandbox`: stay small until sample improves")
        st.markdown("- `Pilot`: reasonable early edge, keep controlled sizing")
        st.markdown("- `Scale Candidate`: close to full-scale thresholds")
        st.markdown("- `Scale`: strong sample and sustained edge")

    with tabs[5]:
        st.subheader("Leaderboard")
        if lb.empty:
            st.info("No graded system records yet.")
        else:
            st.dataframe(lb, use_container_width=True, height=420)

    with tabs[6]:
        st.subheader("Historical Backtest")
        st.caption("Runs each system over historical daily slates using F5 outcomes and flat 1u sizing.")
        if backtest_generated_at:
            st.caption(f"Showing saved cache from: {backtest_generated_at}")
        if backtest_1y.empty and backtest_2y.empty and backtest_3y.empty:
            st.info("Open **Controls** (top expander) to rebuild the backtest cache when needed.")
        else:
            bt_tabs = st.tabs(["1Y", "2Y", "3Y"])
            with bt_tabs[0]:
                st.dataframe(backtest_1y, use_container_width=True, height=360)
                if not backtest_1y.empty:
                    st.download_button(
                        "Download 1Y Backtest CSV",
                        data=backtest_1y.to_csv(index=False),
                        file_name="f5_backtest_1y.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
            with bt_tabs[1]:
                st.dataframe(backtest_2y, use_container_width=True, height=360)
                if not backtest_2y.empty:
                    st.download_button(
                        "Download 2Y Backtest CSV",
                        data=backtest_2y.to_csv(index=False),
                        file_name="f5_backtest_2y.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
            with bt_tabs[2]:
                st.dataframe(backtest_3y, use_container_width=True, height=360)
                if not backtest_3y.empty:
                    st.download_button(
                        "Download 3Y Backtest CSV",
                        data=backtest_3y.to_csv(index=False),
                        file_name="f5_backtest_3y.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

    for i, s in enumerate(SYSTEMS, start=7):
        with tabs[i]:
            render_section_header(s.name, "System rules, active picks, and tracked outcomes.")
            for rule in s.rules:
                st.markdown(f"- {rule}")

            df = system_tables[s.key]
            if df.empty:
                st.info("No games parsed for this system today.")
                continue

            sys_tracker = tracker_df[tracker_df["system_name"] == s.name].copy() if not tracker_df.empty else pd.DataFrame()
            summary = summary_from_tracker(sys_tracker)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("System Record", summary["record"])
            m2.metric("Win Rate", f"{summary['win_rate']:.1f}%")
            m3.metric("Open Bets", summary["open"])
            m4.metric("Today Qualified", int((df["Qualifies Strategy"] == "Yes").sum()))
            bt = backtest_lookup.get(s.name)
            if bt is not None:
                b1, b2, b3 = st.columns(3)
                bt2 = backtest_2y[backtest_2y["System"] == s.name].head(1)
                bt3 = backtest_3y[backtest_3y["System"] == s.name].head(1)
                roi2 = float(bt2["ROI %"].iloc[0]) if not bt2.empty else 0.0
                roi3 = float(bt3["ROI %"].iloc[0]) if not bt3.empty else 0.0
                b1.metric("Backtest Win Rate (1Y)", f"{float(bt['Win Rate %']):.1f}%")
                b2.metric("Backtest ROI (1Y)", f"{float(bt['ROI %']):.2f}%")
                b3.metric("Backtest ROI (2Y/3Y)", f"{roi2:.2f}% / {roi3:.2f}%")

            show_df = annotate_with_game_state(df, today_score_map)
            show_cols = [
                "System",
                "Matchup",
                "Away Team",
                "Home Team",
                "Game",
                "Away Pitcher",
                "Home Pitcher",
                "Away ERA",
                "Home ERA",
                "Away WHIP",
                "Home WHIP",
                "Away K",
                "Home K",
                "Away GS",
                "Home GS",
                "Away IP",
                "Home IP",
                "Sample Reliability",
                "ERA Diff",
                "Edge Score",
                "Confidence",
                "Suggested F5 Pick",
                "Qualifies Strategy",
                "Edge Band",
            ]

            styled = (
                show_df[show_cols]
                .style
                .format(
                    {
                        "Away ERA": "{:.2f}",
                        "Home ERA": "{:.2f}",
                        "Away WHIP": "{:.2f}",
                        "Home WHIP": "{:.2f}",
                        "Away IP": "{:.1f}",
                        "Home IP": "{:.1f}",
                        "Sample Reliability": "{:.2f}",
                        "ERA Diff": "{:.2f}",
                        "Edge Score": "{:.2f}",
                        "Confidence": "{:.1f}",
                    },
                    na_rep="-",
                )
                .apply(row_color, axis=1)
            )
            st.dataframe(styled, use_container_width=True, height=420)

            render_section_header("Today's Bets", "Current qualifying picks with live game status.")
            bets = show_df[show_df["Qualifies Strategy"] == "Yes"].head(10)
            if bets.empty:
                st.info("No games meet this system today.")
            else:
                bets_view = bets[
                    [
                        "Matchup",
                        "Game",
                        "Suggested F5 Pick",
                        "Edge Score",
                        "Confidence",
                        "Sample Reliability",
                    ]
                ]
                bets_styled = (
                    bets_view.style.format(
                        {
                            "Edge Score": "{:.2f}",
                            "Confidence": "{:.1f}",
                            "Sample Reliability": "{:.2f}",
                        },
                        na_rep="-",
                    ).apply(row_color_matchup_only, axis=1)
                )
                st.dataframe(bets_styled, use_container_width=True, height=240)

            render_section_header("System Tracker", "Historical outcomes for this system.")
            if sys_tracker.empty:
                st.info("No tracked bets for this system yet.")
            else:
                stf = prettify_tracker_view(sys_tracker, limit=None)
                st.dataframe(
                    stf[
                        [
                            "bet_date",
                            "matchup",
                            "suggested_pick",
                            "confidence",
                            "edge_score",
                            "status",
                            "result_note",
                            "logged_at",
                            "resolved_at",
                        ]
                    ],
                    use_container_width=True,
                    height=260,
                )


if __name__ == "__main__":
    main()


# ==================== ADDED FEATURES ====================

ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"

def implied_prob_from_american(odds):
    if odds is None:
        return None
    if odds > 0:
        return 100 / (odds + 100)
    return -odds / (-odds + 100)

def fetch_odds(api_key: str):
    if not api_key:
        return {}
    try:
        r = requests.get(
            ODDS_API_URL,
            params={
                "apiKey": api_key,
                "regions": "us",
                "markets": "h2h_1st_5_innings",
                "oddsFormat": "american"
            },
            timeout=20
        )
        data = r.json()
    except Exception:
        return {}

    odds_map = {}
    for game in data:
        key = f"{game['away_team']} @ {game['home_team']}"
        odds_map[key] = {}
        for book in game.get("bookmakers", []):
            for market in book.get("markets", []):
                for o in market.get("outcomes", []):
                    odds_map[key][o["name"]] = o["price"]
    return odds_map


def build_risk_cards(consensus_df: pd.DataFrame):
    if consensus_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    sharp = consensus_df[
        (consensus_df["Models On Bet"] >= 3) &
        (consensus_df["Avg Confidence"] >= 70) &
        (consensus_df["Avg Sample"] >= 0.55) &
        (consensus_df["EV"] > 0.05)
    ]

    balanced = consensus_df[
        (consensus_df["Models On Bet"] >= 2) &
        (consensus_df["Avg Confidence"] >= 65) &
        (consensus_df["Avg Sample"] >= 0.45) &
        (consensus_df["EV"] > 0.02)
    ]

    aggressive = consensus_df[
        (consensus_df["Models On Bet"] >= 1) &
        (consensus_df["Avg Confidence"] >= 58) &
        (consensus_df["EV"] > 0)
    ]

    return sharp, balanced, aggressive

# ==================== END ADDED FEATURES ====================
