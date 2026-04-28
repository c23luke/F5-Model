"""
gamecast_ui.py — Drop-in gamecast renderer for STACKED ANALYTICS · F5

Replaces the existing best-bet card renderers (render_best_bets_hero,
render_featured_pick_cards_html, render_best_bet_cards_v2) with a single
visual language: the live "gamecast" layout you saw in the mock —
inning-by-inning line score, live hit-probability gauge, key-reasons panel,
and an optional bet slip side panel that appears when American odds exist.

Three card sizes:
    render_gamecast_hero(...)        # Lock of the Day — full 2-column
    render_gamecast_card(...)        # Featured / Top Picks #2-#5 — same layout, tighter
    render_gamecast_mini_list(...)   # Long best-bets list — compact one-row gamecast each

Plus helpers:
    render_gamecast_theme()          # call once after render_premium_theme()
    fetch_innings_for_date(date_iso) # innings array for the line score grid
    build_matchup_intel(system_tables) # optional richer reasons (ERA gap, K diff, etc.)

This module is self-contained and only depends on pandas, requests, streamlit.
It does NOT mutate the user's existing fetch_scores_for_date cache.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import html as _html
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st


# =============================================================================
# Constants
# =============================================================================

_MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# Tier color tokens — keep in sync with render_gamecast_theme below.
_TIER_META: Dict[str, Dict[str, str]] = {
    "100% LOCK": {"label": "100% LOCK", "color": "#a78bfa", "soft": "rgba(167,139,250,0.18)"},
    "CORE":      {"label": "CORE",      "color": "#10b981", "soft": "rgba(16,185,129,0.18)"},
    "SHARP":     {"label": "SHARP",     "color": "#38bdf8", "soft": "rgba(56,189,248,0.16)"},
    "VALUE":     {"label": "VALUE",     "color": "#f59e0b", "soft": "rgba(245,158,11,0.18)"},
    "BT-ONLY":   {"label": "BT ONLY",   "color": "#f472b6", "soft": "rgba(244,114,182,0.16)"},
    "FEATURED":  {"label": "FEATURED",  "color": "#22d3ee", "soft": "rgba(34,211,238,0.16)"},
}


# =============================================================================
# Tiny helpers (re-implemented locally so this module is portable)
# =============================================================================

def _esc(val: Any) -> str:
    """HTML-escape and collapse whitespace (Streamlit markdown is line-sensitive)."""
    return _html.escape(" ".join(str(val if val is not None else "").split()))


def _to_int(val: Any) -> Optional[int]:
    if val is None:
        return None
    if isinstance(val, (int, float)) and not pd.isna(val):
        return int(val)
    s = str(val).strip()
    if not s or s in {"-", "--", "N/A"}:
        return None
    m = re.search(r"-?\d+", s)
    return int(m.group()) if m else None


def _norm_key(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def _canonical_team_key(text: Optional[str]) -> str:
    k = _norm_key(text)
    aliases = {
        "as": "athletics", "oaklandathletics": "athletics", "athletics": "athletics",
        "dbacks": "arizonadiamondbacks", "diamondbacks": "arizonadiamondbacks",
        "arizonadiamondbacks": "arizonadiamondbacks",
        "chicagowhitesox": "cws", "whitesox": "cws", "cws": "cws",
        "chicagocubs": "chc", "cubs": "chc", "chc": "chc",
    }
    return aliases.get(k, k)


def _short_team_label(full_name: str) -> str:
    """Last word, uppercased, capped at 8 chars."""
    full_name = str(full_name).strip()
    if not full_name:
        return ""
    parts = full_name.split()
    nick = parts[-1] if len(parts) >= 2 else full_name
    return (nick if len(nick) <= 8 else nick[:8]).upper()


def _team_abbr(full_name: str) -> str:
    """MLB team abbreviation for compact mobile-friendly matchup labels."""
    key = _canonical_team_key(full_name)
    abbr_map = {
        "arizonadiamondbacks": "ARI",
        "atlantabraves": "ATL",
        "baltimoreorioles": "BAL",
        "bostonredsox": "BOS",
        "chc": "CHC",
        "cws": "CWS",
        "cincinnatireds": "CIN",
        "clevelandguardians": "CLE",
        "coloradorockies": "COL",
        "detroittigers": "DET",
        "houstonastros": "HOU",
        "kansascityroyals": "KC",
        "losangelesangels": "LAA",
        "losangelesdodgers": "LAD",
        "miamimarlins": "MIA",
        "milwaukeebrewers": "MIL",
        "minnesotatwins": "MIN",
        "newyorkmets": "NYM",
        "newyorkyankees": "NYY",
        "athletics": "OAK",
        "philadelphiaphillies": "PHI",
        "pittsburghpirates": "PIT",
        "sandiegopadres": "SD",
        "sanfranciscogiants": "SF",
        "seattlemariners": "SEA",
        "stlouiscardinals": "STL",
        "tampabayrays": "TB",
        "texasrangers": "TEX",
        "torontobluejays": "TOR",
        "washingtonnationals": "WSH",
    }
    return abbr_map.get(key, _short_team_label(full_name)[:3])


def _abbr_matchup(matchup: str) -> str:
    if " @ " not in str(matchup):
        return str(matchup)
    away, home = [s.strip() for s in str(matchup).split(" @ ", 1)]
    return f"{_team_abbr(away)} @ {_team_abbr(home)}"


def _responsive_matchup_html(matchup: str, extra_class: str = "") -> str:
    """Full matchup on desktop, abbreviated matchup on mobile."""
    cls = f"gc-matchup {extra_class}".strip()
    return (
        f'<div class="{_esc(cls)}">'
        f'<span class="gc-matchup-full">{_esc(matchup)}</span>'
        f'<span class="gc-matchup-abbr">{_esc(_abbr_matchup(matchup))}</span>'
        '</div>'
    )


def _ordinal(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def _matchup_tone_class(matchup: str) -> str:
    """Stable color bucket per matchup so cards from the same game share a tone accent."""
    m = " ".join(str(matchup).split()).strip()
    if " @ " not in m:
        key = _norm_key(m)
    else:
        away, home = m.split(" @ ", 1)
        key = f"{_canonical_team_key(away.strip())}|{_canonical_team_key(home.strip())}"
    if not key:
        return "gc-tone-0"
    return f"gc-tone-{int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % 12}"


def _tier_class_suffix(tier: str) -> str:
    return {
        "100% LOCK": "lock",
        "CORE":      "core",
        "SHARP":     "sharp",
        "VALUE":     "value",
        "BT-ONLY":   "btonly",
        "FEATURED":  "featured",
    }.get(str(tier).upper(), "core")


# Pre-game keywords from MLB Stats API detailedState. These all mean the game
# has NOT started yet (even if currentInning is reported as 1 by the API).
_PRE_GAME_STATUS_KEYWORDS = (
    "scheduled",
    "pre-game",
    "pregame",
    "warmup",
    "delayed start",
    "postponed",
)


def _phase_from_inn(inn_data: Optional[Dict[str, Any]]) -> str:
    """Return one of 'pre' | 'live' | 'final' from an innings_map entry.

    The MLB Stats API will sometimes report `currentInning: 1` for a game whose
    detailedState is still 'Scheduled'/'Pre-Game'/'Warmup'. Trusting
    currentInning alone produces a phantom 'Live' label hours before first
    pitch — so we always check status first.
    """
    if not inn_data:
        return "pre"
    if inn_data.get("is_final"):
        return "final"
    status_l = str(inn_data.get("status") or "").lower()
    if any(k in status_l for k in _PRE_GAME_STATUS_KEYWORDS):
        return "pre"
    if "in progress" in status_l or "manager challenge" in status_l or "umpire review" in status_l:
        return "live"
    cur = inn_data.get("current_inning")
    if cur and int(cur) >= 1:
        # current_inning alone is not enough — but if status is unknown and we
        # have any innings runs reported, treat as live.
        runs_a = inn_data.get("innings_away_runs") or []
        runs_h = inn_data.get("innings_home_runs") or []
        any_runs_seen = any(r is not None for r in list(runs_a)[:max(1, int(cur))]) or any(
            r is not None for r in list(runs_h)[:max(1, int(cur))]
        )
        away_R = inn_data.get("away_R", 0) or 0
        home_R = inn_data.get("home_R", 0) or 0
        if any_runs_seen or away_R or home_R:
            return "live"
    return "pre"


# =============================================================================
# Data fetch — innings array (separate cache, doesn't disturb fetch_scores_for_date)
# =============================================================================

@st.cache_data(ttl=45, show_spinner=False)
def fetch_innings_for_date(game_date_iso: str) -> Dict[str, Dict[str, Any]]:
    """
    Returns a dict keyed by "<away_canonical>|<home_canonical>" containing
    inning-by-inning runs/hits/errors plus current state.

    Output per game:
        {
          "away": "New York Yankees",
          "home": "Houston Astros",
          "innings_away_runs":  [0,0,1,0,1,None,None,None,None],
          "innings_home_runs":  [0,0,0,0,1,None,None,None,None],
          "away_R": 2, "home_R": 1,
          "away_H": 5, "home_H": 3,
          "away_E": 0, "home_E": 0,
          "current_inning": 6,
          "inning_state": "bottom",
          "is_final": False,
          "status": "In Progress",
        }
    """
    params = {"sportId": 1, "date": game_date_iso, "hydrate": "linescore,team"}
    try:
        r = requests.get(_MLB_SCHEDULE_URL, params=params, headers=_HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for date_node in data.get("dates", []) or []:
        for game in date_node.get("games", []) or []:
            away_team = ((game.get("teams") or {}).get("away") or {}).get("team", {}).get("name") or "Away"
            home_team = ((game.get("teams") or {}).get("home") or {}).get("team", {}).get("name") or "Home"
            key = f"{_canonical_team_key(away_team)}|{_canonical_team_key(home_team)}"

            ls = game.get("linescore", {}) or {}
            innings_raw = ls.get("innings") or []
            inn_a: List[Optional[int]] = []
            inn_h: List[Optional[int]] = []
            for inn in innings_raw[:9]:
                a = _to_int((inn.get("away") or {}).get("runs"))
                h = _to_int((inn.get("home") or {}).get("runs"))
                inn_a.append(a)
                inn_h.append(h)
            # Pad to length 9
            while len(inn_a) < 9:
                inn_a.append(None)
            while len(inn_h) < 9:
                inn_h.append(None)

            ls_teams = ls.get("teams") or {}
            away_R = _to_int((ls_teams.get("away") or {}).get("runs")) or 0
            home_R = _to_int((ls_teams.get("home") or {}).get("runs")) or 0
            away_H = _to_int((ls_teams.get("away") or {}).get("hits")) or 0
            home_H = _to_int((ls_teams.get("home") or {}).get("hits")) or 0
            away_E = _to_int((ls_teams.get("away") or {}).get("errors")) or 0
            home_E = _to_int((ls_teams.get("home") or {}).get("errors")) or 0

            status = (game.get("status") or {}).get("detailedState", "") or ""
            is_final = "final" in status.lower() or status.lower() in {"game over", "completed early"}

            out[key] = {
                "away": away_team,
                "home": home_team,
                "innings_away_runs": inn_a,
                "innings_home_runs": inn_h,
                "away_R": away_R, "home_R": home_R,
                "away_H": away_H, "home_H": home_H,
                "away_E": away_E, "home_E": home_E,
                "current_inning": _to_int(ls.get("currentInning")),
                "inning_state": str(ls.get("inningState") or "").lower(),
                "is_final": is_final,
                "status": status,
            }
    return out


def _innings_for_matchup(matchup: str, innings_map: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if " @ " not in matchup:
        return None
    away, home = matchup.split(" @ ", 1)
    key = f"{_canonical_team_key(away.strip())}|{_canonical_team_key(home.strip())}"
    return innings_map.get(key)


# =============================================================================
# Live probability + ticket lookup
# =============================================================================

def _hit_probability(
    matchup: str,
    pick: str,
    base_confidence: float,
    score_map: Dict[str, Dict[str, Any]],
) -> Tuple[float, str, str]:
    """
    Returns (probability_0_100, trend_label, trend_color_hex).
    Blends model confidence with current F5 run differential and inning progress.
    """
    if " @ " not in matchup:
        return max(0.0, min(100.0, base_confidence)), "Pre-game", "#94a3b8"

    away, home = [s.strip() for s in matchup.split(" @ ", 1)]
    key = f"{_canonical_team_key(away)}|{_canonical_team_key(home)}"
    game = score_map.get(key) or {}

    pick_team = re.sub(r"\bF5\b", "", pick, flags=re.I).strip().lower()

    if game.get("can_grade"):
        af5 = int(game.get("away_f5", 0) or 0)
        hf5 = int(game.get("home_f5", 0) or 0)
        if pick_team == away.lower():
            won = af5 > hf5; pushed = af5 == hf5
        else:
            won = hf5 > af5; pushed = hf5 == af5
        if pushed:
            return 50.0, "Push", "#f59e0b"
        if won:
            return 100.0, "Won", "#10b981"
        return 0.0, "Lost", "#f43f5e"

    if not game:
        return max(0.0, min(100.0, base_confidence)), "Pre-game", "#94a3b8"

    af5 = int(game.get("away_f5", 0) or 0)
    hf5 = int(game.get("home_f5", 0) or 0)
    if pick_team == away.lower():
        run_diff = af5 - hf5
    elif pick_team == home.lower():
        run_diff = hf5 - af5
    else:
        run_diff = 0

    inning = _to_int(game.get("current_inning")) or 1
    inning_state = str(game.get("inning_state", "")).lower()
    if inning_state in {"end", "bottom"}:
        progress = min(1.0, (inning + 0.5) / 5.0)
    else:
        progress = min(1.0, inning / 5.0)

    state_bonus = (run_diff * 11.0) * (0.35 + 0.65 * progress)
    blended = (base_confidence * (1.0 - 0.35 * progress)) + (base_confidence + state_bonus) * (0.35 * progress)
    p = max(0.0, min(100.0, blended))

    if p >= 85:
        return p, "Very High", "#10b981"
    if p >= 65:
        return p, "High", "#34d399"
    if p >= 45:
        return p, "Toss-up", "#f59e0b"
    if p >= 25:
        return p, "Low", "#fb7185"
    return p, "Cooked", "#f43f5e"


def _ticket_status_for(row: pd.Series, tracker_df: pd.DataFrame) -> str:
    """Return 'WON' / 'LOST' / 'PUSH' / 'OPEN' / '' for today's tracker."""
    if tracker_df is None or tracker_df.empty:
        return "OPEN"
    matchup = str(row.get("Matchup", ""))
    pick = str(row.get("Suggested F5 Pick", ""))
    today = dt.date.today().isoformat()
    sub = tracker_df[
        (tracker_df["matchup"].astype(str) == matchup)
        & (tracker_df["suggested_pick"].astype(str) == pick)
        & (tracker_df["bet_date"].astype(str) == today)
    ]
    if sub.empty:
        return "OPEN"
    sub = sub.copy()
    sub["logged_at_dt"] = pd.to_datetime(sub.get("logged_at"), errors="coerce")
    sub = sub.sort_values(by=["logged_at_dt"], ascending=False)
    s = str(sub.iloc[0].get("status", "")).strip().lower()
    return {"win": "WON", "won": "WON", "loss": "LOST", "lost": "LOST",
            "push": "PUSH"}.get(s, "OPEN")


# =============================================================================
# Reason generation
# =============================================================================

def build_matchup_intel(system_tables: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    """
    Optional helper. Pass your `system_tables` dict here once per render and the
    gamecast functions will use it to produce richer Key Reasons / Why So High
    bullets (ERA gap, K differential, etc.).

    Returns: { "Away Team @ Home Team": {era_gap, k_diff, whip_gap, away_era, home_era, ...} }
    """
    intel: Dict[str, Dict[str, Any]] = {}
    for _, df in (system_tables or {}).items():
        if df is None or df.empty:
            continue
        for _, r in df.iterrows():
            m = str(r.get("Matchup", ""))
            if not m or m in intel:
                continue
            try:
                a_era = float(r.get("Away ERA")) if pd.notna(r.get("Away ERA")) else None
                h_era = float(r.get("Home ERA")) if pd.notna(r.get("Home ERA")) else None
                a_whip = float(r.get("Away WHIP")) if pd.notna(r.get("Away WHIP")) else None
                h_whip = float(r.get("Home WHIP")) if pd.notna(r.get("Home WHIP")) else None
                a_k = int(r.get("Away K")) if pd.notna(r.get("Away K")) else None
                h_k = int(r.get("Home K")) if pd.notna(r.get("Home K")) else None
                intel[m] = {
                    "away_era": a_era, "home_era": h_era,
                    "away_whip": a_whip, "home_whip": h_whip,
                    "away_k": a_k, "home_k": h_k,
                    "era_gap": (abs(a_era - h_era) if a_era is not None and h_era is not None else None),
                    "whip_gap": (abs(a_whip - h_whip) if a_whip is not None and h_whip is not None else None),
                    "k_diff": ((a_k - h_k) if a_k is not None and h_k is not None else None),
                    "away_pitcher": str(r.get("Away Pitcher", "") or ""),
                    "home_pitcher": str(r.get("Home Pitcher", "") or ""),
                }
            except Exception:
                continue
    return intel


def _why_so_high_reasons(row: pd.Series, intel: Optional[Dict[str, Any]]) -> List[str]:
    """Produce 3-5 reasons explaining why the model likes this play."""
    reasons: List[str] = []
    tier = str(row.get("Tier", "")).upper()
    models = int(row.get("Models On Bet", 0) or 0)
    conf = float(row.get("Avg Confidence", 0.0) or 0.0)
    sample = float(row.get("Avg Sample", 0.0) or 0.0)
    ev = row.get("EV", None)

    if tier == "100% LOCK":
        reasons.append("Maxed-out model confidence")
    elif tier == "CORE":
        reasons.append("Core priority signal")

    if models >= 4:
        reasons.append(f"{models} models converged")
    elif models >= 2:
        reasons.append(f"{models}-model consensus")

    if conf >= 90:
        reasons.append("Confidence ≥90%")
    elif conf >= 80:
        reasons.append("Confidence ≥80%")

    if sample >= 0.65:
        reasons.append("Established starter sample")
    elif sample >= 0.50:
        reasons.append("Solid sample reliability")

    if ev is not None and pd.notna(ev) and float(ev) > 0:
        reasons.append(f"+EV: {float(ev):+.2f}u/1u")

    if intel:
        era_gap = intel.get("era_gap")
        k_diff = intel.get("k_diff")
        whip_gap = intel.get("whip_gap")
        if era_gap is not None and era_gap >= 1.5:
            reasons.append(f"ERA gap {era_gap:.2f}")
        if k_diff is not None and abs(k_diff) >= 15:
            reasons.append(f"K edge {k_diff:+d}")
        if whip_gap is not None and whip_gap >= 0.15:
            reasons.append(f"WHIP gap {whip_gap:.2f}")

    # Dedup, preserve order, cap at 5
    seen = set(); out: List[str] = []
    for r in reasons:
        if r not in seen:
            seen.add(r); out.append(r)
    return out[:5] if out else ["Multi-model agreement", "Sample reliability cleared", "Tier threshold hit"]


def _key_reasons(row: pd.Series, intel: Optional[Dict[str, Any]]) -> List[str]:
    """Produce 3-4 longer-form key reasons for the side panel."""
    reasons: List[str] = []
    matchup = str(row.get("Matchup", ""))
    pick = str(row.get("Suggested F5 Pick", "")).replace("F5 ", "").strip()
    away = matchup.split(" @ ")[0].strip() if " @ " in matchup else ""
    home = matchup.split(" @ ")[1].strip() if " @ " in matchup else ""
    pick_is_away = pick.lower() == away.lower()

    if intel:
        a_era = intel.get("away_era"); h_era = intel.get("home_era")
        a_whip = intel.get("away_whip"); h_whip = intel.get("home_whip")
        a_k = intel.get("away_k"); h_k = intel.get("home_k")
        a_p = intel.get("away_pitcher") or "Away starter"
        h_p = intel.get("home_pitcher") or "Home starter"

        if a_era is not None and h_era is not None:
            if pick_is_away:
                reasons.append(f"{a_p} {a_era:.2f} ERA vs {h_p} {h_era:.2f} ERA")
            else:
                reasons.append(f"{h_p} {h_era:.2f} ERA vs {a_p} {a_era:.2f} ERA")

        if a_whip is not None and h_whip is not None:
            gap = abs(a_whip - h_whip)
            if gap >= 0.10:
                better = a_whip if pick_is_away else h_whip
                worse = h_whip if pick_is_away else a_whip
                reasons.append(f"WHIP edge {better:.2f} vs {worse:.2f}")

        if a_k is not None and h_k is not None:
            diff = (a_k - h_k) if pick_is_away else (h_k - a_k)
            if abs(diff) >= 5:
                reasons.append(f"Strikeout edge {diff:+d} on starter side")

    models = int(row.get("Models On Bet", 0) or 0)
    sample = float(row.get("Avg Sample", 0.0) or 0.0)
    if models >= 2:
        reasons.append(f"{models} models on the same side")
    if sample >= 0.45:
        reasons.append(f"Sample reliability {sample:.2f}")

    ev = row.get("EV", None)
    odds = row.get("American Odds", None)
    if ev is not None and pd.notna(ev) and odds is not None and pd.notna(odds):
        reasons.append(f"+EV {float(ev):+.2f}u at {int(float(odds)):+d}")

    if not reasons:
        reasons = [
            "Multi-model agreement on F5 side",
            "Sample reliability above threshold",
            "Tier-qualified priority signal",
        ]
    # Dedup + cap
    seen = set(); out: List[str] = []
    for r in reasons:
        if r not in seen:
            seen.add(r); out.append(r)
    return out[:4]


# =============================================================================
# CSS — gamecast theme
# =============================================================================

_GAMECAST_CSS = """
<style>
:root {
    --gc-bg-0: #06080f;
    --gc-bg-1: #0b1220;
    --gc-bg-2: #111a2e;
    --gc-bg-3: #162038;
    --gc-card-grad-from: rgba(11, 18, 32, 0.92);
    --gc-card-grad-to:   rgba(13, 23, 42, 0.95);
    --gc-border:        rgba(148, 163, 184, 0.14);
    --gc-border-strong: rgba(148, 163, 184, 0.28);
    --gc-text-0: #f8fafc;
    --gc-text-1: #e2e8f0;
    --gc-text-2: #94a3b8;
    --gc-text-3: #64748b;
    --gc-emerald: #10b981;
    --gc-emerald-soft: rgba(16,185,129,0.18);
    --gc-violet: #a78bfa;
    --gc-violet-soft: rgba(167,139,250,0.18);
    --gc-cyan: #38bdf8;
    --gc-cyan-soft: rgba(56,189,248,0.16);
    --gc-amber: #f59e0b;
    --gc-amber-soft: rgba(245,158,11,0.18);
    --gc-rose: #f43f5e;
    --gc-rose-soft: rgba(244,63,94,0.16);
}

/* ============== Card shell ============== */
.gc-card {
    position: relative;
    background:
        radial-gradient(800px 320px at -10% -40%, rgba(167,139,250,0.10), transparent 65%),
        radial-gradient(700px 280px at 110% -30%, rgba(16,185,129,0.08), transparent 65%),
        linear-gradient(160deg, var(--gc-card-grad-from) 0%, var(--gc-card-grad-to) 100%);
    border: 1px solid var(--gc-border);
    border-radius: 18px;
    padding: 18px 20px;
    margin: 12px 0 14px 0;
    overflow: hidden;
    box-shadow: 0 18px 48px rgba(0,0,0,0.42), inset 0 1px 0 rgba(255,255,255,0.04);
}
.gc-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--gc-violet), var(--gc-emerald), var(--gc-cyan));
    opacity: 0.85;
}

/* tier-tinted left rail */
.gc-card.tier-lock     { border-left: 3px solid var(--gc-violet); }
.gc-card.tier-core     { border-left: 3px solid var(--gc-emerald); }
.gc-card.tier-sharp    { border-left: 3px solid var(--gc-cyan); }
.gc-card.tier-value    { border-left: 3px solid var(--gc-amber); }
.gc-card.tier-btonly   { border-left: 3px solid #f472b6; }
.gc-card.tier-featured { border-left: 3px solid #22d3ee; }

/* ============== Header strip ============== */
.gc-head {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 6px; gap: 10px; flex-wrap: wrap;
}
.gc-head-left { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.gc-live-pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 10px; border-radius: 999px;
    background: rgba(16,185,129,0.12); border: 1px solid rgba(16,185,129,0.45);
    color: #6ee7b7;
    font-size: 0.62rem; font-weight: 800; letter-spacing: 0.16em;
}
.gc-live-pill::before {
    content: ""; width: 6px; height: 6px; border-radius: 50%;
    background: #34d399; box-shadow: 0 0 10px #10b981;
    animation: gcPulse 1.6s infinite;
}
@keyframes gcPulse {
    0% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.55; transform: scale(0.85); }
    100% { opacity: 1; transform: scale(1); }
}
.gc-head-title { font-size: 1.1rem; font-weight: 800; color: var(--gc-text-0); letter-spacing: -0.01em; }
.gc-tier-pill {
    display: inline-block; padding: 5px 12px; border-radius: 999px;
    font-size: 0.62rem; font-weight: 900; letter-spacing: 0.14em; text-transform: uppercase;
    border: 1px solid; background: transparent;
}
.gc-tier-pill.tier-lock     { color: var(--gc-violet);  border-color: var(--gc-violet);  background: var(--gc-violet-soft); }
.gc-tier-pill.tier-core     { color: var(--gc-emerald); border-color: var(--gc-emerald); background: var(--gc-emerald-soft); }
.gc-tier-pill.tier-sharp    { color: var(--gc-cyan);    border-color: var(--gc-cyan);    background: var(--gc-cyan-soft); }
.gc-tier-pill.tier-value    { color: var(--gc-amber);   border-color: var(--gc-amber);   background: var(--gc-amber-soft); }
.gc-tier-pill.tier-btonly   { color: #f472b6;           border-color: #f472b6;           background: rgba(244,114,182,0.16); }
.gc-tier-pill.tier-featured { color: #22d3ee;           border-color: #22d3ee;           background: rgba(34,211,238,0.16); }

.gc-sub {
    color: var(--gc-text-2); font-size: 0.82rem; font-weight: 500;
    margin-bottom: 12px; padding-bottom: 12px;
    border-bottom: 1px solid var(--gc-border);
}

/* ============== Hero / card 2-col grid ============== */
.gc-grid {
    display: grid;
    grid-template-columns: 1.55fr 1fr;
    gap: 18px;
}
@media (max-width: 980px) {
    .gc-grid { grid-template-columns: 1fr; }
}
@media (max-width: 760px) {
    .gc-card { padding: 12px 12px; border-radius: 14px; }
    .gc-head-title { font-size: 0.95rem; }
    .gc-sub { font-size: 0.74rem; margin-bottom: 8px; padding-bottom: 8px; }
    .gc-topbar { flex-direction: column; align-items: flex-start; gap: 10px; margin-bottom: 10px; }
    .gc-matchup { font-size: 1.05rem; line-height: 1.2; }
    .gc-matchup-card { font-size: 0.98rem; }
    .gc-pickline { font-size: 0.84rem; margin-top: 2px; }
    .gc-metrics { width: 100%; border-left: none; border-top: 1px solid var(--gc-border); padding-top: 8px; }
    .gc-metric { flex: 1; min-width: 0; padding: 0 8px; }
    .gc-metric-val { font-size: 1.02rem; }
    .gc-line-wrap { padding: 8px 8px; gap: 8px; grid-template-columns: 1fr; }
    .gc-line-table th, .gc-line-table td { font-size: 0.68rem; padding: 4px 3px; }
    .gc-line-table td.team { font-size: 0.72rem; padding-right: 8px; }
    .gc-inning-marker { border-left: none; border-top: 1px solid var(--gc-border); padding-top: 8px; }
    .gc-prob-big { font-size: 2rem; }
    .gc-status { padding: 8px 10px; }
    .gc-status-text { font-size: 0.74rem; }
    .gc-side { gap: 8px; }
    .gc-side-card { padding: 10px; }
    .gc-matchup-full { display: none; }
    .gc-matchup-abbr { display: inline; }
}

.gc-main { min-width: 0; }

/* ============== Eyebrow + matchup ============== */
.gc-eyebrow {
    font-size: 0.66rem; font-weight: 900; letter-spacing: 0.20em; text-transform: uppercase;
    color: var(--gc-violet); margin-bottom: 4px;
}
.gc-matchup {
    font-size: 1.85rem; font-weight: 800; color: var(--gc-text-0);
    letter-spacing: -0.018em; line-height: 1.15;
}
.gc-matchup-card { font-size: 1.45rem; }
.gc-matchup-abbr { display: none; }
.gc-pickline {
    color: var(--gc-text-1); font-size: 0.95rem; margin-top: 6px; margin-bottom: 4px;
}
.gc-pickline b { color: var(--gc-emerald); font-weight: 800; }

/* ============== Top right metrics ============== */
.gc-topbar {
    display: flex; align-items: flex-start; justify-content: space-between;
    gap: 18px; margin: 4px 0 14px 0;
}
.gc-topbar-meta { flex: 1; min-width: 0; }
.gc-metrics {
    display: flex; gap: 0; align-items: stretch;
    border-left: 1px solid var(--gc-border);
}
.gc-metric {
    padding: 0 14px;
    border-right: 1px solid var(--gc-border);
    text-align: center; min-width: 78px;
}
.gc-metric:last-child { border-right: none; }
.gc-metric-lab {
    font-size: 0.60rem; font-weight: 800; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--gc-text-3); margin-bottom: 4px;
}
.gc-metric-val {
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
    font-size: 1.32rem; font-weight: 900; color: var(--gc-text-0);
    line-height: 1;
}
.gc-metric-val.is-conf { color: var(--gc-emerald); }

/* ============== Linescore ============== */
.gc-line-wrap {
    background: rgba(2, 6, 23, 0.55);
    border: 1px solid var(--gc-border-strong);
    border-radius: 12px;
    padding: 12px 14px;
    margin: 6px 0 14px 0;
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 14px;
    align-items: center;
}
.gc-line-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
    font-variant-numeric: tabular-nums;
}
.gc-line-table th, .gc-line-table td {
    padding: 6px 6px;
    text-align: center;
    font-size: 0.78rem;
    color: var(--gc-text-1);
}
.gc-line-table th {
    color: var(--gc-text-3);
    font-weight: 700;
    font-size: 0.62rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    border-bottom: 1px solid var(--gc-border);
}
.gc-line-table td.team {
    text-align: left;
    font-family: "Lexend", system-ui, sans-serif;
    font-weight: 800;
    color: var(--gc-text-0);
    font-size: 0.86rem;
    padding-right: 14px;
    white-space: nowrap;
}
.gc-line-table td.rec {
    color: var(--gc-text-3);
    font-size: 0.66rem;
    font-weight: 600;
    margin-left: 4px;
}
.gc-line-table td.tot {
    font-weight: 900;
    color: var(--gc-emerald);
    font-size: 0.92rem;
}
.gc-line-table td.tot-h, .gc-line-table td.tot-e {
    color: var(--gc-text-0);
    font-weight: 800;
}
.gc-line-table td.future {
    color: var(--gc-text-3);
}
.gc-line-table th.divider, .gc-line-table td.divider {
    border-left: 1px solid var(--gc-border);
}

.gc-inning-marker {
    text-align: center;
    padding: 6px 14px;
    border-left: 1px solid var(--gc-border);
}
.gc-inning-lab {
    font-size: 0.60rem; font-weight: 800; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--gc-text-3); margin-bottom: 6px;
}
.gc-inning-val {
    font-size: 1.05rem; font-weight: 900; color: var(--gc-text-0);
    margin-bottom: 8px; letter-spacing: 0.04em;
}
.gc-diamond {
    width: 36px; height: 36px; margin: 0 auto;
    position: relative;
    transform: rotate(45deg);
    border: 1px solid var(--gc-border-strong);
    background: linear-gradient(135deg, rgba(167,139,250,0.18), rgba(16,185,129,0.10));
}
.gc-diamond::after {
    content: ""; position: absolute; left: 50%; top: 50%;
    width: 8px; height: 8px; transform: translate(-50%, -50%);
    background: var(--gc-amber); border-radius: 50%;
    box-shadow: 0 0 8px var(--gc-amber);
}

/* ============== Live Hit Probability ============== */
.gc-prob-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
    align-items: stretch;
    margin-bottom: 12px;
}
@media (max-width: 760px) { .gc-prob-row { grid-template-columns: 1fr; } }

.gc-prob-card {
    background:
        linear-gradient(135deg, rgba(16,185,129,0.10) 0%, rgba(2,6,23,0.55) 65%);
    border: 1px solid rgba(16,185,129,0.22);
    border-radius: 12px;
    padding: 14px 16px;
}
.gc-prob-lab {
    font-size: 0.62rem; font-weight: 900; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--gc-text-3); margin-bottom: 6px;
}
.gc-prob-big {
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
    font-size: 2.6rem; font-weight: 900;
    color: var(--gc-emerald);
    letter-spacing: -0.04em; line-height: 1;
}
.gc-prob-trend {
    margin-top: 6px;
    font-size: 0.85rem; font-weight: 700;
    color: var(--gc-emerald);
}
.gc-prob-trend.amber { color: var(--gc-amber); }
.gc-prob-trend.rose  { color: var(--gc-rose); }
.gc-prob-trend.muted { color: var(--gc-text-2); }
.gc-prob-bar {
    margin-top: 10px;
    width: 100%; height: 6px; border-radius: 999px;
    background: rgba(148,163,184,0.18); overflow: hidden;
}
.gc-prob-bar-fill {
    height: 100%; border-radius: 999px;
    background: linear-gradient(90deg, var(--gc-emerald) 0%, #34d399 60%, #6ee7b7 100%);
}

.gc-why {
    background: rgba(15, 23, 42, 0.55);
    border: 1px solid var(--gc-border);
    border-radius: 12px;
    padding: 12px 14px;
}
.gc-why-lab {
    font-size: 0.62rem; font-weight: 900; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--gc-text-3); margin-bottom: 8px;
}
.gc-why-list { list-style: none; margin: 0; padding: 0; }
.gc-why-list li {
    color: var(--gc-text-1); font-size: 0.82rem; font-weight: 600;
    padding: 3px 0; display: flex; align-items: center; gap: 8px;
}
.gc-why-check {
    flex: 0 0 auto; width: 14px; height: 14px; border-radius: 50%;
    background: var(--gc-emerald-soft); color: var(--gc-emerald);
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 0.62rem; font-weight: 900;
}

/* ============== Status bar (under main column) ============== */
.gc-status {
    margin-top: 10px;
    display: flex; align-items: center; justify-content: space-between;
    gap: 10px; flex-wrap: wrap;
    padding: 10px 14px; border-radius: 10px;
    background: rgba(2, 6, 23, 0.55);
    border: 1px solid var(--gc-border);
}
.gc-status-left, .gc-status-right { display: flex; align-items: center; gap: 8px; }
.gc-status-lab {
    font-size: 0.62rem; font-weight: 900; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--gc-text-3);
}
.gc-status-dot {
    width: 8px; height: 8px; border-radius: 50%;
}
.gc-status-dot.open { background: var(--gc-amber); box-shadow: 0 0 8px var(--gc-amber); }
.gc-status-dot.won  { background: var(--gc-emerald); box-shadow: 0 0 8px var(--gc-emerald); }
.gc-status-dot.lost { background: var(--gc-rose); box-shadow: 0 0 6px var(--gc-rose); }
.gc-status-dot.push { background: var(--gc-text-2); }
.gc-status-text {
    font-size: 0.82rem; color: var(--gc-text-1); font-weight: 700;
}

/* ============== Side panel (bet slip + reasons + tracking) ============== */
.gc-side { display: flex; flex-direction: column; gap: 12px; min-width: 0; }
.gc-side-card {
    background: rgba(2, 6, 23, 0.62);
    border: 1px solid var(--gc-border);
    border-radius: 12px;
    padding: 12px 14px;
}
.gc-side-head {
    display: flex; align-items: center; gap: 8px;
    font-size: 0.70rem; font-weight: 900; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--gc-text-2);
    margin-bottom: 10px;
}
.gc-side-icon {
    width: 16px; height: 16px; border-radius: 4px;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 0.70rem;
    background: var(--gc-violet-soft); color: var(--gc-violet);
}
.gc-slip-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 6px 0;
}
.gc-slip-label { color: var(--gc-text-3); font-size: 0.78rem; font-weight: 600; }
.gc-slip-pick-name {
    color: var(--gc-text-0); font-size: 0.92rem; font-weight: 800; letter-spacing: -0.01em;
}
.gc-slip-odds-chip {
    background: var(--gc-violet-soft); color: var(--gc-violet);
    border: 1px solid rgba(167,139,250,0.45);
    border-radius: 8px; padding: 4px 10px;
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
    font-size: 0.92rem; font-weight: 900;
}
.gc-slip-input {
    background: var(--gc-bg-2); color: var(--gc-text-0);
    border: 1px solid var(--gc-border-strong); border-radius: 8px;
    padding: 6px 12px; font-family: ui-monospace, "SF Mono", Menlo, monospace;
    font-size: 0.92rem; font-weight: 800; min-width: 90px; text-align: right;
}
.gc-slip-towin { color: var(--gc-emerald); font-weight: 900; font-family: ui-monospace, monospace; font-size: 0.95rem; }
.gc-slip-button {
    margin-top: 10px; width: 100%; padding: 11px;
    border-radius: 10px;
    background: linear-gradient(135deg, var(--gc-violet) 0%, #6366f1 100%);
    color: white; font-weight: 900; letter-spacing: 0.10em;
    text-align: center; font-size: 0.82rem;
    box-shadow: 0 6px 20px rgba(99,102,241,0.32);
    cursor: pointer; transition: transform .15s ease, box-shadow .15s ease;
}
.gc-slip-button::after { content: " →"; opacity: 0.85; }
.gc-slip-button:hover { transform: translateY(-1px); box-shadow: 0 10px 24px rgba(99,102,241,0.42); }

.gc-key-item {
    display: flex; align-items: flex-start; gap: 10px;
    padding: 6px 0;
    color: var(--gc-text-1); font-size: 0.84rem; font-weight: 600;
}
.gc-key-item-icon {
    flex: 0 0 18px; height: 18px; border-radius: 5px;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 0.70rem; font-weight: 900;
    background: var(--gc-emerald-soft); color: var(--gc-emerald);
}

.gc-track-row {
    display: flex; align-items: center; justify-content: space-between; gap: 10px;
}
.gc-track-stack {
    display: flex; align-items: center; gap: -6px;
}
.gc-track-avatar {
    width: 28px; height: 28px; border-radius: 50%;
    background: linear-gradient(135deg, #475569 0%, #1e293b 100%);
    border: 2px solid var(--gc-bg-1);
    margin-left: -8px;
    display: inline-flex; align-items: center; justify-content: center;
    color: var(--gc-text-0); font-weight: 800; font-size: 0.70rem;
}
.gc-track-avatar:first-child { margin-left: 0; }
.gc-track-count {
    margin-left: 8px; color: var(--gc-text-2);
    font-weight: 700; font-size: 0.80rem;
    font-variant-numeric: tabular-nums;
}
.gc-track-roi {
    display: inline-flex; align-items: center; gap: 6px;
    color: var(--gc-emerald); font-weight: 900;
    font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.95rem;
}
.gc-track-roi.neg { color: var(--gc-rose); }

/* ============== MINI variant (compact horizontal gamecast) ============== */
.gc-mini {
    display: grid;
    grid-template-columns: 84px 1fr 240px 110px 80px;
    gap: 14px;
    align-items: center;
    padding: 14px 16px;
    background: linear-gradient(160deg, rgba(11,18,32,0.92) 0%, rgba(13,23,42,0.95) 100%);
    border: 1px solid var(--gc-border);
    border-radius: 14px;
    margin-bottom: 10px;
    position: relative;
    overflow: hidden;
    transition: transform .14s ease, border-color .14s ease;
}
.gc-mini:hover { transform: translateY(-1px); border-color: var(--gc-border-strong); }
.gc-mini.tier-lock     { border-left: 4px solid var(--gc-violet); }
.gc-mini.tier-core     { border-left: 4px solid var(--gc-emerald); }
.gc-mini.tier-sharp    { border-left: 4px solid var(--gc-cyan); }
.gc-mini.tier-value    { border-left: 4px solid var(--gc-amber); }
.gc-mini.tier-btonly   { border-left: 4px solid #f472b6; }
.gc-mini-rank-col { display: flex; flex-direction: column; gap: 4px; min-width: 0; }
.gc-mini-rank {
    font-family: ui-monospace, monospace;
    font-size: 0.82rem; color: var(--gc-text-3); font-weight: 700;
}
.gc-mini-pick-col { min-width: 0; }
.gc-mini-matchup {
    font-size: 0.95rem; font-weight: 800; color: var(--gc-text-0);
    letter-spacing: -0.005em;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.gc-mini-pick {
    font-size: 0.82rem; color: var(--gc-text-1); margin-top: 2px;
}
.gc-mini-pick b { color: var(--gc-emerald); font-weight: 800; }
.gc-mini-meta {
    display: flex; flex-wrap: wrap; gap: 10px; margin-top: 6px;
    font-family: ui-monospace, monospace;
    font-size: 0.72rem; color: var(--gc-text-2);
}
.gc-mini-meta b { color: var(--gc-text-0); }

.gc-mini-line {
    display: flex; align-items: center; justify-content: space-between;
    background: rgba(2,6,23,0.55); border: 1px solid var(--gc-border);
    border-radius: 10px; padding: 8px 10px;
    font-family: ui-monospace, monospace;
}
.gc-mini-side { display: flex; align-items: center; gap: 8px; }
.gc-mini-team {
    color: var(--gc-text-1); font-weight: 800; font-size: 0.78rem;
}
.gc-mini-score-num {
    color: var(--gc-text-0); font-weight: 900; font-size: 1.10rem;
    letter-spacing: -0.02em; min-width: 16px; text-align: right;
}
.gc-mini-vs { color: var(--gc-text-3); font-weight: 700; font-size: 0.78rem; }
.gc-mini-stat-tag {
    margin-top: 4px;
    text-align: center;
    font-size: 0.62rem; font-weight: 800; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--gc-text-3);
}
.gc-mini-prob-col { text-align: center; min-width: 0; }
.gc-mini-prob {
    font-family: ui-monospace, monospace;
    font-size: 1.28rem; font-weight: 900; color: var(--gc-emerald);
    line-height: 1;
}
.gc-mini-prob.amber { color: var(--gc-amber); }
.gc-mini-prob.rose  { color: var(--gc-rose); }
.gc-mini-prob.muted { color: var(--gc-text-2); }
.gc-mini-prob-bar {
    margin-top: 6px; width: 100%; height: 4px; border-radius: 999px;
    background: rgba(148,163,184,0.20); overflow: hidden;
}
.gc-mini-prob-fill {
    height: 100%; border-radius: 999px;
    background: linear-gradient(90deg, var(--gc-emerald), #34d399);
}
.gc-mini-prob-trend {
    font-size: 0.62rem; color: var(--gc-text-3);
    font-weight: 800; letter-spacing: 0.10em;
    text-transform: uppercase; margin-top: 4px;
}
.gc-mini-status-col { display: flex; flex-direction: column; align-items: flex-end; gap: 4px; }
.gc-mini-ticket {
    font-size: 0.62rem; font-weight: 900; letter-spacing: 0.12em;
    text-transform: uppercase; padding: 4px 9px; border-radius: 6px;
}
.gc-mini-ticket.won  { background: var(--gc-emerald-soft); color: var(--gc-emerald); }
.gc-mini-ticket.lost { background: var(--gc-rose-soft);    color: var(--gc-rose); }
.gc-mini-ticket.open { background: rgba(148,163,184,0.16); color: var(--gc-text-2); }
.gc-mini-ticket.push { background: var(--gc-amber-soft);   color: var(--gc-amber); }
.gc-mini-phase {
    font-size: 0.62rem; color: var(--gc-text-3);
    font-weight: 700; letter-spacing: 0.06em;
}

@media (max-width: 980px) {
    .gc-mini { grid-template-columns: 60px 1fr; }
    .gc-mini-line, .gc-mini-prob-col, .gc-mini-status-col { grid-column: 1 / -1; }
}

/* ============== Tier accents on mini headline ============== */
.gc-mini-tierpill {
    display: inline-block; padding: 3px 8px; border-radius: 6px;
    font-size: 0.58rem; font-weight: 900; letter-spacing: 0.14em; text-transform: uppercase;
    width: fit-content;
}
.gc-mini-tierpill.tier-lock     { background: var(--gc-violet-soft);  color: var(--gc-violet); }
.gc-mini-tierpill.tier-core     { background: var(--gc-emerald-soft); color: var(--gc-emerald); }
.gc-mini-tierpill.tier-sharp    { background: var(--gc-cyan-soft);    color: var(--gc-cyan); }
.gc-mini-tierpill.tier-value    { background: var(--gc-amber-soft);   color: var(--gc-amber); }
.gc-mini-tierpill.tier-btonly   { background: rgba(244,114,182,0.18); color: #f472b6; }
.gc-mini-tierpill.tier-featured { background: rgba(34,211,238,0.18);  color: #22d3ee; }

/* ============== Empty state ============== */
.gc-empty {
    background: rgba(2, 6, 23, 0.55);
    border: 1px dashed var(--gc-border-strong);
    border-radius: 14px;
    padding: 28px 24px;
    text-align: center;
    color: var(--gc-text-2);
    margin: 12px 0;
}
.gc-empty-title { font-size: 1rem; color: var(--gc-text-1); font-weight: 700; margin-bottom: 4px; }
.gc-empty-sub   { font-size: 0.82rem; color: var(--gc-text-3); }
</style>
"""


def render_gamecast_theme() -> None:
    """Inject the gamecast CSS. Call once after render_premium_theme()."""
    st.markdown(_GAMECAST_CSS, unsafe_allow_html=True)


# =============================================================================
# Internal renderers — shared building blocks
# =============================================================================

def _decimal_payout(american_odds: Optional[float]) -> Optional[float]:
    if american_odds is None or pd.isna(american_odds):
        return None
    o = float(american_odds)
    if o == 0:
        return None
    return 1.0 + (o / 100.0) if o > 0 else 1.0 + (100.0 / -o)


def _render_linescore_html(
    matchup: str,
    pick_team: str,
    score_map: Dict[str, Dict[str, Any]],
    innings_map: Dict[str, Dict[str, Any]],
) -> str:
    """Render the inning-by-inning + R/H/E + inning marker block."""
    if " @ " not in matchup:
        return ""
    away_full, home_full = [s.strip() for s in matchup.split(" @ ", 1)]
    away_lab = _short_team_label(away_full)
    home_lab = _short_team_label(home_full)
    sm_key = f"{_canonical_team_key(away_full)}|{_canonical_team_key(home_full)}"
    sm_game = score_map.get(sm_key) or {}
    inn = _innings_for_matchup(matchup, innings_map)

    if not inn:
        # Pre-game / no data — show empty grid
        cells_a = "".join('<td class="future">-</td>' for _ in range(9))
        cells_h = "".join('<td class="future">-</td>' for _ in range(9))
        return (
            '<div class="gc-line-wrap">'
            '<table class="gc-line-table"><thead><tr>'
            '<th class="team">&nbsp;</th>'
            + "".join(f'<th>{i}</th>' for i in range(1, 10))
            + '<th class="divider">R</th><th>H</th><th>E</th>'
            '</tr></thead><tbody>'
            f'<tr><td class="team">{_esc(away_lab)}</td>{cells_a}'
            f'<td class="tot divider">-</td><td class="tot-h">-</td><td class="tot-e">-</td></tr>'
            f'<tr><td class="team">{_esc(home_lab)}</td>{cells_h}'
            f'<td class="tot divider">-</td><td class="tot-h">-</td><td class="tot-e">-</td></tr>'
            '</tbody></table>'
            '<div class="gc-inning-marker">'
            '<div class="gc-inning-lab">Inning</div>'
            '<div class="gc-inning-val">Pre</div>'
            '<div class="gc-diamond"></div>'
            '</div></div>'
        )

    inn_a = inn["innings_away_runs"]
    inn_h = inn["innings_home_runs"]
    a_R = inn["away_R"]; h_R = inn["home_R"]
    a_H = inn["away_H"]; h_H = inn["home_H"]
    a_E = inn["away_E"]; h_E = inn["home_E"]
    cur = inn.get("current_inning")
    state = (inn.get("inning_state") or "").lower()
    is_final = bool(inn.get("is_final"))

    def _cell(val: Optional[int]) -> str:
        if val is None:
            return '<td class="future">-</td>'
        return f'<td>{int(val)}</td>'

    cells_a = "".join(_cell(v) for v in inn_a)
    cells_h = "".join(_cell(v) for v in inn_h)

    if is_final:
        marker_label = "Final"
    elif cur is not None:
        prefix = {"top": "Top", "middle": "Mid", "bottom": "Bot", "end": "End"}.get(state, "")
        marker_label = f"{prefix} {cur}".strip()
    else:
        marker_label = "Live"

    return (
        '<div class="gc-line-wrap">'
        '<table class="gc-line-table"><thead><tr>'
        '<th class="team">&nbsp;</th>'
        + "".join(f'<th>{i}</th>' for i in range(1, 10))
        + '<th class="divider">R</th><th>H</th><th>E</th>'
        '</tr></thead><tbody>'
        f'<tr><td class="team">{_esc(away_lab)}</td>{cells_a}'
        f'<td class="tot divider">{a_R}</td><td class="tot-h">{a_H}</td><td class="tot-e">{a_E}</td></tr>'
        f'<tr><td class="team">{_esc(home_lab)}</td>{cells_h}'
        f'<td class="tot divider">{h_R}</td><td class="tot-h">{h_H}</td><td class="tot-e">{h_E}</td></tr>'
        '</tbody></table>'
        '<div class="gc-inning-marker">'
        '<div class="gc-inning-lab">Inning</div>'
        f'<div class="gc-inning-val">{_esc(marker_label)}</div>'
        '<div class="gc-diamond"></div>'
        '</div></div>'
    )


def _render_metrics_strip(row: pd.Series) -> str:
    models = int(
        row.get("Models On Bet", row.get("Sharp Model Count", row.get("Model Count", 0))) or 0
    )
    conf = float(row.get("Avg Confidence", row.get("Confidence", 0.0)) or 0.0)
    sample = float(row.get("Avg Sample", row.get("Sample Reliability", 0.0)) or 0.0)
    return (
        '<div class="gc-metrics">'
        f'<div class="gc-metric"><div class="gc-metric-lab">Models</div>'
        f'<div class="gc-metric-val">{models}</div></div>'
        f'<div class="gc-metric"><div class="gc-metric-lab">Confidence</div>'
        f'<div class="gc-metric-val is-conf">{conf:.0f}%</div></div>'
        f'<div class="gc-metric"><div class="gc-metric-lab">Sample</div>'
        f'<div class="gc-metric-val">{sample:.2f}</div></div>'
        '</div>'
    )


def _render_prob_card(prob: float, trend: str, color: str) -> str:
    cls = "amber" if color == "#f59e0b" else ("rose" if color == "#f43f5e" else ("muted" if color == "#94a3b8" else ""))
    fill = max(0.0, min(100.0, prob))
    return (
        '<div class="gc-prob-card">'
        '<div class="gc-prob-lab">Live Hit Probability</div>'
        f'<div class="gc-prob-big" style="color:{color};">{prob:.1f}%</div>'
        f'<div class="gc-prob-trend {cls}">{_esc(trend)} <span style="opacity:0.7;">↗</span></div>'
        f'<div class="gc-prob-bar"><div class="gc-prob-bar-fill" style="width:{fill:.1f}%; background: linear-gradient(90deg, {color}, {color}cc);"></div></div>'
        '</div>'
    )


def _render_why_card(reasons: List[str]) -> str:
    items = "".join(
        f'<li><span class="gc-why-check">✓</span>{_esc(r)}</li>' for r in reasons
    )
    return (
        '<div class="gc-why">'
        '<div class="gc-why-lab">Why this side?</div>'
        f'<ul class="gc-why-list">{items}</ul>'
        '</div>'
    )


def _render_status_bar(
    matchup: str,
    pick_team: str,
    ticket: str,
    inn_data: Optional[Dict[str, Any]],
    prob: float,
) -> str:
    if " @ " not in matchup:
        return ""
    away, home = [s.strip() for s in matchup.split(" @ ", 1)]
    away_lab = _short_team_label(away)
    home_lab = _short_team_label(home)

    dot_cls = ticket.lower() if ticket.lower() in {"won", "lost", "push", "open"} else "open"
    if dot_cls == "won":
        ticket_word = "WON"
    elif dot_cls == "lost":
        ticket_word = "LOST"
    elif dot_cls == "push":
        ticket_word = "PUSH"
    else:
        ticket_word = "OPEN"

    phase_kind = _phase_from_inn(inn_data)
    if phase_kind == "final":
        phase = "Final"
    elif phase_kind == "live":
        cur = (inn_data or {}).get("current_inning") or 0
        state = ((inn_data or {}).get("inning_state") or "").lower()
        prefix = {"top": "Top", "middle": "Mid", "bottom": "Bot", "end": "End"}.get(state, "")
        phase = f"{prefix} {cur}".strip() if cur else "Live"
    else:
        phase = "Pre"

    a_R = inn_data.get("away_R", 0) if inn_data else 0
    h_R = inn_data.get("home_R", 0) if inn_data else 0
    score_text = f"{away_lab} {a_R} - {home_lab} {h_R}" if inn_data else "Score —"

    return (
        '<div class="gc-status">'
        '<div class="gc-status-left">'
        f'<span class="gc-status-lab">Bet status</span>'
        f'<span class="gc-status-dot {dot_cls}"></span>'
        f'<span class="gc-status-text">{ticket_word}</span>'
        '</div>'
        f'<div class="gc-status-right">'
        f'<span class="gc-status-text">{_esc(score_text)}</span>'
        f'<span class="gc-status-lab">·</span>'
        f'<span class="gc-status-text">{_esc(phase)}</span>'
        f'<span class="gc-status-lab">·</span>'
        f'<span class="gc-status-text" style="color:var(--gc-emerald);">{prob:.1f}%</span>'
        '</div>'
        '</div>'
    )


def _render_bet_slip(row: pd.Series, default_wager: float = 100.0) -> str:
    """Quick bet slip side card. Renders only if American Odds is present."""
    odds = row.get("American Odds", None)
    pick = str(row.get("Suggested F5 Pick", "F5"))
    if odds is None or pd.isna(odds):
        return ""
    odds_f = float(odds)
    dec = _decimal_payout(odds_f)
    to_win = (default_wager * (dec - 1.0)) if dec else 0.0
    odds_str = f"{int(odds_f):+d}"
    return (
        '<div class="gc-side-card">'
        '<div class="gc-side-head">'
        '<span class="gc-side-icon">⚡</span>'
        'Quick Bet Slip'
        '</div>'
        '<div class="gc-slip-row">'
        '<div><div class="gc-slip-label">Pick</div>'
        f'<div class="gc-slip-pick-name">{_esc(pick)}</div></div>'
        f'<div class="gc-slip-odds-chip">{_esc(odds_str)}</div>'
        '</div>'
        '<div class="gc-slip-row">'
        '<div class="gc-slip-label">Wager</div>'
        f'<div class="gc-slip-input">$ {default_wager:,.0f}</div>'
        '</div>'
        '<div class="gc-slip-row">'
        '<div class="gc-slip-label">To Win</div>'
        f'<div class="gc-slip-towin">$ {to_win:,.2f}</div>'
        '</div>'
        '<div class="gc-slip-button">Place Bet</div>'
        '</div>'
    )


def _render_key_reasons_card(reasons: List[str]) -> str:
    icons = ["★", "🛡", "📈", "⚙"]
    items = ""
    for i, r in enumerate(reasons[:4]):
        items += (
            '<div class="gc-key-item">'
            f'<span class="gc-key-item-icon">{_esc(icons[i % len(icons)])}</span>'
            f'<span>{_esc(r)}</span>'
            '</div>'
        )
    return (
        '<div class="gc-side-card">'
        '<div class="gc-side-head">'
        '<span class="gc-side-icon" style="background:var(--gc-cyan-soft); color:var(--gc-cyan);">🛡</span>'
        'Key Reasons'
        '</div>'
        f'{items}'
        '</div>'
    )


def evaluate_f5_criteria(row: pd.Series, intel: Dict[str, Any]) -> List[Tuple[str, bool]]:
    """
    Build a compact checklist for quick qualification context in the side panel.
    Uses available row/intel fields only (safe fallback when values are missing).
    """
    conf = float(row.get("Avg Confidence", 0.0) or 0.0)
    sample = float(row.get("Avg Sample", 0.0) or 0.0)
    models = int(row.get("Models On Bet", 0) or 0)
    era_gap = float(intel.get("era_gap", 0.0) or 0.0)
    k_diff = float(intel.get("k_diff", 0.0) or 0.0)

    return [
        ("Confidence >= 70%", conf >= 70.0),
        ("Sample >= 0.45", sample >= 0.45),
        ("Models aligned >= 2", models >= 2),
        ("ERA gap positive", era_gap > 0.0),
        ("K edge positive", k_diff > 0.0),
    ]


def _render_f5_criteria_card(criteria: List[Tuple[str, bool]]) -> str:
    if not criteria:
        return ""
    rows = ""
    for label, ok in criteria:
        mark = "✓" if ok else "•"
        cls = "color:var(--gc-emerald);" if ok else "color:var(--gc-text-3);"
        rows += (
            '<div class="gc-key-item">'
            f'<span class="gc-key-item-icon" style="{cls}">{_esc(mark)}</span>'
            f'<span>{_esc(label)}</span>'
            '</div>'
        )
    return (
        '<div class="gc-side-card">'
        '<div class="gc-side-head">'
        '<span class="gc-side-icon" style="background:rgba(16,185,129,0.14);color:var(--gc-emerald);">✓</span>'
        'F5 Criteria'
        '</div>'
        f'{rows}'
        '</div>'
    )


def _render_tracking_card(
    row: pd.Series,
    tracker_df: pd.DataFrame,
) -> str:
    """Tracking footer — pulls actual ROI/record from the user's tracker for this pick."""
    matchup = str(row.get("Matchup", ""))
    pick = str(row.get("Suggested F5 Pick", ""))

    wins = losses = 0
    if tracker_df is not None and not tracker_df.empty:
        sub = tracker_df[
            (tracker_df["matchup"].astype(str) == matchup)
            & (tracker_df["suggested_pick"].astype(str) == pick)
        ]
        if not sub.empty:
            status = sub["status"].astype(str).str.lower()
            wins = int((status == "win").sum())
            losses = int((status == "loss").sum())

    n_models = int(row.get("Models On Bet", 0) or 0)
    graded = wins + losses
    wr = (wins / graded * 100.0) if graded > 0 else 0.0
    roi = (wins * 0.909 - losses) / max(1, graded) * 100.0 if graded > 0 else 0.0

    avatars = ""
    initials = ["A", "B", "C", "D"]
    for i in range(min(3, max(1, n_models))):
        avatars += f'<div class="gc-track-avatar">{initials[i]}</div>'
    extra = max(0, n_models - 3)
    extra_html = f'<span class="gc-track-count">+{extra} models</span>' if extra > 0 else f'<span class="gc-track-count">{n_models} models</span>'

    if graded > 0:
        roi_cls = "" if roi >= 0 else "neg"
        track_metric = (
            f'<div class="gc-track-roi {roi_cls}">'
            f'<span style="font-size:0.72rem;letter-spacing:0.08em;color:var(--gc-text-3);font-weight:800;">'
            f'{wins}-{losses}</span> '
            f'{wr:.0f}% <span style="font-size:0.72rem;font-weight:700;color:var(--gc-text-3);">WR</span>'
            '</div>'
        )
    else:
        track_metric = (
            '<div class="gc-track-roi" style="color:var(--gc-text-2);">'
            'New play <span style="font-size:0.72rem;font-weight:700;color:var(--gc-text-3);">·</span> tracking'
            '</div>'
        )

    return (
        '<div class="gc-side-card" style="padding-top:10px;padding-bottom:10px;">'
        '<div class="gc-side-head" style="margin-bottom:6px;">'
        '<span class="gc-side-icon" style="background:rgba(34,211,238,0.16);color:#22d3ee;">📈</span>'
        'Tracking This Pick'
        '</div>'
        '<div class="gc-track-row">'
        f'<div class="gc-track-stack">{avatars}{extra_html}</div>'
        f'{track_metric}'
        '</div>'
        '</div>'
    )


# =============================================================================
# Public renderers
# =============================================================================

def render_gamecast_hero(
    row: Optional[pd.Series],
    score_map: Dict[str, Dict[str, Any]],
    tracker_df: pd.DataFrame,
    innings_map: Optional[Dict[str, Dict[str, Any]]] = None,
    matchup_intel: Optional[Dict[str, Dict[str, Any]]] = None,
    label: str = "Lock of the Day",
    eyebrow: str = "Sharp Pick · Lock of the Day",
    badge: str = "#1 Priority",
) -> None:
    """Full-size gamecast for the Lock of the Day."""
    if row is None or (hasattr(row, "empty") and getattr(row, "empty", False)):
        st.markdown(
            '<div class="gc-empty">'
            '<div class="gc-empty-title">No qualified Lock of the Day yet</div>'
            '<div class="gc-empty-sub">Models are still evaluating today\'s slate. Check back closer to first pitch.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    innings_map = innings_map or {}
    tier = str(row.get("Tier", "CORE")).upper()
    tier_cls = _tier_class_suffix(tier)
    matchup = str(row.get("Matchup", ""))
    pick = str(row.get("Suggested F5 Pick", "No Play"))
    pick_team = re.sub(r"\bF5\s*", "", pick, flags=re.I).strip()
    intel = (matchup_intel or {}).get(matchup) or {}

    # Probability + ticket. Fallback confidence prevents nonsense ultra-low live
    # probabilities on cards that don't carry consensus fields.
    base_conf = float(row.get("Avg Confidence", row.get("Confidence", 0.0)) or 0.0)
    models_for_prob = int(
        row.get("Models On Bet", row.get("Sharp Model Count", row.get("Model Count", 0))) or 0
    )
    if base_conf <= 0:
        base_conf = 55.0 if models_for_prob <= 0 else 50.0
    prob, trend, color = _hit_probability(matchup, pick, base_conf, score_map)
    ticket = _ticket_status_for(row, tracker_df)
    inn_data = _innings_for_matchup(matchup, innings_map)

    why = _why_so_high_reasons(row, intel)
    keys = _key_reasons(row, intel)

    linescore_html = _render_linescore_html(matchup, pick_team, score_map, innings_map)
    metrics_html = _render_metrics_strip(row)
    prob_html = _render_prob_card(prob, trend, color)
    why_html = _render_why_card(why)
    status_html = _render_status_bar(matchup, pick_team, ticket, inn_data, prob)
    slip_html = _render_bet_slip(row)
    keys_html = _render_key_reasons_card(keys)
    track_html = _render_tracking_card(row, tracker_df)
    criteria = evaluate_f5_criteria(row, intel)
    criteria_html = _render_f5_criteria_card(criteria)

    # Header strip — LIVE pill only when the game is actually in progress.
    phase = _phase_from_inn(inn_data)
    head_pill = (
        '<span class="gc-live-pill">LIVE</span>' if phase == "live"
        else ('<span class="gc-live-pill" style="background:rgba(148,163,184,0.18);color:#94a3b8;">FINAL</span>' if phase == "final"
              else '<span class="gc-live-pill" style="background:rgba(148,163,184,0.18);color:#94a3b8;">PRE</span>')
    )

    main_html = (
        f'<div class="gc-eyebrow">{_esc(eyebrow)}</div>'
        f'<div class="gc-topbar">'
        f'<div class="gc-topbar-meta">'
        f'{_responsive_matchup_html(matchup)}'
        f'<div class="gc-pickline">Pick: <b>{_esc(pick)}</b></div>'
        f'</div>'
        f'{metrics_html}'
        f'</div>'
        f'{linescore_html}'
        f'<div class="gc-prob-row">{prob_html}{why_html}</div>'
        f'{status_html}'
    )

    side_html = (
        '<div class="gc-side">'
        f'{criteria_html}'
        f'{slip_html}'
        f'{keys_html}'
        f'{track_html}'
        '</div>'
    )

    full_html = (
        f'<div class="gc-card tier-{tier_cls}">'
        '<div class="gc-head">'
        '<div class="gc-head-left">'
        f'{head_pill}'
        f'<div class="gc-head-title">{_esc(label)}</div>'
        f'<span class="gc-tier-pill tier-{tier_cls}">{_esc(badge)}</span>'
        '</div>'
        '<div></div>'
        '</div>'
        '<div class="gc-sub">Highest-tier qualified play across every model.</div>'
        '<div class="gc-grid">'
        f'<div class="gc-main">{main_html}</div>'
        f'{side_html}'
        '</div>'
        '</div>'
    )
    st.markdown(full_html, unsafe_allow_html=True)


def render_gamecast_card(
    row: pd.Series,
    score_map: Dict[str, Dict[str, Any]],
    tracker_df: pd.DataFrame,
    innings_map: Optional[Dict[str, Dict[str, Any]]] = None,
    matchup_intel: Optional[Dict[str, Dict[str, Any]]] = None,
    label: str = "Featured Pick",
    eyebrow: str = "Featured · Season Leader",
    badge: str = "FEATURED",
    rank: Optional[int] = None,
) -> None:
    """Slightly tighter gamecast — same content as hero but smaller matchup type.
    Use this for the season-leader Featured pick and Top Picks #2-#5."""
    innings_map = innings_map or {}
    tier = str(row.get("Tier", "CORE")).upper()
    tier_cls = _tier_class_suffix(tier)
    matchup = str(row.get("Matchup", ""))
    pick = str(row.get("Suggested F5 Pick", "No Play"))
    pick_team = re.sub(r"\bF5\s*", "", pick, flags=re.I).strip()
    intel = (matchup_intel or {}).get(matchup) or {}

    base_conf = float(row.get("Avg Confidence", row.get("Confidence", 0.0)) or 0.0)
    models_for_prob = int(
        row.get("Models On Bet", row.get("Sharp Model Count", row.get("Model Count", 0))) or 0
    )
    if base_conf <= 0:
        base_conf = 55.0 if models_for_prob <= 0 else 50.0
    prob, trend, color = _hit_probability(matchup, pick, base_conf, score_map)
    ticket = _ticket_status_for(row, tracker_df)
    inn_data = _innings_for_matchup(matchup, innings_map)

    why = _why_so_high_reasons(row, intel)
    keys = _key_reasons(row, intel)

    linescore_html = _render_linescore_html(matchup, pick_team, score_map, innings_map)
    metrics_html = _render_metrics_strip(row)
    prob_html = _render_prob_card(prob, trend, color)
    why_html = _render_why_card(why)
    status_html = _render_status_bar(matchup, pick_team, ticket, inn_data, prob)
    slip_html = _render_bet_slip(row)
    keys_html = _render_key_reasons_card(keys)
    track_html = _render_tracking_card(row, tracker_df)
    criteria = evaluate_f5_criteria(row, intel)
    criteria_html = _render_f5_criteria_card(criteria)

    phase = _phase_from_inn(inn_data)
    head_pill = (
        '<span class="gc-live-pill">LIVE</span>' if phase == "live"
        else ('<span class="gc-live-pill" style="background:rgba(148,163,184,0.18);color:#94a3b8;">FINAL</span>' if phase == "final"
              else '<span class="gc-live-pill" style="background:rgba(148,163,184,0.18);color:#94a3b8;">PRE</span>')
    )
    rank_html = f'<span class="gc-tier-pill" style="opacity:0.85;">#{int(rank)}</span>' if rank is not None else ""

    main_html = (
        f'<div class="gc-eyebrow">{_esc(eyebrow)}</div>'
        f'<div class="gc-topbar">'
        f'<div class="gc-topbar-meta">'
        f'{_responsive_matchup_html(matchup, "gc-matchup-card")}'
        f'<div class="gc-pickline">Pick: <b>{_esc(pick)}</b></div>'
        f'</div>'
        f'{metrics_html}'
        f'</div>'
        f'{linescore_html}'
        f'<div class="gc-prob-row">{prob_html}{why_html}</div>'
        f'{status_html}'
    )

    side_html = (
        '<div class="gc-side">'
        f'{criteria_html}'
        f'{slip_html}'
        f'{keys_html}'
        f'{track_html}'
        '</div>'
    )

    full_html = (
        f'<div class="gc-card tier-{tier_cls}">'
        '<div class="gc-head">'
        '<div class="gc-head-left">'
        f'{head_pill}'
        f'<div class="gc-head-title">{_esc(label)}</div>'
        f'<span class="gc-tier-pill tier-{tier_cls}">{_esc(badge)}</span>'
        f'{rank_html}'
        '</div>'
        '</div>'
        '<div class="gc-grid">'
        f'<div class="gc-main">{main_html}</div>'
        f'{side_html}'
        '</div>'
        '</div>'
    )
    st.markdown(full_html, unsafe_allow_html=True)


def render_gamecast_mini_list(
    df: pd.DataFrame,
    score_map: Dict[str, Dict[str, Any]],
    tracker_df: pd.DataFrame,
    innings_map: Optional[Dict[str, Dict[str, Any]]] = None,
    start_rank: int = 1,
    section_subtitle: Optional[str] = None,
) -> None:
    """Compact one-row gamecast for every row of df.
    Use this for the long Best Bets list."""
    if df is None or df.empty:
        st.markdown(
            '<div class="gc-empty">'
            '<div class="gc-empty-title">No qualified bets at this tier today</div>'
            '<div class="gc-empty-sub">Models are still evaluating — check back soon.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    innings_map = innings_map or {}
    if section_subtitle:
        st.caption(section_subtitle)

    blocks: List[str] = []
    for i, (_, row) in enumerate(df.iterrows(), start=start_rank):
        tier = str(row.get("Tier", "CORE")).upper()
        tier_cls = _tier_class_suffix(tier)
        tier_lab = _TIER_META.get(tier, {}).get("label", tier)
        matchup = str(row.get("Matchup", ""))
        pick = str(row.get("Suggested F5 Pick", ""))

        base_conf = float(row.get("Avg Confidence", row.get("Confidence", 0.0)) or 0.0)
        models_for_prob = int(
            row.get("Models On Bet", row.get("Sharp Model Count", row.get("Model Count", 0))) or 0
        )
        if base_conf <= 0:
            base_conf = 55.0 if models_for_prob <= 0 else 50.0
        prob, trend, color = _hit_probability(matchup, pick, base_conf, score_map)
        ticket = _ticket_status_for(row, tracker_df)
        inn_data = _innings_for_matchup(matchup, innings_map)

        # Score line
        if " @ " in matchup:
            away, home = [s.strip() for s in matchup.split(" @ ", 1)]
            away_lab = _short_team_label(away)
            home_lab = _short_team_label(home)
        else:
            away_lab = home_lab = "—"

        if inn_data:
            a_R = inn_data.get("away_R", 0)
            h_R = inn_data.get("home_R", 0)
            cur = inn_data.get("current_inning") or 0
            state = (inn_data.get("inning_state") or "").lower()
            is_final = bool(inn_data.get("is_final"))
            if is_final:
                stat_tag = "Final · F5"
            elif cur:
                prefix = {"top": "Top", "middle": "Mid", "bottom": "Bot", "end": "End"}.get(state, "")
                stat_tag = f"{prefix} {cur}".strip()
            else:
                stat_tag = "Pre · F5"
            score_html = (
                '<div>'
                '<div class="gc-mini-line">'
                '<div class="gc-mini-side">'
                f'<span class="gc-mini-team">{_esc(away_lab)}</span>'
                f'<span class="gc-mini-score-num">{a_R}</span>'
                '</div>'
                '<span class="gc-mini-vs">·</span>'
                '<div class="gc-mini-side">'
                f'<span class="gc-mini-score-num">{h_R}</span>'
                f'<span class="gc-mini-team">{_esc(home_lab)}</span>'
                '</div>'
                '</div>'
                f'<div class="gc-mini-stat-tag">{_esc(stat_tag)}</div>'
                '</div>'
            )
        else:
            score_html = (
                '<div>'
                '<div class="gc-mini-line">'
                f'<div class="gc-mini-side"><span class="gc-mini-team">{_esc(away_lab)}</span><span class="gc-mini-score-num" style="color:var(--gc-text-3);">—</span></div>'
                '<span class="gc-mini-vs">·</span>'
                f'<div class="gc-mini-side"><span class="gc-mini-score-num" style="color:var(--gc-text-3);">—</span><span class="gc-mini-team">{_esc(home_lab)}</span></div>'
                '</div>'
                '<div class="gc-mini-stat-tag">Pre · F5</div>'
                '</div>'
            )

        # Prob column
        prob_cls = "amber" if color == "#f59e0b" else ("rose" if color == "#f43f5e" else ("muted" if color == "#94a3b8" else ""))
        prob_html = (
            '<div class="gc-mini-prob-col">'
            f'<div class="gc-mini-prob {prob_cls}" style="color:{color};">{prob:.0f}%</div>'
            f'<div class="gc-mini-prob-bar"><div class="gc-mini-prob-fill" style="width:{max(0.0, min(100.0, prob)):.0f}%; background:linear-gradient(90deg, {color}, {color}aa);"></div></div>'
            f'<div class="gc-mini-prob-trend">{_esc(trend)}</div>'
            '</div>'
        )

        # Ticket column
        tcls = ticket.lower() if ticket.lower() in {"won", "lost", "push", "open"} else "open"
        ev = row.get("EV", None)
        ev_html = ""
        if ev is not None and pd.notna(ev):
            ev_f = float(ev)
            color_ev = "var(--gc-emerald)" if ev_f > 0 else "var(--gc-rose)"
            ev_html = f'<div class="gc-mini-phase" style="color:{color_ev};">EV {ev_f:+.2f}u</div>'
        odds = row.get("American Odds", None)
        odds_html = ""
        if odds is not None and pd.notna(odds):
            odds_html = f'<div class="gc-mini-phase">{int(float(odds)):+d}</div>'

        status_col = (
            '<div class="gc-mini-status-col">'
            f'<span class="gc-mini-ticket {tcls}">{_esc(ticket)}</span>'
            f'{odds_html}'
            f'{ev_html}'
            '</div>'
        )

        # Pick column
        models = int(row.get("Models On Bet", 0) or 0)
        sample = float(row.get("Avg Sample", 0.0) or 0.0)
        meta_bits = (
            f'<span>Models <b>{models}</b></span>'
            f'<span>Conf <b>{base_conf:.0f}%</b></span>'
            f'<span>Sample <b>{sample:.2f}</b></span>'
        )
        pick_col = (
            '<div class="gc-mini-pick-col">'
            f'<div class="gc-mini-matchup"><span class="gc-matchup-full">{_esc(matchup)}</span><span class="gc-matchup-abbr">{_esc(_abbr_matchup(matchup))}</span></div>'
            f'<div class="gc-mini-pick">Pick: <b>{_esc(pick)}</b></div>'
            f'<div class="gc-mini-meta">{meta_bits}</div>'
            '</div>'
        )

        # Rank column
        rank_col = (
            '<div class="gc-mini-rank-col">'
            f'<span class="gc-mini-tierpill tier-{tier_cls}">{_esc(tier_lab)}</span>'
            f'<span class="gc-mini-rank">#{i}</span>'
            '</div>'
        )

        blocks.append(
            f'<div class="gc-mini tier-{tier_cls}">'
            f'{rank_col}'
            f'{pick_col}'
            f'{score_html}'
            f'{prob_html}'
            f'{status_col}'
            '</div>'
        )

    st.markdown("".join(blocks), unsafe_allow_html=True)


def render_gamecast_empty(message: str = "No qualified bets yet", sub: str = "Models are still evaluating today's slate.") -> None:
    st.markdown(
        '<div class="gc-empty">'
        f'<div class="gc-empty-title">{_esc(message)}</div>'
        f'<div class="gc-empty-sub">{_esc(sub)}</div>'
        '</div>',
        unsafe_allow_html=True,
    )
