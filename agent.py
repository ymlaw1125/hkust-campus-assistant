# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging
import os
import re
import time
import datetime as dt
from datetime import datetime
from typing import Optional

import httpx
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.identity import AzureCliCredential, get_bearer_token_provider

from agent_interface import AgentInterface
from microsoft_agents.hosting.core import Authorization, TurnContext
from bus_client import get_all_hkust_etas, format_etas_for_agent
from library_client import (
    get_slot_availability,
    get_slots_range,
    find_rooms_free_until,
    find_best_room,
    book_room,
    format_slot_for_agent,
    format_range_for_agent,
    format_continuous_for_agent,
    format_room_recommendation,
    format_booking_confirmation,
    print_full_day_text,
    get_room_conflict_info,
    LC_ROOMS,
    STUDY_ROOMS,
)
from sports_client import (
    resolve_sport,
    match_venue_hint,
    find_available_venue,
    free_courts,
    book_facility,
    format_venue_recommendation,
    format_sports_confirmation,
    list_venues_for_sport,
    print_sports_day_text,
    SPORT_INFO,
    SPORT_LABELS,
    SPORT_VENUES,
    SPORT_ALIASES,
    FACILITIES,
)

load_dotenv()
logger = logging.getLogger(__name__)

# Strip ANY internal control tag like [BOOKING CONFIRMED], [NOTE], [SLOT LIMIT REACHED]
# that the LLM might echo. Matches an opening UPPERCASE bracketed tag generically.
_INTERNAL_TAGS = re.compile(r'\[[A-Z][A-Z0-9 /_-]{1,40}\]')


def _parse_hour(h: int, meridiem: Optional[str], fallback_pm_if_low: bool = True) -> int:
    """Normalise an hour value with optional am/pm. If no meridiem and h < 8, assume PM."""
    if meridiem == "pm" and h < 12:
        return h + 12
    if meridiem == "am" and h == 12:
        return 0
    if meridiem is None and fallback_pm_if_low and 1 <= h <= 7:
        return h + 12
    return h


def _parse_time_str(text: str) -> Optional[tuple]:
    """
    Extract the first (hour, minute) from a text fragment.
    Returns (hour, minute) in 24h, or None.
    Handles: "7pm", "7:30pm", "7:30", "19:00", "7", "now"
    """
    if "now" in text:
        n = datetime.now()
        return (n.hour, 30 if n.minute >= 30 else 0)

    # HH:MM am/pm
    m = re.search(r'\b(\d{1,2}):(\d{2})\s*(am|pm)?\b', text)
    if m:
        h = _parse_hour(int(m.group(1)), m.group(3))
        minute = 30 if int(m.group(2)) >= 30 else 0
        return (h, minute)

    # H am/pm
    m = re.search(r'\b(\d{1,2})\s*(am|pm)\b', text)
    if m:
        h = _parse_hour(int(m.group(1)), m.group(2))
        return (h, 0)

    # bare integer — use PM heuristic
    m = re.search(r'\b(\d{1,2})\b', text)
    if m:
        h = _parse_hour(int(m.group(1)), None)
        return (h, 0)

    return None


def _parse_booking_intent(msg: str, now: datetime) -> dict:
    """
    Parse a booking request from natural language.
    Returns dict with keys: room_id (opt), date, start_hour, start_minute,
                             end_hour (opt), end_minute (opt), num_slots (opt, 0=unknown),
                             capacity.
    """
    low = msg.lower()
    result: dict = {}

    # ── Specific room ────────────────────────────────────────────────
    room_match = re.search(r'\b(lc-\d+|1f-r\d+|lg[134]-r\d+)\b', low)
    if room_match:
        result["room_id"] = room_match.group(1).upper()

    # ── Date ────────────────────────────────────────────────────────
    if any(w in low for w in ["tomorrow", "tmr", "tmrw"]):
        result["date"] = (now + dt.timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        result["date"] = now.strftime("%Y-%m-%d")

    cur_h = now.hour
    cur_m = 30 if now.minute >= 30 else 0

    # ── Time range  "X-Y" / "X to Y" / "X–Y" ───────────────────────
    range_pat = re.compile(
        r'\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*(?:-|–|to)\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b'
    )
    rm = range_pat.search(low)
    if rm:
        sh = int(rm.group(1))
        sm = int(rm.group(2)) if rm.group(2) else 0
        s_mer = rm.group(3)
        eh = int(rm.group(4))
        em = int(rm.group(5)) if rm.group(5) else 0
        e_mer = rm.group(6)

        # Infer missing meridiem: if end has "pm", assume start is also pm unless before
        if e_mer == "pm" and s_mer is None and sh <= eh:
            s_mer = "pm"
        if s_mer == "am" and e_mer is None and eh > sh:
            e_mer = "am"

        sh = _parse_hour(sh, s_mer)
        eh = _parse_hour(eh, e_mer)
        sm = 30 if sm >= 30 else 0
        em = 30 if em >= 30 else 0

        result["start_hour"] = sh
        result["start_minute"] = sm
        result["end_hour"] = eh
        result["end_minute"] = em
        total_min = (eh * 60 + em) - (sh * 60 + sm)
        result["num_slots"] = max(1, total_min // 30)
        result["capacity"] = _parse_capacity(low)
        return result

    # ── Start time ──────────────────────────────────────────────────
    if re.search(r'\bnow\b', low):
        # "now", "from now", "starting now" — use current slot
        result["start_hour"] = cur_h
        result["start_minute"] = cur_m
    else:
        sm = re.search(r'(?:at|from|@|starting)\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', low)
        if sm:
            h = _parse_hour(int(sm.group(1)), sm.group(3))
            m = 30 if sm.group(2) and int(sm.group(2)) >= 30 else 0
            result["start_hour"] = h
            result["start_minute"] = m

    # ── End time ────────────────────────────────────────────────────
    um = re.search(r'(?:until|till|to)\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', low)
    if um:
        h = _parse_hour(int(um.group(1)), um.group(3))
        m = 30 if um.group(2) and int(um.group(2)) >= 30 else 0
        result["end_hour"] = h
        result["end_minute"] = m

    # ── Apply start default BEFORE deriving num_slots ────────────────
    if "start_hour" not in result:
        result["start_hour"] = cur_h
        result["start_minute"] = cur_m
    if "start_minute" not in result:
        result["start_minute"] = 0

    # ── Duration ────────────────────────────────────────────────────
    dur_h = re.search(r'(\d+)\s*hour', low)
    dur_m = re.search(r'(\d+)\s*(?:min|mins|minutes)', low)
    if dur_h:
        result["num_slots"] = min(4, int(dur_h.group(1)) * 2)
    elif dur_m:
        result["num_slots"] = max(1, min(4, round(int(dur_m.group(1)) / 30)))
    elif any(k in low for k in ["meeting", "interview", "presentation", "discussion"]):
        result["num_slots"] = 2
    elif any(k in low for k in ["lecture", "seminar", "workshop", "class"]):
        result["num_slots"] = 4
    elif any(k in low for k in ["quick", "brief", "short", "half hour", "half an hour"]):
        result["num_slots"] = 1

    # Derive num_slots from start+end
    if "num_slots" not in result and "end_hour" in result:
        sh = result["start_hour"]
        sm2 = result["start_minute"]
        eh = result["end_hour"]
        em2 = result.get("end_minute", 0)
        total_min = (eh * 60 + em2) - (sh * 60 + sm2)
        result["num_slots"] = max(1, min(4, total_min // 30))

    if "num_slots" not in result:
        result["num_slots"] = 0  # unknown — will ask

    result["capacity"] = _parse_capacity(low)
    return result


def _parse_capacity(low: str) -> int:
    m = re.search(r'(\d+)\s*(?:people|person|ppl|pax|of us|students|attendees)', low)
    return int(m.group(1)) if m else 1


def _round_up_slot(hour: int, minute: int) -> tuple:
    """Round a time UP to the next 30-min slot boundary."""
    if minute == 0:
        return (hour, 0)
    if minute <= 30:
        return (hour, 30)
    return (hour + 1, 0)


def _parse_floor_pref(low: str) -> Optional[str]:
    """Extract a floor/room-type preference from a message. Returns '1/F','LG1','LG3','LG4','LC' or None."""
    if re.search(r'\blg\s*1\b|\blg1\b', low):
        return "LG1"
    if re.search(r'\blg\s*3\b|\blg3\b', low):
        return "LG3"
    if re.search(r'\blg\s*4\b|\blg4\b', low):
        return "LG4"
    if re.search(r'\b1\s*/?\s*f\b|\bfirst floor\b|\b1f\b', low):
        return "1/F"
    if "learning commons" in low or re.search(r'\blc\b', low):
        return "LC"
    return None


# ── HK origins → how to reach HKUST ──────────────────────────────────────────
# Only 5 bus routes serve HKUST, each from a fixed interchange. For origins AT an
# interchange ("near"), you board the bus directly and the live ETA is meaningful.
# For everywhere else you MTR to the interchange ("via") first, so we DON'T show a
# precise next-bus time (you're not there yet) — we route honestly, no made-up ETA.
# travel_min is a rough door-to-door estimate to campus; gate is where you alight.
HK_ORIGINS = {
    # On/near an HKUST interchange — board the bus directly, live ETA applies
    "hang hau":       {"route": "11M",  "gate": "North Gate", "travel_min": 22, "near": True},
    "po lam":         {"route": "91M",  "gate": "North Gate", "travel_min": 28, "near": True},
    "diamond hill":   {"route": "91M",  "gate": "South Gate", "travel_min": 35, "near": True},
    "choi hung":      {"route": "91M",  "gate": "South Gate", "travel_min": 32, "near": True},
    "tseung kwan o":  {"route": "792M", "gate": "North Gate", "travel_min": 28, "near": True},
    "tko":            {"route": "792M", "gate": "North Gate", "travel_min": 28, "near": True},
    "tiu keng leng":  {"route": "792M", "gate": "North Gate", "travel_min": 30, "near": True},
    "sai kung":       {"route": "12",   "gate": "North Gate", "travel_min": 35, "near": True},
    # East Kowloon → MTR to Choi Hung / Diamond Hill, then 91M
    "san po kong":    {"route": "91M",  "via": "Choi Hung",   "gate": "South Gate", "travel_min": 42},
    "wong tai sin":   {"route": "91M",  "via": "Diamond Hill","gate": "South Gate", "travel_min": 44},
    "ngau chi wan":   {"route": "91M",  "via": "Choi Hung",   "gate": "South Gate", "travel_min": 38},
    "kowloon bay":    {"route": "91M",  "via": "Choi Hung",   "gate": "South Gate", "travel_min": 38},
    "ngau tau kok":   {"route": "91M",  "via": "Choi Hung",   "gate": "South Gate", "travel_min": 40},
    "kwun tong":      {"route": "91M",  "via": "Choi Hung",   "gate": "South Gate", "travel_min": 44},
    "lam tin":        {"route": "792M", "via": "Tseung Kwan O","gate": "North Gate","travel_min": 40},
    "yau tong":       {"route": "792M", "via": "Tseung Kwan O","gate": "North Gate","travel_min": 38},
    "kowloon tong":   {"route": "91M",  "via": "Diamond Hill","gate": "South Gate", "travel_min": 46},
    "lok fu":         {"route": "91M",  "via": "Diamond Hill","gate": "South Gate", "travel_min": 46},
    # Central / West Kowloon → MTR to Diamond Hill, then 91M
    "mong kok":       {"route": "91M",  "via": "Diamond Hill","gate": "South Gate", "travel_min": 52},
    "prince edward":  {"route": "91M",  "via": "Diamond Hill","gate": "South Gate", "travel_min": 53},
    "yau ma tei":     {"route": "91M",  "via": "Diamond Hill","gate": "South Gate", "travel_min": 54},
    "jordan":         {"route": "91M",  "via": "Diamond Hill","gate": "South Gate", "travel_min": 55},
    "tsim sha tsui":  {"route": "91M",  "via": "Diamond Hill","gate": "South Gate", "travel_min": 56},
    "tst":            {"route": "91M",  "via": "Diamond Hill","gate": "South Gate", "travel_min": 56},
    "sham shui po":   {"route": "91M",  "via": "Diamond Hill","gate": "South Gate", "travel_min": 54},
    "hung hom":       {"route": "91M",  "via": "Diamond Hill","gate": "South Gate", "travel_min": 50},
    # HK Island East → on/near the TKO line, so MTR to Hang Hau, then 11M
    "north point":    {"route": "11M",  "via": "Hang Hau",    "gate": "North Gate", "travel_min": 50},
    "quarry bay":     {"route": "11M",  "via": "Hang Hau",    "gate": "North Gate", "travel_min": 48},
    "tai koo":        {"route": "11M",  "via": "Hang Hau",    "gate": "North Gate", "travel_min": 48},
    # HK Island Central/West → MTR east to the TKO line at North Point, then 11M
    "central":        {"route": "11M",  "via": "Hang Hau",    "gate": "North Gate", "travel_min": 58},
    "admiralty":      {"route": "11M",  "via": "Hang Hau",    "gate": "North Gate", "travel_min": 56},
    "wan chai":       {"route": "11M",  "via": "Hang Hau",    "gate": "North Gate", "travel_min": 55},
    "causeway bay":   {"route": "11M",  "via": "Hang Hau",    "gate": "North Gate", "travel_min": 54},
    # New Territories → MTR to Diamond Hill, then 91M
    "sha tin":        {"route": "91M",  "via": "Diamond Hill","gate": "South Gate", "travel_min": 55},
    "tai wai":        {"route": "91M",  "via": "Diamond Hill","gate": "South Gate", "travel_min": 52},
    "tai po":         {"route": "91M",  "via": "Diamond Hill","gate": "South Gate", "travel_min": 65},
    "tsuen wan":      {"route": "91M",  "via": "Diamond Hill","gate": "South Gate", "travel_min": 70},
    "airport":        {"route": "91M",  "via": "Diamond Hill","gate": "South Gate", "travel_min": 85},
}

ARRIVAL_PHRASES = [
    "when i arrive", "when i get there", "when i get to", "when i reach",
    "on arrival", "by the time i arrive", "once i arrive", "after i arrive",
    "when im there", "when i'm there", "when i get in",
]


class AgentFrameworkAgent(AgentInterface):

    AGENT_PROMPT = """You are a helpful HKUST campus assistant for bus transport, library room bookings, and sports facility bookings.

BUS STOP LOCATIONS AT HKUST:
- NORTH GATE: 91M (towards Po Lam / Hang Hau), 11, 11M, 12, 792M
- SOUTH GATE: 91M (towards Diamond Hill / Choi Hung), 11

BOARDING RULES:
- 91M to Diamond Hill or Choi Hung → South Gate
- 91M to Hang Hau or Po Lam → North Gate
- 11M to Hang Hau → North Gate
- 792M to TKO or Sai Kung → North Gate
- 12 to Po Lam or Sai Kung → North Gate
- 11 towards Choi Hung → South Gate
- 11 towards Hang Hau → North Gate

BUSES SERVING HKUST — ONLY these 5 routes:
- 91M: KMB — Diamond Hill MTR ↔ HKUST ↔ Po Lam
- 11: GMB — Choi Hung MTR ↔ HKUST ↔ Hang Hau Village
- 11M: GMB — Hang Hau MTR ↔ HKUST
- 12: GMB — Po Lam ↔ HKUST ↔ Sai Kung
- 792M: Citybus — TKO Station ↔ HKUST ↔ Sai Kung

ROUTING:
- TKO: 792M direct, or 11M to Hang Hau MTR then MTR
- Diamond Hill / Kowloon: 91M
- Sai Kung: 12 or 792M
- Hang Hau MTR: 11M (fastest), 11, or 792M
- Choi Hung MTR: 91M or 11
- Po Lam: 91M or 12
- ONLY use [LIVE BUS DATA] for ETAs — never invent times
- If direction is ambiguous, ask ONE clarifying question

LIBRARY (lbbooking.ust.hk):
- Rooms bookable in 30-min slots, max 4 consecutive (2 hours).
- Learning Commons LC-1–13 (seat 7), LC-14–17 (seat 10), LC-18 (seat 6). All on LG1.
- Study Rooms: 1/F (1F-R1–4), LG1 (LG1-R1–9), LG3 (LG3-R1–9), LG4 (LG4-R1–11). All seat ~4.
- Hours: 8am–11pm.

SPORTS FACILITIES (fbs.hkust.edu.hk):
- 1-hour slots, 1 slot per sport per day (all venues of a sport share one quota), booking opens 8am 7 days ahead.
- Bookable: badminton, squash, table tennis, tennis, basketball, football, volleyball, handball, netball, pickleball, climbing, gym, dance, fencing, martial arts, archery.
- Walk-in (no booking): running (400m track), swimming (pool).
- Contact only: water sports (dragon boat, rowing, windsurfing, kayaking, sailing).
- The system handles all sports availability and booking — never invent venues or court availability.

BOOKING FLOW:
- [BOOKING RECOMMENDATION]: Present the room name, floor, seats, date, and time range. Ask "Shall I book this for you?" — one short question only.
- [BOOKING CONFIRMED]: Booking succeeded. Show the reference number and say it's confirmed.
- [BOOKING FAILED]: Explain what went wrong in one sentence.
- [BOOKING DURATION NEEDED]: Ask ONLY "How long do you need the room?" — nothing else.
- [SLOT LIMIT REACHED]: Tell the user they've used their 2-hour daily limit.
- [MY BOOKINGS]: Show the user's bookings as a simple list.
- [CANCEL CONFIRMED]: Tell the user the booking was cancelled.
- [CANCEL NOT FOUND]: Tell the user the booking reference wasn't found.
- [PLANNING RESULT]: Show the bus plan (route, next bus, depart time) and the room recommendation together in one response. Be concise. Ask to confirm the room booking at the end.
- [NOTE]: Follow the note instruction — don't repeat the note to the user.
- Never say "I'll book it" or "Let me book" — the system handles actual booking.
- Never invent reference numbers. Only show references from [BOOKING CONFIRMED].
- Never ask the user what time it is — the system always knows the current time.
- Don't ask for room type preferences unless the user specifically asks.

RESPONSE RULES:
- Be concise and direct. No thinking out loud.
- Never show internal tags in your response.
- Never invent bus times or room availability.
- Keep responses short — one or two sentences for confirmations.
"""

    MAX_SLOTS_PER_DAY = 4  # 2 hours total per user per day

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.conversation_history = []
        self.pending_booking = None
        self._waiting_duration = False
        self._duration_context = None
        self.my_bookings: list = []
        self._pending_cancel_ref: Optional[str] = None
        self._edit_after_cancel: bool = False
        self._rejected_booking_ctx: Optional[dict] = None
        self._pending_edit: Optional[dict] = None
        self.pending_sports: Optional[dict] = None
        self._waiting_sports_time: Optional[dict] = None
        self._pending_multi: Optional[list] = None
        self._plan_ctx: Optional[dict] = None
        self._create_client()

    def _create_client(self):
        load_dotenv(override=True)  # ensure .env wins over any injected/stale process env
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        endpoint = endpoint.rstrip("/").replace("/openai/v1", "").replace("/openai", "")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
        self.deployment = deployment

        # HTTP client for the model calls. By default we IGNORE proxy env vars
        # (HTTP_PROXY/HTTPS_PROXY) and connect directly to Azure — a local proxy
        # (e.g. 127.0.0.1:8118) often can't tunnel to the model endpoint and yields
        # "APIConnectionError: Connection error". Azure is directly reachable, so
        # trust_env=False is the reliable choice. Set AZURE_USE_PROXY=1 to opt back in.
        use_proxy = os.getenv("AZURE_USE_PROXY", "").strip() in ("1", "true", "True")
        http_client = httpx.Client(trust_env=use_proxy, timeout=60.0)

        # ── Preferred: API-key auth ───────────────────────────────────────────
        # Reliable and instant — avoids the Azure CLI / managed-identity probing
        # that fails when `az` isn't on the debug process PATH. Set AZURE_OPENAI_API_KEY
        # in .env (Azure Portal → your AI resource → Keys and Endpoint).
        api_key = (os.getenv("AZURE_OPENAI_API_KEY") or "").strip()
        if api_key and api_key.lower() != "placeholder":
            self.client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=api_version,
                http_client=http_client,
            )
            logger.info(f"✅ Client created (API key, direct) → {deployment}")
            return

        # ── Fallback: Microsoft Entra (AAD) token ─────────────────────────────
        # NOTE: this path is much slower and brittle (needs `az` on the process PATH).
        # Prefer the API key above. We bound the CLI probe and skip the managed-identity
        # (IMDS) probe, which otherwise hangs for ~seconds on non-Azure machines.
        logger.warning("AZURE_OPENAI_API_KEY not set — falling back to Entra auth "
                       "(slower; set the key in .env for instant, reliable responses).")
        try:
            credential = AzureCliCredential(process_timeout=8)
            token_provider = get_bearer_token_provider(credential, "https://ai.azure.com/.default")
            token_provider()
            logger.info("Using Azure CLI credential")
        except Exception:
            try:
                from azure.identity import DefaultAzureCredential
                credential = DefaultAzureCredential(
                    exclude_managed_identity_credential=True,
                    exclude_shared_token_cache_credential=True,
                    process_timeout=8,
                )
                token_provider = get_bearer_token_provider(credential, "https://ai.azure.com/.default")
                logger.info("Using DefaultAzureCredential")
            except Exception as e:
                raise ValueError(
                    f"No valid Azure credential found: {e}. "
                    f"Tip: set AZURE_OPENAI_API_KEY in .env to use key-based auth instead."
                )

        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider,
            api_version=api_version,
            api_key="placeholder",
            http_client=http_client,
        )
        logger.info(f"✅ Client created (Entra token, direct) → {deployment}")

    async def initialize(self) -> None:
        logger.info("Agent ready")

    # ──────────────────────────────────────────────────────────────────────
    # MAIN ENTRY POINT
    # ──────────────────────────────────────────────────────────────────────
    async def process_user_message(
        self,
        message: str,
        auth: Authorization,
        auth_handler_name: Optional[str],
        context: TurnContext,
    ) -> str:
        try:
            low = message.lower().strip()
            now = datetime.now()
            extra_context = ""

            # ── Debug ─────────────────────────────────────────────────────
            if message.strip() == "$DATA" or message.strip().startswith("$DATA "):
                parts = message.strip().split(None, 1)
                floor = parts[1].upper() if len(parts) > 1 else None
                # Show only next 6 hours to keep response small
                from_h = now.hour
                from_m = 30 if now.minute >= 30 else 0
                return print_full_day_text(
                    now.strftime("%Y-%m-%d"),
                    floor_filter=floor,
                    from_hour=from_h,
                    from_minute=from_m,
                    num_slots=12,  # 6 hours
                )

            if message.strip() == "$SPORTS" or message.strip().startswith("$SPORTS "):
                parts = message.strip().split(None, 1)
                sport = resolve_sport(parts[1]) if len(parts) > 1 else None
                return print_sports_day_text(
                    now.strftime("%Y-%m-%d"),
                    sport_filter=sport,
                    from_hour=now.hour,
                    num_slots=8,
                )

            confirm_words = {"yes", "yeah", "yep", "ok", "okay", "sure", "go ahead",
                             "confirm", "book it", "do it", "please", "yup", "correct",
                             "just book", "book that", "book any", "book one", "sounds good",
                             "go for it"}
            cancel_words  = {"no", "nope", "cancel", "don't", "dont", "nevermind",
                             "never mind", "stop", "skip", "abort", "not", "prefer",
                             "instead", "different", "another", "other", "change"}

            # ── Multi-booking plan confirm flow (chained sequence) ────────
            if self._pending_multi:
                # A preference tweak ("no, basketball at the sports hall") revises the
                # plan in place instead of discarding it.
                revised = self._revise_multi(low)
                if revised:
                    return revised
                if self._matches(low, confirm_words):
                    return self._book_multi()
                elif self._matches(low, cancel_words):
                    self._pending_multi = None
                    return "No problem — I haven't booked anything. Tell me what you'd like."

            # ── Edit-confirm flow (change an existing booking) ────────────
            if self._pending_edit:
                if self._matches(low, confirm_words):
                    return self._apply_edit()
                elif self._matches(low, cancel_words):
                    self._pending_edit = None
                    return "No change made. Your original booking is still active."

            # ── Sports booking confirm flow ───────────────────────────────
            if self.pending_sports:
                ps = self.pending_sports
                # User naming a different venue ("I said the multi-purpose room") →
                # switch to it at the same time instead of treating it as confirm/cancel.
                hint = match_venue_hint(ps["sport"], low)
                if hint and hint != ps["facility"]:
                    repick = self._repick_sport_venue(ps, hint)
                    if repick:
                        return repick
                if self._matches(low, confirm_words):
                    self.pending_sports = None
                    booking = book_facility(ps["date"], ps["facility"], ps["hour"], ps["sport"])
                    if booking["success"]:
                        self.my_bookings.append({
                            "ref": booking["booking_ref"],
                            "room_id": booking["name"],
                            "fac_id": ps["facility"],
                            "sport": ps["sport"],
                            "date": booking["date"],
                            "start": booking["start"],
                            "end": booking["end"],
                            "num_slots": 1,
                            "kind": "sports",
                        })
                    return format_sports_confirmation(booking)
                elif self._matches(low, cancel_words):
                    self.pending_sports = None
                    return "No problem — I haven't booked anything. Want a different time or sport?"

            # ── Sports time follow-up ─────────────────────────────────────
            if self._waiting_sports_time and not self.pending_sports:
                t = _parse_time_str(low)
                if t:
                    ctx = self._waiting_sports_time
                    self._waiting_sports_time = None
                    return self._do_sports_search(ctx["sport"], ctx["date"], t[0], t[1])

            # ── Cancel-confirm flow ───────────────────────────────────────
            if self._pending_cancel_ref:
                if self._matches(low, confirm_words):
                    from db import cancel_booking as db_cancel
                    ref = self._pending_cancel_ref
                    self._pending_cancel_ref = None
                    ok = db_cancel(ref)
                    if ok:
                        self.my_bookings = [b for b in self.my_bookings if b["ref"] != ref]
                        return f"✅ Cancelled. Booking {ref} has been removed."
                    else:
                        return f"Couldn't find booking {ref} in the system — it may have already been cancelled."
                elif self._matches(low, cancel_words):
                    self._pending_cancel_ref = None
                    return "Cancellation aborted. Your booking is still active."

            # ── Preference change while a booking is pending ───────────────
            # e.g. "I want one in LG1" / "I don't like 1F rooms" / "anything but LG4"
            if self.pending_booking:
                floor_pref = _parse_floor_pref(low)
                dislike = any(w in low for w in
                              ["dont", "don't", "not ", "no ", "dislike", "hate",
                               "avoid", "rather not", "anything but", "instead of",
                               "other than", "besides"])
                if floor_pref:
                    pb = self.pending_booking
                    pref = None if dislike else floor_pref
                    excl = floor_pref if dislike else None
                    new = self._reprice_room(pb, pref, excl)
                    if new:
                        return new
                    # No room matched the preference — tell them, keep old pending
                    kind = f"not on {floor_pref}" if dislike else f"on {floor_pref}"
                    return (f"Sorry, no rooms {kind} are free for that time. "
                            f"Your current option is still {pb['room_id']} "
                            f"({pb['hour']:02d}:{pb['minute']:02d}). Want me to book that, or try another floor?")

            # ── Booking confirmation ───────────────────────────────────────
            if self.pending_booking:
                if self._matches(low, confirm_words):
                    pb = self.pending_booking
                    self.pending_booking = None
                    booking = book_room(
                        date_str=pb["date"],
                        room_id=pb["room_id"],
                        hour=pb["hour"],
                        minute=pb["minute"],
                        num_slots=pb["num_slots"],
                    )
                    if booking["success"]:
                        self.my_bookings.append({
                            "ref": booking["booking_ref"],
                            "room_id": booking["room_id"],
                            "date": booking["date"],
                            "start": booking["start"],
                            "end": booking["end"],
                            "num_slots": pb["num_slots"],
                        })
                    return format_booking_confirmation(booking)
                elif self._matches(low, cancel_words):
                    # Save time window so we can reuse it if user asks for a different room
                    self._rejected_booking_ctx = {
                        "date":      self.pending_booking["date"],
                        "hour":      self.pending_booking["hour"],
                        "minute":    self.pending_booking["minute"],
                        "num_slots": self.pending_booking["num_slots"],
                        "capacity":  self.pending_booking.get("capacity", 1),
                    }
                    self.pending_booking = None
                    # Don't return — fall through so we can process the rest of the message
                    extra_context += "\n\n[NOTE] User declined the room suggestion. Don't say 'booking cancelled'. Address their actual request — they may want a different room type or floor."

            # ── Duration follow-up ────────────────────────────────────────
            if self._waiting_duration and not self.pending_booking:
                dur_reply = self._extract_duration_from_reply(low)
                if dur_reply:
                    ctx = self._duration_context
                    self._waiting_duration = False
                    self._duration_context = None
                    extra_context = self._do_booking_search(
                        ctx["date"], ctx["start_hour"], ctx["start_minute"],
                        dur_reply, ctx["capacity"], ctx.get("room_id")
                    )
                    return await self._call_model(message, extra_context)

            self._waiting_duration = False
            self._duration_context = None
            self._waiting_sports_time = None

            # ── Sport swap: "instead of table tennis, I'll play basketball" ──
            swap_reply = self._try_sport_swap(low, now)
            if swap_reply:
                return swap_reply

            # ── Cancel / edit / view-bookings intent ──────────────────────
            edit_verbs   = ["change", "edit", "reschedule", "move", "switch", "swap", "rebook", "replace"]
            booking_nouns = ["booking", "reservation", "room", "reserve", "study", "court"]
            is_edit_explicit = any(v in low for v in edit_verbs) and any(n in low for n in booking_nouns)

            # Preference-based edit of an EXISTING booking, e.g. "I prefer LG1 to study"
            # or "replace my study room" — route to the deterministic edit flow (not the
            # LLM, which only shows availability). Only fires when a matching booking exists.
            today_str = now.strftime("%Y-%m-%d")
            today_bk = [b for b in self.my_bookings if b["date"] == today_str]
            has_lib = any(b.get("kind") != "sports" for b in today_bk)
            wants_lib_change = bool(_parse_floor_pref(low)) or "study" in low or "library" in low
            named_sports_bk = any(b.get("kind") == "sports" and b.get("sport") in self._sports_in_text(low)
                                  for b in today_bk)
            pref_edit = (any(w in low for w in ["prefer", "rather", "replace", "instead"])
                         and ((wants_lib_change and has_lib) or named_sports_bk))
            # "change/move/reschedule my badminton to 5" — edit a named sport booking
            edits_named_sport = any(v in low for v in edit_verbs) and named_sports_bk

            is_edit = is_edit_explicit or pref_edit or edits_named_sport
            is_cancel = "cancel" in low and not is_edit
            is_my_bookings = (not is_edit and not is_cancel) and any(
                k in low for k in ["my booking", "my bookings", "show booking", "show my",
                                   "see my booking", "what did i book", "list booking", "view booking"])

            if is_my_bookings:
                return self._format_my_bookings()

            if is_edit:
                return self._start_edit(message, low, now)

            if is_cancel:
                today = now.strftime("%Y-%m-%d")
                today_bookings = [b for b in self.my_bookings if b["date"] == today]
                if not today_bookings:
                    return "You have no bookings to cancel today."

                # Did they name a specific booking? ("cancel the pingpong", a room id…)
                matched = [b for b in today_bookings if self._booking_matches(b, low)]
                target = matched[0] if len(matched) == 1 else None
                if target is None and len(today_bookings) == 1:
                    target = today_bookings[0]

                # Compound "cancel X and book/play Y" → swap to a different sport
                if target and ("book" in low or "play" in low):
                    tsport = target.get("sport")
                    swap_sport = next((k for k in self._sports_in_text(low)
                                       if k != tsport and k in SPORT_VENUES), None)
                    if swap_sport:
                        from db import cancel_booking as db_cancel
                        db_cancel(target["ref"])
                        self.my_bookings = [b for b in self.my_bookings if b["ref"] != target["ref"]]
                        hour = int(target["start"][:2])  # reuse the freed slot's time
                        search = self._do_sports_search(swap_sport, today, hour, 0)
                        return (f"✅ Cancelled {target['room_id']} "
                                f"({target['start']}–{target['end']}).\n\n{search}")

                if target:
                    self._pending_cancel_ref = target["ref"]
                    return (
                        f"Cancel {target['room_id']} on {target['date']} {target['start']}–{target['end']} "
                        f"(ref: {target['ref']})? Reply yes to confirm."
                    )

                lines = ["Which booking would you like to cancel? Reply with the room name or reference:"]
                for b in today_bookings:
                    lines.append(f"• {b['room_id']} {b['start']}–{b['end']}  (ref: {b['ref']})")
                return "\n".join(lines)

            # ── Combined planning: "I'm at X, want to study/play/book at HKUST" ──
            planning_triggers = [
                "im at", "i'm at", "i am at", "coming from", "leaving from",
                "on my way", "heading to", "going to school", "going to campus",
                "going to hkust", "going to ust", "going to uni", "to school",
                "when i get", "by the time i get", "arrive at school",
            ]
            arrival_phrase = any(p in low for p in ARRIVAL_PHRASES)
            sport_for_plan = resolve_sport(low)
            detected_origin = next((loc for loc in HK_ORIGINS if loc in low), None)

            # "book one when I arrive" — reuse the origin from an earlier planning turn
            if detected_origin is None and self._plan_ctx and arrival_phrase:
                detected_origin = self._plan_ctx.get("origin")

            is_planning = detected_origin is not None and (
                any(t in low for t in planning_triggers)
                or arrival_phrase
                or sport_for_plan is not None
                or any(k in low for k in ["study", "book", "room", "library", "play"])
            )

            if is_planning:
                sport = sport_for_plan
                # Inherit the remembered sport only if they aren't asking for a room/library
                if sport is None and not any(k in low for k in
                                             ["room", "study", "library", "desk"]):
                    sport = (self._plan_ctx or {}).get("sport")
                plan_reply = await self._handle_planning(low, now, detected_origin, sport, arrival_phrase)
                self.conversation_history.append({"role": "user", "content": message})
                self.conversation_history.append({"role": "assistant", "content": plan_reply[:300]})
                if len(self.conversation_history) > 8:
                    self.conversation_history = self.conversation_history[-8:]
                return plan_reply

            # ── Chained booking: "study at lg3 after pingpong / after badminton" ──
            # Fires deterministically (no LLM) whenever the user wants to study
            # "after/once/then/later" relative to an existing booking today.
            chain_words = ["after", "once", "then", "afterward", "afterwards",
                           "later", "when i finish", "when im done", "when i'm done",
                           "subsequently", "following"]
            wants_study = ("study" in low or "studdy" in low or "sstudy" in low
                           or "library" in low
                           or _parse_floor_pref(low) in ("LG1", "LG3", "LG4", "1/F", "LC"))
            today_str = now.strftime("%Y-%m-%d")
            has_today_booking = any(b["date"] == today_str for b in self.my_bookings)
            if (any(w in low for w in chain_words) and wants_study and has_today_booking):
                # If the chain also names a NEW sport (e.g. "play table tennis, then
                # study"), book the whole sequence; otherwise just the study.
                booked_sports = {b.get("sport") for b in self.my_bookings
                                 if b.get("kind") == "sports" and b["date"] == today_str}
                new_sports = [sk for sk in self._sports_in_text(low)
                              if sk not in booked_sports and sk in SPORT_VENUES]
                if new_sports:
                    seq = self._handle_chain_sequence(low, now, new_sports)
                    if seq:
                        return seq
                chained = self._handle_chained_study(low, now)
                if chained:
                    return chained

            # ── Sports facilities ─────────────────────────────────────────
            # A "study" request that merely mentions a sport (e.g. "after badminton,
            # study at lg3") is a LIBRARY booking — don't let the sport word grab it.
            sport = resolve_sport(low)
            if sport and "study" not in low and "library" not in low:
                return await self._handle_sports(message, low, now, sport)

            # ── Detect intent ─────────────────────────────────────────────
            booking_kws = {"book", "reserve", "booking", "reservation", "need a room",
                           "get a room", "find a room", "want a room", "find me a room"}
            bus_kws = {"bus", "eta", "next bus", "91m", "11m", "792", "route", "transport",
                       "ride", "get to", "going to", "leave campus", "depart",
                       "hang hau", "diamond hill", "po lam", "choi hung", "tseung kwan o",
                       "tko", "sai kung", "north gate", "south gate"}
            library_kws = {"study room", "learning commons", "lc-", "lg1", "lg3",
                           "lg4", "1f", "library", "available", "free room", "study space",
                           "quiet", "study", "reserve"}

            is_booking = any(k in low for k in booking_kws) or bool(re.search(r'\b(lc-\d+|1f-r\d+|lg[134]-r\d+)\b', low))
            is_bus     = any(k in low for k in bus_kws)
            is_library = any(k in low for k in library_kws)

            # ── Bus fetch ─────────────────────────────────────────────────
            if is_bus:
                dest_map = {
                    "diamond_hill": ["diamond hill", "kowloon", "choi hung mtr", "choi hung station"],
                    "po_lam":       ["po lam"],
                    "tko":          ["tseung kwan o", "tko", "popcorn", "tiu keng leng"],
                    "sai_kung":     ["sai kung", "marina cove"],
                    "hkust":        ["to hkust", "to ust", "to campus", "to school",
                                     "get to hkust", "going to hkust", "at hang hau",
                                     "from hang hau", "at po lam", "from po lam",
                                     "at diamond hill", "from diamond hill",
                                     "at choi hung", "from choi hung", "at tko", "from tko"],
                }
                matched = "all"
                for fk, kws in dest_map.items():
                    if any(k in low for k in kws):
                        matched = fk
                        break
                t0 = time.time()
                etas = await get_all_hkust_etas(matched)
                logger.info(f"Bus fetch {time.time()-t0:.2f}s filter={matched}")
                extra_context += f"\n\n[LIVE BUS DATA]\n{format_etas_for_agent(etas)}"

            # ── Booking intent ────────────────────────────────────────────
            if is_booking or (self._rejected_booking_ctx and any(k in low for k in {"room", "study", "lg", "lc", "1f", "book", "quiet", "large"})):
                intent    = _parse_booking_intent(message, now)
                req_date  = intent["date"]
                req_h     = intent["start_hour"]
                req_m     = intent["start_minute"]
                num_slots = intent["num_slots"]
                capacity  = intent["capacity"]
                room_id   = intent.get("room_id")

                # Reuse rejected booking's time/duration when user just expresses preference
                rctx = self._rejected_booking_ctx
                if rctx:
                    self._rejected_booking_ctx = None
                    if num_slots == 0 or (req_h == now.hour and req_m == (30 if now.minute >= 30 else 0) and not room_id):
                        # User didn't give new times — reuse previous
                        req_date  = rctx["date"]
                        req_h     = rctx["hour"]
                        req_m     = rctx["minute"]
                        num_slots = rctx["num_slots"]
                        capacity  = rctx["capacity"]

                if num_slots == 0:
                    self._waiting_duration = True
                    self._duration_context = {
                        "date": req_date, "start_hour": req_h,
                        "start_minute": req_m, "capacity": capacity,
                        "room_id": room_id,
                    }
                    extra_context += "\n\n[BOOKING DURATION NEEDED]\nAsk: 'How long do you need the room?'"
                else:
                    # Check daily LIBRARY slot limit (sports bookings don't count —
                    # those are limited separately: 1 slot per sport per day).
                    today_slots = sum(
                        b["num_slots"] for b in self.my_bookings
                        if b["date"] == req_date and b.get("kind") != "sports"
                    )
                    slots_remaining = self.MAX_SLOTS_PER_DAY - today_slots
                    if slots_remaining <= 0:
                        extra_context += (
                            f"\n\n[SLOT LIMIT REACHED]\n"
                            f"User has used all {self.MAX_SLOTS_PER_DAY} slots (2 hours) for {req_date}. "
                            f"They cannot make another booking today."
                        )
                    elif num_slots > slots_remaining:
                        # Trim to what's left
                        num_slots = slots_remaining
                        extra_context += (
                            f"\n\n[SLOT LIMIT PARTIAL]\n"
                            f"User only has {slots_remaining} slot(s) ({slots_remaining*30} min) remaining today. "
                            f"Booking trimmed accordingly.\n"
                        )
                        extra_context += self._do_booking_search(req_date, req_h, req_m, num_slots, capacity, room_id)
                    else:
                        extra_context += self._do_booking_search(req_date, req_h, req_m, num_slots, capacity, room_id)

            # ── Library availability (non-booking) ────────────────────────
            elif is_library:
                date_str = now.strftime("%Y-%m-%d")
                cur_h    = now.hour
                cur_m    = 30 if now.minute >= 30 else 0

                until_m = re.search(r'(?:until|till)\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', low)
                from_m  = re.search(r'(?:from|at|@)\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', low)
                bare_m  = re.search(r'\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b', low)

                if until_m:
                    until_h = _parse_hour(int(until_m.group(1)), until_m.group(3))
                    result  = find_rooms_free_until(date_str, cur_h, cur_m, until_h)
                    extra_context += f"\n\n[LIVE LIBRARY DATA]\n{format_continuous_for_agent(result)}"
                elif from_m or bare_m:
                    match  = from_m or bare_m
                    h      = _parse_hour(int(match.group(1)), match.group(3))
                    m2_raw = int(match.group(2)) if match.group(2) else 0
                    m2     = 30 if m2_raw >= 30 else 0
                    slots  = get_slots_range(date_str, h, m2, 4)
                    slot   = get_slot_availability(date_str, h, m2)
                    extra_context += f"\n\n[LIVE LIBRARY DATA from {h:02d}:{m2:02d}]\n{format_range_for_agent(slots)}\n{format_slot_for_agent(slot)}"
                else:
                    slots = get_slots_range(date_str, cur_h, cur_m, 4)
                    slot  = get_slot_availability(date_str, cur_h, cur_m)
                    extra_context += (
                        f"\n\n[LIVE LIBRARY DATA - now {now.hour:02d}:{now.minute:02d}]\n"
                        f"{format_range_for_agent(slots)}\n{format_slot_for_agent(slot)}"
                    )

            return await self._call_model(message, extra_context)

        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            return f"Sorry, I encountered an error: {str(e)}"

    # ──────────────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────────────

    def _format_my_bookings(self) -> str:
        if not self.my_bookings:
            return "You haven't made any bookings in this session."
        lines = ["**Your bookings this session:**", "",
                 "| Room | Start | End | Reference |",
                 "|------|-------|-----|-----------|"]
        for b in self.my_bookings:
            if b.get("kind") == "sports":
                # 1-hour slot, stored start/end directly
                lines.append(f"| {b['room_id']} | {b['start']} | {b['end']} | {b['ref']} |")
                continue
            h, m = int(b["start"][:2]), int(b["start"][3:])
            for _ in range(b["num_slots"]):
                end_m, end_h = m + 30, h
                if end_m >= 60:
                    end_m -= 60
                    end_h += 1
                lines.append(f"| {b['room_id']} | {h:02d}:{m:02d} | {end_h:02d}:{end_m:02d} | {b['ref']} |")
                m += 30
                if m >= 60:
                    m -= 60
                    h += 1
        return "\n".join(lines)

    def _start_edit(self, message: str, low: str, now: datetime) -> str:
        """
        Begin changing an existing booking. Deterministically finds a replacement room
        (new floor and/or new time, defaulting to the same time) and sets _pending_edit.
        """
        today = now.strftime("%Y-%m-%d")
        today_bookings = [b for b in self.my_bookings if b["date"] == today]
        if not today_bookings:
            return "You don't have any bookings to change today."

        # Pick which booking to change:
        #  1) an explicitly named existing room id,
        #  2) a library booking if they said "study"/"library"/a floor,
        #  3) the sport they named,
        #  4) otherwise the most recent booking.
        target = None
        for b in today_bookings:
            if b["room_id"].lower() in low:
                target = b
                break
        if target is None and ("study" in low or "library" in low
                               or "study room" in low or _parse_floor_pref(low)):
            target = next((b for b in reversed(today_bookings)
                           if b.get("kind") != "sports"), None)
        if target is None:
            named = self._sports_in_text(low)
            target = next((b for b in reversed(today_bookings)
                           if b.get("kind") == "sports" and b.get("sport") in named), None)
        if target is None:
            target = today_bookings[-1]

        # Sports bookings: edit in place (new time and/or venue), atomically.
        if target.get("kind") == "sports":
            return self._start_sport_edit(target, low)

        old_h, old_m = int(target["start"][:2]), int(target["start"][3:])
        old_slots = target["num_slots"]

        # New floor preference (the floor they want to move TO).
        new_floor = _parse_floor_pref(low)
        # Ignore a floor mention that just refers to the booking's current floor
        cur_floor = (LC_ROOMS.get(target["room_id"]) or STUDY_ROOMS.get(target["room_id"]))["floor"].upper()
        if new_floor and new_floor != "LC" and new_floor == cur_floor and target["room_id"].lower() not in low:
            new_floor = new_floor  # still allow (user may want a different room same floor)

        # New time? Only if the message explicitly states one (and not "same time").
        if "same time" in low or "same slot" in low:
            new_h, new_m, new_slots = old_h, old_m, old_slots
        elif self._message_has_time(low):
            intent = _parse_booking_intent(message, now)
            new_h, new_m = intent["start_hour"], intent["start_minute"]
            new_slots = intent["num_slots"] or old_slots
        else:
            new_h, new_m, new_slots = old_h, old_m, old_slots

        result = find_best_room(today, new_h, new_m, new_slots, 1, preferred_floor=new_floor)
        # Don't suggest the exact same room they already have
        if result.get("found") and result["room_id"] == target["room_id"] and result.get("alternatives"):
            alt_id, alt_info = result["alternatives"][0]
            result = {"found": True, "room_id": alt_id,
                      "capacity": alt_info["capacity"], "floor": alt_info["floor"],
                      "room_type": alt_info["type"], "alternatives": []}

        if not result.get("found"):
            where = f"on {new_floor}" if new_floor else "free"
            return (f"Sorry, no rooms {where} are available at "
                    f"{new_h:02d}:{new_m:02d} for {new_slots*30} min. "
                    f"Your booking {target['room_id']} is unchanged.")

        new_room = result["room_id"]
        ninfo = LC_ROOMS.get(new_room) or STUDY_ROOMS.get(new_room)
        end_h = new_h + (new_m + new_slots * 30) // 60
        end_m = (new_m + new_slots * 30) % 60
        self._pending_edit = {
            "old_ref": target["ref"],
            "old_room": target["room_id"],
            "room_id": new_room,
            "date": today,
            "hour": new_h,
            "minute": new_m,
            "num_slots": new_slots,
        }
        return (
            f"Change **{target['room_id']}** → **{new_room}** "
            f"({ninfo['type']}, {ninfo['floor']}, seats {ninfo['capacity']}) "
            f"for {new_h:02d}:{new_m:02d}–{end_h:02d}:{end_m:02d}?\n\nReply yes to confirm."
        )

    def _start_sport_edit(self, target: dict, low: str) -> str:
        """Edit a sports booking in place: new time and/or venue. Sets _pending_edit."""
        date_str = target["date"]
        sport = target.get("sport")
        label = SPORT_LABELS.get(sport, (sport or "").title())
        old_h = int(target["start"][:2])
        cur_fac = target.get("fac_id")
        if not sport or cur_fac is None:
            return (f"I can't tell which sport {target['room_id']} is — "
                    f"please cancel it and rebook.")

        # New time?
        if self._message_has_time(low):
            t = _parse_time_str(low)
            new_h = (t[0] + (1 if t[1] >= 30 else 0)) if t else old_h
        else:
            new_h = old_h

        # New venue? An explicit venue is honored strictly; otherwise keep the current one.
        hint = match_venue_hint(sport, low)
        if hint:
            courts = next((v["courts"] for v in SPORT_VENUES.get(sport, []) if v["id"] == hint), None)
            if not courts or free_courts(date_str, hint, sport, courts, new_h) <= 0:
                return (f"{FACILITIES[hint]['name']} has no {label} court free at {new_h:02d}:00 — "
                        f"your {target['room_id']} booking ({target['start']}–{target['end']}) is unchanged.")
            new_fac = hint
        else:
            res = find_available_venue(date_str, sport, new_h, preferred_facility=cur_fac)
            if res["status"] != "ok":
                return (f"No {label} court is free at {new_h:02d}:00 — your "
                        f"{target['room_id']} booking ({target['start']}–{target['end']}) is unchanged.")
            new_fac = res["facility"]

        if new_fac == cur_fac and new_h == old_h:
            return (f"That's the same as your current booking — {target['room_id']} "
                    f"{target['start']}–{target['end']}.")

        self._pending_edit = {
            "kind": "sports", "old_ref": target["ref"], "old_room": target["room_id"],
            "sport": sport, "facility": new_fac, "date": date_str, "hour": new_h,
        }
        new_slot = f"{new_h:02d}:00–{new_h + 1:02d}:00"
        if new_fac == cur_fac:
            return (f"Move your {label} booking from {target['start']}–{target['end']} to "
                    f"**{new_slot}** at {FACILITIES[new_fac]['name']}? Reply yes to confirm.")
        return (f"Change **{target['room_id']}** ({target['start']}–{target['end']}) → "
                f"**{FACILITIES[new_fac]['name']}** ({label}) at {new_slot}? Reply yes to confirm.")

    def _apply_edit(self) -> str:
        """Book the new room first, then cancel the old one. Deterministic — no LLM."""
        from db import cancel_booking as db_cancel
        ed = self._pending_edit
        self._pending_edit = None

        # ── Sports edit ───────────────────────────────────────────────────────
        if ed.get("kind") == "sports":
            booking = book_facility(ed["date"], ed["facility"], ed["hour"], ed["sport"])
            if not booking["success"]:
                return (f"Couldn't switch: {booking['error']} "
                        f"Your original booking {ed['old_room']} is still active.")
            db_cancel(ed["old_ref"])
            self.my_bookings = [b for b in self.my_bookings if b["ref"] != ed["old_ref"]]
            self.my_bookings.append({
                "ref": booking["booking_ref"], "room_id": booking["name"],
                "fac_id": ed["facility"], "sport": ed["sport"], "date": booking["date"],
                "start": booking["start"], "end": booking["end"],
                "num_slots": 1, "kind": "sports",
            })
            return (f"✅ Changed! {ed['old_room']} → **{booking['name']}** "
                    f"on {booking['date']} from {booking['start']} to {booking['end']}.\n"
                    f"New reference: {booking['booking_ref']}")

        booking = book_room(
            date_str=ed["date"], room_id=ed["room_id"],
            hour=ed["hour"], minute=ed["minute"], num_slots=ed["num_slots"],
        )
        if not booking["success"]:
            return (f"Couldn't switch to {ed['room_id']}: {booking['error']} "
                    f"Your original booking {ed['old_room']} is still active.")

        # New booking succeeded — release the old one
        db_cancel(ed["old_ref"])
        self.my_bookings = [b for b in self.my_bookings if b["ref"] != ed["old_ref"]]
        self.my_bookings.append({
            "ref": booking["booking_ref"],
            "room_id": booking["room_id"],
            "date": booking["date"],
            "start": booking["start"],
            "end": booking["end"],
            "num_slots": ed["num_slots"],
        })
        return (
            f"✅ Changed! {ed['old_room']} → **{booking['room_id']}** "
            f"({booking['floor']}, seats {booking['capacity']}) on {booking['date']} "
            f"from {booking['start']} to {booking['end']}.\n"
            f"New reference: {booking['booking_ref']}"
        )

    @staticmethod
    def _message_has_time(low: str) -> bool:
        """True if the message explicitly mentions a clock time or time range."""
        return bool(
            re.search(r'\b\d{1,2}:\d{2}\b', low)
            or re.search(r'\b\d{1,2}\s*(am|pm)\b', low)
            or re.search(r'\b\d{1,2}\s*(?:-|–|to)\s*\d{1,2}\b', low)
            or re.search(r'(?:from|at|until|till)\s+\d', low)
        )

    @staticmethod
    def _sports_in_text(low: str) -> set:
        """
        All canonical sport keys named in the text. Longest alias wins per span — e.g.
        "table tennis" registers table_tennis only, not also tennis (its substring).
        """
        found = set()
        work = low
        for alias in sorted(SPORT_ALIASES, key=len, reverse=True):
            if alias in work:
                found.add(SPORT_ALIASES[alias])
                work = work.replace(alias, " " * len(alias))  # mask so substrings don't re-match
        return found

    @staticmethod
    def _booking_matches(b: dict, low: str) -> bool:
        """True if the message names this booking (by room id, or by its sport)."""
        if b["room_id"].lower() in low:
            return True
        bsport = b.get("sport")
        if bsport and any(a in low for a, k in SPORT_ALIASES.items() if k == bsport):
            return True
        return False

    @staticmethod
    def _matches(low: str, words: set) -> bool:
        """
        Word-aware membership test. Single-word entries must match a whole token
        (so "no" no longer matches "north"); multi-word entries match as substrings.
        """
        tokens = set(re.findall(r"[a-z0-9']+", low))
        for w in words:
            if " " in w:
                if w in low:
                    return True
            elif w in tokens:
                return True
        return False

    def _reprice_room(self, pb: dict, preferred_floor: Optional[str],
                      excluded_floor: Optional[str]) -> Optional[str]:
        """
        Re-search for a room at the same time window as the pending booking, applying a
        floor preference/exclusion. Updates pending_booking and returns a reply, or None
        if nothing matched.
        """
        result = find_best_room(
            pb["date"], pb["hour"], pb["minute"], pb["num_slots"],
            pb.get("capacity", 1),
            preferred_floor=preferred_floor,
            excluded_floor=excluded_floor,
        )
        if not result["found"]:
            return None

        room_id = result["room_id"]
        room_info = LC_ROOMS.get(room_id) or STUDY_ROOMS.get(room_id)
        end_h = pb["hour"] + (pb["minute"] + pb["num_slots"] * 30) // 60
        end_m = (pb["minute"] + pb["num_slots"] * 30) % 60
        self.pending_booking = {
            "room_id": room_id,
            "date": pb["date"],
            "hour": pb["hour"],
            "minute": pb["minute"],
            "num_slots": pb["num_slots"],
            "capacity": result["capacity"],
        }
        return (
            f"📚 **{room_id}** ({room_info['type']}, {room_info['floor']}, "
            f"seats {room_info['capacity']}) is free "
            f"{pb['hour']:02d}:{pb['minute']:02d}–{end_h:02d}:{end_m:02d}.\n\n"
            f"Shall I book {room_id}?"
        )

    async def _handle_planning(self, low: str, now: datetime, origin: str,
                               sport: Optional[str] = None,
                               arrival_phrase: bool = False) -> str:
        """
        Combined bus + (sports OR library) planning. Returns a direct reply (no LLM).
        1. Estimate arrival at HKUST from the origin (door-to-door + live next-bus).
        2. If a sport is in play → recommend & hold a sports court at arrival.
           Otherwise → recommend & hold a library room at arrival.
        Remembers the origin/sport in _plan_ctx so "book one when I arrive" works.
        """
        info = HK_ORIGINS[origin]
        route = info["route"]
        travel_min = info["travel_min"]
        gate = info.get("gate", "")
        via = info.get("via")  # interchange to transfer at (None = you're already there)

        date_str = now.strftime("%Y-%m-%d")
        if any(w in low for w in ["tomorrow", "tmr", "tmrw"]):
            date_str = (now + dt.timedelta(days=1)).strftime("%Y-%m-%d")

        # Live next-bus ONLY makes sense if you're at the interchange (near origin).
        # For far origins we don't invent a precise ETA — you haven't reached the stop.
        wait_min = None
        if via is None:
            try:
                etas = await get_all_hkust_etas("all")
                for entry in etas:
                    if entry.get("label") == route and entry.get("etas"):
                        wait_min = entry["etas"][0]["minutes"]
                        break
            except Exception:
                wait_min = None

        arrival = now + dt.timedelta(minutes=travel_min)

        # Remember context for follow-ups like "book one when I arrive"
        self._plan_ctx = {"origin": origin, "sport": sport, "date": date_str}
        # Planning takes over — clear any half-started library/sports prompts
        self._waiting_duration = False
        self._duration_context = None
        self._waiting_sports_time = None

        if via:
            route_text = f"MTR to **{via}**, then **{route}** to HKUST {gate}"
            eta_str = ""  # no live ETA for a stop you haven't reached yet
        else:
            route_text = f"take **{route}** to HKUST {gate}"
            eta_str = f" Next **{route}** in ~{wait_min} min." if wait_min is not None else ""
        bus_line = (
            f"🚌 From **{origin.title()}**: {route_text} "
            f"(~{travel_min} min door-to-door).{eta_str}"
        )

        # If the user gave an explicit clock time (and isn't saying "when I arrive"),
        # honour that instead of the computed arrival.
        explicit = _parse_time_str(low) if (self._message_has_time(low) and not arrival_phrase) else None

        # ── SPORTS planning ───────────────────────────────────────────────
        if sport:
            label = SPORT_LABELS.get(sport, sport.title())

            if sport in SPORT_INFO:
                fac = FACILITIES[SPORT_INFO[sport]]
                return (f"{bus_line} ETA campus ~{arrival.hour:02d}:{arrival.minute:02d}.\n\n"
                        f"🏃 **{label}** is at **{fac['name']}** ({fac['location']}) — "
                        f"{fac['note']}. No booking needed, just turn up when you arrive.")

            existing = self._sport_booked_today(sport, date_str)
            if existing:
                return (f"{bus_line}\n\nYou already have a {label} booking today "
                        f"({existing['room_id']} {existing['start']}–{existing['end']}). "
                        f"It's one slot per sport per day — cancel that first to rebook {label}.")

            if explicit:
                sp_h = explicit[0] + (1 if explicit[1] >= 30 else 0)
            else:
                # Round arrival UP to the next bookable 1-hour slot
                sp_h = arrival.hour + (1 if arrival.minute > 0 else 0)

            pref_fac = match_venue_hint(sport, low)
            result = find_available_venue(date_str, sport, sp_h,
                                          preferred_facility=pref_fac)

            arr_note = (f" ETA campus ~{arrival.hour:02d}:{arrival.minute:02d}"
                        f" → first slot {sp_h:02d}:00." if not explicit else "")
            # If the user named a venue we couldn't honor, say so (don't substitute silently).
            pref_note = ""
            if pref_fac and result.get("status") == "ok" and result["facility"] != pref_fac:
                pref_note = (f"\n\n_Heads up: {FACILITIES[pref_fac]['name']} is full at "
                             f"{sp_h:02d}:00, so here's the next-best option._")
            if result["status"] == "ok":
                self.pending_sports = {
                    "sport": sport, "facility": result["facility"],
                    "date": date_str, "hour": result["hour"],
                }
                rec = format_venue_recommendation(result, date_str, sport)
                return f"{bus_line}{arr_note}{pref_note}\n\n{rec}\n\nShall I book it?"
            if result["status"] == "full" and result.get("alt_times"):
                alt = result["alt_times"][0]
                fac = FACILITIES[alt["facility"]]
                self.pending_sports = {
                    "sport": sport, "facility": alt["facility"],
                    "date": date_str, "hour": alt["hour"],
                }
                return (f"{bus_line}{arr_note}\n\n"
                        f"{label} is full at {sp_h:02d}:00, but **{alt['hour']:02d}:00–"
                        f"{alt['hour'] + 1:02d}:00** is free at **{fac['name']}**. "
                        f"Shall I book that?")
            self.pending_sports = None
            return (f"{bus_line}{arr_note}\n\n"
                    f"{format_venue_recommendation(result, date_str, sport)}")

        # ── LIBRARY planning (default) ────────────────────────────────────
        if explicit:
            arr_h, arr_m = _round_up_slot(explicit[0], explicit[1])
        else:
            arr_h, arr_m = _round_up_slot(arrival.hour, arrival.minute)

        dur_h = re.search(r'(\d+)\s*hour', low)
        dur_m_match = re.search(r'(\d+)\s*(?:min|mins|minutes)', low)
        if dur_h:
            num_slots = min(4, int(dur_h.group(1)) * 2)
        elif dur_m_match:
            num_slots = max(1, min(4, round(int(dur_m_match.group(1)) / 30)))
        else:
            num_slots = 4  # default 2 hours

        end_h = arr_h + (arr_m + num_slots * 30) // 60
        end_m = (arr_m + num_slots * 30) % 60

        preferred_floor = _parse_floor_pref(low)
        result = find_best_room(date_str, arr_h, arr_m, num_slots, 1, preferred_floor)

        arr_note = (f" ETA campus ~{arrival.hour:02d}:{arrival.minute:02d}"
                    f" → first slot {arr_h:02d}:{arr_m:02d}." if not explicit else "")

        if result["found"]:
            room_id = result["room_id"]
            room_info = LC_ROOMS.get(room_id) or STUDY_ROOMS.get(room_id)
            self.pending_booking = {
                "room_id": room_id,
                "date": date_str,
                "hour": arr_h,
                "minute": arr_m,
                "num_slots": num_slots,
                "capacity": result["capacity"],
            }
            return (
                f"{bus_line}{arr_note}\n\n"
                f"📚 **{room_id}** ({room_info['type']}, {room_info['floor']}, "
                f"seats {room_info['capacity']}) is free {arr_h:02d}:{arr_m:02d}–{end_h:02d}:{end_m:02d}.\n\n"
                f"Shall I book {room_id}?"
            )
        else:
            alt_str = ""
            if result.get("alt_times"):
                alts = [f"{t} ({r})" for t, r, _ in result["alt_times"][:2]]
                alt_str = f"\nAlternative slots: {', '.join(alts)}."
            return (
                f"{bus_line}{arr_note}\n\n"
                f"No rooms available at {arr_h:02d}:{arr_m:02d} for {num_slots*30} min.{alt_str}\n"
                f"Would you like to try a different time?"
            )

    # ──────────────────────────────────────────────────────────────────────
    # SPORTS
    # ──────────────────────────────────────────────────────────────────────
    async def _handle_sports(self, message: str, low: str, now: datetime, sport: str) -> str:
        """Route a sports request: list venues, give walk-in info, or recommend a slot."""
        date_str = now.strftime("%Y-%m-%d")
        if any(w in low for w in ["tomorrow", "tmr", "tmrw"]):
            date_str = (now + dt.timedelta(days=1)).strftime("%Y-%m-%d")
        label = SPORT_LABELS.get(sport, sport.title())

        # "where can I play X" / "which courts" → just list venues
        if any(w in low for w in ["where can", "which venue", "what venue", "which court",
                                  "where do", "list venue", "venues for", "facilities for",
                                  "where is", "where are"]):
            return list_venues_for_sport(sport)

        # Walk-in / contact-only activities — no booking, just info
        if sport in SPORT_INFO:
            res = find_available_venue(date_str, sport, 12)
            return format_venue_recommendation(res, date_str, sport)

        # Need an explicit time for a 1-hour slot — offer the free slots to pick from
        if (not self._message_has_time(low) and "now" not in low) or not _parse_time_str(low):
            self._waiting_sports_time = {"sport": sport, "date": date_str}
            opts = self._sport_time_options(sport, date_str, now)
            if opts:
                return (f"What time would you like to play {label}? Free 1-hour slots today:\n"
                        + "\n".join(f"• {o}" for o in opts)
                        + "\n\nJust tell me a time.")
            return f"What time would you like to play {label}? Courts run in 1-hour slots."

        t = _parse_time_str(low)
        return self._do_sports_search(sport, date_str, t[0], t[1], match_venue_hint(sport, low))

    def _sport_time_options(self, sport: str, date_str: str, now: datetime, limit: int = 5) -> list:
        """List the next few free 1-hour slots for a sport, with the least-busy venue."""
        out = []
        for h in range(max(now.hour, 7), 23):
            res = find_available_venue(date_str, sport, h)
            if res.get("status") == "ok":
                out.append(f"{h:02d}:00–{h + 1:02d}:00 — {res['facility_info']['name']} "
                           f"({res['free']} free)")
                if len(out) >= limit:
                    break
        return out

    def _try_sport_swap(self, low: str, now: datetime) -> Optional[str]:
        """
        "Instead of table tennis, I'll play basketball" → cancel the table-tennis
        booking and offer basketball at the SAME time. Returns a reply, or None to
        fall through. Verifies the new sport is free before cancelling anything.
        """
        if not any(w in low for w in ["instead", "rather than", "replace", "swap", "switch", "change"]):
            return None
        date_str = now.strftime("%Y-%m-%d")
        today = [b for b in self.my_bookings if b["date"] == date_str]
        if not today:
            return None
        named = self._sports_in_text(low)
        if not named:
            return None
        booked_sports = {b.get("sport") for b in today if b.get("kind") == "sports"}
        target = next((b for b in today if b.get("kind") == "sports"
                       and b.get("sport") in named), None)
        new_sport = next((s for s in named if s not in booked_sports and s in SPORT_VENUES), None)
        if not target or not new_sport:
            return None

        hour = int(target["start"][:2])
        pref = match_venue_hint(new_sport, low)
        label = SPORT_LABELS.get(new_sport, new_sport.title())
        # Verify availability BEFORE cancelling, so a swap never loses both bookings.
        if find_available_venue(date_str, new_sport, hour, preferred_facility=pref)["status"] != "ok":
            return (f"{label} isn't free at {hour:02d}:00, so I kept your "
                    f"{target['room_id']} booking ({target['start']}–{target['end']}) as is.")

        from db import cancel_booking as db_cancel
        db_cancel(target["ref"])
        self.my_bookings = [x for x in self.my_bookings if x["ref"] != target["ref"]]
        search = self._do_sports_search(new_sport, date_str, hour, 0, pref)
        return (f"✅ Cancelled {target['room_id']} ({target['start']}–{target['end']}).\n\n{search}")

    def _sport_booked_today(self, sport: str, date_str: str) -> Optional[dict]:
        """Return the user's existing booking for this sport today, if any (1/sport/day)."""
        return next((b for b in self.my_bookings
                     if b.get("kind") == "sports" and b["date"] == date_str
                     and b.get("sport") == sport), None)

    def _repick_sport_venue(self, ps: dict, hint: str) -> Optional[str]:
        """Switch a pending sports booking to a specific venue the user named, same time."""
        courts = next((v["courts"] for v in SPORT_VENUES.get(ps["sport"], [])
                       if v["id"] == hint), None)
        if courts is None:
            return None
        fac = FACILITIES[hint]
        h = ps["hour"]
        fc = free_courts(ps["date"], hint, ps["sport"], courts, h)
        if fc <= 0:
            cur = FACILITIES[ps["facility"]]["name"]
            return (f"{fac['name']} has no free courts at {h:02d}:00. "
                    f"I've still got {cur} held for you — book that, or try another time?")
        self.pending_sports = {**ps, "facility": hint}
        note = f" · {fac['note']}" if fac["note"] else ""
        return (f"Got it — **{fac['name']}** ({fac['location']}){note}: "
                f"{fc} free at {h:02d}:00–{h + 1:02d}:00. Shall I book it?")

    def _do_sports_search(self, sport: str, date_str: str, hour: int, minute: int,
                          preferred_facility: Optional[str] = None) -> str:
        """Find a venue for a sport at a time, set pending_sports if bookable."""
        if minute >= 30:
            hour += 1  # 1-hour slots — round up to the next hour
        label = SPORT_LABELS.get(sport, sport.title())

        # 1 booking per SPORT per day (all venues of a sport share the quota, e.g.
        # both soccer pitches count as one football booking).
        existing = self._sport_booked_today(sport, date_str)
        if existing:
            self.pending_sports = None
            return (f"You already have a {label} booking today — {existing['room_id']} "
                    f"{existing['start']}–{existing['end']}. It's one slot per sport per day, "
                    f"so cancel that first if you'd like a different {label} slot.")

        result = find_available_venue(date_str, sport, hour,
                                      preferred_facility=preferred_facility)
        if result["status"] == "ok":
            self.pending_sports = {
                "sport": sport,
                "facility": result["facility"],
                "date": date_str,
                "hour": result["hour"],
            }
            rec = format_venue_recommendation(result, date_str, sport)
            return f"{rec}\n\nShall I book it?"

        # Requested time is full — offer the nearest free slot AS A PENDING booking,
        # so a plain "yes" books it (instead of falling through to the LLM).
        if result["status"] == "full" and result.get("alt_times"):
            alt = result["alt_times"][0]
            fac = FACILITIES[alt["facility"]]
            self.pending_sports = {
                "sport": sport,
                "facility": alt["facility"],
                "date": date_str,
                "hour": alt["hour"],
            }
            others = result["alt_times"][1:3]
            other_str = ""
            if others:
                other_str = (" Other options: "
                             + ", ".join(f"{a['hour']:02d}:00" for a in others) + ".")
            return (
                f"{label} is fully booked at {hour:02d}:00. The nearest free slot is "
                f"**{alt['hour']:02d}:00–{alt['hour'] + 1:02d}:00** at **{fac['name']}** "
                f"({fac['location']}).{other_str}\n\nShall I book that instead?"
            )

        self.pending_sports = None
        return format_venue_recommendation(result, date_str, sport)

    def _parse_study_duration(self, low: str) -> int:
        """Library duration → number of 30-min slots. Handles 'an hour', '90 min', etc."""
        m = re.search(r'(\d+)\s*hour', low)
        if m:
            return max(1, min(self.MAX_SLOTS_PER_DAY, int(m.group(1)) * 2))
        mm = re.search(r'(\d+)\s*(?:min|mins|minutes)', low)
        if mm:
            return max(1, min(self.MAX_SLOTS_PER_DAY, round(int(mm.group(1)) / 30)))
        if "half an hour" in low or "half hour" in low:
            return 1
        if re.search(r'\b(an|a|one)\s+hour\b', low):
            return 2  # 1 hour
        if re.search(r'\b(two|2)\s+hours?\b', low):
            return 4
        return 4  # default 2 hours

    def _handle_chain_sequence(self, low: str, now: datetime, new_sports: list) -> Optional[str]:
        """
        Book a whole sequence in one go, e.g. "after badminton, play table tennis,
        then study 1h at lg3" → table tennis right after badminton, then LG3 study
        right after table tennis. Deterministic; sets _pending_multi for confirmation.
        """
        date_str = now.strftime("%Y-%m-%d")
        today = [b for b in self.my_bookings if b["date"] == date_str]
        if not today:
            return None

        anchor = max(today, key=lambda b: b["end"])
        cur_h = int(anchor["end"][:2])
        cur_m = 0 if int(anchor["end"][3:]) < 30 else 30

        # Order the new activities by where they appear in the sentence.
        items = []  # (position, kind, payload)
        for sk in new_sports:
            pos = min((low.find(a) for a, k in SPORT_ALIASES.items()
                       if k == sk and a in low), default=10 ** 9)
            items.append((pos, "sport", sk))
        if "study" in low or "studdy" in low or "sstudy" in low or "library" in low \
                or _parse_floor_pref(low):
            spos = low.find("study")
            if spos < 0:
                spos = low.find("library")
            if spos < 0:
                fp = (_parse_floor_pref(low) or "").lower()
                spos = low.find(fp) if fp else 10 ** 9
            items.append((spos, "study", _parse_floor_pref(low)))
        items.sort(key=lambda x: x[0])

        anchor_fac = anchor.get("fac_id")
        plan = []
        notes = []
        lib_used = sum(b["num_slots"] for b in today if b.get("kind") != "sports")
        plan_sports = set()
        for _pos, kind, payload in items:
            if cur_h >= 23:
                break
            if kind == "sport":
                if payload in plan_sports:
                    continue  # one slot per sport per day
                label = SPORT_LABELS.get(payload, payload.title())
                pref = self._sport_venue_pref(payload, low, anchor_fac)
                if pref:
                    # Explicit venue choice → honor it strictly; never substitute.
                    courts = next((v["courts"] for v in SPORT_VENUES.get(payload, [])
                                   if v["id"] == pref), None)
                    chosen_h = next((h for h in range(cur_h, min(cur_h + 5, 23))
                                     if courts and free_courts(date_str, pref, payload, courts, h) > 0), None)
                    if chosen_h is None:
                        notes.append(f"⚠️ {FACILITIES[pref]['name']} has no {label} court free "
                                     f"around {cur_h:02d}:00, so I left {label} out — "
                                     f"book it separately or pick another venue.")
                        continue
                    item = {"kind": "sports", "sport": payload, "facility": pref,
                            "label": FACILITIES[pref]["name"], "date": date_str,
                            "hour": chosen_h, "pref_facility": pref}
                else:
                    chosen_h = next((h for h in range(cur_h, min(cur_h + 5, 23))
                                     if find_available_venue(date_str, payload, h)["status"] == "ok"), None)
                    if chosen_h is None:
                        continue
                    res = find_available_venue(date_str, payload, chosen_h)
                    item = {"kind": "sports", "sport": payload, "facility": res["facility"],
                            "label": FACILITIES[res["facility"]]["name"], "date": date_str,
                            "hour": chosen_h}
                plan.append(item)
                plan_sports.add(payload)
                cur_h, cur_m = chosen_h + 1, 0
            else:  # study
                remaining = self.MAX_SLOTS_PER_DAY - lib_used
                if remaining <= 0:
                    continue
                n = min(self._parse_study_duration(low), remaining)
                res = find_best_room(date_str, cur_h, cur_m, n, _parse_capacity(low),
                                     preferred_floor=payload)
                if not res.get("found"):
                    continue
                plan.append({"kind": "library", "room_id": res["room_id"], "date": date_str,
                             "hour": cur_h, "minute": cur_m, "num_slots": n,
                             "floor": res["floor"], "capacity": res["capacity"],
                             "type": res["room_type"]})
                lib_used += n
                tot = cur_m + n * 30
                cur_h, cur_m = cur_h + tot // 60, tot % 60

        if not plan:
            # Nothing could be scheduled — surface any venue note so the user knows why.
            return "\n".join(notes) if notes else None
        self._pending_multi = {"plan": plan, "anchor_label": anchor["room_id"],
                               "anchor_end": anchor["end"], "anchor_fac": anchor_fac}
        body = self._format_multi_plan(plan, anchor["room_id"], anchor["end"])
        return ("\n".join(notes) + "\n\n" + body) if notes else body

    def _sport_venue_pref(self, sport: str, low: str, anchor_fac: Optional[str]) -> Optional[str]:
        """Resolve a venue preference for a sport: an explicit hint, or 'same place'
        meaning the anchor booking's venue (if it hosts that sport)."""
        hint = match_venue_hint(sport, low)
        if hint:
            return hint
        if anchor_fac and any(p in low for p in
                              ["same place", "same venue", "same spot", "same hall",
                               "same court", "same building"]):
            if any(v["id"] == anchor_fac for v in SPORT_VENUES.get(sport, [])):
                return anchor_fac
        return None

    def _revise_multi(self, low: str) -> Optional[str]:
        """
        The user tweaked attribute(s) on a pending multi-plan and the rest should be
        preserved. Supports:
          • sport venue   ("basketball at the sports hall")
          • study floor   ("make the study lg4")
          • study duration("make the study 1 hour instead")
          • an item's time("start the study at 6", "basketball at 5 instead")
        Time/duration changes re-flow the plan so items stay back-to-back.
        Returns the re-presented plan, or None if nothing matched.
        """
        ctx = self._pending_multi or {}
        work = [dict(p) for p in (ctx.get("plan") or [])]
        if not work:
            return None
        anchor_fac = ctx.get("anchor_fac")
        changed = False
        need_reflow = False

        # 0) Swap the order/times of two items ("swap the study and basketball times")
        if any(w in low for w in ["swap", "switch"]):
            def _label_pos(it):
                if it["kind"] == "library":
                    return low.find("study") if "study" in low else low.find("library")
                aliases = [a for a, k in SPORT_ALIASES.items() if k == it.get("sport") and a in low]
                return min((low.find(a) for a in aliases), default=-1)
            referenced = sorted((( _label_pos(it), idx) for idx, it in enumerate(work)
                                 if _label_pos(it) >= 0), key=lambda x: x[0])
            if len(referenced) >= 2:
                i, j = referenced[0][1], referenced[1][1]
                work[i], work[j] = work[j], work[i]
                changed = True
                need_reflow = True

        # 1) Sport venue (lock it — it was the user's explicit choice)
        for sk in self._sports_in_text(low):
            pref = self._sport_venue_pref(sk, low, anchor_fac)
            if not pref:
                continue
            for item in work:
                if item["kind"] == "sports" and item["sport"] == sk and item["facility"] != pref:
                    courts = next((v["courts"] for v in SPORT_VENUES.get(sk, [])
                                   if v["id"] == pref), None)
                    if courts and free_courts(item["date"], pref, sk, courts, item["hour"]) > 0:
                        item["facility"] = pref
                        item["label"] = FACILITIES[pref]["name"]
                        item["pref_facility"] = pref
                        changed = True
                    else:
                        return (f"{FACILITIES[pref]['name']} has no free courts at "
                                f"{item['hour']:02d}:00, so I kept {item['label']}. "
                                f"Want a different time or venue?")

        # 2) Study floor (in place, same time)
        floor = _parse_floor_pref(low)
        if floor:
            for item in work:
                if item["kind"] == "library" and item["floor"] != floor:
                    item["floor"] = floor      # actual room re-picked in re-flow / below
                    changed = True
                    need_reflow = True

        # 3) Study duration ("make the study 1 hour")
        has_dur = bool(re.search(r'\d+\s*hour', low) or re.search(r'\d+\s*(?:min|mins|minutes)', low)
                       or re.search(r'\b(an|a|one|two)\s+hours?\b', low)
                       or "half an hour" in low or "half hour" in low)
        if has_dur:
            n = self._parse_study_duration(low)
            for item in work:
                if item["kind"] == "library" and item["num_slots"] != n:
                    item["num_slots"] = n
                    changed = True
                    need_reflow = True

        # 4) Explicit time on an item ("start study at 6", "basketball at 5")
        if self._message_has_time(low):
            t = _parse_time_str(low)
            if t:
                named = self._sports_in_text(low)
                target = None
                for item in work:
                    if item["kind"] == "sports" and item["sport"] in named:
                        target = item
                        break
                if target is None:
                    target = next((it for it in work if it["kind"] == "library"), None)
                if target is not None and target.get("hour") != t[0]:
                    target["_forced"] = (t[0], 0 if target["kind"] == "sports"
                                         else (30 if t[1] >= 30 else 0))
                    changed = True
                    need_reflow = True

        if not changed:
            return None

        if need_reflow:
            sh, sm = int(ctx["anchor_end"][:2]), int(ctx["anchor_end"][3:])
            new_plan, err = self._reflow(work, sh, sm)
            if err:
                return f"{err} Your current plan is unchanged."
            work = new_plan

        self._pending_multi["plan"] = work
        return self._format_multi_plan(work, ctx["anchor_label"], ctx["anchor_end"])

    def _reflow(self, plan: list, start_h: int, start_m: int):
        """
        Recompute every item's time sequentially from (start_h, start_m), keeping items
        back-to-back and preferring each item's current venue/floor. An item carrying a
        '_forced' (h, m) jumps the cursor to that time first. Returns (new_plan, None)
        or (None, error_message) if something can't be placed.
        """
        cur_h, cur_m = start_h, start_m
        out = []
        for item in plan:
            it = dict(item)
            forced = it.pop("_forced", None)
            if forced:
                cur_h, cur_m = forced
            if cur_h >= 23:
                return None, "That runs past closing time."
            if it["kind"] == "sports":
                cur_m = 0  # 1-hour slots sit on the hour
                hours = [cur_h] if forced else list(range(cur_h, min(cur_h + 5, 23)))
                locked = it.get("pref_facility")
                if locked:
                    # User explicitly chose this venue — never silently substitute.
                    courts = next((v["courts"] for v in SPORT_VENUES.get(it["sport"], [])
                                   if v["id"] == locked), None)
                    chosen = next((h for h in hours if courts
                                   and free_courts(it["date"], locked, it["sport"], courts, h) > 0), None)
                    if chosen is None:
                        return None, (f"{FACILITIES[locked]['name']} has no "
                                      f"{SPORT_LABELS.get(it['sport'], it['sport'].title())} court free "
                                      f"around {cur_h:02d}:00.")
                    it["facility"], it["label"] = locked, FACILITIES[locked]["name"]
                    it["hour"] = chosen
                    cur_h, cur_m = chosen + 1, 0
                else:
                    chosen = next((h for h in hours
                                   if find_available_venue(it["date"], it["sport"], h,
                                                           preferred_facility=it["facility"])["status"] == "ok"), None)
                    if chosen is None:
                        return None, (f"{SPORT_LABELS.get(it['sport'], it['sport'].title())} "
                                      f"isn't free around {cur_h:02d}:00.")
                    res = find_available_venue(it["date"], it["sport"], chosen,
                                               preferred_facility=it["facility"])
                    it["facility"] = res["facility"]
                    it["label"] = FACILITIES[res["facility"]]["name"]
                    it["hour"] = chosen
                    cur_h, cur_m = chosen + 1, 0
            else:
                res = find_best_room(it["date"], cur_h, cur_m, it["num_slots"], 1,
                                     preferred_floor=it.get("floor"))
                if not res.get("found"):
                    return None, (f"No {it.get('floor') or 'study'} room is free for "
                                  f"{it['num_slots'] * 30} min at {cur_h:02d}:{cur_m:02d}.")
                it.update({"room_id": res["room_id"], "floor": res["floor"],
                           "capacity": res["capacity"], "type": res["room_type"],
                           "hour": cur_h, "minute": cur_m})
                tot = cur_m + it["num_slots"] * 30
                cur_h, cur_m = cur_h + tot // 60, tot % 60
            out.append(it)
        return out, None

    def _format_multi_plan(self, plan: list, anchor_label: str, anchor_end: str) -> str:
        lines = [f"Here's your plan after **{anchor_label}** (ends {anchor_end}):", ""]
        for i, p in enumerate(plan, 1):
            if p["kind"] == "sports":
                lines.append(f"{i}. 🏸 {SPORT_LABELS.get(p['sport'], p['sport'].title())} — "
                             f"{p['label']} {p['hour']:02d}:00–{p['hour'] + 1:02d}:00")
            else:
                eh = p["hour"] + (p["minute"] + p["num_slots"] * 30) // 60
                em = (p["minute"] + p["num_slots"] * 30) % 60
                lines.append(f"{i}. 📚 {p['room_id']} ({p['floor']}, seats {p['capacity']}) "
                             f"{p['hour']:02d}:{p['minute']:02d}–{eh:02d}:{em:02d}")
        lines.append("")
        lines.append("Shall I book all of these?")
        return "\n".join(lines)

    def _book_multi(self) -> str:
        """Book every item in the pending multi-plan. Deterministic — no LLM."""
        plan = (self._pending_multi or {}).get("plan", [])
        self._pending_multi = None
        booked, failed = [], []
        for p in plan:
            if p["kind"] == "sports":
                b = book_facility(p["date"], p["facility"], p["hour"], p["sport"])
                if b["success"]:
                    self.my_bookings.append({
                        "ref": b["booking_ref"], "room_id": b["name"], "fac_id": p["facility"],
                        "sport": p["sport"], "date": b["date"], "start": b["start"],
                        "end": b["end"], "num_slots": 1, "kind": "sports",
                    })
                    booked.append((f"{SPORT_LABELS.get(p['sport'], p['sport'].title())} · {b['name']}",
                                   b["start"], b["end"], b["booking_ref"]))
                else:
                    failed.append((p.get("label", p["sport"]), b["error"]))
            else:
                b = book_room(p["date"], p["room_id"], p["hour"], p["minute"], p["num_slots"])
                if b["success"]:
                    self.my_bookings.append({
                        "ref": b["booking_ref"], "room_id": b["room_id"], "date": b["date"],
                        "start": b["start"], "end": b["end"], "num_slots": p["num_slots"],
                    })
                    booked.append((b["room_id"], b["start"], b["end"], b["booking_ref"]))
                else:
                    failed.append((p["room_id"], b["error"]))

        lines = ["✅ All booked!" if not failed else "Booked what I could:", ""]
        for name, s, e, ref in booked:
            lines.append(f"• {name} {s}–{e}  ·  {ref}")
        for name, err in failed:
            lines.append(f"• ⚠️ {name}: {err}")
        return "\n".join(lines)

    def _handle_chained_study(self, low: str, now: datetime) -> Optional[str]:
        """
        "After I finish badminton, study at lg3 for 2 hours" → book a study room
        starting when an existing booking (the anchor) ends. Deterministic, no LLM.
        Returns a reply, or None to fall through if there's nothing to chain from.
        """
        date_str = now.strftime("%Y-%m-%d")
        today = [b for b in self.my_bookings if b["date"] == date_str]
        if not today:
            return None

        # Anchor = the booking we chain after. If a sport is named (e.g. "after
        # pingpong"), anchor on THAT sport's booking; else any sports booking; else
        # the latest-ending booking today.
        sport = resolve_sport(low)
        anchor = None
        if sport:
            anchor = next((b for b in reversed(today)
                           if b.get("kind") == "sports" and b.get("sport") == sport), None)
            if anchor is None:
                anchor = next((b for b in reversed(today) if b.get("kind") == "sports"), None)
        if anchor is None:
            anchor = max(today, key=lambda b: b["end"])

        start_h = int(anchor["end"][:2])
        start_m = 0 if int(anchor["end"][3:]) < 30 else 30

        num_slots = self._parse_study_duration(low)

        # Library daily slot limit (sports bookings don't count)
        used = sum(b["num_slots"] for b in today if b.get("kind") != "sports")
        remaining = self.MAX_SLOTS_PER_DAY - used
        if remaining <= 0:
            return (f"You've used your 2-hour daily library limit for {date_str}, "
                    f"so I can't add a study room after your {anchor['room_id']} booking.")
        num_slots = min(num_slots, remaining)

        floor = _parse_floor_pref(low)
        capacity = _parse_capacity(low)
        result = find_best_room(date_str, start_h, start_m, num_slots, capacity,
                                preferred_floor=floor)

        end_h = start_h + (start_m + num_slots * 30) // 60
        end_m = (start_m + num_slots * 30) % 60
        after = f"after your {anchor['room_id']} booking ends at {anchor['end']}"

        if not result.get("found"):
            where = f" on {floor}" if floor else ""
            alt = ""
            if result.get("alt_times"):
                alts = [f"{t} ({rm})" for t, rm, _ in result["alt_times"][:2]]
                alt = f" Free instead at: {', '.join(alts)}."
            return (f"No study room{where} is free {start_h:02d}:{start_m:02d}–"
                    f"{end_h:02d}:{end_m:02d} ({after}).{alt}")

        room_id = result["room_id"]
        room_info = LC_ROOMS.get(room_id) or STUDY_ROOMS.get(room_id)
        self.pending_booking = {
            "room_id": room_id, "date": date_str, "hour": start_h,
            "minute": start_m, "num_slots": num_slots, "capacity": result["capacity"],
        }
        return (
            f"Nice — {after}, you can study here:\n\n"
            f"📚 **{room_id}** ({room_info['type']}, {room_info['floor']}, "
            f"seats {room_info['capacity']}) is free "
            f"{start_h:02d}:{start_m:02d}–{end_h:02d}:{end_m:02d}.\n\n"
            f"Shall I book {room_id}?"
        )

    def _do_booking_search(
        self, date: str, hour: int, minute: int,
        num_slots: int, capacity: int, room_id: Optional[str]
    ) -> str:
        """Run booking search and set pending_booking. Returns extra_context string."""
        end_h = hour + (minute + num_slots * 30) // 60
        end_m = (minute + num_slots * 30) % 60

        if room_id:
            # Specific room requested — check if it's available for all slots
            all_rooms = {**LC_ROOMS, **STUDY_ROOMS}
            if room_id not in all_rooms:
                return f"\n\n[BOOKING SEARCH RESULT]\nRoom {room_id} doesn't exist. Valid rooms: LC-1–18, 1F-R1–4, LG1-R1–9, LG3-R1–9, LG4-R1–11."

            room_info = all_rooms[room_id]
            # Check availability for all slots
            h, m = hour, minute
            available = True
            for _ in range(num_slots):
                slot = get_slot_availability(date, h, m)
                room_type_key = "learning_commons" if room_id in LC_ROOMS else "study_rooms"
                if room_id not in slot.get(room_type_key, {}):
                    available = False
                    break
                m += 30
                if m >= 60:
                    m = 0
                    h += 1

            if available:
                self.pending_booking = {
                    "room_id": room_id,
                    "date": date,
                    "hour": hour,
                    "minute": minute,
                    "num_slots": num_slots,
                    "capacity": room_info["capacity"],
                }
                return (
                    f"\n\n[BOOKING RECOMMENDATION]\n"
                    f"Room: {room_id} ({room_info['type']}, {room_info['floor']}, "
                    f"seats {room_info['capacity']})\n"
                    f"Date: {date}\n"
                    f"Time: {hour:02d}:{minute:02d}–{end_h:02d}:{end_m:02d}\n"
                    f"Ask: 'Shall I book {room_id} for you?'"
                )
            else:
                conflict_info = get_room_conflict_info(date, room_id, hour, minute, num_slots)
                return (
                    f"\n\n[BOOKING SEARCH RESULT]\n"
                    f"{conflict_info}. Finding alternatives with similar capacity...\n" +
                    self._best_room_context(date, hour, minute, num_slots, capacity, end_h, end_m)
                )

        return self._best_room_context(date, hour, minute, num_slots, capacity, end_h, end_m)

    def _best_room_context(
        self, date: str, hour: int, minute: int,
        num_slots: int, capacity: int, end_h: int, end_m: int
    ) -> str:
        result = find_best_room(date, hour, minute, num_slots, capacity)
        rec_str = format_room_recommendation(
            result, date, f"{hour:02d}:{minute:02d}", f"{end_h:02d}:{end_m:02d}"
        )
        if result["found"]:
            self.pending_booking = {
                "room_id": result["room_id"],
                "date": date,
                "hour": hour,
                "minute": minute,
                "num_slots": num_slots,
                "capacity": result["capacity"],
            }
            return (
                f"\n\n[BOOKING RECOMMENDATION]\n{rec_str}\n"
                f"Ask: 'Shall I book {result['room_id']} for you?'"
            )
        else:
            return f"\n\n[BOOKING SEARCH RESULT]\n{rec_str}"

    def _extract_duration_from_reply(self, low: str) -> Optional[int]:
        """Extract num_slots from a short duration reply like '1', '2 hours', '30 min'."""
        dur_h = re.search(r'(\d+)\s*hour', low)
        dur_m = re.search(r'(\d+)\s*(?:min|mins|minutes)', low)
        bare  = re.search(r'^\s*(\d+)\s*$', low)
        if dur_h:
            return int(dur_h.group(1)) * 2
        if dur_m:
            return max(1, round(int(dur_m.group(1)) / 30))
        if bare:
            n = int(bare.group(1))
            # "1" = 1 hour = 2 slots; "2" = 2 hours = 4 slots; "30" = 1 slot
            if n <= 4:
                return n * 2
            if n <= 120:
                return max(1, round(n / 30))
        return None

    async def _call_model(self, user_message: str, extra_context: str) -> str:
        user_content = f"{user_message}\n{extra_context}" if extra_context else user_message
        self.conversation_history.append({"role": "user", "content": user_content})
        messages = [{"role": "system", "content": self.AGENT_PROMPT}] + self.conversation_history

        t0 = time.time()
        # Run the (blocking) SDK call off the event loop and bound it with a hard
        # timeout, so a slow/failing Azure auth or network can never freeze the bot.
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.deployment,
                    messages=messages,
                    max_tokens=600,
                    timeout=25,
                ),
                timeout=30,
            )
        except asyncio.TimeoutError:
            logger.error("Model call timed out after 30s")
            self.conversation_history.pop()  # drop the unanswered user turn
            return ("⏳ The AI model didn't respond in time — this is usually an Azure "
                    "auth/network issue. Tip: set AZURE_OPENAI_API_KEY in .env for instant, "
                    "reliable responses, then restart.")
        except Exception as e:
            logger.error(f"Model call failed: {e}")
            self.conversation_history.pop()
            return (f"⚠️ I couldn't reach the AI model ({type(e).__name__}). "
                    f"If this keeps happening, set AZURE_OPENAI_API_KEY in .env and restart.")
        logger.info(f"Model {time.time()-t0:.2f}s tokens={response.usage.prompt_tokens}")

        reply = response.choices[0].message.content
        # Strip internal tags that leaked into the reply
        reply = _INTERNAL_TAGS.sub("", reply).strip()

        # Compact history storage
        self.conversation_history[-1] = {"role": "user", "content": user_message}
        self.conversation_history.append({
            "role": "assistant",
            "content": reply[:300] + "..." if len(reply) > 300 else reply,
        })
        if len(self.conversation_history) > 8:
            self.conversation_history = self.conversation_history[-8:]

        return reply

    async def cleanup(self) -> None:
        self.conversation_history = []
        self.pending_booking = None
        self._waiting_duration = False
        self._duration_context = None
        self.my_bookings = []
        self._pending_cancel_ref = None
        self._edit_after_cancel = False
        self._rejected_booking_ctx = None
        self._pending_edit = None
        self.pending_sports = None
        self._waiting_sports_time = None
        self._pending_multi = None
        self._plan_ctx = None
        logger.info("Cleanup done")
