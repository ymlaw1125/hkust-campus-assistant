"""
HKUST Sports Facilities — availability + booking (demo data).

Mirrors library_client.py but models the sports booking rules:
  • 1-HOUR slots (not 30-min)
  • 1 slot per facility per day per user
  • Booking opens 8:00am, 7 days in advance
  • Some activities are walk-in / contact-only / separate system (not bookable here)

Data source: official HKUST FBS facility list (fbs.hkust.edu.hk). Court counts
marked "*est" are not published by HKUST and are reasonable estimates for the demo.

Availability is simulated deterministically (seeded by date) so the same day always
looks the same, but each day differs. Real bookings made via db.py are layered on top.
"""
from datetime import datetime
import random

BOOKING_URL = "https://fbs.hkust.edu.hk"
SLOT_MINUTES = 60          # 1-hour slots
MAX_SLOTS_PER_FACILITY = 1 # 1 slot per facility per day

# ============================================================
# FACILITY INVENTORY
#   eligibility: "all" | "students_only" | "staff_only"
#   bookable:    True  -> 1-hr slot booking via FBS
#                False -> walk-in / contact / separate system (info only)
# ============================================================
FACILITIES = {
    # ---- Indoor Sports Complex ----
    "SHHO": {
        "name": "S. H. Ho Sports Hall", "location": "LG1, Indoor Sports Complex",
        "open": 9, "close": 22, "eligibility": "all", "bookable": True,
        "note": "Shared 1,600 sqm hall · Mon–Fri 2–5pm students only",
    },
    "SQUASH-LG4": {
        "name": "Squash Courts", "location": "LG4, Indoor Sports Complex",
        "open": 8, "close": 22, "eligibility": "all", "bookable": True, "note": "",
    },
    "TT-LG1031": {
        "name": "Table Tennis Room (LG1031)", "location": "LG1, Indoor Sports Complex",
        "open": 8, "close": 22, "eligibility": "all", "bookable": True, "note": "",
    },
    "CLIMB-LG4": {
        "name": "Climbing Wall", "location": "LG4, Indoor Sports Complex",
        "open": 8, "close": 22, "eligibility": "all", "bookable": True, "note": "",
    },
    "MPR-LG1027": {
        "name": "Multi-Purpose Room LG1027", "location": "LG1, Indoor Sports Complex",
        "open": 8, "close": 22, "eligibility": "all", "bookable": True, "note": "",
    },
    "MPR-LG4204": {
        "name": "Multi-Purpose Room LG4204", "location": "LG4, Indoor Sports Complex",
        "open": 8, "close": 22, "eligibility": "all", "bookable": True, "note": "",
    },
    "FITNESS-LG4": {
        "name": "Fitness Center (LG4)", "location": "LG4, Indoor Sports Complex",
        "open": 7, "close": 22, "eligibility": "all", "bookable": False,
        "note": "Separate system: fitness-booking.hkust.edu.hk",
    },
    # ---- Tsang Shiu Tim Sports Center ----
    "ARENA": {
        "name": "Arena", "location": "2/F, Tsang Shiu Tim Sports Center",
        "open": 9, "close": 22, "eligibility": "all", "bookable": True, "note": "",
    },
    "TT-TST": {
        "name": "Table Tennis / Multi-Purpose Room", "location": "2/F, Tsang Shiu Tim Sports Center",
        "open": 8, "close": 22, "eligibility": "all", "bookable": True, "note": "",
    },
    "EXERCISE-TST": {
        "name": "Exercise Zone (2/F) + Exercise Room (3/F)", "location": "Tsang Shiu Tim Sports Center",
        "open": 8, "close": 22, "eligibility": "all", "bookable": True,
        "note": "Opens 7:15am",
    },
    # ---- Seafront / outdoor ----
    "TENNIS-12": {
        "name": "Tennis Courts No. 1 & 2", "location": "Seafront (floodlit)",
        "open": 7, "close": 22, "eligibility": "students_only", "bookable": True, "note": "",
    },
    "TENNIS-4": {
        "name": "Tennis Court No. 4", "location": "Staff Quarters (floodlit)",
        "open": 7, "close": 21, "eligibility": "staff_only", "bookable": True, "note": "",
    },
    "TENNIS-7": {
        "name": "Tennis Court No. 7", "location": "Staff Quarters (floodlit)",
        "open": 7, "close": 21, "eligibility": "all", "bookable": True, "note": "",
    },
    "BBALL-OUT": {
        "name": "Outdoor Basketball Court", "location": "Seafront (floodlit)",
        "open": 7, "close": 22, "eligibility": "all", "bookable": True, "note": "",
    },
    "SOCCER-TURF": {
        "name": "Artificial Turf Soccer Pitch", "location": "Seafront, Fok Ying Tung (floodlit)",
        "open": 7, "close": 22, "eligibility": "all", "bookable": True, "note": "Standard size",
    },
    "SOCCER-MINI": {
        "name": "Mini-Soccer Pitch", "location": "Seafront (floodlit)",
        "open": 7, "close": 22, "eligibility": "all", "bookable": True, "note": "",
    },
    # ---- Walk-in / contact (info only) ----
    "TRACK": {
        "name": "400m Athletic Track", "location": "Seafront, Fok Ying Tung (floodlit)",
        "open": 6, "close": 22, "eligibility": "all", "bookable": False,
        "note": "Walk-in, no booking · 8 lanes · from 6:30am",
    },
    "POOL": {
        "name": "Swimming Pool", "location": "Seafront",
        "open": 7, "close": 21, "eligibility": "all", "bookable": False,
        "note": "Walk-in, no booking · 50m outdoor (Apr–Oct) / 25m indoor (Oct–Apr) · cleaning break 2–3pm",
    },
    "WATERSPORTS": {
        "name": "Water Sports Center", "location": "Seafront",
        "open": 8, "close": 18, "eligibility": "all", "bookable": False,
        "note": "Contact 2358-7385 for equipment & arrangements",
    },
}

# ============================================================
# SPORT -> VENUES  (each venue carries its court/unit count for that sport)
#   courts marked here are simultaneous bookable units (*est where unpublished)
# ============================================================
SPORT_VENUES = {
    "badminton":    [{"id": "SHHO", "courts": 4}, {"id": "ARENA", "courts": 4}],
    "squash":       [{"id": "SQUASH-LG4", "courts": 4}],
    "table_tennis": [{"id": "TT-LG1031", "courts": 6}, {"id": "TT-TST", "courts": 4}],
    "tennis":       [{"id": "TENNIS-12", "courts": 2}, {"id": "TENNIS-7", "courts": 1},
                     {"id": "TENNIS-4", "courts": 1}],
    "basketball":   [{"id": "SHHO", "courts": 1}, {"id": "ARENA", "courts": 1},
                     {"id": "BBALL-OUT", "courts": 1}],
    "football":     [{"id": "SOCCER-TURF", "courts": 1}, {"id": "SOCCER-MINI", "courts": 1}],
    "volleyball":   [{"id": "SHHO", "courts": 1}, {"id": "ARENA", "courts": 1}],
    "handball":     [{"id": "SHHO", "courts": 1}, {"id": "ARENA", "courts": 1}],
    "netball":      [{"id": "SHHO", "courts": 1}, {"id": "ARENA", "courts": 1}],
    "pickleball":   [{"id": "ARENA", "courts": 2}],
    "climbing":     [{"id": "CLIMB-LG4", "courts": 1}],
    "gym":          [{"id": "EXERCISE-TST", "courts": 20}],
    "dance":        [{"id": "MPR-LG1027", "courts": 1}, {"id": "MPR-LG4204", "courts": 1}],
    "fencing":      [{"id": "MPR-LG1027", "courts": 1}, {"id": "MPR-LG4204", "courts": 1}],
    "martial_arts": [{"id": "MPR-LG1027", "courts": 1}, {"id": "MPR-LG4204", "courts": 1}],
    "archery":      [{"id": "MPR-LG1027", "courts": 1}, {"id": "MPR-LG4204", "courts": 1}],
}

# Non-bookable activities -> the facility to point the user at.
SPORT_INFO = {
    "running":    "TRACK",
    "swimming":   "POOL",
    "dragon_boat": "WATERSPORTS",
    "rowing":     "WATERSPORTS",
    "windsurfing": "WATERSPORTS",
    "kayaking":   "WATERSPORTS",
    "sailing":    "WATERSPORTS",
}

# Pretty labels
SPORT_LABELS = {
    "badminton": "Badminton", "squash": "Squash", "table_tennis": "Table Tennis",
    "tennis": "Tennis", "basketball": "Basketball", "football": "Football",
    "volleyball": "Volleyball", "handball": "Handball", "netball": "Netball",
    "pickleball": "Pickleball", "climbing": "Rock Climbing", "gym": "Gym / Fitness",
    "dance": "Dance", "fencing": "Fencing", "martial_arts": "Martial Arts",
    "archery": "Archery", "running": "Running", "swimming": "Swimming",
    "dragon_boat": "Dragon Boat", "rowing": "Rowing", "windsurfing": "Windsurfing",
    "kayaking": "Kayaking", "sailing": "Sailing",
}

# Word -> canonical sport key (checked longest-first so multi-word wins)
SPORT_ALIASES = {
    "badminton": "badminton",
    "squash": "squash",
    "table tennis": "table_tennis", "ping pong": "table_tennis",
    "ping-pong": "table_tennis", "pingpong": "table_tennis",
    "tennis": "tennis",
    "basketball": "basketball", "bball": "basketball", "hoops": "basketball",
    "football": "football", "soccer": "football", "futsal": "football",
    "volleyball": "volleyball",
    "handball": "handball",
    "netball": "netball",
    "pickleball": "pickleball",
    "rock climbing": "climbing", "bouldering": "climbing", "climbing": "climbing",
    "gym": "gym", "fitness": "gym", "workout": "gym", "work out": "gym",
    "weights": "gym", "weight training": "gym", "exercise": "gym",
    "dance": "dance", "dancing": "dance",
    "fencing": "fencing",
    "martial arts": "martial_arts", "kung fu": "martial_arts", "taekwondo": "martial_arts",
    "judo": "martial_arts", "karate": "martial_arts",
    "archery": "archery",
    "running": "running", "jogging": "running", "jog": "running",
    "athletics": "running", "400m": "running",
    "swimming": "swimming", "swim": "swimming",
    "dragon boat": "dragon_boat",
    "rowing": "rowing",
    "windsurfing": "windsurfing", "windsurf": "windsurfing",
    "kayaking": "kayaking", "kayak": "kayaking",
    "sailing": "sailing",
}


def resolve_sport(text: str) -> str | None:
    """Find the canonical sport key mentioned in a message (longest alias wins)."""
    low = text.lower()
    for alias in sorted(SPORT_ALIASES, key=len, reverse=True):
        if alias in low:
            return SPORT_ALIASES[alias]
    return None


# Location keywords that point at a specific venue.
_VENUE_HINTS = {
    "lg1031": "TT-LG1031", "lg1027": "MPR-LG1027", "lg4204": "MPR-LG4204",
    "lg1": "LG1", "lg4": "LG4", "lg3": "LG3",
    "arena": "ARENA", "tsang shiu tim": "TST", "tst": "TST", "shiu tim": "TST",
    "multi purpose": "MPR", "multipurpose": "MPR", "multi purpose room": "MPR",
    "seafront": "SEAFRONT", "outdoor": "OUTDOOR", "indoor sports complex": "ISC",
    "mini": "MINI", "turf": "TURF",
    "fok ying tung": "FYT", "staff quarters": "STAFF", "s. h. ho": "SHHO",
    "sh ho": "SHHO", "ho sports hall": "SHHO",
}


def _norm(s: str) -> str:
    """Lowercase and treat hyphens/slashes as spaces so 'multi-purpose' == 'multi purpose'."""
    return s.lower().replace("-", " ").replace("/", " ")


def match_venue_hint(sport: str, text: str) -> str | None:
    """If the text names a specific venue for this sport, return that facility id."""
    low = _norm(text)
    venues = SPORT_VENUES.get(sport, [])
    for v in venues:
        fid = v["id"]
        fac = FACILITIES[fid]
        blob = _norm(f"{fid} {fac['name']} {fac['location']}")
        # Direct id / name match
        if _norm(fid) in low or _norm(fac["name"]) in low:
            return fid
        # Keyword hint that also appears in this venue's id/name/location
        for kw in sorted(_VENUE_HINTS, key=len, reverse=True):
            if kw in low and kw in blob:
                return fid
    return None


# ============================================================
# OCCUPANCY MODEL  (per facility/sport/hour)
# ============================================================
def _occupancy_rate(hour: int) -> float:
    if hour < 9:    return 0.10   # early
    if hour < 12:   return 0.35
    if hour < 14:   return 0.45
    if hour < 17:   return 0.55
    if 17 <= hour < 21: return 0.80  # after-work / after-class peak
    return 0.40

_schedule_cache: dict = {}


def _facility_seed(date_str: str, fac_id: str, sport: str) -> int:
    date_num = int(date_str.replace("-", ""))
    key = fac_id + "|" + sport
    word_num = sum(ord(c) * (i + 1) for i, c in enumerate(key))
    return date_num + word_num * 31


def _get_facility_schedule(date_str: str, fac_id: str, sport: str, courts: int) -> dict:
    """Return {hour: occupied_court_count} for this facility+sport for the day."""
    key = (date_str, fac_id, sport)
    if key in _schedule_cache:
        return _schedule_cache[key]

    fac = FACILITIES[fac_id]
    rng = random.Random(_facility_seed(date_str, fac_id, sport))
    schedule = {}
    for h in range(fac["open"], fac["close"]):
        rate = _occupancy_rate(h)
        occ = sum(1 for _ in range(courts) if rng.random() < rate)
        schedule[h] = min(occ, courts)
    _schedule_cache[key] = schedule
    return schedule


def _eligible(fac: dict, user_type: str) -> bool:
    if fac["eligibility"] == "all":
        return True
    if fac["eligibility"] == "students_only":
        return user_type == "student"
    if fac["eligibility"] == "staff_only":
        return user_type == "staff"
    return True


def free_courts(date_str: str, fac_id: str, sport: str, courts: int, hour: int) -> int:
    """How many courts/units are free at this facility+sport+hour."""
    from db import is_booked
    fac = FACILITIES[fac_id]
    if hour < fac["open"] or hour >= fac["close"]:
        return 0
    sched = _get_facility_schedule(date_str, fac_id, sport, courts)
    occ = sched.get(hour, courts)
    free = courts - occ
    if is_booked(date_str, fac_id, hour, 0):
        free -= 1  # the user's own booking holds a court
    return max(0, free)


# ============================================================
# QUERIES
# ============================================================
def find_available_venue(date_str: str, sport: str, hour: int,
                         user_type: str = "student", exclude: set = None,
                         preferred_facility: str = None) -> dict:
    """
    Find the best bookable venue for a sport at a given hour.
    exclude: facility ids the user has already booked today (1 slot/facility/day rule).
    preferred_facility: if set and free, prefer this specific venue.
    Returns a dict describing the outcome (bookable, info-only, or none).
    """
    exclude = exclude or set()
    if sport in SPORT_INFO:
        fac_id = SPORT_INFO[sport]
        return {"status": "info", "facility": fac_id, "facility_info": FACILITIES[fac_id]}

    venues = SPORT_VENUES.get(sport)
    if not venues:
        return {"status": "unknown"}

    options = []
    for v in venues:
        fac_id, courts = v["id"], v["courts"]
        fac = FACILITIES[fac_id]
        if not fac["bookable"] or not _eligible(fac, user_type) or fac_id in exclude:
            continue
        if hour < fac["open"] or hour >= fac["close"]:
            continue
        fc = free_courts(date_str, fac_id, sport, courts, hour)
        if fc > 0:
            options.append({"id": fac_id, "free": fc, "courts": courts, "info": fac})

    if options:
        # Prefer the named venue if it's free, else the least-crowded one
        options.sort(key=lambda o: (o["id"] != preferred_facility, -o["free"]))
        best = options[0]
        return {
            "status": "ok",
            "sport": sport,
            "facility": best["id"],
            "facility_info": best["info"],
            "free": best["free"],
            "courts": best["courts"],
            "hour": hour,
            "alternatives": options[1:],
        }

    # Nothing free now — look for nearby hours at any venue
    alt_times = []
    for delta in (1, -1, 2, -2, 3):
        ah = hour + delta
        for v in venues:
            fac_id, courts = v["id"], v["courts"]
            fac = FACILITIES[fac_id]
            if not fac["bookable"] or not _eligible(fac, user_type) or fac_id in exclude:
                continue
            if ah < fac["open"] or ah >= fac["close"]:
                continue
            if free_courts(date_str, fac_id, sport, courts, ah) > 0:
                alt_times.append({"hour": ah, "facility": fac_id})
                break
        if len(alt_times) >= 3:
            break
    return {"status": "full", "sport": sport, "hour": hour, "alt_times": alt_times}


def book_facility(date_str: str, fac_id: str, hour: int, sport: str = "") -> dict:
    """Reserve one 1-hour slot at a facility. Writes to db.py."""
    from db import add_booking, is_booked
    fac = FACILITIES.get(fac_id)
    if not fac:
        return {"success": False, "error": f"Facility {fac_id} does not exist."}
    if not fac["bookable"]:
        return {"success": False, "error": f"{fac['name']} is not bookable ({fac['note']})."}
    if hour < fac["open"] or hour >= fac["close"]:
        return {"success": False,
                "error": f"{fac['name']} is closed at {hour:02d}:00 (hours {fac['open']:02d}:00–{fac['close']:02d}:00)."}
    if is_booked(date_str, fac_id, hour, 0):
        return {"success": False, "error": f"{fac['name']} already has your booking at {hour:02d}:00."}

    ref = f"DEMO-{fac_id}-{date_str}-{hour:02d}00-{random.randint(1000, 9999)}"
    if not add_booking(date_str, fac_id, hour, 0, 1, ref):
        return {"success": False, "error": "Booking failed — slot just taken."}

    return {
        "success": True, "facility_id": fac_id, "name": fac["name"],
        "location": fac["location"], "sport": sport, "date": date_str,
        "start": f"{hour:02d}:00", "end": f"{hour + 1:02d}:00", "booking_ref": ref,
    }


# ============================================================
# FORMATTING
# ============================================================
def format_venue_recommendation(result: dict, date_str: str, sport: str) -> str:
    label = SPORT_LABELS.get(sport, sport.title())
    status = result["status"]

    if status == "info":
        f = result["facility_info"]
        return (f"🏃 **{label}** at HKUST is at the **{f['name']}** ({f['location']}).\n"
                f"{f['note']}.\nNo online booking needed — just turn up.")

    if status == "unknown":
        return (f"I'm not sure which HKUST facility covers '{label}'. "
                f"Check the full list at {BOOKING_URL}.")

    if status == "full":
        lines = [f"All {label} venues are fully booked at {result['hour']:02d}:00."]
        if result.get("alt_times"):
            slots = ", ".join(
                f"{a['hour']:02d}:00 ({FACILITIES[a['facility']]['name']})"
                for a in result["alt_times"])
            lines.append(f"Next free: {slots}.")
        return "\n".join(lines)

    # ok
    f = result["facility_info"]
    h = result["hour"]
    note = f" · {f['note']}" if f["note"] else ""
    alts = ""
    if result.get("alternatives"):
        alt_strs = [f"{a['info']['name']} ({a['free']} free)" for a in result["alternatives"]]
        alts = f"\nAlso open: {', '.join(alt_strs)}."
    return (
        f"🏸 **{label}** — **{f['name']}** ({f['location']}){note}\n"
        f"{result['free']} of {result['courts']} courts free at {h:02d}:00–{h + 1:02d}:00 on {date_str}.{alts}"
    )


def format_sports_confirmation(booking: dict) -> str:
    if not booking["success"]:
        return f"Booking failed: {booking['error']}"
    label = SPORT_LABELS.get(booking.get("sport", ""), "")
    label_str = f"{label} · " if label else ""
    return (
        f"✅ Booked! {label_str}{booking['name']} ({booking['location']}) "
        f"on {booking['date']} from {booking['start']} to {booking['end']}.\n"
        f"Reference: {booking['booking_ref']}\n"
        f"Note: Demo booking. For real bookings visit {BOOKING_URL} "
        f"(opens 8am, 7 days ahead · unclaimed slots released after 10 min)."
    )


def list_venues_for_sport(sport: str) -> str:
    label = SPORT_LABELS.get(sport, sport.title())
    if sport in SPORT_INFO:
        f = FACILITIES[SPORT_INFO[sport]]
        return f"**{label}** — {f['name']} ({f['location']}). {f['note']}."
    venues = SPORT_VENUES.get(sport)
    if not venues:
        return f"No HKUST facility found for {label}."
    lines = [f"**{label}** venues:"]
    for v in venues:
        f = FACILITIES[v["id"]]
        elig = "" if f["eligibility"] == "all" else f" · {f['eligibility'].replace('_', ' ')}"
        lines.append(f"• {f['name']} ({f['location']}) — {f['open']:02d}:00–{f['close']:02d}:00, "
                     f"{v['courts']} court(s){elig}")
    return "\n".join(lines)


def print_sports_day_text(date_str: str = None, sport_filter: str = None,
                          from_hour: int = None, num_slots: int = None,
                          user_type: str = "student") -> str:
    """Markdown availability tables (rows = hours, cols = venues) per sport."""
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    sports = [sport_filter] if sport_filter and sport_filter in SPORT_VENUES else list(SPORT_VENUES)

    lines = [f"## Sports Availability — {date_str}", ""]
    for sport in sports:
        venues = [v for v in SPORT_VENUES[sport]
                  if FACILITIES[v["id"]]["bookable"] and _eligible(FACILITIES[v["id"]], user_type)]
        if not venues:
            continue
        label = SPORT_LABELS.get(sport, sport.title())
        lines.append(f"### {label}")
        lines.append("")
        names = [FACILITIES[v["id"]]["name"] for v in venues]
        lines.append("| Time | " + " | ".join(names) + " |")
        lines.append("|------|" + "|".join(["------"] * len(names)) + "|")

        opens = min(FACILITIES[v["id"]]["open"] for v in venues)
        closes = max(FACILITIES[v["id"]]["close"] for v in venues)
        start_h = from_hour if from_hour is not None else opens
        end_h = (start_h + num_slots) if num_slots else closes

        for h in range(max(start_h, opens), min(end_h, closes)):
            cells = []
            for v in venues:
                fac = FACILITIES[v["id"]]
                if h < fac["open"] or h >= fac["close"]:
                    cells.append("—")
                    continue
                fc = free_courts(date_str, v["id"], sport, v["courts"], h)
                cells.append(f"✅ {fc}" if fc > 0 else "❌")
            lines.append(f"| {h:02d}:00 | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("### Legend")
    lines.append("- ✅ N = N courts/units free")
    lines.append("- ❌ = Fully booked · — = Closed")
    lines.append(f"- 1-hour slots · 1 slot per facility per day · Book at: {BOOKING_URL}")
    return "\n".join(lines)


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    print(format_venue_recommendation(
        find_available_venue(date, "badminton", 19), date, "badminton"))
    print()
    print(format_venue_recommendation(
        find_available_venue(date, "swimming", 19), date, "swimming"))
    print()
    print(list_venues_for_sport("tennis"))
    print()
    print(print_sports_day_text(date, "badminton"))
