from datetime import datetime
import random

# ============================================================
# LIBRARY HOURS
# ============================================================
LIBRARY_OPEN  = 8    # 8:00am
LIBRARY_CLOSE = 23   # 11:00pm
SLOT_MINUTES  = 30   # 30-minute slots
MAX_CONSECUTIVE_SLOTS = 4  # max 2 hours per booking

# ============================================================
# ROOM INVENTORY
# ============================================================

# Learning Commons — LG1
LC_ROOMS = {
    **{f"LC-{i}":  {"capacity": 7,  "floor": "LG1", "type": "Learning Commons"} for i in range(1, 14)},
    **{f"LC-{i}":  {"capacity": 10, "floor": "LG1", "type": "Learning Commons"} for i in range(14, 18)},
    "LC-18":       {"capacity": 6,  "floor": "LG1", "type": "Learning Commons"},
}

# Library Study Rooms — scattered across floors
STUDY_ROOMS = {
    **{f"1F-R{i}":  {"capacity": 4, "floor": "1/F",  "type": "Library Study Room"} for i in range(1, 5)},
    **{f"LG1-R{i}": {"capacity": 4, "floor": "LG1",  "type": "Library Study Room"} for i in range(1, 10)},
    **{f"LG3-R{i}": {"capacity": 4, "floor": "LG3",  "type": "Library Study Room"} for i in range(1, 10)},
    **{f"LG4-R{i}": {"capacity": 4, "floor": "LG4",  "type": "Library Study Room"} for i in range(1, 12)},
}

ALL_ROOMS = {**LC_ROOMS, **STUDY_ROOMS}

# ============================================================
# OCCUPANCY MODEL
# Peak hours = higher occupancy
# ============================================================
def _occupancy_rate(hour: int) -> float:
    if 8 <= hour < 10:   return 0.15   # early morning — quiet
    if 10 <= hour < 13:  return 0.55   # mid-morning peak
    if 13 <= hour < 14:  return 0.70   # lunch rush — very busy
    if 14 <= hour < 18:  return 0.65   # afternoon peak
    if 18 <= hour < 20:  return 0.50   # evening moderate
    if 20 <= hour < 22:  return 0.35   # late evening quieter
    if 22 <= hour < 23:  return 0.20   # nearly closing
    return 0.10                         # off-hours fallback

# ============================================================
# AVAILABILITY GENERATOR
# Generates per-slot (30min) availability for every room
# Seeded by date so results are consistent for the same day
# but change each day
# ============================================================
def _get_day_seed(date_str: str) -> int:
    return int(date_str.replace("-", ""))

def _room_seed(date_str: str, room_id: str, hour: int, minute: int) -> int:
    """Stable seed that doesn't use Python's hash() which changes per session"""
    date_num = int(date_str.replace("-", ""))
    room_num = sum(ord(c) * (i + 1) for i, c in enumerate(room_id))
    return date_num + room_num * 31 + hour * 100 + (1 if minute == 30 else 0)

_schedule_cache: dict = {}

def _get_room_schedule(date_str: str, room_id: str) -> set:
    """
    Generate a set of (hour, minute) tuples when this room is OCCUPIED for the day.
    Simulates students booking 1-4 consecutive slots (30min each).
    Results are cached per (date, room) so repeated calls are free.
    """
    key = (date_str, room_id)
    if key in _schedule_cache:
        return _schedule_cache[key]

    seed = _room_seed(date_str, room_id, 0, 0)
    rng = random.Random(seed)

    occupied = set()
    # Generate all 30-min slots in the day
    all_slots = []
    for h in range(LIBRARY_OPEN, LIBRARY_CLOSE):
        all_slots.append((h, 0))
        all_slots.append((h, 30))

    # Simulate bookings throughout the day
    # Each booking is 1-4 slots long, with gaps between
    i = 0
    while i < len(all_slots):
        h, m = all_slots[i]
        occupancy = _occupancy_rate(h)

        # Decide if someone books starting at this slot
        if rng.random() < occupancy * 0.6:
            # Book 1-4 consecutive slots
            duration = rng.randint(1, 4)
            for j in range(duration):
                if i + j < len(all_slots):
                    occupied.add(all_slots[i + j])
            i += duration + rng.randint(0, 2)  # gap before next booking
        else:
            i += 1

    _schedule_cache[key] = occupied
    return occupied

def get_slot_availability(date_str: str, hour: int, minute: int) -> dict:
    """Returns availability for a single 30-min slot."""
    assert minute in (0, 30), "minute must be 0 or 30"

    if hour < LIBRARY_OPEN or hour >= LIBRARY_CLOSE:
        return {
            "date": date_str, "hour": hour, "minute": minute,
            "closed": True,
            "message": f"Library closed at {hour:02d}:{minute:02d}. Hours: {LIBRARY_OPEN:02d}:00–{LIBRARY_CLOSE:02d}:00"
        }

    from db import is_booked

    available_lc = {}
    available_sr = {}

    for room_id, info in LC_ROOMS.items():
        occupied = _get_room_schedule(date_str, room_id)
        if (hour, minute) not in occupied and not is_booked(date_str, room_id, hour, minute):
            available_lc[room_id] = info

    for room_id, info in STUDY_ROOMS.items():
        occupied = _get_room_schedule(date_str, room_id)
        if (hour, minute) not in occupied and not is_booked(date_str, room_id, hour, minute):
            available_sr[room_id] = info

    return {
        "date": date_str,
        "hour": hour,
        "minute": minute,
        "closed": False,
        "learning_commons": available_lc,
        "study_rooms": available_sr,
        "booking_url": "https://lbbooking.ust.hk",
    }

def get_slots_range(date_str: str, from_hour: int, from_minute: int, num_slots: int) -> list:
    """Get availability for consecutive 30-min slots starting from a given time."""
    slots = []
    h, m = from_hour, from_minute
    for _ in range(num_slots):
        if h >= LIBRARY_CLOSE:
            break
        slots.append(get_slot_availability(date_str, h, m))
        m += 30
        if m >= 60:
            m = 0
            h += 1
    return slots

def find_rooms_free_until(date_str: str, from_hour: int, from_minute: int, until_hour: int) -> dict:
    """
    Find rooms that are continuously free from from_hour:from_minute until until_hour:00.
    Returns rooms free in ALL slots in that range.
    """
    # Build list of all slots in range
    slots = []
    h, m = from_hour, from_minute
    while h < until_hour:
        slots.append(get_slot_availability(date_str, h, m))
        m += 30
        if m >= 60:
            m = 0
            h += 1

    if not slots:
        return {"learning_commons": {}, "study_rooms": {}}

    # Start with all available rooms in first slot
    free_lc = set(slots[0]["learning_commons"].keys())
    free_sr = set(slots[0]["study_rooms"].keys())

    # Intersect with each subsequent slot
    for slot in slots[1:]:
        free_lc &= set(slot["learning_commons"].keys())
        free_sr &= set(slot["study_rooms"].keys())

    return {
        "date": date_str,
        "from": f"{from_hour:02d}:{from_minute:02d}",
        "until": f"{until_hour:02d}:00",
        "learning_commons": {k: LC_ROOMS[k] for k in free_lc},
        "study_rooms": {k: STUDY_ROOMS[k] for k in free_sr},
        "booking_url": "https://lbbooking.ust.hk",
    }

# ============================================================
# FORMATTING
# ============================================================
def format_slot_for_agent(slot: dict) -> str:
    if slot.get("closed"):
        return slot["message"]

    h, m = slot["hour"], slot["minute"]
    time_str = f"{h:02d}:{m:02d}–{h:02d}:{'30' if m == 0 else '00'}" if m == 0 else f"{h:02d}:30–{h+1:02d}:00"
    lc = slot["learning_commons"]
    sr = slot["study_rooms"]

    lines = [f"Slot {time_str}:"]

    if lc:
        sample = list(lc.items())[:4]
        sample_str = ", ".join(f"{r} ({info['capacity']}p)" for r, info in sample)
        extra = f" +{len(lc)-4} more" if len(lc) > 4 else ""
        lines.append(f"  Learning Commons (LG1): {sample_str}{extra}")
    else:
        lines.append("  Learning Commons: Fully booked")

    if sr:
        by_floor = {}
        for r, info in sr.items():
            by_floor.setdefault(info["floor"], []).append(r)
        for floor, rooms in sorted(by_floor.items()):
            sample = rooms[:3]
            extra = f" +{len(rooms)-3} more" if len(rooms) > 3 else ""
            lines.append(f"  Study Rooms {floor}: {', '.join(sample)}{extra}")
    else:
        lines.append("  Study Rooms: Fully booked")

    return "\n".join(lines)

def format_range_for_agent(slots: list) -> str:
    lines = []
    for slot in slots:
        if slot.get("closed"):
            continue
        h, m = slot["hour"], slot["minute"]
        end_m = 30 if m == 0 else 0
        end_h = h if m == 0 else h + 1
        time_str = f"{h:02d}:{m:02d}–{end_h:02d}:{end_m:02d}"
        lc_count = len(slot["learning_commons"])
        sr_count = len(slot["study_rooms"])
        lines.append(f"• {time_str}: {lc_count} LC rooms, {sr_count} study rooms free")
    return "\n".join(lines) if lines else "No availability in this range."

def format_continuous_for_agent(result: dict) -> str:
    lc = result["learning_commons"]
    sr = result["study_rooms"]
    frm = result["from"]
    until = result["until"]
    url = result["booking_url"]

    lines = [f"Rooms free continuously from {frm} to {until}:"]

    if lc:
        sample = list(lc.items())[:5]
        sample_str = ", ".join(f"{r} ({info['capacity']}p)" for r, info in sample)
        extra = f" +{len(lc)-5} more" if len(lc) > 5 else ""
        lines.append(f"• Learning Commons (LG1): {sample_str}{extra}")
    else:
        lines.append("• Learning Commons: No rooms free for the full period")

    if sr:
        by_floor = {}
        for r, info in sr.items():
            by_floor.setdefault(info["floor"], []).append(r)
        for floor, rooms in sorted(by_floor.items()):
            sample = rooms[:3]
            extra = f" +{len(rooms)-3} more" if len(rooms) > 3 else ""
            lines.append(f"• Study Rooms {floor}: {', '.join(sample)}{extra}")
    else:
        lines.append("• Study Rooms: No rooms free for the full period")

    lines.append(f"• Book at: {url}")
    return "\n".join(lines)

def print_full_day(date_str: str = None):
    """Print full day availability for all rooms — for debugging"""
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    
    print(f"\n{'='*70}")
    print(f"FULL DAY AVAILABILITY — {date_str}")
    print(f"{'='*70}")
    
    all_slots = []
    for h in range(LIBRARY_OPEN, LIBRARY_CLOSE):
        all_slots.append((h, 0))
        all_slots.append((h, 30))
    
    # Header row
    room_ids = list(LC_ROOMS.keys()) + list(STUDY_ROOMS.keys())
    print(f"\n{'Time':<8}", end="")
    for r in room_ids:
        print(f"{r:<10}", end="")
    print()
    print("-" * (8 + 10 * len(room_ids)))
    
    # Each slot
    for h, m in all_slots:
        slot = get_slot_availability(date_str, h, m)
        print(f"{h:02d}:{m:02d}   ", end="")
        for r in room_ids[:18]:  # LC rooms
            avail = "✓" if r in slot["learning_commons"] else "✗"
            print(f"{avail:<10}", end="")
        for r in room_ids[18:]:  # study rooms
            avail = "✓" if r in slot["study_rooms"] else "✗"
            print(f"{avail:<10}", end="")
        print()


def book_room(date_str: str, room_id: str, hour: int, minute: int, num_slots: int) -> dict:
    from db import add_booking, is_booked
    
    room_info = LC_ROOMS.get(room_id) or STUDY_ROOMS.get(room_id)
    if not room_info:
        return {"success": False, "error": f"Room {room_id} does not exist."}

    # Check all slots available
    h, m = hour, minute
    slots_to_book = []
    for _ in range(num_slots):
        if h >= LIBRARY_CLOSE:
            return {"success": False, "error": "Booking extends beyond library closing time."}
        slot = get_slot_availability(date_str, h, m)
        if slot.get("closed"):
            return {"success": False, "error": f"Library closed at {h:02d}:{m:02d}."}
        if is_booked(date_str, room_id, h, m):
            return {"success": False, "error": f"{room_id} is already booked at {h:02d}:{m:02d}."}
        room_type = "learning_commons" if room_id in LC_ROOMS else "study_rooms"
        if room_id not in slot.get(room_type, {}):
            return {"success": False, "error": f"{room_id} is not available at {h:02d}:{m:02d}."}
        slots_to_book.append((h, m))
        m += 30
        if m >= 60:
            m = 0
            h += 1

    # Generate reference
    import random
    ref = f"DEMO-{room_id}-{date_str}-{hour:02d}{minute:02d}-{random.randint(1000,9999)}"

    # Write to DB
    success = add_booking(date_str, room_id, hour, minute, num_slots, ref)
    if not success:
        return {"success": False, "error": "Booking failed — slot taken."}

    end_h, end_m = slots_to_book[-1]
    end_m += 30
    if end_m >= 60:
        end_m = 0
        end_h += 1

    return {
        "success": True,
        "room_id": room_id,
        "floor": room_info["floor"],
        "capacity": room_info["capacity"],
        "room_type": room_info["type"],
        "date": date_str,
        "start": f"{slots_to_book[0][0]:02d}:{slots_to_book[0][1]:02d}",
        "end": f"{end_h:02d}:{end_m:02d}",
        "num_slots": num_slots,
        "booking_ref": ref,
    }

def get_room_conflict_info(date_str: str, room_id: str, req_h: int, req_m: int, num_slots: int) -> str:
    """
    For a specific room, describe which requested slots are unavailable and why.
    Returns a human-readable string.
    """
    from db import is_booked
    occupied_sim = _get_room_schedule(date_str, room_id)
    conflicts = []
    h, m = req_h, req_m
    for _ in range(num_slots):
        if h >= LIBRARY_CLOSE:
            conflicts.append(f"{h:02d}:{m:02d} (after closing)")
        elif (h, m) in occupied_sim:
            conflicts.append(f"{h:02d}:{m:02d} (already booked)")
        elif is_booked(date_str, room_id, h, m):
            conflicts.append(f"{h:02d}:{m:02d} (already booked)")
        m += 30
        if m >= 60:
            m = 0
            h += 1
    if conflicts:
        return f"{room_id} is unavailable at: {', '.join(conflicts)}"
    return f"{room_id} is available"

def is_room_session_booked(date_str: str, room_id: str, hour: int, minute: int) -> bool:
    from db import is_booked
    return is_booked(date_str, room_id, hour, minute)

def find_best_room(date_str: str, hour: int, minute: int, num_slots: int,
                   min_capacity: int, preferred_floor: str = None,
                   excluded_floor: str = None) -> dict:
    """
    Find the best available room matching capacity requirement for all requested slots.
    preferred_floor: '1/F', 'LG1', 'LG3', 'LG4', or 'LC' to restrict to that floor/type.
    excluded_floor: same values — exclude rooms on that floor/type.
    Returns the recommended room or alternatives if exact match not found.
    """
    # Get rooms free for ALL requested slots
    h, m = hour, minute
    free_sets = []
    for _ in range(num_slots):
        if h >= LIBRARY_CLOSE:
            break
        slot = get_slot_availability(date_str, h, m)
        lc_free = set(slot.get("learning_commons", {}).keys())
        sr_free = set(slot.get("study_rooms", {}).keys())
        # Exclude session-booked rooms
        lc_free = {r for r in lc_free if not is_room_session_booked(date_str, r, h, m)}
        sr_free = {r for r in sr_free if not is_room_session_booked(date_str, r, h, m)}
        free_sets.append(lc_free | sr_free)
        m += 30
        if m >= 60:
            m = 0
            h += 1

    if not free_sets:
        return {"found": False, "reason": "No slots available"}

    # Rooms free in ALL slots
    continuously_free = free_sets[0]
    for s in free_sets[1:]:
        continuously_free &= s

    # Restrict to preferred floor / room type if requested
    def _on_floor(rid: str, target: str) -> bool:
        inf = LC_ROOMS.get(rid) or STUDY_ROOMS.get(rid)
        if not inf:
            return False
        if target == "LC":
            return rid in LC_ROOMS
        return inf["floor"].upper() == target

    if preferred_floor:
        pf = preferred_floor.upper()
        continuously_free = {r for r in continuously_free if _on_floor(r, pf)}
    if excluded_floor:
        ef = excluded_floor.upper()
        continuously_free = {r for r in continuously_free if not _on_floor(r, ef)}

    # Filter by capacity
    matching = []
    for room_id in continuously_free:
        info = LC_ROOMS.get(room_id) or STUDY_ROOMS.get(room_id)
        if info and info["capacity"] >= min_capacity:
            matching.append((room_id, info))

    if matching:
        # Sort: prefer exact capacity match, then smallest that fits
        matching.sort(key=lambda x: x[1]["capacity"])
        best_room, best_info = matching[0]
        return {
            "found": True,
            "room_id": best_room,
            "capacity": best_info["capacity"],
            "floor": best_info["floor"],
            "room_type": best_info["type"],
            "alternatives": [(r, i) for r, i in matching[1:4]],
        }

    # No exact match — find alternatives
    # Try smaller capacity
    smaller = []
    for room_id in continuously_free:
        info = LC_ROOMS.get(room_id) or STUDY_ROOMS.get(room_id)
        if info:
            smaller.append((room_id, info))
    smaller.sort(key=lambda x: -x[1]["capacity"])  # largest first

    # Try different time slots (±1 hour)
    alt_times = []
    for delta in [-2, -1, 1, 2, 3, 4]:
        alt_h = hour + delta // 2
        alt_m = minute + (delta % 2) * 30
        if alt_m >= 60:
            alt_m -= 60
            alt_h += 1
        if LIBRARY_OPEN <= alt_h < LIBRARY_CLOSE:
            alt_result = find_best_room(date_str, alt_h, alt_m, num_slots, min_capacity)
            if alt_result["found"]:
                alt_times.append((f"{alt_h:02d}:{alt_m:02d}", alt_result["room_id"], alt_result["capacity"]))
            if len(alt_times) >= 2:
                break

    return {
        "found": False,
        "reason": f"No rooms with capacity ≥{min_capacity} free for all {num_slots} slots",
        "smaller_options": [(r, i) for r, i in smaller[:3]],
        "alt_times": alt_times,
    }

def format_booking_confirmation(booking: dict) -> str:
    if not booking["success"]:
        return f"Booking failed: {booking['error']}"
    return (
        f"✅ Booked! {booking['room_id']} ({booking['room_type']}, {booking['floor']}, "
        f"seats {booking['capacity']}) on {booking['date']} "
        f"from {booking['start']} to {booking['end']}.\n"
        f"Reference: {booking['booking_ref']}\n"
        f"Note: This is a demo booking. For real bookings visit https://lbbooking.ust.hk"
    )

def format_room_recommendation(result: dict, date_str: str, start: str, end: str) -> str:
    if not result["found"]:
        lines = [f"No rooms with enough capacity free from {start} to {end}."]
        if result.get("smaller_options"):
            lines.append("Smaller rooms available:")
            for r, info in result["smaller_options"]:
                lines.append(f"  • {r} ({info['type']}, {info['floor']}, seats {info['capacity']})")
        if result.get("alt_times"):
            lines.append("Same capacity available at:")
            for t, r, cap in result["alt_times"]:
                lines.append(f"  • {t} — {r} (seats {cap})")
        return "\n".join(lines)

    alts = ""
    if result.get("alternatives"):
        alt_strs = [f"{r} (seats {i['capacity']})" for r, i in result["alternatives"]]
        alts = f"\nAlternatives: {', '.join(alt_strs)}"

    return (
        f"Best match: {result['room_id']} ({result['room_type']}, "
        f"{result['floor']}, seats {result['capacity']}) "
        f"from {start} to {end} on {date_str}.{alts}"
    )

def print_full_day_text(
    date_str: str = None,
    floor_filter: str = None,
    from_hour: int = None,
    from_minute: int = 0,
    num_slots: int = None,
) -> str:
    """Returns availability as Markdown tables.
    floor_filter: 'LC', '1F', 'LG1', 'LG3', or 'LG4' to show only that section.
    from_hour/from_minute/num_slots: limit to a time window (default: full day).
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    # Build slot list
    all_slots = []
    if from_hour is not None:
        h, m = from_hour, from_minute
        count = num_slots or 30
        for _ in range(count):
            if h >= LIBRARY_CLOSE:
                break
            all_slots.append((h, m))
            m += 30
            if m >= 60:
                m = 0
                h += 1
    else:
        for h in range(LIBRARY_OPEN, LIBRARY_CLOSE):
            all_slots.append((h, 0))
            all_slots.append((h, 30))

    # Normalise floor_filter
    ff = (floor_filter or "").upper().strip()

    # Group study rooms by floor
    rooms_by_floor = {"1/F": [], "LG1": [], "LG3": [], "LG4": []}
    for room_id, info in STUDY_ROOMS.items():
        floor = info["floor"]
        if floor in rooms_by_floor:
            rooms_by_floor[floor].append(room_id)
    for floor in rooms_by_floor:
        rooms_by_floor[floor].sort()

    if from_hour is not None and all_slots:
        end_h, end_m = all_slots[-1]
        end_m += 30
        if end_m >= 60: end_m = 0; end_h += 1
        window = f"{from_hour:02d}:{from_minute:02d}–{end_h:02d}:{end_m:02d}"
        lines = [f"## Availability {window} — {date_str}", ""]
    else:
        lines = [f"## Full Day Availability — {date_str}", ""]
    
    # === Learning Commons Table ===
    lc_rooms = list(LC_ROOMS.keys())
    if not ff or ff in ("LC", "LG1"):
        lines.append("### Learning Commons (LG1)")
        lines.append("")
        header = "| Time | " + " | ".join(lc_rooms) + " |"
        separator = "|------|" + "|".join(["------" for _ in lc_rooms]) + "|"
        lines.append(header)
        lines.append(separator)
        for h, m in all_slots:
            slot = get_slot_availability(date_str, h, m)
            row_icons = []
            for room_id in lc_rooms:
                if slot.get("closed"):
                    row_icons.append("❌")
                elif room_id in slot.get("learning_commons", {}):
                    row_icons.append("✅")
                else:
                    row_icons.append("❌")
            lines.append(f"| {h:02d}:{m:02d} | " + " | ".join(row_icons) + " |")
        lines.append("")
        lines.append("---")
        lines.append("")

    # === Study Rooms by Floor ===
    floor_map = {"1F": "1/F", "1/F": "1/F", "LG1": "LG1", "LG3": "LG3", "LG4": "LG4"}
    for floor, room_ids in rooms_by_floor.items():
        if not room_ids:
            continue
        if ff and floor_map.get(ff, ff) != floor:
            continue
        lines.append(f"### Study Rooms — {floor}")
        lines.append("")
        
        # Header
        header = "| Time | " + " | ".join(room_ids) + " |"
        separator = "|------|" + "|".join(["------" for _ in room_ids]) + "|"
        lines.append(header)
        lines.append(separator)
        
        for h, m in all_slots:
            slot = get_slot_availability(date_str, h, m)
            row_icons = []
            for room_id in room_ids:
                if slot.get("closed"):
                    row_icons.append("❌")
                elif room_id in slot.get("study_rooms", {}):
                    row_icons.append("✅")
                else:
                    row_icons.append("❌")
            time_str = f"{h:02d}:{m:02d}"
            lines.append(f"| {time_str} | " + " | ".join(row_icons) + " |")
        
        lines.append("")
        lines.append("---")
        lines.append("")
    
    # Legend
    lines.append("### Legend")
    lines.append("- ✅ = Available (free to book)")
    lines.append("- ❌ = Booked/Occupied")
    lines.append(f"- Hours: {LIBRARY_OPEN}:00 – {LIBRARY_CLOSE}:00")
    lines.append("- Book at: https://lbbooking.ust.hk")
    
    return "\n".join(lines)

# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    h, m = now.hour, 0 if now.minute < 30 else 30

    print("=== Current slot ===")
    slot = get_slot_availability(date, h, m)
    print(format_slot_for_agent(slot))

    print("\n=== Next 4 slots (2 hours) ===")
    slots = get_slots_range(date, h, m, 4)
    print(format_range_for_agent(slots))

    print("\n=== Rooms free continuously now → 7pm ===")
    result = find_rooms_free_until(date, h, m, 19)
    print(format_continuous_for_agent(result))

    # Uncomment to see full day grid:
    print_full_day(date)