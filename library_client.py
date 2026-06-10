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

def _get_room_schedule(date_str: str, room_id: str) -> set:
    """
    Generate a set of (hour, minute) tuples when this room is OCCUPIED for the day.
    Simulates students booking 1-4 consecutive slots (30min each).
    Returns occupied slots as a set.
    """
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

    available_lc = {}
    available_sr = {}

    for room_id, info in LC_ROOMS.items():
        occupied = _get_room_schedule(date_str, room_id)
        if (hour, minute) not in occupied:
            available_lc[room_id] = info

    for room_id, info in STUDY_ROOMS.items():
        occupied = _get_room_schedule(date_str, room_id)
        if (hour, minute) not in occupied:
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