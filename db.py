import json
import os
from datetime import datetime

DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bookings_db.json")

def _load() -> dict:
    if not os.path.exists(DB_FILE):
        return {"bookings": {}}
    with open(DB_FILE, "r") as f:
        return json.load(f)

def _save(data: dict):
    with open(DB_FILE, "w") as f:
        json.dump(data, f, indent=2)

def add_booking(date_str: str, room_id: str, hour: int, minute: int, num_slots: int, ref: str) -> bool:
    """Mark all slots for a booking as occupied. Returns True if successful."""
    data = _load()
    h, m = hour, minute
    keys_to_add = []
    for _ in range(num_slots):
        key = f"{date_str}_{room_id}_{h}_{m}"
        if key in data["bookings"]:
            return False  # already booked
        keys_to_add.append((key, h, m))
        m += 30
        if m >= 60:
            m = 0
            h += 1

    for key, bh, bm in keys_to_add:
        data["bookings"][key] = {
            "room_id": room_id,
            "date": date_str,
            "hour": bh,
            "minute": bm,
            "ref": ref,
            "booked_at": datetime.now().isoformat(),
        }
    _save(data)
    return True

def is_booked(date_str: str, room_id: str, hour: int, minute: int) -> bool:
    data = _load()
    key = f"{date_str}_{room_id}_{hour}_{minute}"
    return key in data["bookings"]

def get_all_bookings() -> list:
    data = _load()
    return list(data["bookings"].values())

def cancel_booking(ref: str) -> bool:
    """Remove all slots for a booking by reference. Returns True if found and removed."""
    data = _load()
    keys_to_remove = [k for k, v in data["bookings"].items() if v.get("ref") == ref]
    if not keys_to_remove:
        return False
    for k in keys_to_remove:
        del data["bookings"][k]
    _save(data)
    return True

def get_bookings_for_date(date_str: str) -> list:
    data = _load()
    seen_refs = set()
    bookings = []
    for v in data["bookings"].values():
        if v["date"] == date_str and v["ref"] not in seen_refs:
            seen_refs.add(v["ref"])
            bookings.append(v)
    return bookings

def clear_old_bookings():
    """Remove bookings from past dates to keep the file small."""
    data = _load()
    today = datetime.now().strftime("%Y-%m-%d")
    data["bookings"] = {
        k: v for k, v in data["bookings"].items()
        if v["date"] >= today
    }
    _save(data)