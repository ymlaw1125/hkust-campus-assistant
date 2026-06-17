"""
Comprehensive test harness for AgentFrameworkAgent.

Runs the full deterministic logic (booking, editing, cancelling, combined
bus+library planning, slot limits, floor preferences, time parsing) WITHOUT
needing Azure or the live KMB API:

  • The Azure client is never created (we bypass __init__).
  • _call_model is stubbed to echo the [TAGGED] context that would be sent to
    the LLM, so we can assert on what data the agent prepared.
  • The bus ETA fetch is stubbed with deterministic data.

Run:  ./.venv/Scripts/python.exe test_agent.py
"""
import asyncio
import sys

import agent as agent_mod
from agent import AgentFrameworkAgent
import db
import library_client
import sports_client


# ── Stubs ────────────────────────────────────────────────────────────────────
FAKE_ETAS = [
    {"label": "11M",  "etas": [{"minutes": 1}, {"minutes": 16}]},
    {"label": "91M",  "etas": [{"minutes": 5}, {"minutes": 20}]},
    {"label": "792M", "etas": [{"minutes": 8}]},
]


async def fake_get_all_hkust_etas(filter_key="all"):
    return FAKE_ETAS


def fake_format_etas_for_agent(etas):
    return "\n".join(f"{e['label']}: {e['etas'][0]['minutes']} min" for e in etas)


async def stub_call_model(self, user_message, extra_context):
    # Echo the prepared context so tests can inspect what was fed to the LLM.
    return f"<LLM>{extra_context.strip()}</LLM>"


def make_agent():
    """Construct an agent without touching Azure."""
    a = AgentFrameworkAgent.__new__(AgentFrameworkAgent)
    import logging
    a.logger = logging.getLogger("test")
    a.conversation_history = []
    a.pending_booking = None
    a._waiting_duration = False
    a._duration_context = None
    a.my_bookings = []
    a._pending_cancel_ref = None
    a._edit_after_cancel = False
    a._rejected_booking_ctx = None
    a._pending_edit = None
    a.pending_sports = None
    a._waiting_sports_time = None
    a._pending_multi = None
    a._plan_ctx = None
    a.client = None
    a.deployment = "stub"
    return a


# ── Test runner ──────────────────────────────────────────────────────────────
class Runner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.agent = make_agent()

    async def say(self, msg):
        reply = await self.agent.process_user_message(msg, auth=None, auth_handler_name=None, context=None)
        print(f"\n  USER: {msg}")
        print(f"  BOT : {reply}")
        return reply

    def check(self, name, cond):
        mark = "PASS" if cond else "FAIL"
        if cond:
            self.passed += 1
        else:
            self.failed += 1
        print(f"    [{mark}] {name}")
        return cond


def reset_db():
    db._save({"bookings": {}})
    library_client._schedule_cache.clear()
    sports_client._schedule_cache.clear()


async def main():
    # Patch module-level deps
    agent_mod.get_all_hkust_etas = fake_get_all_hkust_etas
    agent_mod.format_etas_for_agent = fake_format_etas_for_agent
    AgentFrameworkAgent._call_model = stub_call_model

    reset_db()
    r = Runner()

    DATE = library_client.datetime.now().strftime("%Y-%m-%d")

    print("=" * 70)
    print("TEST 1 — Bus query routes to live bus data")
    print("=" * 70)
    reply = await r.say("when's the next bus to diamond hill")
    r.check("bus data fetched", "11M" in reply or "91M" in reply)

    print("\n" + "=" * 70)
    print("TEST 2 — Library availability query (no booking)")
    print("=" * 70)
    reply = await r.say("what study rooms are free at 3pm")
    r.check("library data prepared", "LIVE LIBRARY DATA" in reply)
    r.check("no pending booking yet", r.agent.pending_booking is None)

    print("\n" + "=" * 70)
    print("TEST 3 — Combined planning: at Hang Hau, wants a room on arrival")
    print("=" * 70)
    reply = await r.say("im at hang hau now, book me a room when i arrive to study 2 hours")
    r.check("bus plan present (11M)", "11M" in reply)
    r.check("room recommended", r.agent.pending_booking is not None)
    r.check("pending is 2 hours (4 slots)", r.agent.pending_booking and r.agent.pending_booking["num_slots"] == 4)
    arr_h = r.agent.pending_booking["hour"]
    arr_m = r.agent.pending_booking["minute"]
    r.check("arrival rounded to a valid slot", arr_m in (0, 30))
    print(f"    (planned arrival slot: {arr_h:02d}:{arr_m:02d}, room {r.agent.pending_booking['room_id']})")

    print("\n" + "=" * 70)
    print("TEST 4 — Floor preference change while pending")
    print("=" * 70)
    prev_room = r.agent.pending_booking["room_id"]
    cur_floor = (library_client.LC_ROOMS.get(prev_room) or library_client.STUDY_ROOMS.get(prev_room))["floor"]
    n_slots = r.agent.pending_booking["num_slots"]
    # Choose a different floor; whether it's free at the (clock-derived) pending slot
    # varies by date/time, so assert the correct behaviour in BOTH cases.
    target_floor = next((f for f in ["LG1", "LG3", "LG4", "1/F"]
                         if f != cur_floor and
                         library_client.find_best_room(DATE, arr_h, arr_m, n_slots, 1,
                                                       preferred_floor=f)["found"]), None)
    if target_floor is None:
        # Nothing else free at this slot — agent should keep the original booking, not crash.
        reply = await r.say("actually i want one in lg1")
        r.check("kept a valid pending booking when no alt floor is free",
                r.agent.pending_booking is not None)
    else:
        phrase = {"LG1": "lg1", "LG3": "lg3", "LG4": "lg4", "1/F": "1f"}[target_floor]
        reply = await r.say(f"actually i want one in {phrase}")
        new_floor = (library_client.LC_ROOMS.get(r.agent.pending_booking["room_id"]) or
                     library_client.STUDY_ROOMS.get(r.agent.pending_booking["room_id"]))["floor"]
        r.check("repriced to the requested floor", new_floor == target_floor)
    new_room = r.agent.pending_booking["room_id"]
    r.check("same time window kept", r.agent.pending_booking["hour"] == arr_h and r.agent.pending_booking["minute"] == arr_m)

    print("\n" + "=" * 70)
    print("TEST 5 — Confirm booking (deterministic, real reference)")
    print("=" * 70)
    reply = await r.say("ok book it thanks")
    r.check("booking confirmed", "Booked" in reply or "✅" in reply)
    r.check("recorded in my_bookings", len(r.agent.my_bookings) == 1)
    r.check("reference is real (DEMO-)", r.agent.my_bookings[0]["ref"].startswith("DEMO-"))
    booked_room = r.agent.my_bookings[0]["room_id"]
    booked_ref = r.agent.my_bookings[0]["ref"]
    r.check("booked room is the repriced one", booked_room == new_room)
    r.check("slot persisted in db", db.is_booked(DATE, booked_room, arr_h, arr_m))

    print("\n" + "=" * 70)
    print("TEST 6 — View my bookings (per-slot rows)")
    print("=" * 70)
    reply = await r.say("show me my bookings")
    r.check("table header present", "| Room | Start | End | Reference |" in reply)
    r.check("4 slot rows for a 2h booking", reply.count(booked_ref) == 4)

    print("\n" + "=" * 70)
    print("TEST 7 — Daily 4-slot limit enforced")
    print("=" * 70)
    reply = await r.say("book me another room at 8pm for 2 hours")
    r.check("slot limit reached", "SLOT LIMIT REACHED" in reply)
    r.check("no new pending booking", r.agent.pending_booking is None)

    print("\n" + "=" * 70)
    print("TEST 8 — Edit booking to a different floor, same time")
    print("=" * 70)
    # Pick a floor (other than the booked LG1) that actually has a 2h slot today,
    # so the test is robust to the date-seeded availability.
    booked_slots = r.agent.my_bookings[0]["num_slots"]
    target_floor = next(
        (fl for fl in ["LG3", "LG4", "1/F"]
         if library_client.find_best_room(DATE, arr_h, arr_m, booked_slots, 1,
                                          preferred_floor=fl)["found"]),
        None,
    )
    if target_floor is None:
        r.check("(skipped — no alternative floor has a 2h slot at this hour today)", True)
    else:
        floor_phrase = {"LG3": "lg3", "LG4": "lg4", "1/F": "1f"}[target_floor]
        reply = await r.say(f"change my booking to a {floor_phrase} room")
        r.check("pending edit created", r.agent._pending_edit is not None)
        r.check("edit target on chosen floor",
                r.agent._pending_edit and
                (library_client.LC_ROOMS.get(r.agent._pending_edit["room_id"]) or
                 library_client.STUDY_ROOMS.get(r.agent._pending_edit["room_id"]))["floor"] == target_floor)
        r.check("edit keeps same time", r.agent._pending_edit and r.agent._pending_edit["hour"] == arr_h)
        reply = await r.say("yes")
        r.check("edit applied", "Changed" in reply)
        r.check("still one booking", len(r.agent.my_bookings) == 1)
        r.check("new room on chosen floor",
                (library_client.LC_ROOMS.get(r.agent.my_bookings[0]["room_id"]) or
                 library_client.STUDY_ROOMS.get(r.agent.my_bookings[0]["room_id"]))["floor"] == target_floor)
        r.check("old LG1 slot released in db", not db.is_booked(DATE, booked_room, arr_h, arr_m))

    print("\n" + "=" * 70)
    print("TEST 9 — Cancel booking")
    print("=" * 70)
    reply = await r.say("cancel my booking")
    r.check("cancel confirmation asked", "Reply yes" in reply or "confirm" in reply.lower())
    reply = await r.say("yes")
    r.check("cancelled message", "Cancelled" in reply or "removed" in reply.lower())
    r.check("my_bookings empty", len(r.agent.my_bookings) == 0)
    r.check("db slot freed", len(db.get_bookings_for_date(DATE)) == 0)

    print("\n" + "=" * 70)
    print("TEST 10 — Specific-room booking with explicit time range")
    print("=" * 70)
    reset_db()
    r.agent.my_bookings = []
    r.agent.pending_booking = None
    # pick a room guaranteed free
    free = library_client.find_best_room(DATE, 15, 0, 2, 1)
    target = free["room_id"]
    reply = await r.say(f"book {target} from 3pm to 4pm")
    r.check("recommendation for the exact room", r.agent.pending_booking and r.agent.pending_booking["room_id"] == target)
    r.check("start parsed as 15:00", r.agent.pending_booking and r.agent.pending_booking["hour"] == 15)
    r.check("duration 1h (2 slots)", r.agent.pending_booking and r.agent.pending_booking["num_slots"] == 2)
    await r.say("yes")
    r.check("booked", len(r.agent.my_bookings) == 1)

    print("\n" + "=" * 70)
    print("TEST 11 — Duration follow-up ('book a room at 7' then '1 hour')")
    print("=" * 70)
    reset_db()
    r.agent.my_bookings = []
    r.agent.pending_booking = None
    reply = await r.say("book me a study room at 7")
    r.check("asks for duration", "DURATION NEEDED" in reply)
    r.check("waiting-duration flag set", r.agent._waiting_duration is True)
    reply = await r.say("1 hour")
    r.check("recommendation after duration", r.agent.pending_booking is not None)
    r.check("start parsed as 19:00 (PM heuristic)", r.agent.pending_booking and r.agent.pending_booking["hour"] == 19)
    r.check("2 slots for 1 hour", r.agent.pending_booking and r.agent.pending_booking["num_slots"] == 2)

    print("\n" + "=" * 70)
    print("TEST 12 — $DATA returns a bounded table")
    print("=" * 70)
    reply = await r.say("$DATA LG3")
    r.check("LG3 table returned", "Study Rooms — LG3" in reply)
    r.check("response bounded (< 4000 chars)", len(reply) < 4000)
    r.check("only LG3 (no LC table)", "Learning Commons" not in reply)

    print("\n" + "=" * 70)
    print("TEST 13 — Reject a suggestion, then ask differently (no false cancel)")
    print("=" * 70)
    reset_db()
    r.agent.my_bookings = []
    r.agent.pending_booking = None
    await r.say("book me a room at 4pm for 1 hour")
    had_pending = r.agent.pending_booking is not None
    reply = await r.say("no i dont like that floor, anywhere else")
    r.check("had a pending suggestion", had_pending)
    # Correct behaviour: fall through with a NOTE, NOT the old hard-coded cancel sentence.
    r.check("fell through (NOTE present)", "[NOTE]" in reply)
    r.check("did not hard-return the cancel sentence",
            "let me know if you'd like a different room or time" not in reply.lower())

    def _free_sport_hour(sport, lo=8, hi=22):
        for h in range(lo, hi):
            if sports_client.find_available_venue(DATE, sport, h)["status"] == "ok":
                return h
        return None

    print("\n" + "=" * 70)
    print("TEST 14 — Walk-in sport (swimming) returns info, no booking")
    print("=" * 70)
    reset_db()
    r.agent.my_bookings = []
    r.agent.pending_sports = None
    reply = await r.say("i want to go swimming later")
    r.check("walk-in info given", "turn up" in reply.lower() or "walk-in" in reply.lower())
    r.check("no pending sports booking", r.agent.pending_sports is None)

    print("\n" + "=" * 70)
    print("TEST 15 — 'where can I play tennis' lists venues")
    print("=" * 70)
    reply = await r.say("where can i play tennis")
    r.check("lists the student tennis courts", "Tennis Courts No. 1 & 2" in reply)
    r.check("no booking started", r.agent.pending_sports is None)

    print("\n" + "=" * 70)
    print("TEST 16 — Book a badminton court at a specific time")
    print("=" * 70)
    reset_db()
    r.agent.my_bookings = []
    r.agent.pending_sports = None
    bh = _free_sport_hour("badminton", 9, 22)
    r.check("a free badminton hour exists", bh is not None)
    reply = await r.say(f"book a badminton court at {bh}:00")
    r.check("venue recommended", r.agent.pending_sports is not None)
    r.check("asks to confirm", "shall i book" in reply.lower())
    reply = await r.say("yes")
    r.check("sports booking confirmed", "Booked" in reply or "✅" in reply)
    r.check("recorded in my_bookings", len(r.agent.my_bookings) == 1)
    r.check("booking is a sports kind", r.agent.my_bookings[0].get("kind") == "sports")
    fac_id = r.agent.my_bookings[0]["fac_id"]
    r.check("slot persisted in db", db.is_booked(DATE, fac_id, bh, 0))
    booked_name = r.agent.my_bookings[0]["room_id"]

    print("\n" + "=" * 70)
    print("TEST 17 — Sport with no time asks for one, then completes")
    print("=" * 70)
    reset_db()
    r.agent.my_bookings = []
    r.agent.pending_sports = None
    reply = await r.say("book a squash court")
    r.check("asks for a time", "what time" in reply.lower())
    r.check("waiting-sports-time flag set", r.agent._waiting_sports_time is not None)
    sh = _free_sport_hour("squash", 8, 22)
    reply = await r.say(f"{sh}:00")
    r.check("recommendation after time given", r.agent.pending_sports is not None)
    r.check("squash venue picked", r.agent.pending_sports["facility"] == "SQUASH-LG4")

    print("\n" + "=" * 70)
    print("TEST 18 — 1 slot per SPORT per day (all venues share the quota)")
    print("=" * 70)
    reset_db()
    r.agent.my_bookings = []
    r.agent.pending_sports = None
    free_hrs = [h for h in range(8, 22)
                if sports_client.find_available_venue(DATE, "squash", h)["status"] == "ok"]
    r.check("at least two free squash hours", len(free_hrs) >= 2)
    await r.say(f"book squash at {free_hrs[0]}:00")
    await r.say("yes")
    r.check("first squash booking made", len(r.agent.my_bookings) == 1)
    reply = await r.say(f"book squash at {free_hrs[1]}:00")
    r.check("second squash booking blocked", "one slot per sport per day" in reply.lower())
    r.check("no second pending", r.agent.pending_sports is None)
    # Football: both pitches are the SAME sport → second pitch also blocked
    reset_db()
    r.agent.my_bookings = []
    r.agent.pending_sports = None
    fh = _free_sport_hour("football", 8, 22)
    await r.say(f"book the turf soccer pitch at {fh}:00")
    await r.say("yes")
    first_pitch = r.agent.my_bookings[0]["fac_id"]
    reply = await r.say(f"book the mini soccer pitch at {fh}:00")
    r.check("second soccer pitch blocked (same sport)", "one slot per sport per day" in reply.lower())

    print("\n" + "=" * 70)
    print("TEST 18b — Different sports are independent (TT + badminton both allowed)")
    print("=" * 70)
    reset_db()
    r.agent.my_bookings = []
    r.agent.pending_sports = None
    tth = _free_sport_hour("table_tennis", 9, 22)
    await r.say(f"book table tennis at {tth}:00")
    await r.say("yes")
    bdh = _free_sport_hour("badminton", 9, 22)
    reply = await r.say(f"book badminton at {bdh}:00")
    r.check("badminton NOT blocked by the table-tennis booking", r.agent.pending_sports is not None)
    await r.say("yes")
    r.check("both sports booked", len(r.agent.my_bookings) == 2)

    print("\n" + "=" * 70)
    print("TEST 19 — $SPORTS returns a bounded table")
    print("=" * 70)
    reply = await r.say("$SPORTS badminton")
    r.check("sports table returned", "Sports Availability" in reply and "Badminton" in reply)
    r.check("response bounded (< 4000 chars)", len(reply) < 4000)

    print("\n" + "=" * 70)
    print("TEST 20 — Sports booking shows in my bookings, then cancel")
    print("=" * 70)
    reset_db()
    r.agent.my_bookings = []
    r.agent.pending_sports = None
    bh2 = _free_sport_hour("basketball", 9, 22)
    await r.say(f"book basketball at {bh2}:00")
    await r.say("yes")
    reply = await r.say("show my bookings")
    r.check("table header present", "| Room | Start | End | Reference |" in reply)
    r.check("sports facility listed", r.agent.my_bookings[0]["room_id"] in reply)
    reply = await r.say("cancel my booking")
    r.check("cancel confirmation asked", "reply yes" in reply.lower())
    reply = await r.say("yes")
    r.check("cancelled", "Cancelled" in reply or "removed" in reply.lower())
    r.check("my_bookings empty", len(r.agent.my_bookings) == 0)

    print("\n" + "=" * 70)
    print("TEST 21 — Combined bus + sports planning from a broad HK origin")
    print("=" * 70)
    reset_db()
    r.agent.my_bookings = []
    r.agent.pending_sports = None
    r.agent.pending_booking = None
    r.agent._plan_ctx = None
    reply = await r.say("im at san po kong now, going to school to play badminton, book me a court at lg1")
    r.check("bus plan present (route shown)", "91M" in reply or "11" in reply or "792M" in reply)
    r.check("did NOT fall into library", r.agent.pending_booking is None)
    r.check("sports court held", r.agent.pending_sports is not None)
    r.check("badminton chosen", r.agent.pending_sports and r.agent.pending_sports["sport"] == "badminton")
    r.check("lg1 venue (SHHO) honoured", r.agent.pending_sports and r.agent.pending_sports["facility"] == "SHHO")
    r.check("plan context remembered", r.agent._plan_ctx and r.agent._plan_ctx["origin"] == "san po kong")
    reply = await r.say("yes")
    r.check("sports booking confirmed", "Booked" in reply or "✅" in reply)
    r.check("one sports booking", len(r.agent.my_bookings) == 1 and r.agent.my_bookings[0].get("kind") == "sports")

    print("\n" + "=" * 70)
    print("TEST 22 — 'book one when I arrive' reuses origin + sport (no ping-pong)")
    print("=" * 70)
    reset_db()
    r.agent.my_bookings = []
    r.agent.pending_sports = None
    r.agent.pending_booking = None
    r.agent._plan_ctx = None
    await r.say("im at choi hung, heading to campus to play table tennis")
    first_pending = r.agent.pending_sports is not None
    r.agent.pending_sports = None  # pretend the user ignored the first suggestion
    reply = await r.say("actually book one when i arrive")
    r.check("first turn recommended a venue", first_pending)
    r.check("did NOT ask library duration", "how long" not in reply.lower())
    r.check("re-recommended a sports venue", r.agent.pending_sports is not None)
    r.check("kept table tennis from context",
            r.agent.pending_sports and r.agent.pending_sports["sport"] == "table_tennis")

    print("\n" + "=" * 70)
    print("TEST 23 — Combined bus + library planning still works (no sport)")
    print("=" * 70)
    reset_db()
    r.agent.my_bookings = []
    r.agent.pending_sports = None
    r.agent.pending_booking = None
    r.agent._plan_ctx = None
    reply = await r.say("im at mong kok, going to school, book me a study room when i arrive for 1 hour")
    r.check("bus plan present", "91M" in reply)
    r.check("library room held (not sports)", r.agent.pending_booking is not None)
    r.check("no sports pending", r.agent.pending_sports is None)
    r.check("1 hour = 2 slots", r.agent.pending_booking and r.agent.pending_booking["num_slots"] == 2)

    print("\n" + "=" * 70)
    print("TEST 24 — Honest routing from far origin (no hallucinated direct bus/ETA)")
    print("=" * 70)
    reset_db()
    r.agent.my_bookings = []
    r.agent.pending_sports = None
    r.agent.pending_booking = None
    r.agent._plan_ctx = None
    reply = await r.say("im at central, going to school to play badminton when i arrive")
    r.check("routes via an interchange (MTR transfer shown)", "MTR to" in reply)
    r.check("does NOT claim a precise next-bus ETA from a far origin", "in ~" not in reply)
    r.check("still recommends a court", r.agent.pending_sports is not None)

    print("\n" + "=" * 70)
    print("TEST 25 — Chained booking: study room starting after a sports booking")
    print("=" * 70)
    reset_db()
    r.agent.my_bookings = []
    r.agent.pending_sports = None
    r.agent.pending_booking = None
    r.agent._plan_ctx = None
    chosen = None
    for bh in range(9, 20):
        if sports_client.find_available_venue(DATE, "badminton", bh)["status"] != "ok":
            continue
        sh = bh + 1
        fl = next((f for f in ["LG3", "LG4", "LG1", "1/F"]
                   if library_client.find_best_room(DATE, sh, 0, 4, 1, preferred_floor=f)["found"]), None)
        if fl and sh + 2 <= 22:
            chosen = (bh, sh, fl)
            break
    r.check("found a chainable badminton+study slot", chosen is not None)
    bh, sh, fl = chosen
    await r.say(f"book badminton at {bh}:00")
    await r.say("yes")
    r.check("badminton booked", len(r.agent.my_bookings) == 1)
    floor_phrase = {"LG3": "lg3", "LG4": "lg4", "LG1": "lg1", "1/F": "1f"}[fl]
    reply = await r.say(f"after i finish badminton, i wanna study at {floor_phrase} for 2 hours")
    r.check("did NOT re-trigger a sports booking", r.agent.pending_sports is None)
    r.check("study room recommended (library pending)", r.agent.pending_booking is not None)
    r.check("starts when badminton ends", r.agent.pending_booking and r.agent.pending_booking["hour"] == sh)
    r.check("2 hours = 4 slots", r.agent.pending_booking and r.agent.pending_booking["num_slots"] == 4)
    r.check("on the requested floor",
            (library_client.LC_ROOMS.get(r.agent.pending_booking["room_id"]) or
             library_client.STUDY_ROOMS.get(r.agent.pending_booking["room_id"]))["floor"] == fl)
    reply = await r.say("yes")
    r.check("study room booked", "Booked" in reply or "✅" in reply)
    r.check("now two bookings (sports + study)", len(r.agent.my_bookings) == 2)

    print("\n" + "=" * 70)
    print("TEST 26 — Planning honors a named venue ('multi purpose room')")
    print("=" * 70)
    reset_db()
    r.agent.my_bookings = []
    r.agent.pending_sports = None
    r.agent.pending_booking = None
    r.agent._plan_ctx = None
    reply = await r.say("im in north point, going to ust to play table tennis. "
                        "book me a slot when i arrive in the multi purpose room")
    r.check("picked the multi-purpose table-tennis venue (TST)",
            r.agent.pending_sports and r.agent.pending_sports["facility"] == "TT-TST")

    print("\n" + "=" * 70)
    print("TEST 27 — A word containing 'no' ('north') does NOT false-cancel")
    print("=" * 70)
    # A pending sports booking must survive a message that merely contains 'north'.
    r.check("a booking is pending", r.agent.pending_sports is not None)
    fac_before = r.agent.pending_sports["facility"]
    reply = await r.say("yes from north point that works, book it")
    r.check("did not get false-cancelled / library cascade", "I haven't booked anything" not in reply)
    r.check("sports booking actually completed", "Booked" in reply or "✅" in reply)
    r.check("booked the multi-purpose venue", r.agent.my_bookings[0]["fac_id"] == fac_before)
    r.check("exactly one slot (1 hour) — not 2", r.agent.my_bookings[0]["num_slots"] == 1)
    r.check("end is one hour after start",
            int(r.agent.my_bookings[0]["end"][:2]) - int(r.agent.my_bookings[0]["start"][:2]) == 1)

    print("\n" + "=" * 70)
    print("TEST 28 — Correct a pending venue mid-flow, no false cancel")
    print("=" * 70)
    reset_db()
    r.agent.my_bookings = []
    r.agent.pending_sports = None
    # Find an hour where BOTH table-tennis venues are free, so we can switch between them
    tt_h = next((h for h in range(8, 22)
                 if sports_client.free_courts(DATE, "TT-LG1031", "table_tennis", 6, h) > 0
                 and sports_client.free_courts(DATE, "TT-TST", "table_tennis", 4, h) > 0), None)
    r.check("an hour with both TT venues free exists", tt_h is not None)
    await r.say(f"book table tennis room lg1031 at {tt_h}:00")
    r.check("first pick is the LG1031 room", r.agent.pending_sports["facility"] == "TT-LG1031")
    reply = await r.say("no i said the multi purpose room")  # 'no' + a venue correction
    r.check("switched venue (not cancelled)", r.agent.pending_sports is not None)
    r.check("now on the multi-purpose venue", r.agent.pending_sports["facility"] == "TT-TST")
    r.check("kept the same hour", r.agent.pending_sports["hour"] == tt_h)

    print("\n" + "=" * 70)
    print("TEST 29 — Internal control tags never leak to the user")
    print("=" * 70)
    leaky = "[BOOKING CONFIRMED] Ref: TT-06-001. [NOTE] hi [SLOT LIMIT REACHED]"
    cleaned = agent_mod._INTERNAL_TAGS.sub("", leaky).strip()
    r.check("BOOKING CONFIRMED stripped", "BOOKING CONFIRMED" not in cleaned)
    r.check("NOTE stripped", "[NOTE]" not in cleaned)
    r.check("SLOT LIMIT REACHED stripped", "SLOT LIMIT REACHED" not in cleaned)

    print("\n" + "=" * 70)
    print("TEST 30 — Full sport time → 'yes' books the offered alternative (no LLM)")
    print("=" * 70)
    reset_db()
    r.agent.my_bookings = []
    r.agent.pending_sports = None
    # Find an hour where ALL badminton venues are full, but some other hour is free
    full_h = None
    for h in range(9, 21):
        venues = sports_client.SPORT_VENUES["badminton"]
        if all(sports_client.free_courts(DATE, v["id"], "badminton", v["courts"], h) == 0
               for v in venues):
            full_h = h
            break
    if full_h is not None:
        reply = await r.say(f"book badminton at {full_h}:00")
        r.check("told it's full", "fully booked" in reply.lower() or "is full" in reply.lower())
        r.check("an alternative is pending (so 'yes' works)", r.agent.pending_sports is not None)
        # Crucial: confirming must NOT hit the model stub — it books deterministically.
        reply = await r.say("yes")
        r.check("alternative booked deterministically", "Booked" in reply or "✅" in reply)
        r.check("did not fall through to the LLM stub", "<LLM>" not in reply)
        r.check("recorded one sports booking", len(r.agent.my_bookings) == 1)
    else:
        r.check("(skipped — no fully-booked badminton hour today)", True)

    print("\n" + "=" * 70)
    print("TEST 31 — Misspelled sport ('badinton') still routes to sports")
    print("=" * 70)
    reset_db()
    r.agent.my_bookings = []
    r.agent.pending_sports = None
    r.agent.pending_booking = None
    bh = _free_sport_hour("badminton", 9, 22)
    reply = await r.say(f"book badinton at {bh}:00")
    r.check("misspelling recognized as a sports request", r.agent.pending_sports is not None)
    r.check("resolved to badminton", r.agent.pending_sports and r.agent.pending_sports["sport"] == "badminton")

    print("\n" + "=" * 70)
    print("TEST 32 — Sports booking does NOT consume the library 2-hour limit")
    print("=" * 70)
    reset_db()
    r.agent.my_bookings = []
    r.agent.pending_sports = None
    r.agent.pending_booking = None
    sh = _free_sport_hour("squash", 8, 22)
    await r.say(f"book squash at {sh}:00")
    await r.say("yes")
    r.check("one sports booking", len(r.agent.my_bookings) == 1)
    lh = next((h for h in range(8, 21)
               if library_client.find_best_room(DATE, h, 0, 4, 1)["found"]), None)
    reply = await r.say(f"book a study room at {lh}:00 for 2 hours")
    r.check("library NOT blocked by the sports booking", "SLOT LIMIT" not in reply.upper())
    r.check("library room recommended", r.agent.pending_booking is not None)
    await r.say("yes")
    r.check("now two bookings (sport + 2h library)", len(r.agent.my_bookings) == 2)

    print("\n" + "=" * 70)
    print("TEST 33 — 'cancel the pingpong and book badminton' swaps in one go")
    print("=" * 70)
    reset_db()
    r.agent.my_bookings = []
    r.agent.pending_sports = None
    r.agent.pending_booking = None
    th = next((h for h in range(9, 21)
               if sports_client.free_courts(DATE, "TT-LG1031", "table_tennis", 6, h) > 0
               and (sports_client.free_courts(DATE, "SHHO", "badminton", 4, h) > 0
                    or sports_client.free_courts(DATE, "ARENA", "badminton", 4, h) > 0)), None)
    r.check("found an hour with TT and badminton free", th is not None)
    await r.say(f"book table tennis at {th}:00")
    await r.say("yes")
    r.check("table tennis booked", r.agent.my_bookings and r.agent.my_bookings[0]["sport"] == "table_tennis")
    reply = await r.say("cancel the pingpong and book badminton")
    r.check("cancelled the table-tennis booking", "Cancelled" in reply)
    r.check("offered a badminton court (pending)", r.agent.pending_sports is not None)
    r.check("new pending sport is badminton", r.agent.pending_sports and r.agent.pending_sports["sport"] == "badminton")
    r.check("table-tennis booking removed", all(b.get("sport") != "table_tennis" for b in r.agent.my_bookings))
    await r.say("yes")
    r.check("badminton now booked", any(b.get("sport") == "badminton" for b in r.agent.my_bookings))

    print("\n" + "=" * 70)
    print("TEST 34 — 'study at lg3 after pingpong' chains deterministically (no LLM)")
    print("=" * 70)
    reset_db()
    r.agent.my_bookings = []
    r.agent.pending_sports = None
    r.agent.pending_booking = None
    th = next((h for h in range(9, 20)
               if sports_client.find_available_venue(DATE, "table_tennis", h)["status"] == "ok"
               and library_client.find_best_room(DATE, h + 1, 0, 4, 1, preferred_floor="LG3")["found"]
               and h + 3 <= 22), None)
    r.check("found a chainable TT+LG3 slot", th is not None)
    await r.say(f"book table tennis at {th}:00")
    await r.say("yes")
    reply = await r.say("i wanna sstudy at lg3 after pingpong")
    r.check("handled deterministically (no LLM stub)", "<LLM>" not in reply)
    r.check("library room recommended", r.agent.pending_booking is not None)
    r.check("starts when table tennis ends", r.agent.pending_booking and r.agent.pending_booking["hour"] == th + 1)
    r.check("on LG3",
            r.agent.pending_booking and
            (library_client.LC_ROOMS.get(r.agent.pending_booking["room_id"]) or
             library_client.STUDY_ROOMS.get(r.agent.pending_booking["room_id"]))["floor"] == "LG3")
    reply = await r.say("yes thanks")
    r.check("real DEMO reference (not hallucinated)", "DEMO-" in reply)
    r.check("two bookings: table tennis + study", len(r.agent.my_bookings) == 2)

    print("\n" + "=" * 70)
    print("TEST 35 — Multi-chain: 'after badminton, table tennis, then study 1h at lg3'")
    print("=" * 70)
    reset_db()
    r.agent.my_bookings = []
    r.agent.pending_sports = None
    r.agent.pending_booking = None
    r.agent._pending_multi = None
    # Need: badminton@bh free; table tennis@bh+1 free; LG3 free 1h at bh+2.
    chain = None
    for bh in range(9, 18):
        if sports_client.find_available_venue(DATE, "badminton", bh)["status"] != "ok":
            continue
        if sports_client.find_available_venue(DATE, "table_tennis", bh + 1)["status"] != "ok":
            continue
        if not library_client.find_best_room(DATE, bh + 2, 0, 2, 1, preferred_floor="LG3")["found"]:
            continue
        chain = bh
        break
    r.check("found a viable badminton→TT→study chain", chain is not None)
    bh = chain
    await r.say(f"book badminton at {bh}:00")
    await r.say("yes")
    reply = await r.say("after badminton, i wanna play table tennis, and then study for an hour at lg3")
    r.check("built a multi-step plan", r.agent._pending_multi is not None)
    multi_plan = (r.agent._pending_multi or {}).get("plan", [])
    r.check("plan has 2 new items", len(multi_plan) == 2)
    p_sport = next((p for p in multi_plan if p["kind"] == "sports"), None)
    p_lib = next((p for p in multi_plan if p["kind"] == "library"), None)
    r.check("table tennis right after badminton", p_sport and p_sport["sport"] == "table_tennis" and p_sport["hour"] == bh + 1)
    r.check("study right after table tennis", p_lib and p_lib["hour"] == bh + 2)
    r.check("study is 1 hour (2 slots), not 2h", p_lib and p_lib["num_slots"] == 2)
    r.check("study on LG3", p_lib and p_lib["floor"] == "LG3")
    reply = await r.say("yes")
    r.check("all three booked (badminton + TT + study)", len(r.agent.my_bookings) == 3)
    r.check("real refs, no hallucination", reply.count("DEMO-") == 2)
    r.check("no LLM fallback used", "<LLM>" not in reply)

    print("\n" + "=" * 70)
    print("TEST 36 — Sport swap: 'instead of table tennis, I'll play basketball'")
    print("=" * 70)
    reset_db()
    r.agent.my_bookings = []
    r.agent.pending_sports = None
    r.agent.pending_booking = None
    def _bball_free(h):
        return any(sports_client.free_courts(DATE, v["id"], "basketball", v["courts"], h) > 0
                   for v in sports_client.SPORT_VENUES["basketball"])
    sh = next((h for h in range(9, 21)
               if sports_client.find_available_venue(DATE, "table_tennis", h)["status"] == "ok"
               and _bball_free(h)), None)
    r.check("found hour with TT and basketball free", sh is not None)
    await r.say(f"book table tennis at {sh}:00")
    await r.say("yes")
    r.check("table tennis booked", r.agent.my_bookings[0]["sport"] == "table_tennis")
    reply = await r.say("actually instead of table tennis, ill play basketball")
    r.check("told it cancelled the table tennis", "Cancelled" in reply)
    r.check("now offering basketball (pending)", r.agent.pending_sports is not None)
    r.check("pending sport is basketball", r.agent.pending_sports and r.agent.pending_sports["sport"] == "basketball")
    r.check("kept the same hour", r.agent.pending_sports and r.agent.pending_sports["hour"] == sh)
    r.check("table tennis removed", all(b.get("sport") != "table_tennis" for b in r.agent.my_bookings))
    reply = await r.say("yes")
    r.check("basketball booked", any(b.get("sport") == "basketball" for b in r.agent.my_bookings))
    r.check("did not double-book (still one sports booking)",
            sum(1 for b in r.agent.my_bookings if b.get("kind") == "sports") == 1)

    print("\n" + "=" * 70)
    print("TEST 37 — Multi-plan: 'same place' venue + revise one item, keep the rest")
    print("=" * 70)
    reset_db()
    r.agent.my_bookings = []
    r.agent.pending_sports = None
    r.agent.pending_booking = None
    r.agent._pending_multi = None
    # badminton@h at SHHO; basketball@h+1 free at SHHO; LG3 free 2h at h+2
    h0 = None
    for h in range(9, 17):
        if sports_client.free_courts(DATE, "SHHO", "badminton", 4, h) <= 0:
            continue
        if sports_client.free_courts(DATE, "SHHO", "basketball", 1, h + 1) <= 0:
            continue
        if not library_client.find_best_room(DATE, h + 2, 0, 4, 1, preferred_floor="LG3")["found"]:
            continue
        h0 = h
        break
    r.check("found SHHO badminton+basketball+LG3 chain", h0 is not None)
    await r.say(f"book badminton at sports hall at {h0}:00")
    await r.say("yes")
    r.check("badminton at SHHO", r.agent.my_bookings[0]["fac_id"] == "SHHO")
    reply = await r.say("after that i want to play basketball at the same place, then study 2 hours at lg3")
    plan = (r.agent._pending_multi or {}).get("plan", [])
    bball = next((p for p in plan if p.get("sport") == "basketball"), None)
    r.check("'same place' put basketball at SHHO", bball and bball["facility"] == "SHHO")
    # Now revise ONLY the study floor; basketball + times must be preserved
    reply = await r.say("actually make the study at lg4")
    plan2 = (r.agent._pending_multi or {}).get("plan", [])
    r.check("plan still intact (2 items)", len(plan2) == 2)
    bball2 = next((p for p in plan2 if p.get("sport") == "basketball"), None)
    lib2 = next((p for p in plan2 if p["kind"] == "library"), None)
    r.check("basketball venue preserved (SHHO)", bball2 and bball2["facility"] == "SHHO")
    r.check("study floor revised to LG4", lib2 and lib2["floor"] == "LG4")
    r.check("study still 2h (4 slots)", lib2 and lib2["num_slots"] == 4)
    reply = await r.say("yes")
    r.check("all booked (badminton + basketball + study)", len(r.agent.my_bookings) == 3)

    def _setup_chain():
        """Book badminton@SHHO and stage a basketball@SHHO + LG3-study plan; return (h0, plan)."""
        reset_db()
        r.agent.my_bookings = []
        r.agent.pending_sports = None
        r.agent.pending_booking = None
        r.agent._pending_multi = None
        h0 = None
        for h in range(9, 16):
            if sports_client.free_courts(DATE, "SHHO", "badminton", 4, h) <= 0:
                continue
            if sports_client.free_courts(DATE, "SHHO", "basketball", 1, h + 1) <= 0:
                continue
            if not library_client.find_best_room(DATE, h + 2, 0, 4, 1, preferred_floor="LG3")["found"]:
                continue
            if not library_client.find_best_room(DATE, h + 2, 0, 2, 1, preferred_floor="LG3")["found"]:
                continue
            h0 = h
            break
        return h0

    print("\n" + "=" * 70)
    print("TEST 38 — Revise study DURATION mid-plan (2h → 1h), keep the rest")
    print("=" * 70)
    h0 = _setup_chain()
    r.check("found a setup hour for duration test", h0 is not None)
    await r.say(f"book badminton at sports hall at {h0}:00")
    await r.say("yes")
    await r.say("after that basketball at the same place, then study 2 hours at lg3")
    plan = (r.agent._pending_multi or {}).get("plan", [])
    lib0 = next((p for p in plan if p["kind"] == "library"), None)
    r.check("study starts as 2h (4 slots)", lib0 and lib0["num_slots"] == 4)
    await r.say("actually make the study just 1 hour instead")
    plan2 = (r.agent._pending_multi or {}).get("plan", [])
    bball = next((p for p in plan2 if p.get("sport") == "basketball"), None)
    lib = next((p for p in plan2 if p["kind"] == "library"), None)
    r.check("study now 1 hour (2 slots)", lib and lib["num_slots"] == 2)
    r.check("study still on LG3", lib and lib["floor"] == "LG3")
    r.check("basketball preserved at SHHO", bball and bball["facility"] == "SHHO" and bball["hour"] == h0 + 1)
    r.check("study still starts right after basketball", lib and lib["hour"] == h0 + 2)
    await r.say("yes")
    r.check("all three booked", len(r.agent.my_bookings) == 3)

    print("\n" + "=" * 70)
    print("TEST 39 — Revise an item's TIME mid-plan ('start the study at HH')")
    print("=" * 70)
    h0 = _setup_chain()
    r.check("found a setup hour for time test", h0 is not None)
    fh = next((h for h in range(h0 + 3, 21)
               if library_client.find_best_room(DATE, h, 0, 2, 1, preferred_floor="LG3")["found"]), None)
    r.check("found a later free study hour", fh is not None)
    await r.say(f"book badminton at sports hall at {h0}:00")
    await r.say("yes")
    await r.say("after that basketball at the same place, then study an hour at lg3")
    await r.say(f"actually start the study at {fh}:00")
    plan3 = (r.agent._pending_multi or {}).get("plan", [])
    bball = next((p for p in plan3 if p.get("sport") == "basketball"), None)
    lib = next((p for p in plan3 if p["kind"] == "library"), None)
    r.check("study moved to the requested time", lib and lib["hour"] == fh)
    r.check("basketball preserved at SHHO right after badminton",
            bball and bball["facility"] == "SHHO" and bball["hour"] == h0 + 1)
    await r.say("yes")
    r.check("both booked after time change", len(r.agent.my_bookings) == 3)

    print("\n" + "=" * 70)
    print("TEST 40 — No time given → agent offers available slots to choose")
    print("=" * 70)
    reset_db()
    r.agent.my_bookings = []
    r.agent.pending_sports = None
    reply = await r.say("book a badminton court")
    r.check("offers free slots to pick", "Free 1-hour slots" in reply and "•" in reply)
    r.check("waiting for a time", r.agent._waiting_sports_time is not None)
    fhh = _free_sport_hour("badminton", 9, 22)
    reply = await r.say(f"{fhh}:00")
    r.check("picking a listed time books it", r.agent.pending_sports is not None)

    print("\n" + "=" * 70)
    print("TEST 41 — Swap the study/basketball times in a pending plan")
    print("=" * 70)
    reset_db()
    r.agent.my_bookings = []
    r.agent.pending_sports = None
    r.agent.pending_booking = None
    r.agent._pending_multi = None
    hs = None
    for h in range(9, 15):
        if sports_client.free_courts(DATE, "SHHO", "badminton", 4, h) <= 0:
            continue
        if sports_client.free_courts(DATE, "SHHO", "basketball", 1, h + 1) <= 0:
            continue
        if sports_client.free_courts(DATE, "SHHO", "basketball", 1, h + 2) <= 0:
            continue
        if not library_client.find_best_room(DATE, h + 1, 0, 2, 1, preferred_floor="LG3")["found"]:
            continue
        if not library_client.find_best_room(DATE, h + 2, 0, 2, 1, preferred_floor="LG3")["found"]:
            continue
        hs = h
        break
    r.check("found a swap-test setup", hs is not None)
    await r.say(f"book badminton at sports hall at {hs}:00")
    await r.say("yes")
    await r.say("after that basketball at the same place, then study an hour at lg3")
    await r.say("swap the study and basketball times")
    plan = (r.agent._pending_multi or {}).get("plan", [])
    bball = next((p for p in plan if p.get("sport") == "basketball"), None)
    lib = next((p for p in plan if p["kind"] == "library"), None)
    r.check("study now in the earlier slot", lib and lib["hour"] == hs + 1)
    r.check("basketball now in the later slot", bball and bball["hour"] == hs + 2)
    r.check("basketball stayed at SHHO (locked venue)", bball and bball["facility"] == "SHHO")

    print("\n" + "=" * 70)
    print("TEST 42 — Locked venue: warn (don't substitute) when it's full at the new time")
    print("=" * 70)
    reset_db()
    r.agent.my_bookings = []
    r.agent.pending_sports = None
    r.agent.pending_booking = None
    r.agent._pending_multi = None
    h1 = next((h for h in range(9, 14)
               if sports_client.free_courts(DATE, "SHHO", "badminton", 4, h) > 0
               and sports_client.free_courts(DATE, "SHHO", "basketball", 1, h + 1) > 0
               and library_client.find_best_room(DATE, h + 2, 0, 2, 1, preferred_floor="LG3")["found"]), None)
    r.check("found a lock-warn setup", h1 is not None)
    full_h = h1 + 4 if h1 + 4 <= 21 else h1 + 3
    # Occupy the single SHHO court at full_h so the locked venue is unavailable there.
    db.add_booking(DATE, "SHHO", full_h, 0, 1, "OCCUPY-TEST")
    r.check("forced hour is full for SHHO basketball",
            sports_client.free_courts(DATE, "SHHO", "basketball", 1, full_h) == 0)
    await r.say(f"book badminton at sports hall at {h1}:00")
    await r.say("yes")
    await r.say("after that basketball at the same place, then study an hour at lg3")
    before = (r.agent._pending_multi or {}).get("plan", [])
    bb_before = next((p for p in before if p.get("sport") == "basketball"), None)["hour"]
    reply = await r.say(f"actually start the basketball at {full_h}:00")
    r.check("warned that the locked venue has no court", "S. H. Ho Sports Hall has no" in reply)
    r.check("said the plan is unchanged", "unchanged" in reply.lower())
    after = (r.agent._pending_multi or {}).get("plan", [])
    bb_after = next((p for p in after if p.get("sport") == "basketball"), None)["hour"]
    r.check("basketball time was NOT changed", bb_after == bb_before)

    print("\n" + "=" * 70)
    print("TEST 43 — Misspelled sport ('basketbal') still chains correctly")
    print("=" * 70)
    r.check("resolve_sport('basketbal') == basketball", sports_client.resolve_sport("basketbal") == "basketball")
    r.check("resolve_sport('basket ball') == basketball", sports_client.resolve_sport("basket ball") == "basketball")
    h0 = _setup_chain()
    r.check("found setup for misspelling chain", h0 is not None)
    await r.say(f"book badminton at sports hall at {h0}:00")
    await r.say("yes")
    reply = await r.say("after badminton i wanna play basketbal at the same place then study 2 hours at lg3")
    plan = (r.agent._pending_multi or {}).get("plan", [])
    r.check("multi-plan built despite misspelling", len(plan) == 2)
    bball = next((p for p in plan if p.get("sport") == "basketball"), None)
    r.check("basketball recognized + at SHHO", bball and bball["facility"] == "SHHO")

    print("\n" + "=" * 70)
    print("TEST 44 — Chain honors 'same place' strictly (warn, don't substitute)")
    print("=" * 70)
    reset_db()
    r.agent.my_bookings = []
    r.agent.pending_sports = None
    r.agent.pending_booking = None
    r.agent._pending_multi = None
    # badminton@SHHO@h0; occupy SHHO basketball for the whole window so 'same place' can't be honored
    h0 = next((h for h in range(9, 14)
               if sports_client.free_courts(DATE, "SHHO", "badminton", 4, h) > 0
               and library_client.find_best_room(DATE, h + 1, 0, 4, 1)["found"]), None)
    r.check("found setup for strict-venue test", h0 is not None)
    for hh in range(h0 + 1, h0 + 6):
        db.add_booking(DATE, "SHHO", hh, 0, 1, f"OCCUPY-{hh}")
    await r.say(f"book badminton at sports hall at {h0}:00")
    await r.say("yes")
    reply = await r.say("after badminton, basketball at the same place, then study 2 hours")
    r.check("warned SHHO has no basketball court", "S. H. Ho Sports Hall has no" in reply)
    plan = (r.agent._pending_multi or {}).get("plan", [])
    r.check("did NOT silently substitute Outdoor",
            not any(p.get("facility") == "BBALL-OUT" for p in plan))

    print("\n" + "=" * 70)
    print("TEST 45 — 'change my study room to lg3' targets the LIBRARY booking")
    print("=" * 70)
    reset_db()
    r.agent.my_bookings = []
    r.agent.pending_sports = None
    r.agent.pending_booking = None
    r.agent._pending_edit = None
    # Make a sports booking AND a library booking; the library one must be the target.
    sp_h = _free_sport_hour("badminton", 9, 22)
    await r.say(f"book badminton at {sp_h}:00")
    await r.say("yes")
    lh = next((h for h in range(9, 20)
               if library_client.find_best_room(DATE, h, 0, 2, 1, preferred_floor="LG1")["found"]
               and library_client.find_best_room(DATE, h, 0, 2, 1, preferred_floor="LG3")["found"]), None)
    r.check("found a library hour with LG1+LG3 free", lh is not None)
    await r.say(f"book a study room in lg1 at {lh}:00 for 1 hour")
    await r.say("yes")
    lib_ref = next(b["ref"] for b in r.agent.my_bookings if b.get("kind") != "sports")
    reply = await r.say("change my study room to a lg3 room")
    r.check("did NOT mis-target the sports booking", "can't be moved" not in reply.lower())
    r.check("started a library edit to LG3",
            r.agent._pending_edit is not None and
            (library_client.LC_ROOMS.get(r.agent._pending_edit["room_id"]) or
             library_client.STUDY_ROOMS.get(r.agent._pending_edit["room_id"]))["floor"] == "LG3")

    print("\n" + "=" * 70)
    print("TEST 46 — 'I prefer LG1 to study' edits the existing study booking (no LLM)")
    print("=" * 70)
    reset_db()
    r.agent.my_bookings = []
    r.agent.pending_sports = None
    r.agent.pending_booking = None
    r.agent._pending_edit = None
    # A sports booking AND a study booking on LG3, with LG1 free at the same hour.
    sp = _free_sport_hour("badminton", 9, 22)
    await r.say(f"book badminton at {sp}:00")
    await r.say("yes")
    lh = next((h for h in range(9, 20)
               if library_client.find_best_room(DATE, h, 0, 2, 1, preferred_floor="LG3")["found"]
               and library_client.find_best_room(DATE, h, 0, 2, 1, preferred_floor="LG1")["found"]), None)
    r.check("found an LG3+LG1 hour", lh is not None)
    lg3room = library_client.find_best_room(DATE, lh, 0, 2, 1, preferred_floor="LG3")["room_id"]
    await r.say(f"book {lg3room} at {lh}:00 for 1 hour")
    await r.say("yes")
    r.check("two bookings (badminton + LG3 study)", len(r.agent.my_bookings) == 2)
    reply = await r.say("actually i prefer lg1 to study")
    r.check("did NOT mis-target the sports booking", "can't be moved" not in reply.lower())
    r.check("started a deterministic edit (no LLM stub)", "<LLM>" not in reply and r.agent._pending_edit is not None)
    r.check("edit target is on LG1",
            r.agent._pending_edit and
            (library_client.LC_ROOMS.get(r.agent._pending_edit["room_id"]) or
             library_client.STUDY_ROOMS.get(r.agent._pending_edit["room_id"]))["floor"] == "LG1")
    r.check("edit keeps the same hour", r.agent._pending_edit and r.agent._pending_edit["hour"] == lh)
    reply = await r.say("any. just book thanks")   # exercises the new confirm phrase
    r.check("edit applied via 'just book'", "Changed" in reply)
    r.check("still two bookings", len(r.agent.my_bookings) == 2)
    r.check("study now on LG1",
            any(b["room_id"].startswith("LG1-R") for b in r.agent.my_bookings))
    r.check("badminton untouched", any(b.get("sport") == "badminton" for b in r.agent.my_bookings))

    # ── Summary ──────────────────────────────────────────────────────────────
    reset_db()
    print("\n" + "=" * 70)
    print(f"RESULTS:  {r.passed} passed, {r.failed} failed")
    print("=" * 70)
    sys.exit(1 if r.failed else 0)


if __name__ == "__main__":
    asyncio.run(main())
