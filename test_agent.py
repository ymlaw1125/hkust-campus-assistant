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
    print("TEST 4 — Floor preference change while pending (want LG1)")
    print("=" * 70)
    prev_room = r.agent.pending_booking["room_id"]
    reply = await r.say("actually i want one in lg1")
    new_room = r.agent.pending_booking["room_id"]
    r.check("new room is on LG1", new_room.startswith("LG1"))
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
    r.check("booked room is the LG1 one", booked_room == new_room)
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
    r.check("an alternative floor is free today", target_floor is not None)
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
    new_ref = r.agent.my_bookings[0]["ref"]

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
    print("TEST 18 — 1 slot per facility per day enforced")
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
    r.check("second squash booking blocked", "already used" in reply.lower())
    r.check("no second pending", r.agent.pending_sports is None)

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

    # ── Summary ──────────────────────────────────────────────────────────────
    reset_db()
    print("\n" + "=" * 70)
    print(f"RESULTS:  {r.passed} passed, {r.failed} failed")
    print("=" * 70)
    sys.exit(1 if r.failed else 0)


if __name__ == "__main__":
    asyncio.run(main())
