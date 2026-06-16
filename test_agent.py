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
    reply = await r.say("change my booking to an lg3 room")
    r.check("pending edit created", r.agent._pending_edit is not None)
    r.check("edit target on LG3", r.agent._pending_edit and r.agent._pending_edit["room_id"].startswith("LG3"))
    r.check("edit keeps same time", r.agent._pending_edit and r.agent._pending_edit["hour"] == arr_h)
    reply = await r.say("yes")
    r.check("edit applied", "Changed" in reply)
    r.check("still one booking", len(r.agent.my_bookings) == 1)
    r.check("new room on LG3", r.agent.my_bookings[0]["room_id"].startswith("LG3"))
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

    # ── Summary ──────────────────────────────────────────────────────────────
    reset_db()
    print("\n" + "=" * 70)
    print(f"RESULTS:  {r.passed} passed, {r.failed} failed")
    print("=" * 70)
    sys.exit(1 if r.failed else 0)


if __name__ == "__main__":
    asyncio.run(main())
