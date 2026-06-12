# Copyright (c) Microsoft. All rights reserved.

import logging
import os
import re
import time
import datetime as dt
from datetime import datetime
from typing import Optional

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
)

load_dotenv()
logger = logging.getLogger(__name__)


class AgentFrameworkAgent(AgentInterface):

    AGENT_PROMPT = """You are a helpful HKUST campus assistant helping students with bus transport and library room bookings.

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
- 91M: KMB bus — Diamond Hill MTR ↔ HKUST ↔ Po Lam
- 11: GMB minibus — Choi Hung MTR ↔ HKUST ↔ Hang Hau Village
- 11M: GMB minibus — Hang Hau MTR ↔ HKUST
- 12: GMB minibus — Po Lam ↔ HKUST ↔ Sai Kung
- 792M: Citybus — TKO Station ↔ HKUST ↔ Sai Kung

KEY STOPS:
91M: Diamond Hill ↔ Choi Hung ↔ Tseng Lan Shue ↔ Pik Uk ↔ Tai Po Tsai ↔ HKUST ↔ Hang Hau ↔ Po Lam
11: Choi Hung ↔ Tseng Lan Shue ↔ Pik Uk ↔ Tai Po Tsai ↔ HKUST ↔ TKO Hospital ↔ Hang Hau
11M: Hang Hau MTR ↔ Tai Po Tsai ↔ HKUST
12: Po Lam ↔ Tseng Lan Shue ↔ Pik Uk ↔ Tai Po Tsai ↔ HKUST ↔ Marina Cove ↔ Sai Kung
792M: TKO Station ↔ Tiu Keng Leng ↔ TKO Hospital ↔ Tai Po Tsai ↔ HKUST ↔ Nam Pin Wai ↔ Sai Kung

ROUTING:
- TKO: 792M direct, or 11M to Hang Hau MTR then MTR
- Diamond Hill / Kowloon: 91M
- Sai Kung: 12 or 792M
- Hang Hau MTR: 11M (fastest), 11, or 792M
- Choi Hung MTR: 91M or 11
- Po Lam: 91M or 12
- ONLY use [LIVE BUS DATA] for ETAs — never invent times
- If direction is ambiguous, ask ONE clarifying question first

LIBRARY (lbbooking.ust.hk):
- All rooms are bookable in 30-minute slots, up to 4 consecutive slots (2 hours max).
- Learning Commons LC-1 to LC-18 (LG1): LC-1–13 seat 7, LC-14–17 seat 10, LC-18 seats 6. Max 2hr booking.
- Study Rooms: 1/F (1F-R1–4), LG1 (LG1-R1–9), LG3 (LG3-R1–9), LG4 (LG4-R1–11). All seat ~4.
- Hours: 8am–11pm. Library data covers the FULL day — shown slots are relevant to query, not a data limit.

BOOKING FLOW:
- Find best room → present recommendation → ask "Shall I book this for you?"
- Never book without explicit confirmation (yes/ok/sure/go ahead)
- After confirmed booking show the reference number
- If no match: suggest smaller rooms or alternative times
- [BOOKING RECOMMENDATION] = found a room, present it and ask to confirm
- [BOOKING SEARCH RESULT] = no exact match, present alternatives
- [BOOKING DURATION NEEDED] = ask how long they need the room

RESPONSE RULES:
- Be concise and direct. No thinking out loud.
- Ambiguous bus direction → ask ONE short question first
- Never show [LIVE BUS DATA] / [LIVE LIBRARY DATA] tags in responses
- Never invent bus times or room availability
- Common sense: "3pm" not "3am" for meetings
- Keep responses short, use bullets only for lists"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.conversation_history = []
        self.pending_booking = None
        self._create_client()

    def _create_client(self):
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        endpoint = endpoint.rstrip("/").replace("/openai/v1", "").replace("/openai", "")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

        try:
            credential = AzureCliCredential()
            token_provider = get_bearer_token_provider(credential, "https://ai.azure.com/.default")
            token_provider()
            logger.info("Using Azure CLI credential")
        except Exception:
            try:
                from azure.identity import DefaultAzureCredential
                credential = DefaultAzureCredential()
                token_provider = get_bearer_token_provider(credential, "https://ai.azure.com/.default")
                logger.info("Using DefaultAzureCredential")
            except Exception as e:
                raise ValueError(f"No valid Azure credential found: {e}")

        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider,
            api_version=api_version,
            api_key="placeholder",
        )
        self.deployment = deployment
        logger.info(f"✅ Client created → {deployment}")

    async def initialize(self) -> None:
        logger.info("Agent ready")

    async def process_user_message(
        self,
        message: str,
        auth: Authorization,
        auth_handler_name: Optional[str],
        context: TurnContext,
    ) -> str:
        try:
            message_lower = message.lower().strip()

            # ── Debug command ─────────────────────────────────────────
            if message.strip() == "$DATA":
                today = datetime.now().strftime("%Y-%m-%d")
                return print_full_day_text(today)

            # ── Booking confirmation flow ─────────────────────────────
            confirm_words = ["yes", "yeah", "yep", "ok", "sure", "go ahead", "confirm", "book it", "do it", "please"]
            cancel_words  = ["no", "nope", "cancel", "don't", "nevermind", "never mind", "stop", "skip"]

            if self.pending_booking:
                if any(w in message_lower for w in confirm_words):
                    pb = self.pending_booking
                    self.pending_booking = None
                    booking = book_room(
                        date_str=pb["date"],
                        room_id=pb["room_id"],
                        hour=pb["hour"],
                        minute=pb["minute"],
                        num_slots=pb["num_slots"],
                    )
                    return format_booking_confirmation(booking)
                elif any(w in message_lower for w in cancel_words):
                    self.pending_booking = None
                    return "Booking cancelled. Let me know if you'd like a different room or time."

            # ── Bus data fetch ────────────────────────────────────────
            bus_keywords = ["bus", "eta", "next bus", "91m", "11m", "792", "route",
                            "transport", "ride", "get to", "going to", "leave campus",
                            "depart", "hang hau", "diamond hill", "po lam", "choi hung",
                            "tseung kwan o", "tko", "sai kung", "north gate", "south gate"]
            is_bus_query = any(k in message_lower for k in bus_keywords)

            extra_context = ""

            if is_bus_query:
                dest_keywords = {
                    "diamond_hill": ["diamond hill", "kowloon", "choi hung mtr", "choi hung station"],
                    "po_lam":       ["po lam"],
                    "tko":          ["tseung kwan o", "tko", "popcorn", "tiu keng leng"],
                    "sai_kung":     ["sai kung", "marina cove"],
                    "hkust":        ["to hkust", "to ust", "to campus", "to school",
                                     "get to hkust", "going to hkust", "at hang hau",
                                     "from hang hau", "at po lam", "from po lam",
                                     "at diamond hill", "from diamond hill",
                                     "at choi hung", "from choi hung",
                                     "at tko", "from tko"],
                }
                matched_filter = "all"
                for fk, kws in dest_keywords.items():
                    if any(k in message_lower for k in kws):
                        matched_filter = fk
                        break

                t1 = time.time()
                etas = await get_all_hkust_etas(matched_filter)
                t2 = time.time()
                logger.info(f"Bus fetch: {t2-t1:.2f}s filter={matched_filter} results={len(etas)}")
                extra_context += f"\n\n[LIVE BUS DATA]\n{format_etas_for_agent(etas)}"

            # ── Library data fetch ────────────────────────────────────
            library_keywords = ["room", "study room", "learning commons", "lc-", "lg1", "lg3",
                                 "lg4", "1f", "library", "book", "available", "free room",
                                 "study space", "quiet", "seat", "study", "meeting room",
                                 "reserve", "booking"]
            is_library_query = any(k in message_lower for k in library_keywords)

            if is_library_query:
                now = datetime.now()
                date_str = now.strftime("%Y-%m-%d")
                current_hour = now.hour
                current_minute = 0 if now.minute < 30 else 30

                until_match = re.search(r'(?:until|till)\s*(\d{1,2})(?::00)?\s*(pm|am)?', message_lower)
                from_match  = re.search(r'(?:from|at|@)\s*(\d{1,2})(?::00)?\s*(pm|am)?', message_lower)
                time_match  = re.search(r'\b(\d{1,2})(?::00)?\s*(pm|am)\b', message_lower)

                # Booking request detection
                booking_keywords = ["book", "reserve", "booking", "meeting", "need a room",
                                     "get a room", "find a room", "want a room"]
                is_booking_request = any(k in message_lower for k in booking_keywords)

                if is_booking_request:
                    # Parse time
                    req_match = from_match or time_match
                    if req_match:
                        req_h = int(req_match.group(1))
                        meridiem = req_match.group(2)
                        if meridiem == "pm" and req_h < 12:
                            req_h += 12
                        elif meridiem == "am" and req_h == 12:
                            req_h = 0
                    else:
                        req_h = current_hour

                    req_m = 0

                    # Parse date
                    tomorrow = (dt.datetime.now() + dt.timedelta(days=1)).strftime("%Y-%m-%d")
                    req_date = tomorrow if any(w in message_lower for w in ["tomorrow", "tmr", "tmrw"]) else date_str

                    # Parse capacity
                    people_match = re.search(r'(\d+)\s*(?:people|person|ppl|pax|of us|students|attendees)', message_lower)
                    capacity = int(people_match.group(1)) if people_match else 1

                    # Infer duration from context
                    duration_match = re.search(r'(\d+)\s*hour', message_lower)
                    if duration_match:
                        num_slots = int(duration_match.group(1)) * 2
                    elif re.search(r'(\d+)\s*(?:min|mins|minutes)', message_lower):
                        mins = int(re.search(r'(\d+)\s*(?:min|mins|minutes)', message_lower).group(1))
                        num_slots = max(1, round(mins / 30))
                    elif any(k in message_lower for k in ["meeting", "interview", "presentation", "discussion"]):
                        num_slots = 2  # 1 hour
                    elif any(k in message_lower for k in ["lecture", "seminar", "workshop", "session", "class"]):
                        num_slots = 4  # 2 hours
                    elif any(k in message_lower for k in ["quick", "brief", "short", "30 min", "half hour", "half an hour"]):
                        num_slots = 1  # 30 mins
                    else:
                        num_slots = 0  # unknown

                    if num_slots == 0:
                        extra_context += "\n\n[BOOKING DURATION NEEDED]\nThe user wants to book a room but duration is unclear. Ask: 'How long do you need the room for?'"
                    else:
                        result = find_best_room(req_date, req_h, req_m, num_slots, capacity)
                        end_h = req_h + (num_slots * 30) // 60
                        end_m = (num_slots * 30) % 60
                        rec_str = format_room_recommendation(
                            result, req_date,
                            f"{req_h:02d}:{req_m:02d}",
                            f"{end_h:02d}:{end_m:02d}"
                        )
                        if result["found"]:
                            self.pending_booking = {
                                "room_id": result["room_id"],
                                "date": req_date,
                                "hour": req_h,
                                "minute": req_m,
                                "num_slots": num_slots,
                                "capacity": result["capacity"],
                            }
                            extra_context += f"\n\n[BOOKING RECOMMENDATION]\n{rec_str}\nAsk: 'Shall I book {result['room_id']} for you?'"
                        else:
                            extra_context += f"\n\n[BOOKING SEARCH RESULT]\n{rec_str}"

                elif until_match:
                    until_h = int(until_match.group(1))
                    if until_match.group(2) == "pm" and until_h < 12:
                        until_h += 12
                    result = find_rooms_free_until(date_str, current_hour, current_minute, until_h)
                    extra_context += f"\n\n[LIVE LIBRARY DATA]\n{format_continuous_for_agent(result)}"

                elif from_match or time_match:
                    match = from_match or time_match
                    h = int(match.group(1))
                    meridiem = match.group(2)
                    if meridiem == "pm" and h < 12:
                        h += 12
                    elif meridiem == "am" and h == 12:
                        h = 0
                    slots = get_slots_range(date_str, h, 0, 4)
                    slot  = get_slot_availability(date_str, h, 0)
                    extra_context += f"\n\n[LIVE LIBRARY DATA from {h:02d}:00]\n{format_range_for_agent(slots)}\n{format_slot_for_agent(slot)}"

                elif any(k in message_lower for k in ["next few", "later", "upcoming", "rest of", "today"]):
                    slots = get_slots_range(date_str, current_hour, current_minute, 6)
                    extra_context += f"\n\n[LIVE LIBRARY DATA - next 3 hours]\n{format_range_for_agent(slots)}"

                else:
                    lead_hour, lead_minute = current_hour, current_minute
                    if now.minute >= 40 and current_minute == 0:
                        lead_minute = 30
                    elif now.minute >= 10 and current_minute == 30:
                        lead_hour += 1
                        lead_minute = 0
                    slot  = get_slot_availability(date_str, lead_hour, lead_minute)
                    slots = get_slots_range(date_str, lead_hour, lead_minute, 4)
                    extra_context += (
                        f"\n\n[LIVE LIBRARY DATA - {now.hour:02d}:{now.minute:02d}]\n"
                        f"{format_range_for_agent(slots)}\n"
                        f"{format_slot_for_agent(slot)}"
                    )

            # ── Build prompt and call model ───────────────────────────
            user_content = f"{message}\n{extra_context}" if extra_context else message
            self.conversation_history.append({"role": "user", "content": user_content})
            messages = [{"role": "system", "content": self.AGENT_PROMPT}] + self.conversation_history

            t3 = time.time()
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                max_tokens=600,
                timeout=25,
            )
            t4 = time.time()
            logger.info(f"Model: {t4-t3:.2f}s tokens={response.usage.prompt_tokens}")

            reply = response.choices[0].message.content

            # Store compact history
            self.conversation_history[-1] = {"role": "user", "content": message}
            history_reply = reply[:300] + "..." if len(reply) > 300 else reply
            self.conversation_history.append({"role": "assistant", "content": history_reply})

            if len(self.conversation_history) > 6:
                self.conversation_history = self.conversation_history[-6:]

            return reply

        except Exception as e:
            logger.error(f"Error: {e}")
            return f"Sorry, I encountered an error: {str(e)}"

    async def cleanup(self) -> None:
        self.conversation_history = []
        self.pending_booking = None
        logger.info("Cleanup done")