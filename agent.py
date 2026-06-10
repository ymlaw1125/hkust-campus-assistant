from datetime import datetime 
import logging
import os
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
    format_slot_for_agent,
    format_range_for_agent,
    format_continuous_for_agent,
)

load_dotenv()
logger = logging.getLogger(__name__)


class AgentFrameworkAgent(AgentInterface):
    AGENT_PROMPT = """You are a helpful HKUST campus assistant helping students with bus transport and campus facilities.

BUSES SERVING HKUST — ONLY these 5 routes exist, do not invent others:
- Route 91M: KMB bus — Diamond Hill MTR ↔ HKUST ↔ Po Lam
- Route 11: GMB minibus — Choi Hung MTR ↔ HKUST ↔ Hang Hau Village
- Route 11M: GMB minibus — Hang Hau MTR ↔ HKUST (short direct route)
- Route 12: GMB minibus — Po Lam ↔ HKUST ↔ Sai Kung
- Route 792M: Citybus — Tseung Kwan O Station ↔ HKUST ↔ Sai Kung

FULL STOP LISTS (use these to answer questions about intermediate stops):

Route 91M (to terminus):
  DIAMOND HILL STATION → LUNG POON COURT → TAI YAU STREET SAN PO KONG → CHOI HUNG BBI → NGAU CHI WAN BBI → GOOD HOPE SCHOOL → ANDERSON ROAD → DENON TERRACE → TSENG LAN SHUE → PAK SHEK WO → PIK UK → TA KU LING SAN TSUEN → TAI PO TSAI KAU → H.K.U.S.T. NORTH → TAI PO TSAI VILLAGE → NGAN YING ROAD → SHUI BIN TSUEN → BOON KIN VILLAGE → MING TAK ESTATE → EAST POINT CITY → HANG HAU STATION → HAU TAK ESTATE → KING LAM ESTATE → METRO CITY → PO LAM
Route 91M (reverse):
  PO LAM → KING LAM ESTATE → HAU TAK ESTATE → EAST POINT CITY → HANG HAU STATION → TSEUNG KWAN O HOSPITAL → SHUI BIN TSUEN → YING YIP ROAD → NGAN YING ROAD → H.K.U.S.T. SOUTH → TAI PO TSAI → TAI PO TSAI KAU → PIK UK → PAK SHEK WO → TSENG LAN SHUE → GOOD HOPE SCHOOL → CHOI WAN ESTATE → CHOI HUNG STATION → DIAMOND HILL STATION

Route 11 (to terminus):
  Ngau Chi Wan → Choi Wan → Good Hope School → Denon Terrace → Tseng Lan Shue → Pik Uk → Ta Ku Ling San Tsuen → Tai Po Tsai Kau → HKUST North → Tai Po Tsai Tsuen → Shui Pin Tsuen → Boon Kin Village → TKO Hospital → East Point City → Hang Hau Station → Hang Hau Village
Route 11 (reverse):
  Hang Hau Village → Hang Hau Station → Hang Hau North → HKUST South → Tai Po Tsai Tsuen → Tai Po Tsai Kau → Ta Ku Ling San Tsuen → Pik Uk → Pak Shek Wo → Tseng Lan Shue → Good Hope School → Choi Wan → Ping Shek Estate → Choi Hung Estate

Route 11M (to terminus):
  Hang Hau Station → Tai Po Tsai Tsuen → HKUST
Route 11M (reverse):
  HKUST → Tai Po Tsai Tsuen → Shui Pin Tsuen → Boon Kin Village → TKO Hospital → Hau Tak Estate → Hang Hau Station

Route 12 (to terminus):
  Po Lam → Sau Mau Ping → Shun Lee → Tseng Lan Shue → Pik Uk → Ta Ku Ling San Tsuen → Tai Po Tsai Kau → HKUST → Hiram's Highway → Marina Cove → Hebe Haven → Pak Kong → Po Lo Che → Lakeside Garden → Sai Kung
Route 12 (reverse):
  Sai Kung → Lakeside Garden → Pak Kong → Hebe Haven → Marina Cove → HKUST → Pik Uk → Tseng Lan Shue → Sau Mau Ping → Po Lam

Route 792M (to terminus):
  Tseung Kwan O Station → Tong Ming Street → Tiu Keng Leng Station → Kin Ming Estate → TKO Hospital → St Vincent Catholic Church → Shui Bin Tsuen → Ying Yip Road → Ngan Ying Road → Tai Po Tsai Village → HKUST → Tai Po Tsai Kau → Wo Mei → Nam Pin Wai → Ho Chung → Pak Wai → Fisherman Village → Pak Sha Wan → Habitat → Pak Kong → Po Lo Che → Lakeside Garden → Sai Kung
Route 792M (reverse):
  Sai Kung → Lakeside Garden → Po Lo Che → Pak Kong → Pak Sha Wan → Fisherman Village → Pak Wai → Marina Cove → Nam Pin Wai → Wo Mei → Tai Po Tsai Kau → HKUST → Tai Po Tsai Village → Ngan Ying Road → Shui Bin Tsuen → Boon Kin Village → TKO Hospital → Hau Tak Estate → Tiu Keng Leng Station → Tseung Kwan O Station

ROUTING RULES:
- Check the stop lists above before saying a destination has no service
- If the destination is an intermediate stop on a route, suggest that route
- For Tseung Kwan O city centre: 792M direct, or 11M to Hang Hau MTR then MTR
- For Diamond Hill / Kowloon: 91M direct
- For Sai Kung: 12 or 792M
- For Hang Hau MTR: 11M (fastest), 11, or 792M
- For Choi Hung MTR: 91M or 11
- For Po Lam: 91M or 12
- If truly no direct route, suggest nearest stop + short taxi/walk
- ONLY use [LIVE BUS DATA] for ETAs — never invent bus times
- When [LIVE BUS DATA] is provided, always show the actual times

When you receive [LIVE BUS DATA], use those exact times to answer.

LIBRARY ROOMS (HKUST Library, lbbooking.ust.hk):
- Learning Commons (LC-1 to LC-18): LG1 floor. LC-1 to LC-13 seat 7 people, LC-14 to LC-17 seat 10 people, LC-18 seats 6 people. Bookable in 30-min slots, max 4 slots (2 hours) per booking.
- Library Study Rooms: Smaller rooms on 1/F (1F-R1 to R4), LG1 (LG1-R1 to R9), LG3 (LG3-R1 to R9), LG4 (LG4-R1 to R11). All seat ~4 people.
- Opening hours: 8:00am–11:00pm daily
- Booking URL: https://lbbooking.ust.hk

LIBRARY RESPONSE RULES:
- NEVER dump all room data unless explicitly asked for a full list
- When showing availability, be concise: mention how many rooms/pods are free and name 2-3 preferred ones, e.g. "3 study rooms free, including LC-5, LC-14, and LG1-R3"
- Always consider current time — if it's late in an hour, prioritize the NEXT hour's availability
- When asked "free from X to Y", use the continuously free data provided
- When asked about a specific time, show that slot's availability
- If a student asks about a specific room like LC-4, check if it appears in the data
- If [LIVE LIBRARY DATA] shows multiple slots, summarize them smartly: "3 slots available in the next 4 hours"
- Always end library responses with the booking URL
- Never invent room availability — only use [LIVE LIBRARY DATA]"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.conversation_history = []
        self._create_client()

    def _create_client(self):
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        endpoint = endpoint.rstrip("/").replace("/openai/v1", "").replace("/openai", "")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

        # Try Azure CLI credential first, fall back to other credential types
        try:
            from azure.identity import AzureCliCredential
            credential = AzureCliCredential()
            token_provider = get_bearer_token_provider(
                credential,
                "https://ai.azure.com/.default"
            )
            # Test it works
            token_provider()
            logger.info("Using Azure CLI credential")
        except Exception:
            try:
                from azure.identity import DefaultAzureCredential
                credential = DefaultAzureCredential()
                token_provider = get_bearer_token_provider(
                    credential,
                    "https://ai.azure.com/.default"
                )
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
            # Check if the message is bus-related and fetch live data
            bus_keywords = ["bus", "eta", "arrive", "next", "91m", "11m", "11", "12", "792", "route", "transport", "ride", "get to", "going to", "leave", "depart", "hkust", "hang hau", "diamond hill", "po lam", "choi hung"]
            message_lower = message.lower()
            is_bus_query = any(k in message_lower for k in bus_keywords)

            extra_context = ""
            if is_bus_query:
                # Determine direction from message
                to_keywords = ["to hkust", "to campus", "going to school", "get to hkust", "arrive at hkust", "reach hkust"]
                from_keywords = ["from hkust", "leave hkust", "leaving campus", "going home", "from campus"]

                if any(k in message_lower for k in from_keywords):
                    etas = await get_all_hkust_etas("from")
                    direction_label = "FROM HKUST"
                elif any(k in message_lower for k in to_keywords):
                    etas = await get_all_hkust_etas("to")
                    direction_label = "TO HKUST"
                else:
                    # Fetch both directions
                    etas_to = await get_all_hkust_etas("to")
                    etas_from = await get_all_hkust_etas("from")
                    to_str = format_etas_for_agent(etas_to)
                    from_str = format_etas_for_agent(etas_from)
                    extra_context = f"\n\n[LIVE BUS DATA]\nBuses TO HKUST:\n{to_str}\n\nBuses FROM HKUST:\n{from_str}"
                    direction_label = None

                if direction_label:
                    etas = await get_all_hkust_etas("to" if "TO" in direction_label else "from")
                    extra_context = f"\n\n[LIVE BUS DATA - {direction_label}]\n{format_etas_for_agent(etas)}"
            
            # Check if library-related
            library_keywords = ["room", "study room", "learning commons", "lc-", "lg1", "lg3", "lg4", "1f", "pod", "nap", "library", "book", "available", "free room", "study space", "quiet", "space", "seat", "study"]
            is_library_query = any(k in message_lower for k in library_keywords)

            if is_library_query:
                now = datetime.now()
                date_str = now.strftime("%Y-%m-%d")
                current_hour = now.hour
                current_minute = 0 if now.minute < 30 else 30

                # Detect "free until X" or "from now to X" pattern
                import re
                until_match = re.search(r'(?:until|till|to|until)\s*(\d{1,2})(?::00)?\s*(pm|am)?', message_lower)
                from_match  = re.search(r'(?:from|at|@)\s*(\d{1,2})(?::00)?\s*(pm|am)?', message_lower)
                time_match  = re.search(r'\b(\d{1,2})(?::00)?\s*(pm|am)\b', message_lower)

                if until_match:
                    # "rooms free from now to 7pm" — find continuously free rooms
                    until_h = int(until_match.group(1))
                    if until_match.group(2) == "pm" and until_h < 12:
                        until_h += 12
                    result = find_rooms_free_until(date_str, current_hour, current_minute, until_h)
                    extra_context += f"\n\n[LIVE LIBRARY DATA - continuously free until {until_h:02d}:00]\n{format_continuous_for_agent(result)}"

                elif from_match or time_match:
                    # "rooms at 5pm" — show that specific slot + next 2
                    match = from_match or time_match
                    h = int(match.group(1))
                    meridiem = match.group(2)
                    if meridiem == "pm" and h < 12:
                        h += 12
                    elif meridiem == "am" and h == 12:
                        h = 0
                    m = 0
                    slots = get_slots_range(date_str, h, m, 4)
                    extra_context += f"\n\n[LIVE LIBRARY DATA from {h:02d}:00]\n{format_range_for_agent(slots)}"
                    # Also give full detail for the requested slot
                    slot = get_slot_availability(date_str, h, m)
                    extra_context += f"\nFull detail for {h:02d}:00:\n{format_slot_for_agent(slot)}"

                elif any(k in message_lower for k in ["next few", "later", "upcoming", "next hour", "rest of", "today"]):
                    # "what's available later" — show next 6 slots (3 hours)
                    slots = get_slots_range(date_str, current_hour, current_minute, 6)
                    extra_context += f"\n\n[LIVE LIBRARY DATA - next 3 hours]\n{format_range_for_agent(slots)}"

                else:
                    # Default: show current slot + next 3 slots
                    # Smart timing: if >40 min into current slot, lead with next slot
                    lead_hour = current_hour
                    lead_minute = current_minute
                    if now.minute >= 40 and current_minute == 0:
                        lead_minute = 30
                    elif now.minute >= 10 and current_minute == 30:
                        lead_hour = current_hour + 1
                        lead_minute = 0

                    slot = get_slot_availability(date_str, lead_hour, lead_minute)
                    slots = get_slots_range(date_str, lead_hour, lead_minute, 4)
                    extra_context += f"\n\n[LIVE LIBRARY DATA - current time {now.hour:02d}:{now.minute:02d}]\n"
                    extra_context += f"Upcoming slots:\n{format_range_for_agent(slots)}\n"
                    extra_context += f"Full detail for {lead_hour:02d}:{lead_minute:02d}:\n{format_slot_for_agent(slot)}"
                    
                           
            # Build message with live data injected
            user_content = message
            if extra_context:
                user_content = f"{message}\n{extra_context}"

            self.conversation_history.append({"role": "user", "content": user_content})
            messages = [{"role": "system", "content": self.AGENT_PROMPT}] + self.conversation_history

            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
            )
            reply = response.choices[0].message.content

            # Store clean message in history (without the injected data)
            self.conversation_history[-1] = {"role": "user", "content": message}
            self.conversation_history.append({"role": "assistant", "content": reply})

            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]

            return reply

        except Exception as e:
            logger.error(f"Error: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    async def cleanup(self) -> None:
        self.conversation_history = []
        logger.info("Cleanup done")