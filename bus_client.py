import asyncio
from datetime import datetime
from hk_bus_eta import HKEta

# ============================================================
# HKUST ROUTES — hardcoded stop indices from discovery run
# ============================================================
# Each entry: route_id as used by hk-bus-eta, stop index for HKUST
HKUST_ROUTES = {
    # KMB 91M — need to confirm stop index from your earlier 91M output
    # GMB routes TO HKUST (arriving at campus)
    "11M_to_hkust": {
        "route_id": "11M+1+Hang Hau Station Public Transport Interchange+Hong Kong University of Science and Technology (North Station)",
        "stop_seq": 2,  # index 2 = HKUST North
        "label": "11M",
        "from": "Hang Hau MTR",
        "to": "HKUST North",
    },
    "11_to_hkust": {
        "route_id": "11+1+Choi Hung Station (Lung Cheung Road)+Hang Hau Village",
        "stop_seq": 9,  # index 9 = HKUST North Station
        "label": "11",
        "from": "Choi Hung MTR",
        "to": "HKUST North",
    },
    "11B_to_hkust": {
        "route_id": "11B+3+Choi Hung Station (Lung Cheung Road)+Hong Kong University of Science and Technology (North Station)",
        "stop_seq": 9,  # index 9 = HKUST
        "label": "11B",
        "from": "Choi Hung MTR",
        "to": "HKUST North",
    },
    # GMB routes FROM HKUST (leaving campus)
    "11M_from_hkust": {
        "route_id": "11M+1+Hong Kong University of Science and Technology (North Station)+Hang Hau Station Public Transport Interchange",
        "stop_seq": 0,  # index 0 = HKUST (first stop)
        "label": "11M",
        "from": "HKUST North",
        "to": "Hang Hau MTR",
    },
    "11_from_hkust_south": {
        "route_id": "11+1+Hang Hau Village+Choi Hung Station (Lung Cheung Road)",
        "stop_seq": 6,  # index 6 = HKUST South Station
        "label": "11",
        "from": "HKUST South",
        "to": "Choi Hung MTR",
    },
    "12_to_saikung": {
        "route_id": "12+1+Po Lam+Sai Kung",
        "stop_seq": 17,  # index 17 = HKUST
        "label": "12",
        "from": "Po Lam",
        "to": "Sai Kung via HKUST",
    },
    "91M_to_hkust": {
        "route_id": "91M+1+DIAMOND HILL STATION+PO LAM",
        "stop_seq": 16,  # 0-based: seq 17 = index 16
        "label": "91M",
        "from": "Diamond Hill MTR",
        "to": "HKUST North",
        "type": "kmb",
    },
    "91M_from_hkust": {
        "route_id": "91M+1+PO LAM+DIAMOND HILL STATION",
        "stop_seq": 12,  # 0-based: seq 13 = index 12
        "label": "91M",
        "from": "HKUST South",
        "to": "Diamond Hill MTR",
        "type": "kmb",
    },
    "792M_to_hkust": {
        "route_id": "792M+1+Tseung Kwan O Station+Sai Kung",
        "stop_seq": 13,
        "label": "792M",
        "from": "Tseung Kwan O Station",
        "to": "HKUST",
    },
    "792M_from_hkust": {
        "route_id": "792M+1+Sai Kung+Tseung Kwan O Station",
        "stop_seq": 16,
        "label": "792M",
        "from": "HKUST",
        "to": "Tseung Kwan O Station",
    },
}

# ============================================================
# ETA FETCHER
# ============================================================
_hketa = None

def get_hketa() -> HKEta:
    global _hketa
    if _hketa is None:
        _hketa = HKEta()
    return _hketa

async def get_eta_for_route(route_key: str) -> dict:
    route = HKUST_ROUTES.get(route_key)
    if not route:
        return {"error": f"Unknown route key: {route_key}"}
    if route.get("stop_seq") is None:
        return {"error": f"Stop not configured for {route['label']}"}

    hketa = get_hketa()
    try:
        raw_etas = hketa.getEtas(
            route_id=route["route_id"],
            seq=route["stop_seq"],
            language="en"
        )
        etas = _parse_etas(raw_etas)
        return {
            "route": route["label"],
            "from": route["from"],
            "to": route["to"],
            "etas": etas,
        }
    except Exception as e:
        return {"route": route["label"], "error": str(e), "etas": []}
    
def _parse_etas(raw_etas: list) -> list:
    """Parse raw ETA list into clean dicts with minutes remaining"""
    results = []
    for item in raw_etas:
        eta_time = item.get("eta")
        if not eta_time:
            continue
        try:
            eta_dt = datetime.fromisoformat(eta_time)
            now = datetime.now(eta_dt.tzinfo)
            minutes = int((eta_dt - now).total_seconds() / 60)
            if minutes >= 0:
                remark = item.get("remark", {})
                remark_en = remark.get("en", "") if isinstance(remark, dict) else str(remark)
                results.append({
                    "minutes": minutes,
                    "time": eta_dt.strftime("%H:%M"),
                    "remark": remark_en,
                })
        except Exception:
            continue
    return results

async def get_all_hkust_etas(direction: str = "to") -> list:
    """
    Get ETAs for all routes in a given direction.
    direction: "to" = arriving at HKUST, "from" = leaving HKUST
    """
    keys = [k for k in HKUST_ROUTES if direction in k]
    results = []
    for key in keys:
        eta_data = await get_eta_for_route(key)
        if eta_data.get("etas"):
            results.append(eta_data)
    return results

def format_etas_for_agent(eta_results: list) -> str:
    """Format ETA results into a clean string for the AI agent"""
    if not eta_results:
        return "No bus ETAs available right now (service may be outside operating hours)."

    lines = []
    for r in eta_results:
        if r.get("error"):
            continue
        etas = r["etas"]
        if not etas:
            lines.append(f"• {r['route']} ({r['from']} → {r['to']}): No upcoming buses")
            continue
        eta_strs = [f"{e['minutes']} min ({e['time']})" for e in etas[:3]]
        lines.append(f"• {r['route']} ({r['from']} → {r['to']}): {', '.join(eta_strs)}")

    return "\n".join(lines) if lines else "No bus data available."

async def generate_route_stops_summary():
    """Print all stops for all HKUST routes for the agent prompt"""
    hketa = get_hketa()
    
    routes_to_summarize = [
        ("91M", "91M+1+DIAMOND HILL STATION+PO LAM", "Diamond Hill MTR → Po Lam"),
        ("91M reverse", "91M+1+PO LAM+DIAMOND HILL STATION", "Po Lam → Diamond Hill MTR"),
        ("11", "11+1+Choi Hung Station (Lung Cheung Road)+Hang Hau Village", "Choi Hung MTR → Hang Hau Village"),
        ("11 reverse", "11+1+Hang Hau Village+Choi Hung Station (Lung Cheung Road)", "Hang Hau Village → Choi Hung MTR"),
        ("11M", "11M+1+Hang Hau Station Public Transport Interchange+Hong Kong University of Science and Technology (North Station)", "Hang Hau MTR → HKUST"),
        ("11M reverse", "11M+1+Hong Kong University of Science and Technology (North Station)+Hang Hau Station Public Transport Interchange", "HKUST → Hang Hau MTR"),
        ("12", "12+1+Po Lam+Sai Kung", "Po Lam → Sai Kung"),
        ("12 reverse", "12+1+Sai Kung+Po Lam", "Sai Kung → Po Lam"),
        ("792M", "792M+1+Tseung Kwan O Station+Sai Kung", "TKO Station → Sai Kung"),
        ("792M reverse", "792M+1+Sai Kung+Tseung Kwan O Station", "Sai Kung → TKO Station"),
    ]
    
    for label, route_id, description in routes_to_summarize:
        route = hketa.route_list.get(route_id)
        if not route:
            print(f"\n{label}: NOT FOUND")
            continue
        print(f"\n{label} ({description}):")
        for op, stop_list in route.get("stops", {}).items():
            stop_names = []
            for sid in stop_list:
                name = hketa.stop_list.get(sid, {}).get("name", {}).get("en", sid)
                stop_names.append(name)
            print(f"  {' → '.join(stop_names)}")
# ============================================================
# TEST
# ============================================================
async def test():
    print("Loading HK Bus ETA data...")
    
    print("\n=== Buses TO HKUST ===")
    results = await get_all_hkust_etas("to")
    print(format_etas_for_agent(results))

    print("\n=== Buses FROM HKUST ===")
    results = await get_all_hkust_etas("from")
    print(format_etas_for_agent(results))


if __name__ == "__main__":
    asyncio.run(generate_route_stops_summary())