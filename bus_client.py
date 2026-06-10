import asyncio
from datetime import datetime
from hk_bus_eta import HKEta

# ============================================================
# HKUST ROUTES — hardcoded stop indices from discovery run
# ============================================================
# Each entry: route_id as used by hk-bus-eta, stop index for HKUST
HKUST_ROUTES = {
    "11M_to_hkust": {
        "route_id": "11M+1+Hang Hau Station Public Transport Interchange+Hong Kong University of Science and Technology (North Station)",
        "stop_seq": 2,
        "label": "11M",
        "headsign": "HKUST",
        "hkust_gate": "North Gate",
    },
    "11M_to_hang_hau": {
        "route_id": "11M+1+Hong Kong University of Science and Technology (North Station)+Hang Hau Station Public Transport Interchange",
        "stop_seq": 0,
        "label": "11M",
        "headsign": "Hang Hau MTR",
        "hkust_gate": "North Gate",
    },
    "11_to_choi_hung": {
        "route_id": "11+1+Hang Hau Village+Choi Hung Station (Lung Cheung Road)",
        "stop_seq": 6,
        "label": "11",
        "headsign": "Choi Hung MTR",
        "hkust_gate": "South Gate",
    },
    "11_to_hang_hau": {
        "route_id": "11+1+Choi Hung Station (Lung Cheung Road)+Hang Hau Village",
        "stop_seq": 9,
        "label": "11",
        "headsign": "Hang Hau Village",
        "hkust_gate": "North Gate",
    },
    "12_to_sai_kung": {
        "route_id": "12+1+Po Lam+Sai Kung",
        "stop_seq": 16,
        "label": "12",
        "headsign": "Sai Kung",
        "hkust_gate": "North Gate",
    },
    "12_to_po_lam": {
        "route_id": "12+1+Sai Kung+Po Lam",
        "stop_seq": 7,
        "label": "12",
        "headsign": "Po Lam",
        "hkust_gate": "North Gate",
    },
    "91M_to_diamond_hill": {
        "route_id": "91M+1+PO LAM+DIAMOND HILL STATION",
        "stop_seq": 12,
        "label": "91M",
        "headsign": "Diamond Hill",
        "hkust_gate": "South Gate",
    },
    "91M_to_po_lam": {
        "route_id": "91M+1+DIAMOND HILL STATION+PO LAM",
        "stop_seq": 16,
        "label": "91M",
        "headsign": "Po Lam",
        "hkust_gate": "North Gate",
    },
    "792M_to_tko": {
        "route_id": "792M+1+Sai Kung+Tseung Kwan O Station",
        "stop_seq": 16,
        "label": "792M",
        "headsign": "Tseung Kwan O Station",
        "hkust_gate": "North Gate",
    },
    "792M_to_sai_kung": {
        "route_id": "792M+1+Tseung Kwan O Station+Sai Kung",
        "stop_seq": 13,
        "label": "792M",
        "headsign": "Sai Kung",
        "hkust_gate": "North Gate",
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
            "label": route["label"],
            "headsign": route["headsign"],
            "hkust_gate": route["hkust_gate"],
            "etas": etas,
        }
    except Exception as e:
        return {"route": route["label"], "label": route["label"], "headsign": route.get("headsign",""), "hkust_gate": route.get("hkust_gate",""), "error": str(e), "etas": []}
    
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

async def get_all_hkust_etas(filter: str = "all") -> list:
    """
    Get ETAs for all routes or filtered by keyword in route key.
    filter: "all", "to_hkust", "from_hkust", "tko", "diamond_hill", "po_lam", "hang_hau", "sai_kung" etc.
    """
    if filter == "all":
        keys = list(HKUST_ROUTES.keys())
    elif filter == "to_hkust":
        keys = [k for k in HKUST_ROUTES if "to_hkust" in k]
    else:
        keys = [k for k in HKUST_ROUTES if filter in k]

    results = []
    for key in keys:
        eta_data = await get_eta_for_route(key)
        if eta_data.get("etas") is not None:
            results.append(eta_data)
    return results

def format_etas_for_agent(eta_results: list) -> str:
    if not eta_results:
        return "No bus ETAs available right now (service may be outside operating hours)."

    lines = []
    for r in eta_results:
        if r.get("error"):
            continue
        etas = r["etas"]
        gate = r.get("hkust_gate", "")
        headsign = r.get("headsign", "")
        gate_str = f" [{gate}]" if gate else ""
        if not etas:
            lines.append(f"• {r['label']} → {headsign}{gate_str}: No upcoming buses")
            continue
        eta_strs = [f"{e['minutes']} min ({e['time']})" for e in etas[:3]]
        lines.append(f"• {r['label']} → {headsign}{gate_str}: {', '.join(eta_strs)}")
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
    print("\n=== All HKUST Bus ETAs ===")
    results = await get_all_hkust_etas("all")
    print(format_etas_for_agent(results))

# Pre-load on import so first query doesn't time out
_hketa = HKEta()

if __name__ == "__main__":
    asyncio.run(test())