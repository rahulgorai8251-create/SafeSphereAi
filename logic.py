"""
SafeSphere AI - Safe Zone Ranking + Risk-Aware Route logic.
Demo-scale but genuinely computed (not hardcoded picks).
"""
import math

# Demo candidate safe zones around a base location (lat, lng offsets in degrees)
CANDIDATE_ZONES = [
    {"name": "Community Hall - Sector 12", "dlat": 0.020, "dlng": 0.010, "base_risk": 25, "capacity": 500},
    {"name": "Municipal School - Sector 7", "dlat": -0.008, "dlng": 0.025, "base_risk": 60, "capacity": 300},
    {"name": "Hilltop Relief Camp",         "dlat": 0.035, "dlng": -0.015, "base_risk": 15, "capacity": 800},
    {"name": "Riverside Stadium",           "dlat": 0.005, "dlng": 0.008, "base_risk": 78, "capacity": 1000},
    {"name": "District Office Complex",     "dlat": -0.018, "dlng": -0.020, "base_risk": 35, "capacity": 400},
]


def haversine_km(lat1, lng1, lat2, lng2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def rank_safe_zones(user_lat, user_lng, current_risk_score):
    """
    Score each candidate zone using hazard risk + distance + accessibility.
    Lower composite score = better. Nearest zone is NOT automatically picked
    if it has higher hazard exposure.
    """
    results = []
    for z in CANDIDATE_ZONES:
        zlat, zlng = user_lat + z["dlat"], user_lng + z["dlng"]
        dist_km = haversine_km(user_lat, user_lng, zlat, zlng)

        # Zone risk scales a bit with current event severity (worse event -> wider impact)
        adj_risk = min(100, z["base_risk"] + current_risk_score * 0.15)

        # Composite suitability score: risk weighted heaviest, then distance, then capacity
        distance_penalty = min(dist_km * 8, 40)          # cap so far-but-safe zones aren't unfairly punished
        capacity_bonus = max(0, (z["capacity"] - 300) / 100)  # bigger camps score slightly better
        composite = adj_risk * 0.6 + distance_penalty * 0.35 - capacity_bonus * 0.05

        results.append({
            "name": z["name"],
            "lat": zlat,
            "lng": zlng,
            "risk_level": "HIGH" if adj_risk >= 60 else "MODERATE" if adj_risk >= 30 else "LOW",
            "risk_score": round(adj_risk, 1),
            "distance_km": round(dist_km, 2),
            "capacity": z["capacity"],
            "composite_score": round(composite, 2),
        })

    results.sort(key=lambda r: r["composite_score"])
    best = results[0]

    nearest = min(results, key=lambda r: r["distance_km"])
    if best["name"] == nearest["name"]:
        reason = "Lowest hazard risk AND closest available option."
    else:
        reason = (
            f"Chosen over the nearer option ({nearest['name']}, "
            f"{nearest['distance_km']} km) because it has a lower estimated "
            f"hazard risk ({best['risk_score']} vs {nearest['risk_score']})."
        )

    return results, best, reason


def build_route(user_lat, user_lng, zone_lat, zone_lng, current_risk_score):
    """
    Risk-aware route prototype: generates 2 candidate paths (direct vs. detour)
    and scores them on distance + simulated hazard exposure, picks the lower-risk one.
    NOT a real routing engine (no road network) - clearly a prototype.
    """
    direct_dist = haversine_km(user_lat, user_lng, zone_lat, zone_lng)
    direct_exposure = current_risk_score * 0.9  # direct path cuts through hazard area

    # Detour path: bends away from the hazard center (midpoint offset)
    mid_lat = (user_lat + zone_lat) / 2 + 0.012
    mid_lng = (user_lng + zone_lng) / 2 - 0.010
    detour_dist = (haversine_km(user_lat, user_lng, mid_lat, mid_lng) +
                   haversine_km(mid_lat, mid_lng, zone_lat, zone_lng))
    detour_exposure = current_risk_score * 0.4  # avoids the worst-hit zone

    direct_score = direct_dist * 0.3 + direct_exposure * 0.7
    detour_score = detour_dist * 0.3 + detour_exposure * 0.7

    routes = {
        "direct": {
            "path": [(user_lat, user_lng), (zone_lat, zone_lng)],
            "distance_km": round(direct_dist, 2),
            "exposure_score": round(direct_exposure, 1),
            "composite_score": round(direct_score, 2),
        },
        "detour": {
            "path": [(user_lat, user_lng), (mid_lat, mid_lng), (zone_lat, zone_lng)],
            "distance_km": round(detour_dist, 2),
            "exposure_score": round(detour_exposure, 1),
            "composite_score": round(detour_score, 2),
        },
    }

    chosen_key = min(routes, key=lambda k: routes[k]["composite_score"])
    return routes, chosen_key
