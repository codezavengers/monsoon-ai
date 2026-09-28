"""
District mapping and geographic reference data for Indian meteorological zones.
"""

from typing import Dict, List, Any

# Representative Indian Meteorological Districts with coordinates, state, elevation (m), and coastal distance (km)
INDIAN_DISTRICTS: List[Dict[str, Any]] = [
    # Western Ghats / Coastal (High Orography & Coastal moisture)
    {"district": "Mumbai City", "state": "Maharashtra", "lat": 18.96, "lon": 72.82, "elevation": 14, "coast_dist_km": 2, "zone": "West Coast"},
    {"district": "Ratnagiri", "state": "Maharashtra", "lat": 16.99, "lon": 73.30, "elevation": 68, "coast_dist_km": 5, "zone": "West Coast"},
    {"district": "Pune", "state": "Maharashtra", "lat": 18.52, "lon": 73.85, "elevation": 560, "coast_dist_km": 120, "zone": "Western Ghats Leeward"},
    {"district": "Wayanad", "state": "Kerala", "lat": 11.68, "lon": 76.13, "elevation": 950, "coast_dist_km": 65, "zone": "Western Ghats Windward"},
    {"district": "Kozhikode", "state": "Kerala", "lat": 11.25, "lon": 75.78, "elevation": 12, "coast_dist_km": 1, "zone": "West Coast"},
    {"district": "Udupi", "state": "Karnataka", "lat": 13.34, "lon": 74.74, "elevation": 27, "coast_dist_km": 4, "zone": "West Coast"},
    {"district": "Uttara Kannada", "state": "Karnataka", "lat": 14.80, "lon": 74.13, "elevation": 350, "coast_dist_km": 15, "zone": "West Coast"},
    
    # Central India / Monsoon Trough & Low Depression Tracks
    {"district": "Nagpur", "state": "Maharashtra", "lat": 21.14, "lon": 79.08, "elevation": 310, "coast_dist_km": 540, "zone": "Central India"},
    {"district": "Raipur", "state": "Chhattisgarh", "lat": 21.25, "lon": 81.63, "elevation": 298, "coast_dist_km": 360, "zone": "Central India"},
    {"district": "Jabalpur", "state": "Madhya Pradesh", "lat": 23.18, "lon": 79.98, "elevation": 411, "coast_dist_km": 620, "zone": "Central India"},
    {"district": "Bhopal", "state": "Madhya Pradesh", "lat": 23.25, "lon": 77.41, "elevation": 527, "coast_dist_km": 580, "zone": "Central India"},
    {"district": "Sambalpur", "state": "Odisha", "lat": 21.46, "lon": 83.97, "elevation": 150, "coast_dist_km": 240, "zone": "East Central India"},
    {"district": "Cuttack", "state": "Odisha", "lat": 20.46, "lon": 85.88, "elevation": 36, "coast_dist_km": 50, "zone": "East Coast"},
    {"district": "Balasore", "state": "Odisha", "lat": 21.49, "lon": 86.93, "elevation": 16, "coast_dist_km": 15, "zone": "East Coast"},
    {"district": "Ranchi", "state": "Jharkhand", "lat": 23.34, "lon": 85.30, "elevation": 651, "coast_dist_km": 320, "zone": "East Central India"},
    
    # Eastern & North-Eastern India (Intense Orographic Monsoon)
    {"district": "East Khasi Hills", "state": "Meghalaya", "lat": 25.57, "lon": 91.88, "elevation": 1496, "coast_dist_km": 310, "zone": "Northeast Hills"},
    {"district": "Kamrup", "state": "Assam", "lat": 26.14, "lon": 91.73, "elevation": 55, "coast_dist_km": 380, "zone": "Brahmaputra Valley"},
    {"district": "Dibrugarh", "state": "Assam", "lat": 27.47, "lon": 94.91, "elevation": 108, "coast_dist_km": 680, "zone": "Brahmaputra Valley"},
    {"district": "Kolkata", "state": "West Bengal", "lat": 22.57, "lon": 88.36, "elevation": 9, "coast_dist_km": 75, "zone": "Gangetic West Bengal"},
    {"district": "Darjeeling", "state": "West Bengal", "lat": 27.03, "lon": 88.26, "elevation": 2042, "coast_dist_km": 510, "zone": "Sub-Himalayan West Bengal"},
    
    # Northern Plains & Western Disturbance / Foothills
    {"district": "Dehradun", "state": "Uttarakhand", "lat": 30.31, "lon": 78.03, "elevation": 640, "coast_dist_km": 1150, "zone": "Western Himalayan Foothills"},
    {"district": "Shimla", "state": "Himachal Pradesh", "lat": 31.10, "lon": 77.17, "elevation": 2206, "coast_dist_km": 1280, "zone": "Western Himalayas"},
    {"district": "New Delhi", "state": "Delhi", "lat": 28.61, "lon": 77.20, "elevation": 216, "coast_dist_km": 1050, "zone": "Northern Plains"},
    {"district": "Lucknow", "state": "Uttar Pradesh", "lat": 26.84, "lon": 80.94, "elevation": 123, "coast_dist_km": 780, "zone": "East Uttar Pradesh"},
    {"district": "Varanasi", "state": "Uttar Pradesh", "lat": 25.31, "lon": 82.97, "elevation": 81, "coast_dist_km": 590, "zone": "East Uttar Pradesh"},
    {"district": "Patna", "state": "Bihar", "lat": 25.59, "lon": 85.13, "elevation": 53, "coast_dist_km": 460, "zone": "Bihar Plains"},
    
    # Northwest / Semi-Arid & Gujarat
    {"district": "Ahmedabad", "state": "Gujarat", "lat": 23.02, "lon": 72.57, "elevation": 53, "coast_dist_km": 75, "zone": "Gujarat Plains"},
    {"district": "Surat", "state": "Gujarat", "lat": 21.17, "lon": 72.83, "elevation": 13, "coast_dist_km": 18, "zone": "Gujarat Coast"},
    {"district": "Jaipur", "state": "Rajasthan", "lat": 26.91, "lon": 75.78, "elevation": 431, "coast_dist_km": 680, "zone": "East Rajasthan"},
    {"district": "Jodhpur", "state": "Rajasthan", "lat": 26.29, "lon": 73.01, "elevation": 231, "coast_dist_km": 490, "zone": "West Rajasthan"},
    
    # Southern Peninsula
    {"district": "Bengaluru Urban", "state": "Karnataka", "lat": 12.97, "lon": 77.59, "elevation": 920, "coast_dist_km": 280, "zone": "South Interior Karnataka"},
    {"district": "Hyderabad", "state": "Telangana", "lat": 17.38, "lon": 78.48, "elevation": 542, "coast_dist_km": 290, "zone": "Telangana Plateau"},
    {"district": "Visakhapatnam", "state": "Andhra Pradesh", "lat": 17.68, "lon": 83.21, "elevation": 45, "coast_dist_km": 2, "zone": "Coastal Andhra"},
    {"district": "Chennai", "state": "Tamil Nadu", "lat": 13.08, "lon": 80.27, "elevation": 7, "coast_dist_km": 1, "zone": "Tamil Nadu Coast"},
    {"district": "Coimbatore", "state": "Tamil Nadu", "lat": 11.01, "lon": 76.96, "elevation": 411, "coast_dist_km": 120, "zone": "Tamil Nadu Interior"},
]

def get_all_districts() -> List[Dict[str, Any]]:
    """Returns the list of meteorological reference districts."""
    return INDIAN_DISTRICTS

def get_district_by_name(name: str) -> Dict[str, Any]:
    """Finds district details by case-insensitive name."""
    name_clean = name.strip().lower()
    for d in INDIAN_DISTRICTS:
        if d["district"].lower() == name_clean:
            return d
    return INDIAN_DISTRICTS[0]
