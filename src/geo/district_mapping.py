"""
Indian District Geospatial Coordinates, Topography, and Coastal Proximity.
"""

from typing import Dict, Any

INDIAN_DISTRICTS: Dict[str, Dict[str, Any]] = {
    "Mumbai City": {"state": "Maharashtra", "lat": 18.96, "lon": 72.82, "elevation": 14, "coast_dist_km": 2, "zone": "West Coast"},
    "Ratnagiri": {"state": "Maharashtra", "lat": 16.99, "lon": 73.30, "elevation": 11, "coast_dist_km": 1, "zone": "West Coast"},
    "Pune": {"state": "Maharashtra", "lat": 18.52, "lon": 73.85, "elevation": 560, "coast_dist_km": 120, "zone": "Western Ghats Leeward"},
    "Wayanad": {"state": "Kerala", "lat": 11.68, "lon": 76.13, "elevation": 950, "coast_dist_km": 75, "zone": "Western Ghats Windward"},
    "Kozhikode": {"state": "Kerala", "lat": 11.25, "lon": 75.78, "elevation": 1, "coast_dist_km": 0, "zone": "West Coast"},
    "Udupi": {"state": "Karnataka", "lat": 13.34, "lon": 74.74, "elevation": 27, "coast_dist_km": 5, "zone": "West Coast"},
    "Uttara Kannada": {"state": "Karnataka", "lat": 14.80, "lon": 74.13, "elevation": 18, "coast_dist_km": 3, "zone": "West Coast"},
    "Nagpur": {"state": "Maharashtra", "lat": 21.14, "lon": 79.08, "elevation": 310, "coast_dist_km": 680, "zone": "Central India"},
    "Raipur": {"state": "Chhattisgarh", "lat": 21.25, "lon": 81.63, "elevation": 298, "coast_dist_km": 540, "zone": "Central India"},
    "Jabalpur": {"state": "Madhya Pradesh", "lat": 23.18, "lon": 79.98, "elevation": 411, "coast_dist_km": 720, "zone": "Central India"},
    "Bhopal": {"state": "Madhya Pradesh", "lat": 23.25, "lon": 77.41, "elevation": 527, "coast_dist_km": 630, "zone": "Central India"},
    "Sambalpur": {"state": "Odisha", "lat": 21.46, "lon": 83.98, "elevation": 150, "coast_dist_km": 280, "zone": "East Central India"},
    "Cuttack": {"state": "Odisha", "lat": 20.46, "lon": 85.88, "elevation": 36, "coast_dist_km": 65, "zone": "East Coast"},
    "Balasore": {"state": "Odisha", "lat": 21.49, "lon": 86.93, "elevation": 16, "coast_dist_km": 15, "zone": "East Coast"},
    "Ranchi": {"state": "Jharkhand", "lat": 23.34, "lon": 85.30, "elevation": 651, "coast_dist_km": 320, "zone": "East Central India"},
    "East Khasi Hills": {"state": "Meghalaya", "lat": 25.57, "lon": 91.88, "elevation": 1496, "coast_dist_km": 380, "zone": "Northeast Hills"},
    "Kamrup": {"state": "Assam", "lat": 26.14, "lon": 91.73, "elevation": 55, "coast_dist_km": 420, "zone": "Brahmaputra Valley"},
    "Dibrugarh": {"state": "Assam", "lat": 27.47, "lon": 94.91, "elevation": 108, "coast_dist_km": 680, "zone": "Brahmaputra Valley"},
    "Kolkata": {"state": "West Bengal", "lat": 22.57, "lon": 88.36, "elevation": 9, "coast_dist_km": 95, "zone": "Gangetic West Bengal"},
    "Darjeeling": {"state": "West Bengal", "lat": 27.04, "lon": 88.26, "elevation": 2042, "coast_dist_km": 540, "zone": "Sub-Himalayan West Bengal"},
    "Dehradun": {"state": "Uttarakhand", "lat": 30.31, "lon": 78.03, "elevation": 640, "coast_dist_km": 1100, "zone": "Western Himalayan Foothills"},
    "Shimla": {"state": "Himachal Pradesh", "lat": 31.10, "lon": 77.17, "elevation": 2205, "coast_dist_km": 1180, "zone": "Western Himalayas"},
    "New Delhi": {"state": "Delhi", "lat": 28.61, "lon": 77.20, "elevation": 216, "coast_dist_km": 980, "zone": "Northern Plains"},
    "Lucknow": {"state": "Uttar Pradesh", "lat": 26.84, "lon": 80.94, "elevation": 123, "coast_dist_km": 780, "zone": "East Uttar Pradesh"},
    "Varanasi": {"state": "Uttar Pradesh", "lat": 25.31, "lon": 82.97, "elevation": 81, "coast_dist_km": 600, "zone": "East Uttar Pradesh"},
    "Patna": {"state": "Bihar", "lat": 25.59, "lon": 85.13, "elevation": 53, "coast_dist_km": 490, "zone": "Bihar Plains"},
    "Ahmedabad": {"state": "Gujarat", "lat": 23.02, "lon": 72.57, "elevation": 53, "coast_dist_km": 75, "zone": "Gujarat Plains"},
    "Surat": {"state": "Gujarat", "lat": 21.17, "lon": 72.83, "elevation": 13, "coast_dist_km": 18, "zone": "Gujarat Coast"},
    "Jaipur": {"state": "Rajasthan", "lat": 26.91, "lon": 75.78, "elevation": 431, "coast_dist_km": 680, "zone": "East Rajasthan"},
    "Jodhpur": {"state": "Rajasthan", "lat": 26.29, "lon": 73.01, "elevation": 231, "coast_dist_km": 490, "zone": "West Rajasthan"},
    "Bengaluru Urban": {"state": "Karnataka", "lat": 12.97, "lon": 77.59, "elevation": 920, "coast_dist_km": 280, "zone": "South Interior Karnataka"},
    "Hyderabad": {"state": "Telangana", "lat": 17.38, "lon": 78.48, "elevation": 542, "coast_dist_km": 290, "zone": "Telangana Plateau"},
    "Visakhapatnam": {"state": "Andhra Pradesh", "lat": 17.68, "lon": 83.21, "elevation": 45, "coast_dist_km": 2, "zone": "Coastal Andhra"},
    "Chennai": {"state": "Tamil Nadu", "lat": 13.08, "lon": 80.27, "elevation": 7, "coast_dist_km": 1, "zone": "Tamil Nadu Coast"},
    "Coimbatore": {"state": "Tamil Nadu", "lat": 11.01, "lon": 76.96, "elevation": 411, "coast_dist_km": 120, "zone": "Tamil Nadu Interior"}
}
