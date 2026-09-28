"""Geospatial utilities, district mappings, and 2D Indian domains."""
from src.geo.district_mapping import INDIAN_DISTRICTS
from src.geo.spatial_utils import haversine_distance_km, find_nearest_district, aggregate_grid_to_districts
from src.geo.grid import generate_india_grid, aggregate_2d_grid_to_districts, get_geojson_feature_collection

__all__ = [
    "INDIAN_DISTRICTS",
    "haversine_distance_km",
    "find_nearest_district",
    "aggregate_grid_to_districts",
    "generate_india_grid",
    "aggregate_2d_grid_to_districts",
    "get_geojson_feature_collection"
]
