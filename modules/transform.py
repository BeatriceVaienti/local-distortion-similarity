
# ============================
# Define your own EPSG codes
# ============================
from pyproj import Transformer

# Initialize CRS transformer (EPSG:28193 to EPSG:4326)
transformer = Transformer.from_crs("epsg:28193", "epsg:4326", always_xy=True)
reverse_transformer = Transformer.from_crs("epsg:4326", "epsg:28193", always_xy=True)


def token_to_latlon(x, y):
    """Convert token coordinates to latitude/longitude."""
    lon, lat = transformer.transform(x, y)
    return lat, lon

def latlon_to_token(lat, lon):
    """Convert latitude/longitude to token coordinates."""
    x, y = reverse_transformer.transform(lon, lat)
    return x, y