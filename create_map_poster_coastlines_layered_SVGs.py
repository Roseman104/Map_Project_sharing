#!/usr/bin/env python3
"""
City Map Poster Generator with Coastlines

This module generates beautiful, minimalist map posters for any city in the world.
It fetches OpenStreetMap data using OSMnx, applies customizable themes, and creates
high-quality poster-ready images with roads, water features, parks, and coastlines.
"""

import argparse
import asyncio
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import cast

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
from geopandas import GeoDataFrame
from geopy.geocoders import Nominatim
from lat_lon_parser import parse
from matplotlib.font_manager import FontProperties
from networkx import MultiDiGraph
from shapely.geometry import Point, LineString, MultiLineString, MultiPolygon, Polygon
from shapely.affinity import rotate
from shapely.ops import unary_union
from tqdm import tqdm

from font_management import load_fonts


class CacheError(Exception):
    """Raised when a cache operation fails."""


CACHE_DIR_PATH = os.environ.get("CACHE_DIR", "cache")
CACHE_DIR = Path(CACHE_DIR_PATH)
CACHE_DIR.mkdir(exist_ok=True)

THEMES_DIR = "themes"
FONTS_DIR = "fonts"
POSTERS_DIR = "posters"

FILE_ENCODING = "utf-8"

FONTS = load_fonts()


def _cache_path(key: str) -> str:
    """
    Generate a safe cache file path from a cache key.

    Args:
        key: Cache key identifier

    Returns:
        Path to cache file with .pkl extension
    """
    safe = key.replace(os.sep, "_")
    return os.path.join(CACHE_DIR, f"{safe}.pkl")


def cache_get(key: str):
    """
    Retrieve a cached object by key.

    Args:
        key: Cache key identifier

    Returns:
        Cached object if found, None otherwise

    Raises:
        CacheError: If cache read operation fails
    """
    try:
        path = _cache_path(key)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise CacheError(f"Cache read failed: {e}") from e


def cache_set(key: str, value):
    """
    Store an object in the cache.

    Args:
        key: Cache key identifier
        value: Object to cache (must be picklable)

    Raises:
        CacheError: If cache write operation fails
    """
    try:
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
        path = _cache_path(key)
        with open(path, "wb") as f:
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        raise CacheError(f"Cache write failed: {e}") from e


# Font loading now handled by font_management.py module


def is_latin_script(text):
    """
    Check if text is primarily Latin script.
    Used to determine if letter-spacing should be applied to city names.

    :param text: Text to analyze
    :return: True if text is primarily Latin script, False otherwise
    """
    if not text:
        return True

    latin_count = 0
    total_alpha = 0

    for char in text:
        if char.isalpha():
            total_alpha += 1
            # Latin Unicode ranges:
            # - Basic Latin: U+0000 to U+007F
            # - Latin-1 Supplement: U+0080 to U+00FF
            # - Latin Extended-A: U+0100 to U+017F
            # - Latin Extended-B: U+0180 to U+024F
            if ord(char) < 0x250:
                latin_count += 1

    # If no alphabetic characters, default to Latin (numbers, symbols, etc.)
    if total_alpha == 0:
        return True

    # Consider it Latin if >80% of alphabetic characters are Latin
    return (latin_count / total_alpha) > 0.8


def generate_output_filename(city, theme_name, output_format):
    """
    Generate unique output filename with city, theme, and datetime.
    """
    if not os.path.exists(POSTERS_DIR):
        os.makedirs(POSTERS_DIR)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    city_slug = city.lower().replace(" ", "_")
    
    # Map format to file extension (layered-svg should use .svg extension)
    ext = output_format.lower()
    if ext == "layered-svg":
        ext = "svg"
    
    filename = f"{city_slug}_{theme_name}_{timestamp}.{ext}"
    return os.path.join(POSTERS_DIR, filename)


def get_available_themes():
    """
    Scans the themes directory and returns a list of available theme names.
    """
    if not os.path.exists(THEMES_DIR):
        os.makedirs(THEMES_DIR)
        return []

    themes = []
    for file in sorted(os.listdir(THEMES_DIR)):
        if file.endswith(".json"):
            theme_name = file[:-5]  # Remove .json extension
            themes.append(theme_name)
    return themes


def load_theme(theme_name="terracotta"):
    """
    Load theme from JSON file in themes directory.
    """
    theme_file = os.path.join(THEMES_DIR, f"{theme_name}.json")

    if not os.path.exists(theme_file):
        print(f"⚠ Theme file '{theme_file}' not found. Using default terracotta theme.")
        # Fallback to embedded terracotta theme
        return {
            "name": "Terracotta",
            "description": "Mediterranean warmth - burnt orange and clay tones on cream",
            "bg": "#F5EDE4",
            "text": "#8B4513",
            "gradient_color": "#F5EDE4",
            "water": "#A8C4C4",
            "parks": "#E8E0D0",
            "coastline": "#4A90A4",
            "road_motorway": "#A0522D",
            "road_primary": "#B8653A",
            "road_secondary": "#C9846A",
            "road_tertiary": "#D9A08A",
            "road_residential": "#E5C4B0",
            "road_default": "#D9A08A",
        }

    with open(theme_file, "r", encoding=FILE_ENCODING) as f:
        theme = json.load(f)
        print(f"✓ Loaded theme: {theme.get('name', theme_name)}")
        if "description" in theme:
            print(f"  {theme['description']}")
        # Add default coastline color if not present in theme
        if "coastline" not in theme:
            theme["coastline"] = theme.get("water", "#4A90A4")
        return theme


# Load theme (can be changed via command line or input)
THEME = dict[str, str]()  # Will be loaded later


def create_gradient_fade(ax, color, location="bottom", zorder=10):
    """
    Creates a fade effect at the top or bottom of the map.
    """
    vals = np.linspace(0, 1, 256).reshape(-1, 1)
    gradient = np.hstack((vals, vals))

    rgb = mcolors.to_rgb(color)
    my_colors = np.zeros((256, 4))
    my_colors[:, 0] = rgb[0]
    my_colors[:, 1] = rgb[1]
    my_colors[:, 2] = rgb[2]

    if location == "bottom":
        my_colors[:, 3] = np.linspace(1, 0, 256)
        extent_y_start = 0
        extent_y_end = 0.25
    else:
        my_colors[:, 3] = np.linspace(0, 1, 256)
        extent_y_start = 0.75
        extent_y_end = 1.0

    custom_cmap = mcolors.ListedColormap(my_colors)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0]

    y_bottom = ylim[0] + y_range * extent_y_start
    y_top = ylim[0] + y_range * extent_y_end

    ax.imshow(
        gradient,
        extent=[xlim[0], xlim[1], y_bottom, y_top],
        aspect="auto",
        cmap=custom_cmap,
        zorder=zorder,
        origin="lower",
    )


def draw_railroad_ties(ax, line_geom, color, tie_spacing=50, tie_length=80, tie_width=0.5, line_width=0.6, zorder=2.5):
    """
    Draw a railroad line with perpendicular ties (cross marks).
    
    Args:
        ax: Matplotlib axis to draw on
        line_geom: Shapely LineString representing the railroad
        color: Color for the railroad
        tie_spacing: Distance between ties in map units (default: 50)
        tie_length: Length of each tie perpendicular to track (default: 80)
        tie_width: Width of tie lines (default: 0.5)
        line_width: Width of main track line (default: 0.6)
        zorder: Z-order for drawing (default: 2.5)
    """
    from shapely.geometry import LineString
    import numpy as np
    
    if not isinstance(line_geom, LineString) or line_geom.is_empty:
        return
    
    # Draw the main railroad line
    x, y = line_geom.xy
    ax.plot(x, y, color=color, linewidth=line_width, alpha=0.9, zorder=zorder,
           solid_capstyle='butt', solid_joinstyle='miter')
    
    # Calculate the total length and number of ties
    total_length = line_geom.length
    if total_length < tie_spacing:
        return
    
    num_ties = int(total_length / tie_spacing)
    
    # Place ties along the line
    for i in range(num_ties + 1):
        distance = i * tie_spacing
        if distance > total_length:
            break
        
        # Get point along the line
        point = line_geom.interpolate(distance)
        
        # Calculate the tangent angle at this point
        # Sample a small section around the point to get direction
        epsilon = min(1.0, total_length * 0.001)
        if distance + epsilon <= total_length:
            next_point = line_geom.interpolate(distance + epsilon)
        else:
            next_point = line_geom.interpolate(distance)
            point = line_geom.interpolate(max(0, distance - epsilon))
        
        # Calculate perpendicular direction
        dx = next_point.x - point.x
        dy = next_point.y - point.y
        length = np.sqrt(dx**2 + dy**2)
        
        if length > 0:
            # Normalize and rotate 90 degrees for perpendicular
            perp_dx = -dy / length
            perp_dy = dx / length
            
            # Calculate tie endpoints
            half_tie = tie_length / 2
            tie_x1 = point.x + perp_dx * half_tie
            tie_y1 = point.y + perp_dy * half_tie
            tie_x2 = point.x - perp_dx * half_tie
            tie_y2 = point.y - perp_dy * half_tie
            
            # Draw the tie
            ax.plot([tie_x1, tie_x2], [tie_y1, tie_y2], 
                   color=color, linewidth=tie_width, alpha=0.9, zorder=zorder,
                   solid_capstyle='butt')


def create_crosshatch_lines(geometry, spacing=100, angle=45, bounds=None):
    """
    Generate cross-hatch lines for a polygon or multipolygon.
    
    Args:
        geometry: Shapely Polygon or MultiPolygon to hatch
        spacing: Distance between hatch lines in map units (default: 100)
        angle: Angle of hatch lines in degrees (default: 45)
        bounds: Optional bounds tuple (minx, miny, maxx, maxy) to limit hatch area
    
    Returns:
        List of LineString objects representing the hatch pattern
    """
    if geometry is None or geometry.is_empty:
        return []
    
    # Handle MultiPolygon
    if isinstance(geometry, MultiPolygon):
        all_lines = []
        for poly in geometry.geoms:
            all_lines.extend(create_crosshatch_lines(poly, spacing, angle, bounds))
        return all_lines
    
    if not isinstance(geometry, Polygon):
        return []
    
    # Get bounds
    if bounds is None:
        minx, miny, maxx, maxy = geometry.bounds
    else:
        minx, miny, maxx, maxy = bounds
    
    # Calculate the diagonal length to ensure lines cover the entire area
    width = maxx - minx
    height = maxy - miny
    diagonal = np.sqrt(width**2 + height**2)
    
    # Generate parallel lines across the bounding box
    lines = []
    num_lines = int(diagonal / spacing) + 2
    
    for i in range(-num_lines, num_lines):
        offset = i * spacing
        # Create a horizontal line and rotate it
        y = miny + height / 2 + offset
        line = LineString([(minx - diagonal, y), (maxx + diagonal, y)])
        
        # Rotate the line around the center of the bounds
        center_x = (minx + maxx) / 2
        center_y = (miny + maxy) / 2
        rotated_line = rotate(line, angle, origin=(center_x, center_y))
        
        # Intersect with the polygon
        try:
            intersection = rotated_line.intersection(geometry)
            if not intersection.is_empty:
                if isinstance(intersection, LineString):
                    lines.append(intersection)
                elif isinstance(intersection, MultiLineString):
                    lines.extend(list(intersection.geoms))
        except Exception:
            continue
    
    return lines


def get_edge_colors_by_type(g):
    """
    Assigns colors to edges based on road type hierarchy.
    Returns a list of colors corresponding to each edge in the graph.
    """
    edge_colors = []

    for _u, _v, data in g.edges(data=True):
        # Get the highway type (can be a list or string)
        highway = data.get('highway', 'unclassified')

        # Handle list of highway types (take the first one)
        if isinstance(highway, list):
            highway = highway[0] if highway else 'unclassified'

        # Assign color based on road type
        if highway in ["motorway", "motorway_link"]:
            color = THEME["road_motorway"]
        elif highway in ["trunk", "trunk_link", "primary", "primary_link"]:
            color = THEME["road_primary"]
        elif highway in ["secondary", "secondary_link"]:
            color = THEME["road_secondary"]
        elif highway in ["tertiary", "tertiary_link"]:
            color = THEME["road_tertiary"]
        elif highway in ["residential", "living_street", "unclassified"]:
            color = THEME["road_residential"]
        else:
            color = THEME['road_default']

        edge_colors.append(color)

    return edge_colors


def get_edge_widths_by_type(g):
    """
    Assigns line widths to edges based on road type.
    Major roads get thicker lines.
    """
    edge_widths = []

    for _u, _v, data in g.edges(data=True):
        highway = data.get('highway', 'unclassified')

        if isinstance(highway, list):
            highway = highway[0] if highway else 'unclassified'

        # Assign width based on road importance
        if highway in ["motorway", "motorway_link"]:
            #width = 1.0
            width = 0.8
        elif highway in ["trunk", "trunk_link", "primary", "primary_link"]:
            #width = 0.8
            width = 0.6
        elif highway in ["secondary", "secondary_link"]:
            #width = 0.6
            width = 0.4
        elif highway in ["tertiary", "tertiary_link"]:
            #width = 0.5
            width = 0.3
        else:
            width = 0.2

        edge_widths.append(width)

    return edge_widths


def get_coordinates(city, country):
    """
    Fetches coordinates for a given city and country using geopy.
    Includes rate limiting to be respectful to the geocoding service.
    """
    coords = f"coords_{city.lower()}_{country.lower()}"
    cached = cache_get(coords)
    if cached:
        print(f"✓ Using cached coordinates for {city}, {country}")
        print(f"✓ Coordinates: {cached[0]}, {cached[1]}")
        return cached

    print("Looking up coordinates...")
    geolocator = Nominatim(user_agent="city_map_poster", timeout=10)

    # Add a small delay to respect Nominatim's usage policy
    time.sleep(1)

    try:
        location = geolocator.geocode(f"{city}, {country}")
    except Exception as e:
        raise ValueError(f"Geocoding failed for {city}, {country}: {e}") from e

    # If geocode returned a coroutine in some environments, run it to get the result.
    if asyncio.iscoroutine(location):
        try:
            location = asyncio.run(location)
        except RuntimeError as exc:
            # If an event loop is already running, try using it to complete the coroutine.
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Running event loop in the same thread; raise a clear error.
                raise RuntimeError(
                    "Geocoder returned a coroutine while an event loop is already running. "
                    "Run this script in a synchronous environment."
                ) from exc
            location = loop.run_until_complete(location)

    if location:
        # Use getattr to safely access address (helps static analyzers)
        addr = getattr(location, "address", None)
        if addr:
            print(f"✓ Found: {addr}")
        else:
            print("✓ Found location (address not available)")
        print(f"✓ Coordinates: {location.latitude}, {location.longitude}")
        try:
            cache_set(coords, (location.latitude, location.longitude))
        except CacheError as e:
            print(e)
        return (location.latitude, location.longitude)

    raise ValueError(f"Could not find coordinates for {city}, {country}")


def get_crop_limits(g_proj, center_lat_lon, fig, dist):
    """
    Crop inward to preserve aspect ratio while guaranteeing
    full coverage of the requested radius.
    """
    lat, lon = center_lat_lon

    # Project center point into graph CRS
    center = (
        ox.projection.project_geometry(
            Point(lon, lat),
            crs="EPSG:4326",
            to_crs=g_proj.graph["crs"]
        )[0]
    )
    center_x, center_y = center.x, center.y

    fig_width, fig_height = fig.get_size_inches()
    aspect = fig_width / fig_height

    # Start from the *requested* radius
    half_x = dist
    half_y = dist

    # Cut inward to match aspect
    if aspect > 1:  # landscape → reduce height
        half_y = half_x / aspect
    else:  # portrait → reduce width
        half_x = half_y * aspect

    return (
        (center_x - half_x, center_x + half_x),
        (center_y - half_y, center_y + half_y),
    )


def fetch_graph(point, dist) -> MultiDiGraph | None:
    """
    Fetch street network graph from OpenStreetMap.

    Uses caching to avoid redundant downloads. Fetches all network types
    within the specified distance from the center point.

    Args:
        point: (latitude, longitude) tuple for center point
        dist: Distance in meters from center point

    Returns:
        MultiDiGraph of street network, or None if fetch fails
    """
    lat, lon = point
    graph = f"graph_{lat}_{lon}_{dist}"
    cached = cache_get(graph)
    if cached is not None:
        print("✓ Using cached street network")
        return cast(MultiDiGraph, cached)

    try:
        g = ox.graph_from_point(point, dist=dist, dist_type='bbox', network_type='all', truncate_by_edge=True)
        # Rate limit between requests
        time.sleep(0.5)
        try:
            cache_set(graph, g)
        except CacheError as e:
            print(e)
        return g
    except Exception as e:
        print(f"OSMnx error while fetching graph: {e}")
        return None


def fetch_features(point, dist, tags, name) -> GeoDataFrame | None:
    """
    Fetch geographic features (water, parks, etc.) from OpenStreetMap.

    Uses caching to avoid redundant downloads. Fetches features matching
    the specified OSM tags within distance from center point.

    Args:
        point: (latitude, longitude) tuple for center point
        dist: Distance in meters from center point
        tags: Dictionary of OSM tags to filter features
        name: Name for this feature type (for caching and logging)

    Returns:
        GeoDataFrame of features, or None if fetch fails
    """
    lat, lon = point
    tag_str = "_".join(tags.keys())
    features = f"{name}_{lat}_{lon}_{dist}_{tag_str}"
    cached = cache_get(features)
    if cached is not None:
        print(f"✓ Using cached {name}")
        return cast(GeoDataFrame, cached)

    try:
        data = ox.features_from_point(point, tags=tags, dist=dist)
        # Rate limit between requests
        time.sleep(0.3)
        try:
            cache_set(features, data)
        except CacheError as e:
            print(e)
        return data
    except Exception as e:
        print(f"OSMnx error while fetching features: {e}")
        return None


def create_poster(
    city,
    country,
    point,
    dist,
    output_file,
    output_format,
    width=12,
    height=16,
    country_label=None,
    name_label=None,
    display_city=None,
    display_country=None,
    fonts=None,
    coastline_width=0.8,
    crosshatch_water=False,
    crosshatch_spacing=100,
    crosshatch_angle=45,
    crosshatch_width=0.3,
    pen_plot_mode=False,
    crop_buffer=0.0,
):
    """
    Generate a complete map poster with roads, water, parks, coastlines, and typography.

    Creates a high-quality poster by fetching OSM data, rendering map layers,
    applying the current theme, and adding text labels with coordinates.

    Args:
        city: City name for display on poster
        country: Country name for display on poster
        point: (latitude, longitude) tuple for map center
        dist: Map radius in meters
        output_file: Path where poster will be saved
        output_format: File format ('png', 'svg', or 'pdf')
        width: Poster width in inches (default: 12)
        height: Poster height in inches (default: 16)
        country_label: Optional override for country text on poster
        _name_label: Optional override for city name (unused, reserved for future use)
        coastline_width: Width of coastline lines (default: 1.5)
        crosshatch_water: Enable cross-hatching for water features (default: False)
        crosshatch_spacing: Distance between hatch lines in meters (default: 100)
        crosshatch_angle: Angle of hatch lines in degrees (default: 45)
        crosshatch_width: Width of hatch lines (default: 0.3)
        pen_plot_mode: Use solid black lines for pen plotting (default: True)

    Raises:
        RuntimeError: If street network data cannot be retrieved
    """
    # Handle display names for i18n support
    # Priority: display_city/display_country > name_label/country_label > city/country
    display_city = display_city or name_label or city
    display_country = display_country or country_label or country

    print(f"\nGenerating map for {city}, {country}...")

    # Progress bar for data fetching
    with tqdm(
        total=8,
        desc="Fetching map data",
        unit="step",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
    ) as pbar:
        # 1. Fetch Street Network
        pbar.set_description("Downloading street network")
        # Add modest buffer to ensure all visible area features are fetched
        # Use 1.15x factor to provide just enough coverage to capture all visible roads/features
        compensated_dist = dist * (max(height, width) / min(height, width)) * 1.15
        g = fetch_graph(point, compensated_dist)
        if g is None:
            raise RuntimeError("Failed to retrieve street network data.")
        pbar.update(1)

        # 2. Fetch Water Features
        pbar.set_description("Downloading water features")
        water = fetch_features(
            point,
            compensated_dist,
            tags={"natural": ["water", "bay", "strait"], "waterway": "riverbank"},
            name="water",
        )
        pbar.update(1)

        # 3. Fetch Parks
        pbar.set_description("Downloading parks/green spaces")
        parks = fetch_features(
            point,
            compensated_dist,
            tags={"leisure": "park", "landuse": "grass"},
            name="parks",
        )
        pbar.update(1)

        # 4. Fetch Coastlines
        pbar.set_description("Downloading coastlines")
        coastlines = fetch_features(
            point,
            compensated_dist,
            tags={"natural": "coastline"},
            name="coastlines",
        )
        pbar.update(1)

        # 5. Fetch Ocean/Sea/Bay areas (larger water bodies and tidal areas)
        pbar.set_description("Downloading ocean/bay areas")
        ocean = fetch_features(
            point,
            compensated_dist,
            tags={"natural": ["bay", "strait"], "place": ["sea", "ocean"], "water": ["bay", "lagoon", "ocean"]},
            name="ocean",
        )
        pbar.update(1)

        # 6. Fetch Islands and landmass features
        pbar.set_description("Downloading islands/landmass")
        islands = fetch_features(
            point,
            compensated_dist,
            tags={"place": ["island", "islet"], "natural": "land"},
            name="islands",
        )
        pbar.update(1)

        # 7. Fetch Airport/Aeroway features
        pbar.set_description("Downloading airports/runways")
        aeroways = fetch_features(
            point,
            compensated_dist,
            tags={"aeroway": ["runway", "taxiway", "aerodrome"]},
            name="aeroways",
        )
        pbar.update(1)

        # 8. Fetch Railroad/Railway features
        pbar.set_description("Downloading railroads/railways")
        railways = fetch_features(
            point,
            compensated_dist,
            tags={"railway": ["rail", "light_rail", "subway", "tram", "monorail", "narrow_gauge"]},
            name="railways",
        )
        pbar.update(1)

    print("✓ All data retrieved successfully!")
    
    # Debug output for water features
    if crosshatch_water:
        water_count = len(water) if water is not None and not water.empty else 0
        ocean_count = len(ocean) if ocean is not None and not ocean.empty else 0
        islands_count = len(islands) if islands is not None and not islands.empty else 0
        aeroway_count = len(aeroways) if aeroways is not None and not aeroways.empty else 0
        railway_count = len(railways) if railways is not None and not railways.empty else 0
        print(f"  Water features found: {water_count}")
        print(f"  Ocean/bay features found: {ocean_count}")
        print(f"  Islands/landmass found: {islands_count}")
        print(f"  Aeroway features found: {aeroway_count}")
        print(f"  Railway features found: {railway_count}")

    # Handle layered-svg format separately (skip matplotlib rendering)
    if output_format == "layered-svg":
        from svg_export import create_layered_svg_poster_coastlines

        # Project data to metric CRS
        g_proj = ox.project_graph(g)

        # Project water features if present
        water_proj = None
        if water is not None and not water.empty:
            water_polys = water[water.geometry.type.isin(["Polygon", "MultiPolygon"])]
            if not water_polys.empty:
                try:
                    water_proj = ox.projection.project_gdf(water_polys)
                except Exception:
                    water_proj = water_polys.to_crs(g_proj.graph['crs'])

        # Project parks features if present
        parks_proj = None
        if parks is not None and not parks.empty:
            parks_polys = parks[parks.geometry.type.isin(["Polygon", "MultiPolygon"])]
            if not parks_polys.empty:
                try:
                    parks_proj = ox.projection.project_gdf(parks_polys)
                except Exception:
                    parks_proj = parks_polys.to_crs(g_proj.graph['crs'])

        # Project coastlines if present
        coastlines_proj = None
        if coastlines is not None and not coastlines.empty:
            coastline_lines = coastlines[coastlines.geometry.type.isin(["LineString", "MultiLineString"])]
            if not coastline_lines.empty:
                try:
                    coastlines_proj = ox.projection.project_gdf(coastline_lines)
                except Exception:
                    coastlines_proj = coastline_lines.to_crs(g_proj.graph['crs'])

        # Project ocean features if present
        ocean_proj = None
        if ocean is not None and not ocean.empty:
            ocean_polys = ocean[ocean.geometry.type.isin(["Polygon", "MultiPolygon"])]
            if not ocean_polys.empty:
                try:
                    ocean_proj = ox.projection.project_gdf(ocean_polys)
                except Exception:
                    ocean_proj = ocean_polys.to_crs(g_proj.graph['crs'])

        # Project islands if present
        islands_proj = None
        if islands is not None and not islands.empty:
            islands_polys = islands[islands.geometry.type.isin(["Polygon", "MultiPolygon"])]
            if not islands_polys.empty:
                try:
                    islands_proj = ox.projection.project_gdf(islands_polys)
                except Exception:
                    islands_proj = islands_polys.to_crs(g_proj.graph['crs'])

        # Project aeroways if present
        aeroways_proj = None
        if aeroways is not None and not aeroways.empty:
            try:
                aeroways_proj = ox.projection.project_gdf(aeroways)
            except Exception:
                aeroways_proj = aeroways.to_crs(g_proj.graph['crs'])

        # Project railways if present
        railways_proj = None
        if railways is not None and not railways.empty:
            railway_lines = railways[railways.geometry.type.isin(["LineString", "MultiLineString"])]
            if not railway_lines.empty:
                try:
                    railways_proj = ox.projection.project_gdf(railway_lines)
                except Exception:
                    railways_proj = railway_lines.to_crs(g_proj.graph['crs'])

        # Create layered SVG with all features
        create_layered_svg_poster_coastlines(
            g_proj, water_proj, parks_proj, coastlines_proj, ocean_proj, 
            islands_proj, aeroways_proj, railways_proj,
            width, height, output_file, THEME,
            coastline_width=coastline_width,
            center_point=point,
            dist=dist,
            crop_buffer=crop_buffer
        )
        return

    # 2. Setup Plot
    print("Rendering map...")
    
    # For pen plotting, use pure white/black for clean bitmaps
    if pen_plot_mode and crosshatch_water:
        bg_color = "#FFFFFF"  # Pure white
        line_color = "#000000"  # Pure black
        print("  Using pen plot mode: solid black lines on white")
    else:
        bg_color = THEME["bg"]
        line_color = THEME['water']
    
    fig, ax = plt.subplots(figsize=(width, height), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    ax.set_position((0.0, 0.0, 1.0, 1.0))

    # Project graph to a metric CRS so distances and aspect are linear (meters)
    g_proj = ox.project_graph(g)

    # 3. Plot Layers
    # Layer 1: Polygons (filter to only plot polygon/multipolygon geometries, not points)
    if water is not None and not water.empty:
        # Filter to only polygon/multipolygon geometries to avoid point features showing as dots
        water_polys = water[water.geometry.type.isin(["Polygon", "MultiPolygon"])]
        if not water_polys.empty:
            # Project water features in the same CRS as the graph
            try:
                water_polys = ox.projection.project_gdf(water_polys)
            except Exception:
                water_polys = water_polys.to_crs(g_proj.graph['crs'])
            
            if crosshatch_water:
                # Use cross-hatching for pen plotting
                print("Generating cross-hatch pattern for water...")
                all_hatch_lines = []
                for geom in water_polys.geometry:
                    hatch_lines = create_crosshatch_lines(
                        geom, 
                        spacing=crosshatch_spacing,
                        angle=crosshatch_angle
                    )
                    all_hatch_lines.extend(hatch_lines)
                
                # Plot the hatch lines
                for line in all_hatch_lines:
                    x, y = line.xy
                    ax.plot(x, y, color=line_color, linewidth=crosshatch_width, 
                           solid_capstyle='round', solid_joinstyle='round', antialiased=False, zorder=0.5)
                
                # Plot the outlines of water polygons
                water_polys.plot(ax=ax, facecolor='none', edgecolor=line_color, linewidth=crosshatch_width * 2, zorder=0.6)
            else:
                # Show only subtle outlines (no fill, no crosshatch)
                water_polys.plot(ax=ax, facecolor='none', edgecolor=THEME['water'], linewidth=0.3, alpha=0.4, zorder=0.5)

    # Ocean/Sea areas (with different hatch angle if cross-hatching)
    if ocean is not None and not ocean.empty:
        ocean_polys = ocean[ocean.geometry.type.isin(["Polygon", "MultiPolygon"])]
        if not ocean_polys.empty:
            try:
                ocean_polys = ox.projection.project_gdf(ocean_polys)
            except Exception:
                ocean_polys = ocean_polys.to_crs(g_proj.graph['crs'])
            
            if crosshatch_water:
                # Use cross-hatching with a different angle for ocean/sea
                print("Generating cross-hatch pattern for ocean/sea...")
                all_ocean_hatch = []
                # Use perpendicular angle for ocean to distinguish from rivers/lakes
                ocean_angle = crosshatch_angle - 90 if crosshatch_angle >= 90 else crosshatch_angle + 90
                for geom in ocean_polys.geometry:
                    hatch_lines = create_crosshatch_lines(
                        geom,
                        spacing=crosshatch_spacing,
                        angle=ocean_angle
                    )
                    all_ocean_hatch.extend(hatch_lines)
                
                # Plot the ocean hatch lines
                for line in all_ocean_hatch:
                    x, y = line.xy
                    ax.plot(x, y, color=line_color, linewidth=crosshatch_width,
                           solid_capstyle='round', solid_joinstyle='round', antialiased=False, zorder=0.5)
                
                # Plot the outlines of ocean polygons
                ocean_polys.plot(ax=ax, facecolor='none', edgecolor=line_color, linewidth=crosshatch_width * 2, zorder=0.6)
            else:
                # Show only subtle outlines (no fill, no crosshatch)
                ocean_polys.plot(ax=ax, facecolor='none', edgecolor=THEME['water'], linewidth=0.3, alpha=0.4, zorder=0.5)

    if parks is not None and not parks.empty:
        # Filter to only polygon/multipolygon geometries to avoid point features showing as dots
        parks_polys = parks[parks.geometry.type.isin(["Polygon", "MultiPolygon"])]
        if not parks_polys.empty:
            # Project park features in the same CRS as the graph
            try:
                parks_polys = ox.projection.project_gdf(parks_polys)
            except Exception:
                parks_polys = parks_polys.to_crs(g_proj.graph['crs'])
            if crosshatch_water and pen_plot_mode:
                # For pen plotting, skip park fill - just outline if needed
                pass
            else:
                parks_polys.plot(ax=ax, facecolor=THEME['parks'], edgecolor='none', zorder=0.8)
            
            # Outline parks/landmasses if cross-hatching water (for pen plotting)
            if crosshatch_water:
                parks_polys.plot(ax=ax, facecolor='none', edgecolor=line_color, linewidth=crosshatch_width * 2, zorder=0.9)

    # Layer 2: Roads with hierarchy coloring
    print("Applying road hierarchy colors...")
    if pen_plot_mode and crosshatch_water:
        # Use uniform black for all roads in pen plot mode
        edge_colors = [line_color] * len(list(g_proj.edges()))
    else:
        edge_colors = get_edge_colors_by_type(g_proj)
    edge_widths = get_edge_widths_by_type(g_proj)

    # Determine cropping limits to maintain the poster aspect ratio
    # Use the original 'dist' here to crop to the requested area, not the larger fetch area
    crop_xlim, crop_ylim = get_crop_limits(g_proj, point, fig, dist)
    # Plot the projected graph and then apply the cropped limits
    ox.plot_graph(
        g_proj, ax=ax, bgcolor=bg_color,
        node_size=0,
        edge_color=edge_colors,
        edge_linewidth=edge_widths,
        show=False,
        close=False,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(crop_xlim)
    ax.set_ylim(crop_ylim)

    # Layer 3: Coastlines (render after roads so they appear on top)
    if coastlines is not None and not coastlines.empty:
        print("Rendering coastlines...")
        # Filter to only LineString and MultiLineString geometries
        coastline_lines = coastlines[coastlines.geometry.type.isin(["LineString", "MultiLineString"])]
        if not coastline_lines.empty:
            # Project coastline features in the same CRS as the graph
            try:
                coastline_lines = ox.projection.project_gdf(coastline_lines)
            except Exception:
                coastline_lines = coastline_lines.to_crs(g_proj.graph['crs'])
            
            # For pen plotting with crosshatch, use thicker coastlines to define islands/land
            coastline_color = line_color if (pen_plot_mode and crosshatch_water) else THEME.get('coastline', THEME['water'])
            if crosshatch_water:
                coastline_lines.plot(
                    ax=ax,
                    edgecolor=coastline_color,
                    linewidth=coastline_width,
                    zorder=3,
                )
            else:
                coastline_lines.plot(
                    ax=ax,
                    edgecolor=coastline_color,
                    linewidth=coastline_width,
                    zorder=2,
                )

    # Layer 3.5: Islands and landmass outlines (for pen plotting)
    if crosshatch_water and islands is not None and not islands.empty:
        print("Rendering island outlines...")
        islands_polys = islands[islands.geometry.type.isin(["Polygon", "MultiPolygon"])]
        if not islands_polys.empty:
            try:
                islands_polys = ox.projection.project_gdf(islands_polys)
            except Exception:
                islands_polys = islands_polys.to_crs(g_proj.graph['crs'])
            
            # Draw outlines for islands
            island_color = line_color if pen_plot_mode else THEME.get('text', '#000000')
            islands_polys.plot(
                ax=ax,
                facecolor='none',
                edgecolor=island_color,
                linewidth=coastline_width * 0.4,
                zorder=3.5,
            )
            print(f"  Drew outlines for {len(islands_polys)} island features")

    # Layer 3.6: Airport runways and taxiways
    if aeroways is not None and not aeroways.empty:
        print("Rendering airport runways...")
        try:
            aeroways_proj = ox.projection.project_gdf(aeroways)
        except Exception:
            aeroways_proj = aeroways.to_crs(g_proj.graph['crs'])
        
        # Separate runways from taxiways for different rendering
        runways = aeroways_proj[aeroways_proj.get('aeroway', '') == 'runway']
        taxiways = aeroways_proj[aeroways_proj.get('aeroway', '') == 'taxiway']
        aerodromes = aeroways_proj[aeroways_proj.get('aeroway', '') == 'aerodrome']
        
        # Render aerodrome areas (airport boundaries) first if they exist
        if not aerodromes.empty:
            aerodrome_polys = aerodromes[aerodromes.geometry.type.isin(["Polygon", "MultiPolygon"])]
            if not aerodrome_polys.empty:
                aerodrome_color = line_color if pen_plot_mode else THEME.get('text', '#000000')
                aerodrome_polys.plot(
                    ax=ax,
                    facecolor='none',
                    edgecolor=aerodrome_color,
                    linewidth=coastline_width * 0.5,
                    linestyle='--',
                    zorder=3.6,
                )
        
        # Render taxiways (thinner lines)
        if not taxiways.empty:
            taxiway_color = line_color if pen_plot_mode else THEME.get('road_primary', '#333333')
            taxiway_lines = taxiways[taxiways.geometry.type.isin(["LineString", "MultiLineString"])]
            taxiway_polys = taxiways[taxiways.geometry.type.isin(["Polygon", "MultiPolygon"])]
            
            if not taxiway_lines.empty:
                taxiway_lines.plot(
                    ax=ax,
                    edgecolor=taxiway_color,
                    linewidth=1.0,
                    zorder=3.7,
                )
            if not taxiway_polys.empty:
                taxiway_polys.plot(
                    ax=ax,
                    facecolor='none',
                    edgecolor=taxiway_color,
                    linewidth=0.8,
                    zorder=3.7,
                )
        
        # Render runways (thicker, more prominent)
        if not runways.empty:
            runway_color = line_color if pen_plot_mode else THEME.get('text', '#000000')
            runway_lines = runways[runways.geometry.type.isin(["LineString", "MultiLineString"])]
            runway_polys = runways[runways.geometry.type.isin(["Polygon", "MultiPolygon"])]
            
            if not runway_lines.empty:
                runway_lines.plot(
                    ax=ax,
                    edgecolor=runway_color,
                    linewidth=2.5,
                    zorder=3.8,
                )
            if not runway_polys.empty:
                runway_polys.plot(
                    ax=ax,
                    facecolor='none',
                    edgecolor=runway_color,
                    linewidth=2.0,
                    zorder=3.8,
                )
            print(f"  Drew {len(runways)} runway features")

    # Layer 3.7: Railroads/Railways
    if railways is not None and not railways.empty:
        print("Rendering railroads/railways...")
        try:
            railways_proj = ox.projection.project_gdf(railways)
        except Exception:
            railways_proj = railways.to_crs(g_proj.graph['crs'])
        
        # Filter to only LineString and MultiLineString geometries
        railway_lines = railways_proj[railways_proj.geometry.type.isin(["LineString", "MultiLineString"])]
        
        if not railway_lines.empty:
            # Use theme color if available, otherwise default to a dark color
            railway_color = THEME.get('railway', THEME.get('text', '#2C3E50'))
            if pen_plot_mode and crosshatch_water:
                railway_color = line_color
            
            # Render railways with cross ties (traditional map symbol)
            for idx, row in railway_lines.iterrows():
                geom = row.geometry
                railway_type = row.get('railway', 'rail')
                
                # Adjust parameters based on railway type
                if railway_type in ['rail', 'narrow_gauge']:
                    line_width = 0.4
                    tie_width = 0.3
                    tie_spacing = 25  # meters between ties
                    tie_length = 25   # length of perpendicular ties
                elif railway_type in ['light_rail', 'tram']:
                    line_width = 0.3
                    tie_width = 0.2
                    tie_spacing = 30
                    tie_length = 30
                elif railway_type == 'subway':
                    # Subway: use dashed line without ties (underground)
                    line_width = 0.4
                    if isinstance(geom, LineString):
                        x, y = geom.xy
                        ax.plot(x, y, color=railway_color, linewidth=line_width, 
                               linestyle='--', alpha=0.7, zorder=2.5)
                    elif isinstance(geom, MultiLineString):
                        for line in geom.geoms:
                            x, y = line.xy
                            ax.plot(x, y, color=railway_color, linewidth=line_width, 
                                   linestyle='--', alpha=0.7, zorder=2.5)
                    continue
                else:
                    line_width = 0.3
                    tie_width = 0.2
                    tie_spacing = 35
                    tie_length = 25
                
                # Draw railroad with ties
                if isinstance(geom, LineString):
                    draw_railroad_ties(ax, geom, railway_color, 
                                     tie_spacing=tie_spacing, 
                                     tie_length=tie_length,
                                     tie_width=tie_width,
                                     line_width=line_width,
                                     zorder=2.5)
                elif isinstance(geom, MultiLineString):
                    for line in geom.geoms:
                        draw_railroad_ties(ax, line, railway_color, 
                                         tie_spacing=tie_spacing, 
                                         tie_length=tie_length,
                                         tie_width=tie_width,
                                         line_width=line_width,
                                         zorder=2.5)
            
            print(f"  Drew {len(railway_lines)} railway features")

    # Layer 4: Gradients (Top and Bottom) - REMOVED for no fade effect
    # create_gradient_fade(ax, THEME['gradient_color'], location='bottom', zorder=10)
    # create_gradient_fade(ax, THEME['gradient_color'], location='top', zorder=10)

    # Calculate scale factor based on smaller dimension (reference 12 inches)
    # This ensures text scales properly for both portrait and landscape orientations
    scale_factor = min(height, width) / 12.0

    # Base font sizes (at 12 inches width)
    base_main = 60
    base_sub = 22
    base_coords = 14
    base_attr = 8

    # 4. Typography - use custom fonts if provided, otherwise use default FONTS
    active_fonts = fonts or FONTS
    if active_fonts:
        # font_main is calculated dynamically later based on length
        font_sub = FontProperties(
            fname=active_fonts["light"], size=base_sub * scale_factor
        )
        font_coords = FontProperties(
            fname=active_fonts["regular"], size=base_coords * scale_factor
        )
        font_attr = FontProperties(
            fname=active_fonts["light"], size=base_attr * scale_factor
        )
    else:
        # Fallback to system fonts
        font_sub = FontProperties(
            family="monospace", weight="normal", size=base_sub * scale_factor
        )
        font_coords = FontProperties(
            family="monospace", size=base_coords * scale_factor
        )
        font_attr = FontProperties(family="monospace", size=base_attr * scale_factor)

    # Format city name based on script type
    # Latin scripts: apply uppercase and letter spacing for aesthetic
    # Non-Latin scripts (CJK, Thai, Arabic, etc.): no spacing, preserve case structure
    if is_latin_script(display_city):
        # Latin script: uppercase with letter spacing (e.g., "P  A  R  I  S")
        spaced_city = "  ".join(list(display_city.upper()))
    else:
        # Non-Latin script: no spacing, no forced uppercase
        # For scripts like Arabic, Thai, Japanese, etc.
        spaced_city = display_city

    # Dynamically adjust font size based on city name length to prevent truncation
    # We use the already scaled "main" font size as the starting point.
    base_adjusted_main = base_main * scale_factor
    city_char_count = len(display_city)

    # Heuristic: If length is > 10, start reducing.
    if city_char_count > 10:
        length_factor = 10 / city_char_count
        adjusted_font_size = max(base_adjusted_main * length_factor, 10 * scale_factor)
    else:
        adjusted_font_size = base_adjusted_main

    if active_fonts:
        font_main_adjusted = FontProperties(
            fname=active_fonts["bold"], size=adjusted_font_size
        )
    else:
        font_main_adjusted = FontProperties(
            family="monospace", weight="bold", size=adjusted_font_size
        )

    # # --- BOTTOM TEXT ---
    # ax.text(
    #     0.5,
    #     0.14,
    #     spaced_city,
    #     transform=ax.transAxes,
    #     color=THEME["text"],
    #     ha="center",
    #     fontproperties=font_main_adjusted,
    #     zorder=11,
    # )

    # ax.text(
    #     0.5,
    #     0.10,
    #     display_country.upper(),
    #     transform=ax.transAxes,
    #     color=THEME["text"],
    #     ha="center",
    #     fontproperties=font_sub,
    #     zorder=11,
    # )

    # lat, lon = point
    # coords = (
    #     f"{lat:.4f}° N / {lon:.4f}° E"
    #     if lat >= 0
    #     else f"{abs(lat):.4f}° S / {lon:.4f}° E"
    # )
    # if lon < 0:
    #     coords = coords.replace("E", "W")

    # ax.text(
    #     0.5,
    #     0.07,
    #     coords,
    #     transform=ax.transAxes,
    #     color=THEME["text"],
    #     alpha=0.7,
    #     ha="center",
    #     fontproperties=font_coords,
    #     zorder=11,
    # )

    # ax.plot(
    #     [0.4, 0.6],
    #     [0.125, 0.125],
    #     transform=ax.transAxes,
    #     color=THEME["text"],
    #     linewidth=1 * scale_factor,
    #     zorder=11,
    # )

    # # --- ATTRIBUTION (bottom right) ---
    # if FONTS:
    #     font_attr = FontProperties(fname=FONTS["light"], size=8)
    # else:
    #     font_attr = FontProperties(family="monospace", size=8)

    # ax.text(
    #     0.98,
    #     0.02,
    #     "© OpenStreetMap contributors",
    #     transform=ax.transAxes,
    #     color=THEME["text"],
    #     alpha=0.5,
    #     ha="right",
    #     va="bottom",
    #     fontproperties=font_attr,
    #     zorder=11,
    # )

    # 5. Save
    print(f"Saving to {output_file}...")

    fmt = output_format.lower()
    
    # Matplotlib doesn't support BMP directly, so save as PNG first then convert
    if fmt == "bmp":
        # Save as PNG temporarily
        temp_png = output_file.replace('.bmp', '_temp.png')
        save_kwargs = dict(
            facecolor=bg_color,
            bbox_inches="tight",
            pad_inches=0.05,
            dpi=300,
        )
        
        # Disable anti-aliasing for pen plot mode to get clean black lines
        if pen_plot_mode and crosshatch_water:
            plt.rcParams['path.simplify'] = False
            plt.rcParams['lines.antialiased'] = False
            plt.rcParams['patch.antialiased'] = False
        
        plt.savefig(temp_png, format='png', **save_kwargs)
        plt.close()
        
        # Convert PNG to BMP
        try:
            from PIL import Image
            img = Image.open(temp_png)
            # Convert to RGB if necessary (BMP doesn't support transparency)
            if img.mode in ('RGBA', 'LA', 'P'):
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = rgb_img
            img.save(output_file, 'BMP')
            # Clean up temporary PNG
            os.remove(temp_png)
            print(f"✓ Done! Bitmap saved as {output_file}")
        except ImportError:
            print("⚠ PIL/Pillow not installed. Saving as PNG instead.")
            os.rename(temp_png, output_file.replace('.bmp', '.png'))
            print(f"✓ Done! Poster saved as {output_file.replace('.bmp', '.png')}")
    else:
        save_kwargs = dict(
            facecolor=bg_color,
            bbox_inches="tight",
            pad_inches=0.05,
        )

        # DPI matters mainly for raster formats
        if fmt == "png":
            save_kwargs["dpi"] = 300
        
        # Disable anti-aliasing for pen plot mode to get clean black lines
        if pen_plot_mode and crosshatch_water:
            plt.rcParams['path.simplify'] = False
            plt.rcParams['lines.antialiased'] = False
            plt.rcParams['patch.antialiased'] = False

        plt.savefig(output_file, format=fmt, **save_kwargs)
        plt.close()
        print(f"✓ Done! Poster saved as {output_file}")


def print_examples():
    """Print usage examples."""
    print("""
City Map Poster Generator with Coastlines
==========================================

Usage:
  python create_map_poster_coastlines.py --city <city> --country <country> [options]

Examples:
  # Iconic grid patterns
  python create_map_poster_coastlines.py -c "New York" -C "USA" -t noir -d 12000           # Manhattan grid
  python create_map_poster_coastlines.py -c "Barcelona" -C "Spain" -t warm_beige -d 8000   # Eixample district grid

  # Waterfront & canals
  python create_map_poster_coastlines.py -c "Venice" -C "Italy" -t blueprint -d 4000       # Canal network
  python create_map_poster_coastlines.py -c "Amsterdam" -C "Netherlands" -t ocean -d 6000  # Concentric canals
  python create_map_poster_coastlines.py -c "Dubai" -C "UAE" -t midnight_blue -d 15000     # Palm & coastline

  # Radial patterns
  python create_map_poster_coastlines.py -c "Paris" -C "France" -t pastel_dream -d 10000   # Haussmann boulevards
  python create_map_poster_coastlines.py -c "Moscow" -C "Russia" -t noir -d 12000          # Ring roads

  # Organic old cities
  python create_map_poster_coastlines.py -c "Tokyo" -C "Japan" -t japanese_ink -d 15000    # Dense organic streets
  python create_map_poster_coastlines.py -c "Marrakech" -C "Morocco" -t terracotta -d 5000 # Medina maze
  python create_map_poster_coastlines.py -c "Rome" -C "Italy" -t warm_beige -d 8000        # Ancient street layout

  # Coastal cities (coastlines highlighted!)
  python create_map_poster_coastlines.py -c "San Francisco" -C "USA" -t sunset -d 10000    # Peninsula grid
  python create_map_poster_coastlines.py -c "Sydney" -C "Australia" -t ocean -d 12000      # Harbor city
  python create_map_poster_coastlines.py -c "Mumbai" -C "India" -t contrast_zones -d 18000 # Coastal peninsula
  python create_map_poster_coastlines.py -c "Eureka" -C "USA" -t contrast_zones -d 8000 --coastline-width 2.0

  # Pen plotter output with cross-hatching
  python create_map_poster_coastlines.py -c "Eureka" -C "USA" -t contrast_zones -d 8000 --crosshatch-water --crosshatch-spacing 150
  python create_map_poster_coastlines.py -c "Venice" -C "Italy" -t blueprint -d 4000 --crosshatch-water --crosshatch-angle 30 --crosshatch-width 0.5

  # Layered SVG output (for Inkscape and pen plotters)
  python create_map_poster_coastlines.py -c "San Francisco" -C "USA" -t ocean -d 10000 --format layered-svg
  python create_map_poster_coastlines.py -c "Sydney" -C "Australia" -t blueprint -d 12000 -f layered-svg --coastline-width 2.0
  python create_map_poster_coastlines.py -c "Venice" -C "Italy" -t noir -d 4000 --format layered-svg

  # River cities
  python create_map_poster_coastlines.py -c "London" -C "UK" -t noir -d 15000              # Thames curves
  python create_map_poster_coastlines.py -c "Budapest" -C "Hungary" -t copper_patina -d 8000  # Danube split

  # List themes
  python create_map_poster_coastlines.py --list-themes

Options:
  --city, -c           City name (required)
  --country, -C        Country name (required)
  --country-label      Override country text displayed on poster
  --theme, -t          Theme name (default: terracotta)
  --all-themes         Generate posters for all themes
  --distance, -d       Map radius in meters (default: 4500)
  --format, -f         Output format: png, svg, pdf, bmp, layered-svg (default: png)
  --coastline-width    Width of coastline lines (default: 1.5)
  --crosshatch-water   Enable cross-hatching for water (ideal for pen plotting)
  --crosshatch-spacing Distance between hatch lines in meters (default: 100)
  --crosshatch-angle   Angle of hatch lines in degrees (default: 45)
  --crosshatch-width   Width of hatch lines (default: 0.3)
  --list-themes        List all available themes

Distance guide:
  4000-6000m   Small/dense cities (Venice, Amsterdam old center, Eureka)
  8000-12000m  Medium cities, focused downtown (Paris, Barcelona)
  15000-20000m Large metros, full city view (Tokyo, Mumbai)

Available themes can be found in the 'themes/' directory.
Generated posters are saved to 'posters/' directory.
""")


def list_themes():
    """List all available themes with descriptions."""
    available_themes = get_available_themes()
    if not available_themes:
        print("No themes found in 'themes/' directory.")
        return

    print("\nAvailable Themes:")
    print("-" * 60)
    for theme_name in available_themes:
        theme_path = os.path.join(THEMES_DIR, f"{theme_name}.json")
        try:
            with open(theme_path, "r", encoding=FILE_ENCODING) as f:
                theme_data = json.load(f)
                display_name = theme_data.get('name', theme_name)
                description = theme_data.get('description', '')
        except (OSError, json.JSONDecodeError):
            display_name = theme_name
            description = ""
        print(f"  {theme_name}")
        print(f"    {display_name}")
        if description:
            print(f"    {description}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate beautiful map posters for any city with coastline support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_map_poster_coastlines.py --city "New York" --country "USA"
  python create_map_poster_coastlines.py --city "New York" --country "USA" -l 40.776676 -73.971321 --theme neon_cyberpunk
  python create_map_poster_coastlines.py --city Tokyo --country Japan --theme midnight_blue
  python create_map_poster_coastlines.py --city Paris --country France --theme noir --distance 15000
  python create_map_poster_coastlines.py --city "San Francisco" --country USA --coastline-width 2.0
  python create_map_poster_coastlines.py --list-themes
        """,
    )

    parser.add_argument("--city", "-c", type=str, help="City name")
    parser.add_argument("--country", "-C", type=str, help="Country name")
    parser.add_argument(
        "--latitude",
        "-lat",
        dest="latitude",
        type=str,
        help="Override latitude center point",
    )
    parser.add_argument(
        "--longitude",
        "-long",
        dest="longitude",
        type=str,
        help="Override longitude center point",
    )
    parser.add_argument(
        "--country-label",
        dest="country_label",
        type=str,
        help="Override country text displayed on poster",
    )
    parser.add_argument(
        "--theme",
        "-t",
        type=str,
        default="terracotta",
        help="Theme name (default: terracotta)",
    )
    parser.add_argument(
        "--all-themes",
        "--All-themes",
        dest="all_themes",
        action="store_true",
        help="Generate posters for all themes",
    )
    parser.add_argument(
        "--distance",
        "-d",
        type=int,
        default=4500,
        help="Map radius in meters (default: 4500)",
    )
    parser.add_argument(
        "--width",
        "-W",
        type=float,
        default=12,
        help="Image width in inches (default: 12, max: 20 )",
    )
    parser.add_argument(
        "--height",
        "-H",
        type=float,
        default=16,
        help="Image height in inches (default: 16, max: 20)",
    )
    parser.add_argument(
        "--coastline-width",
        type=float,
        default=1.5,
        help="Width of coastline lines (default: 1.5)",
    )
    parser.add_argument(
        "--crop-buffer",
        type=float,
        default=0.0,
        help="Percentage buffer/margin around map edges (0-20, default: 0). Creates white space for framing.",
    )
    parser.add_argument(
        "--crosshatch-water",
        action="store_true",
        help="Use cross-hatching for water features (ideal for pen plotting)",
    )
    parser.add_argument(
        "--crosshatch-spacing",
        type=float,
        default=100,
        help="Distance between hatch lines in meters (default: 100)",
    )
    parser.add_argument(
        "--crosshatch-angle",
        type=float,
        default=45,
        help="Angle of hatch lines in degrees (default: 45)",
    )
    parser.add_argument(
        "--crosshatch-width",
        type=float,
        default=0.3,
        help="Width of hatch lines (default: 0.3)",
    )
    parser.add_argument(
        "--list-themes", action="store_true", help="List all available themes"
    )
    parser.add_argument(
        "--display-city",
        "-dc",
        type=str,
        help="Custom display name for city (for i18n support)",
    )
    parser.add_argument(
        "--display-country",
        "-dC",
        type=str,
        help="Custom display name for country (for i18n support)",
    )
    parser.add_argument(
        "--font-family",
        type=str,
        help='Google Fonts family name (e.g., "Noto Sans JP", "Open Sans"). If not specified, uses local Roboto fonts.',
    )
    parser.add_argument(
        "--format",
        "-f",
        default="png",
        choices=["png", "svg", "pdf", "bmp", "layered-svg"],
        help="Output format for the poster (default: png). Use 'bmp' for bitmap, 'layered-svg' for Inkscape-compatible layers.",
    )

    args = parser.parse_args()

    # If no arguments provided, show examples
    if len(sys.argv) == 1:
        print_examples()
        sys.exit(0)

    # List themes if requested
    if args.list_themes:
        list_themes()
        sys.exit(0)

    # Validate required arguments
    if not args.city or not args.country:
        print("Error: --city and --country are required.\n")
        print_examples()
        sys.exit(1)

    # Enforce maximum dimensions
    if args.width > 20:
        print(
            f"⚠ Width {args.width} exceeds the maximum allowed limit of 20. It's enforced as max limit 20."
        )
        args.width = 20.0
    if args.height > 20:
        print(
            f"⚠ Height {args.height} exceeds the maximum allowed limit of 20. It's enforced as max limit 20."
        )
        args.height = 20.0

    available_themes = get_available_themes()
    if not available_themes:
        print("No themes found in 'themes/' directory.")
        sys.exit(1)

    if args.all_themes:
        themes_to_generate = available_themes
    else:
        if args.theme not in available_themes:
            print(f"Error: Theme '{args.theme}' not found.")
            print(f"Available themes: {', '.join(available_themes)}")
            sys.exit(1)
        themes_to_generate = [args.theme]

    print("=" * 50)
    print("City Map Poster Generator with Coastlines")
    print("=" * 50)

    # Load custom fonts if specified
    custom_fonts = None
    if args.font_family:
        custom_fonts = load_fonts(args.font_family)
        if not custom_fonts:
            print(f"⚠ Failed to load '{args.font_family}', falling back to Roboto")

    # Get coordinates and generate poster
    try:
        if args.latitude and args.longitude:
            lat = parse(args.latitude)
            lon = parse(args.longitude)
            coords = [lat, lon]
            print(f"✓ Coordinates: {', '.join([str(i) for i in coords])}")
        else:
            coords = get_coordinates(args.city, args.country)

        for theme_name in themes_to_generate:
            THEME = load_theme(theme_name)
            output_file = generate_output_filename(args.city, theme_name, args.format)
            create_poster(
                args.city,
                args.country,
                coords,
                args.distance,
                output_file,
                args.format,
                args.width,
                args.height,
                country_label=args.country_label,
                display_city=args.display_city,
                display_country=args.display_country,
                fonts=custom_fonts,
                coastline_width=args.coastline_width,
                crosshatch_water=args.crosshatch_water,
                crosshatch_spacing=args.crosshatch_spacing,
                crosshatch_angle=args.crosshatch_angle,
                crosshatch_width=args.crosshatch_width,
                pen_plot_mode=args.crosshatch_water,  # Enable pen plot mode only when crosshatching
                crop_buffer=args.crop_buffer,
            )

        print("\n" + "=" * 50)
        print("✓ Poster generation complete!")
        print("=" * 50)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
