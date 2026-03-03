"""
SVG Export Module for Layered Map Posters

Generates Inkscape-compatible layered SVG files from OpenStreetMap data,
with separate layers for each road type, water, and parks.
Optimized for pen plotter workflows with multiple colors.
"""

import svgwrite
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString
import networkx as nx


# Road hierarchy mapping from OSM highway tags to poster road types
ROAD_HIERARCHY = {
    'motorway': ['motorway', 'motorway_link'],
    'primary': ['trunk', 'trunk_link', 'primary', 'primary_link'],
    'secondary': ['secondary', 'secondary_link'],
    'tertiary': ['tertiary', 'tertiary_link'],
    'residential': ['residential', 'living_street', 'unclassified', 'service', 'road'],
    'track': ['track', 'path', 'bridleway', 'footway', 'cycleway', 'pedestrian', 'steps']
}

# Stroke width multipliers for each road type (relative to base scale)
ROAD_WIDTHS = {
    'motorway': 1.2,
    'primary': 1.0,
    'secondary': 0.8,
    'tertiary': 0.6,
    'residential': 0.4,
    'track': 0.25
}


def transform_coords(x, y, bounds, svg_width, svg_height, margin_x=0, margin_y=0):
    """
    Transform projected CRS coordinates to SVG coordinate space.

    Args:
        x, y: Coordinates in projected CRS (meters)
        bounds: (minx, miny, maxx, maxy) bounding box
        svg_width, svg_height: Full SVG canvas dimensions in pixels
        margin_x, margin_y: Margin in pixels on each side (crop buffer)

    Returns:
        (svg_x, svg_y): Transformed coordinates in SVG canvas space,
                        already offset so content lives in the inner area.
    """
    minx, miny, maxx, maxy = bounds

    # Calculate scale to fit the inner (margin-inset) area (preserve aspect ratio)
    inner_w = svg_width - 2 * margin_x
    inner_h = svg_height - 2 * margin_y
    data_width = maxx - minx
    data_height = maxy - miny

    if data_width == 0 or data_height == 0:
        return svg_width / 2, svg_height / 2

    scale_x = inner_w / data_width
    scale_y = inner_h / data_height
    scale = min(scale_x, scale_y)

    # Center within the inner area, then offset by margin
    scaled_data_width = data_width * scale
    scaled_data_height = data_height * scale
    offset_x = (inner_w - scaled_data_width) / 2
    offset_y = (inner_h - scaled_data_height) / 2

    # Output directly in canvas coordinates (margin baked in)
    svg_x = margin_x + (x - minx) * scale + offset_x
    # Y-axis flip: SVG Y increases downward; subtract from bottom of inner area
    svg_y = (svg_height - margin_y) - ((y - miny) * scale + offset_y)

    # Round to 2 decimal places for smaller file size
    return round(svg_x, 2), round(svg_y, 2)


def polygon_to_svg_path(polygon, bounds, svg_width, svg_height, margin_x=0, margin_y=0):
    """
    Convert Shapely Polygon to SVG path data string.

    Args:
        polygon: Shapely Polygon or MultiPolygon
        bounds: Coordinate transformation bounds
        svg_width, svg_height: Full SVG canvas dimensions
        margin_x, margin_y: Margin in pixels on each side (crop buffer)

    Returns:
        SVG path data string (M...L...Z format)
    """
    path_data = []

    # Handle exterior ring
    coords = list(polygon.exterior.coords)
    if len(coords) < 3:
        return ""

    # Move to first point
    x, y = transform_coords(coords[0][0], coords[0][1], bounds, svg_width, svg_height, margin_x, margin_y)
    path_data.append(f"M {x},{y}")

    # Line to remaining points
    for coord in coords[1:]:
        x, y = transform_coords(coord[0], coord[1], bounds, svg_width, svg_height, margin_x, margin_y)
        path_data.append(f"L {x},{y}")

    # Close path
    path_data.append("Z")

    # Handle interior rings (holes)
    for interior in polygon.interiors:
        hole_coords = list(interior.coords)
        if len(hole_coords) < 3:
            continue

        x, y = transform_coords(hole_coords[0][0], hole_coords[0][1], bounds, svg_width, svg_height, margin_x, margin_y)
        path_data.append(f"M {x},{y}")

        for coord in hole_coords[1:]:
            x, y = transform_coords(coord[0], coord[1], bounds, svg_width, svg_height, margin_x, margin_y)
            path_data.append(f"L {x},{y}")

        path_data.append("Z")

    return " ".join(path_data)


def linestring_to_svg_path(linestring, bounds, svg_width, svg_height, margin_x=0, margin_y=0):
    """
    Convert Shapely LineString to SVG path data string.

    Args:
        linestring: Shapely LineString
        bounds: Coordinate transformation bounds
        svg_width, svg_height: Full SVG canvas dimensions
        margin_x, margin_y: Margin in pixels on each side (crop buffer)

    Returns:
        SVG path data string (M...L... format)
    """
    coords = list(linestring.coords)
    if len(coords) < 2:
        return ""

    path_data = []

    # Move to first point
    x, y = transform_coords(coords[0][0], coords[0][1], bounds, svg_width, svg_height, margin_x, margin_y)
    path_data.append(f"M {x},{y}")

    # Line to remaining points
    for coord in coords[1:]:
        x, y = transform_coords(coord[0], coord[1], bounds, svg_width, svg_height, margin_x, margin_y)
        path_data.append(f"L {x},{y}")

    return " ".join(path_data)


def runway_to_centerline(geom):
    """
    Extract a LineString centerline from a runway geometry.

    For LineStrings the geometry is used directly.
    For Polygons the long axis of the minimum rotated rectangle is used.

    Args:
        geom: Shapely Polygon or LineString

    Returns:
        Shapely LineString representing the runway centerline, or None.
    """
    if isinstance(geom, LineString):
        return geom
    if not isinstance(geom, Polygon) or geom.is_empty:
        return None
    rect = geom.minimum_rotated_rectangle
    coords = list(rect.exterior.coords)[:-1]  # 4 corners, drop repeat
    # Measure the two distinct side lengths
    side_a = LineString([coords[0], coords[1]]).length
    side_b = LineString([coords[1], coords[2]]).length
    if side_a >= side_b:
        # Long axis: midpoints of sides 0-3 and 1-2
        mid1 = ((coords[0][0] + coords[3][0]) / 2, (coords[0][1] + coords[3][1]) / 2)
        mid2 = ((coords[1][0] + coords[2][0]) / 2, (coords[1][1] + coords[2][1]) / 2)
    else:
        # Long axis: midpoints of sides 0-1 and 2-3
        mid1 = ((coords[0][0] + coords[1][0]) / 2, (coords[0][1] + coords[1][1]) / 2)
        mid2 = ((coords[2][0] + coords[3][0]) / 2, (coords[2][1] + coords[3][1]) / 2)
    return LineString([mid1, mid2])


def classify_roads(graph):
    """
    Classify road edges by type based on highway attribute.

    Args:
        graph: NetworkX MultiDiGraph with highway attributes

    Returns:
        Dict mapping road types to lists of (u, v, key, data) tuples
    """
    classified = {road_type: [] for road_type in ROAD_HIERARCHY.keys()}
    classified['other'] = []

    for u, v, key, data in graph.edges(data=True, keys=True):
        highway = data.get('highway', '')

        # Handle list of highway types (can happen in OSM data)
        if isinstance(highway, list):
            highway = highway[0] if highway else ''

        # Find which road type this highway belongs to
        found = False
        for road_type, highway_tags in ROAD_HIERARCHY.items():
            if highway in highway_tags:
                classified[road_type].append((u, v, key, data))
                found = True
                break

        if not found and highway:  # Non-empty highway tag not in hierarchy
            classified['other'].append((u, v, key, data))

    return classified


def calculate_bounds(graph, water_gdf=None, parks_gdf=None):
    """
    Calculate combined bounding box for all geometries.

    Args:
        graph: NetworkX graph with node coordinates
        water_gdf: Optional GeoDataFrame with water features
        parks_gdf: Optional GeoDataFrame with parks features

    Returns:
        (minx, miny, maxx, maxy) bounding box
    """
    # Start with graph bounds
    nodes = graph.nodes(data=True)
    if not nodes:
        return (0, 0, 1, 1)

    xs = [data['x'] for _, data in nodes]
    ys = [data['y'] for _, data in nodes]

    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    # Expand to include water and parks
    for gdf in [water_gdf, parks_gdf]:
        if gdf is not None and not gdf.empty:
            bounds = gdf.total_bounds  # Returns [minx, miny, maxx, maxy]
            minx = min(minx, bounds[0])
            miny = min(miny, bounds[1])
            maxx = max(maxx, bounds[2])
            maxy = max(maxy, bounds[3])

    return (minx, miny, maxx, maxy)


def clip_to_bounds(geometry, bounds):
    """
    Clip a Shapely geometry to the given bounding box.

    This ensures pen plotter path coordinates never stray outside the
    intended draw area — SVG clip paths are visual only and ignored by plotters.

    Args:
        geometry: Shapely geometry (Polygon, MultiPolygon, LineString, etc.)
        bounds: (minx, miny, maxx, maxy) bounding box

    Returns:
        Clipped geometry, or None if result is empty.
    """
    from shapely.geometry import box as shapely_box
    minx, miny, maxx, maxy = bounds
    clip_box = shapely_box(minx, miny, maxx, maxy)
    try:
        clipped = geometry.intersection(clip_box)
        return clipped if not clipped.is_empty else None
    except Exception:
        return None


class LayeredSVGRenderer:
    """
    Renders map data as layered SVG with Inkscape-compatible layer structure.
    """

    def __init__(self, width_inches, height_inches, theme, dpi=96, crop_buffer=0.0):
        """
        Initialize SVG renderer.

        Args:
            width_inches: Canvas width in inches
            height_inches: Canvas height in inches
            theme: Theme dictionary with colors
            dpi: Dots per inch for inch-to-pixel conversion (default: 96)
            crop_buffer: Percentage buffer for margins (0-20, default: 0)
        """
        self.width_px = width_inches * dpi
        self.height_px = height_inches * dpi
        self.theme = theme
        self.bounds = None

        # Compute margins for crop_buffer. Content is mapped into the inner area
        # and offset by the margin, so all paths land in the center of the full page.
        # The SVG canvas stays at full page size — the plotter knows the real paper
        # edges and the paths are genuinely inset from them.
        margin_frac = crop_buffer / 100.0 / 2
        self.margin_x = self.width_px * margin_frac
        self.margin_y = self.height_px * margin_frac
        self.inner_width = self.width_px - 2 * self.margin_x
        self.inner_height = self.height_px - 2 * self.margin_y

        # Create SVG drawing at FULL page size.
        # Use debug=False to disable validation that would reject Inkscape namespace
        self.dwg = svgwrite.Drawing(
            size=(f"{self.width_px}px", f"{self.height_px}px"),
            profile='full',
            debug=False
        )

        # Add Inkscape namespace
        self.dwg.attribs['xmlns:inkscape'] = 'http://www.inkscape.org/namespaces/inkscape'

        # Clip path covers only the inner area — keeps any stray geometry out of margins
        clip_path = self.dwg.clipPath(id='canvas-clip')
        clip_rect = self.dwg.rect(
            insert=(self.margin_x, self.margin_y),
            size=(f"{self.inner_width}px", f"{self.inner_height}px")
        )
        clip_path.add(clip_rect)
        self.dwg.defs.add(clip_path)

    def set_bounds(self, bounds):
        """Set coordinate transformation bounds."""
        self.bounds = bounds

    def create_layer(self, layer_id, layer_label, layer_number=None):
        """
        Create an Inkscape-compatible layer group.

        Args:
            layer_id: Unique ID for the layer
            layer_label: Human-readable layer name shown in Inkscape
            layer_number: Optional number prefix for the layer label

        Returns:
            svgwrite Group element configured as an Inkscape layer
        """
        layer = self.dwg.g(id=layer_id)
        layer.attribs['inkscape:groupmode'] = 'layer'
        # Add number prefix if provided
        if layer_number is not None:
            layer.attribs['inkscape:label'] = f"{layer_number} {layer_label}"
        else:
            layer.attribs['inkscape:label'] = layer_label
        # Apply clip-path to keep content within the inner (margin-inset) area
        layer.attribs['clip-path'] = 'url(#canvas-clip)'
        return layer

    def add_background_layer(self, layer_number=None):
        """Add solid background layer."""
        layer = self.create_layer('layer-background', 'background', layer_number)

        # Full canvas rectangle (full page size)
        rect = self.dwg.rect(
            insert=(0, 0),
            size=(f"{self.width_px}px", f"{self.height_px}px"),
            fill=self.theme['bg']
        )
        layer.add(rect)
        self.dwg.add(layer)

    def add_polygon_layer(self, gdf, layer_id, layer_label, fill_color, layer_number=None):
        """
        Add layer with polygon features (water, parks).

        Args:
            gdf: GeoDataFrame with Polygon/MultiPolygon geometries
            layer_id: Unique layer ID
            layer_label: Layer name in Inkscape
            fill_color: Fill color for polygons
            layer_number: Optional number prefix for layer label
        """
        layer = self.create_layer(layer_id, layer_label, layer_number)

        if gdf is None or gdf.empty or self.bounds is None:
            # Add empty layer for consistent structure
            self.dwg.add(layer)
            return

        for idx, row in gdf.iterrows():
            geom = clip_to_bounds(row.geometry, self.bounds)
            if geom is None:
                continue

            if isinstance(geom, Polygon):
                polygons = [geom]
            elif isinstance(geom, MultiPolygon):
                polygons = list(geom.geoms)
            else:
                continue  # Skip non-polygon geometries

            for polygon in polygons:
                if polygon.is_empty:
                    continue

                path_data = polygon_to_svg_path(
                    polygon, self.bounds, self.width_px, self.height_px, self.margin_x, self.margin_y
                )

                if path_data:
                    path = self.dwg.path(
                        d=path_data,
                        fill=fill_color,
                        stroke='none'
                    )
                    layer.add(path)

        self.dwg.add(layer)

    def add_road_layer(self, road_edges, layer_id, layer_label, color, width_multiplier, graph, layer_number=None):
        """
        Add layer with road edges.

        Args:
            road_edges: List of (u, v, key, data) tuples
            layer_id: Unique layer ID
            layer_label: Layer name in Inkscape
            color: Stroke color
            width_multiplier: Base width multiplier
            graph: NetworkX graph for node lookups
            layer_number: Optional number prefix for layer label
        """
        layer = self.create_layer(layer_id, layer_label, layer_number)

        if not road_edges or self.bounds is None:
            # Add empty layer for consistent structure
            self.dwg.add(layer)
            return

        # Calculate stroke width based on canvas size
        # Base width is relative to canvas dimensions
        base_width = min(self.inner_width, self.inner_height) / 1000
        stroke_width = base_width * width_multiplier

        for u, v, key, data in road_edges:
            # Get geometry from edge data
            if 'geometry' in data:
                geom = data['geometry']
            else:
                # Fallback: create LineString from node coordinates
                u_data = graph.nodes[u]
                v_data = graph.nodes[v]
                geom = LineString([(u_data['x'], u_data['y']), (v_data['x'], v_data['y'])])

            geom = clip_to_bounds(geom, self.bounds)
            if geom is None:
                continue

            if isinstance(geom, LineString):
                linestrings = [geom]
            elif isinstance(geom, MultiLineString):
                linestrings = list(geom.geoms)
            else:
                continue

            for linestring in linestrings:
                if linestring.is_empty:
                    continue

                path_data = linestring_to_svg_path(
                    linestring, self.bounds, self.width_px, self.height_px, self.margin_x, self.margin_y
                )

                if path_data:
                    path = self.dwg.path(
                        d=path_data,
                        fill='none',
                        stroke=color,
                        stroke_width=stroke_width,
                        stroke_linecap='round',
                        stroke_linejoin='round'
                    )
                    layer.add(path)

        self.dwg.add(layer)

    def add_linestring_layer(self, gdf, layer_id, layer_label, color, width_multiplier, layer_number=None):
        """
        Add layer with LineString features (coastlines, railways).

        Args:
            gdf: GeoDataFrame with LineString/MultiLineString geometries
            layer_id: Unique layer ID
            layer_label: Layer name in Inkscape
            color: Stroke color
            width_multiplier: Width multiplier for stroke
            layer_number: Optional number prefix for layer label
        """
        layer = self.create_layer(layer_id, layer_label, layer_number)

        if gdf is None or gdf.empty or self.bounds is None:
            self.dwg.add(layer)
            return

        # Calculate stroke width based on canvas size
        base_width = min(self.inner_width, self.inner_height) / 1000
        stroke_width = base_width * width_multiplier

        for idx, row in gdf.iterrows():
            geom = clip_to_bounds(row.geometry, self.bounds)
            if geom is None:
                continue

            if isinstance(geom, LineString):
                linestrings = [geom]
            elif isinstance(geom, MultiLineString):
                linestrings = list(geom.geoms)
            else:
                continue

            for linestring in linestrings:
                if linestring.is_empty:
                    continue

                path_data = linestring_to_svg_path(
                    linestring, self.bounds, self.width_px, self.height_px, self.margin_x, self.margin_y
                )

                if path_data:
                    path = self.dwg.path(
                        d=path_data,
                        fill='none',
                        stroke=color,
                        stroke_width=stroke_width,
                        stroke_linecap='round',
                        stroke_linejoin='round'
                    )
                    layer.add(path)

        self.dwg.add(layer)

    def add_polygon_outline_layer(self, gdf, layer_id, layer_label, color, width_multiplier, layer_number=None):
        """
        Add layer with polygon outlines only (no fill).

        Args:
            gdf: GeoDataFrame with Polygon/MultiPolygon geometries
            layer_id: Unique layer ID
            layer_label: Layer name in Inkscape
            color: Stroke color
            width_multiplier: Width multiplier for stroke
            layer_number: Optional number prefix for layer label
        """
        layer = self.create_layer(layer_id, layer_label, layer_number)

        if gdf is None or gdf.empty or self.bounds is None:
            self.dwg.add(layer)
            return

        # Calculate stroke width based on canvas size
        base_width = min(self.inner_width, self.inner_height) / 1000
        stroke_width = base_width * width_multiplier

        for idx, row in gdf.iterrows():
            geom = clip_to_bounds(row.geometry, self.bounds)
            if geom is None:
                continue

            if isinstance(geom, Polygon):
                polygons = [geom]
            elif isinstance(geom, MultiPolygon):
                polygons = list(geom.geoms)
            else:
                continue

            for polygon in polygons:
                if polygon.is_empty:
                    continue

                path_data = polygon_to_svg_path(
                    polygon, self.bounds, self.width_px, self.height_px, self.margin_x, self.margin_y
                )

                if path_data:
                    path = self.dwg.path(
                        d=path_data,
                        fill='none',
                        stroke=color,
                        stroke_width=stroke_width,
                        stroke_linecap='round',
                        stroke_linejoin='round'
                    )
                    layer.add(path)

        self.dwg.add(layer)

    def add_mixed_geometry_layer(self, gdf, layer_id, layer_label, color, width_multiplier, layer_number=None):
        """
        Add layer with mixed geometry types (lines and polygons).

        Args:
            gdf: GeoDataFrame with mixed geometry types
            layer_id: Unique layer ID
            layer_label: Layer name in Inkscape
            color: Stroke color
            width_multiplier: Width multiplier for stroke
            layer_number: Optional number prefix for layer label
        """
        layer = self.create_layer(layer_id, layer_label, layer_number)

        if gdf is None or gdf.empty or self.bounds is None:
            self.dwg.add(layer)
            return

        # Calculate stroke width based on canvas size
        base_width = min(self.inner_width, self.inner_height) / 1000
        stroke_width = base_width * width_multiplier

        for idx, row in gdf.iterrows():
            geom = clip_to_bounds(row.geometry, self.bounds)
            if geom is None:
                continue

            # Handle LineStrings
            if isinstance(geom, LineString):
                path_data = linestring_to_svg_path(
                    geom, self.bounds, self.width_px, self.height_px, self.margin_x, self.margin_y
                )
                if path_data:
                    path = self.dwg.path(
                        d=path_data,
                        fill='none',
                        stroke=color,
                        stroke_width=stroke_width,
                        stroke_linecap='round',
                        stroke_linejoin='round'
                    )
                    layer.add(path)

            # Handle MultiLineStrings
            elif isinstance(geom, MultiLineString):
                for linestring in geom.geoms:
                    if linestring.is_empty:
                        continue
                    path_data = linestring_to_svg_path(
                        linestring, self.bounds, self.width_px, self.height_px, self.margin_x, self.margin_y
                    )
                    if path_data:
                        path = self.dwg.path(
                            d=path_data,
                            fill='none',
                            stroke=color,
                            stroke_width=stroke_width,
                            stroke_linecap='round',
                            stroke_linejoin='round'
                        )
                        layer.add(path)

            # Handle Polygons (render as outline only)
            elif isinstance(geom, Polygon):
                path_data = polygon_to_svg_path(
                    geom, self.bounds, self.width_px, self.height_px, self.margin_x, self.margin_y
                )
                if path_data:
                    path = self.dwg.path(
                        d=path_data,
                        fill='none',
                        stroke=color,
                        stroke_width=stroke_width,
                        stroke_linecap='round',
                        stroke_linejoin='round'
                    )
                    layer.add(path)

            # Handle MultiPolygons (render as outline only)
            elif isinstance(geom, MultiPolygon):
                for polygon in geom.geoms:
                    if polygon.is_empty:
                        continue
                    path_data = polygon_to_svg_path(
                        polygon, self.bounds, self.width_px, self.height_px, self.margin_x, self.margin_y
                    )
                    if path_data:
                        path = self.dwg.path(
                            d=path_data,
                            fill='none',
                            stroke=color,
                            stroke_width=stroke_width,
                            stroke_linecap='round',
                            stroke_linejoin='round'
                        )
                        layer.add(path)

        self.dwg.add(layer)

    def add_runway_layer(self, gdf, layer_id, layer_label, color, line_spacing_m=20, layer_number=None):
        """
        Render runway geometries as 3 parallel lines (edge, center, edge).

        Each runway is reduced to its centerline then offset left and right
        by line_spacing_m metres so the plotter draws 3 distinct strokes.

        Args:
            gdf: GeoDataFrame with runway Polygon/LineString geometries
            layer_id: Unique layer ID
            layer_label: Layer name in Inkscape
            color: Stroke color
            line_spacing_m: Distance in metres between the parallel lines (default 20)
            layer_number: Optional number prefix for layer label
        """
        layer = self.create_layer(layer_id, layer_label, layer_number)

        if gdf is None or gdf.empty or self.bounds is None:
            self.dwg.add(layer)
            return

        base_width = min(self.inner_width, self.inner_height) / 1000
        stroke_width = base_width * 1.0

        for idx, row in gdf.iterrows():
            geom = clip_to_bounds(row.geometry, self.bounds)
            if geom is None:
                continue

            # Handle MultiPolygon / MultiLineString by iterating parts
            if isinstance(geom, (MultiPolygon, MultiLineString)):
                parts = list(geom.geoms)
            else:
                parts = [geom]

            for part in parts:
                centerline = runway_to_centerline(part)
                if centerline is None or centerline.is_empty or centerline.length == 0:
                    continue

                # Generate center + two offset lines
                try:
                    left  = centerline.parallel_offset(line_spacing_m, 'left',  join_style=2)
                    right = centerline.parallel_offset(line_spacing_m, 'right', join_style=2)
                except Exception:
                    left, right = None, None

                for line in [left, centerline, right]:
                    if line is None or line.is_empty:
                        continue
                    # parallel_offset can return MultiLineString
                    if isinstance(line, MultiLineString):
                        segments = list(line.geoms)
                    else:
                        segments = [line]
                    for seg in segments:
                        path_data = linestring_to_svg_path(
                            seg, self.bounds, self.width_px, self.height_px,
                            self.margin_x, self.margin_y
                        )
                        if path_data:
                            path = self.dwg.path(
                                d=path_data,
                                fill='none',
                                stroke=color,
                                stroke_width=stroke_width,
                                stroke_linecap='round',
                                stroke_linejoin='round'
                            )
                            layer.add(path)

        self.dwg.add(layer)

    def save(self, filepath):
        """
        Write SVG to file.

        Args:
            filepath: Output file path
        """
        self.dwg.saveas(filepath)


def create_layered_svg_poster(graph, water_gdf, parks_gdf, width, height, output_file, theme):
    """
    Create layered SVG poster from projected geodata.

    Args:
        graph: NetworkX MultiDiGraph (projected to metric CRS)
        water_gdf: GeoDataFrame with water polygons (projected)
        parks_gdf: GeoDataFrame with park polygons (projected)
        width: Canvas width in inches
        height: Canvas height in inches
        output_file: Output file path
        theme: Theme dictionary with colors
    """
    print("Rendering layered SVG...")

    # Calculate bounds for coordinate transformation
    bounds = calculate_bounds(graph, water_gdf, parks_gdf)

    # Initialize renderer
    renderer = LayeredSVGRenderer(width, height, theme)
    renderer.set_bounds(bounds)

    # Add background layer
    renderer.add_background_layer()

    # Add water layer
    renderer.add_polygon_layer(
        water_gdf,
        'layer-water',
        'water',
        theme.get('water', '#A0C0C0')
    )

    # Add parks layer
    renderer.add_polygon_layer(
        parks_gdf,
        'layer-parks',
        'parks',
        theme.get('parks', '#E0E0D0')
    )

    # Classify roads by type
    classified_roads = classify_roads(graph)

    # Add road layers (order: track → residential → tertiary → secondary → primary → motorway)
    road_layers = [
        ('track', 'roads-track', theme.get('road_residential', '#E0E0E0')),
        ('residential', 'roads-residential', theme.get('road_residential', '#E0E0E0')),
        ('tertiary', 'roads-tertiary', theme.get('road_tertiary', '#D0D0D0')),
        ('secondary', 'roads-secondary', theme.get('road_secondary', '#C0C0C0')),
        ('primary', 'roads-primary', theme.get('road_primary', '#B0B0B0')),
        ('motorway', 'roads-motorway', theme.get('road_motorway', '#A0A0A0')),
    ]

    for road_type, layer_label, color in road_layers:
        edges = classified_roads.get(road_type, [])
        width_multiplier = ROAD_WIDTHS.get(road_type, 0.5)

        renderer.add_road_layer(
            edges,
            f'layer-{layer_label}',
            layer_label,
            color,
            width_multiplier,
            graph
        )

    # Add "other" roads to residential layer if any exist
    if classified_roads.get('other'):
        print(f"  Note: {len(classified_roads['other'])} edges with unclassified highway tags added to residential layer")

    # Save SVG
    renderer.save(output_file)
    print(f"✓ Layered SVG saved to {output_file}")
    print(f"  Layers: background, water, parks, roads-residential, roads-tertiary, roads-secondary, roads-primary, roads-motorway")


def create_layered_svg_poster_coastlines(
    graph, water_gdf, parks_gdf, coastlines_gdf, ocean_gdf, 
    islands_gdf, aeroways_gdf, railways_gdf,
    width, height, output_file, theme, coastline_width=1.5,
    center_point=None, dist=None, crop_buffer=0.0
):
    """
    Create layered SVG poster with coastlines, islands, aeroways, and railways.

    Args:
        graph: NetworkX MultiDiGraph (projected to metric CRS)
        water_gdf: GeoDataFrame with water polygons (projected)
        parks_gdf: GeoDataFrame with park polygons (projected)
        coastlines_gdf: GeoDataFrame with coastline lines (projected)
        ocean_gdf: GeoDataFrame with ocean/sea polygons (projected)
        islands_gdf: GeoDataFrame with island polygons (projected)
        aeroways_gdf: GeoDataFrame with aeroway features (projected)
        railways_gdf: GeoDataFrame with railway lines (projected)
        width: Canvas width in inches
        height: Canvas height in inches
        output_file: Output file path
        theme: Theme dictionary with colors
        coastline_width: Width multiplier for coastline rendering (default: 1.5)
        center_point: (latitude, longitude) tuple for map center (optional)
        dist: Distance in meters from center point to crop to (optional)
        crop_buffer: Percentage of area to use as buffer/margin (0-20, default: 0)
    """
    print("Rendering layered SVG with coastlines...")

    # Calculate bounds for coordinate transformation
    if center_point is not None and dist is not None:
        # Use cropping logic similar to get_crop_limits
        import osmnx as ox
        from shapely.geometry import Point
        
        lat, lon = center_point
        # Project center point into graph CRS
        center = ox.projection.project_geometry(
            Point(lon, lat),
            crs="EPSG:4326",
            to_crs=graph.graph["crs"]
        )[0]
        center_x, center_y = center.x, center.y
        
        # Use FULL dimensions for aspect ratio (keeps geographic area the same)
        aspect = width / height
        
        # Start from the requested radius
        half_x = dist
        half_y = dist
        
        # Cut inward to match aspect
        if aspect > 1:  # landscape → reduce height
            half_y = half_x / aspect
        else:  # portrait → reduce width
            half_x = half_y * aspect
        
        bounds = (
            center_x - half_x,
            center_y - half_y,
            center_x + half_x,
            center_y + half_y
        )
    else:
        # Fall back to calculating bounds from all data
        bounds = calculate_bounds(graph, water_gdf, parks_gdf)

        # Expand bounds to include coastlines, ocean, islands, aeroways, and railways
        for gdf in [coastlines_gdf, ocean_gdf, islands_gdf, aeroways_gdf, railways_gdf]:
            if gdf is not None and not gdf.empty:
                gdf_bounds = gdf.total_bounds
                minx, miny, maxx, maxy = bounds
                bounds = (
                    min(minx, gdf_bounds[0]),
                    min(miny, gdf_bounds[1]),
                    max(maxx, gdf_bounds[2]),
                    max(maxy, gdf_bounds[3])
                )

    # Initialize renderer with crop buffer for margins
    renderer = LayeredSVGRenderer(width, height, theme, crop_buffer=crop_buffer)
    renderer.set_bounds(bounds)

    # Add background layer
    renderer.add_background_layer(layer_number=1)

    # Add ocean/sea layer (larger water bodies) - outline only for pen plotter
    if ocean_gdf is not None and not ocean_gdf.empty:
        renderer.add_polygon_outline_layer(
            ocean_gdf,
            'layer-ocean',
            'ocean',
            theme.get('water', '#A0C0C0'),
            1.5,
            layer_number=2
        )

    # Add water layer (rivers, lakes) - outline only for pen plotter
    if water_gdf is not None and not water_gdf.empty:
        renderer.add_polygon_outline_layer(
            water_gdf,
            'layer-water',
            'water',
            theme.get('water', '#A0C0C0'),
            1.0,
            layer_number=3
        )

    # Add parks layer
    if parks_gdf is not None and not parks_gdf.empty:
        renderer.add_polygon_layer(
            parks_gdf,
            'layer-parks',
            'parks',
            theme.get('parks', '#E0E0D0'),
            layer_number=4
        )

    # Classify roads by type
    classified_roads = classify_roads(graph)

    # Add road layers (order: track → residential → tertiary → secondary → primary → motorway)
    road_layers = [
        (5, 'track', 'roads-track', theme.get('road_residential', '#E0E0E0')),
        (6, 'residential', 'roads-residential', theme.get('road_residential', '#E0E0E0')),
        (7, 'tertiary', 'roads-tertiary', theme.get('road_tertiary', '#D0D0D0')),
        (8, 'secondary', 'roads-secondary', theme.get('road_secondary', '#C0C0C0')),
        (9, 'primary', 'roads-primary', theme.get('road_primary', '#B0B0B0')),
        (10, 'motorway', 'roads-motorway', theme.get('road_motorway', '#A0A0A0')),
    ]

    for layer_num, road_type, layer_label, color in road_layers:
        edges = classified_roads.get(road_type, [])
        width_multiplier = ROAD_WIDTHS.get(road_type, 0.5)

        renderer.add_road_layer(
            edges,
            f'layer-{layer_label}',
            layer_label,
            color,
            width_multiplier,
            graph,
            layer_number=layer_num
        )

    # Add railway layer
    if railways_gdf is not None and not railways_gdf.empty:
        railway_color = theme.get('railway', theme.get('text', '#2C3E50'))
        renderer.add_linestring_layer(
            railways_gdf,
            'layer-railways',
            'railways',
            railway_color,
            0.5,  # width multiplier
            layer_number=10
        )

    # Add aeroway layer (runways and taxiways)
    if aeroways_gdf is not None and not aeroways_gdf.empty:
        # Separate runways and taxiways
        runways = aeroways_gdf[aeroways_gdf.get('aeroway', '') == 'runway']
        taxiways = aeroways_gdf[aeroways_gdf.get('aeroway', '') == 'taxiway']
        
        # Render taxiways first (thinner, underneath)
        if not taxiways.empty:
            taxiway_color = theme.get('road_primary', '#333333')
            renderer.add_mixed_geometry_layer(
                taxiways,
                'layer-taxiways',
                'taxiways',
                taxiway_color,
                0.8 * 2,  # width multiplier
                layer_number=11
            )
        
        # Render runways as 3 parallel lines (edge / center / edge)
        if not runways.empty:
            runway_color = theme.get('text', '#000000')
            renderer.add_runway_layer(
                runways,
                'layer-runways',
                'runways',
                runway_color,
                line_spacing_m=10,  # spacing between parallel lines in metres
                layer_number=12
            )

    # Add coastlines layer (on top of water and roads)
    if coastlines_gdf is not None and not coastlines_gdf.empty:
        coastline_color = theme.get('coastline', theme.get('water', '#4A90A4'))
        renderer.add_linestring_layer(
            coastlines_gdf,
            'layer-coastlines',
            'coastlines',
            coastline_color,
            coastline_width,
            layer_number=13
        )

    # Add island outlines layer
    if islands_gdf is not None and not islands_gdf.empty:
        island_color = theme.get('text', '#000000')
        # Render as outlines only (no fill)
        renderer.add_polygon_outline_layer(
            islands_gdf,
            'layer-islands',
            'islands',
            island_color,
            coastline_width * 0.5,
            layer_number=14
        )

    # Save SVG
    renderer.save(output_file)
    print(f"✓ Layered SVG with coastlines saved to {output_file}")
    layers = ["1 background", "2 ocean", "3 water", "4 parks", 
              "5 roads-residential", "6 roads-tertiary", "7 roads-secondary", 
              "8 roads-primary", "9 roads-motorway", "10 railways", "11 taxiways", 
              "12 runways", "13 coastlines", "14 islands"]
    print(f"  Layers: {', '.join(layers)}")
