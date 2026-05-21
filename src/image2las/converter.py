from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Literal

import laspy
import numpy as np
from PIL import Image

ChannelName = Literal["luma", "red", "green", "blue"]


@dataclass(slots=True)
class ConversionConfig:
    input_path: Path
    output_path: Path
    channel: ChannelName = "luma"
    x_scale: float = 1.0
    y_scale: float = 1.0
    z_scale: float = 1.0
    x_offset: float = 0.0
    y_offset: float = 0.0
    z_offset: float = 0.0
    invert_y: bool = True
    band_index: int | None = None
    x_band_index: int | None = None
    y_band_index: int | None = None
    z_band_index: int | None = None
    use_envi_coordinates: bool = False
    x_meter_channel: int = 227
    x_fraction_channel: int = 228
    y_meter_channel: int = 229
    y_fraction_channel: int = 230
    z_meter_channel: int = 231
    z_fraction_channel: int = 232
    use_rgb_colors: bool = True
    red_channel: int = 93
    green_channel: int = 54
    blue_channel: int = 24
    rgb_clip_low_percentile: float = 1.0
    rgb_clip_high_percentile: float = 99.5
    write_las: bool = True
    plot_offset_delta_rd: tuple[float, float] | None = None
    plot_offset_reference_plot_enter_rd: tuple[float, float] | None = None


def _offset_source_label(metadata: dict) -> str:
    if metadata.get("offsetlat") is not None and metadata.get("offsetlong") is not None:
        return "ENVI keys 'offsetlat' and 'offsetlong'"
    if isinstance(metadata.get("extrainfo"), str):
        return "ENVI 'extrainfo' (OffsetLat/OffsetLong)"
    return "unknown"


def _plot_position_source_label(metadata: dict) -> str:
    if metadata.get("plotpositionx") is not None and metadata.get("plotpositiony") is not None:
        return "ENVI keys 'plotpositionx' and 'plotpositiony'"
    if isinstance(metadata.get("extrainfo"), str):
        return "ENVI 'extrainfo' (PlotPositionX/PlotPositionY)"
    return "not present; defaults to 0,0"


def _plot_point_source_label(metadata: dict, point_name: str) -> str:
    lower_name = point_name.lower()
    if metadata.get(f"{lower_name}lat") is not None and metadata.get(f"{lower_name}long") is not None:
        return f"ENVI keys '{lower_name}lat' and '{lower_name}long'"
    if isinstance(metadata.get("extrainfo"), str):
        return f"ENVI 'extrainfo' ({point_name}Lat/{point_name}Long)"
    return "unknown"


def _write_georef_report(
    txt_path: Path,
    input_name: str,
    offset_source: str | None,
    plot_position_source: str | None,
    plot_enter_source: str | None,
    plot_exit_source: str | None,
    offset_wgs84: tuple[float, float] | None,
    offset_rd: tuple[float, float] | None,
    plot_position: tuple[float, float] | None,
    plot_offset_delta_rd: tuple[float, float] | None,
    plot_offset_rd: tuple[float, float] | None,
    plot_enter_wgs84: tuple[float, float] | None,
    plot_enter_reference_rd: tuple[float, float] | None,
    plot_enter_rd: tuple[float, float] | None,
    plot_exit_wgs84: tuple[float, float] | None,
    plot_exit_rd: tuple[float, float] | None,
) -> None:
    """Write georeference report with source fields and explicit calculations."""
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    generated_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with txt_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("Georeference calculation report\n\n")
        handle.write("Context\n")
        handle.write(f"- Input image: {input_name}\n")
        handle.write(f"- Generated (UTC): {generated_utc}\n\n")

        handle.write("Sources\n")
        if offset_source is not None:
            handle.write(f"- OffsetLat/OffsetLong source: {offset_source}\n")
        if plot_position_source is not None:
            handle.write(f"- PlotPositionX/PlotPositionY source: {plot_position_source}\n")
        if plot_enter_source is not None:
            handle.write(f"- PlotEnterLat/PlotEnterLong source: {plot_enter_source}\n")
        if plot_exit_source is not None:
            handle.write(f"- PlotExitLat/PlotExitLong source: {plot_exit_source}\n")
        handle.write("- CRS conversion: WGS84 (EPSG:4326) -> RD New (EPSG:28992) via pyproj Transformer\n\n")

        handle.write("Input values\n")
        if offset_wgs84 is not None:
            handle.write(f"- OffsetLat = {offset_wgs84[0]:.12f}\n")
            handle.write(f"- OffsetLong = {offset_wgs84[1]:.12f}\n")
        if plot_position is not None:
            handle.write(f"- PlotPositionX = {plot_position[0]:.12f}\n")
            handle.write(f"- PlotPositionY = {plot_position[1]:.12f}\n")
        if plot_enter_wgs84 is not None:
            handle.write(f"- PlotEnterLat = {plot_enter_wgs84[0]:.12f}\n")
            handle.write(f"- PlotEnterLong = {plot_enter_wgs84[1]:.12f}\n")
        if plot_enter_reference_rd is not None:
            handle.write(f"- PlotEnterRef.X (Plot 1) = {plot_enter_reference_rd[0]:.6f}\n")
            handle.write(f"- PlotEnterRef.Y (Plot 1) = {plot_enter_reference_rd[1]:.6f}\n")
        if plot_exit_wgs84 is not None:
            handle.write(f"- PlotExitLat = {plot_exit_wgs84[0]:.12f}\n")
            handle.write(f"- PlotExitLong = {plot_exit_wgs84[1]:.12f}\n")
        handle.write("\n")

        handle.write("Calculations\n")
        step = 1
        if offset_wgs84 is not None and offset_rd is not None:
            handle.write(f"{step}) Offset (RD New)\n")
            handle.write("   [Offset.X, Offset.Y] = transform_wgs84_to_rdnew(OffsetLong, OffsetLat)\n")
            handle.write(
                "   [Offset.X, Offset.Y] = "
                f"transform_wgs84_to_rdnew({offset_wgs84[1]:.12f}, {offset_wgs84[0]:.12f}) "
                f"= [{offset_rd[0]:.6f}, {offset_rd[1]:.6f}]\n\n"
            )
            step += 1

        if offset_rd is not None and plot_offset_rd is not None:
            if plot_offset_delta_rd is not None and plot_enter_rd is not None and plot_enter_reference_rd is not None:
                handle.write(f"{step}) PlotOffsetNew (delta from PlotEnter of Plot 1)\n")
                handle.write("   Delta.X = PlotEnter.X(current) - PlotEnter.X(plot1)\n")
                handle.write("   Delta.Y = PlotEnter.Y(current) - PlotEnter.Y(plot1)\n")
                handle.write(
                    f"   Delta.X = {plot_enter_rd[0]:.6f} - {plot_enter_reference_rd[0]:.6f} = {plot_offset_delta_rd[0]:.6f}\n"
                )
                handle.write(
                    f"   Delta.Y = {plot_enter_rd[1]:.6f} - {plot_enter_reference_rd[1]:.6f} = {plot_offset_delta_rd[1]:.6f}\n"
                )
                handle.write("   PlotOffset.X = Offset.X + Delta.X\n")
                handle.write("   PlotOffset.Y = Offset.Y + Delta.Y\n")
                handle.write(
                    f"   PlotOffset.X = {offset_rd[0]:.6f} + {plot_offset_delta_rd[0]:.6f} = {plot_offset_rd[0]:.6f}\n"
                )
                handle.write(
                    f"   PlotOffset.Y = {offset_rd[1]:.6f} + {plot_offset_delta_rd[1]:.6f} = {plot_offset_rd[1]:.6f}\n\n"
                )
            else:
                px = plot_position[0] if plot_position is not None else 0.0
                py = plot_position[1] if plot_position is not None else 0.0
                handle.write(f"{step}) PlotOffset (translated from Offset)\n")
                handle.write("   PlotOffset.X = Offset.X + PlotPositionX\n")
                handle.write("   PlotOffset.Y = Offset.Y + PlotPositionY\n")
                handle.write(f"   PlotOffset.X = {offset_rd[0]:.6f} + {px:.12f} = {plot_offset_rd[0]:.6f}\n")
                handle.write(f"   PlotOffset.Y = {offset_rd[1]:.6f} + {py:.12f} = {plot_offset_rd[1]:.6f}\n\n")
            step += 1

        if plot_enter_wgs84 is not None and plot_enter_rd is not None:
            handle.write(f"{step}) PlotEnter (RD New)\n")
            handle.write("   [PlotEnter.X, PlotEnter.Y] = transform_wgs84_to_rdnew(PlotEnterLong, PlotEnterLat)\n")
            handle.write(
                "   [PlotEnter.X, PlotEnter.Y] = "
                f"transform_wgs84_to_rdnew({plot_enter_wgs84[1]:.12f}, {plot_enter_wgs84[0]:.12f}) "
                f"= [{plot_enter_rd[0]:.6f}, {plot_enter_rd[1]:.6f}]\n\n"
            )
            step += 1

        if plot_exit_wgs84 is not None and plot_exit_rd is not None:
            handle.write(f"{step}) PlotExit (RD New)\n")
            handle.write("   [PlotExit.X, PlotExit.Y] = transform_wgs84_to_rdnew(PlotExitLong, PlotExitLat)\n")
            handle.write(
                "   [PlotExit.X, PlotExit.Y] = "
                f"transform_wgs84_to_rdnew({plot_exit_wgs84[1]:.12f}, {plot_exit_wgs84[0]:.12f}) "
                f"= [{plot_exit_rd[0]:.6f}, {plot_exit_rd[1]:.6f}]\n\n"
            )

        handle.write("Results\n")
        handle.write("Name\tX\tY\n")
        if offset_rd is not None:
            handle.write(f"Offset\t{offset_rd[0]:.6f}\t{offset_rd[1]:.6f}\n")
        if plot_offset_rd is not None:
            handle.write(f"PlotOffset\t{plot_offset_rd[0]:.6f}\t{plot_offset_rd[1]:.6f}\n")
        if plot_enter_rd is not None:
            handle.write(f"PlotEnter\t{plot_enter_rd[0]:.6f}\t{plot_enter_rd[1]:.6f}\n")
        if plot_exit_rd is not None:
            handle.write(f"PlotExit\t{plot_exit_rd[0]:.6f}\t{plot_exit_rd[1]:.6f}\n")


def _read_envi_metadata(hdr_path: Path) -> dict:
    """Parse ENVI header file for metadata."""
    metadata = {}
    if hdr_path.suffix.lower() != ".hdr":
        return metadata

    try:
        with open(hdr_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if "=" in line and not line.startswith("{"):
                key, val = line.split("=", 1)
                key = key.strip().lower()
                val = val.strip()
                metadata[key] = val
    except Exception:
        pass

    return metadata


def _extract_offset_wgs84(metadata: dict) -> tuple[float, float] | None:
    """Extract OffsetLat/OffsetLong from ENVI metadata when available."""
    lat = metadata.get("offsetlat")
    lon = metadata.get("offsetlong")
    if lat is not None and lon is not None:
        try:
            return float(lat), float(lon)
        except (TypeError, ValueError):
            return None

    extra_info = metadata.get("extrainfo")
    if not isinstance(extra_info, str):
        return None

    lat_match = re.search(r"OffsetLat\s*:\s*([-+]?\d+(?:\.\d+)?)", extra_info)
    lon_match = re.search(r"OffsetLong\s*:\s*([-+]?\d+(?:\.\d+)?)", extra_info)
    if not lat_match or not lon_match:
        return None

    return float(lat_match.group(1)), float(lon_match.group(1))


def _extract_plot_angle(metadata: dict) -> float | None:
    """Extract PlotAngle (degrees, 0 = North) from ENVI metadata when available."""
    angle = metadata.get("plotangle")
    if angle is not None:
        try:
            return float(angle)
        except (TypeError, ValueError):
            return None

    extra_info = metadata.get("extrainfo")
    if not isinstance(extra_info, str):
        return None

    angle_match = re.search(r"PlotAngle\s*:\s*([-+]?\d+(?:\.\d+)?)", extra_info)
    if not angle_match:
        return None

    return float(angle_match.group(1))


def _extract_plot_position(metadata: dict) -> tuple[float, float] | None:
    """Extract PlotPositionX/PlotPositionY in meters from ENVI metadata."""
    pos_x = metadata.get("plotpositionx")
    pos_y = metadata.get("plotpositiony")
    if pos_x is not None and pos_y is not None:
        try:
            return float(pos_x), float(pos_y)
        except (TypeError, ValueError):
            return None

    extra_info = metadata.get("extrainfo")
    if not isinstance(extra_info, str):
        return None

    x_match = re.search(r"PlotPositionX\s*:\s*([-+]?\d+(?:\.\d+)?)", extra_info)
    y_match = re.search(r"PlotPositionY\s*:\s*([-+]?\d+(?:\.\d+)?)", extra_info)
    if not x_match or not y_match:
        return None

    return float(x_match.group(1)), float(y_match.group(1))


def _extract_plot_point_wgs84(metadata: dict, point_name: str) -> tuple[float, float] | None:
    """Extract PlotEnter/PlotExit WGS84 lat/long from ENVI metadata."""
    lower_name = point_name.lower()
    lat = metadata.get(f"{lower_name}lat")
    lon = metadata.get(f"{lower_name}long")
    if lat is not None and lon is not None:
        try:
            return float(lat), float(lon)
        except (TypeError, ValueError):
            return None

    extra_info = metadata.get("extrainfo")
    if not isinstance(extra_info, str):
        return None

    lat_match = re.search(rf"{point_name}Lat\s*:\s*([-+]?\d+(?:\.\d+)?)", extra_info)
    lon_match = re.search(rf"{point_name}Long\s*:\s*([-+]?\d+(?:\.\d+)?)", extra_info)
    if not lat_match or not lon_match:
        return None

    return float(lat_match.group(1)), float(lon_match.group(1))


def _rotate_xy_from_north_clockwise(x: np.ndarray, y: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate local XY where angle is clockwise from North.

    In RD New convention, X points East and Y points North.
    """
    theta = np.deg2rad(angle_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    x_rot = (x * cos_t) + (y * sin_t)
    y_rot = (-x * sin_t) + (y * cos_t)
    return x_rot, y_rot


def _wgs84_to_rd_new(lat: float, lon: float) -> tuple[float, float]:
    """Convert WGS84 latitude/longitude to RD New (EPSG:28992)."""
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
    x_rd, y_rd = transformer.transform(lon, lat)
    return float(x_rd), float(y_rd)


def _write_offset_shapefile(shp_path: Path, rd_x: float, rd_y: float, source_name: str) -> None:
    """Write a point shapefile in RD New (EPSG:28992) for the offset origin."""
    import shapefile

    shp_path.parent.mkdir(parents=True, exist_ok=True)

    with shapefile.Writer(str(shp_path), shapeType=shapefile.POINT) as writer:
        writer.autoBalance = 1
        writer.field("name", "C", size=80)
        writer.field("x_rd", "F", size=18, decimal=3)
        writer.field("y_rd", "F", size=18, decimal=3)
        writer.point(rd_x, rd_y)
        writer.record(source_name, rd_x, rd_y)

    # Coordinate reference for RD New.
    prj_wkt = (
        'PROJCS["Amersfoort / RD New",GEOGCS["Amersfoort",DATUM["Amersfoort",'
        'SPHEROID["Bessel 1841",6377397.155,299.1528128]],PRIMEM["Greenwich",0],'
        'UNIT["degree",0.0174532925199433]],PROJECTION["Oblique_Stereographic"],'
        'PARAMETER["latitude_of_origin",52.15616055555555],'
        'PARAMETER["central_meridian",5.38763888888889],'
        'PARAMETER["scale_factor",0.9999079],'
        'PARAMETER["false_easting",155000],PARAMETER["false_northing",463000],'
        'UNIT["metre",1],AXIS["X",EAST],AXIS["Y",NORTH]]'
    )
    shp_path.with_suffix(".prj").write_text(prj_wkt, encoding="ascii")
    shp_path.with_suffix(".cpg").write_text("UTF-8\n", encoding="ascii")


def _load_image_array(input_path: Path) -> np.ndarray:
    if input_path.suffix.lower() == ".hdr":
        try:
            import spectral as spy

            img = spy.open_image(str(input_path))
            return np.asarray(img.load())
        except Exception as exc:
            raise ValueError(
                "Unable to load ENVI header. Ensure the matching ENVI data file (for example .raw) exists next to the .hdr file."
            ) from exc

    if input_path.suffix.lower() in {".tif", ".tiff"}:
        try:
            import tifffile as tiff

            return tiff.imread(input_path)
        except Exception:
            pass

    with Image.open(input_path) as image:
        return np.asarray(image)


def _extract_height_map(array: np.ndarray, channel: ChannelName, band_index: int | None = None) -> tuple[np.ndarray, np.ndarray | None]:

    if array.ndim == 2:
        return array.astype(np.float64), None

    if array.ndim != 3:
        raise ValueError("Unsupported image format: expected grayscale or hyperspectral/RGB data")

    if band_index is not None and array.shape[2] > band_index:
        return array[..., band_index].astype(np.float64), None

    if array.shape[2] > 4:
        return array.mean(axis=2).astype(np.float64), None

    if channel == "red":
        return array[..., 0].astype(np.float64), array[..., 3] if array.shape[2] > 3 else None
    if channel == "green":
        return array[..., 1].astype(np.float64), array[..., 3] if array.shape[2] > 3 else None
    if channel == "blue":
        return array[..., 2].astype(np.float64), array[..., 3] if array.shape[2] > 3 else None

    rgb = array[..., :3].astype(np.float64)
    return (0.2126 * rgb[..., 0]) + (0.7152 * rgb[..., 1]) + (0.0722 * rgb[..., 2]), array[..., 3] if array.shape[2] > 3 else None


def _decode_envi_pair(array: np.ndarray, meter_channel: int, fraction_channel: int) -> np.ndarray:
    """Decode one ENVI coordinate from two 1-based channels.

    Meter channel stores whole meters around an offset of 32767.
    Fraction channel stores 0.1 mm units around the same offset.
    """
    meter_index = meter_channel - 1
    fraction_index = fraction_channel - 1
    if meter_index < 0 or fraction_index < 0 or meter_index >= array.shape[2] or fraction_index >= array.shape[2]:
        raise ValueError("Configured ENVI channel is out of range")

    meters = array[..., meter_index].astype(np.float64) - 32767.0
    fraction = (array[..., fraction_index].astype(np.float64) - 32767.0) / 10000.0
    return meters + fraction


def _extract_xyz_from_envi_bands(
    array: np.ndarray,
    x_meter_channel: int,
    x_fraction_channel: int,
    y_meter_channel: int,
    y_fraction_channel: int,
    z_meter_channel: int,
    z_fraction_channel: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if array.ndim != 3:
        raise ValueError("ENVI coordinate extraction requires a 3D array")

    x = _decode_envi_pair(array, x_meter_channel, x_fraction_channel)
    y = _decode_envi_pair(array, y_meter_channel, y_fraction_channel)
    z = _decode_envi_pair(array, z_meter_channel, z_fraction_channel)
    return x.ravel(), y.ravel(), z.ravel()


def _normalize_to_uint16(values: np.ndarray, clip_low_percentile: float, clip_high_percentile: float) -> np.ndarray:
    values = values.astype(np.float64)
    low = max(0.0, min(100.0, clip_low_percentile))
    high = max(low + 1e-6, min(100.0, clip_high_percentile))
    v_min = float(np.percentile(values, low))
    v_max = float(np.percentile(values, high))
    if v_max <= v_min:
        return np.zeros(values.shape, dtype=np.uint16)
    clipped = np.clip(values, v_min, v_max)
    scaled = (clipped - v_min) / (v_max - v_min)
    return np.clip(np.round(scaled * 65535.0), 0, 65535).astype(np.uint16)


def _extract_rgb(
    array: np.ndarray,
    mask: np.ndarray | None,
    red_channel: int,
    green_channel: int,
    blue_channel: int,
    clip_low_percentile: float,
    clip_high_percentile: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if array.ndim == 2:
        gray = _normalize_to_uint16(array, clip_low_percentile, clip_high_percentile)
        if mask is not None:
            gray = gray[mask].ravel()
        else:
            gray = gray.ravel()
        return gray, gray, gray

    if array.ndim != 3:
        raise ValueError("Unsupported array dimensionality for RGB extraction")

    # Hyperspectral channels are configured as 1-based values.
    if array.shape[2] > 4:
        r_idx = red_channel - 1
        g_idx = green_channel - 1
        b_idx = blue_channel - 1
        if min(r_idx, g_idx, b_idx) < 0 or max(r_idx, g_idx, b_idx) >= array.shape[2]:
            raise ValueError("Configured RGB channel is out of range")
        r = _normalize_to_uint16(array[..., r_idx], clip_low_percentile, clip_high_percentile)
        g = _normalize_to_uint16(array[..., g_idx], clip_low_percentile, clip_high_percentile)
        b = _normalize_to_uint16(array[..., b_idx], clip_low_percentile, clip_high_percentile)
    else:
        # RGB/RGBA image channels are 0-based.
        r = _normalize_to_uint16(array[..., 0], clip_low_percentile, clip_high_percentile)
        g = _normalize_to_uint16(array[..., 1], clip_low_percentile, clip_high_percentile)
        b = _normalize_to_uint16(array[..., 2], clip_low_percentile, clip_high_percentile)

    if mask is not None:
        return r[mask].ravel(), g[mask].ravel(), b[mask].ravel()
    return r.ravel(), g.ravel(), b.ravel()


def convert_image_to_las(config: ConversionConfig) -> Path:
    array = _load_image_array(config.input_path)
    metadata = _read_envi_metadata(config.input_path)
    plot_angle_deg = _extract_plot_angle(metadata)
    plot_position = _extract_plot_position(metadata)
    plot_enter_wgs84 = _extract_plot_point_wgs84(metadata, "PlotEnter")
    plot_exit_wgs84 = _extract_plot_point_wgs84(metadata, "PlotExit")
    plot_enter_rd: tuple[float, float] | None = None
    plot_exit_rd: tuple[float, float] | None = None
    rd_origin: tuple[float, float] | None = None
    wgs84_origin = _extract_offset_wgs84(metadata)
    if wgs84_origin is not None:
        rd_origin = _wgs84_to_rd_new(wgs84_origin[0], wgs84_origin[1])
    if plot_enter_wgs84 is not None:
        plot_enter_rd = _wgs84_to_rd_new(plot_enter_wgs84[0], plot_enter_wgs84[1])
    if plot_exit_wgs84 is not None:
        plot_exit_rd = _wgs84_to_rd_new(plot_exit_wgs84[0], plot_exit_wgs84[1])

    red_points: np.ndarray | None = None
    green_points: np.ndarray | None = None
    blue_points: np.ndarray | None = None

    # Check if we should use ENVI encoded coordinates
    if config.use_envi_coordinates and array.ndim == 3 and array.shape[2] >= 233:
        x_points, y_points, z_points = _extract_xyz_from_envi_bands(
            array,
            config.x_meter_channel,
            config.x_fraction_channel,
            config.y_meter_channel,
            config.y_fraction_channel,
            config.z_meter_channel,
            config.z_fraction_channel,
        )
        if config.use_rgb_colors:
            red_points, green_points, blue_points = _extract_rgb(
                array,
                None,
                config.red_channel,
                config.green_channel,
                config.blue_channel,
                config.rgb_clip_low_percentile,
                config.rgb_clip_high_percentile,
            )
    else:
        # Fall back to height map from single band or channel
        height_map, alpha = _extract_height_map(array, config.channel, config.band_index)

        rows, cols = height_map.shape

        x_indices = np.arange(cols, dtype=np.float64)
        y_indices = np.arange(rows, dtype=np.float64)
        grid_x, grid_y = np.meshgrid(x_indices, y_indices)

        if config.invert_y:
            grid_y = (rows - 1) - grid_y

        mask = np.ones_like(height_map, dtype=bool)
        if alpha is not None:
            mask = np.asarray(alpha) > 0

        z_values = height_map * config.z_scale + config.z_offset
        x_values = grid_x * config.x_scale + config.x_offset
        y_values = grid_y * config.y_scale + config.y_offset

        x_points = x_values[mask].ravel()
        y_points = y_values[mask].ravel()
        z_points = z_values[mask].ravel()
        if config.use_rgb_colors:
            red_points, green_points, blue_points = _extract_rgb(
                array,
                mask,
                config.red_channel,
                config.green_channel,
                config.blue_channel,
                config.rgb_clip_low_percentile,
                config.rgb_clip_high_percentile,
            )

    plot_origin_x: float | None = None
    plot_origin_y: float | None = None
    if rd_origin is not None:
        origin_x = rd_origin[0]
        origin_y = rd_origin[1]
        if config.plot_offset_delta_rd is not None:
            origin_x += config.plot_offset_delta_rd[0]
            origin_y += config.plot_offset_delta_rd[1]
        elif plot_position is not None:
            origin_x += plot_position[0]
            origin_y += plot_position[1]

        # First move points to the translated plot origin.
        x_points = x_points + origin_x
        y_points = y_points + origin_y
        plot_origin_x = origin_x
        plot_origin_y = origin_y

    # Then rotate around the new plot origin if available.
    if plot_angle_deg is not None:
        if plot_origin_x is not None and plot_origin_y is not None:
            local_x = x_points - plot_origin_x
            local_y = y_points - plot_origin_y
            local_x, local_y = _rotate_xy_from_north_clockwise(local_x, local_y, plot_angle_deg)
            x_points = local_x + plot_origin_x
            y_points = local_y + plot_origin_y
        else:
            x_points, y_points = _rotate_xy_from_north_clockwise(x_points, y_points, plot_angle_deg)

    # Set LAS header with proper scales for 0.1mm precision
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = [0.0001, 0.0001, 0.0001]
    if plot_origin_x is not None and plot_origin_y is not None:
        header.offsets = [plot_origin_x, plot_origin_y, float(z_points.min(initial=0.0))]
    else:
        header.offsets = [float(x_points.min(initial=0.0)), float(y_points.min(initial=0.0)), float(z_points.min(initial=0.0))]

    if config.write_las:
        las = laspy.LasData(header)
        las.x = x_points
        las.y = y_points
        las.z = z_points
        if red_points is not None and green_points is not None and blue_points is not None:
            las.red = red_points
            las.green = green_points
            las.blue = blue_points
        las.write(config.output_path)

    plot_offset_rd: tuple[float, float] | None = None
    if rd_origin is not None:
        offset_shp_path = config.output_path.parent / f"{config.input_path.stem}_Offset.shp"
        _write_offset_shapefile(offset_shp_path, rd_origin[0], rd_origin[1], config.input_path.stem)

        translated_x = rd_origin[0]
        translated_y = rd_origin[1]
        if config.plot_offset_delta_rd is not None:
            translated_x += config.plot_offset_delta_rd[0]
            translated_y += config.plot_offset_delta_rd[1]
        elif plot_position is not None:
            translated_x += plot_position[0]
            translated_y += plot_position[1]
        plot_offset_rd = (translated_x, translated_y)
        plot_offset_shp_path = config.output_path.parent / f"{config.input_path.stem}_plotoffset.shp"
        _write_offset_shapefile(plot_offset_shp_path, translated_x, translated_y, f"{config.input_path.stem}_plotoffset")

    if plot_enter_rd is not None:
        plot_enter_shp_path = config.output_path.parent / f"{config.input_path.stem}_PlotEnter.shp"
        _write_offset_shapefile(plot_enter_shp_path, plot_enter_rd[0], plot_enter_rd[1], f"{config.input_path.stem}_PlotEnter")

    if plot_exit_rd is not None:
        plot_exit_shp_path = config.output_path.parent / f"{config.input_path.stem}_PlotExit.shp"
        _write_offset_shapefile(plot_exit_shp_path, plot_exit_rd[0], plot_exit_rd[1], f"{config.input_path.stem}_PlotExit")

    if rd_origin is not None or plot_enter_rd is not None or plot_exit_rd is not None:
        uses_plot_offset_new = config.plot_offset_delta_rd is not None and config.plot_offset_reference_plot_enter_rd is not None
        report_path = config.output_path.parent / f"{config.input_path.stem}_offsets.txt"
        _write_georef_report(
            report_path,
            config.input_path.name,
            _offset_source_label(metadata) if wgs84_origin is not None else None,
            _plot_position_source_label(metadata) if (rd_origin is not None and not uses_plot_offset_new) else None,
            _plot_point_source_label(metadata, "PlotEnter") if plot_enter_wgs84 is not None else None,
            _plot_point_source_label(metadata, "PlotExit") if plot_exit_wgs84 is not None else None,
            wgs84_origin,
            rd_origin,
            plot_position if not uses_plot_offset_new else None,
            config.plot_offset_delta_rd,
            plot_offset_rd,
            plot_enter_wgs84,
            config.plot_offset_reference_plot_enter_rd,
            plot_enter_rd,
            plot_exit_wgs84,
            plot_exit_rd,
        )

    return config.output_path


