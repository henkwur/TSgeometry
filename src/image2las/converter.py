from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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


def _load_image_array(input_path: Path) -> np.ndarray:
    if input_path.suffix.lower() == ".hdr":
        try:
            import spectral as spy

            img = spy.open_image(str(input_path))
            return np.asarray(img.load())
        except Exception:
            pass

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

    # Set LAS header with proper scales for 0.1mm precision
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = [0.0001, 0.0001, 0.0001]
    header.offsets = [float(x_points.min(initial=0.0)), float(y_points.min(initial=0.0)), float(z_points.min(initial=0.0))]

    las = laspy.LasData(header)
    las.x = x_points
    las.y = y_points
    las.z = z_points
    if red_points is not None and green_points is not None and blue_points is not None:
        las.red = red_points
        las.green = green_points
        las.blue = blue_points
    las.write(config.output_path)

    return config.output_path

